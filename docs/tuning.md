# vLLM 최적화 설정

DeepSeek-V4-Flash-0731을 RTX 6000 Pro Blackwell ×2에서 서빙할 때의 튜닝 가이드입니다. 모든 플래그는 **vLLM 0.25.0**에서 `vllm serve --help=all`로 존재를 확인했고, 기본값과 실측값은 구동 중인 서버의 로그와 `/metrics`에서 가져왔습니다.

- 기본 구성과 실행: [README.md](../README.md)
- 실측 성능: [benchmark.md](benchmark.md)
- 게이트웨이 계층: [gateway.md](gateway.md)

## 이 환경의 병목은 KV 캐시입니다

튜닝 방향을 정하는 유일한 숫자예요. 서버 기동 로그입니다.

```
Available KV cache memory: 7.75 GiB
GPU KV cache size: 301,529 tokens
Maximum concurrency for 131,072 tokens per request: 2.30x
```

가중치 155.4 GiB가 192 GiB(96×2) 중 대부분을 차지하고, KV 캐시로 남는 것이 **GPU당 7.75 GiB뿐**입니다. 결과가 마지막 줄이에요. `--max-model-len 131072`를 꽉 채운 요청은 **동시에 2.3개**밖에 들어가지 않습니다.

여기서 두 가지가 따라옵니다.

1. **`--max-model-len`을 낮추는 것이 동시성을 늘리는 가장 강력한 수단입니다.** 컨텍스트를 절반으로 줄이면 동시성이 대략 두 배가 돼요.
2. `--gpu-memory-utilization`을 올려서 얻는 이득이 불균형하게 큽니다. 전체의 1%p(약 0.96 GiB)가 KV 캐시 7.75 GiB 기준으로는 12%거든요.

벤치마크 중 `/metrics`를 보면 동시 요청 124개가 돌아가고 있었어요. 요청들이 131K를 다 쓰지 않았기 때문입니다. 동시성 한계는 `max-model-len`이 아니라 **실제 사용 토큰 총합**으로 결정됩니다.

```
vllm:num_requests_running     124
vllm:kv_cache_usage_perc      0.565   # 56.5%
vllm:num_preemptions_total    42      # 42회 선점 발생
```

`num_preemptions_total`이 0이 아니라는 건 KV 캐시가 부족해서 진행 중인 요청을 되돌린 적이 있다는 뜻이에요. 이 값이 계속 오르면 부하가 캐시 용량을 넘고 있다는 신호입니다.

## 이미 켜져 있는 것

`--help`만 보고 "이걸 추가해야 한다"고 판단하기 전에, vLLM 0.25.0이 자동으로 켜는 것을 확인해야 합니다. 실제 엔진 로그예요.

```
enable_prefix_caching=True
enable_chunked_prefill=True
Chunked prefill is enabled with max_num_batched_tokens=8192.
block_size=4        # 지정한 256이 아님
kv_cache_dtype=fp8
quantization=deepseek_v4_fp8
cudagraph_mode=FULL_AND_PIECEWISE
```

| 항목 | 상태 | 의미 |
|---|---|---|
| Prefix caching | **자동 활성** | `--enable-prefix-caching`을 따로 줄 필요 없음. 실측 히트율 2.6~6.3% |
| Chunked prefill | **자동 활성** | 긴 프롬프트가 디코딩을 막지 않게 8,192토큰 단위로 쪼갬 |
| `max_num_batched_tokens` | **8192** (자동) | 명시하지 않았는데 이 값으로 설정됨 |
| `block_size` | **4** (덮어써짐) | `--block-size 256`을 줬지만 실제는 4. sparse attention(`sliding_window=128`)에 맞춰 vLLM이 재조정 |

`--block-size 256`이 무시된다는 점이 중요해요. `vllm:cache_config_info`에서 확인할 수 있습니다.

```
block_size="4"  user_specified_block_size="True"  num_gpu_blocks="8008"
```

즉 이 모델에서 `--block-size`는 튜닝 레버가 아닙니다.

### torch.compile이 무력화되어 있습니다

로그에 이 경고가 있어요.

```
Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1.
VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline.
Equivalent to -cc.mode=none.
```

`serve.sh`가 `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`을 넘기지만, vLLM이 breakable CUDA graph를 자동 활성화하면서 **torch.compile 파이프라인을 껐습니다**(`CompilationMode.NONE`). CUDA graph 캡처 자체는 남아 있고(배치 크기 1~512로 51개 캡처), inductor 컴파일만 빠져요.

`VLLM_USE_BREAKABLE_CUDAGRAPH=0`으로 opt-out할 수 있다고 로그가 안내합니다. 다만 breakable CUDA graph는 vLLM이 이 모델에 대해 **의도적으로** 켠 것이라 끄면 다른 문제가 생길 수 있어요. 실측으로 비교한 뒤에만 바꾸세요.

## 프리셋

[benchmark.md](benchmark.md)의 S1~S5를 근거로 세 방향을 제시합니다. 공통 플래그(TP, 파서, FP8 KV)는 [README.md](../README.md)와 동일하고 아래는 **차이나는 부분만** 표시했어요.

### 기준선 (현재 `serve.sh`)

```bash
--max-model-len 131072 \
--gpu-memory-utilization 0.92 \
--kv-cache-dtype fp8
```

실측: TTFT p50 62.8ms(c=1), 210ms(c=8), 처리량 700 tok/s(c=32), 동시성 상한 2.30x

### Latency 우선

응답 하나를 최대한 빨리 받는 구성입니다.

```bash
--max-model-len 32768 \
--gpu-memory-utilization 0.92 \
--kv-cache-dtype fp8 \
--max-num-seqs 16 \
--async-scheduling
```

| 플래그 | 근거 |
|---|---|
| `--max-model-len 32768` | 131K를 32K로 줄이면 요청당 KV 점유가 1/4. 동시성 상한이 2.3x에서 약 9x로 올라 큐 대기가 사라짐 |
| `--max-num-seqs 16` | 배치를 작게 유지해 TPOT를 낮게 붙듦. S3(c=32)에서 33.2ms까지 올랐던 것을 S2 수준(17.1ms)으로 억제 |
| `--async-scheduling` | 스케줄링과 GPU 실행을 겹쳐 유휴 구간 제거. 도움말: "helps to avoid gaps in GPU utilization, leading to better latency and throughput" |

트레이드오프로 32K를 넘는 요청은 거절됩니다. 긴 문서 요약이 필요하면 부적합해요.

### Balanced

```bash
--max-model-len 65536 \
--gpu-memory-utilization 0.93 \
--kv-cache-dtype fp8 \
--max-num-seqs 64 \
--async-scheduling
```

컨텍스트를 절반만 포기해 동시성을 약 4.6x로 올립니다. 대부분의 챗봇이나 RAG 트래픽이 64K를 넘지 않으므로 실사용에서 체감 손실이 가장 작은 지점이에요.

### Throughput 우선

```bash
--max-model-len 32768 \
--gpu-memory-utilization 0.94 \
--kv-cache-dtype fp8 \
--max-num-seqs 256 \
--max-num-batched-tokens 16384 \
--async-scheduling \
--long-prefill-token-threshold 4096 \
--max-num-partial-prefills 4
```

| 플래그 | 근거 |
|---|---|
| `--gpu-memory-utilization 0.94` | 로그가 직접 안내함. CUDA graph 메모리 프로파일링 때문에 0.92는 실효 0.8993이며 **"increase --gpu-memory-utilization to 0.9407"**로 이전 KV 캐시 크기를 회복할 수 있다고 명시 |
| `--max-num-seqs 256` | 배치를 키워 GPU 활용률을 올림. S3에서 c=32로 700 tok/s였으므로 여지가 있음 |
| `--max-num-batched-tokens 16384` | 기본 8,192의 2배. prefill 처리량이 올라감(S4의 총 4,716 tok/s가 여기서 나옴) |
| `--long-prefill-token-threshold 4096` | 4K 넘는 프롬프트를 "긴 요청"으로 분류. 기본값 0은 이 구분이 비활성 |
| `--max-num-partial-prefills 4` | 기본 1에서 4로. 긴 prefill 여러 개를 동시에 쪼개 처리 |

트레이드오프로 배치가 커지면 TPOT가 악화됩니다. S3(c=32) 실측에서 TPOT p50이 c=8의 17.1ms에서 33.2ms로 거의 두 배가 됐고, **TTFT p95는 228ms에서 10,458ms로 46배** 튀었어요. 개별 사용자 체감은 확실히 나빠집니다.

## 각 레버가 무엇을 바꾸나

### `--max-model-len`이 가장 강력합니다

KV 캐시가 7.75 GiB뿐이라 컨텍스트 길이가 동시성을 직접 결정해요.

| `--max-model-len` | 이론적 동시성 | 언제 |
|---|---|---|
| 131,072 (현재) | 2.30x | 긴 문서를 통째로 넣어야 할 때 |
| 65,536 | 약 4.6x | 대부분의 실사용 |
| 32,768 | 약 9.2x | 챗봇, 짧은 RAG |
| 16,384 | 약 18x | 분류나 추출 등 짧은 작업 |

동시성 배수는 `GPU KV cache size ÷ max-model-len`입니다(301,529 ÷ 131,072 = 2.30). 기동 로그의 `Maximum concurrency` 줄에서 항상 확인할 수 있어요.

### `--gpu-memory-utilization`은 작은 변화로 큰 효과

전체 메모리의 1%p는 약 0.96 GiB지만 KV 캐시 7.75 GiB 기준으로는 **12%**입니다. 로그가 0.9407을 권하는 이유가 여기 있어요.

올릴 때 주의할 점은 OOM이 기동 시점이 아니라 **부하가 걸린 뒤** 터질 수 있다는 것입니다. 0.94를 넘기면 S3 수준의 부하로 반드시 검증하세요. 현재 GPU 사용량은 92,959 MiB / 97,887 MiB(95%)입니다. GiB로는 90.8 / 95.6 GiB입니다.

### `--kv-cache-dtype fp8`은 이미 최선

FP8은 FP16 대비 KV 캐시를 절반으로 줄여 동시성을 두 배로 만듭니다. 이 환경에서는 필수예요.

0.25.0이 지원하는 더 공격적인 옵션(`int8_per_token_head`, `nvfp4`, `turboquant_*` 계열)도 있지만, DeepSeek-V4는 `fp8_ds_mla` 같은 전용 경로가 있고 모델 기본값이 이미 fp8입니다. 검증 없이 바꾸면 품질 저하나 커널 미지원 크래시로 이어질 수 있으니 [README.md](../README.md)에서 검증된 `fp8`을 유지하는 것을 권합니다.

### `--max-num-seqs`는 TPOT와 처리량의 저울

배치 크기 상한입니다. 벤치마크가 이 관계를 그대로 보여줘요.

| 동시성 | TPOT p50 | 출력 tok/s | TTFT p95 |
|---|---|---|---|
| 1 (S1) | 9.2ms | 105 | 64ms |
| 8 (S2) | 17.1ms | 446 | 228ms |
| 32 (S3) | 33.2ms | 701 | **10,458ms** |

동시성을 32배 늘려 처리량은 6.7배 얻었지만 TTFT p95는 163배 나빠졌습니다. `--max-num-seqs`로 이 지점을 어디에 둘지 고르는 것이에요.

### `--async-scheduling`은 부작용이 없습니다

기본값이 `None`이고 도움말이 "leading to better latency and throughput"이라고 명시합니다. 지연과 처리량 양쪽에 도움이 되므로 세 프리셋 모두에 넣었어요.

### `--scheduling-policy priority`로 티어 분리

기본은 `fcfs`(선착순)입니다. `priority`로 바꾸면 요청의 `priority` 필드(낮은 값이 우선)를 따라요. 유료 사용자를 배치 작업보다 먼저 처리할 때 유용하고, [gateway.md](gateway.md)의 LiteLLM 계층과 조합하면 키별로 우선순위를 줄 수 있습니다.

## 튜닝 절차

추측하지 말고 측정하세요. 이 환경은 **한 번 기동에 5~6분**이 걸리므로 무작정 조합을 시도하면 시간이 녹습니다.

**1) 목표를 하나 정합니다.** "TTFT p95를 500ms 아래로" 또는 "출력 1,000 tok/s 이상" 같은 단일 숫자여야 해요. 지연과 처리량을 동시에 개선하려는 시도는 위 표대로 실패합니다.

**2) 기준선을 다시 측정합니다.**

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --model deepseek-v4-flash \
  --dataset-name random --random-input-len 204 --random-output-len 200 \
  --max-concurrency 8 --num-prompts 80 \
  --save-result --result-filename baseline.json
```

**3) 한 번에 하나만 바꿉니다.** `--max-num-seqs`와 `--max-model-len`을 같이 바꾸면 어느 쪽이 효과였는지 알 수 없어요.

**4) 기동 로그에서 예상과 실제를 대조합니다.**

```bash
docker logs <container> 2>&1 | grep -E "KV cache size|Maximum concurrency|max_num_batched_tokens|non-default args"
```

`non-default args` 줄에 내가 준 플래그가 다 들어갔는지, `block_size`처럼 덮어써진 것이 없는지 확인합니다.

**5) 부하 중 `/metrics`로 병목을 확인합니다.**

```bash
curl -s http://localhost:8000/metrics | grep -E \
  '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total|prefix_cache)'
```

| 관측 | 해석 | 조치 |
|---|---|---|
| `num_requests_waiting` > 0, `kv_cache_usage_perc` ≈ 1.0 | KV 캐시 고갈 | `--max-model-len` ↓ 또는 `--gpu-memory-utilization` ↑ |
| `num_preemptions_total`이 계속 증가 | 진행 중 요청을 되돌리는 중이라 처리량 낭비 | 동시성을 낮추거나 컨텍스트 축소 |
| `num_requests_waiting` > 0, `kv_cache_usage_perc` 낮음 | 캐시가 아니라 `--max-num-seqs`가 병목 | `--max-num-seqs` ↑ |
| `prefix_cache_hits / queries`가 매우 낮음 | 프롬프트에 공통 접두사가 없음 | system prompt를 앞쪽에 고정 배치 |
| 모두 여유로운데 처리량이 안 오름 | GPU 자체가 한계 | 하드웨어 증설 외에 답 없음 |

prefix cache 히트율은 나눗셈으로 직접 계산합니다. 측정 시점 실측값이에요.

```
prefix_cache_queries_total  14,160,454
prefix_cache_hits_total        898,048   → 6.3%
```

랜덤 데이터셋 벤치마크라 낮게 나온 값입니다. 실제 서비스에서 공통 system prompt를 쓰면 훨씬 높아지고 그만큼 prefill이 절약돼요.

**6) 같은 조건으로 재측정하고 비교합니다.**

```bash
python3 -c "
import json
for f in ['baseline.json','tuned.json']:
    d = json.load(open(f))
    print(f\"{f:14} TTFT p50={d['p50_ttft_ms']:8.1f} p95={d['p95_ttft_ms']:9.1f} \"
          f\"TPOT p50={d['p50_tpot_ms']:6.2f} out_tok/s={d['output_throughput']:7.1f}\")
"
```

## 손대지 말아야 하는 것

| 플래그 | 이유 |
|---|---|
| `--tensor-parallel-size` | 가중치 155.4 GiB는 96 GiB 1장에 안 들어감. TP=2가 하한이고 GPU가 2장이라 선택지 없음 |
| `--speculative-config` | 0731 체크포인트는 DSpark만 내장하고 SM120에서 크래시. [dspark-sm120-crash.md](../troubleshooting/dspark-sm120-crash.md) |
| `--block-size` | 이 모델에서 무시됨(지정 256, 실제 4). sparse attention 구조가 결정 |
| `--enable-prefix-caching` | 이미 자동 활성 |
| `--enable-chunked-prefill` | 이미 자동 활성 |
| `--quantization` | 체크포인트가 `deepseek_v4_fp8`로 이미 양자화됨 |
| `--enable-expert-parallel` | MoE 전문가를 GPU에 분산. TP=2에 GPU 2장뿐인 구성에서는 이득 없이 통신만 늘어남 |

## 관측 설정

부하 상황을 계속 봐야 한다면 다음을 켭니다.

```bash
--enable-server-load-tracking \   # /metrics에 서버 부하 지표 추가
--enable-request-id-headers \     # 응답에 X-Request-Id. 게이트웨이 로그와 대조 가능
--max-log-len 100                 # 프롬프트 전문이 로그에 남지 않게 절단
```

`--max-log-len`은 개인정보 보호 측면에서도 중요해요. 기본값은 프롬프트 전체를 로그에 남깁니다.

Prometheus로 수집한다면 `/metrics`를 **내부망으로만** 제한하세요. 이 엔드포인트는 프롬프트 토큰 통계와 모델 구성을 그대로 노출합니다.

## 참고 링크

- [README.md](../README.md) 기본 서빙 구성과 각 플래그의 선택 근거
- [benchmark.md](benchmark.md) 이 문서가 인용한 S1~S5 실측값
- [gateway.md](gateway.md) 게이트웨이 계층(인증, 쿼터, 스트리밍)
- [serving.md](serving.md) 메모리 사이징 계산
- [vLLM 최적화 문서](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- [vLLM 공식 recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) 하드웨어별 권장 실행 명령
