# GLM-5.2, DeepSeek-V4 (Pro/Flash), Kimi K3 서빙 GPU 사이징

> 라이브 검증 2026-08-04. ⚠️ 항목은 배포 전 재확인.
> 검증 소스: HF `config.json` raw + HF API `?blobs=true`(safetensors 실측 bytes), `docs.aws.amazon.com/ec2/latest/instancetypes/ac.html`, `aws-samples/sagemaker-genai-hosting-examples` 노트북 원문, `vllm.ai/blog/2026-07-27-k3`.

## TL;DR

**네 모델이 네 등급으로 갈립니다. K3만 8×B300(`p6-b300`)이 필요하고, 나머지 셋은 단일 노드로 충분합니다.**

| 모델 | weights (실측) | 최소 인스턴스 | 권장 | 이유 |
|---|---|---|---|---|
| **DeepSeek-V4-Flash** | **148.7 GiB** | g7e.12xlarge (2 GPU) | **`g7e.24xlarge`** | G7e로 충분, P 불필요 |
| **GLM-5.2-FP8** | **701.6 GiB** | p5e (1128 GiB) | **`ml.p5en.48xlarge`** TP8 | H200 이상 필수 |
| **DeepSeek-V4-Pro** | **805.4 GiB** | p5e (1128 GiB) | **`ml.p5en.48xlarge`** TP8 | H200 이상 필수, 여유 적음 |
| **Kimi K3** | **1453.7 GiB** | **`p6-b300`**(2148 GiB) | **`p6-b300.48xlarge`** TP8 | **B300 미만은 단일 노드 불가** |

기준선 둘:
- **707 GiB**(= `g7e.48xlarge` 예산). 이 아래는 G7e, 위는 H200.
- **1318 GiB**(= `p6-b200` 예산). 이 위는 **B300 아니면 멀티노드**. K3(1454)가 여기 걸립니다.

## 왜 헷갈리나

1. **GB vs GiB**: safetensors는 GB(10진), GPU 메모리는 GiB(2진). 753 GB를 753 GiB로 읽으면 7% 과소평가.
2. **파라미터 수 ≠ weights**: V4-Pro는 1.6T인데 805 GiB(0.54 B/param), GLM-5.2는 753B인데 701 GiB(1.00 B/param). **1.6T 모델과 753B 모델의 메모리 차이가 15%뿐입니다.**
3. **MoE active는 메모리와 무관**: V4-Flash active 13B는 *연산량*. 메모리는 total 284B 전량.

# 1단계: weights 실측

파라미터 수로 추정하지 말고 HF API로 직접 셉니다.

```bash
for m in zai-org/GLM-5.2-FP8 deepseek-ai/DeepSeek-V4-Flash deepseek-ai/DeepSeek-V4-Pro moonshotai/Kimi-K3; do
  echo -n "$m  "
  curl -s "https://huggingface.co/api/models/$m?blobs=true" | jq '.usedStorage'
done
```

| 모델 | params | bytes | GB | **GiB** | B/param | GiB/GPU @TP8 | shards |
|---|---|---|---|---|---|---|---|
| `DeepSeek-V4-Flash` | 284B / 13B act | 159,641,337,663 | 159.6 | **148.7** | 0.562 | 18.6 | 46 |
| `GLM-5.2-FP8` | 753B | 753,375,793,584 | 753.4 | **701.6** | 1.000 | 87.7 | 141 |
| `DeepSeek-V4-Pro` | 1.6T / 49B act | 864,761,623,612 | 864.8 | **805.4** | 0.540 | 100.7 | 64 |
| `Kimi-K3` | 2.8T / 16-of-896 exp | 1,560,936,091,448 | 1560.9 | **1453.7** | 0.557 | **181.7** | 96 |

GiB 환산: `bytes / 1024³`

**B/param이 갈리는 이유**: DeepSeek V4 둘 다 `expert_dtype: "fp4"`이고 expert가 전체의 ~98%라 0.54 B/param로 수렴. GLM-5.2-FP8은 expert까지 FP8이라 정확히 1.00. K3는 **MXFP4**(`format: "mxfp4-pack-quantized"`, `num_bits: 4`, `group_size: 32`)로 0.557입니다. DeepSeek FP4와 거의 같습니다.

**→ 이것 때문에 1.6T V4-Pro(805 GiB)가 753B GLM-5.2(702 GiB)와 같은 등급입니다.** FP4가 파라미터 수 차이 2배를 흡수합니다.

**→ 하지만 4-bit로도 한계가 있습니다.** K3는 2.8T × 0.557 = **1454 GiB**. TP8이면 GPU당 **181.7 GiB**입니다. B200(179 GiB)조차 넘습니다. **여기서 "4-bit니까 어떻게든 되겠지"가 깨집니다.**

❓ **K3 quantization은 attention, shared expert를 제외합니다.** `ignore: ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\.(gate|up|down)_proj.*", "re:.*lm_head.*", "re:.*vision_tower.*"]`. 라우팅되는 expert만 MXFP4이고 나머지는 BF16입니다. 실측 1454 GiB에 이미 반영되어 있습니다.

# 2단계: KV cache per token

셋 다 **MLA(compressed KV)** 라 head 수와 무관하게 latent 하나만 캐시합니다.

**기본 공식**
```
per layer per token = (kv_lora_rank + qk_rope_head_dim) × bytes_per_elem
                    = (512 + 64) × 1 (FP8) = 576 B
```

## GLM-5.2: 압축 없음, 78 layers

```
576 B × 78 = 44,928 B = 43.88 KiB/token
```

## DeepSeek V4: CSA/HCA 압축 적용

`compress_ratios`가 layer별 압축률을 지정합니다 (`0` = 압축 없음, `128` = 1/128).

**V4-Flash** (43 layers + MTP 1): `[0, 0, 4, 128, 4, 128, ..., 4, 0]`

| layer | 개수 | B/token | 소계 |
|---|---|---|---|
| full (`0`) | 3 | 576 | 1,728 |
| ratio 4 | 21 | 144 | 3,024 |
| ratio 128 | 20 | 4.5 | 90 |
| | | | **4,842 B = 4.73 KiB** |

**V4-Pro** (61 layers + MTP 1): `[128, 128, 4, 128, ..., 4, 0]`

| layer | 개수 | B/token | 소계 |
|---|---|---|---|
| full (`0`) | **1** | 576 | 576 |
| ratio 4 | 30 | 144 | 4,320 |
| ratio 128 | 31 | 4.5 | 140 |
| | | | **5,036 B = 4.92 KiB** |

❓ **V4-Pro가 1.6T인데 KV는 V4-Flash와 거의 같습니다** (4.92 vs 4.73 KiB). layer가 43→61로 늘었지만 full layer를 3개→1개로 줄여 상쇄했습니다. **모델 크기와 KV 비용은 별개 축입니다.**

## Kimi K3: hybrid (KDA + full-attention)

K3는 압축이 아니라 **layer 종류를 나눕니다**. `linear_attn_config`에서:

```
num_hidden_layers   = 93
full_attn_layers    = [4, 8, 12, ..., 88, 92, 93]   → 24개 (4의 배수 + 마지막)
kda_layers          = 나머지 69개
```

**① full-attention layer 24개만 토큰당 KV가 자랍니다** (MLA, `kv_lora_rank=512` + `qk_rope_head_dim=64`)

```
576 B × 24 = 13,824 B = 13.50 KiB/token
```

**② KDA layer 69개는 토큰 수와 무관한 고정 크기 recurrent state** (`num_heads=96`, `head_dim=128`)

```
96 × 128 × 128 × 2 B (BF16) = 3.0 MiB/layer
3.0 MiB × 69 = 0.202 GiB/seq   ← 컨텍스트 길이와 무관한 상수
```

⚠️ KDA state의 정확한 레이아웃, dtype은 vLLM 커널 구현에 달려 있어 위 값은 **추정**입니다(`short_conv_kernel_size=4` convolution state 별도). 배포 전 실측 필요.

## 시퀀스 1개 비용 (FP8 KV)

| | 65K | 128K | 1M |
|---|---|---|---|
| V4-Flash | 0.30 GiB | 0.59 GiB | **4.73 GiB** |
| V4-Pro | 0.31 GiB | 0.61 GiB | **4.92 GiB** |
| **Kimi K3** | 1.05 GiB | 1.89 GiB | **13.70 GiB** |
| GLM-5.2 | 2.74 GiB | 5.48 GiB | **43.88 GiB** |

**GLM-5.2가 V4 대비 KV 9배 무겁습니다.** 긴 컨텍스트 동시성에서 이게 결정적입니다.

**K3는 2.8T인데 KV가 GLM-5.2(753B)의 1/3입니다.** 93 layer 중 24개만 KV를 키우기 때문입니다. 블로그가 "그래서 1M 컨텍스트가 affordable하다"고 쓴 부분이 이 계산입니다.

# 3단계: 인스턴스 예산

**공식** (vLLM `gpu_memory_utilization=0.92` 가정)
```
KV 예산 = GPU총메모리 × 0.92 − weights
동시 시퀀스 = KV 예산 ÷ (KV/token × ctx_len)
```

## DeepSeek-V4-Flash (148.7 GiB)

| 인스턴스 | GPU 메모리 | KV 예산 | 65K | 1M | 판정 |
|---|---|---|---|---|---|
| g7e.8xlarge (1×96) | 96 | - | - | - | ❌ weights 초과 |
| g7e.12xlarge (2×96) | 192 | 28.0 | 94 | 5.9 | ✅ 최소, 가성비 |
| **g7e.24xlarge (4×96)** | **384** | **204.6** | **692** | **43** | ✅ **권장** |
| g7e.48xlarge (8×96) | 768 | 557.9 | 1888 | 118 | 과잉 |
| p5en.48xlarge | 1128 | 889.1 | 3008 | 188 | 과잉 (weights가 13%) |

## GLM-5.2-FP8 (701.6 GiB)

| 인스턴스 | GPU 메모리 | KV 예산 | 65K | 1M | 판정 |
|---|---|---|---|---|---|
| g7e.24xlarge (4×96) | 384 | - | - | - | ❌ weights 초과 |
| p5.48xlarge (8×H100 80) | 640 | - | - | - | ❌ weights 초과 |
| g7e.48xlarge (8×96) | 768 | **4.9** | 2 | 0.1 | ⚠️ 산술상 뜨나 비권장 |
| **p5e / p5en.48xlarge (8×H200 141)** | **1128** | **336.1** | **123** | **7.7** | ✅ **권장** |
| p6-b200.48xlarge | 1432 | 615.8 | 225 | 14 | ✅ 여유 |
| p6-b300.48xlarge | 2148 | 1274.5 | 465 | 29 | ✅ BF16 원본도 가능 |

## DeepSeek-V4-Pro (805.4 GiB)

| 인스턴스 | GPU 메모리 | KV 예산 | 65K | 1M | 판정 |
|---|---|---|---|---|---|
| g7e.48xlarge (8×96) | 768 | - | - | - | ❌ 예산 707 < 805 |
| p5.48xlarge (8×H100 80) | 640 | - | - | - | ❌ 예산 589 < 805 |
| **p5e / p5en.48xlarge (8×H200 141)** | **1128** | **232.4** | **756** | **47** | ✅ **권장** |
| p6-b200.48xlarge | 1432 | 512.1 | 1666 | 104 | ✅ 여유 |
| p6-b300.48xlarge | 2148 | 1170.8 | 3809 | 238 | ✅ 대여유 |

## Kimi K3 (1453.7 GiB)

| 인스턴스 | GPU 메모리 | 예산(×0.92) | KV 예산 | 65K | 1M | 판정 |
|---|---|---|---|---|---|---|
| g7e.48xlarge (8×96) | 768 | 707 | - | - | - | ❌ 707 < 1454 |
| p5e / p5en.48xlarge (8×H200 141) | 1128 | 1038 | - | - | - | ❌ 1038 < 1454 |
| p6-b200.48xlarge (8×B200 179) | 1432 | 1317 | - | - | - | ❌ **1317 < 1454, 아깝게 안 됨** |
| **p6-b300.48xlarge (8×B300 268)** | **2148** | **1976** | **522.4** | **499** | **38** | ✅ **권장 (단일 노드)** |
| 16×H200 (2× p5en) | 2256 | 2076 | 621.8 | 594 | 45 | ✅ 멀티노드 |
| 16×B200 (2× p6-b200) | 2864 | 2635 | 1181.1 | 1129 | 86 | ✅ 멀티노드 |

**K3는 앞의 셋과 성격이 다릅니다. 단일 노드로 되는 인스턴스가 `p6-b300` 하나뿐입니다.**

❓ **B200 8장으로 안 되는 이유가 "조금 부족"입니다.** 1317 vs 1454 GiB(**137 GiB 차이**). util을 1.0으로 올려도 1432 < 1454로 weights조차 안 들어갑니다. TP8 기준 GPU당 181.7 GiB가 필요한데 B200은 179 GiB입니다. **GPU당 2.7 GiB가 부족해서 노드가 2배로 뜁니다.**

## vLLM 블로그와의 대조

블로그 FAQ("How many GPUs do I need to serve Kimi K3?") 원문:

> *"At least one 8× B300 (or GB300 NVL72) node is required; 16× B200 is also supported."*

**위 계산과 정확히 일치합니다.** 8×B300 = 1976 GiB 예산으로 단일 노드 가능, 8×B200 = 1317 GiB로 불가 → 16장으로 가야 함. AWS 매핑:

| 블로그 표현 | AWS 인스턴스 |
|---|---|
| 8× B300 | **`p6-b300.48xlarge`** (8×268 GiB) |
| 16× B200 | `p6-b200.48xlarge` × 2 노드 (EFA) |
| GB300 NVL72 | AWS에 대응 인스턴스 미확인 ⚠️ |

⚠️ 블로그 성능 수치(118 → 370 tok/s, DSpark 3.14×)는 **16× GB300 NVL72** 기준입니다. `p6-b300` 8장 단일 노드의 tok/s는 다른 값이며 실측 필요.

**p5e와 p5en은 GPU 메모리가 동일합니다** (둘 다 8×H200 141 GiB = 1128 GiB). 차이는 CPU(AMD EPYC vs Intel Sapphire Rapids), EBS 대역폭, 네트워크뿐 → **메모리 관점에서 p5e도 됩니다.**

❓ **V4-Pro(805 GiB)가 GLM-5.2(702 GiB)보다 KV 예산이 적은데 1M 동시성은 6배 높습니다** (47 vs 7.7). KV/token이 4.92 vs 43.88 KiB이기 때문입니다. **weights만 보고 사이징하면 틀립니다. KV까지 봐야 합니다.**

# 4단계: 메모리 다음, G7e의 구조적 제약

메모리가 맞아도 G7e에는 두 가지 한계가 있습니다.

| | G7e (RTX PRO 6000 Blackwell) | p5e/p5en (H200) | p6-b200 (B200) |
|---|---|---|---|
| GPU 메모리 | 96 GiB **GDDR7** | 141 GiB HBM3e | 179 GiB HBM3e |
| 메모리 대역폭 | **~1.8 TB/s** | ~4.8 TB/s | ~8 TB/s |
| GPU 간 연결 | **NVLink 없음 / PCIe Gen5** | NVLink | NVLink |
| FP4 native | ✅ | ❌ (marlin 커널 경유) | ✅ |

1. **대역폭**: decode는 memory-bandwidth bound. GDDR7 1.8 TB/s는 H200의 약 1/3 → 토큰 생성 속도 열위. 용량이 맞아도 TPS는 별개 문제.
2. **NVLink 부재**: TP는 layer마다 all-reduce. PCIe Gen5로 TP8을 돌리면 통신이 지배적이 되어 GPU를 늘려도 스케일이 안 붙습니다.

**→ G7e는 TP2~4로 끝나는 모델에 유리, TP8+에 불리.** V4-Flash가 G7e에 맞는 이유입니다 (TP2~4 + active 13B + FP4 native).

❓ **`g7e.48xlarge`로 GLM-5.2를 돌리면 안 되나**: KV 예산 4.9 GiB로 65K 시퀀스 2개가 한계. 게다가 NVLink 없이 78 layers × TP8 all-reduce. **"메모리는 겨우 되지만 실용성 없음."**

# 5단계: AWS 공식 샘플 교차검증

## GLM-5.2 → [`01-models/GLM/GLM-5.2`](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/GLM/GLM-5.2)

```python
instance_type = "ml.p5en.48xlarge"
env = {
    "SM_VLLM_MODEL": "zai-org/GLM-5.2-FP8",
    "SM_VLLM_TENSOR_PARALLEL_SIZE": 8,
    "SM_VLLM_MAX_MODEL_LEN": "65535",
    "SM_VLLM_KV_CACHE_DTYPE": "fp8",      # ← 2단계 계산 근거
}
image = ".../vllm:0.23.0-gpu-py312-cu130-ubuntu22.04-sagemaker"
```

3단계와 일치: 701.6 GiB + KV 336 GiB → 65K를 123 시퀀스. `max_model_len=65535`는 메모리 한계가 아닌 보수적 설정(1M도 7.7 시퀀스 가능).

## DeepSeek V4 → [`01-models/DeepSeek/DeepSeek-V4`](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/DeepSeek/DeepSeek-V4)

⚠️ **이 노트북은 Flash 전용이 아니라 Flash/Pro 공용입니다.** 주석을 바꿔 쓰는 구조:

```python
instance = {"type": "ml.p5en.48xlarge", "num_gpu": 8}
model_id = "deepseek-ai/DeepSeek-V4-Flash"
#model_id = "deepseek-ai/DeepSeek-V4-Pro"       # ← 이 줄로 전환
```

> cell 9 원문: *"The configuration below can be used to deploy both `Flash` and `Pro` DeepSeek v4 models."*

**p5en은 Pro 기준으로 선택된 것이고, Flash 단독으로는 과잉입니다.** 근거: 노트북에 Pro 전용 주석이 따로 있습니다:

```python
#"VLLM_ENGINE_READY_TIMEOUT_S": "3600",                  # 805 GiB 로딩 시간
#"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
#"SM_VLLM_GPU_MEMORY_UTILIZATION": "0.95",               # 기본 0.9로는 빡빡
#"SM_VLLM_MAX_NUM_SEQS": "512",
#"SM_VLLM_MAX_NUM_BATCHED_TOKENS": "512",
```

util 0.95가 필요한 이유가 계산으로 확인됩니다: **0.92 → KV 232 GiB, 0.95 → KV 266 GiB.** Pro는 여유가 없어 짜내는 것이고, Flash는 889 GiB가 남아 이런 튜닝이 불필요합니다.

**병렬화 3가지 프리셋** (노트북 제공):

| 목표 | 설정 |
|---|---|
| latency | `TENSOR_PARALLEL_SIZE`만 |
| balanced | `TENSOR_PARALLEL_SIZE` + `ENABLE_EXPERT_PARALLEL` |
| throughput | `TENSOR_PARALLEL_SIZE` 제거, `ENABLE_EXPERT_PARALLEL` + `DATA_PARALLEL_SIZE` |

기타 설정: `SM_VLLM_KV_CACHE_DTYPE: "fp8"`, `SM_VLLM_BLOCK_SIZE: "256"`, MTP speculative decoding(`{"method":"mtp","num_speculative_tokens":3}`), vLLM 0.20.1.

SGLang(BYOC) 경로엔 **`SGLANG_DSV4_FP4_EXPERTS: "1"` + `SM_SGLANG_MOE_RUNNER_BACKEND: "marlin"`** 이 있습니다. **H200(Hopper)에서 FP4 expert가 marlin 커널로 지원됩니다.**

# Kimi K3 서빙 설정 (vLLM 블로그 원문)

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

**⚠️ 배포 팁 (블로그 "Important Deployment Tips" 원문 기반)**

| 항목 | 내용 |
|---|---|
| **prefix caching** | K3는 **기본 비활성** (hybrid cache 설계 진행 중) → `--enable-prefix-caching` 명시 필수 |
| **all2all backend** | NVLink = `flashinfer_nvlink_one_sided`, RDMA(EFA) = `deepep_v2` |
| **MoE backend** | DEP 환경 = `deep_gemm_mega_moe`, TP > 1 = `flashinfer_trtllm` |
| **ViT parallelism** | `--mm-encoder-tp-mode=data` 기본값. vision encoder `head_size=12`라 TP8로 균등 분할 불가 |
| **tool calling** | ⚠️ 블로그가 직접 경고, 자체 parser가 빈 `tool_calls`를 내는 케이스 관측. 프로덕션은 retry/fallback + structured tool calling |
| **Docker only** | pre-release 의존성(FlashInfer 등) 때문에 현재 Docker 이미지만 사용 가능 |

DSpark speculative decoding (블로그 주장 ~3배 single-stream decode):

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark",
  "num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA",
  "draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

⚠️ SageMaker endpoint에서 K3를 서빙하려면 위 pre-release 의존성을 담은 **BYOC 이미지**가 필요합니다. 현재 `aws-samples`에 K3 예제는 확인되지 않았습니다.

# 결정 트리

```
weights(GiB) = safetensors bytes / 1024³        ← 파라미터 수로 추정 금지
                      │
        ┌─────────────┼──────────────────────┐
   < 707 GiB      707~1318 GiB           > 1318 GiB
  (g7e.48xl)      (p6-b200 예산)          (B200 8장 초과)
        │              │                      │
   V4-Flash 148.7      │                 Kimi K3 1453.7
   G7e로 충분     H200 이상 (p5e/p5en TP8)  KV 13.5 KiB/tok
   TP2~4               │                 (93 layer 중 24개만)
   g7e.24xlarge   ┌────┴────┐                 │
              GLM-5.2    V4-Pro          p6-b300 단일 노드
              701.6      805.4           (8×268 = 2148 GiB)
              KV 43.9    KV 4.92              │
              1M 7.7seq  1M 47seq        또는 16×B200 멀티노드
              util 0.92  util 0.95
```

# ⚠️ 미확정: 배포 전 실측

1. **DSA indexer cache 미포함**: 셋 다 sparse attention indexer가 KV와 **별도 캐시**를 가집니다. 위 표에 반영 안 됨.
   - V4-Flash: `index_head_dim=128`, `index_n_heads=64`, `index_topk=512`
   - V4-Pro: `index_head_dim=128`, `index_n_heads=64`, `index_topk=1024`
   - GLM-5.2: `index_topk=2048`, `index_topk_freq=4`(IndexShare로 4 layer마다 공유). **`index_head_dim`을 확인하지 못했습니다.**
   - indexer key가 layer별 캐시되면 V4 KV가 **~2배**(4.7→9-10 KiB)가 될 수 있음. V4는 여유가 커서 결론 불변이나, **GLM-5.2 1M을 노리면 반드시 실측.**
2. **`ml.g7e` SageMaker 지원 여부**: EC2 G7e는 공식 문서에 있으나 SageMaker endpoint 인스턴스 목록에서 확정 못 함. 미지원이면 EC2 self-managed / HyperPod 경로.
3. **가격**: AWS 가격 페이지가 동적 렌더링이라 시간당 단가 미확보. 비용 비교 미포함.
4. **동시 시퀀스 수치**는 `gpu_memory_utilization=0.92` 근사치. activation, CUDA graph 실제 오버헤드는 `--max-num-seqs`를 올려가며 실측.
5. **`compress_ratios` 해석**: layer별 KV 압축률로 읽었으나 커널 구현(CSA/HCA)에 따라 실제 할당이 다를 수 있음. 공식 기술 리포트로 재확인 권장.
6. **K3 KDA state 크기 추정**: `num_heads × head_dim × head_dim × 2 B × 69 layer = 0.202 GiB/seq`로 계산했으나 **vLLM 커널의 실제 state 레이아웃, dtype 미확인**. `short_conv_kernel_size=4` convolution state도 별도. K3는 KV 예산이 522 GiB로 크므로 결론(= `p6-b300` 단일 노드)은 불변이지만, 동시성 수치는 실측 필요.
7. **K3 vision encoder 메모리 미분리**: 실측 1454 GiB에 vision tower가 포함되어 있으나(1B 미만) DP로 복제되면 GPU당 추가. 블로그가 `--mm-encoder-tp-mode=data`를 기본값으로 둔 이유.
8. **`p6-b300`의 SageMaker endpoint 지원 여부 미확인**: EC2로는 존재. K3는 pre-release 의존성 때문에 BYOC 필요.
9. **GB300 NVL72의 AWS 대응 인스턴스 미확인**: 블로그 성능 수치(370 tok/s)는 16× GB300 기준이며 `p6-b300` 8장에 그대로 적용되지 않음.

# 출처

**AWS**
- [EC2 accelerated computing 인스턴스 스펙](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html) G7e/P5/P5e/P5en/P6 GPU 메모리 원본
- [EC2 G7e 제품 페이지](https://aws.amazon.com/ec2/instance-types/g7e/)
- [aws-samples/sagemaker-genai-hosting-examples: GLM-5.2](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/GLM/GLM-5.2)
- [aws-samples/sagemaker-genai-hosting-examples: DeepSeek-V4](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/DeepSeek/DeepSeek-V4)

**모델**
- [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8), [config.json](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json)
- [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json)
- [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json)
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3), [config.json](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json), [tech report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)

**vLLM**
- [Kimi K3 Is Here: Efficient Day-0 Support on vLLM](https://vllm.ai/blog/2026-07-27-k3) GPU 요구사항 FAQ, 배포 팁 원문
- [A Preview of Production-Scale Kimi K3 Support on vLLM](https://vllm.ai/blog/2026-07-22-k3-preview) KDA prefix caching, 커널 딥다이브
- [recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3) Docker 이미지, 배포 레시피
- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) speculative decoding draft
