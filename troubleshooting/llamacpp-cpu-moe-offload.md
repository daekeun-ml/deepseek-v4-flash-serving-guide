# GPU 메모리가 부족할 때: `-ncmoe`로 MoE를 CPU로 넘기기

## 요약

GGUF를 GPU에 다 올릴 수 없을 때, llama.cpp는 **MoE 전문가 가중치만 CPU RAM으로 보내는** 옵션을 제공합니다.

```bash
llama-server -m model.gguf -ngl 999 -ncmoe 20   # 앞 20개 레이어의 MoE를 CPU로
```

VRAM을 크게 줄이지만 **절감분만큼 느려집니다.** 최적화 수단이 아니라, 아예 못 돌리는 상황의 탈출구입니다.

| 설정 | GPU0 VRAM | c=1 처리량 |
|---|---|---|
| 기본 | 47.5 GB | 60.1 tok/s |
| `-ncmoe 10` | 28.5 GB (-40%) | 37.3 tok/s (-38%) |
| `-ncmoe 20` | 8.8 GB (-81%) | 28.4 tok/s (-53%) |

vLLM에는 이에 해당하는 기능이 없습니다.

## 배경 지식

### MoE와 활성 파라미터

DeepSeek-V4-Flash는 **MoE**(Mixture of Experts) 구조입니다. 전체 파라미터는 284B인데 토큰 하나를 처리할 때는 그중 **13B만** 사용합니다. 라우터가 토큰마다 몇 개의 전문가만 골라 쓰기 때문입니다.

이 구조가 메모리 관점에서 특이한 결과를 낳습니다.

- **가중치 대부분이 전문가**입니다. 그리고 각 토큰은 그중 일부만 씁니다
- 반면 어텐션과 라우터는 **모든 토큰이 매번** 사용합니다

즉 전문가 가중치는 "용량은 크지만 접근 빈도가 상대적으로 낮은" 부분입니다. 이 점이 오프로딩 대상으로 적합한 이유입니다.

### 오프로딩

GPU 메모리에 안 들어가는 부분을 CPU RAM에 두고, 필요할 때 가져와 계산하는 방식입니다. PCIe로 전송해야 하니 느려지지만, 아예 못 돌리는 것보다는 낫습니다.

llama.cpp가 제공하는 관련 플래그입니다.

| 플래그 | 동작 |
|---|---|
| `-ngl N` | GPU에 올릴 레이어 수. `999`면 전부 |
| `-cmoe`, `--cpu-moe` | **모든** MoE 가중치를 CPU로 |
| `-ncmoe N`, `--n-cpu-moe N` | **앞 N개 레이어**의 MoE만 CPU로 |

`-ncmoe`가 세밀한 조절에 유용합니다. GPU 여유에 맞춰 N을 늘리거나 줄입니다.

## 언제 필요한가

이 모델의 양자화별 메모리 요구량입니다(컨텍스트 8K, 실측).

| 양자화 | GPU 합계 | GPU 1장(96 GiB) | GPU 2장(192 GiB) |
|---|---|---|---|
| Q2 | 92.3 GiB | 가능 | 여유 |
| Q3 | 122.5 GiB | 불가 | 가능 |
| Q4 | 130.3 GiB | 불가 | 가능 |
| Q8 | 152.2 GiB | 불가 | 가능 |

GPU 2장이면 Q8까지 다 들어가므로 `-ncmoe`가 필요 없습니다. 필요해지는 경우는 이렇습니다.

- **GPU 1장뿐인데 Q3 이상을 쓰고 싶을 때**
- 컨텍스트를 크게 늘려 KV 캐시가 커질 때
- GPU를 다른 작업과 공유해야 할 때

## 측정

Q2로 고정하고 GPU 2장 환경에서 측정했습니다. 기준선은 c=1 60.1 tok/s, c=8 226.5 tok/s입니다.

| 설정 | GPU0 | GPU1 | c=1 처리량 | c=8 처리량 |
|---|---|---|---|---|
| 기본 | 47.5 GB | 46.9 GB | 60.1 | 226.5 |
| `-ncmoe 10` | **28.5 GB** | 46.6 GB | 37.3 (-38%) | 88.6 (-61%) |
| `-ncmoe 20` | **8.8 GB** | 46.6 GB | 28.4 (-53%) | 54.8 (-76%) |

두 가지가 눈에 띕니다.

**GPU0만 줄어듭니다.** `layer` 분할에서 앞쪽 레이어가 GPU0에 배치되므로, 앞 N개의 MoE를 CPU로 보내면 GPU0만 비워집니다. 양쪽을 균등하게 줄이려면 `--tensor-split` 비율을 함께 조정해야 합니다.

**동시 요청이 많을 때 더 나빠집니다.** c=1에서 -38%인 `-ncmoe 10`이 c=8에서는 -61%입니다. 요청이 늘면 CPU와 GPU 사이 전송량도 비례해 늘어나므로 PCIe가 병목이 됩니다.

## 쓸 때의 판단

**메모리 절감분과 속도 손실이 거의 1:1입니다.**

| 설정 | VRAM 절감 | 속도 손실 |
|---|---|---|
| `-ncmoe 10` | -40% | -38% |
| `-ncmoe 20` | -81% | -53% |

`-ncmoe 20`이 절감 대비 손실이 조금 낫지만, 어느 쪽도 "공짜 이득"은 아닙니다.

그래서 이렇게 접근하는 게 맞습니다.

1. **먼저 낮은 양자화를 시도합니다.** Q4를 `-ncmoe`로 밀어넣기보다 Q2를 그냥 쓰는 편이 빠릅니다. 이 모델은 양자화를 낮춰도 속도가 거의 같고([backends.md](../backends.md)), 메모리만 줄어듭니다
2. **그래도 부족하면 N을 최소로** 잡습니다. GPU에 들어갈 만큼만 넘기세요
3. 동시 요청이 많은 서빙 용도라면 `-ncmoe`는 부적합합니다

## RAM 요구량

CPU로 넘긴 만큼 시스템 RAM이 필요합니다. 이 서버는 499 GB라 여유가 크지만, RAM이 부족하면 스왑이 발생해 성능이 급락합니다.

```bash
free -g   # available이 넘길 용량보다 충분한지 확인
```

llama.cpp의 GPT-OSS 튜닝 논의에서도 같은 점을 지적합니다.

> "It is not necessary to fit the entire model in VRAM to get good performance. Offloading just the attention tensors and the KV cache in VRAM can provide decent performance."

단 시스템 RAM으로 스왑이 넘어가면 "very bad performance"라고 명시합니다.

## 참고

- [../backends.md](../backends.md) 양자화별 메모리와 속도 실측
- [../llamacpp/README.md](../llamacpp/README.md) llama-server 서빙
- [llamacpp-split-mode.md](llamacpp-split-mode.md) GPU 분할 방식 제약
- [llama.cpp GPT-OSS 튜닝 논의](https://github.com/ggml-org/llama.cpp/discussions/15396)
