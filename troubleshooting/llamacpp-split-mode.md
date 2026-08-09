# llama.cpp에서 GPU 병렬 분할(`--split-mode row`/`tensor`)이 실패하는 문제

## 요약

llama.cpp에는 여러 GPU에 모델을 나누는 방식이 4가지 있는데, **병렬 계산을 하는 두 방식이 RTX 6000 Pro(SM120)에서 로드 실패합니다.**

```
llama_model_load: error loading model: device CUDA0 does not support split buffers
```

기본값 `layer`는 동작하지만 **파이프라인 방식**이라 한 GPU가 계산할 때 다른 쪽이 대기합니다. vLLM의 텐서 병렬(TP=2)에 해당하는 구성을 쓸 수 없고, 이것이 처리량 격차(vLLM 407 tok/s 대 llama-server 218 tok/s)의 주요 원인으로 보입니다.

우회 방법은 없습니다. 다중 사용자 처리량이 필요하면 vLLM을 쓰세요.

## 배경 지식

### GPU 2장에 모델을 나누는 방식

가중치가 GPU 1장에 안 들어갈 때 여러 장에 나눠야 합니다. 나누는 방식에 따라 성능이 크게 달라집니다.

`llama-server --help`가 제시하는 4가지입니다.

| 방식 | 설명 | 특성 |
|---|---|---|
| `none` | GPU 1장만 사용 | 나누지 않음 |
| **`layer`** (기본) | 레이어를 나눠 배치 | **파이프라인** |
| `row` | 가중치를 행 단위로 분할 | 병렬 |
| `tensor` | 가중치와 KV를 분할 | 병렬 (EXPERIMENTAL) |

### 파이프라인과 병렬의 차이

**`layer`(파이프라인)**는 레이어 1~21을 GPU0에, 22~43을 GPU1에 둡니다. 토큰 하나를 처리할 때 GPU0이 먼저 계산하고, 그 결과를 GPU1에 넘겨 이어서 계산합니다.

```
시간 →
GPU0: [레이어 1-21 계산] ......대기......
GPU1: ......대기...... [레이어 22-43 계산]
```

**한 시점에 GPU 하나만 일합니다.** 요청이 여러 개 있으면 겹쳐서 채울 수 있지만(파이프라인 버블 감소), 완전히 채우기는 어렵습니다.

**`row`/`tensor`(병렬)**는 각 레이어의 가중치 자체를 쪼개서 두 GPU가 동시에 계산하고 결과를 합칩니다. vLLM의 `--tensor-parallel-size 2`가 이 방식입니다.

```
시간 →
GPU0: [레이어 1의 절반] [레이어 2의 절반] ...
GPU1: [레이어 1의 절반] [레이어 2의 절반] ...
      ↑ 매 레이어마다 all-reduce 통신
```

**항상 두 GPU가 함께 일합니다.** 대신 레이어마다 GPU 간 통신이 필요합니다.

### `--tensor-split`과 `--split-mode`는 다릅니다

이름이 비슷해 혼동하기 쉽습니다.

| 플래그 | 역할 |
|---|---|
| `--split-mode` | **어떻게** 나눌지 (layer/row/tensor) |
| `--tensor-split 1,1` | **얼마씩** 나눌지 (비율) |

`--tensor-split 1,1`만 주면 `layer` 모드에서 레이어를 절반씩 나눕니다. 이름과 달리 텐서 병렬이 아닙니다.

## 재현

```bash
docker run -d --name lcpp --gpus all -p 8001:8000 \
  -v /opt/dlami/nvme/models/gguf:/gguf:ro \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /gguf/UD-Q2_K_XL/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf \
  --host 0.0.0.0 --port 8000 -ngl 999 --split-mode row -c 8192
```

```
E llama_model_load: error loading model: device CUDA0 does not support split buffers
E llama_model_load_from_file_impl: failed to load model
E srv llama_server: exiting due to model loading error
```

`--split-mode tensor`도 같은 단계에서 실패합니다. 도움말에 EXPERIMENTAL로 표기된 방식입니다.

## 원인

`row` 모드는 ggml의 **split buffer** 기능을 요구합니다. 하나의 텐서를 여러 GPU 메모리에 걸쳐 배치하고, 각 GPU가 자기 몫만 읽어 계산하는 버퍼 타입입니다.

이 기능은 백엔드가 지원을 선언해야 쓸 수 있는데, 이 GPU의 CUDA 백엔드에서는 지원하지 않는다고 보고합니다. GPU 자체 한계인지 ggml 쪽 구현 범위인지는 오류 메시지만으로 단정할 수 없습니다.

참고로 이 GPU는 **NVLink가 없고 PCIe로만 연결**되어 있습니다(`nvidia-smi topo -m`에서 PIX). 병렬 분할은 레이어마다 GPU 간 통신이 필요하므로, 설령 동작했더라도 PCIe 대역폭이 병목이 됐을 가능성이 있습니다. vLLM은 TP=2에서 이 환경에서도 잘 동작하지만, 그건 vLLM이 자체 통신 커널을 최적화해뒀기 때문입니다.

## 영향

`layer` 모드로만 돌린 결과를 vLLM과 비교하면 이렇습니다.

**동시 8명 (c=8, 출력 200토큰)**

| 백엔드 | 분할 방식 | TPOT p50 | 처리량 |
|---|---|---|---|
| vLLM FP8 | 텐서 병렬 (TP=2) | 16.3ms | **407.3 tok/s** |
| llama-server Q2 | `layer` (파이프라인) | 34.0ms | 218.1 tok/s |

TPOT가 정확히 2배 차이입니다. GPU 하나씩 번갈아 쓰는 구조와 두 개를 동시에 쓰는 구조의 차이로 설명이 됩니다.

단일 사용자(c=1)에서는 격차가 작습니다(vLLM 10.2ms 대 llama-server 15.6ms). 요청이 하나면 파이프라인을 채울 일이 없어 두 방식의 이론적 차이가 덜 드러납니다.

## 해결

**우회 방법이 없습니다.** 선택지는 이렇습니다.

| 상황 | 선택 |
|---|---|
| 다중 사용자 처리량이 필요 | **vLLM** (텐서 병렬 동작) |
| 단일 사용자면 충분 | llama-server + DSpark (75 tok/s) |
| GPU 1장만 있음 | llama-server + Q2 (92.3 GiB, 애초에 분할 불필요) |

GPU 1장으로 돌리면 이 문제가 사라집니다. Q2는 92.3 GiB로 96 GiB 카드 하나에 들어갑니다.

```bash
docker run -d --gpus '"device=0"' ... --split-mode none
```

다만 컨텍스트를 8K 이상으로 늘리면 KV 캐시가 커져 넘칠 수 있습니다.

## 확인해두면 좋은 것

로드 후 실제 배치를 확인하려면 GPU별 메모리를 보세요. `layer` 모드가 정상 동작하면 두 GPU에 비슷하게 올라갑니다.

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# 0, 47533 MiB
# 1, 46949 MiB
```

한쪽만 차 있으면 `--tensor-split`이 반영되지 않은 것입니다.

## 참고

- [backends.md](../docs/backends.md) 백엔드 실측 비교
- [../llamacpp/README.md](../llamacpp/README.md) llama-server 서빙
- [tuning.md](../docs/tuning.md) vLLM 쪽 튜닝 (TP는 조정 대상이 아님)
- [llama.cpp server 플래그](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
