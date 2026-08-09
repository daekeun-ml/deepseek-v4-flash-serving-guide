# 서빙 백엔드 비교: vLLM vs llama.cpp vs Ollama

DeepSeek-V4-Flash-0731을 RTX 6000 Pro Blackwell ×2에서 세 가지 방식으로 서빙하고 같은 조건으로 측정한 결과입니다.

- 실행 스크립트: [llamacpp/](../llamacpp/README.md), [ollama/](../ollama/README.md), vLLM은 [README.md](../README.md)
- vLLM 튜닝: [tuning.md](tuning.md)
- 측정 스크립트: [quantbench/latency.py](../quantbench/latency.py), 결과 JSON: `quantbench/results/`

## 결론부터

| 용도 | 선택 | 처리량 (c=8) |
|---|---|---|
| 다중 사용자 서빙 | **vLLM** (FP8) | **407 tok/s** |
| 혼자 로컬 실험 | **llama-server** (Q2 + DSpark) | 75 tok/s (c=1) |
| GPU 1장만 있음 | llama-server (Q2, 92 GiB) | 218 tok/s |
| 빠르게 한번 써보기 | Ollama | 124 tok/s |

vLLM이 c=8에서 llama-server의 1.9배, Ollama의 3.3배입니다. 대신 llama.cpp 계열은 **양자화로 메모리를 줄일 수 있고 DSpark 추측 디코딩이 동작합니다**.

## 측정 조건

- 입력 약 200토큰, 출력 200토큰 (benchmark.md의 S1/S2와 동일)
- c=1은 5회, c=8은 32회 요청
- llama-server: `-ngl 999 --tensor-split 1,1 -c 8192 -np 8 --jinja`
- Ollama: `OLLAMA_NUM_PARALLEL=8`, 워밍업 후 측정
- vLLM: `docker/serve.sh` 기본 구성

## 백엔드별 결과

**단일 사용자 (c=1)**

| 백엔드 | 가중치 | TTFT p50 | TPOT p50 | 처리량 |
|---|---|---|---|---|
| **vLLM** FP8 | 148.7 GB | **51ms** | **10.2ms** | **94.4 tok/s** |
| llama-server Q2 | 96.8 GB | 71ms | 15.6ms | 61.2 tok/s |
| Ollama Q2 | 96.8 GB | 206ms | 16.4ms | 57.7 tok/s |

**동시 8명 (c=8)**

| 백엔드 | TTFT p50 | TTFT p95 | TPOT p50 | 처리량 |
|---|---|---|---|---|
| **vLLM** FP8 | **186ms** | 2,011ms | **16.3ms** | **407.3 tok/s** |
| llama-server Q2 | 417ms | **654ms** | 34.0ms | 218.1 tok/s |
| Ollama Q2 | 590ms | 7,430ms | 46.3ms | 123.5 tok/s |

vLLM은 처리량이 앞서지만 **꼬리 지연은 llama-server가 가장 안정적**입니다. TTFT p95가 654ms인데 vLLM은 2,011ms, Ollama는 7,430ms입니다.

## 양자화별 (llama-server)

unsloth GGUF 4종을 같은 조건으로 측정했습니다.

| 양자화 | 파일 | GPU0+GPU1 | c=1 TPOT | c=8 처리량 | 로드 |
|---|---|---|---|---|---|
| **Q2** (UD-Q2_K_XL) | 96.8 GB | **92.3 GiB** | **15.6ms** | 218.1 | 30초 |
| Q3 (UD-Q3_K_XL) | 129 GB | 122.5 GiB | 15.9ms | **233.3** | 30초 |
| Q4 (UD-IQ4_XS) | 138 GB | 130.3 GiB | 16.3ms | 223.5 | 30초 |
| Q8 (UD-Q8_K_XL) | 162 GB | 152.2 GiB | 20.1ms | 200.6 | 30초 |
| (vLLM FP8) | 148.7 GB | 180.5 GiB | 10.2ms | 407.3 | 5~6분 |

**양자화를 낮춰도 빨라지지 않습니다.** Q2와 Q8은 크기가 1.67배 차이인데 처리량 차이는 9%입니다. MoE라 활성 파라미터(13B)만 계산하므로 가중치 크기가 연산량을 좌우하지 않습니다.

메모리는 다릅니다. **Q2만 GPU 1장(96 GiB)에 들어갑니다.**

| 양자화 | 단일 GPU |
|---|---|
| Q2 | 가능 (92.3 GiB, 여유 3 GiB) |
| Q3 | 불가 (26.9 GiB 초과) |
| Q4 | 불가 (34.7 GiB 초과) |
| Q8 | 불가 (56.6 GiB 초과) |

Q2도 여유가 3 GiB뿐이라 컨텍스트를 8K 이상으로 늘리면 넘칩니다.

## llama-server 최적화 옵션

Q2로 고정하고 9가지를 측정했습니다. 기준선은 c=1 60.1 tok/s, c=8 226.5 tok/s입니다.

| 옵션 | c=1 | c=8 | 판정 |
|---|---|---|---|
| **`--spec-type draft-dspark`** | **+25%** | **-41%** | 단일 사용자에만 효과 |
| `-fa on` | +1% | 0% | 기본이 `auto`로 이미 활성 |
| `-b 4096 -ub 2048` | +1% | 0% | 기본값이 충분 |
| `-np 16` | +1% | 0% | c=8에는 8슬롯이면 충분 |
| `-ctk/-ctv q8_0` | -3% | -1% | 8K 컨텍스트에선 절감 없음 |
| unsloth 권장 샘플링 | -10% | 0% | 품질용 값, 속도 무관 |
| `-ncmoe 10` | -38% | -61% | 메모리 절감용 |
| `-ncmoe 20` | -53% | -76% | 메모리 절감용 |

`--cont-batching`(continuous batching)은 기본 활성이라 지정할 필요가 없습니다.

### DSpark 추측 디코딩

**vLLM에서 포기했던 기능이 llama.cpp에서는 동작합니다.** vLLM + SM120에서는 FlashInfer 커널 미지원으로 크래시하는데([dspark-sm120-crash.md](../troubleshooting/dspark-sm120-crash.md)), llama.cpp는 자체 CUDA 커널을 써서 문제없이 로드됩니다.

드래프트 모델은 **0731 GGUF 저장소에만** 있습니다(10.9 GB).

```bash
uvx --from huggingface_hub hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  --include "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf" \
  --local-dir /opt/dlami/nvme/models/gguf/dspark

DSPARK=1 bash llamacpp/serve.sh
```

| 구성 | c=1 TPOT | c=1 처리량 | c=8 처리량 |
|---|---|---|---|
| 기본 | 16.0ms | 60.1 tok/s | 226.5 tok/s |
| **+ DSpark** | **12.6ms** | **75.3 tok/s** | 132.9 tok/s |

실측 드래프트 수락률은 56.1%였습니다(32/57 수락, 평균 2.68토큰). 배치가 이미 GPU를 채우면 드래프트 계산이 낭비되므로 **동시 요청이 많을 때는 끄세요.**

Ollama에서는 이 기능을 쓸 수 없습니다: [ollama-dspark-unsupported.md](../troubleshooting/ollama-dspark-unsupported.md)

### `-ncmoe`로 GPU 메모리 절약

MoE 전문가 가중치를 CPU RAM으로 넘깁니다. GPU가 부족해 아예 못 돌리는 상황의 탈출구입니다.

| 설정 | GPU0 VRAM | c=1 처리량 |
|---|---|---|
| 기본 | 47.5 GB | 60.1 tok/s |
| `-ncmoe 10` | 28.5 GB (-40%) | 37.3 (-38%) |
| `-ncmoe 20` | 8.8 GB (-81%) | 28.4 (-53%) |

메모리 절감분과 속도 손실이 거의 비례합니다. 상세: [llamacpp-cpu-moe-offload.md](../troubleshooting/llamacpp-cpu-moe-offload.md)

### 동작하지 않는 옵션

| 옵션 | 오류 |
|---|---|
| `--split-mode row` | `device CUDA0 does not support split buffers` |
| `--split-mode tensor` | 로드 실패 (EXPERIMENTAL) |

vLLM의 TP=2에 해당하는 병렬 계산을 쓸 수 없습니다. 기본값 `layer`는 파이프라인 방식이라 한 GPU가 계산할 때 다른 쪽이 대기합니다. **c=8에서 TPOT가 정확히 2.09배 차이나는 것이 이 구조로 설명됩니다**(vLLM 16.3ms, llama-server 34.0ms). 상세: [llamacpp-split-mode.md](../troubleshooting/llamacpp-split-mode.md)

## 겪은 문제

| 문제 | 해결 |
|---|---|
| Ollama가 분할 GGUF를 거부 | 단일 파일로 병합 ([ollama/merge.sh](../ollama/merge.sh), [상세](../troubleshooting/ollama-split-gguf.md)) |
| Ollama + DSpark 크래시 | 우회 불가. llama-server를 쓸 것 ([상세](../troubleshooting/ollama-dspark-unsupported.md)) |
| `--split-mode row`/`tensor` 로드 실패 | 우회 불가 ([상세](../troubleshooting/llamacpp-split-mode.md)) |
| 디스크가 3배로 늘어남 | Ollama는 원본, 병합본, 저장소 복사본을 모두 가짐 (Q2 기준 약 301 GB) |

**reasoning 필드명이 백엔드마다 다릅니다.** 측정 스크립트를 만들 때 걸리는 함정입니다.

| 백엔드 | 필드명 |
|---|---|
| vLLM, llama.cpp | `reasoning_content` |
| **Ollama** | **`reasoning`** |

이 모델은 기본이 thinking 모드라 토큰이 대부분 이 필드로 나옵니다. 한쪽만 세면 토큰 수가 0으로 집계됩니다.

**llama-server는 `-np`를 동시 요청 수 이상으로 두세요.** `-np 4`에 c=8을 보내면 슬롯 부족으로 TTFT가 417ms에서 4,931ms로 12배 나빠집니다.

## 설치 방법

셋 다 Docker가 가장 깔끔합니다.

| 백엔드 | 이미지 | uv 설치 |
|---|---|---|
| vLLM | `vllm/vllm-openai:v0.25.0` | 가능하지만 우회 4가지 필요 ([uv-native-setup.md](../troubleshooting/uv-native-setup.md)) |
| llama.cpp | `ghcr.io/ggml-org/llama.cpp:server-cuda` | `llama-cpp-python`은 CUDA 휠이 cu125까지만 제공(이 서버는 CUDA 13) |
| Ollama | `ollama/ollama:latest` | 불가 (Go 단일 바이너리) |

llama.cpp의 Linux 사전 빌드 바이너리에는 CUDA 버전이 없고, 소스 빌드는 `nvcc`가 필요합니다.

## 참고 링크

- [unsloth/DeepSeek-V4-Flash-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF) 양자화 13종
- [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) DSpark 드래프트 포함
- [unsloth 실행 가이드](https://unsloth.ai/docs/models/deepseek-v4.md) 권장 샘플링, DSpark 명령
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) 플래그 전체
- [Ollama OpenAI 호환](https://docs.ollama.com/openai) 지원 엔드포인트와 미지원 파라미터
- [benchmark.md](benchmark.md) vLLM 상세 벤치마크
- [tuning.md](tuning.md) vLLM 튜닝
