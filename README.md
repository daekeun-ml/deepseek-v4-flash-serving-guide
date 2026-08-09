# DeepSeek-V4-Flash-0731 Serving Guide

RTX 6000 Pro Blackwell 2장으로 284B MoE 모델을 서빙한 기록입니다. 구성 근거, 실제로 부딪힌 문제, 실측 성능을 함께 정리했습니다.

[Why this guide](#why-this-guide) | [Results](#results) | [Is this for you](#is-this-for-you) | [Setup](#setup) | [Run](#run) | [Cost & cleanup](#cost--cleanup) | [Documentation](#documentation)

## Why this guide

**H100/H200/B200/B300이 부담스러울 때 가장 먼저 떠오르는 대안이 RTX 6000 Pro Blackwell입니다.** 96 GiB를 장당 제공하고 FP4를 네이티브로 지원해서, 두 장이면 148.7 GiB 모델이 이론상 들어갑니다. AWS에서는 `g7e` 계열이 이 GPU를 씁니다.

문제는 "이론상 들어간다"와 "실제로 서빙된다" 사이의 간격입니다. 이 GPU는 데이터센터 GPU와 세 가지가 다릅니다.

| | RTX 6000 Pro (g7e) | H200 (p5en) | B200 (p6-b200) |
|---|---|---|---|
| GPU 메모리 | 96 GiB **GDDR7** | 141 GiB HBM3e | 179 GiB HBM3e |
| 메모리 대역폭 | **~1.8 TB/s** | ~4.8 TB/s | ~8 TB/s |
| GPU 간 연결 | **NVLink 없음, PCIe Gen5** | NVLink | NVLink |
| FP4 native | 지원 | 미지원 (marlin 경유) | 지원 |

그리고 **SM120 아키텍처는 커널 지원이 아직 고르지 않습니다.** 이 차이가 실제로 문제를 만듭니다.

- 추측 디코딩(DSpark)이 vLLM에서 크래시합니다. FlashInfer에 SM120용 커널이 없습니다
- FlashInfer 버전이 vLLM이 pin한 것보다 높아야 기동됩니다
- uv로 설치하면 CUDA 툴킷 관련 우회 4가지가 필요합니다

이 저장소는 그 문제들을 하나씩 원인까지 파고들어 [troubleshooting/](#documentation)에 정리했고, **실측 벤치마크로 "쓸 만한가"에 답합니다.**

결론부터 말하면 **잘 돌아갑니다.** 다만 대역폭 차이 때문에 토큰 생성 속도는 H200급에 못 미치고, KV 캐시 여유가 적어 컨텍스트를 길게 쓰면 동시 요청 수가 제한됩니다. 그 한계가 어디인지 숫자로 확인해 뒀습니다.

## Results

RTX 6000 Pro Blackwell ×2, TP=2, FP8 KV 캐시 구성입니다.

| 항목 | 값 |
|---|---|
| GPU당 메모리 사용 | 92.4 / 97.9 GiB (94%) |
| KV 캐시 | 7.75 GiB, **301,529 토큰** |
| 131K 컨텍스트 기준 동시 요청 | **2.3개** |
| 기동 시간 | 5~6분 (가중치 로딩 + CUDA 그래프 캡처) |

**처리량과 지연시간** (입력 204, 출력 200토큰)

| 동시성 | TTFT p50 | TTFT p95 | TPOT p50 | 출력 tok/s |
|---|---|---|---|---|
| 1 | 62.8ms | 64.0ms | 9.2ms | 105 |
| 8 | 210ms | 228ms | 17.1ms | 446 |
| 32 | 497ms | **10,458ms** | 33.2ms | **701** |

동시성을 32로 올리면 처리량은 6.7배가 되지만 TTFT p95는 163배 나빠집니다. **SLA는 p50이 아니라 p95로 잡아야 합니다.** 시나리오별 상세는 [benchmark.md](benchmark.md)에 있습니다.

**검증한 기능**: 일반 채팅, think-high 추론 모드, tool calling (uv와 Docker 양쪽)

**병목은 KV 캐시입니다.** 가중치 148.7 GiB가 192 GiB 중 대부분을 차지해 KV 캐시로 7.75 GiB만 남습니다. 컨텍스트를 65K로 줄이면 동시 요청이 4.6개, 32K면 9.2개로 늘어납니다. 프리셋 3종은 [tuning.md](tuning.md)에 정리했습니다.

## Is this for you

**적합합니다**

- GPU 메모리 96 GiB 카드 2장 이상 보유 (또는 AWS `g7e.12xlarge` 이상)
- 이 모델을 로컬이나 온프레미스에 직접 배포
- 데이터가 외부로 나가면 안 되는 환경

**다른 방법이 낫습니다**

| 상황 | 대안 |
|---|---|
| GPU 1장뿐 | GGUF Q2 양자화(92.3 GiB)로 llama.cpp 서빙 → [backends.md](backends.md) |
| 모델만 빠르게 써보고 싶다 | OpenRouter (`$0.09/$0.18` per 1M tokens) 또는 DeepSeek API |
| 매니지드 엔드포인트가 필요 | [sagemaker/](sagemaker/README.md) 또는 [AWS 공식 노트북](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/DeepSeek/DeepSeek-V4) |
| H100/H200/B200을 쓸 수 있다 | 이 가이드의 우회는 대부분 SM120 전용이라 불필요 |
| 혼자 실험용, GPU 여유 적음 | llama.cpp + DSpark (75 tok/s) → [llamacpp/](llamacpp/README.md) |

## Setup

두 방법 중 하나를 고릅니다. **서버가 뜨면 완전히 동일하게 동작합니다.**

| | Docker (권장) | uv |
|---|---|---|
| 격리 | 완전 격리 | 없음 (호스트에 직접) |
| 사전 준비 | Docker + GPU 패스스루 | `python3.12-dev`, sudo |
| 손이 가는 부분 | 없음 | CUDA 버전 정합, 심볼릭 링크 4가지 |
| 언제 | 처음 써보거나 호스트를 건드리지 않고 검증할 때 | 이미 uv/CUDA를 관리하고 있고 컨테이너 오버헤드 없이 개발할 때 |

**요구사항**

| 항목 | 값 |
|---|---|
| GPU | 96 GiB × 2 이상 (가중치 148.7 GiB가 1장엔 안 들어감) |
| 디스크 | 여유 200 GiB 이상 (가중치 155 GiB + 여유분) |
| vLLM | `0.25.0` 고정. `0.26.0`은 DeepSeek-V4 회귀 버그 있음 |
| 추측 디코딩 | vLLM에서는 비활성화. SM120에서 크래시하므로 필요하면 [llamacpp/](llamacpp/README.md) |

스크립트에 이미 반영된 우회는 `docker/serve.sh` 주석과 [troubleshooting/](#documentation)에 근거를 남겼습니다.

**모델 가중치 다운로드** (155 GiB, 공통)

```bash
uvx --from huggingface_hub hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /opt/dlami/nvme/models/DeepSeek-V4-Flash-0731
```

## Run

**Docker**

```bash
bash docker/serve.sh
```

**uv**

```bash
cd uv
bash setup.sh   # 최초 1회. CUDA 관련 4가지 문제를 순서대로 해결
bash serve.sh
```

`setup.sh`가 처리하는 문제(CUDA_HOME 미설정, nvcc/헤더 버전 불일치, Python.h 부재, `-lcudart` 링크 실패)의 원인은 [troubleshooting/uv-native-setup.md](troubleshooting/uv-native-setup.md)에 있습니다.

약 5~6분 후 이 로그가 나오면 준비 완료입니다.

```
INFO:     Application startup complete.
```

**동작 확인**

```bash
curl http://localhost:8000/v1/models

# non-think, think-high, tool-calling 세 가지를 한 번에
uv run --with openai python3 client_example.py
```

**다른 백엔드로도 서빙할 수 있습니다**

GGUF 양자화본을 쓰면 메모리를 줄일 수 있고, vLLM에서 막힌 DSpark 추측 디코딩도 동작합니다.

```bash
bash llamacpp/download.sh UD-Q2_K_XL   # GGUF 96.8 GiB (공통)

# llama.cpp: 약 30초 후 :8001
bash llamacpp/serve.sh
DSPARK=1 bash llamacpp/serve.sh        # 추측 디코딩 (단일 사용자에만 이득)

# Ollama: 병합 후 등록 (:11434)
bash ollama/merge.sh
bash ollama/serve.sh
```

| 백엔드 | c=8 처리량 | 특징 |
|---|---|---|
| vLLM FP8 | **407 tok/s** | 처리량 최상. 다중 사용자 서빙 |
| [llama.cpp](llamacpp/README.md) | 218 tok/s | 양자화로 메모리 절감, DSpark 동작, 로드 30초 |
| [Ollama](ollama/README.md) | 124 tok/s | 모델 관리 편의. 분할 GGUF 병합 필요 |

실측 비교와 선택 기준은 [backends.md](backends.md)를 보세요.

## Cost & cleanup

**GPU는 켜져 있는 동안 계속 과금됩니다.** 요청이 0건이어도 마찬가지입니다. 검증이 끝나면 반드시 정리하세요.

```bash
# Docker로 띄운 경우
docker ps                  # 실행 중인 컨테이너 확인
docker stop <container>

# GPU가 비었는지 확인
nvidia-smi --query-gpu=index,memory.used --format=csv
```

`docker/serve.sh`는 `--rm`으로 실행하므로 stop 시 컨테이너가 삭제됩니다. 재기동에 5~6분이 걸리니 잠시 쉴 때는 그대로 두는 편이 나을 수 있습니다.

**SageMaker 엔드포인트를 만들었다면** 삭제하지 않는 한 시간당 과금이 계속됩니다. 노트북 맨 아래 Cleanup 셀을 실행하세요([sagemaker/](sagemaker/README.md)).

**디스크도 확인하세요.** 가중치 155 GiB에 GGUF 양자화본까지 받으면 수백 GiB가 됩니다. 인스턴스 스토어(`/opt/dlami/nvme`)에 뒀다면 인스턴스 중지 시 소실되므로, 다시 받는 시간을 감안하세요.

**트래픽이 적다면 자체 서빙이 손해일 수 있습니다.** OpenRouter 기준 이 모델은 입력 $0.09, 출력 $0.18 per 1M tokens입니다. 하루 몇백 건 수준이면 API가 압도적으로 싸고, GPU를 꽉 채워 돌릴 때 자체 서빙이 유리해집니다.

## Documentation

**구성과 성능**

| 문서 | 내용 |
|---|---|
| [deepseek-v4-flash.md](deepseek-v4-flash.md) | 모델 아키텍처, 체크포인트 4종 차이 |
| [benchmark.md](benchmark.md) | 시나리오 5종 처리량과 지연시간 (p50/p95/p99) |
| [tuning.md](tuning.md) | vLLM 최적화. 프리셋 3종, 튜닝 절차, `/metrics` 진단 |
| [backends.md](backends.md) | vLLM vs llama.cpp vs Ollama 실측, GGUF 양자화 4종 |
| [serving.md](serving.md) | 모델별 GPU 사이징 계산 (V4-Pro, GLM-5.2, Kimi K3 비교) |
| [gateway.md](gateway.md) | API 게이트웨이 (LiteLLM / ALB / API Gateway) |

**실행 스크립트**

| 경로 | 내용 |
|---|---|
| `docker/serve.sh` | vLLM Docker 실행 |
| `uv/setup.sh`, `uv/serve.sh` | vLLM uv 설치와 실행 |
| [llamacpp/](llamacpp/README.md) | GGUF 다운로드, llama-server 실행, DSpark |
| [ollama/](ollama/README.md) | GGUF 병합, Ollama 실행 |
| [quantbench/](quantbench/README.md) | 레이턴시 측정 스크립트와 결과 JSON |
| `client_example.py` | OpenAI 호환 클라이언트 예제 |
| [sagemaker/](sagemaker/README.md) | SageMaker 실시간 엔드포인트 배포 |

**트러블슈팅**

[troubleshooting/](troubleshooting/README.md)에 7개 문서와 **에러 메시지로 찾는 표**가 있습니다. SM120 커널 미지원, FlashInfer 버전 불일치, uv 설치 문제, llama.cpp GPU 분할 실패, Ollama 제약을 다룹니다.

## References

- [vLLM 공식 recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
- [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [unsloth/DeepSeek-V4-Flash-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF) 양자화본

## Disclaimer

저자의 개인 실측 경험을 정리한 것으로, 재직 중인 회사의 공식 문서나 입장을 대변하지 않습니다. 내용이 공식 문서와 다를 경우 공식 문서가 우선합니다.

이 가이드의 수치는 특정 시점, 특정 환경(RTX 6000 Pro Blackwell ×2, vLLM 0.25.0, PCIe 연결)에서 측정한 값입니다. GPU 모델, 드라이버, 라이브러리 버전, 네트워크 구성에 따라 달라질 수 있습니다.

vLLM과 llama.cpp 버전, Docker 이미지 태그, DLC 이미지 태그, 리전 지원, 서비스 제약, 모델 가격은 자주 바뀌므로 배포 전에 재확인하세요. SM120 관련 커널 지원 문제는 상당수가 업스트림에서 수정 진행 중이므로, 최신 버전에서는 이 가이드의 우회가 불필요할 수 있습니다.
