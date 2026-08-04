# deepseek-v4-flash-serving-guide

RTX 6000 Pro Blackwell 2장에서 DeepSeek-V4-Flash-0731을 vLLM으로 서빙하는 가이드입니다. 구성 근거와 실제로 부딪힌 문제, 해결 과정을 함께 정리했습니다.

- 모델 아키텍처: [deepseek-v4-flash.md](deepseek-v4-flash.md)
- 성능 벤치마크: [benchmark.md](benchmark.md)
- SageMaker 배포: [sagemaker/](sagemaker/README.md)
- 트러블슈팅: [아래 섹션](#트러블슈팅) 또는 이슈별 문서

## 적용 대상

**적합한 경우**
- GPU 메모리 96 GiB 카드 2장 이상을 보유
- DeepSeek-V4-Flash를 로컬/온프레미스에 직접 배포
- vLLM을 Docker로 실행하는 데 익숙함

**적합하지 않은 경우**
- GPU 1장뿐이거나 메모리가 96 GiB 미만 — 가중치가 148.7 GiB라 최소 2장이 필요
- SageMaker 같은 매니지드 서비스로 배포하려는 경우 — [AWS 공식 노트북](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/DeepSeek/DeepSeek-V4)이 더 적합
- H100/H200/B200 등 데이터센터 GPU 사용 — 이 가이드의 우회 방법은 대부분 워크스테이션용 GPU(RTX 6000 Pro 등, SM120 계열) 전용

## 빠른 시작

### 1) 모델 가중치 다운로드 (155 GiB)

```bash
uvx --from huggingface_hub hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /opt/dlami/nvme/models/DeepSeek-V4-Flash-0731
```

### 2) 서버 실행

```bash
bash serve_deepseek_v4_flash.sh
```

가중치 로딩, GPU 워밍업, CUDA 그래프 캡처까지 약 5~6분 소요됩니다. 아래 로그가 출력되면 준비 완료입니다:

```
INFO:     Application startup complete.
```

### 3) 동작 확인

```bash
curl http://localhost:8000/v1/models
```

또는 준비된 클라이언트 예제로 non-think / think-high / tool-calling 세 가지를 한 번에 테스트:

```bash
uv run --with openai python3 client_example.py
```

## 요구사항

| 항목 | 값 | 비고 |
|---|---|---|
| GPU | 96 GiB × 2 이상 | 가중치 148.7 GiB가 1장엔 안 들어감 (2장 이상 필수) |
| 디스크 | 여유 200 GiB 이상 | 가중치 155 GiB + 여유분 |
| Docker | GPU 패스스루 가능 | `docker run --gpus all`이 동작해야 함 |
| vLLM 이미지 | `v0.25.0` 고정 | 최신(`v0.26.0`)은 DeepSeek-V4에 회귀 버그가 있음, 아래 참고 |

## 설계 근거

### vLLM 버전을 0.25.0으로 고정한 이유

최신 버전(`0.26.0`)에는 DeepSeek-V4 전용 회귀 버그가 있습니다(FlashMLA 커널이 긴 문맥에서 크래시). vLLM 공식 recipe도 DeepSeek-V4-Flash-0731 기준 `v0.25.0`을 권장하므로 이를 따랐습니다.

### 추측 디코딩을 비활성화한 이유

DeepSeek-V4-Flash-0731 체크포인트는 **DSpark** 방식의 추측 디코딩 모듈만 내장하고 있는데, RTX 6000 Pro에서 알려진 버그로 크래시합니다. 자세한 내용: [troubleshooting/dspark-sm120-crash.md](troubleshooting/dspark-sm120-crash.md)

### 컨테이너 시작 시 pip install을 추가로 실행하는 이유

vLLM 0.25.0 이미지에 포함된 FlashInfer 라이브러리 버전이 실제 코드가 요구하는 버전보다 낮아 서버 기동 중 크래시합니다. 컨테이너 시작 시마다 패치 버전으로 자동 업그레이드하도록 스크립트에 반영했습니다. 자세한 내용: [troubleshooting/flashinfer-version-mismatch.md](troubleshooting/flashinfer-version-mismatch.md)

## 파일 구성

| 파일 | 내용 |
|---|---|
| `serve_deepseek_v4_flash.sh` | 서버 실행 스크립트 (우회 방법 포함) |
| `client_example.py` | OpenAI 호환 클라이언트 예제 |
| `deepseek-v4-flash.md` | 모델 아키텍처 설명 (MoE, 양자화 등) |
| `benchmark.md` | 시나리오별 처리량/지연시간(p50/p95/p99) 벤치마크 |
| `sagemaker/` | SageMaker 실시간 엔드포인트 배포 노트북 |
| `troubleshooting/dspark-sm120-crash.md` | DSpark 크래시 원인과 해결 |
| `troubleshooting/flashinfer-version-mismatch.md` | FlashInfer 버전 크래시 원인과 해결 |

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `KeyError: model.layers.43.mtp_block.main_norm.weight` | `--speculative-config`에 `method:mtp`를 지정했지만 체크포인트엔 DSpark 모듈만 있음 | 추측 디코딩 설정 제거 (이 스크립트는 이미 반영됨) |
| `TypeError: ...got an unexpected keyword argument 'swa_topk_lens'` | FlashInfer 버전이 vLLM 코드보다 낮음 | [troubleshooting/flashinfer-version-mismatch.md](troubleshooting/flashinfer-version-mismatch.md) |
| `Check failed: num_tokens > 64` (DSpark 추측 디코딩 활성화 시) | RTX 6000 Pro 계열 GPU에서 DSpark 커널 지원 누락 | [troubleshooting/dspark-sm120-crash.md](troubleshooting/dspark-sm120-crash.md) — 추측 디코딩 비활성화 |
| `RuntimeError: Failed to infer device type` | GPU 없이(`--gpus all` 누락) 컨테이너 실행 | `docker run`에 `--gpus all` 확인 |

## 검증 결과

- GPU당 메모리 사용: 92.4 / 97.9 GiB
- KV 캐시: 301,529 토큰 (131K 컨텍스트 기준 동시 요청 2.3개)
- 검증 완료: 일반 채팅, think-high 추론 모드, tool calling
- 성능 벤치마크(처리량, TTFT/TPOT/ITL/E2EL p50/p95/p99): [benchmark.md](benchmark.md)

## 참고 링크

- [vLLM 공식 recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
- [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
