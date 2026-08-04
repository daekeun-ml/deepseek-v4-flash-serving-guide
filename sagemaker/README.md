# DeepSeek-V4-Flash SageMaker 배포

로컬(RTX 6000 Pro ×2)에서 검증한 vLLM 서빙 구성을 SageMaker 실시간 엔드포인트로 옮기는 노트북입니다.

> ⚠️ SageMaker 실시간 엔드포인트는 삭제하기 전까지 시간당 계속 과금됩니다. `ml.p5en.48xlarge`(8×H200) 기준 시간당 비용이 높으므로, 테스트가 끝나면 반드시 노트북의 [Cleanup](deploy_endpoint.ipynb) 셀을 실행하세요.

## 적용 대상

**적합한 경우**
- [상위 폴더의 서빙 가이드](../README.md)로 DeepSeek-V4-Flash를 로컬에서 이미 검증했고, 매니지드 엔드포인트로 이전하려는 경우
- SageMaker 실시간 엔드포인트의 기본 개념(모델/엔드포인트 설정/엔드포인트)을 이해하고 있는 경우

**적합하지 않은 경우**
- 모델을 아직 로컬에서 실행해보지 않은 경우 — 먼저 [../README.md](../README.md)로 로컬 검증을 권장합니다. 이 노트북의 설정값 대부분이 그 검증 결과를 반영한 것입니다
- 서버리스/배치 추론이 필요한 경우 — 이 문서는 실시간 엔드포인트 전용입니다

## 빠른 시작

### 1) 배포

[`deploy_endpoint.ipynb`](deploy_endpoint.ipynb)를 위에서부터 순서대로 실행하세요:

1. 설치 및 SageMaker 세션 준비
2. 인스턴스·모델 변수 설정
3. vLLM 서빙 환경변수 구성 (성능 프리셋 3가지 중 택1)
4. 모델 생성 → 엔드포인트 배포 (가중치 로딩 포함 10~15분)
5. 추론 테스트 (기본 / reasoning / 스트리밍)

### 2) 정리 (반드시)

같은 노트북 맨 아래 **Cleanup** 셀을 실행하세요. endpoint → endpoint config → model 순서로 삭제합니다.

## 로컬 검증과의 차이

이 노트북의 설정은 로컬 서빙에서 실제로 확인한 결과를 반영합니다.

| 설정 | 로컬(RTX 6000 Pro) | SageMaker 노트북(H200) | 사유 |
|---|---|---|---|
| 인스턴스 | GPU 2장, 96 GiB×2 | `ml.p5en.48xlarge`, 8×H200 141 GiB | 노트북 기본값을 여유 있게 설정. 비용을 낮추려면 더 작은 인스턴스로 변경 가능 (아래 참고) |
| 추측 디코딩 | 비활성화 (DSpark가 SM120에서 크래시) | MTP 사용 | H200(SM90)에서는 DSpark도 정상 동작하지만, 로컬 검증과 조건을 맞추기 위해 MTP로 통일 |
| FlashInfer 버전 핫픽스 | 필요 (`0.6.14`로 수동 업그레이드) | 불필요 | 이 문제는 SM120류 GPU 전용이 아니라 vLLM 0.25.0/0.25.1의 버전 pin 자체 문제이므로, H200에서도 이론상 발생 가능 — DLC 이미지가 수정된 버전을 사용하는지 배포 전 확인 권장 ([../troubleshooting/flashinfer-version-mismatch.md](../troubleshooting/flashinfer-version-mismatch.md)) |

더 저렴한 인스턴스로 배포하려면: DeepSeek-V4-Flash 가중치는 148.7 GiB이므로 `ml.g6e.12xlarge`(2×L40S, 96 GiB)처럼 더 작은 인스턴스로도 TP=2 구성이 가능합니다. 사이징 계산은 [../serving.md](../serving.md) 참고.

## 성능 프리셋

노트북 3번 셀의 `PERFORMANCE_PRESET` 값으로 선택합니다.

| 프리셋 | 언제 쓰나 |
|---|---|
| `latency` (기본) | 응답 하나가 최대한 빨리 와야 할 때 |
| `balanced` | 지연시간과 처리량을 절충 |
| `throughput` | 동시 요청이 많고 총 처리량이 중요할 때 |

각 프리셋이 실제로 지연시간·처리량에 어떤 영향을 주는지는 로컬 벤치마크 결과([../benchmark.md](../benchmark.md))를 참고하세요 — 동시성이 올라가면 p50보다 p95/p99가 훨씬 급격히 나빠진다는 점이 프리셋 선택에도 그대로 적용됩니다.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `deploy_endpoint.ipynb` | 배포 → 추론 테스트 → cleanup까지 전체 흐름 |

## 참고 링크

- [../README.md](../README.md) — 로컬 vLLM 서빙 가이드 (이 노트북의 설정 근거)
- [../troubleshooting/dspark-sm120-crash.md](../troubleshooting/dspark-sm120-crash.md) — DSpark가 SM120에서 왜 크래시하는지
- [../troubleshooting/flashinfer-version-mismatch.md](../troubleshooting/flashinfer-version-mismatch.md) — FlashInfer 버전 불일치 문제
- [aws-samples/sagemaker-genai-hosting-examples](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/DeepSeek/DeepSeek-V4) — 이 노트북이 참고한 AWS 공식 예제
