# DSpark 추측 디코딩이 RTX 6000 Pro(SM120)에서 크래시하는 원인

> vLLM 이슈: [vllm-project/vllm#50720](https://github.com/vllm-project/vllm/issues/50720)

## 요약

DSpark 추측 디코딩을 활성화하면 RTX 6000 Pro 같은 SM120 GPU에서 커널 하나가 "이 입력 형태는 준비되어 있지 않다"며 크래시합니다. vLLM 자체의 버그가 아니라, vLLM이 의존하는 FlashInfer 라이브러리의 커널 지원 누락입니다. 이 가이드에서는 해당 기능을 비활성화하고 서빙했습니다.

## 배경 지식

### 추측 디코딩(speculative decoding)

언어모델은 토큰(단어 조각)을 한 번에 하나씩만 생성하며, 이는 지연시간의 주요 원인입니다. **추측 디코딩은 작고 빠른 초안(draft) 모델이 여러 토큰을 먼저 예측해두고, 크고 정확한 본 모델이 그 예측을 한 번에 검증**하는 방식으로 속도를 높이는 기법입니다. 예측이 맞으면 여러 토큰을 한 번에 확정할 수 있어 빨라지고, 틀리면 그 지점부터 본 모델이 다시 생성합니다.

### DSpark

DeepSeek이 만든 추측 디코딩 방식입니다. 초안 모델을 별도로 학습시켜 체크포인트에 함께 포함하는 방식이며, 이 초안 모델이 내부적으로 어텐션(attention) 연산을 수행할 때 **`markov_rank=256`이라는 자체 설정값**을 사용합니다. 이 `256`이라는 값이 이번 버그의 핵심입니다(아래에서 다시 등장합니다).

DSpark와 대비되는 방식이 **MTP**(Multi-Token Prediction)이며, 체크포인트마다 둘 중 하나만 내장됩니다. 이 가이드에서 사용한 `DeepSeek-V4-Flash-0731`은 DSpark만 내장하고 있어, 추측 디코딩을 쓰려면 DSpark를 사용해야 합니다.

### FlashInfer와 커널 디스패치 테이블

**FlashInfer**는 GPU에서 어텐션 연산을 빠르게 처리해주는 라이브러리로, vLLM이 내부적으로 이걸 불러서 씁니다. GPU 코드(커널)는 "이런 모양의 입력이 들어올 것"을 미리 알고 컴파일해둬야 빠릅니다. 그래서 FlashInfer는 자주 쓰이는 입력 모양들을 미리 정리한 **"디스패치 테이블"**을 갖고 있고, 새 연산 요청이 들어오면 이 테이블에서 맞는 커널을 찾아 실행합니다.

문제는 **이 테이블에 없는 모양이 들어오면 어떻게 처리할지**입니다. 그리고 이번 버그가 바로 그 케이스입니다.

## 원인

FlashInfer의 디스패치 테이블은 `topk` 값이 **128, 512, 1024**인 경우에만 디코드용 커널을 준비해두고 있습니다. 그런데 DSpark 초안 모델은 자체 설정값 `markov_rank=256`을 그대로 `topk` 인자로 전달합니다. **256은 테이블에 없는 값**입니다.

테이블에서 값을 찾지 못하면 FlashInfer는 별도 오류 없이 prefill(초기 처리)용 경로로 넘어갑니다. 이 경로는 "토큰 수가 64개를 넘어야 한다"는 전제를 가지고 있는데, 실제로는 소량(토큰 5~7개)의 디코드 요청이라 이 조건 검사에서 실패합니다.

## 실제 에러 메시지

```
tvm.error.InternalError: Check failed: num_tokens > 64 (7 vs. 64) :
Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv4;
got num_tokens=7
```

에러 메시지의 `num_tokens=7`은 실제 원인과 무관합니다. 진짜 원인은 `topk=256`이 디스패치 테이블에 없다는 것이고, `num_tokens=7`은 그 결과로 잘못된 경로에 진입했을 때 걸리는 별개의 검사입니다. 메시지만으로 디버깅하면 원인을 잘못 짚기 쉬운데, 실제로 이 이슈의 참여자들도 며칠간 TP/DP/block-size 등 무관한 변수를 의심했습니다.

## 발생 조건

- 모델: `DeepSeek-V4-Flash-0731` (또는 DSpark 모듈이 내장된 다른 V4 체크포인트)
- GPU: RTX 6000 Pro, GB10, GB300 등 "SM120/SM121" 계열 (NVIDIA Blackwell의 워크스테이션/컨슈머용 세대)
- 조건: `--speculative-config`에서 `"method":"dspark"`를 켰을 때만

MTP 방식을 쓰거나 추측 디코딩을 안 쓰면 이 문제 자체가 발생하지 않습니다.

## 해결 상태

- 원인은 규명되었고, 수정은 vLLM이 아니라 FlashInfer 쪽에서 진행 중입니다: [flashinfer#3989](https://github.com/flashinfer-ai/flashinfer/issues/3989). 디스패치 테이블에 `topk=256` 항목을 추가하는 패치이며, 아직 정식 배포되지 않았습니다.
- 커뮤니티가 이 패치를 로컬에 직접 적용해 GB10(SM121) 환경에서는 검증에 성공했습니다.
- RTX 6000 Pro(SM120)에서 검증된 사례는 아직 없습니다.
- 패치를 적용해도 추가 의존성 문제(tilelang 버전, apache-tvm-ffi 버전 충돌 등)가 발생합니다.

## 우회 방법

vLLM에서는 추측 디코딩을 비활성화했습니다. DSpark 없이도 모델은 정상 동작하며, 추측 디코딩으로 얻는 속도 향상만 없는 상태입니다. 실행 스크립트는 [../README.md](../README.md) 참고.

### llama.cpp에서는 같은 GPU에서 동작합니다

이 문제는 vLLM이 FlashInfer에 의존하기 때문에 생깁니다. **llama.cpp는 자체 CUDA 커널을 쓰므로 같은 RTX 6000 Pro에서 DSpark가 정상 로드됩니다.**

```bash
DSPARK=1 bash llamacpp/serve.sh
```

실측 효과입니다(Q2 양자화 기준).

| 구성 | c=1 처리량 | c=8 처리량 |
|---|---|---|
| 기본 | 60.1 tok/s | 226.5 tok/s |
| DSpark | **75.3 tok/s (+25%)** | 132.9 tok/s (-41%) |

드래프트 수락률은 56.1%였습니다(32/57 수락, 평균 2.68토큰). 동시 요청이 많으면 드래프트 계산이 낭비되므로 단일 사용자일 때만 켜세요.

단 GGUF 양자화본을 써야 하고 처리량은 vLLM보다 낮습니다. 선택 기준은 [../backends.md](../backends.md), 사용법은 [../llamacpp/README.md](../llamacpp/README.md)를 보세요.

**Ollama에서는 불가능합니다.** 추측 디코딩 방식을 MTP로 고정 호출해 크래시합니다: [ollama-dspark-unsupported.md](ollama-dspark-unsupported.md)

| 백엔드 | DSpark |
|---|---|
| llama-server | 동작 (`--spec-type draft-dspark`) |
| Ollama | 등록은 되나 추론 시 크래시 |
| vLLM | 로드 실패 (이 문서의 원인) |

## 참고 링크

- [vllm-project/vllm#50720](https://github.com/vllm-project/vllm/issues/50720) 이 이슈
- [flashinfer-ai/flashinfer#3989](https://github.com/flashinfer-ai/flashinfer/issues/3989) 실제 수정이 올라갈 곳 (미머지)
- [flashinfer-ai/flashinfer#3828](https://github.com/flashinfer-ai/flashinfer/issues/3828) 최초 발견 리포트
