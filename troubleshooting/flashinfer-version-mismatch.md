# vLLM 0.25.0의 FlashInfer 버전 pin 불일치로 인한 크래시

> vLLM 이슈: [vllm-project/vllm#48054](https://github.com/vllm-project/vllm/issues/48054)

## 요약

vLLM 0.25.0 도커 이미지에 설치된 FlashInfer 라이브러리 버전이 실제 코드가 요구하는 버전보다 낮아, DeepSeek-V4 서빙 코드가 호출하는 함수의 인자와 실제 함수 시그니처가 일치하지 않았습니다. vLLM 도커 이미지의 버전 pin이 코드 변경을 따라가지 못한 것이 원인이며, 라이브러리를 상위 버전으로 업그레이드해 해결했습니다.

## 배경 지식

### FlashInfer

GPU에서 어텐션(attention) 연산을 처리하는 라이브러리입니다. vLLM은 어텐션 연산의 상당 부분을 FlashInfer에 위임합니다.

### 버전 pin

여러 소프트웨어 패키지가 서로 의존할 때, 호환성을 보장하기 위해 특정 버전으로 고정하는 것을 "버전 pin"이라 합니다. vLLM 0.25.0 도커 이미지는 FlashInfer를 `0.6.13` 버전으로 고정했습니다.

### 함수 시그니처(signature)

함수가 받는 인자의 이름과 순서를 말합니다. 함수의 새 버전에서 인자가 추가되었는데 이전 버전의 함수를 호출하는 코드가 그 인자를 넘기면, "알 수 없는 인자"라는 에러가 발생합니다. 이번 버그가 정확히 이 상황입니다.

## 원인

vLLM의 DeepSeek-V4 서빙 코드는 FlashInfer의 `trtllm_batch_decode_sparse_mla_dsv4` 함수를 호출할 때 `swa_topk_lens`라는 인자를 전달합니다. 이 인자는 FlashInfer **0.6.14**부터 추가된 것인데, vLLM 0.25.0 이미지에는 그보다 낮은 **0.6.13**이 고정되어 있었습니다. 즉 vLLM 코드는 0.6.14 기준으로 작성되어 있었지만 도커 이미지의 FlashInfer 버전 pin이 갱신되지 않은 상태였고, 이 불일치가 함수 호출 시점에 에러로 드러났습니다.

## 에러 메시지

```
TypeError: trtllm_batch_decode_sparse_mla_dsv4() got an unexpected keyword argument 'swa_topk_lens'
```

## 해결 방법

FlashInfer를 vLLM이 pin한 버전(0.6.13)보다 상위인 0.6.14로 업그레이드합니다.

```bash
pip install --no-deps "flashinfer-python==0.6.14"
```

FlashInfer는 파이썬 패키지(`flashinfer-python`)와 사전 컴파일된 GPU 커널 패키지(`flashinfer-cubin`)가 짝을 이루는데, 이 문서 작성 시점에 `flashinfer-cubin`의 0.6.14 배포가 아직 이루어지지 않아 파이썬 패키지만 올리면 버전 검증에서 실패합니다.

```
RuntimeError: flashinfer-cubin version (0.6.13) does not match flashinfer version (0.6.14).
```

환경변수로 이 검증을 건너뛰면, 사전 컴파일된 커널 대신 JIT(즉석 컴파일) 방식으로 대체됩니다.

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

이 가이드의 서빙 스크립트(`serve_deepseek_v4_flash.sh`)는 컨테이너 시작 시 위 두 단계를 자동으로 수행합니다.

## 영향 범위

이 문제는 특정 GPU에 한정되지 않으며, vLLM 0.25.0/0.25.1로 DeepSeek-V4를 서빙하는 모든 환경에서 발생할 수 있습니다. GitHub 이슈에서는 RTX 6000 Pro 같은 SM120 계열 GPU 사용자의 보고가 많았고(GLM-5.2에서도 동일 원인의 에러가 확인됨), 이 가이드의 환경(RTX 6000 Pro + DeepSeek-V4-Flash + vLLM 0.25.0)에서도 재현되었습니다.

## 검증 결과

패치 적용 후 서버가 정상 기동했고, KV 캐시 확보 및 추론 요청까지 확인했습니다. 전체 실행 흐름은 [../README.md](../README.md) 참고.

## 참고 링크

- [vllm-project/vllm#48054](https://github.com/vllm-project/vllm/issues/48054) 이 이슈
