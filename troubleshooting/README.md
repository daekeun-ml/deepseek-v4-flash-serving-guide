# 트러블슈팅

RTX 6000 Pro Blackwell(SM120)에서 겪은 문제와 원인입니다. 에러 메시지로 바로 찾으려면 아래 표를 보세요.

## 에러 메시지로 찾기

**vLLM**

| 증상 | 원인 | 해결 |
|---|---|---|
| `KeyError: model.layers.43.mtp_block.main_norm.weight` | `--speculative-config`에 `method:mtp`를 줬지만 체크포인트엔 DSpark만 있음 | 추측 디코딩 설정 제거 (스크립트에 반영됨) |
| `TypeError: ...unexpected keyword argument 'swa_topk_lens'` | FlashInfer 버전이 vLLM 코드보다 낮음 | [flashinfer-version-mismatch.md](flashinfer-version-mismatch.md) |
| `Check failed: num_tokens > 64` | SM120에서 DSpark 커널 지원 누락 | [dspark-sm120-crash.md](dspark-sm120-crash.md) |
| `RuntimeError: Failed to infer device type` | `--gpus all` 누락 | `docker run`에 추가 |

**vLLM, uv 설치 시에만**

| 증상 | 원인 | 해결 |
|---|---|---|
| `RuntimeError: ...requires DeepGEMM support` | `CUDA_HOME` 미설정 | [uv-native-setup.md](uv-native-setup.md) |
| `CUDA compiler and CUDA toolkit headers are incompatible` | nvcc와 CUDA 서브패키지 버전 불일치 | [uv-native-setup.md](uv-native-setup.md) |
| `fatal error: Python.h: No such file or directory` | `python3.12-dev` 미설치 | [uv-native-setup.md](uv-native-setup.md) |
| `cannot find -lcudart` | pip CUDA 툴킷의 디렉터리 구조 차이 | [uv-native-setup.md](uv-native-setup.md) |

**llama.cpp**

| 증상 | 원인 | 해결 |
|---|---|---|
| `device CUDA0 does not support split buffers` | SM120에서 `--split-mode row` 미지원 | [llamacpp-split-mode.md](llamacpp-split-mode.md) |
| GPU 메모리 부족으로 로드 실패 | 양자화가 GPU 용량 초과 | [llamacpp-cpu-moe-offload.md](llamacpp-cpu-moe-offload.md) |

**Ollama**

| 증상 | 원인 | 해결 |
|---|---|---|
| `split GGUF ... has 1 shards, expected 3` | 분할 GGUF 미지원 | [ollama-split-gguf.md](ollama-split-gguf.md) |
| `GGML_ASSERT(n_graph_inputs < ...)` | DSpark를 MTP로 실행 시도 | [ollama-dspark-unsupported.md](ollama-dspark-unsupported.md) |

## 문서 목록

| 문서 | 내용 |
|---|---|
| [dspark-sm120-crash.md](dspark-sm120-crash.md) | vLLM에서 DSpark가 크래시하는 원인. llama.cpp에서는 동작함 |
| [flashinfer-version-mismatch.md](flashinfer-version-mismatch.md) | vLLM이 pin한 FlashInfer 버전이 낮아 기동 실패 |
| [uv-native-setup.md](uv-native-setup.md) | uv 설치 시 CUDA 관련 4가지 문제 |
| [llamacpp-split-mode.md](llamacpp-split-mode.md) | GPU 병렬 분할이 SM120에서 실패. 처리량 격차의 원인 |
| [llamacpp-cpu-moe-offload.md](llamacpp-cpu-moe-offload.md) | `-ncmoe`로 GPU 메모리 절약. 절감분만큼 느려짐 |
| [ollama-split-gguf.md](ollama-split-gguf.md) | Ollama가 분할 GGUF를 거부. 병합 필요 |
| [ollama-dspark-unsupported.md](ollama-dspark-unsupported.md) | Ollama에서 DSpark 사용 불가 |

## 문제가 여기 없다면

기동 로그를 먼저 확인하세요. 대부분 원인이 여기 나옵니다.

```bash
docker logs <container> 2>&1 | grep -E " E |ERROR|error|Traceback" | head -20
```

vLLM은 `non-default args` 줄에서 실제 적용된 플래그를 확인할 수 있습니다. 지정한 값이 무시되거나 덮어써진 경우를 잡아낼 수 있습니다.

```bash
docker logs <container> 2>&1 | grep "non-default args"
```

부하 중 병목 진단은 [tuning.md](../docs/tuning.md)의 `/metrics` 절을 보세요.
