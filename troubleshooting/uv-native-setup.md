# Docker 없이 uv로 서빙할 때 겪는 4가지 문제

> 도커 이미지(`vllm/vllm-openai:v0.25.0`)는 시스템에 CUDA 13.0 툴킷과 각종 빌드 도구가 이미 설치되어 있어 문제가 드러나지 않습니다. `uv`로 맨 시스템에 새로 설치하면, 도커가 대신 처리해주던 부분이 그대로 드러납니다. 아래 4가지를 순서대로 겪게 되며, `uv/setup.sh`는 이 순서 그대로 해결합니다.

## 배경 지식

### 도커와 uv의 근본적인 차이

도커 이미지 안에는 **시스템 레벨 CUDA 툴킷**(`/usr/local/cuda-13.0`, OS 이미지 빌드 시점에 apt로 설치)이 있고, `nvcc`가 거기서 실행됩니다. `uv`로 맨 시스템에 설치하면 그런 시스템 툴킷이 없으므로, torch/vLLM이 끌어오는 **pip 패키지 버전 CUDA 툴킷**(`nvidia-cuda-nvcc`, `nvidia-cuda-cccl` 등, `.venv` 안에 설치됨)만으로 전부 해결해야 합니다. 도커에선 이 pip 패키지들이 설치되어 있어도 실제로는 안 쓰이는 반면, uv 환경에서는 이게 유일한 CUDA 툴킷입니다.

## 문제 1. DeepGEMM이 nvcc를 못 찾음

### 증상

```
RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM support in the current vLLM environment.
```

### 원인

vLLM 내부의 DeepGEMM(런타임에 CUDA 커널을 즉석 컴파일하는 모듈)이 `CUDA_HOME` 환경변수를 찾는데, 아무 값도 없습니다. 도커에는 시스템 툴킷이 있어 `CUDA_HOME`이 필요 없었지만, uv 환경엔 pip가 설치한 nvcc가 있어도 그 위치를 가리키는 환경변수가 없습니다.

### 해결

`CUDA_HOME`을 uv 가상환경 안의 nvidia 패키지 경로로 직접 지정합니다.

```bash
export CUDA_HOME="$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cu13"
```

## 문제 2. nvcc와 CUDA 헤더 버전이 서로 안 맞음

### 증상

```
error: "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

### 원인

CUDA 13.x 툴킷은 `nvidia-cuda-nvcc`(컴파일러), `nvidia-cuda-cccl`(C++ 표준 라이브러리 헤더), `nvidia-cuda-crt`(C 런타임), `nvidia-cuda-runtime`(런타임 라이브러리) 등 여러 pip 패키지로 나뉘어 배포됩니다. 이 패키지들은 **서로 독립적으로 버전이 올라가서**, uv가 리졸버로 각각 최신 버전을 고르면 `nvcc`는 13.2.86인데 `cccl`은 13.3.3.4.1처럼 **서로 다른 CUDA 서브버전이 뒤섞입니다.** 컴파일러가 자신과 다른 버전의 헤더를 보면 컴파일을 거부합니다.

**구체적 원인**: `tilelang`이 의존하는 `cuda-tile` 패키지가 `nvidia-cuda-nvcc==13.2.86`을 정확히 고정하는데, 다른 CUDA 서브패키지들은 아무도 특정 버전을 고정하지 않아 uv가 최신 버전을 고릅니다.

### 해결

`nvcc`가 고정된 버전(`13.2.86`)에 나머지 서브패키지들을 맞춥니다. torch가 `nvidia-cuda-runtime==13.0.96`을 요구해서 `pyproject.toml`에는 넣을 수 없고, 설치 후 `--no-deps`로 오버라이드해야 합니다.

```bash
uv pip install --no-deps \
  "nvidia-cuda-cccl==13.2.86" \
  "nvidia-cuda-crt==13.2.86" \
  "nvidia-cuda-runtime==13.2.86"
```

## 문제 3. Triton이 gcc로 컴파일할 때 Python.h가 없음

### 증상

```
fatal error: Python.h: No such file or directory
```

### 원인

Triton은 일부 커널을 gcc로 직접 컴파일하는데, 이때 Python C API 헤더(`Python.h`)가 필요합니다. 도커 이미지엔 이미 설치되어 있었지만, 일반적인 Ubuntu 시스템에는 `python3-dev` 계열 패키지가 기본으로 깔려있지 않습니다.

### 해결

```bash
sudo apt-get install -y python3.12-dev
```

## 문제 4. FlashInfer 링크 단계에서 `-lcudart`를 못 찾음

### 증상

```
/usr/bin/ld: cannot find -lcudart: No such file or directory
```

### 원인

FlashInfer가 JIT 컴파일한 커널을 링크할 때 `-L<경로>/lib64 -lcudart`를 씁니다. 이건 전통적인 시스템 CUDA 툴킷 구조(`lib64` 디렉터리, 버전 없는 `libcudart.so` 심볼릭 링크)를 가정한 것인데, pip 패키지 버전 CUDA 툴킷은 디렉터리 이름이 `lib`(64 없음)이고, 파일도 버전이 붙은 `libcudart.so.13`만 있습니다.

### 해결

두 심볼릭 링크를 만들어 pip 패키지 구조를 전통적 구조처럼 보이게 만듭니다.

```bash
CU13_DIR=".venv/lib/python3.12/site-packages/nvidia/cu13"
ln -sfn lib "$CU13_DIR/lib64"
ln -sfn libcudart.so.13 "$CU13_DIR/lib/libcudart.so"
```

## 참고로, FlashInfer 버전 문제는 별개

위 4가지를 해결해도 [flashinfer-version-mismatch.md](flashinfer-version-mismatch.md)에서 설명하는 FlashInfer 버전 pin 문제는 그대로 남아있습니다 — 이건 도커든 uv든 vLLM 0.25.0을 쓰면 똑같이 겪는 문제라 별도 문서로 분리했습니다.

## 검증 결과

`uv/setup.sh` + `uv/serve.sh`로 도커 없이 완전히 서버가 기동되고, `/v1/chat/completions` 요청에 정상 응답하는 것을 확인했습니다. 실행 방법은 [../README.md](../README.md) 참고.
