#!/usr/bin/env bash
# uv만으로 DeepSeek-V4-Flash-0731 서빙 환경을 구성 (도커 불필요).
#
# 겪은 문제와 이유는 README.md의 "uv로 재현할 때 겪는 문제" 표 참고. 요약:
# 1) DeepGEMM이 nvcc를 못 찾음 -> CUDA_HOME을 pip가 설치한 nvidia/cu13 디렉터리로 지정
# 2) nvcc(13.2.86)와 cccl/crt/runtime(uv가 고른 13.3.x/13.0.x)이 서로 안 맞아
#    "CUDA compiler and CUDA toolkit headers are incompatible" 컴파일 에러
#    -> cccl/crt/runtime을 nvcc와 같은 13.2.86으로 강제 고정 (--no-deps, torch가
#       13.0.96을 요구해서 pyproject.toml에는 못 넣고 설치 후 오버라이드해야 함)
# 3) Triton이 gcc로 컴파일할 때 Python.h가 없어서 실패 -> python3.12-dev 설치
# 4) FlashInfer가 링크할 때 -lcudart를 못 찾음 (pip 패키지는 lib64가 아니라 lib에,
#    버전 붙은 이름으로만 있음) -> lib64 심볼릭 링크 + libcudart.so 심볼릭 링크 생성
# 5) FlashInfer가 vLLM 0.25.0이 pin한 0.6.13보다 높은 버전을 요구
#    (swa_topk_lens 등 kwarg) -> 0.6.14로 업그레이드 + FLASHINFER_DISABLE_VERSION_CHECK=1
#    (도커 버전과 동일 원인, 상세: ../troubleshooting/flashinfer-version-mismatch.md)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/6] python3.12-dev 설치 확인 (Triton JIT 컴파일에 필요)"
if [ ! -f /usr/include/python3.12/Python.h ]; then
  sudo apt-get update -qq && sudo apt-get install -y python3.12-dev
fi

echo "[2/6] uv sync (vllm==0.25.0 + 의존성)"
uv sync

CU13_DIR="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13"

echo "[3/6] CUDA 서브패키지 버전을 nvcc와 맞춤 (cccl/crt/runtime -> 13.2.86)"
uv pip install --no-deps \
  "nvidia-cuda-cccl==13.2.86" \
  "nvidia-cuda-crt==13.2.86" \
  "nvidia-cuda-runtime==13.2.86"

echo "[4/6] FlashInfer를 0.6.14로 업그레이드 (vLLM이 pin한 0.6.13에는 필요한 kwarg가 없음)"
uv pip install --no-deps "flashinfer-python==0.6.14"

echo "[5/6] lib64 심볼릭 링크 + libcudart.so 심볼릭 링크 생성 (FlashInfer 링크 단계에 필요)"
ln -sfn lib "$CU13_DIR/lib64"
ln -sfn libcudart.so.13 "$CU13_DIR/lib/libcudart.so"

echo "[6/6] 완료. serve.sh로 서버를 실행하세요."
