#!/usr/bin/env bash
# uv 가상환경으로 DeepSeek-V4-Flash-0731 서빙 (도커 버전과 동일한 플래그/근거는 ../README.md 참고)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4-Flash-0731}"

source .venv/bin/activate
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_HOME="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13"

exec vllm serve "$MODEL_DIR" \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 131072 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
