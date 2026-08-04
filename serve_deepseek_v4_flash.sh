#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 서빙 (RTX 6000 Pro Blackwell x2, SM120, NVLink 없음/PCIe)
#
# 체크포인트: deepseek-ai/DeepSeek-V4-Flash-0731 (공식 릴리즈, 155.4 GiB, weights 148.7 GiB)
# vLLM 이미지: v0.25.0 고정 (vLLM 공식 recipe 권장값. v0.26.0은 DeepSeek-V4 FlashMLA
#   회귀 버그(#49922)가 있어 회피)
# 추측 디코딩: 사용 안 함. 0731 체크포인트는 MTP 헤드가 없고 DSpark 드래프트 모듈만
#   내장(mtp.0.main_norm/main_proj/hc_attn_* 등 DSpark 전용 키). DSpark는 RTX PRO
#   6000/SM120에서 FlashInfer 커널 지원 누락으로 크래시 — vllm#50720, 원인은
#   flashinfer#3989 미머지, GB10(SM121)에서만 로컬 패치로 검증됨. 상세: troubleshooting/dspark-sm120-crash.md
#
# GPU간 NVLink 없음(PIX/PCIe Gen5) -> TP8급 all-reduce에는 불리하나 TP=2라 문제 되지 않음.
# weights 148.7 GiB는 GPU 1장(96 GiB)에 못 들어가므로 TP=2 필수.
#
# FlashInfer 버전 핫픽스: v0.25.0/0.25.1이 pin한 flashinfer-python==0.6.13에는
# DeepSeek-V4 sparse-SWA decode가 호출하는 swa_topk_lens 등 kwarg가 없음(0.6.14부터
# 추가됨) -> TypeError로 warmup 크래시. vllm#48054. 0.6.14로 올리고 flashinfer-cubin은
# 0.6.13에 머물러 버전 체크에 걸리므로 FLASHINFER_DISABLE_VERSION_CHECK=1로 우회
# (JIT 컴파일로 커널 자체 생성, 실제로 SM120에서 검증된 방법).

set -euo pipefail

MODEL_DIR="/opt/dlami/nvme/models/DeepSeek-V4-Flash-0731"
IMAGE="vllm/vllm-openai:v0.25.0"
PORT="${PORT:-8000}"

docker run --rm --gpus all \
  --ipc=host \
  -p "${PORT}:8000" \
  -v "${MODEL_DIR}:/model" \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  --entrypoint bash \
  "${IMAGE}" \
  -c '
    pip install -q --no-deps "flashinfer-python==0.6.14" &&
    exec vllm serve /model \
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
      --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}"
  '
