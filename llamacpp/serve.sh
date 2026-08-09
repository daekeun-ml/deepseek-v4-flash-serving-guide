#!/usr/bin/env bash
# llama-server로 GGUF 서빙 (RTX 6000 Pro Blackwell x2).
#
# 사용:
#   bash serve.sh                 # Q2, 기본 구성
#   QUANT=UD-IQ4_XS bash serve.sh # 다른 양자화
#   DSPARK=1 bash serve.sh        # 추측 디코딩 (단일 사용자일 때만 이득)
#
# vLLM 대비 특징 (실측 근거: ../backends.md)
#   - 로드 30초 (vLLM은 5~6분)
#   - 양자화로 메모리 절감. Q2는 92.3 GiB로 GPU 1장에도 들어감
#   - c=8 처리량 218 tok/s (vLLM 407 tok/s)
#   - DSpark 추측 디코딩 동작. vLLM+SM120에서는 크래시하는 기능
#
# 설치를 Docker로 하는 이유: Linux 사전 빌드 바이너리에 CUDA 버전이 없고,
# 소스 빌드는 nvcc가 필요하다. llama-cpp-python의 CUDA 휠도 cu125까지만 있어
# 이 서버(CUDA 13)와 맞지 않는다.

set -euo pipefail

GGUF_DIR="${GGUF_DIR:-/opt/dlami/nvme/models/gguf}"
QUANT="${QUANT:-UD-Q2_K_XL}"
PORT="${PORT:-8001}"
CTX="${CTX:-8192}"
# 슬롯 수. 동시 요청 수보다 작으면 큐 대기로 TTFT가 크게 나빠진다
# (실측: -np 4에 c=8 -> 4,931ms, -np 8 -> 417ms)
SLOTS="${SLOTS:-8}"
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
NAME="${NAME:-llamacpp}"

# 분할 GGUF는 첫 shard만 지정하면 나머지를 알아서 찾는다 (Ollama와 다른 점)
# `|| true`가 없으면 pipefail+set -e 때문에 아래 안내 메시지에 도달하지 못한다
FIRST_SHARD=$(ls "${GGUF_DIR}/${QUANT}"/*-00001-of-*.gguf 2>/dev/null | head -1 || true)
if [ -z "$FIRST_SHARD" ]; then
  echo "GGUF를 찾을 수 없습니다: ${GGUF_DIR}/${QUANT}/" >&2
  echo "download.sh로 먼저 받으세요." >&2
  exit 1
fi
MODEL="/gguf/${QUANT}/$(basename "$FIRST_SHARD")"

ARGS=(
  -m "$MODEL"
  --host 0.0.0.0 --port 8000
  -ngl 999                  # 전 레이어를 GPU로
  --tensor-split 1,1        # GPU 2장에 절반씩
                            # split-mode row/tensor는 이 GPU에서 로드 실패
                            # ("device CUDA0 does not support split buffers")
  -c "$CTX"
  -np "$SLOTS"
  --jinja                   # tool calling 활성화에 필요
)

# DSpark 추측 디코딩. c=1에서 +25%, c=8에서 -41%이므로 단일 사용자일 때만 켠다.
if [ "${DSPARK:-0}" = "1" ]; then
  DRAFT="${GGUF_DIR}/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
  if [ ! -f "$DRAFT" ]; then
    echo "드래프트 모델이 없습니다: $DRAFT" >&2
    echo "다음으로 받으세요 (10.9 GiB, 0731 저장소에만 있음):" >&2
    echo "  uvx --from huggingface_hub hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \\" >&2
    echo "    --include 'dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf' --local-dir ${GGUF_DIR}/dspark" >&2
    exit 1
  fi
  ARGS+=(
    -md "/gguf/dspark/$(basename "$DRAFT")"
    --spec-type draft-dspark   # draft-mtp가 아님. 0731은 DSpark 계열
    --spec-draft-n-max 3
    -ngld 999
  )
  echo "DSpark 추측 디코딩 활성화"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all \
  -p "${PORT}:8000" \
  -v "${GGUF_DIR}:/gguf:ro" \
  "$IMAGE" "${ARGS[@]}"

echo "기동 중 (${QUANT}, 약 30초)..."
for i in $(seq 1 60); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://localhost:${PORT}/health" 2>/dev/null)" = "200" ]; then
    echo "준비 완료: http://localhost:${PORT}/v1"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0
  fi
  # 컨테이너가 죽었으면 즉시 중단
  if ! docker ps -q --filter "name=${NAME}" --filter status=running | grep -q .; then
    echo "로드 실패:" >&2
    docker logs "$NAME" 2>&1 | grep " E " | tail -5 >&2
    exit 1
  fi
  sleep 5
done
echo "타임아웃. docker logs ${NAME}으로 확인하세요." >&2
exit 1
