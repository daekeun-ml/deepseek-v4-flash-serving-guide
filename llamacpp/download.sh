#!/usr/bin/env bash
# unsloth GGUF 양자화본 다운로드.
#
# 4종 합계 약 526 GiB. 실측 6분 24초 소요(대역폭에 따라 다름).
# 인스턴스 스토어(/opt/dlami/nvme)는 인스턴스 중지 시 소실되므로 루트 볼륨 용량을
# 아끼려면 여기, 영구 보관이 필요하면 EBS 경로로 DEST를 바꿀 것.
#
# 양자화별 GPU 메모리 실측값은 ../backends.md 참고. Q2만 GPU 1장(96 GiB)에 들어간다.

set -euo pipefail

DEST="${DEST:-/opt/dlami/nvme/models/gguf}"
REPO="${REPO:-unsloth/DeepSeek-V4-Flash-GGUF}"
# 인자로 양자화를 지정하지 않으면 4종 전부
QUANTS=("${@:-UD-Q2_K_XL UD-Q3_K_XL UD-IQ4_XS UD-Q8_K_XL}")

mkdir -p "$DEST"

for Q in ${QUANTS[*]}; do
  echo "=== [$(date '+%H:%M:%S')] $Q ==="
  uvx --from huggingface_hub hf download "$REPO" \
    --include "${Q}/*" \
    --local-dir "$DEST" \
    --max-workers 8
  du -sh "$DEST/$Q"
done

echo "=== 완료 [$(date '+%H:%M:%S')] ==="
df -h "$DEST" | tail -1
