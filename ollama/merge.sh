#!/usr/bin/env bash
# 분할 GGUF를 단일 파일로 병합.
#
# Ollama는 분할 GGUF를 읽지 못한다:
#   Error: split GGUF "...00001-of-00003.gguf" has 1 shards, expected 3
# llama-server는 첫 shard만 지정하면 나머지를 알아서 찾으므로 이 단계가 불필요하다.
#
# 병합 도구(llama-gguf-split)는 server-cuda 이미지에 없고 full 이미지에만 있다.
# Q2(96.8 GiB) 기준 약 1분 소요. 원본과 별도로 저장되므로 디스크를 두 배로 쓴다.

set -euo pipefail

GGUF_DIR="${GGUF_DIR:-/opt/dlami/nvme/models/gguf}"
QUANT="${QUANT:-UD-Q2_K_XL}"
OUT_NAME="${OUT_NAME:-ds-$(echo "$QUANT" | sed 's/UD-//; s/_.*//' | tr 'A-Z' 'a-z').gguf}"

FIRST_SHARD=$(ls "${GGUF_DIR}/${QUANT}"/*-00001-of-*.gguf 2>/dev/null | head -1 || true)
if [ -z "$FIRST_SHARD" ]; then
  echo "GGUF를 찾을 수 없습니다: ${GGUF_DIR}/${QUANT}/" >&2
  echo "../llamacpp/download.sh로 먼저 받으세요." >&2
  exit 1
fi

mkdir -p "${GGUF_DIR}/merged"
OUT="${GGUF_DIR}/merged/${OUT_NAME}"

if [ -f "$OUT" ]; then
  echo "이미 존재합니다: $OUT ($(du -h "$OUT" | cut -f1))"
  exit 0
fi

echo "병합: ${QUANT} -> ${OUT_NAME}"
docker run --rm --entrypoint /app/llama-gguf-split \
  -v "${GGUF_DIR}:/gguf" \
  ghcr.io/ggml-org/llama.cpp:full \
  --merge "/gguf/${QUANT}/$(basename "$FIRST_SHARD")" \
          "/gguf/merged/${OUT_NAME}"

echo "완료: $OUT ($(du -h "$OUT" | cut -f1))"
df -h "$GGUF_DIR" | tail -1
