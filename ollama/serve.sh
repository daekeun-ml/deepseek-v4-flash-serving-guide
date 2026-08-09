#!/usr/bin/env bash
# Ollama로 GGUF 서빙 (RTX 6000 Pro Blackwell x2).
#
# 사용:
#   bash serve.sh          # 서버 기동 + 모델 등록까지
#   bash serve.sh --skip-create   # 이미 등록했으면 기동만
#
# llama-server 대비 제약 (실측 근거: ../backends.md)
#   - 분할 GGUF를 읽지 못한다. 단일 파일로 병합 필요 (merge.sh)
#   - 등록 시 파일을 자기 저장소로 복사한다. 96.8 GiB가 이중으로 존재
#   - DSpark 추측 디코딩 불가. --spec-type을 draft-mtp로 고정 호출해 크래시
#   - c=8 처리량 124 tok/s (llama-server 218, vLLM 407)
#
# 그래도 쓰는 이유: `ollama run` 한 줄로 끝나고 모델 관리가 편하다.

set -euo pipefail

GGUF_DIR="${GGUF_DIR:-/opt/dlami/nvme/models/gguf}"
MERGED="${MERGED:-${GGUF_DIR}/merged/ds-q2.gguf}"
MODEL_NAME="${MODEL_NAME:-ds-q2}"
PORT="${PORT:-11434}"
CTX="${CTX:-8192}"
# 기본값이 1이라 동시 요청을 받으려면 반드시 올려야 한다
PARALLEL="${PARALLEL:-8}"
NAME="${NAME:-ollama}"
# 모델 저장소를 인스턴스 스토어에 둔다. 루트 볼륨에 96.8 GiB를 쓰지 않도록
OLLAMA_HOME="${OLLAMA_HOME:-${GGUF_DIR}/ollama_home}"

if [ ! -f "$MERGED" ]; then
  echo "병합된 GGUF가 없습니다: $MERGED" >&2
  echo "merge.sh로 먼저 병합하세요 (Ollama는 분할 GGUF를 읽지 못함)." >&2
  exit 1
fi

mkdir -p "$OLLAMA_HOME"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all \
  -p "${PORT}:11434" \
  -v "${GGUF_DIR}:/gguf:ro" \
  -v "${OLLAMA_HOME}:/root/.ollama" \
  -e OLLAMA_NUM_PARALLEL="$PARALLEL" \
  -e OLLAMA_KEEP_ALIVE=30m \
  ollama/ollama:latest >/dev/null

echo "서버 기동 중..."
for i in $(seq 1 30); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://localhost:${PORT}/api/version" 2>/dev/null)" = "200" ]; then
    break
  fi
  sleep 2
done

if [ "${1:-}" != "--skip-create" ]; then
  echo "모델 등록 중 (96.8 GiB 복사, 수 분 소요)..."
  docker exec "$NAME" sh -c "cat > /tmp/Modelfile <<EOF
FROM /gguf/$(realpath --relative-to="$GGUF_DIR" "$MERGED")
PARAMETER num_ctx ${CTX}
EOF
ollama create ${MODEL_NAME} -f /tmp/Modelfile"
fi

echo "모델 로드 중 (약 2분)..."
curl -s -m 900 -o /dev/null "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"max_tokens\":300,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"

echo "준비 완료: http://localhost:${PORT}/v1  (model=${MODEL_NAME})"
curl -s -m 5 "http://localhost:${PORT}/api/ps" | python3 -c "
import json, sys
for m in json.load(sys.stdin).get('models', []):
    print(f\"  {m['name']}  VRAM {m.get('size_vram',0)/1e9:.1f} GB  ctx {m.get('context_length')}\")
"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
