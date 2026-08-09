# Ollama 서빙

GGUF를 Ollama로 서빙합니다. 실측 비교는 [backends.md](../docs/backends.md)를 보세요.

## 빠른 시작

```bash
bash ../llamacpp/download.sh UD-Q2_K_XL   # GGUF 받기
bash merge.sh                             # 단일 파일로 병합 (Ollama 필수)
bash serve.sh                             # 서버 기동 + 모델 등록
```

등록이 끝나면 `http://localhost:11434/v1`에서 OpenAI 호환 API를 제공합니다.

```bash
bash serve.sh --skip-create   # 이미 등록했으면 기동만
```

## 옵션

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_NAME` | `ds-q2` | Ollama에 등록할 이름 |
| `MERGED` | `<GGUF_DIR>/merged/ds-q2.gguf` | 병합된 GGUF 경로 |
| `PORT` | `11434` | |
| `CTX` | `8192` | 컨텍스트 길이 |
| `PARALLEL` | `8` | `OLLAMA_NUM_PARALLEL`. **기본값이 1이라 반드시 올려야 함** |
| `OLLAMA_HOME` | `<GGUF_DIR>/ollama_home` | 모델 저장소. 루트 볼륨을 피하려고 nvme에 둠 |

## 제약 3가지

**1. 분할 GGUF를 읽지 못합니다.**

```
Error: split GGUF "...00001-of-00003.gguf" has 1 shards, expected 3
```

`merge.sh`로 병합해야 합니다(Q2 기준 약 1분). llama-server는 첫 shard만 지정하면 나머지를 알아서 찾으므로 이 단계가 없습니다.

**2. 등록 시 파일을 복사합니다.** 96.8 GiB가 원본과 Ollama 저장소에 이중으로 존재합니다. 병합 파일까지 합치면 같은 모델이 세 번 디스크를 차지합니다.

**3. DSpark 추측 디코딩이 불가능합니다.** Modelfile에 `DRAFT` 지시어로 등록은 됩니다.

```
FROM /gguf/merged/ds-q2.gguf
DRAFT /gguf/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
```

manifest에도 정상 반영되지만(`application/vnd.ollama.image.draft`, 10.90 GB) 추론 시 크래시합니다.

```
[spec] failed to measure draft model memory: failed to create llama_context from model
GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed
```

Ollama가 내부 llama-server를 호출할 때 `--spec-type draft-mtp`로 고정하기 때문입니다. 0731 체크포인트는 DSpark 계열이라 맞지 않고, Ollama는 `--spec-type`을 노출하지 않아 우회할 수 없습니다. DSpark가 필요하면 [../llamacpp/](../llamacpp/README.md)를 쓰세요.

## 성능

| | c=1 TTFT | c=8 TTFT p50 | c=8 TTFT p95 | c=8 처리량 |
|---|---|---|---|---|
| Ollama Q2 | 206ms | 590ms | 7,430ms | 124 tok/s |
| llama-server Q2 | 71ms | 417ms | 654ms | 218 tok/s |
| vLLM FP8 | 51ms | 186ms | 2,011ms | 407 tok/s |

셋 중 가장 느리고 꼬리 지연도 가장 나쁩니다. 그래도 쓰는 이유는 모델 관리가 편하기 때문입니다.

## 알아둘 것

**첫 요청에 로딩이 포함됩니다.** 96.8 GiB 모델은 약 2분 걸립니다. `serve.sh`가 워밍업까지 하지만, 직접 측정할 때는 `/api/ps`로 로드 완료를 확인하세요.

```bash
curl -s http://localhost:11434/api/ps | python3 -m json.tool
# size_vram이 모델 크기와 비슷하면 로드 완료
```

**reasoning 필드명이 다릅니다.** vLLM과 llama.cpp는 `reasoning_content`인데 Ollama는 **`reasoning`**입니다. 이 모델은 기본이 thinking 모드라 토큰이 대부분 이 필드로 나옵니다.

**`ollama` 클라우드 모델과 다릅니다.** Ollama 라이브러리의 `deepseek-v4-flash`는 태그가 `:cloud`, `:0731-cloud`, `:preview-cloud`뿐이고 모두 Ollama 서버에서 실행됩니다. 내 GPU로 돌리려면 이 문서처럼 GGUF를 직접 등록해야 합니다.
