# Ollama에서 DSpark 추측 디코딩이 크래시하는 원인

## 요약

Ollama의 Modelfile에는 드래프트 모델을 지정하는 `DRAFT` 지시어가 있고, DSpark 모델을 등록하는 것까지는 **성공합니다.** 하지만 추론을 시도하면 크래시합니다.

```
[spec] failed to measure draft model memory: failed to create llama_context from model
GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed
llama-server terminated: signal: aborted (core dumped)
```

원인은 Ollama가 내부 llama-server를 호출할 때 **추측 디코딩 방식을 `draft-mtp`로 고정**한다는 점입니다. 0731 체크포인트는 MTP가 아니라 DSpark 계열이라 맞지 않습니다.

Ollama는 이 설정을 노출하지 않아 우회할 수 없습니다. DSpark가 필요하면 [../llamacpp/](../llamacpp/README.md)의 llama-server를 직접 쓰세요.

## 배경 지식

### MTP와 DSpark

둘 다 추측 디코딩 방식이지만 구조가 다릅니다. 자세한 설명은 [dspark-sm120-crash.md](dspark-sm120-crash.md)에 있고, 여기서는 차이만 정리합니다.

| | MTP | DSpark |
|---|---|---|
| 구조 | 본 모델에 예측 헤드를 **내장** | 별도 드래프트 모델을 **함께 배포** |
| 파일 | 본 모델 안에 텐서로 존재 | 독립 GGUF 파일 (10.9 GB) |

**체크포인트마다 둘 중 하나만 들어 있습니다.** `DeepSeek-V4-Flash-0731`은 DSpark 계열이고, preview 버전은 MTP 계열입니다.

### llama.cpp의 `--spec-type`

llama.cpp는 여러 방식을 지원하고, 실행할 때 어느 것인지 명시해야 합니다.

```
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,
             ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
```

DSpark를 쓰려면 `draft-dspark`를 지정합니다. 이 저장소에서 llama-server로 직접 실행할 때는 이렇게 했고 정상 동작했습니다.

```bash
--spec-type draft-dspark --spec-draft-n-max 3 -ngld 999
```

## 재현

Modelfile에 `DRAFT`를 씁니다.

```
FROM /gguf/merged/ds-q2.gguf
DRAFT /gguf/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
```

등록은 성공합니다.

```bash
ollama create testdraft -f Modelfile
# success
```

manifest에도 드래프트 레이어가 정상 기록됩니다.

```
application/vnd.ollama.image.model    96.83 GB
application/vnd.ollama.image.draft    10.90 GB
```

여기까지는 문제가 없습니다. 추론을 요청하면 크래시합니다.

## 원인

Ollama 로그에서 내부 llama-server 호출 명령을 볼 수 있습니다.

```bash
docker logs ollama 2>&1 | grep "starting llama-server"
```

```
/usr/lib/ollama/llama-server --model <blob> --port 36627 ...
  --spec-type draft-mtp                    <- 여기
  --spec-draft-n-max 4
  --spec-draft-backend-sampling
  --spec-draft-model <draft-blob>
  --flash-attn auto -b 2048 -ub 2048
```

**`--spec-type draft-mtp`로 고정되어 있습니다.** DSpark 모델에는 MTP 텐서가 없으므로 컨텍스트 생성이 실패하고, 이어서 그래프 스케줄러 단정문이 깨집니다.

```
[spec] failed to measure draft model memory: failed to create llama_context from model
GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed
```

Ollama가 방식을 자동 판별하지 않고 MTP로 단정하는 것이 문제입니다. Modelfile에도, 환경변수에도 `--spec-type`에 해당하는 설정이 없습니다.

`ollama create --draft-quantize`나 `PARAMETER draft_num_predict`는 드래프트 토큰 수와 양자화만 조절하고 방식은 바꾸지 못합니다.

## 해결

**llama-server를 직접 쓰는 것이 유일한 방법입니다.**

```bash
DSPARK=1 bash llamacpp/serve.sh
```

실측 효과입니다.

| 구성 | c=1 처리량 | c=8 처리량 |
|---|---|---|
| 기본 | 60.1 tok/s | 226.5 tok/s |
| DSpark | **75.3 tok/s (+25%)** | 132.9 tok/s (-41%) |

드래프트 수락률은 56.1%였습니다(32/57 수락, 평균 2.68토큰).

```
slot print_timing: draft acceptance = 0.56140 (32 accepted / 57 generated), mean len = 2.68
```

동시 요청이 많으면 드래프트 계산이 낭비되므로 단일 사용자일 때만 켜세요.

## MTP 체크포인트라면 될 수도 있습니다

Ollama가 `draft-mtp`로 고정한다는 것은, **MTP를 내장한 체크포인트에서는 정상 동작할 가능성**을 뜻합니다. preview 계열이 MTP를 씁니다.

| 체크포인트 | 방식 | Ollama 추측 디코딩 |
|---|---|---|
| `DeepSeek-V4-Flash-0731` | DSpark | 불가 |
| `DeepSeek-V4-Flash` (preview) | MTP | 가능할 수 있음 (미검증) |
| `nvidia/DeepSeek-V4-Flash-NVFP4` | MTP | 가능할 수 있음 (미검증) |

다만 0731이 preview보다 agentic 성능이 크게 향상된 정식 릴리즈이므로, 추측 디코딩을 위해 preview로 되돌리는 것은 권하지 않습니다. 체크포인트별 차이는 [deepseek-v4-flash.md](../docs/deepseek-v4-flash.md)를 보세요.

## 세 백엔드 정리

| 백엔드 | DSpark |
|---|---|
| **llama-server** | **동작** (`--spec-type draft-dspark` 명시) |
| Ollama | 등록은 되지만 추론 시 크래시 |
| vLLM | 로드 자체가 실패 (SM120 커널 미지원, [dspark-sm120-crash.md](dspark-sm120-crash.md)) |

같은 GPU에서 llama.cpp만 DSpark를 쓸 수 있습니다. vLLM은 FlashInfer 커널에 의존하는데 SM120용이 없고, llama.cpp는 자체 CUDA 커널을 씁니다.

## 참고

- [dspark-sm120-crash.md](dspark-sm120-crash.md) vLLM에서 DSpark가 크래시하는 원인
- [../llamacpp/README.md](../llamacpp/README.md) DSpark 사용법
- [backends.md](../docs/backends.md) 백엔드 실측 비교
- [Ollama Modelfile](https://docs.ollama.com/modelfile) 지시어 목록
