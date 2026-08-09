# Ollama가 분할 GGUF를 읽지 못하는 문제

## 요약

unsloth가 배포하는 GGUF는 파일이 3~5개로 쪼개져 있습니다. llama-server는 첫 조각만 지정하면 나머지를 알아서 찾지만, **Ollama는 이를 거부합니다.**

```
Error: split GGUF "DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf" has 1 shards, expected 3
```

`llama-gguf-split --merge`로 단일 파일로 합쳐야 합니다. Q2(96.8 GiB) 기준 약 1분 걸립니다.

## 배경 지식

### 분할 GGUF

**GGUF**는 llama.cpp가 쓰는 모델 파일 형식입니다. DeepSeek-V4-Flash는 양자화해도 96~162 GB라 단일 파일로 배포하기 부담스럽습니다. HuggingFace의 파일당 권장 상한을 넘고, 다운로드가 중간에 끊기면 처음부터 다시 받아야 하니까요.

그래서 unsloth는 파일을 쪼개 올립니다.

```
UD-Q2_K_XL/
  DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf     5 MB   메타데이터만
  DeepSeek-V4-Flash-UD-Q2_K_XL-00002-of-00003.gguf    49 GB   텐서 688개
  DeepSeek-V4-Flash-UD-Q2_K_XL-00003-of-00003.gguf    47 GB   텐서 640개
```

첫 조각이 5 MB뿐인 게 정상입니다. 여기에는 모델 설정값(kv 72개)만 있고 실제 가중치는 2번과 3번에 들어 있습니다.

### 두 도구의 처리 방식 차이

llama-server는 첫 조각의 메타데이터에서 "총 3개"라는 정보를 읽고 같은 디렉터리에서 나머지를 찾습니다. Ollama는 이 탐색을 하지 않고, **넘겨받은 파일 개수와 메타데이터의 선언값을 비교해서 다르면 거부합니다.**

Ollama가 파일을 자기 저장소(`~/.ollama/models/blobs/`)로 복사해 관리하기 때문입니다. 복사 시점에 조각이 전부 있어야 하는데, Modelfile의 `FROM`은 경로 하나만 받습니다.

## 원인

Modelfile은 이렇게 씁니다.

```
FROM /gguf/UD-Q2_K_XL/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf
```

Ollama가 이 파일을 열어 메타데이터를 보면 "3개로 분할됨"이라 적혀 있는데, 실제로 전달된 것은 1개입니다. 그래서 개수 불일치로 등록을 중단합니다.

`FROM`에 와일드카드나 디렉터리를 줄 수도 없습니다. 여러 개를 나열하는 문법도 없습니다.

## 해결

`llama-gguf-split --merge`로 병합합니다. 이 도구는 **`server-cuda` 이미지에 없고 `full` 이미지에만** 있습니다.

```bash
docker run --rm --entrypoint /app/llama-gguf-split \
  -v /opt/dlami/nvme/models/gguf:/gguf \
  ghcr.io/ggml-org/llama.cpp:full \
  --merge /gguf/UD-Q2_K_XL/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf \
          /gguf/merged/ds-q2.gguf
```

```
gguf_merge: /gguf/merged/ds-q2.gguf merged from 3 split with 1328 tensors.
```

이 저장소에는 [../ollama/merge.sh](../ollama/merge.sh)로 넣어뒀습니다.

```bash
bash ollama/merge.sh                    # Q2
QUANT=UD-IQ4_XS bash ollama/merge.sh    # 다른 양자화
```

병합 후에는 정상 등록됩니다.

```
FROM /gguf/merged/ds-q2.gguf
PARAMETER num_ctx 8192
```

## 디스크 비용

병합은 원본을 남겨두므로 같은 모델이 여러 번 디스크를 차지합니다.

| 단계 | 용량 (Q2 기준, 실측) |
|---|---|
| 원본 분할 GGUF | 96.8 GB |
| 병합 파일 | 96.8 GB |
| Ollama 저장소 (`ollama_home`) | 107.7 GB |
| **합계** | **약 301 GB** |

llama-server만 쓴다면 원본 96.8 GB로 끝납니다. Ollama를 쓰려면 3배를 감당해야 하고, 양자화 4종을 모두 시험하려면 더 커집니다.

병합 파일은 Ollama 등록이 끝나면 지워도 됩니다. Ollama가 이미 복사해뒀기 때문입니다. 단 다른 이름으로 재등록할 때 다시 병합해야 합니다.

## 참고

- [../ollama/README.md](../ollama/README.md) Ollama 서빙 전체 흐름
- [../backends.md](../backends.md) 백엔드 실측 비교
- [llama.cpp gguf-split](https://github.com/ggml-org/llama.cpp/tree/master/tools/gguf-split) 병합 도구
