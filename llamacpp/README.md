# llama.cpp 서빙

GGUF 양자화본을 `llama-server`로 서빙합니다. 실측 비교는 [backends.md](../docs/backends.md)를 보세요.

## 빠른 시작

```bash
bash download.sh UD-Q2_K_XL    # 96.8 GiB. 생략하면 4종 전부(526 GiB)
bash serve.sh                  # 약 30초 후 http://localhost:8001/v1
```

OpenAI 호환이므로 [../client_example.py](../client_example.py)의 `base_url`만 8001로 바꾸면 그대로 동작합니다.

## 옵션

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `QUANT` | `UD-Q2_K_XL` | 양자화 선택 |
| `PORT` | `8001` | vLLM(8000)과 겹치지 않게 |
| `CTX` | `8192` | 컨텍스트 길이 |
| `SLOTS` | `8` | 동시 요청 슬롯. 동시 요청 수 이상으로 둘 것 |
| `DSPARK` | `0` | `1`이면 추측 디코딩 활성화 |
| `GGUF_DIR` | `/opt/dlami/nvme/models/gguf` | GGUF 위치 |

```bash
QUANT=UD-IQ4_XS CTX=32768 bash serve.sh
DSPARK=1 bash serve.sh          # 단일 사용자일 때만 이득
```

## 양자화 선택 기준

| 양자화 | 파일 | GPU 메모리 | c=8 처리량 | GPU 1장 |
|---|---|---|---|---|
| **UD-Q2_K_XL** | 96.8 GB | 92.3 GiB | 218 tok/s | 가능 |
| UD-Q3_K_XL | 129 GB | 122.5 GiB | 233 tok/s | 불가 |
| UD-IQ4_XS | 138 GB | 130.3 GiB | 224 tok/s | 불가 |
| UD-Q8_K_XL | 162 GB | 152.2 GiB | 201 tok/s | 불가 |

처리량 차이가 9% 이내라 **메모리가 선택 기준입니다.** MoE라서 활성 파라미터(13B)만 계산하므로 가중치 크기가 속도를 좌우하지 않습니다.

## DSpark 추측 디코딩

드래프트 모델을 먼저 받습니다. **0731 저장소에만** 있습니다(10.9 GiB).

```bash
uvx --from huggingface_hub hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  --include "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf" \
  --local-dir /opt/dlami/nvme/models/gguf/dspark
DSPARK=1 bash serve.sh
```

| 구성 | c=1 처리량 | c=8 처리량 |
|---|---|---|
| 기본 | 60.1 tok/s | 226.5 tok/s |
| DSpark | **75.3 tok/s (+25%)** | 132.9 tok/s (-41%) |

실측 드래프트 수락률은 56.1%(32/57, 평균 2.68토큰)였습니다. 동시 요청이 많으면 드래프트 계산이 낭비되므로 끄세요.

vLLM에서는 이 기능이 SM120에서 크래시합니다([../troubleshooting/dspark-sm120-crash.md](../troubleshooting/dspark-sm120-crash.md)). llama.cpp는 자체 CUDA 커널을 써서 동작합니다.

## 알아둘 것

**GPU 병렬화가 제한적입니다.** `--split-mode row`와 `tensor`는 이 GPU에서 로드 실패합니다(`device CUDA0 does not support split buffers`). 기본값 `layer`는 파이프라인 방식이라 한 GPU가 계산할 때 다른 쪽이 대기합니다. vLLM의 TP=2에 해당하는 병렬 계산을 쓸 수 없고, 이것이 처리량 격차(407 대 218)의 주요 원인으로 보입니다.

**튜닝 여지가 거의 없습니다.** `-fa`, `-b/-ub`, `-np`를 조정해도 2% 이내 차이였습니다. `-fa`는 기본이 `auto`로 이미 활성이고 `--cont-batching`도 기본 활성입니다.

**GPU가 부족하면 `-ncmoe`를 쓸 수 있습니다.** MoE 전문가를 CPU RAM으로 넘깁니다. 다만 메모리 절감분만큼 느려집니다(`-ncmoe 10`: VRAM -40%, 속도 -38%).

## Docker를 쓰는 이유

- Linux 사전 빌드 바이너리에 CUDA 버전이 없습니다(`ubuntu-x64`는 CPU 전용)
- 소스 빌드는 `nvcc`가 필요하고 이 서버에는 없습니다
- `llama-cpp-python`의 CUDA 휠은 `cu125`까지만 제공됩니다(이 서버는 CUDA 13)
