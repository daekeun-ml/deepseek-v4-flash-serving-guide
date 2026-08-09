# 레이턴시 측정

vLLM, llama-server, Ollama에 같은 조건으로 요청해 TTFT, TPOT, 처리량을 비교합니다.
측정 결과 해석은 [../backends.md](../backends.md)를 보세요.

## 사용법

```bash
python3 latency.py <endpoint_url> <concurrency> <n_requests> <model_name> [label]
```

label을 주면 `results/lat_<label>.json`으로 저장합니다.

```bash
# vLLM
python3 latency.py http://localhost:8000/v1/chat/completions 8 32 deepseek-v4-flash vllm-fp8-c8

# llama-server
python3 latency.py http://localhost:8001/v1/chat/completions 8 32 x lcpp-q2-c8

# Ollama (워밍업 후 측정할 것, 첫 요청에 로딩 시간이 포함됨)
python3 latency.py http://localhost:11434/v1/chat/completions 8 32 ds-q2 ollama-q2-c8
```

부하 형태는 [../benchmark.md](../benchmark.md)의 S1(c=1), S2(c=8)과 같습니다. 입력 약 200토큰, 출력 200토큰입니다.

## 주의

**백엔드마다 스트리밍 delta 필드명이 다릅니다.**

| 백엔드 | 필드명 |
|---|---|
| vLLM, llama.cpp | `reasoning_content` |
| Ollama | `reasoning` |

이 모델은 기본이 thinking 모드라 토큰이 대부분 이 필드로 나옵니다. 한쪽만 세면 토큰 수가 0으로 집계됩니다. 스크립트는 셋 다 처리합니다.

**Ollama는 첫 요청에 모델 로딩이 포함됩니다.** 워밍업 없이 재면 TTFT p95가 48초로 튑니다. 96.8 GB 모델은 로딩에 약 2분 걸리므로, 짧은 요청 하나로는 워밍업이 끝나지 않습니다. `/api/ps`로 실제 로드 여부를 확인하세요.

```bash
curl -s http://localhost:11434/api/ps | python3 -m json.tool
# size_vram이 모델 크기와 비슷해야 로드 완료
```

**llama-server는 `-np`를 동시 요청 수 이상으로 두세요.** `-np 4`에 c=8을 보내면 슬롯 부족으로 TTFT가 12배 나빠집니다(417ms -> 4,931ms).

## 결과 파일

`results/lat_<label>.json`에 p50/p95를 포함한 요약이 저장됩니다.

```json
{
  "label": "vllm-fp8-c8",
  "concurrency": 8,
  "ttft_p50_ms": 186.2,
  "ttft_p95_ms": 2011.4,
  "tpot_p50_ms": 16.3,
  "output_tok_per_s": 407.3,
  "req_per_s": 2.21
}
```
