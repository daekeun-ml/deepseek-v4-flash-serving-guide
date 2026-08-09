"""서빙 백엔드/양자화별 레이턴시 측정.

vLLM, llama-server, Ollama에 동일 조건으로 요청해 TTFT / TPOT / 처리량을 비교한다.
../benchmark.md의 S1(c=1), S2(c=8)과 같은 부하 형태를 쓴다.

사용:
    python3 latency.py <endpoint_url> <concurrency> <n_requests> <model_name> [label]
예:
    python3 latency.py http://localhost:8000/v1/chat/completions 8 32 deepseek-v4-flash vllm-fp8-c8
    python3 latency.py http://localhost:8001/v1/chat/completions 8 32 x lcpp-q2-c8
    python3 latency.py http://localhost:11434/v1/chat/completions 8 32 ds-q2 ollama-q2-c8

label을 주면 results/lat_<label>.json 으로 저장한다.

주의: 백엔드마다 스트리밍 delta 필드명이 다르다.
  vLLM, llama.cpp -> reasoning_content    Ollama -> reasoning
셋 다 세지 않으면 토큰 수가 0으로 집계된다(실제로 겪은 함정).
"""

import json
import os
import statistics
import sys
import threading
import time
import urllib.request

URL = sys.argv[1]
CONC = int(sys.argv[2])
N = int(sys.argv[3])
MODEL = sys.argv[4] if len(sys.argv) > 4 else "x"
LABEL = sys.argv[5] if len(sys.argv) > 5 else None

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TOKENS = 200      # benchmark.md S1/S2와 동일한 출력 길이
PROMPT = "다음 주제에 대해 설명해줘: " + ("인공지능의 역사와 발전 과정에 대해 자세히 " * 12)

res, lock = [], threading.Lock()


def one():
    body = json.dumps({
        "model": MODEL, "stream": True, "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0, ttft, n = time.time(), None, 0
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                if not raw.startswith(b"data: "):
                    continue
                if raw.strip() == b"data: [DONE]":
                    break
                if ttft is None:
                    ttft = time.time() - t0
                try:
                    d = json.loads(raw[6:])["choices"][0].get("delta", {})
                    if (d.get("content") or d.get("reasoning_content")
                            or d.get("reasoning")):
                        n += 1
                except Exception:
                    pass
    except Exception as e:
        with lock:
            res.append({"err": str(e)[:80]})
        return
    e2e = time.time() - t0
    with lock:
        res.append({"ttft": ttft * 1000, "e2e": e2e * 1000, "toks": n,
                    "tpot": (e2e - ttft) / max(n - 1, 1) * 1000})


def pct(vals, q):
    s = sorted(vals)
    return s[min(int(len(s) * q), len(s) - 1)]


if __name__ == "__main__":
    sem = threading.Semaphore(CONC)

    def worker():
        with sem:
            one()

    start = time.time()
    ts = [threading.Thread(target=worker) for _ in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - start

    ok = [r for r in res if "ttft" in r]
    print(f"완료 {len(ok)}/{N}  실패 {len(res) - len(ok)}  wall {wall:.1f}s")
    if not ok:
        print("  errors:", {r.get("err") for r in res})
        sys.exit(1)

    tt = [r["ttft"] for r in ok]
    tp = [r["tpot"] for r in ok]
    e2 = [r["e2e"] for r in ok]
    out_tps = sum(r["toks"] for r in ok) / wall

    print(f"  TTFT  p50={statistics.median(tt):8.1f}  p95={pct(tt, .95):9.1f} ms")
    print(f"  TPOT  p50={statistics.median(tp):8.2f}  p95={pct(tp, .95):9.2f} ms")
    print(f"  E2EL  p50={statistics.median(e2):8.1f}  p95={pct(e2, .95):9.1f} ms")
    print(f"  처리량  {out_tps:7.1f} out tok/s   {len(ok) / wall:.2f} req/s")

    if LABEL:
        summary = {
            "label": LABEL, "url": URL, "model": MODEL,
            "concurrency": CONC, "n_requests": N,
            "completed": len(ok), "failed": len(res) - len(ok),
            "wall_seconds": round(wall, 1),
            "ttft_p50_ms": round(statistics.median(tt), 1),
            "ttft_p95_ms": round(pct(tt, .95), 1),
            "tpot_p50_ms": round(statistics.median(tp), 2),
            "tpot_p95_ms": round(pct(tp, .95), 2),
            "e2el_p50_ms": round(statistics.median(e2), 1),
            "e2el_p95_ms": round(pct(e2, .95), 1),
            "output_tok_per_s": round(out_tps, 1),
            "req_per_s": round(len(ok) / wall, 2),
        }
        outdir = os.path.join(HERE, "results")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, f"lat_{LABEL}.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  -> results/lat_{LABEL}.json")
