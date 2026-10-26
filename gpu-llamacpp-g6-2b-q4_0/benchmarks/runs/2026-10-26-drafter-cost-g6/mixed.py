"""Concurrency with mixed chat prompts: c requests at once, each one of the six prompts of
2026-10-14-drafter-gaps-3050oem/speed.py, 256 tokens out, prompt cache off, thinking at the
server default (off). Records per-request TTFT, tokens and draft counts from llama-server's
final stream chunk; a level's aggregate is total output tokens over the batch's wall time.

    python3 mixed.py --model NAME --out FILE.jsonl --levels 1,2,4 --repeats 5 --mode greedy|sampled
"""
import argparse, concurrent.futures, itertools, json, time, urllib.request

PROMPTS = [
    "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "I just moved to a new city and feel a bit lonely. Any advice?",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator works to a ten-year-old.",
    "List the planets of the solar system with one fact about each.",
    "Summarize the causes of the French Revolution in one paragraph.",
]
SEED = itertools.count(1)
THINK = False   # --thinking: chat_template_kwargs enable_thinking, as sweep.py sends

def one(base, model, prompt, mode):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
            "cache_prompt": False, "stream": True, "stream_options": {"include_usage": True}}
    if THINK:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    body.update({"temperature": 0} if mode == "greedy" else {"temperature": 0.8, "seed": next(SEED)})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; timings = {}
    with urllib.request.urlopen(req, timeout=1800) as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ch = json.loads(line[5:])
            if ttft is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                                    for c in ch.get("choices", [])):
                ttft = time.perf_counter() - t0
            if ch.get("timings"):
                timings = ch["timings"]
    return {"ttft_ms": round((ttft or 0) * 1000, 1), "wall_s": round(time.perf_counter() - t0, 3),
            "n": timings.get("predicted_n", 0), "draft_n": timings.get("draft_n", 0),
            "draft_accepted": timings.get("draft_n_accepted", 0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="1,2,4,8,16,32,64")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--mode", choices=("greedy", "sampled"), required=True)
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    global THINK
    THINK = a.thinking
    one(a.base, a.model, PROMPTS[0], a.mode)                      # warm-up, not recorded
    with open(a.out, "a") as out:
        for c in map(int, a.levels.split(",")):
            for r in range(a.repeats):
                prompts = [PROMPTS[(i + r) % len(PROMPTS)] for i in range(c)]
                t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    runs = list(pool.map(lambda p: one(a.base, a.model, p, a.mode), prompts))
                wall = time.perf_counter() - t0
                rec = {"mode": a.mode, "thinking": THINK, "c": c, "rep": r, "wall_s": round(wall, 3), "runs": runs}
                out.write(json.dumps(rec) + "\n"); out.flush()
                tok = sum(x["n"] for x in runs)
                print(a.mode, c, r, round(tok / wall, 1), flush=True)

main()
