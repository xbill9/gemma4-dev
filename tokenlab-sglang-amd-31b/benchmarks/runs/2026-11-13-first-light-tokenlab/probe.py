"""First-light probe: one streamed request at a time against Token Lab, timestamped per event.

Records for each request: retries before headers, time to headers, time to first token
(reasoning or content), the time of every content-bearing event, usage, whether [DONE]
arrived. Writes one JSON line per request to probe.jsonl beside this file.

    python3 probe.py [--n 3] [--think] [--max-tokens 256]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import tokenlab  # noqa: E402

PROMPTS = [
    "Why is the sky blue? One paragraph.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Give me three facts about octopuses, one line each.",
]

def one(cfg, prompt, think, max_tokens):
    body = {"model": cfg["MODEL_NAME"], "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}
    retries = []
    t0 = time.time()
    r = tokenlab.open_api(cfg, "/chat/completions", json.dumps(body).encode(), on_retry=lambda *a: retries.append(a[1]))
    rec = {"prompt": prompt, "think": think, "max_tokens": max_tokens, "status": r.status,
           "retries": len(retries), "retry_codes": retries, "t_headers": round(time.time() - t0, 3),
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    if r.status != 200:
        rec["error"] = tokenlab.error_message(r)
        return rec
    times, usage, done, pings, reasoning, content, finish = [], None, False, 0, 0, 0, None
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith(":"):
            pings += 1
            continue
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            done = True
            break
        j = json.loads(d)
        usage = j.get("usage") or usage
        for ch in j.get("choices", []):
            dl = ch.get("delta") or {}
            rc = dl.get("reasoning_content") or dl.get("reasoning") or ""
            c = dl.get("content") or ""
            if rc or c:
                times.append(round(time.time() - t0, 4))
            reasoning += len(rc)
            content += len(c)
            finish = ch.get("finish_reason") or finish
    rec.update({"t_total": round(time.time() - t0, 3), "done": done, "pings": pings, "finish": finish,
                "reasoning_chars": reasoning, "content_chars": content, "usage": usage,
                "ttft": times[0] if times else None, "events": len(times),
                "t_last_token": times[-1] if times else None})
    n = (usage or {}).get("completion_tokens")
    if n and n > 1 and times and times[-1] > times[0]:
        rec["decode_tok_s"] = round((n - 1) / (times[-1] - times[0]), 2)
    return rec

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--think", action="store_true")
    p.add_argument("--max-tokens", type=int, default=256)
    a = p.parse_args()
    cfg = tokenlab.load_config()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe.jsonl")
    for i in range(a.n):
        rec = one(cfg, PROMPTS[i % len(PROMPTS)], a.think, a.max_tokens)
        print(json.dumps({k: v for k, v in rec.items() if k != "usage"}), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
