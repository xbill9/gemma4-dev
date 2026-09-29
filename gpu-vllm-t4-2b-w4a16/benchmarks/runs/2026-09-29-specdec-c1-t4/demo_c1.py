"""Single-user interactive latency on the T4: TTFT, TPOT and spec-decode acceptance.

    python3 demo_c1.py LABEL [--url http://127.0.0.1:8000] [--max-tokens 256] [--temperature T|default]

Sends each prompt below once, sequentially (c=1), as a streamed chat completion at
temperature 0 (or T; `default` sends none, so the server applies the checkpoint's
generation_config, as ~/bin/ask-t4 does), after two untimed warmups. Real chat prompts on purpose: random
tokens (`vllm bench serve --dataset-name random`) make a draft model's acceptance
look far worse than it is on the text a demo produces. Acceptance comes from the
server's /metrics counters, differenced across the timed requests only.
"""

import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = [
    "Why is the sky blue? Answer in two short paragraphs.",
    "Write a Python function that checks whether a string is a palindrome, with a docstring.",
    "Explain what a TPU is to a high-school student.",
    "Give me five name ideas for a coffee shop run by robots.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "What are the pros and cons of running a language model locally instead of in the cloud?",
    "Translate into French: 'The meeting has been moved to Thursday afternoon.'",
    "Write a haiku about a GPU that is running out of memory.",
    "How do I reverse a linked list? Explain, then show code in C.",
    "List the planets of the solar system with one interesting fact each.",
    "What is the difference between a process and a thread?",
    "Draft a polite email declining a meeting invitation.",
    "Explain quantization of neural network weights in simple terms.",
    "Write a SQL query that returns the top 3 customers by total order value.",
    "What should I pack for a three-day hiking trip?",
    "Explain the Monty Hall problem and why switching is better.",
    "Write a short bash script that backs up a directory with a timestamp.",
    "What is Kubernetes and why would someone use it?",
    "Give a one-paragraph history of the Internet.",
    "Tell me a short story about a cat who learns to code.",
]


def metrics(url):
    out = {}
    with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_") and not line.startswith("#"):
                name, val = line.rsplit(" ", 1)
                key = name.split("{")[0]
                out[key] = out.get(key, 0.0) + float(val)
    return out


def one(url, model, prompt, max_tokens, temperature=0.0):
    req = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        req["temperature"] = temperature
    body = json.dumps(req).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices") and d["choices"][0]["delta"].get("content") and first is None:
                first = time.perf_counter()
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
    t1 = time.perf_counter()
    return {"ttft_ms": (first - t0) * 1e3, "tpot_ms": (t1 - first) * 1e3 / max(n - 1, 1), "tokens": n, "e2e_s": t1 - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", default="0")
    a = ap.parse_args()
    temp = None if a.temperature == "default" else float(a.temperature)
    with urllib.request.urlopen(a.url + "/v1/models", timeout=10) as r:
        model = json.load(r)["data"][0]["id"]
    for p in PROMPTS[:2]:
        one(a.url, model, p, 32)
    m0 = metrics(a.url)
    rows = [dict(prompt=p, **one(a.url, model, p, a.max_tokens, temp)) for p in PROMPTS]
    m1 = metrics(a.url)
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    drafts = d.get("vllm:spec_decode_num_draft_tokens_total", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    summ = {
        "label": a.label, "model": model, "n": len(rows), "temperature": a.temperature,
        "median_ttft_ms": statistics.median(r["ttft_ms"] for r in rows),
        "median_tpot_ms": statistics.median(r["tpot_ms"] for r in rows),
        "decode_tok_s": 1e3 / statistics.median(r["tpot_ms"] for r in rows),
        "mean_tokens": statistics.mean(r["tokens"] for r in rows),
        "acceptance_rate": (acc / drafts) if drafts else None,
        "spec_metrics_delta": d,
    }
    json.dump({"summary": summ, "rows": rows}, open(f"{a.label}.json", "w"), indent=1)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k != "spec_metrics_delta"}))


if __name__ == "__main__":
    main()
