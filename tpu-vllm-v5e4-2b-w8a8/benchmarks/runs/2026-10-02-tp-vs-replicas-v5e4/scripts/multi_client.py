#!/usr/bin/env python3
"""Copy of jev-tpu-v5e1/tpu/w4a16_client.py that spreads requests over several servers.

JEV_LOAD_URLS (comma-separated base URLs, default http://localhost:8000): request i of a pass goes
to URLS[i % len(URLS)], so a concurrency of N splits evenly over the servers. Method unchanged:
one untimed warm-up pass, three timed passes, median; throughput counts every server's tokens
over the pass's wall clock.

Original docstring follows.

Client for the W4A16 probe. Stdlib only; runs on the VM host against :8000.

record <model> <out.json>   greedy chat completions on fixed prompts, with top-5 logprobs
compare <ref.json> <test.json>
                            per prompt: does the first token match, how many leading
                            tokens match, and the mean |logprob| gap over that prefix
load <model> <out.json>     CONCURRENCY parallel requests of exactly MAX_TOKENS each
                            (ignore_eos, so every model does the same work), REPEATS
                            times; output tokens per second from the server's usage
                            counts, median and range over the repeats
loadlong <model> <out.json> the same with prompts of about JEV_LOAD_INPUT_TOKENS tokens,
                            streamed: a unique prefix on every request of every pass (the
                            prefix cache never serves one), plus total (prompt + output)
                            tokens per second and median time to first token
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URLS = [u.rstrip("/") + "/v1/chat/completions" for u in os.environ.get("JEV_LOAD_URLS", "http://localhost:8000").split(",")]
URL = URLS[0]
PROMPTS = [
    "What is the capital of Australia? Answer in one sentence.",
    "Compute 17 * 23 and show the steps.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Translate into French: The library opens at nine in the morning.",
    "List three differences between TCP and UDP.",
    "Explain in two sentences why the sky is blue.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "A train leaves at 14:10 and arrives at 17:45. How long is the trip?",
]
CONCURRENCY = int(os.environ.get("JEV_LOAD_CONCURRENCY", "16"))  # parallel requests in load
MAX_TOKENS = 256
REPEATS = 3


def chat(model, prompt, max_tokens, logprobs, ignore_eos=False, url=URL):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if ignore_eos:
        body["ignore_eos"] = True
    if logprobs:
        body.update(logprobs=True, top_logprobs=5)
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def record(model, out):
    rows = []
    for p in PROMPTS:
        r = chat(model, p, 64, True)
        c = r["choices"][0]
        toks = c["logprobs"]["content"]
        rows.append(
            {
                "prompt": p,
                "text": c["message"]["content"],
                "tokens": [t["token"] for t in toks],
                "logprobs": [t["logprob"] for t in toks],
            }
        )
        print(f"--- {p}\n{c['message']['content']}\n")
    json.dump({"model": model, "rows": rows}, open(out, "w"), indent=1)


def compare(ref_path, test_path):
    ref, test = json.load(open(ref_path)), json.load(open(test_path))
    print(f"reference {ref['model']}  vs  test {test['model']}")
    first = 0
    prefixes = []
    for a, b in zip(ref["rows"], test["rows"]):
        n = 0
        while n < min(len(a["tokens"]), len(b["tokens"])) and a["tokens"][n] == b["tokens"][n]:
            n += 1
        gap = (sum(abs(x - y) for x, y in zip(a["logprobs"][:n], b["logprobs"][:n])) / n) if n else float("nan")
        first += n > 0
        prefixes.append(n)
        print(
            f"  leading tokens equal {n:3d} of {len(a['tokens']):3d}   mean |dlogprob| {gap:.4f}   {a['prompt'][:50]}"
        )
    print(
        f"first token equal on {first} of {len(prefixes)} prompts; "
        f"median leading-equal run {sorted(prefixes)[len(prefixes) // 2]} tokens"
    )


def load(model, out):
    prompts = [(i, PROMPTS[i % len(PROMPTS)] + f" (request {i})") for i in range(CONCURRENCY)]
    run = lambda ip: chat(model, ip[1], MAX_TOKENS, False, ignore_eos=True, url=URLS[ip[0] % len(URLS)])
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        list(ex.map(run, prompts))  # warm-up pass at the timed shape, not timed
        passes = []
        for _ in range(REPEATS):
            t0 = time.time()
            results = list(ex.map(run, prompts))
            wall = time.time() - t0
            completion = sum(r["usage"]["completion_tokens"] for r in results)
            passes.append(
                {
                    "wall_s": round(wall, 3),
                    "completion_tokens": completion,
                    "output_tok_per_s": round(completion / wall, 1),
                }
            )
    rates = sorted(p["output_tok_per_s"] for p in passes)
    summary = {
        "model": model,
        "concurrency": CONCURRENCY,
        "servers": len(URLS),
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "repeats": REPEATS,
        "passes": passes,
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in results),
        "output_tok_per_s": rates[len(rates) // 2],
        "output_tok_per_s_min": rates[0],
        "output_tok_per_s_max": rates[-1],
    }
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(summary))


FILLER = (
    "Record {i}: the warehouse in district {d} shipped {n} crates of parts on day {day}, "
    "and the audit team logged the delivery as complete with no damage reported. "
)
INPUT_TOKENS = int(os.environ.get("JEV_LOAD_INPUT_TOKENS", "2048"))  # target prompt length in loadlong


def long_prompt(nonce, sentences):
    body = "".join(FILLER.format(i=i, d=i % 17, n=(i * 37) % 500, day=i % 28 + 1) for i in range(sentences))
    return f"[{nonce}] Read the log below, then summarize it.\n\n{body}\nSummary:"


def stream(model, prompt, max_tokens, url=URL):
    """One streamed request: (time to first token, total seconds, prompt tokens, completion tokens)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if first is None and any((c.get("delta") or {}).get("content") for c in ev.get("choices", [])):
                first = time.time() - t0
            if ev.get("usage"):
                usage = ev["usage"]
    return first, time.time() - t0, usage["prompt_tokens"], usage["completion_tokens"]


def loadlong(model, out):
    """Throughput with long prompts: every request of every pass has a unique prefix, so the prefix
    cache never serves a prompt; prompt length is calibrated to JEV_LOAD_INPUT_TOKENS."""
    probe = stream(model, long_prompt("probe", 40), 1)[2]
    base = stream(model, long_prompt("probe", 0), 1)[2]
    sentences = max(1, round((INPUT_TOKENS - base) / ((probe - base) / 40)))
    tag = f"{time.time_ns()}"
    run = lambda inonce: stream(model, long_prompt(inonce[1], sentences), MAX_TOKENS, url=URLS[inonce[0] % len(URLS)])
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        list(ex.map(run, [(i, f"{tag}-w-{i}") for i in range(CONCURRENCY)]))  # warm-up, not timed
        passes = []
        for p in range(REPEATS):
            t0 = time.time()
            results = list(ex.map(run, [(i, f"{tag}-{p}-{i}") for i in range(CONCURRENCY)]))
            wall = time.time() - t0
            ttft = sorted(r[0] for r in results)
            passes.append(
                {
                    "wall_s": round(wall, 3),
                    "prompt_tokens": sum(r[2] for r in results),
                    "completion_tokens": sum(r[3] for r in results),
                    "output_tok_per_s": round(sum(r[3] for r in results) / wall, 1),
                    "total_tok_per_s": round(sum(r[2] + r[3] for r in results) / wall, 1),
                    "ttft_median_s": round(ttft[len(ttft) // 2], 3),
                    "ttft_max_s": round(ttft[-1], 3),
                }
            )
    key = lambda k: sorted(p[k] for p in passes)
    rates, ttfts = key("output_tok_per_s"), key("ttft_median_s")
    summary = {
        "model": model,
        "concurrency": CONCURRENCY,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "repeats": REPEATS,
        "servers": len(URLS),
        "target_input_tokens": INPUT_TOKENS,
        "prompt_tokens_per_request": results[0][2],
        "unique_prefix_per_request": True,
        "passes": passes,
        "output_tok_per_s": rates[len(rates) // 2],
        "output_tok_per_s_min": rates[0],
        "output_tok_per_s_max": rates[-1],
        "total_tok_per_s": key("total_tok_per_s")[len(passes) // 2],
        "ttft_median_s": ttfts[len(ttfts) // 2],
    }
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"record": record, "compare": compare, "load": load, "loadlong": loadlong}[cmd](*args)
