#!/usr/bin/env python3
"""Client for the W4A16 probe. Stdlib only; runs on the VM host against :8000.

  record <model> <out.json>   greedy chat completions on fixed prompts, with top-5 logprobs
  compare <ref.json> <test.json>
                              per prompt: does the first token match, how many leading
                              tokens match, and the mean |logprob| gap over that prefix
  load <model> <out.json>     CONCURRENCY parallel requests of exactly MAX_TOKENS each
                              (ignore_eos, so every model does the same work), REPEATS
                              times; output tokens per second from the server's usage
                              counts, median and range over the repeats
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"
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
CONCURRENCY = 16
MAX_TOKENS = 256
REPEATS = 3


def chat(model, prompt, max_tokens, logprobs, ignore_eos=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    if ignore_eos:
        body["ignore_eos"] = True
    if logprobs:
        body.update(logprobs=True, top_logprobs=5)
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def record(model, out):
    rows = []
    for p in PROMPTS:
        r = chat(model, p, 64, True)
        c = r["choices"][0]
        toks = c["logprobs"]["content"]
        rows.append({"prompt": p, "text": c["message"]["content"],
                     "tokens": [t["token"] for t in toks],
                     "logprobs": [t["logprob"] for t in toks]})
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
        print(f"  leading tokens equal {n:3d} of {len(a['tokens']):3d}   mean |dlogprob| {gap:.4f}   {a['prompt'][:50]}")
    print(f"first token equal on {first} of {len(prefixes)} prompts; "
          f"median leading-equal run {sorted(prefixes)[len(prefixes) // 2]} tokens")


def load(model, out):
    prompts = [PROMPTS[i % len(PROMPTS)] + f" (request {i})" for i in range(CONCURRENCY)]
    run = lambda p: chat(model, p, MAX_TOKENS, False, ignore_eos=True)
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        list(ex.map(run, prompts))  # warm-up pass at the timed shape, not timed
        passes = []
        for _ in range(REPEATS):
            t0 = time.time()
            results = list(ex.map(run, prompts))
            wall = time.time() - t0
            completion = sum(r["usage"]["completion_tokens"] for r in results)
            passes.append({"wall_s": round(wall, 3), "completion_tokens": completion,
                           "output_tok_per_s": round(completion / wall, 1)})
    rates = sorted(p["output_tok_per_s"] for p in passes)
    summary = {"model": model, "concurrency": CONCURRENCY, "max_tokens": MAX_TOKENS,
               "ignore_eos": True, "repeats": REPEATS, "passes": passes,
               "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in results),
               "output_tok_per_s": rates[len(rates) // 2],
               "output_tok_per_s_min": rates[0], "output_tok_per_s_max": rates[-1]}
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"record": record, "compare": compare, "load": load}[cmd](*args)
