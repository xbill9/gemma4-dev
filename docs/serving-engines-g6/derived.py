"""Ratios and rounded figures the article quotes, computed from the 2026-11-06 suite run."""

import json
import os

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gpu-vllm-g6-2b-q4w4a16emb4", "benchmarks", "runs", "2026-11-06-suite-g6")


def wall(arm):
    return json.loads(open(os.path.join(RUN, f"acc-{arm}", "gsm8k.log")).read().strip().splitlines()[-1])["wall_s"]


def tps(arm, shape, c):
    d = json.load(open(os.path.join(RUN, f"sweep-{arm}", shape, "concurrency.json")))
    return next(x["output_tokens_per_second"] for x in d["concurrency"] if x["concurrency"] == c)


for a, b in (("vllm", "vllm-d4"), ("sglang", "sglang-d4"), ("llama", "llama-d2")):
    print(f"GSM8K time {a} {wall(a):.1f} s / {b} {wall(b):.1f} s = {wall(a) / wall(b):.2f}x")
s, v = tps("sglang", "in128-out512", 64), tps("vllm-d4", "in128-out512", 64)
print(f"128/512 at 64: SGLang {s:.1f} / vLLM+drafter {v:.1f} = {s / v:.2f}x")
for arm in ("vllm", "llama-d2"):
    print(f"128/512 at 64: {arm} {tps(arm, 'in128-out512', 64):,.0f} tokens per second")
