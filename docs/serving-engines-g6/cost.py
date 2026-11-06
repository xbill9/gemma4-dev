"""Arithmetic, not a measurement: on-demand dollars per million output tokens at measured throughput, from the
g6.4xlarge price in evidence/g6-pricing.txt and the 2026-11-06 sweep (median of 5 repeats). Also the price of
the suite's own run time from its driver log."""

import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "..", "gpu-vllm-g6-2b-q4w4a16emb4", "benchmarks", "runs", "2026-11-06-suite-g6")
price = {l.split()[0]: float(l.split()[-1]) for l in open(os.path.join(HERE, "evidence", "g6-pricing.txt")) if l.startswith("g6.")}
P = price["g6.4xlarge"]


def tps(arm, shape, c):
    d = json.load(open(os.path.join(RUN, f"sweep-{arm}", shape, "concurrency.json")))
    return next(x["output_tokens_per_second"] for x in d["concurrency"] if x["concurrency"] == c)


print(f"g6.4xlarge on-demand ${P}/hr, us-east-1; g6.2xlarge ${price['g6.2xlarge']}/hr (same L4, not measured)")
print(f"{'setting':34s} {'shape':13s} {'tok/s':>8s} {'$ per M output tokens':>22s}")
for label, arm, c in (("vLLM + drafter, 1 request", "vllm-d4", 1), ("vLLM + drafter (4), 64 requests", "vllm-d4", 64),
                      ("vLLM + drafter (1), 64 requests", "vllm-d1", 64), ("SGLang, 64 requests", "sglang", 64),
                      ("mistral.rs (32 seqs), 64 requests", "mistralrs-card", 64), ("llama.cpp + drafter, 64 requests", "llama-d2", 64),
                      ("llama.cpp, 64 requests", "llama", 64)):
    for shape in ("in128-out512", "in512-out128"):
        t = tps(arm, shape, c)
        c_m = P / (t * 3600) * 1e6
        print(f"{label:34s} {shape:13s} {t:8.1f} {c_m:22.4f}  (${c_m:.2f})")
lines = [l for l in open(os.path.join(RUN, "driver.log")) if l[:4] == "2026"]
t0 = datetime.datetime.strptime(lines[0][:19], "%Y-%m-%d %H:%M:%S")
t1 = datetime.datetime.strptime(lines[-1][:19], "%Y-%m-%d %H:%M:%S")
h = (t1 - t0).total_seconds() / 3600
print(f"suite run {t0:%H:%M}-{t1:%H:%M} UTC = {h:.2f} h, ${h * P:.2f} on-demand")
