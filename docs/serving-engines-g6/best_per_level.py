"""G6 (one NVIDIA L4): each batching engine's best setting (no drafter or a measured draft length) per
concurrency level and shape, aggregate output tok/s, median of 5 repeats, from the 2026-11-06 suite run.
Also peak card memory during each sweep (gpu.csv) and the median time per output token at 64 requests.
Prints Markdown."""

import csv
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUN = os.path.join(ROOT, "gpu-vllm-g6-2b-q4w4a16emb4", "benchmarks", "runs", "2026-11-06-suite-g6")
ARMS = {
    "vLLM": {"none": "vllm", "1 drafted": "vllm-d1", "4 drafted": "vllm-d4"},
    "SGLang": {"none": "sglang", "4 drafted": "sglang-d4"},
    "llama.cpp": {"none": "llama", "2 drafted": "llama-d2"},
    "mistral.rs": {"none": "mistralrs-card"},
}
LEVELS = (1, 2, 4, 8, 16, 32, 64)
SHAPES = ("in128-out512", "in512-out128")


def rows(arm, shape):
    return {c["concurrency"]: c for c in json.load(open(os.path.join(RUN, f"sweep-{arm}", shape, "concurrency.json")))["concurrency"]
            if c.get("status", "ok") == "ok"}


def peak_mib(arm):
    best = 0
    with open(os.path.join(RUN, f"sweep-{arm}", "gpu.csv")) as f:
        for r in csv.reader(f):
            for x in r:
                x = x.strip()
                if x.endswith("MiB"):
                    try:
                        best = max(best, int(x.split()[0]))
                    except ValueError:
                        pass
    return best


for shape in SHAPES:
    i, o = shape[2:].split("-out")
    print(f"\n{i} in / {o} out, best setting per level:\n")
    print("| Requests | " + " | ".join(ARMS) + " |")
    print("| :--- |" + " :--- |" * len(ARMS))
    for c in LEVELS:
        cells = {}
        for eng, settings in ARMS.items():
            cand = [(rows(a, shape)[c]["output_tokens_per_second"], s) for s, a in settings.items() if c in rows(a, shape)]
            cells[eng] = max(cand)
        top = max(v for v, _ in cells.values())
        print(f"| {c} | " + " | ".join(f"{'🥇 ' if v == top else ''}{v:,.0f} ({s})" for v, s in cells.values()) + " |")

print("\nPeak card memory during each sweep, and median time per output token at 64 requests:\n")
print("| Arm | Peak MiB | 128/512 ms per token at 64 | 512/128 ms per token at 64 |")
print("| :--- | :--- | :--- | :--- |")
for settings in ARMS.values():
    for s, a in settings.items():
        t = [rows(a, sh)[64]["tpot_ms_median"] for sh in SHAPES]
        print(f"| {a} | {peak_mib(a):,} | {t[0]} | {t[1]} |")
