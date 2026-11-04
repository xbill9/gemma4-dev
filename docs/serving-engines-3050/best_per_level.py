"""Each batching engine's best setting (no drafter or any measured draft length) per concurrency level and
shape, aggregate output tok/s, median of 5 repeats, from the run directories. Prints Markdown."""

import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def run(rig, d, *sub):
    return os.path.join(ROOT, rig, "benchmarks", "runs", d, *sub)


V, S, L, M = ("local-vllm-3050oem-2b-q4w4a16emb4", "local-sglang-3050oem-2b-q4w4a16emb4",
              "local-llamacpp-3050oem-2b-q4_0", "local-mistralrs-3050oem-2b-q4_0")
ARMS = {
    "vLLM": {"none": run(V, "2026-11-03-concurrency-3050oem"),
             **{f"{n} drafted": run(V, "2026-11-03-concurrency-drafters-3050oem", f"w4a16-n{n}") for n in (1, 2, 3, 4)}},
    "SGLang": {"none": run(S, "2026-11-04-concurrency-3050oem"),
               **{f"{n} drafted": run(S, "2026-11-04-concurrency-drafters-3050oem", f"w4a16-n{n}") for n in (1, 2, 3, 4)}},
    "llama.cpp": {"none": run(L, "2026-11-03-concurrency-3050oem"),
                  **{f"{n} drafted": run(L, "2026-11-03-concurrency-drafters-3050oem", f"q4x-n{n}") for n in (1, 2)}},
    "mistral.rs": {"none": run(M, "2026-11-04-concurrency-3050oem")},
}
LEVELS = (1, 2, 4, 8, 16, 32, 64)


def cells(d, shape):
    return {c["concurrency"]: c["output_tokens_per_second"]
            for c in json.load(open(os.path.join(d, shape, "concurrency.json")))["concurrency"] if c.get("status", "ok") == "ok"}


for shape, label in (("in128-out512", "128 in / 512 out"), ("in512-out128", "512 in / 128 out")):
    data = {e: {a: cells(d, shape) for a, d in arms.items()} for e, arms in ARMS.items()}
    print(f"\n{label}, best setting per level (tok/s, setting)\n")
    print("| requests | " + " | ".join(data) + " |")
    print("|---|" + "---|" * len(data))
    for c in LEVELS:
        row = []
        for e, arms in data.items():
            a = max(arms, key=lambda k: arms[k][c])
            row.append(f"{arms[a][c]:,.0f} ({a})")
        print(f"| {c} | " + " | ".join(row) + " |")
