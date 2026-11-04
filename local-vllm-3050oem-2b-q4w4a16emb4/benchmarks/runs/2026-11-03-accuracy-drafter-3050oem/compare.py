"""vLLM 0.33.0 with the W4A16 drafter at 4 drafted tokens against vLLM without it (../2026-11-03-accuracy-3050oem)
and llama.cpp exact Q4_0 with and without its exact Q4_0 drafter at 2 tokens (local-llamacpp-3050oem-2b-q4_0/
.../2026-10-14-accuracy-fullctx-3050oem), GSM8K and BFCL simple, one request at a time. Paired by question,
95% paired bootstrap range (2,000 resamples, seed 0); wall time and its ratio."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
NO = os.path.join(HERE, "..", "2026-11-03-accuracy-3050oem")
LC = os.path.join(ROOT, "local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem")


def load(d, s):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(d, s + ".jsonl")))}


def wall(d, s):
    return json.loads(open(os.path.join(d, s + ".log")).read().strip().splitlines()[-1])["wall_s"]


def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]


def pair(x, y):
    ids = sorted(set(x) & set(y))
    d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
    lo, hi = boot(d)
    same = sum((x[i].get("text") or "", json.dumps(x[i].get("calls"), sort_keys=True))
               == (y[i].get("text") or "", json.dumps(y[i].get("calls"), sort_keys=True)) for i in ids)
    return f"{sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] (+{d.count(1)} / -{d.count(-1)}) identical {same} of {len(ids)}"


for t in ("gsm8k", "bfcl_simple"):
    arms = {"vllm+drafter": (HERE, f"{t}-vllm-w4a16n4"), "vllm": (NO, f"{t}-vllm"),
            "llama+drafter": (LC, f"{t}-e2b-q4x"), "llama": (LC, f"{t}-e2b-none")}
    rows = {n: load(*p) for n, p in arms.items()}
    walls = {n: wall(*p) for n, p in arms.items()}
    for n, r in rows.items():
        right = sum(bool(x.get("right")) for x in r.values())
        print(f"{t:12s} {n:14s} acc {right / len(r):.4f} ({right}/{len(r)}) wall {walls[n]:7.1f} s")
    print(f"{t:12s} vllm drafter speed-up {walls['vllm'] / walls['vllm+drafter']:.2f}x, "
          f"llama.cpp drafter speed-up {walls['llama'] / walls['llama+drafter']:.2f}x, "
          f"vllm+drafter vs llama+drafter {walls['llama+drafter'] / walls['vllm+drafter']:.2f}x sooner")
    print(f"{t:12s} vllm+drafter - vllm: {pair(rows['vllm+drafter'], rows['vllm'])}")
    print(f"{t:12s} vllm+drafter - llama+drafter: {pair(rows['vllm+drafter'], rows['llama+drafter'])}")
