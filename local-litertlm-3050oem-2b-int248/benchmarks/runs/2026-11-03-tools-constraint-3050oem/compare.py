"""BFCL simple with litert-lm serve's constrained decoding on (stock server) and off (serve_noconstraint.py),
for Google's int248 file and the exact QAT build, paired by question against each other and against
llama.cpp on the same card. Differences with a 95% paired bootstrap range (2,000 resamples, seed 0), the
method of local-llamacpp-3050oem-26b-q4_0/benchmarks/runs/2026-10-18-accuracy-26b-3050oem/compare.py.
Also: how many failures are a missing optional argument, and whether argument keys come out in the
schema's declared order or alphabetically."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
ACC = os.path.join(ROOT, "local-litertlm-3050oem-2b-q4w4a16emb4", "benchmarks", "runs", "2026-11-02-accuracy-3050oem")
LC = os.path.join(ROOT, "local-llamacpp-3050oem-2b-q4_0", "benchmarks", "runs", "2026-10-14-accuracy-fullctx-3050oem")
ARMS = {
    "litert int248 on": os.path.join(ACC, "bfcl_simple-int248.jsonl"),
    "litert int248 off": os.path.join(HERE, "bfcl_simple-int248-off.jsonl"),
    "litert q4exact on": os.path.join(ACC, "bfcl_simple-q4exact.jsonl"),
    "litert q4exact off": os.path.join(HERE, "bfcl_simple-q4exact-off.jsonl"),
    "llama.cpp q4_0": os.path.join(LC, "bfcl_simple-e2b-none.jsonl"),
}
DATA = {r["id"]: r for r in map(json.loads, open(os.path.join(ROOT, "jev-tpu-v5e1", "data", "bfcl_simple.jsonl")))}


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f))} if os.path.exists(f) else {}


def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]


def order(rows):
    """Calls with 2+ keys: (declared order, alphabetical order, neither); a call in both counts as neither."""
    dec = alpha = other = 0
    for i, r in rows.items():
        props = list(DATA[i]["functions"][0].get("parameters", {}).get("properties", {}))
        for c in r.get("calls") or []:
            keys = list(c["arguments"])
            if len(keys) < 2:
                continue
            is_dec = keys == [k for k in props if k in keys]
            is_alpha = keys == sorted(keys)
            if is_dec and not is_alpha:
                dec += 1
            elif is_alpha and not is_dec:
                alpha += 1
            else:
                other += 1
    return dec, alpha, other


arms = {n: load(f) for n, f in ARMS.items()}
for n, rows in arms.items():
    if not rows:
        print(f"{n:20s} (not run)")
        continue
    right = sum(r["right"] for r in rows.values())
    miss_opt = sum(
        1 for i, r in rows.items() if not r["right"] and r.get("reason", "").startswith("missing ")
        and r.get("reason", "").split()[1] not in DATA[i]["functions"][0].get("parameters", {}).get("required", [])
    )
    dec, alpha, other = order(rows)
    print(f"{n:20s} n={len(rows)} right {right} acc {right / len(rows):.4f}  missing-optional {miss_opt}  "
          f"key order: declared {dec} alphabetical {alpha} ambiguous {other}")

pairs = [("litert int248 off", "litert int248 on"), ("litert q4exact off", "litert q4exact on"),
         ("litert int248 off", "llama.cpp q4_0"), ("litert q4exact off", "llama.cpp q4_0"),
         ("litert q4exact on", "llama.cpp q4_0"), ("litert q4exact off", "litert int248 off")]
for a, b in pairs:
    x, y = arms[a], arms[b]
    if not (x and y):
        continue
    ids = sorted(set(x) & set(y))
    d = [int(x[i]["right"]) - int(y[i]["right"]) for i in ids]
    lo, hi = boot(d)
    print(f"{a} - {b}: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}]  (+{d.count(1)} / -{d.count(-1)} of {len(ids)})")
