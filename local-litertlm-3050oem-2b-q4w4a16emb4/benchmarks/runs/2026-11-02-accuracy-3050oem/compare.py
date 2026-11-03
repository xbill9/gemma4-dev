"""GSM8K and BFCL simple for LiteRT-LM on the RTX 3050 OEM, stock server (constrained decoding on), drafter off:
Google's int248 .litertlm against this rig's exact QAT Q4_0 build, and each against llama.cpp's exact Q4_0
E2B on the same card (no drafter, -c 131072; local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/
2026-10-14-accuracy-fullctx-3050oem), paired by question with a 95% paired bootstrap range (2,000 resamples,
seed 0). Also prints identical replies and wall time."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
LC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "local-llamacpp-3050oem-2b-q4_0", "benchmarks",
                                   "runs", "2026-10-14-accuracy-fullctx-3050oem"))


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f))} if os.path.exists(f) else {}


def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]


def wall(log):
    try:
        return json.loads(open(log).read().strip().splitlines()[-1])["wall_s"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


for t in ("gsm8k", "bfcl_simple"):
    arms = {
        "int248": (load(os.path.join(HERE, f"{t}-int248.jsonl")), wall(os.path.join(HERE, f"{t}-int248.log"))),
        "q4exact": (load(os.path.join(HERE, f"{t}-q4exact.jsonl")), wall(os.path.join(HERE, f"{t}-q4exact.log"))),
        "llama.cpp": (load(os.path.join(LC, f"{t}-e2b-none.jsonl")), wall(os.path.join(LC, f"{t}-e2b-none.log"))),
    }
    for n, (rows, w) in arms.items():
        right = sum(bool(r.get("right")) for r in rows.values())
        trunc = sum(r.get("finish") == "length" for r in rows.values())
        print(f"{t:12s} {n:10s} n={len(rows)} right {right} acc {right / len(rows):.4f} truncated {trunc} wall {w} s")
    for a, b in (("q4exact", "llama.cpp"), ("int248", "llama.cpp"), ("q4exact", "int248")):
        x, y = arms[a][0], arms[b][0]
        ids = sorted(set(x) & set(y))
        d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
        lo, hi = boot(d)
        same = sum((x[i].get("text") or "", json.dumps(x[i].get("calls"), sort_keys=True))
                   == (y[i].get("text") or "", json.dumps(y[i].get("calls"), sort_keys=True)) for i in ids)
        print(f"{t:12s} {a} - {b}: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] "
              f"(+{d.count(1)} / -{d.count(-1)}) identical replies {same} of {len(ids)}")
