"""GSM8K and BFCL simple for SGLang 0.5.23 with the W4A16 drafter at 4 drafted tokens (one request at a time,
thinking off) against SGLang without a drafter (../2026-11-04-accuracy-3050oem) and vLLM 0.33.0 with the same
drafter at 4 tokens (local-vllm-3050oem-2b-q4w4a16emb4/.../2026-11-03-accuracy-drafter-3050oem). Paired by
question, 95% paired bootstrap range (2,000 resamples, seed 0). The GSM8K wall time here overlapped an Android
build on the host (12 jobs, 46 GB of RAM, from about 15:54), so it is not a speed result."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
NONE = os.path.join(HERE, "..", "2026-11-04-accuracy-3050oem")
VD = os.path.join(ROOT, "local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-accuracy-drafter-3050oem")
ARMS = {t: {"sglang-d4": (HERE, f"{t}-sglang-w4a16n4"), "sglang": (NONE, f"{t}-sglang"),
            "vllm-d4": (VD, f"{t}-vllm-w4a16n4")} for t in ("gsm8k", "bfcl_simple")}


def load(d, s):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(d, s + ".jsonl")))}


def wall(d, s):
    try:
        return json.loads(open(os.path.join(d, s + ".log")).read().strip().splitlines()[-1])["wall_s"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]


for t, arms in ARMS.items():
    rows = {n: load(*p) for n, p in arms.items()}
    for n, r in rows.items():
        right = sum(bool(x.get("right")) for x in r.values())
        trunc = sum(x.get("finish") == "length" for x in r.values())
        print(f"{t:12s} {n:10s} n={len(r)} right {right} acc {right / len(r):.4f} truncated {trunc} wall {wall(*arms[n])} s")
    for b in ("sglang", "vllm-d4"):
        x, y = rows["sglang-d4"], rows[b]
        ids = sorted(set(x) & set(y))
        d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
        lo, hi = boot(d)
        same = sum((x[i].get("text") or "", json.dumps(x[i].get("calls"), sort_keys=True))
                   == (y[i].get("text") or "", json.dumps(y[i].get("calls"), sort_keys=True)) for i in ids)
        print(f"{t:12s} sglang-d4 - {b}: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] "
              f"(+{d.count(1)} / -{d.count(-1)}) identical replies {same} of {len(ids)}")
