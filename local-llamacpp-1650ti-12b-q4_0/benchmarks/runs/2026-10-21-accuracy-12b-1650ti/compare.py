"""The 12B on the 1650 Ti (layers split between card and CPU) against the same files on the RTX 3050
OEM (all layers on the card), paired by question with a 95% bootstrap range. Same llama.cpp commit
7182ef71e, GGUF, drafter, q4_0 cache and gen_eval.py scoring; the 3050 arms served -c 81920 with 2
drafted tokens (q4x) or -c 131072 without the drafter (q4none), this run -c 32768 with 3. GSM8K here
is the first N questions; pairing uses the questions both have. From the 3050 31B sister's
compare.py."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SISTER = os.path.join(HERE, "..", "..", "..", "..", "local-llamacpp-3050oem-12b-q4_0", "benchmarks", "runs",
                      "2026-10-17-kvcache-12b-3050oem")


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f)) if "id" in r} if os.path.exists(f) else {}


def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]


def acc(rows):
    return sum(bool(r.get("right")) for r in rows.values()) / len(rows)


for t in ("bfcl_simple", "gsm8k"):
    x = load(os.path.join(HERE, f"{t}-q4x.jsonl"))
    if not x:
        continue
    secs = sum(r.get("seconds") or 0 for r in x.values())
    trunc = sum(r.get("finish") == "length" for r in x.values())
    print(f"{t:12s} 1650ti q4x n={len(x)} right {sum(bool(r.get('right')) for r in x.values())} acc {acc(x):.4f} "
          f"truncated {trunc} request time {secs:.0f} s ({secs / len(x):.1f} s per question)")
    for arm in ("q4x", "q4none"):
        y = load(os.path.join(SISTER, f"{t}-{arm}.jsonl"))
        ids = sorted(set(x) & set(y))
        d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
        lo, hi = boot(d)
        same = sum((x[i].get("text") or "", json.dumps(x[i].get("calls"), sort_keys=True))
                   == (y[i].get("text") or "", json.dumps(y[i].get("calls"), sort_keys=True)) for i in ids)
        sy = sum(y[i].get("seconds") or 0 for i in ids)
        sx = sum(x[i].get("seconds") or 0 for i in ids)
        print(f"{t:12s}   vs 3050 {arm:6s} on {len(ids)}: 3050 acc {sum(bool(y[i].get('right')) for i in ids) / len(ids):.4f}, "
              f"1650ti - 3050 {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] disagree {sum(v != 0 for v in d)} "
              f"(+{d.count(1)} / -{d.count(-1)}) identical-text {same}; time 1650ti/3050 {sx / sy:.2f}x")
