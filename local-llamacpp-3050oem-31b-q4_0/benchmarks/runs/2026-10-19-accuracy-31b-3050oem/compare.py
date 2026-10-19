"""31B scores with and without the drafter, paired by question with a 95% bootstrap range, then the
31B against the 26B and 12B sisters at their own serving settings (same binary 7182ef71e, card and
gen_eval.py scoring; context size and drafter length differ). Partial runs are paired on the
questions both have. Adapted from the 26B's 2026-10-18-accuracy-26b-3050oem/compare.py."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(HERE, "..", "..", "..", "..")
SISTERS = {
    "26B": (os.path.join(UP, "local-llamacpp-3050oem-26b-q4_0", "benchmarks", "runs",
                         "2026-10-18-accuracy-26b-3050oem"), {"q4x": "q4x", "none": "none"}),
    "12B": (os.path.join(UP, "local-llamacpp-3050oem-12b-q4_0", "benchmarks", "runs",
                         "2026-10-17-kvcache-12b-3050oem"), {"q4x": "q4x", "none": "q4none"}),
}


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f)) if "id" in r} if os.path.exists(f) else {}


def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]


def paired(x, y):
    ids = sorted(set(x) & set(y))
    d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
    lo, hi = boot(d)
    return ids, d, lo, hi


def acc(rows):
    return sum(bool(r.get("right")) for r in rows.values()) / len(rows)


for t in ("gsm8k", "bfcl_simple"):
    arms = {a: load(os.path.join(HERE, f"{t}-{a}.jsonl")) for a in ("q4x", "none")}
    for a, rows in arms.items():
        if rows:
            secs = sum(r.get("seconds") or 0 for r in rows.values())
            trunc = sum(r.get("finish") == "length" for r in rows.values())
            print(f"{t:12s} 31B {a:5s} n={len(rows)} right {sum(bool(r.get('right')) for r in rows.values())} "
                  f"acc {acc(rows):.4f} truncated {trunc} request time {secs:.0f} s")
    x, y = arms["q4x"], arms["none"]
    if x and y:
        ids, d, lo, hi = paired(x, y)
        same = sum((x[i].get("text") or "", json.dumps(x[i].get("calls"), sort_keys=True))
                   == (y[i].get("text") or "", json.dumps(y[i].get("calls"), sort_keys=True)) for i in ids)
        sx, sy = (sum(r.get("seconds") or 0 for i, r in z.items() if i in ids) for z in (x, y))
        print(f"{t:12s} 31B drafter - none: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] "
              f"disagree {sum(v != 0 for v in d)} (+{d.count(1)} / -{d.count(-1)}) "
              f"identical-text {same} of {len(ids)}; time none/drafter {sy / sx:.2f}x")
    for name, (base, m) in SISTERS.items():
        for a, a2 in m.items():
            z = load(os.path.join(base, f"{t}-{a2}.jsonl"))
            if not (z and arms[a]):
                continue
            ids, d, lo, hi = paired(arms[a], z)
            za = sum(bool(z[i].get("right")) for i in ids) / len(ids)
            print(f"{t:12s} 31B {a} - {name} {a}: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] "
                  f"(+{d.count(1)} / -{d.count(-1)} of {len(ids)}; {name} {za:.4f} on these)")
