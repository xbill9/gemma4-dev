#!/usr/bin/env python3
"""Statistical audit of devto-gemma4-e2b-long-context-3050.md, recomputed from longctx.jsonl alone.
Independent of derived.py: per-cell facts right with Wilson 95% (facts treated as independent) and a
trial-clustered interval (bootstrap over the 10 trials, since the 8 facts of a trial share one reply),
every trial-level exact sign test the article quotes, Holm over the 13 trial-level tests derived.py
corrects for, and the ratios behind 'about twice'. Writes stat_audit.json."""

import json
import random
from math import sqrt
from scipy.stats import binomtest

RUN = "/home/xbill/gemma4-dev/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-25-longctx-e2b-3050oem"
rows = [json.loads(line) for line in open(f"{RUN}/longctx.jsonl")]
R = {(r["arm"], r["task"], r["depth"], r["trial"]): r["right"] for r in rows}
DEPTHS = sorted({r["depth"] for r in rows})
MID = [d for d in DEPTHS if d <= 98304]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def boot(vals, n_facts=8, reps=20000, seed=1):
    rng = random.Random(seed)
    m = len(vals)
    s = sorted(sum(rng.choice(vals) for _ in range(m)) / (m * n_facts) for _ in range(reps))
    return [round(s[int(0.025 * reps)], 3), round(s[int(0.975 * reps)], 3)]


def keys(arm, depths, tasks=("recall", "link")):
    return [(arm, t, d, i) for t in tasks for d in depths for i in range(10) if (arm, t, d, i) in R]


def sign(a, b, depths, tasks=("recall", "link")):
    ka = keys(a, depths, tasks)
    pairs = [(R[k], R[(b,) + k[1:]]) for k in ka if (b,) + k[1:] in R]
    up = sum(x > y for x, y in pairs)
    dn = sum(x < y for x, y in pairs)
    facts = (sum(x for x, _ in pairs), sum(y for _, y in pairs))
    return {
        "trials": len(pairs),
        "ahead": up,
        "behind": dn,
        "p": binomtest(up, up + dn).pvalue if up + dn else 1.0,
        "facts_a": facts[0],
        "facts_b": facts[1],
    }


out = {"cells": {}, "tests": {}}
arms = sorted({r["arm"] for r in rows})
for a in arms:
    for t in ("recall", "link"):
        for d in DEPTHS:
            v = [R[(a, t, d, i)] for i in range(10) if (a, t, d, i) in R]
            if v:
                out["cells"][f"{a} {t} {d}"] = {
                    "k": sum(v),
                    "n": 8 * len(v),
                    "wilson95": wilson(sum(v), 8 * len(v)),
                    "trial_bootstrap95": boot(v),
                }

T = out["tests"]
T["no drafter vs drafter, all five"] = sign("f16-none", "f16-draft", DEPTHS)
T["q8 vs f16, all five"] = sign("q8-draft", "f16-draft", DEPTHS)
T["q4 vs f16, all five"] = sign("q4-draft", "f16-draft", DEPTHS)
T["q8 vs f16, 16k-98k"] = sign("q8-draft", "f16-draft", MID)
T["q4 vs f16, 16k-98k"] = sign("q4-draft", "f16-draft", MID)
T["no drafter vs drafter, 16k-98k"] = sign("f16-none", "f16-draft", MID)
T["sampled off vs f16-draft, 16k-98k"] = sign("f16-sampled-off", "f16-draft", MID)
T["sampled on vs f16-draft, 16k-98k"] = sign("f16-sampled-on", "f16-draft", MID)
T["thinking on vs off, both sampled, 16k-98k"] = sign("f16-sampled-on", "f16-sampled-off", MID)
T["sampled on vs f16-none, 16k-98k"] = sign("f16-sampled-on", "f16-none", MID)
T["sampled off vs f16-none, all five"] = sign("f16-sampled-off", "f16-none", DEPTHS)
T["sampled on vs f16-draft, recall, 16k-98k"] = sign("f16-sampled-on", "f16-draft", MID, ("recall",))
T["sampled on vs f16-draft, link, 16k-98k"] = sign("f16-sampled-on", "f16-draft", MID, ("link",))
names = sorted(T, key=lambda k: T[k]["p"])
run = 0.0
for i, k in enumerate(names):
    run = max(run, min(1.0, (len(names) - i) * T[k]["p"]))
    T[k]["p_holm"] = run

c = out["cells"]
tot = lambda a, t: sum(c[f"{a} {t} {d}"]["k"] for d in MID)
out["ratios"] = {
    "recall on/draft": round(tot("f16-sampled-on", "recall") / tot("f16-draft", "recall"), 2),
    "link on/draft": round(tot("f16-sampled-on", "link") / tot("f16-draft", "link"), 2),
    "both": [
        tot("f16-sampled-on", "recall") + tot("f16-sampled-on", "link"),
        tot("f16-draft", "recall") + tot("f16-draft", "link"),
    ],
}
json.dump(out, open(__file__.replace("audit_stats.py", "stat_audit.json"), "w"), indent=1)
for k in names:
    v = T[k]
    print(
        f"{k}: trials {v['trials']} +{v['ahead']}/-{v['behind']} p={v['p']:.3g} holm={v['p_holm']:.3g} facts {v['facts_a']} vs {v['facts_b']}"
    )
print(out["ratios"])
for k in ("f16-draft recall 16384", "f16-draft recall 32768", "f16-draft recall 65536", "f16-sampled-on recall 32768"):
    pass
for k, v in c.items():
    if k.startswith(("f16-draft", "f16-sampled-on")):
        print(k, v)
