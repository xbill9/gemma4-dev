#!/usr/bin/env python3
"""Statistical audit of devto-gemma4-e2b-thinking-1650ti.md, recomputed from the per-problem rows.
Wilson 95% intervals per cell, exact McNemar (two-sided binomial on discordant pairs) for every
paired comparison the article makes, and a minimum detectable difference for AIME's 30 problems.
Writes stat_audit.json and prints the report rows."""

import json
from itertools import product
from scipy.stats import binomtest
from math import sqrt


def proportion_confint(k, n, z=1.959963984540054, method="wilson"):
    ph = k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

G = "/home/xbill/gemma4-dev"
RUN = {
    "1650": f"{G}/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-11-23-thinking-e2b-1650ti",
    "3050": f"{G}/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem",
}
TASKS, W, D, T = ("gsm8k", "aime2025"), ("repack", "google"), ("none", "q4x"), ("off", "on")


def rows(card, task, w, d, t):
    f = f"{RUN[card]}/{task}-{w}-{d}-think{t}.jsonl"
    return {r["id"]: bool(r["right"]) for r in map(json.loads, open(f))}


def mcnemar(a, b):
    ids = sorted(a)
    assert sorted(b) == ids
    bb = sum(a[i] and not b[i] for i in ids)
    cc = sum(b[i] and not a[i] for i in ids)
    p = binomtest(bb, bb + cc).pvalue if bb + cc else 1.0
    return {
        "n": len(ids),
        "a_right": sum(a.values()),
        "b_right": sum(b.values()),
        "only_a": bb,
        "only_b": cc,
        "diff": sum(a.values()) - sum(b.values()),
        "p": p,
    }


out = {"cells": {}, "tests": {}}
for card, task, w, d, t in product(RUN, TASKS, W, D, T):
    r = rows(card, task, w, d, t)
    k, n = sum(r.values()), len(r)
    lo, hi = proportion_confint(k, n, method="wilson")
    out["cells"][f"{card} {task} {w}-{d} think{t}"] = {"k": k, "n": n, "rate": k / n, "wilson95": [lo, hi]}

fam = {"card": [], "thinking": [], "drafter": [], "build": []}
for task, w, d, t in product(TASKS, W, D, T):
    x = mcnemar(rows("1650", task, w, d, t), rows("3050", task, w, d, t))
    out["tests"][f"card 1650-vs-3050 {task} {w}-{d} think{t}"] = x
    fam["card"].append((task, x))
for card in RUN:
    for task, w, d in product(TASKS, W, D):
        x = mcnemar(rows(card, task, w, d, "on"), rows(card, task, w, d, "off"))
        out["tests"][f"thinking on-vs-off {card} {task} {w}-{d}"] = x
        fam["thinking"].append((task, x)) if card == "1650" else None
    for task, w, t in product(TASKS, W, T):
        x = mcnemar(rows(card, task, w, "q4x", t), rows(card, task, w, "none", t))
        out["tests"][f"drafter yes-vs-none {card} {task} {w} think{t}"] = x
        fam["drafter"].append((task, x)) if card == "1650" else None
    for task, d, t in product(TASKS, D, T):
        x = mcnemar(rows(card, task, "repack", d, t), rows(card, task, "google", d, t))
        out["tests"][f"build repack-vs-google {card} {task} {d} think{t}"] = x
        fam["build"].append((task, x)) if card == "1650" else None

summ = {}
for f, xs in fam.items():
    for task in TASKS:
        s = [x for tk, x in xs if tk == task]
        summ[f"{f} {task}"] = {
            "n_tests": len(s),
            "diff_min": min(x["diff"] for x in s),
            "diff_max": max(x["diff"] for x in s),
            "p_min": min(x["p"] for x in s),
            "p_max": max(x["p"] for x in s),
            "discordant_min": min(x["only_a"] + x["only_b"] for x in s),
            "discordant_max": max(x["only_a"] + x["only_b"] for x in s),
        }
within = [x["p"] for f in ("thinking", "drafter", "build") for _, x in fam[f]]
summ["1650 within-card 24 tests p_min"] = min(within)
gs = [c for k, c in out["cells"].items() if "gsm8k" in k]
summ["gsm8k rate range"] = {
    c: [
        min(v["rate"] for k, v in out["cells"].items() if k.startswith(c + " gsm8k")),
        max(v["rate"] for k, v in out["cells"].items() if k.startswith(c + " gsm8k")),
    ]
    for c in RUN
}

# AIME: smallest one-sided split that reaches p < 0.05 two-sided when every discordant pair goes one way,
# and the same for a 3:0 split -- what 30 problems with ~3-6 discordant pairs can show.
aime_min = {}
for m in range(1, 31):
    if binomtest(m, m).pvalue < 0.05:
        aime_min["all-one-way discordant pairs needed for p<0.05"] = m
        break
summ.update(aime_min)
out["summary"] = summ
json.dump(out, open(__file__.replace("audit_stats.py", "stat_audit.json"), "w"), indent=1)
for k, v in summ.items():
    print(k, v)
