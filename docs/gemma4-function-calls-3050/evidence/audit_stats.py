#!/usr/bin/env python3
"""Statistical audit of devto-gemma4-function-calls-3050.md, recomputed from the 36 per-problem files.
Independent of derived.py: re-reads the jsonl rows, recomputes every pooled score with a Wilson 95%
interval, every paired comparison the article quotes (exact McNemar = two-sided binomial on the
discordant pairs), every size step at the two recommended settings, and a Holm correction over the
family of paired tests the article reports. Writes stat_audit.json."""

import json
from itertools import combinations
from math import sqrt
from scipy.stats import binomtest

RUN = "/home/xbill/gemma4-dev/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-23-functions-e2b-e4b-12b-3050oem"
MODELS, SAMP, THINK = ("e2b", "e4b", "12b"), ("greedy", "sampled"), ("thinkoff", "thinkon")
CATS = ("simple", "multiple", "irrelevance")


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def cell(m, s, t, cats=CATS):
    out = {}
    for c in cats:
        for r in map(json.loads, open(f"{RUN}/bfcl_{c}-{m}-{s}-{t}.jsonl")):
            out[f"{c}:{r['id']}"] = bool(r["right"])
    return out


def mc(a, b):
    assert sorted(a) == sorted(b)
    x = sum(a[i] and not b[i] for i in a)
    y = sum(b[i] and not a[i] for i in a)
    return {
        "a_only": x,
        "b_only": y,
        "a_right": sum(a.values()),
        "b_right": sum(b.values()),
        "p": binomtest(x, x + y).pvalue if x + y else 1.0,
    }


C = {(m, s, t): cell(m, s, t) for m in MODELS for s in SAMP for t in THINK}
out = {"cells": {}, "tests": {}}
for (m, s, t), r in C.items():
    k = sum(r.values())
    out["cells"][f"{m} {s} {t}"] = {
        "k": k,
        "n": len(r),
        "rate": round(k / len(r), 4),
        "wilson95": wilson(k, len(r)),
        **{c: sum(v for i, v in r.items() if i.startswith(c + ":")) for c in CATS},
    }

T = out["tests"]
# settings within each model (article lines 27, 28, 38, 138, 140)
for m in MODELS:
    T[f"{m} thinkoff greedy-vs-sampled"] = mc(C[m, "greedy", "thinkoff"], C[m, "sampled", "thinkoff"])
    T[f"{m} sampled thinkon-vs-thinkoff"] = mc(C[m, "sampled", "thinkon"], C[m, "sampled", "thinkoff"])
    T[f"{m} greedy thinkon-vs-thinkoff"] = mc(C[m, "greedy", "thinkon"], C[m, "greedy", "thinkoff"])
    T[f"{m} thinkon greedy-vs-sampled"] = mc(C[m, "greedy", "thinkon"], C[m, "sampled", "thinkon"])
# every size step at the two recommended settings (lines 25, 26, 40, 218)
for s, t in (("greedy", "thinkoff"), ("sampled", "thinkon")):
    for a, b in combinations(MODELS, 2):
        T[f"size {b}-vs-{a} {s} {t}"] = mc(C[b, s, t], C[a, s, t])
# per-category 12B vs E4B (line 42) and the fitting-tool subset (line 155)
for c in CATS:
    T[f"12b-vs-e4b greedy thinkoff {c}"] = mc(
        cell("12b", "greedy", "thinkoff", (c,)), cell("e4b", "greedy", "thinkoff", (c,))
    )
fit = ("simple", "multiple")
T["12b greedy off vs e4b sampled on, fitting 600"] = mc(
    cell("12b", "greedy", "thinkoff", fit), cell("e4b", "sampled", "thinkon", fit)
)
T["12b greedy off vs e4b greedy off, fitting 600"] = mc(
    cell("12b", "greedy", "thinkoff", fit), cell("e4b", "greedy", "thinkoff", fit)
)
T["e4b sampled on vs 12b greedy off, all"] = mc(C["e4b", "sampled", "thinkon"], C["12b", "greedy", "thinkoff"])

# Holm over every paired test computed here
names = sorted(T, key=lambda k: T[k]["p"])
m_ = len(names)
running = 0.0
for i, k in enumerate(names):
    running = max(running, min(1.0, (m_ - i) * T[k]["p"]))
    T[k]["p_holm"] = running


# no-call counts on the 600 fitting problems (line 9, 38): reply made no tool call
def nocall(m, s, t):
    n = 0
    for c in fit:
        for r in map(json.loads, open(f"{RUN}/bfcl_{c}-{m}-{s}-{t}.jsonl")):
            n += not r.get("calls")
    return n


out["nocall_600"] = {f"{m} {s} {t}": nocall(m, s, t) for m in MODELS for s in SAMP for t in THINK}
json.dump(out, open(__file__.replace("audit_stats.py", "stat_audit.json"), "w"), indent=1)
for k in names:
    v = T[k]
    print(f"{k}: +{v['a_only']}/-{v['b_only']} p={v['p']:.3g} holm={v['p_holm']:.3g}")
for k, v in out["cells"].items():
    print(k, v["k"], v["wilson95"], v["simple"], v["multiple"], v["irrelevance"], "nocall", out["nocall_600"][k])
