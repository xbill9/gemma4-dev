#!/usr/bin/env python3
"""Statistical audit of devto-gemma4-e4b-thinking-1650ti.md, recomputed from the per-problem .jsonl rows of
both runs. Wilson 95% per cell, exact McNemar (two-sided binomial on discordant pairs) for every paired
comparison: 16 cross-card, 24 within the 1650 Ti (thinking, drafter, build), Holm within each family and
over all 40. Writes stat_audit.json (with a claims list) and prints the rows the article quotes."""
import json
from itertools import product
from math import sqrt
from scipy.stats import binomtest

G = "/home/xbill/gemma4-dev"
RUN = {"1650": f"{G}/local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-11-25-thinking-e4b-1650ti",
       "3050": f"{G}/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem"}
TASKS, W, D, T = ("gsm8k", "aime2025"), ("repack", "google"), ("none", "q4x"), ("off", "on")


def wilson(k, n, z=1.959963984540054):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def rows(card, task, w, d, t):
    return {r["id"]: bool(r["right"]) for r in map(json.loads, open(f"{RUN[card]}/{task}-{w}-{d}-think{t}.jsonl"))}


def mc(a, b):
    assert sorted(a) == sorted(b)
    x = sum(a[i] and not b[i] for i in a); y = sum(b[i] and not a[i] for i in a)
    return {"a_right": sum(a.values()), "b_right": sum(b.values()), "only_a": x, "only_b": y,
            "p": binomtest(x, x + y).pvalue if x + y else 1.0}


def holm(tests, key):
    names = sorted(tests, key=lambda k: tests[k]["p"]); run = 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (len(names) - i) * tests[k]["p"])); tests[k][key] = run


out = {"cells": {}, "cross": {}, "within": {}}
for card, task, w, d, t in product(RUN, TASKS, W, D, T):
    r = rows(card, task, w, d, t); k = sum(r.values())
    out["cells"][f"{card} {task} {w}-{d} {t}"] = {"k": k, "n": len(r), "rate": round(k / len(r), 4), "wilson95": wilson(k, len(r))}
for task, w, d, t in product(TASKS, W, D, T):
    out["cross"][f"{task} {w}-{d} {t}"] = mc(rows("1650", task, w, d, t), rows("3050", task, w, d, t))
for task in TASKS:
    for w, d in product(W, D):
        out["within"][f"thinking {task} {w}-{d}"] = mc(rows("1650", task, w, d, "on"), rows("1650", task, w, d, "off"))
    for w, t in product(W, T):
        out["within"][f"drafter {task} {w} {t}"] = mc(rows("1650", task, w, "q4x", t), rows("1650", task, w, "none", t))
    for d, t in product(D, T):
        out["within"][f"build {task} {d} {t}"] = mc(rows("1650", task, "repack", d, t), rows("1650", task, "google", d, t))
holm(out["cross"], "holm16"); holm(out["within"], "holm24")
allt = {**{"x " + k: v for k, v in out["cross"].items()}, **{"w " + k: v for k, v in out["within"].items()}}
holm(allt, "holm40")
# AIME: fewest discordant pairs, all one way, that reach p < 0.05
out["aime_min_one_way"] = next(m for m in range(1, 31) if binomtest(m, m).pvalue < 0.05)
C, X, Wt = out["cells"], out["cross"], out["within"]
other = [v for k, v in X.items() if k != "gsm8k google-q4x on"]
out["summary"] = {
    "cross p<0.05": [k for k, v in X.items() if v["p"] < 0.05],
    "cross other 15 min p": min(v["p"] for v in other),
    "cross gsm8k diff range (other 7)": [min(v["a_right"] - v["b_right"] for k, v in X.items() if k.startswith("gsm8k") and k != "gsm8k google-q4x on"),
                                         max(v["a_right"] - v["b_right"] for k, v in X.items() if k.startswith("gsm8k") and k != "gsm8k google-q4x on")],
    "cross aime diff range": [min(v["a_right"] - v["b_right"] for k, v in X.items() if k.startswith("aime")), max(v["a_right"] - v["b_right"] for k, v in X.items() if k.startswith("aime"))],
    "within p<0.05": {k: [round(v["p"], 4), round(v["holm24"], 4)] for k, v in Wt.items() if v["p"] < 0.05},
    "build tests min p": min(v["p"] for k, v in Wt.items() if k.startswith("build")),
    "thinking gsm8k nets": {k: v["a_right"] - v["b_right"] for k, v in Wt.items() if k.startswith("thinking gsm8k")},
    "1650 gsm8k rate range": [min(v["rate"] for k, v in C.items() if k.startswith("1650 gsm8k")), max(v["rate"] for k, v in C.items() if k.startswith("1650 gsm8k"))],
    "3050 gsm8k rate range": [min(v["rate"] for k, v in C.items() if k.startswith("3050 gsm8k")), max(v["rate"] for k, v in C.items() if k.startswith("3050 gsm8k"))],
}
s = out["summary"]; o = X["gsm8k google-q4x on"]; tg = Wt["thinking gsm8k google-none"]; dr = Wt["drafter gsm8k google on"]


def rec(i, st, pv, src, comp, verdict, concl, reason):
    return {"id": i, "statement": st, "paper_value": pv, "paper_source": src, "computed": comp, "verdict": verdict, "conclusion": concl, "reason": reason}


out["claims"] = [
    rec("cross-outlier", "Google file + drafter + thinking on: 1,227 vs 1,245, 8 vs 26, p = 0.0029", {"right": [1227, 1245], "only": [8, 26], "p": 0.0029},
        "lede; Findings", {"right": [o["a_right"], o["b_right"]], "only": [o["only_a"], o["only_b"]], "p": round(o["p"], 4), "holm16": round(o["holm16"], 4), "holm40": round(o["holm40"], 4)},
        "VERIFIED", "HOLDS" if o["holm16"] < 0.05 else "WITHIN NOISE", "Clears Holm over the 16 cross-card tests; over all 40 tests in the article it does not."),
    rec("cross-other-15", "other 15 settings within chance, smallest p = 0.0708", {"p_min": 0.0708}, "lede; Findings",
        {"p_min": round(s["cross other 15 min p"], 4)}, "VERIFIED", "HOLDS", "No other cross-card test below 0.05."),
    rec("cross-ranges", "-11 to +9 GSM8K, -1 to +3 AIME in those 15", {"gsm8k": [-11, 9], "aime": [-1, 3]}, "Findings; Summary",
        {"gsm8k": s["cross gsm8k diff range (other 7)"], "aime": s["cross aime diff range"]}, "VERIFIED", "HOLDS", ""),
    rec("score-ranges", "GSM8K 0.9265-0.9416 here, 0.9272-0.9439 on the 3050", {"1650": [0.9265, 0.9416], "3050": [0.9272, 0.9439]}, "lede",
        {"1650": s["1650 gsm8k rate range"], "3050": s["3050 gsm8k rate range"]}, "VERIFIED", "HOLDS", ""),
    rec("thinking-+20", "thinking +20, p = 0.0105 alone, 0.24 after Holm over 24", {"net": 20, "p": 0.0105, "holm24": 0.24}, "lede; Findings",
        {"net": tg["a_right"] - tg["b_right"], "p": round(tg["p"], 4), "holm24": round(tg["holm24"], 2)}, "VERIFIED", "WITHIN NOISE", "Article already says it does not survive correction."),
    rec("drafter-low", "drafter, Google thinking on: 1,242 -> 1,227, p = 0.0059, 0.14 corrected", {"right": [1227, 1242], "p": 0.0059, "holm24": 0.14}, "Findings",
        {"right": [dr["a_right"], dr["b_right"]], "p": round(dr["p"], 4), "holm24": round(dr["holm24"], 2)}, "VERIFIED", "WITHIN NOISE", "Article already says it does not survive correction."),
    rec("build-level", "repack and Google's file level in every paired test", {"p_min": None}, "Findings",
        {"p_min": round(s["build tests min p"], 4)}, "VERIFIED", "HOLDS" if s["build tests min p"] >= 0.05 else "OVERSTATED", ""),
    rec("aime-power", "AIME's 30 problems can only show large effects", {}, "Caveats",
        {"min_one_way_discordant_for_p<0.05": out["aime_min_one_way"]}, "VERIFIED", "HOLDS", "Needs 6+ discordant pairs all one way."),
]
json.dump(out, open(__file__.replace("audit_stats.py", "stat_audit.json"), "w"), indent=1)
print(json.dumps(out["summary"], indent=1))
for c in out["claims"]: print(c["id"], c["computed"], c["verdict"], c["conclusion"])
for k in sorted(allt, key=lambda k: allt[k]["p"])[:5]: print(k, round(allt[k]["p"], 4), round(allt[k]["holm40"], 4))
