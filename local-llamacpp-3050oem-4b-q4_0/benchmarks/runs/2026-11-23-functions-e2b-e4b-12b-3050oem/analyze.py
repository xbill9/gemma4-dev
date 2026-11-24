#!/usr/bin/env python3
"""analyze.py -- scores, 95% Wilson ranges, hours, thinking presence and paired exact McNemar tests for the
36 cells run.sh wrote. Writes RESULTS.md. All arithmetic here; the article quotes it."""
import json, os
from scipy.stats import binomtest, norm

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS, TASKS = ["e2b", "e4b", "12b"], ["bfcl_simple", "bfcl_multiple", "bfcl_irrelevance"]
SETS = [("greedy", "thinkoff"), ("sampled", "thinkoff"), ("sampled", "thinkon"), ("greedy", "thinkon")]


def load(m, t, mode, th):
    return {r["id"]: r for r in map(json.loads, open(f"{HERE}/{t}-{m}-{mode}-{th}.jsonl"))}


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (c - h) / d, (c + h) / d


def mcnemar(a, b):
    """a, b: {id: row} on the same ids. Returns (only a right, only b right, exact two-sided p)."""
    ids = sorted(a); x = sum(a[i]["right"] and not b[i]["right"] for i in ids); y = sum(b[i]["right"] and not a[i]["right"] for i in ids)
    return x, y, (binomtest(x, x + y).pvalue if x + y else 1.0)


def pooled(m, mode, th):
    out = {}
    for t in TASKS:
        out.update({f"{t}:{k}": v for k, v in load(m, t, mode, th).items()})
    return out


L = []
L.append("# Function calling on the 3050: results\n\nBFCL v3 simple (400), multiple (200), irrelevance (240); "
         "exact Q4_0, llama.cpp d1044bc82, no drafter, one request at a time. `sampled` = temperature 1.0, "
         "top_k 64, top_p 0.95, seed 1. `thought` = replies carrying thinking text. Hours = sum of reply times.\n")
L.append("## Per cell\n\n| model | task | setting | right | score | 95% range | thought | cut at 8,192 | mean tokens | hours |\n|---|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for mode, th in SETS:
        for t in TASKS:
            r = load(m, t, mode, th); n = len(r); k = sum(x["right"] for x in r.values()); lo, hi = wilson(k, n)
            th_n = sum(bool(x.get("reasoning")) for x in r.values()); cut = sum(x.get("finish") == "length" for x in r.values())
            tok = sum(x.get("completion_tokens") or 0 for x in r.values()) / n; h = sum(x["seconds"] for x in r.values()) / 3600
            L.append(f"| {m} | {t[5:]} | {mode} {th} | {k}/{n} | {k/n:.4f} | {lo:.4f}-{hi:.4f} | {th_n}/{n} | {cut} | {tok:.0f} | {h:.3f} |")
L.append("\n## Pooled over all 840 problems\n\n| model | setting | right | score | 95% range | thought | hours |\n|---|---|---|---|---|---|---|")
for m in MODELS:
    for mode, th in SETS:
        r = pooled(m, mode, th); n = len(r); k = sum(x["right"] for x in r.values()); lo, hi = wilson(k, n)
        th_n = sum(bool(x.get("reasoning")) for x in r.values()); h = sum(x["seconds"] for x in r.values()) / 3600
        L.append(f"| {m} | {mode} {th} | {k}/{n} | {k/n:.4f} | {lo:.4f}-{hi:.4f} | {th_n}/{n} | {h:.3f} |")

L.append("\n## Paired tests (same problems; exact two-sided McNemar)\n\n| comparison | scope | A right only | B right only | p |\n|---|---|---|---|---|")
def row(name, scope, a, b):
    x, y, p = mcnemar(a, b); L.append(f"| {name} | {scope} | {x} | {y} | {p:.4g} |")
for m in MODELS:
    for th in ("thinkoff", "thinkon"):
        for t in TASKS:
            row(f"{m} {th}: greedy (A) vs sampled (B)", t[5:], load(m, t, "greedy", th), load(m, t, "sampled", th))
        row(f"{m} {th}: greedy (A) vs sampled (B)", "all 840", pooled(m, "greedy", th), pooled(m, "sampled", th))
    for mode in ("greedy", "sampled"):
        for t in TASKS:
            row(f"{m} {mode}: thinking on (A) vs off (B)", t[5:], load(m, t, mode, "thinkon"), load(m, t, mode, "thinkoff"))
        row(f"{m} {mode}: thinking on (A) vs off (B)", "all 840", pooled(m, mode, "thinkon"), pooled(m, mode, "thinkoff"))
for a, b in (("12b", "e4b"), ("e4b", "e2b"), ("12b", "e2b")):
    for mode, th in SETS:
        for t in TASKS:
            row(f"{a} (A) vs {b} (B), {mode} {th}", t[5:], load(a, t, mode, th), load(b, t, mode, th))
        row(f"{a} (A) vs {b} (B), {mode} {th}", "all 840", pooled(a, mode, th), pooled(b, mode, th))
# best E4B setting against the 12B's best-looking settings
for mb, thb in (("greedy", "thinkoff"), ("sampled", "thinkon")):
    row(f"12b {mb} {thb} (A) vs e4b greedy thinkon (B)", "all 840", pooled("12b", mb, thb), pooled("e4b", "greedy", "thinkon"))
    row(f"12b {mb} {thb} (A) vs e4b sampled thinkon (B)", "all 840", pooled("12b", mb, thb), pooled("e4b", "sampled", "thinkon"))

# Thinking-off sampled replies: does carrying thinking go with being wrong?
L.append("\n## Sampled, thinking off: right rate by whether the reply carried thinking\n\n| model | task | thought: right/n | no thought: right/n |\n|---|---|---|---|")
for m in MODELS:
    for t in TASKS:
        r = load(m, t, "sampled", "thinkoff").values()
        a = [x for x in r if x.get("reasoning")]; b = [x for x in r if not x.get("reasoning")]
        L.append(f"| {m} | {t[5:]} | {sum(x['right'] for x in a)}/{len(a)} | {sum(x['right'] for x in b)}/{len(b)} |")

# Why replies are wrong, per cell
L.append("\n## Wrong replies by kind\n\n| model | task | setting | kinds |\n|---|---|---|---|")
for m in MODELS:
    for mode, th in SETS:
        for t in TASKS:
            kinds = {}
            for x in load(m, t, mode, th).values():
                if not x["right"]:
                    k = x.get("reason", "error"); k = "value" if k.startswith("value") else ("missing" if k.startswith("missing") else k)
                    if x.get("finish") == "length": k = "cut at 8,192"
                    kinds[k] = kinds.get(k, 0) + 1
            L.append(f"| {m} | {t[5:]} | {mode} {th} | " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) + " |")
open(f"{HERE}/RESULTS.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
