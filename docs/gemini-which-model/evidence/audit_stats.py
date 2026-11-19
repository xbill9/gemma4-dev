#!/usr/bin/env python3
"""Statistical audit of devto-gemini-which-model.md, recomputed from per-question rows.

Tests: Wilson 95% interval for k/n; exact two-sided McNemar on paired questions (b = only the first
right, c = only the second right); Wilcoxon signed-rank (scipy, two-sided) on paired per-problem cost
and time. Writes stat_audit.json (claims list read by check-facts.py) and audit_stats.txt.
"""

import json
import math
import os
import statistics

from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HL = os.path.join(ROOT, "docs", "gemini-high-vs-low", "evidence")
NEON1 = os.path.join(ROOT, "gemini-api-google-neon4", "benchmarks", "runs", "2026-11-19-thinking-levels", "aime")
SHORT = "benchmarks/runs/2026-11-14-accuracy-api"
NEONSHORT = os.path.join(ROOT, "gemini-api-google-neon4", "benchmarks", "runs", "2026-11-19-accuracy-api")
MODELS = {m["name"]: m for m in json.load(open(os.path.join(HERE, "models.json")))}


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p)) if "id" in r} if os.path.exists(p) else {}


def aime(name):  # pooled AIME I + II for e.g. "argon-4-low"
    base, arm = name.rsplit("-", 1)
    if base == "neon-4":
        one = load(os.path.join(NEON1, f"neon-{arm}.jsonl"))
    else:
        one = load(os.path.join(HL, f"aime-{arm}", base + ".jsonl"))
    return {**one, **load(os.path.join(HERE, "aime2", name + ".jsonl"))}


def aime2(name):
    return load(os.path.join(HERE, "aime2", name + ".jsonl"))


def short(slot, task):
    if slot == "neon4":
        return load(os.path.join(NEONSHORT, task + ".jsonl"))
    return load(os.path.join(ROOT, f"gemini-api-google-{slot}", SHORT, task + ".jsonl"))


def wilson(k, n, z=1.959964):
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def paired(A, B):
    ids = sorted(set(A) & set(B))
    b = sum(bool(A[i].get("right")) and not B[i].get("right") for i in ids)
    c = sum(bool(B[i].get("right")) and not A[i].get("right") for i in ids)
    return b, c, len(ids)


def cost(r, m):
    return (
        (r.get("prompt_tokens") or 0) * m["price_in"]
        + ((r.get("completion_tokens") or 0) + (r.get("reasoning_tokens") or 0)) * m["price_out"]
    ) / 1e6


claims, lines = [], []


def rec(cid, statement, source, paper, computed, verdict, conclusion, test, reason=""):
    claims.append(
        {
            "id": cid,
            "statement": statement,
            "paper_value": paper,
            "paper_source": source,
            "computed": computed,
            "verdict": verdict,
            "conclusion": conclusion,
            "test": test,
            "reason": reason,
        }
    )
    lines.append(f"| {statement} | {paper} | {computed} | {test} | {conclusion} |")


# 1. Wilson on 30/30 and the small-model scores
for name, k_claim in [
    ("argon-4-low", 30),
    ("flash-lite-3.5-low", 18),
    ("flash-lite-3.5-high", 24),
    ("flash-3.8-low", 27),
    ("flash-3.8-high", 29),
]:
    rows = aime(name)
    k, n = sum(bool(r.get("right")) for r in rows.values()), len(rows)
    lo, hi = wilson(k, n)
    v = "VERIFIED" if (k, n) == (k_claim, 30) else "MISMATCH"
    paper = {"k": k_claim, "n": 30, "ci": [0.89, 1.0]} if name == "argon-4-low" else {"k": k_claim, "n": 30}
    rec(
        f"wilson-{name}",
        f"{name} {k_claim} of 30 on AIME 2025 I+II" + (", 95% range 0.89 to 1.00" if name == "argon-4-low" else ""),
        "Step 2 table; line 117",
        paper,
        {"k": k, "n": n, "ci": [round(lo, 4), round(hi, 4)]},
        v,
        "HOLDS" if v == "VERIFIED" else "OVERSTATED",
        "Wilson 95%",
    )
for name, k_claim in [
    ("hydrogen-4-low", 30),
    ("carbon-4-low", 30),
    ("neon-4-low", 30),
    ("hydrogen-4-high", 30),
    ("carbon-4-high", 30),
    ("argon-4-high", 30),
    ("neon-4-high", 30),
]:
    rows = aime(name)
    k, n = sum(bool(r.get("right")) for r in rows.values()), len(rows)
    rec(
        f"score-{name}",
        f"{name} {k_claim} of 30",
        "Step 2 table",
        {"k": k_claim, "n": 30},
        {"k": k, "n": n},
        "VERIFIED" if (k, n) == (k_claim, 30) else "MISMATCH",
        "HOLDS" if (k, n) == (k_claim, 30) else "OVERSTATED",
        "count",
    )

# 2. McNemar claims
for cid, a, b_, paper_bc, paper_p, stmt, concl_expect in [
    (
        "argon-vs-flashlite-low",
        "argon-4-low",
        "flash-lite-3.5-low",
        (12, 0),
        "< 0.001",
        "Argon lowest solved the 12 Flash-Lite lowest missed, lost none, p < 0.001",
        "sig",
    ),
    (
        "flashlite-high-vs-low",
        "flash-lite-3.5-high",
        "flash-lite-3.5-low",
        (6, 0),
        0.031,
        "Flash-Lite high vs lowest +6 / -0, p = 0.031",
        "sig",
    ),
    (
        "flash38-high-vs-low",
        "flash-3.8-high",
        "flash-3.8-low",
        (2, 0),
        0.50,
        "3.8 Flash high vs lowest +2 / -0, p = 0.50 (could be chance)",
        "ns",
    ),
    (
        "argon-vs-flash38-low",
        "argon-4-low",
        "flash-3.8-low",
        (3, 0),
        0.25,
        "3.8 Flash lowest missed 3 Argon solved, p = 0.25 (could be chance)",
        "ns",
    ),
]:
    b, c, n = paired(aime(a), aime(b_))
    p = mcnemar(b, c)
    ok_bc = (b, c) == paper_bc
    ok_p = (p < 0.001) if paper_p == "< 0.001" else abs(round(p, 3 if paper_p < 0.1 else 2) - paper_p) < 1e-9
    concl = (
        ("HOLDS" if p < 0.05 else "OVERSTATED") if concl_expect == "sig" else ("HOLDS" if p >= 0.05 else "OVERSTATED")
    )
    rec(
        cid,
        stmt,
        "The Findings; Step 3 table",
        {"b": paper_bc[0], "c": paper_bc[1], "p": paper_p},
        {"b": b, "c": c, "n": n, "p": round(p, 6)},
        "VERIFIED" if ok_bc and ok_p else "MISMATCH",
        concl,
        "exact McNemar",
    )

# 3. "No Gemini 4 model can be ranked": all pairs 0 discordant
for a, b_ in [
    ("neon-4-high", "carbon-4-high"),
    ("argon-4-low", "hydrogen-4-low"),
    ("argon-4-low", "neon-4-low"),
    ("argon-4-low", "carbon-4-low"),
]:
    b, c, n = paired(aime(a), aime(b_))
    rec(
        f"tie-{a}-{b_}",
        f"{a} and {b_} cannot be ranked on AIME (same answers)",
        "The Findings, Accuracy",
        {"b": 0, "c": 0},
        {"b": b, "c": c, "n": n, "p": mcnemar(b, c)},
        "VERIFIED" if (b, c) == (0, 0) else "MISMATCH",
        "HOLDS" if (b, c) == (0, 0) else "OVERSTATED",
        "exact McNemar",
    )

# 4. Neon short-task comparisons (article line 94), exact McNemar alongside the bootstrap the article used
claims94 = {
    "gsm8k": {
        "31b": "ahead",
        "26b": "ahead",
        "flashlite35": "ahead",
        "flash38": "level",
        "hydrogen4": "level",
        "carbon4": "level",
        "argon4": "level",
    },
    "bfcl_simple": {
        "hydrogen4": "ahead",
        "carbon4": "level",
        "flash38": "ahead",
        "flashlite35": "ahead",
        "argon4": "level",
        "31b": "level",
    },
}
for task, m in claims94.items():
    N = short("neon4", task)
    for slot, said in m.items():
        b, c, n = paired(N, short(slot, task))
        p = mcnemar(b, c)
        holds = (p < 0.05 and b > c) if said == "ahead" else p >= 0.05
        rec(
            f"neon-{task}-{slot}",
            f"Neon {task} {said} {'of' if said == 'ahead' else 'with'} {slot}",
            "Step 1, line 94",
            {"said": said},
            {"b": b, "c": c, "n": n, "p": round(p, 5)},
            "VERIFIED" if holds else "MISMATCH",
            "HOLDS" if holds else "OVERSTATED",
            "exact McNemar",
        )


# 5. Paired resource claims, Wilcoxon signed-rank on AIME II per-problem values
def wil(a, b_, key):
    A, B = aime2(a), aime2(b_)
    ids = sorted(set(A) & set(B))
    if key == "cost":
        x = [cost(A[i], MODELS[a]) for i in ids]
        y = [cost(B[i], MODELS[b_]) for i in ids]
        ratio = sum(x) / sum(y)
    else:
        x = [A[i]["seconds"] for i in ids]
        y = [B[i]["seconds"] for i in ids]
        ratio = statistics.median(x) / statistics.median(y)
    st = wilcoxon(x, y)
    lower = sum(xi < yi for xi, yi in zip(x, y))
    return float(st.pvalue), lower, len(ids), ratio


# The record's verdict: does the printed ratio reproduce. The conclusion: does the paired test support the direction.
for cid, a, b_, key, stmt, ratio, expect_lower in [
    ("neon-carbon-high-cost", "neon-4-high", "carbon-4-high", "cost", "Neon high cost 0.73x Carbon high per problem, cheaper on 13 of 15, AIME II", 0.73, True),
    ("neon-carbon-high-time", "neon-4-high", "carbon-4-high", "time", "Neon high median time 0.68x Carbon high; faster on 9 of 15, could be chance", 0.68, True),
    ("hydrogen-argon-low-time", "hydrogen-4-low", "argon-4-low", "time", "Hydrogen lowest fastest Gemini 4 (11.3 s vs Argon 16.8 s), AIME II", None, True),
    ("argon-hydrogen-low-cost", "argon-4-low", "hydrogen-4-low", "cost", "Argon lowest cheaper per problem than Hydrogen lowest, AIME II", None, True),
    ("argon-carbon-low-cost", "argon-4-low", "carbon-4-low", "cost", "Argon lowest cheaper per problem than Carbon lowest, AIME II", None, True),
    ("neon-hydrogen-low-time", "neon-4-low", "hydrogen-4-low", "time", "Neon lowest slowest (40.8 s vs Hydrogen 11.3 s), AIME II", None, False),
]:
    p, lower, n, r = wil(a, b_, key)
    direction_ok = (lower > n / 2) if expect_lower else (lower < n / 2)
    sig = p < 0.05 and direction_ok
    verdict = "VERIFIED" if ratio is None or round(r, 2) == ratio else "MISMATCH"
    concl = "HOLDS" if sig else ("HOLDS" if "could be chance" in stmt else "WITHIN NOISE")
    rec(cid, stmt, "The Findings; Step 4; Step 5", {"ratio": ratio} if ratio else {"direction": "lower" if expect_lower else "higher"},
        {"ratio": round(r, 4), "wilcoxon_p": round(p, 6), "first_lower_on": lower, "n": n}, verdict, concl,
        "Wilcoxon signed-rank, paired per problem", "" if sig else "difference within noise; the article says so")

json.dump(
    {"article": "docs/gemini-which-model/devto-gemini-which-model.md", "claims": claims},
    open(os.path.join(HERE, "stat_audit.json"), "w"),
    indent=1,
)
hdr = "| Claim | Article value | Recomputed | Test used | Verdict |\n|---|---|---|---|---|"
txt = hdr + "\n" + "\n".join(lines) + "\n"
open(os.path.join(HERE, "audit_stats.txt"), "w").write(txt)
print(txt)
print({v: sum(c["conclusion"] == v for c in claims) for v in ("HOLDS", "WITHIN NOISE", "OVERSTATED", "CANNOT CHECK")})
