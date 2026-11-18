#!/usr/bin/env python3
"""Statistical audit of the E2B-vs-E4B comparison saved for the summary paper, recomputed from the raw replies (not from
analyze.py's output). Wilson intervals, exact McNemar via scipy binomtest, Wilcoxon signed-rank on
generated tokens, and the noise floor. Writes stat_audit.json and audit_stats.txt."""

import json, math, os, itertools
from scipy.stats import binomtest, wilcoxon, norm

H = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.normpath(
    os.path.join(H, "../../../local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem")
)
ARMS = ["repack-none", "repack-q4x", "google-none", "google-q4x"]


def rows(t, a, th):
    return {r["id"]: r for r in map(json.loads, open(f"{RUN}/{t}-{a}-think{th}.jsonl"))}


D = {(t, a, th): rows(t, a, th) for t in ("gsm8k", "aime2025") for a in ARMS for th in ("off", "on")}


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def mc(x, y):
    A, B = D[x], D[y]
    assert A.keys() == B.keys()
    b = sum(A[i]["right"] and not B[i]["right"] for i in A)
    c = sum(B[i]["right"] and not A[i]["right"] for i in A)
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)


claims, out = [], []


def claim(id, st, pv, src, comp, ok, reason, concl):
    claims.append(
        dict(
            id=id,
            statement=st,
            paper_value=pv,
            paper_source=src,
            computed=comp,
            verdict="VERIFIED" if ok else "MISMATCH",
            reason=reason,
            conclusion=concl,
        )
    )
    out.append(f"{'VERIFIED' if ok else 'MISMATCH':9} {concl:13} {id}: {st} | paper {pv} | computed {comp}")


ART = "e2b-vs-e4b.md"
# 5. E4B against E2B, same cell, same problems (both runs' raw replies)
RUN2 = os.path.normpath(
    os.path.join(H, "../../../local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem")
)


def rows2(t, a, th):
    return {r["id"]: r for r in map(json.loads, open(f"{RUN2}/{t}-{a}-think{th}.jsonl"))}


gain, gp, aoff, aon = [], [], [], []
for t in ("gsm8k", "aime2025"):
    for a in ARMS:
        for th in ("off", "on"):
            A, B = rows2(t, a, th), D[(t, a, th)]
            assert A.keys() == B.keys()
            b = sum(B[i]["right"] and not A[i]["right"] for i in A)
            c = sum(A[i]["right"] and not B[i]["right"] for i in A)
            p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
            k2, k4 = sum(x["right"] for x in A.values()), sum(x["right"] for x in B.values())
            if t == "gsm8k":
                gain.append(k4 - k2)
                gp.append(p)
            else:
                (aoff if th == "off" else aon).append((k2, k4, p))
claim(
    "e2b->e4b gsm8k gain",
    "E4B got 35 to 58 more GSM8K problems right than E2B",
    {"min": 35, "max": 58},
    f"{ART} Findings",
    {"min": min(gain), "max": max(gain)},
    [min(gain), max(gain)] == [35, 58],
    "k(E4B) - k(E2B) per cell",
    "HOLDS",
)
claim(
    "e2b->e4b gsm8k p",
    "beyond chance in all eight setups (p = 0.00082 or smaller)",
    {"p_max": 0.00082},
    f"{ART} Findings",
    {"p_max": float(f"{max(gp):.2g}")},
    round(max(gp), 5) == 0.00082,
    "exact McNemar per cell",
    "HOLDS",
)
claim(
    "e2b->e4b aime off",
    "AIME thinking off: E4B 10 to 13 against E2B 5 to 7, beyond chance in two of four",
    {"e4b": [10, 13], "e2b": [5, 7], "passing": 2},
    f"{ART} Step 5",
    {
        "e4b": [min(x[1] for x in aoff), max(x[1] for x in aoff)],
        "e2b": [min(x[0] for x in aoff), max(x[0] for x in aoff)],
        "passing": sum(x[2] < 0.05 for x in aoff),
    },
    (
        [min(x[1] for x in aoff), max(x[1] for x in aoff)],
        [min(x[0] for x in aoff), max(x[0] for x in aoff)],
        sum(x[2] < 0.05 for x in aoff),
    )
    == ([10, 13], [5, 7], 2),
    "exact McNemar per cell",
    "TWO OF FOUR",
)
claim(
    "e2b->e4b aime on",
    "AIME thinking on: level, E4B 9 to 12 against E2B 6 to 10",
    {"e4b": [9, 12], "e2b": [6, 10]},
    f"{ART} Step 5",
    {
        "e4b": [min(x[1] for x in aon), max(x[1] for x in aon)],
        "e2b": [min(x[0] for x in aon), max(x[0] for x in aon)],
        "p_min": round(min(x[2] for x in aon), 4),
    },
    ([min(x[1] for x in aon), max(x[1] for x in aon)], [min(x[0] for x in aon), max(x[0] for x in aon)])
    == ([9, 12], [6, 10])
    and min(x[2] for x in aon) >= 0.05,
    "exact McNemar per cell",
    "WITHIN NOISE",
)
claim(
    "four tests by chance",
    "with four tests, one passing at 0.05 happens by chance alone 18.5 percent of the time",
    {"pct": 18.5},
    f"{ART} Findings",
    {"pct": round(100 * (1 - 0.95**4), 2)},
    round(100 * (1 - 0.95**4), 1) == 18.5,
    "1 - 0.95^4, independent tests assumed",
    "HOLDS",
)
try:
    _prev = json.load(open(f"{H}/stat_audit.json"))
except (OSError, ValueError):
    _prev = {}
json.dump(
    dict(run=RUN, claims=claims, mcp_crosscheck=_prev.get("mcp_crosscheck", {})),
    open(f"{H}/stat_audit.json", "w"),
    indent=1,
    default=lambda o: o.item(),
)
open(f"{H}/audit_stats.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
