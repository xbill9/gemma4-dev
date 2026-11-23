#!/usr/bin/env python3
"""Statistical audit of devto-gemma4-e2b-e4b-12b-3050.md, recomputed from per-problem rows (not rollup.txt).

Wilson 95% interval for k/n; exact two-sided McNemar on paired problems (b = right only for the first,
c = right only for the second). Writes stat_audit.json (claims read by check-facts.py) and audit_stats.txt.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUN = {
    "E2B": "local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem",
    "E4B": "local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem",
    "12B": "local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-17-thinking-12b-3050oem",
}
OCT12 = "local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-accuracy-12b-3050oem"
LOOPS = "local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-20-thinking-loops-12b-3050oem"


def load(rel):
    p = os.path.join(R, rel)
    if not os.path.exists(p) and os.path.exists(p + ".partial"):
        p += ".partial"
    return {r["id"]: r for r in map(json.loads, open(p)) if "id" in r}


def run(m, task, drafter, think):
    return load(f"{RUN[m]}/{task}-repack-{drafter}-think{think}.jsonl")


def ok(r):
    return r.get("right") is True


def wilson(k, n, z=1.959964):
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def pair(A, B, ids=None):
    ids = sorted(ids if ids is not None else set(A) & set(B))
    b = sum(ok(A[i]) and not ok(B[i]) for i in ids)
    c = sum(ok(B[i]) and not ok(A[i]) for i in ids)
    return b, c, len(ids)


claims, lines = [], []


def rec(cid, stmt, src, paper, computed, verdict, conclusion, test, reason=""):
    claims.append(
        {
            "id": cid,
            "statement": stmt,
            "paper_value": paper,
            "paper_source": src,
            "computed": computed,
            "verdict": verdict,
            "conclusion": conclusion,
            "test": test,
            "reason": reason,
        }
    )
    lines.append(f"| {stmt} | {paper} | {computed} | {test} | {conclusion} |")


# 1. GSM8K thinking-off scores and 95% ranges (Step 2 table)
for m, k_, sc, lo_, hi_ in [
    ("E2B", 1189, 0.9014, 0.884, 0.916),
    ("E4B", 1226, 0.9295, 0.914, 0.942),
    ("12B", 1269, 0.9621, 0.950, 0.971),
]:
    rows = run(m, "gsm8k", "none", "off")
    k, n = sum(ok(r) for r in rows.values()), len(rows)
    lo, hi = wilson(k, n)
    good = k == k_ and n == 1319 and round(k / n, 4) == sc and round(lo, 3) == lo_ and round(hi, 3) == hi_
    rec(
        f"gsm8k-{m}",
        f"{m} GSM8K thinking off {k_}/1319 = {sc}, 95% {lo_} to {hi_}",
        "Step 2 table",
        {"k": k_, "score": sc, "ci": [lo_, hi_]},
        {"k": k, "n": n, "score": round(k / n, 4), "ci": [round(lo, 4), round(hi, 4)]},
        "VERIFIED" if good else "MISMATCH",
        "HOLDS" if good else "OVERSTATED",
        "Wilson 95%",
    )

# 2. AIME thinking-off counts
for m, k_ in [("E2B", 7), ("E4B", 12), ("12B", 15)]:
    rows = run(m, "aime2025", "none", "off")
    k, n = sum(ok(r) for r in rows.values()), len(rows)
    lo, hi = wilson(k, n)
    rec(
        f"aime-{m}",
        f"{m} AIME thinking off {k_} of 30",
        "Step 2 table",
        {"k": k_, "n": 30},
        {"k": k, "n": n, "ci": [round(lo, 4), round(hi, 4)]},
        "VERIFIED" if (k, n) == (k_, 30) else "MISMATCH",
        "HOLDS" if (k, n) == (k_, 30) else "OVERSTATED",
        "count + Wilson 95%",
    )

# 3. Size steps (Step 3 table): paper gives b (smaller only), c (bigger only), p
steps = [
    ("gsm8k", "E2B", "E4B", 34, 71, 0.00039, "holds"),
    ("gsm8k", "E4B", "12B", 17, 60, 8.91e-7, "holds"),
    ("aime2025", "E2B", "E4B", 0, 5, 0.0625, "ns"),
    ("aime2025", "E4B", "12B", 2, 5, 0.453, "ns"),
    ("aime2025", "E2B", "12B", 0, 8, 0.0078, "holds"),
]
for task, a, b_, pb, pc, pp, claim in steps:
    b, c, n = pair(run(a, task, "none", "off"), run(b_, task, "none", "off"))
    p = mcnemar(b, c)
    rep = (b, c) == (pb, pc) and abs(p - pp) / pp < 0.01
    concl = (p < 0.05) if claim == "holds" else (p >= 0.05)
    rec(
        f"step-{task}-{a}-{b_}",
        f"{task} {a} to {b_}: {pb} / {pc}, p = {pp} ({'beyond' if claim == 'holds' else 'within'} chance)",
        "Step 3 table",
        {"b": pb, "c": pc, "p": pp},
        {"b": b, "c": c, "n": n, "p": float(f"{p:.4g}")},
        "VERIFIED" if rep else "MISMATCH",
        "HOLDS" if concl else "OVERSTATED",
        "exact McNemar",
    )

# 4. Thinking off -> on (Step 4 table); paper gives totals and p
for m, task, k_off, k_on, pp, claim, ids in [
    ("E2B", "gsm8k", 1189, 1178, 0.294, "ns", None),
    ("E4B", "gsm8k", 1226, 1230, 0.683, "ns", None),
    ("12B", "gsm8k", 152, 87, 5.42e-20, "holds", "on"),
    ("E2B", "aime2025", 7, 10, 0.375, "ns", None),
    ("E4B", "aime2025", 12, 10, 0.688, "ns", None),
    ("12B", "aime2025", 15, 10, 0.227, "ns", None),
]:
    OFF, ON = run(m, task, "none", "off"), run(m, task, "none", "on")
    use = set(ON) if ids == "on" else set(OFF) & set(ON)
    b, c, n = pair(OFF, ON, use)
    p = mcnemar(b, c)
    koff, kon = sum(ok(OFF[i]) for i in use), sum(ok(ON[i]) for i in use)
    rep = (koff, kon) == (k_off, k_on) and abs(p - pp) / pp < 0.01
    concl = (p < 0.05) if claim == "holds" else (p >= 0.05)
    rec(
        f"think-{m}-{task}",
        f"{m} {task} thinking off {k_off} to on {k_on}, p = {pp}",
        "Step 4 table",
        {"off": k_off, "on": k_on, "p": pp},
        {"off": koff, "on": kon, "n": n, "off_only": b, "on_only": c, "p": float(f"{p:.4g}")},
        "VERIFIED" if rep else "MISMATCH",
        "HOLDS" if concl else "OVERSTATED",
        "exact McNemar",
    )

# 5. First 157: E4B 148, E2B 142 with thinking on; 12B lost 65 and gained none
ids157 = set(run("12B", "gsm8k", "none", "on"))
for m, k_ in [("E4B", 148), ("E2B", 142)]:
    k = sum(ok(run(m, "gsm8k", "none", "on")[i]) for i in ids157)
    rec(
        f"first157-{m}",
        f"{m} thinking on, first 157 GSM8K: {k_}",
        "Step 4",
        {"k": k_},
        {"k": k, "n": len(ids157)},
        "VERIFIED" if k == k_ else "MISMATCH",
        "HOLDS" if k == k_ else "OVERSTATED",
        "count",
    )

# 6. Drafter: no score change beyond chance (Step 6 table), thinking off and on, GSM8K and AIME, repack
for m in ("E2B", "E4B"):
    for task in ("gsm8k", "aime2025"):
        for think in ("off", "on"):
            b, c, n = pair(run(m, task, "q4x", think), run(m, task, "none", think))
            p = mcnemar(b, c)
            rec(
                f"drafter-{m}-{task}-{think}",
                f"{m} drafter changes no {task} score (thinking {think})",
                "Step 6 table",
                {"claim": "none beyond chance"},
                {"drafter_only": b, "none_only": c, "n": n, "p": float(f"{p:.4g}")},
                "VERIFIED" if p >= 0.05 else "MISMATCH",
                "HOLDS" if p >= 0.05 else "OVERSTATED",
                "exact McNemar",
            )
b, c, n = pair(load(f"{OCT12}/gsm8k-q4x.jsonl"), load(f"{OCT12}/gsm8k-none.jsonl"))
p = mcnemar(b, c)
rec(
    "drafter-12B-gsm8k-oct",
    "12B drafter 0.9598 vs 0.9629 on GSM8K, no change beyond chance (October run)",
    "Step 6",
    {"claim": "none beyond chance"},
    {"drafter_only": b, "none_only": c, "n": n, "p": float(f"{p:.4g}")},
    "VERIFIED" if p >= 0.05 else "MISMATCH",
    "HOLDS" if p >= 0.05 else "OVERSTATED",
    "exact McNemar",
)

# 7. Loop test, 8 problems: greedy q4_0 0/8 finished, f16 0/8, sampled 8/8 finished and 7 right
G = load(f"{LOOPS}/google-q4kv.jsonl")
F = load(f"{LOOPS}/greedy-f16kv.jsonl")
fin = lambda rows: sum(r.get("finish") != "length" for r in rows.values())  # noqa: E731
kg, kf = sum(ok(r) for r in G.values()), sum(ok(r) for r in F.values())
rec(
    "loops-sampled",
    "Google's sampling: 8 of 8 finished, 7 right",
    "Step 5",
    {"finished": 8, "right": 7, "n": 8},
    {"finished": fin(G), "right": kg, "n": len(G), "right_ci": [round(x, 3) for x in wilson(kg, len(G))]},
    "VERIFIED" if (fin(G), kg, len(G)) == (8, 7, 8) else "MISMATCH",
    "HOLDS",
    "count + Wilson 95%",
    "8 problems: the gap from 0 of 8 is beyond chance (exact McNemar b=7, c=0, p=0.0156) but the rate is wide",
)
rec(
    "loops-f16",
    "f16 cache at temperature 0: 0 of 8 finished",
    "Step 5",
    {"finished": 0, "n": 8},
    {"finished": fin(F), "right": kf, "n": len(F)},
    "VERIFIED" if (fin(F), len(F)) == (0, 8) else "MISMATCH",
    "HOLDS",
    "count",
)
b, c, n = pair(G, F)
rec(
    "loops-sampled-vs-greedy",
    "sampling vs greedy on the 8 problems: 7 right vs 0",
    "Step 5",
    {"b": 7, "c": 0},
    {"b": b, "c": c, "n": n, "p": float(f"{mcnemar(b, c):.4g}")},
    "VERIFIED" if (b, c) == (7, 0) else "MISMATCH",
    "HOLDS" if mcnemar(b, c) < 0.05 else "WITHIN NOISE",
    "exact McNemar",
)

# 8. Google's sampling settings (Step 6): 2026-11-21 (12B thinking on) and 2026-11-22 runs, no drafter
S22 = "local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-22-sampling-e2b-e4b-12b-3050oem/"
S21 = "local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-21-drafter-sampling-12b-3050oem/"
sub = lambda d: {i: d[i] for i in ids157 if i in d}  # noqa: E731
SG = {
    "E2B-on": sub(load(S22 + "gsm8k-e2b-sampled-seed1-thinkon.jsonl")),
    "E4B-on": sub(load(S22 + "gsm8k-e4b-sampled-seed1-thinkon.jsonl")),
    "12B-on": sub(load(S21 + "gsm8k-sampled-none-thinkon.jsonl")),
    "12B-off": sub(load(S22 + "gsm8k-12b-sampled-seed1-thinkoff.jsonl")),
}
GR = {
    "E2B-off": sub(run("E2B", "gsm8k", "none", "off")),
    "E4B-off": sub(run("E4B", "gsm8k", "none", "off")),
    "12B-on": sub(run("12B", "gsm8k", "none", "on")),
    "12B-off": sub(run("12B", "gsm8k", "none", "off")),
}
SA = {
    "E2B-off": load(S22 + "aime2025-e2b-sampled-seed1-thinkoff.jsonl"),
    "E2B-on": load(S22 + "aime2025-e2b-sampled-seed1-thinkon.jsonl"),
    "E4B-off": load(S22 + "aime2025-e4b-sampled-seed1-thinkoff.jsonl"),
    "E4B-on": load(S22 + "aime2025-e4b-sampled-seed1-thinkon.jsonl"),
    "12B-off": load(S22 + "aime2025-12b-sampled-seed1-thinkoff.jsonl"),
    "12B-on": load(S21 + "aime2025-sampled-none-thinkon.jsonl"),
    "12B-on2": load(S22 + "aime2025-12b-sampled-seed2-thinkon.jsonl"),
}
for key, k_ in [("E2B-on", 140), ("E4B-on", 153), ("12B-on", 151), ("12B-off", 153)]:
    d = SG[key]
    k = sum(ok(r) for r in d.values())
    rec(f"sampled-gsm-{key}", f"Google's settings, GSM8K first 157, {key}: {k_}", "Step 6 table",
        {"k": k_}, {"k": k, "n": len(d), "ci": [round(x, 4) for x in wilson(k, len(d))]},
        "VERIFIED" if (k, len(d)) == (k_, 157) else "MISMATCH", "HOLDS" if k == k_ else "OVERSTATED", "count + Wilson 95%")
for key, k_ in [("E2B-off", 7), ("E2B-on", 10), ("E4B-off", 10), ("E4B-on", 15), ("12B-off", 19), ("12B-on", 21), ("12B-on2", 21)]:
    d = SA[key]
    k = sum(ok(r) for r in d.values())
    rec(f"sampled-aime-{key}", f"Google's settings, AIME {key}: {k_} of 30", "Step 6 table",
        {"k": k_}, {"k": k, "n": len(d)},
        "VERIFIED" if (k, len(d)) == (k_, 30) else "MISMATCH", "HOLDS" if k == k_ else "OVERSTATED", "count")
for cid, A, B, pb, pc, pp, claim in [
    ("E4B greedy off -> sampled on, GSM8K", GR["E4B-off"], SG["E4B-on"], 0, 7, 0.0156, "holds"),
    ("12B greedy on -> sampled on, GSM8K", GR["12B-on"], SG["12B-on"], 1, 65, 1.82e-18, "holds"),
    ("12B sampled off vs on, GSM8K", SG["12B-off"], SG["12B-on"], 4, 2, 0.688, "ns"),
    ("E4B sampled on vs 12B sampled off, GSM8K", SG["E4B-on"], SG["12B-off"], 1, 1, 1.0, "ns"),
    ("E2B greedy off -> sampled on, GSM8K", GR["E2B-off"], SG["E2B-on"], 9, 7, 0.804, "ns"),
    ("E2B AIME sampled off -> on", SA["E2B-off"], SA["E2B-on"], 0, 3, 0.25, "ns"),
    ("E4B AIME sampled off -> on", SA["E4B-off"], SA["E4B-on"], 1, 6, 0.125, "ns"),
    ("12B AIME sampled off -> on", SA["12B-off"], SA["12B-on"], 1, 3, 0.625, "ns"),
    ("12B AIME sampled off -> on, seed 2", SA["12B-off"], SA["12B-on2"], 2, 4, 0.688, "ns"),
    ("E4B vs 12B AIME, sampled off", SA["E4B-off"], SA["12B-off"], 0, 9, 0.0039, "holds"),
    ("E4B on vs 12B off AIME, sampled", SA["E4B-on"], SA["12B-off"], 1, 5, 0.219, "ns"),
    ("12B AIME greedy on -> sampled on", run("12B", "aime2025", "none", "on"), SA["12B-on"], 2, 13, 0.0074, "holds"),
]:
    b, c, n = pair(A, B)
    p = mcnemar(b, c)
    rep = (b, c) == (pb, pc) and abs(p - pp) / pp < 0.02
    concl = (p < 0.05) if claim == "holds" else (p >= 0.05)
    rec(f"sampled-{cid}", f"{cid}: {pb} / {pc}, p = {pp}", "Step 6",
        {"b": pb, "c": pc, "p": pp}, {"b": b, "c": c, "n": n, "p": float(f"{p:.4g}")},
        "VERIFIED" if rep else "MISMATCH", "HOLDS" if concl else "OVERSTATED", "exact McNemar")
cut = {m: sum(r.get("finish") == "length" for f in os.listdir(os.path.join(R, S22)) if f.endswith(".jsonl")
              and f"-{m.lower()}-" in f for r in load(S22 + f).values()) for m in ("E2B", "E4B", "12B")}
nrep = {m: sum(len(load(S22 + f)) for f in os.listdir(os.path.join(R, S22)) if f.endswith(".jsonl")
               and f"-{m.lower()}-" in f) for m in ("E2B", "E4B", "12B")}
for f in ("gsm8k-sampled-none-thinkon.jsonl", "aime2025-sampled-none-thinkon.jsonl"):
    cut["12B"] += sum(r.get("finish") == "length" for r in load(S21 + f).values())
    nrep["12B"] += len(load(S21 + f))
good = (cut, nrep) == ({"E2B": 0, "E4B": 0, "12B": 20}, {"E2B": 217, "E4B": 217, "12B": 404})
rec("sampled-cut", "Google's settings, replies at the limit: E2B 0/217, E4B 0/217, 12B 20/404", "Step 6",
    {"E2B": [0, 217], "E4B": [0, 217], "12B": [20, 404]}, {m: [cut[m], nrep[m]] for m in cut},
    "VERIFIED" if good else "MISMATCH", "HOLDS" if good else "OVERSTATED", "count")

json.dump(
    {"article": "docs/gemma4-e2b-e4b-12b-3050/devto-gemma4-e2b-e4b-12b-3050.md", "claims": claims},
    open(os.path.join(HERE, "stat_audit.json"), "w"),
    indent=1,
)
txt = "| Claim | Article value | Recomputed | Test | Verdict |\n|---|---|---|---|---|\n" + "\n".join(lines) + "\n"
open(os.path.join(HERE, "audit_stats.txt"), "w").write(txt)
print(txt)
print(
    {v: sum(c["conclusion"] == v for c in claims) for v in ("HOLDS", "WITHIN NOISE", "OVERSTATED")},
    "| verdict MISMATCH:",
    [c["id"] for c in claims if c["verdict"] == "MISMATCH"],
)
