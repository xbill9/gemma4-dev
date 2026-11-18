#!/usr/bin/env python3
"""Statistical audit of the E4B thinking article, recomputed from the raw replies (not from
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


ART = "devto-gemma4-e4b-thinking-3050.md"
# 1. scores and intervals
paper = {
    ("gsm8k", "repack-none", "off"): 0.9295,
    ("gsm8k", "repack-none", "on"): 0.9325,
    ("gsm8k", "repack-q4x", "off"): 0.9272,
    ("gsm8k", "repack-q4x", "on"): 0.9371,
    ("gsm8k", "google-none", "off"): 0.9348,
    ("gsm8k", "google-none", "on"): 0.9363,
    ("gsm8k", "google-q4x", "off"): 0.9318,
    ("gsm8k", "google-q4x", "on"): 0.9439,
}
ci = {}
for k_, v in paper.items():
    r = D[k_]
    k = sum(x["right"] for x in r.values())
    n = len(r)
    ci[k_] = wilson(k, n)
    claim(
        f"score {' '.join(k_)}",
        "GSM8K score",
        {"rate": v},
        f"{ART} Step 3",
        {"rate": round(k / n, 4), "k": k, "n": n, "ci_pct": [round(c, 2) for c in ci[k_]]},
        round(k / n, 4) == v,
        "k/n from the saved replies",
        "HOLDS",
    )
ov = all(ci[a][0] <= ci[b][1] and ci[b][0] <= ci[a][1] for a, b in itertools.combinations(ci, 2))
claim(
    "gsm8k intervals overlap",
    "every 95% range overlaps every other",
    {"overlap": True},
    f"{ART} Step 3",
    {"overlap": ov},
    ov,
    "pairwise Wilson overlap",
    "HOLDS",
)
aime = {"repack-none": (12, 10), "repack-q4x": (10, 12), "google-none": (12, 12), "google-q4x": (13, 9)}
for a, (o, n_) in aime.items():
    ko = sum(x["right"] for x in D[("aime2025", a, "off")].values())
    kn = sum(x["right"] for x in D[("aime2025", a, "on")].values())
    claim(
        f"aime right {a}",
        "AIME right answers off/on",
        {"off": o, "on": n_},
        f"{ART} Step 3",
        {
            "off": ko,
            "on": kn,
            "ci_off_pct": [round(c, 1) for c in wilson(ko, 30)],
            "ci_on_pct": [round(c, 1) for c in wilson(kn, 30)],
        },
        (ko, kn) == (o, n_),
        "count of right replies",
        "HOLDS",
    )
# 2. paired tests in the article
tab = [
    ("thinking repack-none", ("gsm8k", "repack-none", "off"), ("gsm8k", "repack-none", "on"), 25, 29, 0.6835),
    ("thinking repack-q4x", ("gsm8k", "repack-q4x", "off"), ("gsm8k", "repack-q4x", "on"), 23, 36, 0.1175),
    ("thinking google-none", ("gsm8k", "google-none", "off"), ("gsm8k", "google-none", "on"), 26, 28, 0.8919),
    ("thinking google-q4x", ("gsm8k", "google-q4x", "off"), ("gsm8k", "google-q4x", "on"), 20, 36, 0.0440),
    ("drafter largest", ("gsm8k", "google-none", "on"), ("gsm8k", "google-q4x", "on"), 10, 20, 0.0987),
    ("weights largest", ("gsm8k", "repack-q4x", "on"), ("gsm8k", "google-q4x", "on"), 15, 24, 0.1996),
]
for id_, x, y, b0, c0, p0 in tab:
    b, c, p = mc(x, y)
    claim(
        f"mcnemar {id_}",
        f"GSM8K paired test {id_}",
        {"only_before": b0, "only_after": c0, "p": p0},
        f"{ART} Step 6",
        {"only_before": b, "only_after": c, "p": round(p, 4)},
        (b, c, round(p, 4)) == (b0, c0, p0),
        "exact McNemar (binomial on discordant pairs)",
        "WITHIN NOISE" if p0 >= 0.05 else "SIGNIFICANT",
    )
# all 24 tests: smallest p per family, AIME floor
fam = {}
for t in ("gsm8k", "aime2025"):
    for a in ARMS:
        fam.setdefault(f"{t} thinking", []).append(mc((t, a, "off"), (t, a, "on")))
    for w in ("repack", "google"):
        for th in ("off", "on"):
            fam.setdefault(f"{t} drafter", []).append(mc((t, f"{w}-none", th), (t, f"{w}-q4x", th)))
    for d in ("none", "q4x"):
        for th in ("off", "on"):
            fam.setdefault(f"{t} weights", []).append(mc((t, f"repack-{d}", th), (t, f"google-{d}", th)))
minp = {k: round(min(p for *_, p in v), 4) for k, v in fam.items()}
claim(
    "gsm8k thinking p range",
    "one GSM8K thinking gain passes (p = 0.0440); the other three p = 0.1175 to 0.8919",
    {"p_min": 0.044, "p_max": 0.8919},
    f"{ART} Findings",
    {"p_min": minp["gsm8k thinking"], "p_max": round(max(p for *_, p in fam["gsm8k thinking"]), 4)},
    minp["gsm8k thinking"] == 0.044,
    "min over four tests",
    "ONE OF FOUR",
)
claim(
    "drafter no change",
    "drafter changed no score beyond chance in eight tests",
    {"p_min_ge": 0.05},
    f"{ART} Findings",
    {"p_min_gsm8k": minp["gsm8k drafter"], "p_min_aime": minp["aime2025 drafter"]},
    min(minp["gsm8k drafter"], minp["aime2025 drafter"]) >= 0.05,
    "smallest of 8 exact McNemar",
    "HOLDS",
)
claim(
    "weights level",
    "repack and Google scored level on every paired test",
    {"p_min_ge": 0.05},
    f"{ART} Findings",
    {"p_min_gsm8k": minp["gsm8k weights"], "p_min_aime": minp["aime2025 weights"]},
    min(minp["gsm8k weights"], minp["aime2025 weights"]) >= 0.05,
    "smallest of 8 exact McNemar",
    "HOLDS",
)
aimemin = min(minp[f"aime2025 {f}"] for f in ("thinking", "drafter", "weights"))
claim(
    "aime best p",
    "on AIME no comparison does better than p = 0.2891",
    {"p": 0.2891},
    f"{ART} Step 6",
    {"p": aimemin},
    aimemin == 0.2891,
    "smallest of 12 AIME tests",
    "WITHIN NOISE",
)
disc = [b + c for b, c, _ in fam["gsm8k thinking"]]
claim(
    "thinking discordant",
    "thinking changed the answer on 54 to 59 GSM8K problems",
    {"min": 54, "max": 59},
    f"{ART} Step 6",
    {"min": min(disc), "max": max(disc)},
    (min(disc), max(disc)) == (54, 59),
    "discordant pairs",
    "HOLDS",
)
# 3. tokens: Wilcoxon
ps, more = [], []
for t in ("gsm8k", "aime2025"):
    for a in ARMS:
        off, on = D[(t, a, "off")], D[(t, a, "on")]
        ids = sorted(off)
        ps.append(
            wilcoxon([on[i]["completion_tokens"] for i in ids], [off[i]["completion_tokens"] for i in ids]).pvalue
        )
        more.append((t, sum(on[i]["completion_tokens"] > off[i]["completion_tokens"] for i in ids)))
claim(
    "tokens wilcoxon",
    "thinking used more tokens, signed-rank p below 0.006 in all eight",
    {"p_max_lt": 0.006},
    f"{ART} Step 4",
    {"p_max": float(f"{max(ps):.3g}")},
    max(ps) < 0.006,
    "Wilcoxon signed-rank on completion_tokens",
    "HOLDS",
)
g = [m for t, m in more if t == "gsm8k"]
am = [m for t, m in more if t == "aime2025"]
claim(
    "tokens more count",
    "more tokens on 1,309 to 1,313 of 1,319 GSM8K and 23 to 26 of 30 AIME",
    {"gsm8k": [1309, 1313], "aime": [23, 26]},
    f"{ART} Step 4",
    {"gsm8k": [min(g), max(g)], "aime": sorted(set(am))},
    [min(g), max(g)] == [1309, 1313] and [min(am), max(am)] == [23, 26],
    "per-problem comparison",
    "HOLDS",
)


# 4. noise floor
def mdl(n, p, z=norm.ppf(0.975) + norm.ppf(0.8)):
    return 100 * z * math.sqrt(2 * p * (1 - p) / n)


claim(
    "aime small",
    "AIME's 30 problems can only show large effects",
    {"qualitative": True},
    f"{ART} Findings",
    {"min_detectable_lift_two_runs_pts_n30": round(mdl(30, 0.4), 2), "same_n1319": round(mdl(1319, 0.93), 2)},
    True,
    "minimum lift two runs can detect at 80% power",
    "HOLDS",
)
claim("four tests by chance", "with four tests, one passing at 0.05 happens by chance alone 18.5 percent of the time", {"pct": 18.5}, f"{ART} Findings",
      {"pct": round(100 * (1 - 0.95 ** 4), 2)}, round(100 * (1 - 0.95 ** 4), 1) == 18.5, "1 - 0.95^4, independent tests assumed", "HOLDS")
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
