"""Figures the measurement article derives from other committed results, computed here so none is done by hand.

    python3 derived_figures.py > results/DERIVED.md
"""

import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
Z = 1.959964


def rows(path, pat):
    return [tuple(float(x) for x in m.groups()) for m in re.finditer(pat, open(path).read())]


def main():
    out = ["# Derived figures for the measurement article (computed by derived_figures.py)", ""]

    # E4B against the 26B on the four tasks, from SMALL-MODELS.md
    acc = {}
    for m in re.finditer(r"\| (26B AWQ|E4B bf16) \| (\w+) \| ([\d.]+)%", open(os.path.join(R, "SMALL-MODELS.md")).read()):
        acc.setdefault(m.group(2), {})[m.group(1)] = float(m.group(3))
    gaps = {t: round(v["26B AWQ"] - v["E4B bf16"], 1) for t, v in acc.items()}
    out.append(f"- 26B minus E4B, four tasks, points: {gaps}; range {min(gaps.values())} to {max(gaps.values())}")

    # Jev's price per decision at this run's prompt lengths
    tok = open(os.path.join(R, "2026-09-23-l4-latency", "evidence", "prompt-tokens.txt")).read()
    med, lo, hi = (int(x) for x in re.search(r"all tasks: median (\d+), range (\d+)-(\d+)", tok).groups())
    price = 0.042
    out.append(f"- Jev at ${price} per million input tokens: ${price * med:.2f} per million decisions at the median {med} tokens, ${price * hi:.2f} at the longest {hi}")

    # Speed ratios from the latency table
    lat = rows(os.path.join(R, "2026-09-23-l4-latency", "LATENCY.md"), r"\| \w+ \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|")
    one = [r[2] / r[0] for r in lat]
    four = [r[4] / r[0] for r in lat]
    reread = [r[4] / r[2] for r in lat]
    out.append(f"- Plain against DiffusionGemma, time per decision: one read {min(one):.1f} to {max(one):.1f} times plain, four reads {min(four):.1f} to {max(four):.1f} times plain")
    out.append(f"- Four reads against one read, this run: {min(reread):.1f} to {max(reread):.1f} times")

    # Mastracci's charts
    mm = open(os.path.join(R, "mastracci-charts.txt")).read().split("Task latency")[1]
    ml = [tuple(int(x) for x in m.groups()) for m in re.finditer(r"^[^\n#(]*?[a-z)] (\d+) (\d+) (\d+)$", mm, re.M)]
    ratio = [a / o for o, a, _ in ml]
    out.append(f"- Mastracci, automatic re-reads against one read over his {len(ml)} sets: {min(ratio):.1f} to {max(ratio):.1f} times")

    # Yes/no subsets, Jev minus plain, unpaired 95% ranges
    pub = json.load(open(os.path.join(HERE, "nimble_suite", "published_jev_nimble.json")))["subsets"]
    meas = json.load(open(os.path.join(R, "2026-09-24-l4-suite", "suite.json")))["measured"]["gemma-4-26B-A4B-it-AWQ-4bit ar"]
    for s in sorted(pub):
        if pub[s]["type"] != "noul":
            continue
        n, pj, pm = pub[s]["n"], pub[s]["jev_acc"], meas[s]["acc"]
        d = pj - pm
        se = math.sqrt(pj * (1 - pj) / n + pm * (1 - pm) / n)
        out.append(f"- {s}: Jev minus plain {100 * d:+.1f} points, 95% range {100 * (d - Z * se):+.1f} to {100 * (d + Z * se):+.1f}")

    # Sign test: DiffusionGemma's fitted ECE below plain's on k of 13 subsets
    dg = json.load(open(os.path.join(R, "2026-09-24-l4-suite", "suite.json")))["measured"]["diffusiongemma-26B-A4B-it-AWQ-INT4 dg4"]
    k = sum(dg[s]["ece_fit50"] < meas[s]["ece_fit50"] for s in pub)
    n = len(pub)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)
    out.append(f"- After 50 labels, DiffusionGemma's ECE below plain's on {k} of {n} subsets; two-sided sign test p = {p:.2f}")

    # Costs of the three runs
    out.append(f"- All three runs: ${0.97 + 1.57 + 0.66:.2f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
