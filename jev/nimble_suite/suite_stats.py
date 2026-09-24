"""The figures the measurement article quotes from the public-suite run, computed rather than read off tables.

    python3 nimble_suite/suite_stats.py --run 2026-09-24-l4-suite --run 2026-09-24-l4-suite-e4b

Pooled accuracy by question type with Wilson 95% intervals (Bespoke Labs' interval);
plain Gemma against DiffusionGemma paired per record with an exact McNemar test;
median ECE over the 13 subsets raw and after 50 labels; how many subsets each arm's
fitted ECE is at or below Jev's published raw ECE; DiffusionGemma's median label mass.
Writes results/RUN/STATS.md and stats.json into the first --run.
"""

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_suite as sc  # noqa: E402

HERE = sc.HERE
GROUPS = ("all", "noul", "choice", "score")


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(runs):
    arms = {}
    for run in runs:
        d = os.path.join(HERE, "results", run)
        for f in sorted(os.listdir(d)):
            if f.endswith(".jsonl"):
                for line in open(os.path.join(d, f)):
                    r = json.loads(line)
                    arms.setdefault(r["model"].split("/")[-1], []).append(r)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    args = ap.parse_args()
    pub = json.load(open(os.path.join(HERE, "nimble_suite", "published_jev_nimble.json")))["subsets"]
    measured = json.load(open(os.path.join(HERE, "results", args.run[0], "suite.json")))["measured"]
    recs = load(args.run)
    readout = {m: ("dg4" if "diffusion" in m else "ar") for m in recs}
    correct = {m: {r["id"] + "|" + r["subset"]: sc.assess(sc.dist(r, readout[m]), sc.gold_key(r))["correct"] for r in rs} for m, rs in recs.items()}
    rtype = {r["id"] + "|" + r["subset"]: r["type"] for rs in recs.values() for r in rs}
    out = {"readout": readout, "accuracy": {}, "paired": {}, "ece": {}, "label_mass": {}}

    for g in GROUPS:
        subs = [s for s in pub if g == "all" or pub[s]["type"] == g]
        n = sum(pub[s]["n"] for s in subs)
        jk = round(sum(pub[s]["jev_acc"] * pub[s]["n"] for s in subs))
        row = {"n": n, "jev": [jk / n, *wilson(jk, n)]}
        for m, c in correct.items():
            ks = [k for k in c if g == "all" or rtype[k] == g]
            k = sum(c[x] for x in ks)
            row[m] = [k / len(ks), *wilson(k, len(ks))]
        out["accuracy"][g] = row

    plain, diff = "gemma-4-26B-A4B-it-AWQ-4bit", "diffusiongemma-26B-A4B-it-AWQ-INT4"
    for g in GROUPS:
        ks = [k for k in correct[plain] if g == "all" or rtype[k] == g]
        b = sum(correct[plain][k] and not correct[diff][k] for k in ks)
        c = sum(correct[diff][k] and not correct[plain][k] for k in ks)
        out["paired"][g] = {"plain_only": b, "diffusion_only": c, "p": mcnemar_exact(b, c)}

    out["ece"]["jev"] = {"raw_median": statistics.median(pub[s]["jev_ece"] for s in pub)}
    out["ece"]["nimble"] = {"raw_median": statistics.median(pub[s]["nimble_ece"] for s in pub)}
    for m in recs:
        a = measured[f"{m} {readout[m]}"]
        out["ece"][m] = {
            "raw_median": statistics.median(a[s]["ece"] for s in pub),
            "fit50_median": statistics.median(a[s]["ece_fit50"] for s in pub),
            "fit50_at_or_below_jev": sum(a[s]["ece_fit50"] <= pub[s]["jev_ece"] for s in pub),
            "raw_at_or_below_jev": sum(a[s]["ece"] <= pub[s]["jev_ece"] for s in pub),
        }
    out["label_mass"][diff] = statistics.median(x["label_mass"] for r in recs[diff] for x in r["reads"])
    out["label_mass"][plain] = statistics.median(x["label_mass"] for r in recs[plain] for x in r["reads"])

    md = ["# Public-suite figures for the article", "", f"Readouts: {readout}", "",
          "| Group | n | Jev (published) | " + " | ".join(recs) + " |", "|---|---|---|" + "---|" * len(recs)]
    for g in GROUPS:
        row = out["accuracy"][g]
        fmt = lambda v: f"{100 * v[0]:.1f}% ({100 * v[1]:.1f}–{100 * v[2]:.1f})"  # noqa: E731
        md.append(f"| {g} | {row['n']} | {fmt(row['jev'])} | " + " | ".join(fmt(row[m]) for m in recs) + " |")
    md += ["", "Plain vs DiffusionGemma, paired per record:", ""]
    md += [f"- {g}: plain only right {v['plain_only']}, diffusion only right {v['diffusion_only']}, exact McNemar p = {v['p']:.4f}" for g, v in out["paired"].items()]
    md += ["", "ECE, median over 13 subsets:", ""]
    md += [f"- {m}: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()})}" for m, e in out["ece"].items()]
    md += ["", "Median label mass per read: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in out["label_mass"].items())]
    gaps = {s: measured[f"{plain} ar"][s]["acc"] - pub[s]["jev_acc"] for s in pub if pub[s]["type"] == "choice"}
    worst = min(gaps, key=gaps.get)
    out["largest_choice_gap"] = {"subset": worst, "jev": pub[worst]["jev_acc"], "plain": measured[f"{plain} ar"][worst]["acc"]}
    md += ["", f"Largest multiple-choice gap, plain Gemma against Jev: {worst}, Jev {100 * pub[worst]['jev_acc']:.1f}% against {100 * measured[f'{plain} ar'][worst]['acc']:.1f}%"]
    target = os.path.join(HERE, "results", args.run[0])
    json.dump(out, open(os.path.join(target, "stats.json"), "w"), indent=1)
    open(os.path.join(target, "STATS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
