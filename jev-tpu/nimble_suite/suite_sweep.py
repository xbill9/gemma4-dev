"""Public-suite figures for every arm of the v6e-1 sweep, beside Jev 1.13.0's published results.

    python3 nimble_suite/suite_sweep.py --prefix 2026-09-24-v6e1 --arms e2b e4b 12b 26b-fp8

Uses score_suite.py's definitions (Bespoke Labs' accuracy, 10-bin ECE, multiclass Brier, and
ECE after one temperature fitted on 50 labels, mean of 20 splits) and suite_stats.py's Wilson
interval. Per arm: pooled accuracy by question type with its 95% range, Jev minus the arm with
an unpaired 95% range (Jev's per-record answers are unpublished), median ECE raw and after 50
labels, median Brier, and how many subsets the fitted arm is at or below Jev's as-shipped ECE.
Writes results/<prefix>-SUITE.md and .json.
"""

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_suite as sc  # noqa: E402
from suite_stats import wilson  # noqa: E402

HERE = sc.HERE
GROUPS = ("all", "noul", "choice", "score")
Z = 1.959964


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    args = ap.parse_args()
    pub = json.load(open(os.path.join(HERE, "nimble_suite", "published_jev_nimble.json")))["subsets"]
    out = {}
    md = [f"# Public suite, v6e-1 sweep `{args.prefix}`", "",
          "Jev 1.13.0 and Nimble-9B are Bespoke Labs' published results on the same 3,880 records. Ranges: Wilson 95% for accuracy; Jev minus arm is unpaired. ECE: 10 bins.", "",
          "| Arm | Questions | Records | Accuracy (95% range) | Jev (published) | Jev minus arm, points (95% range) |", "|---|---|---|---|---|---|"]
    for arm in args.arms:
        d = os.path.join(HERE, "results", f"{args.prefix}-{arm}-suite")
        recs = [json.loads(line) for f in sorted(os.listdir(d)) if f.endswith(".jsonl") for line in open(os.path.join(d, f))]
        if not recs:
            continue
        per_sub = {}
        for r in recs:
            per_sub.setdefault(r["subset"], []).append(r)
        subs = {s: sc.score(rs, "ar") for s, rs in per_sub.items()}
        row = {"model": recs[0]["model"], "groups": {}, "subsets": subs}
        for g in GROUPS:
            rs = [r for r in recs if g == "all" or r["type"] == g]
            k = sum(sc.assess(sc.dist(r, "ar"), sc.gold_key(r))["correct"] for r in rs)
            n = len(rs)
            pa = k / n
            lo, hi = wilson(k, n)
            ss = [s for s in pub if g == "all" or pub[s]["type"] == g]
            nj = sum(pub[s]["n"] for s in ss)
            pj = sum(round(pub[s]["jev_acc"] * pub[s]["n"]) for s in ss) / nj
            se = math.sqrt(pj * (1 - pj) / nj + pa * (1 - pa) / n)
            dd = pj - pa
            row["groups"][g] = {"n": n, "acc": pa, "range": [lo, hi], "jev": pj, "jev_minus": [dd, dd - Z * se, dd + Z * se]}
            md.append(f"| {arm} | {g} | {n} | {100 * pa:.1f}% ({100 * lo:.1f}–{100 * hi:.1f}) | {100 * pj:.1f}% | {100 * dd:+.1f} ({100 * (dd - Z * se):+.1f} to {100 * (dd + Z * se):+.1f}) |")
        row["ece_raw_median"] = statistics.median(v["ece"] for v in subs.values())
        row["ece_fit50_median"] = statistics.median(v["ece_fit50"] for v in subs.values())
        row["brier_median"] = statistics.median(v["brier"] for v in subs.values())
        row["fit50_at_or_below_jev"] = sum(subs[s]["ece_fit50"] <= pub[s]["jev_ece"] for s in subs)
        row["raw_at_or_below_jev"] = sum(subs[s]["ece"] <= pub[s]["jev_ece"] for s in subs)
        out[arm] = row
    md += ["", "Calibration, median over the 13 subsets (Jev 1.13.0 published: ECE %.3f, Brier %.3f):" % (statistics.median(pub[s]["jev_ece"] for s in pub), statistics.median(pub[s]["jev_brier"] for s in pub)), "",
           "| Arm | ECE as shipped | ECE after 50 labels | Brier as shipped | Subsets at or below Jev, raw / after 50 labels |", "|---|---|---|---|---|"]
    for arm, row in out.items():
        md.append(f"| {arm} | {row['ece_raw_median']:.3f} | {row['ece_fit50_median']:.3f} | {row['brier_median']:.3f} | {row['raw_at_or_below_jev']} / {row['fit50_at_or_below_jev']} of 13 |")
    json.dump(out, open(os.path.join(HERE, "results", f"{args.prefix}-SUITE.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", f"{args.prefix}-SUITE.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
