"""Score the v6e-1 sweep: one row per model and task, from score.py's own functions.

    python3 sweep_score.py --prefix 2026-09-24-v6e1 --arms e2b e4b 12b [26b-fp8 ...]

Per arm and task: accuracy with a 95% bootstrap range, ECE (15 bins), Brier, the
pre-registered single-split ECE after 50 labels, the mean of 20 splits at 50 labels, the
mean fitted temperature at 50 labels, option-order changes, the share of reads with every
label inside the top 32 (deviation 2), and the latency pass. For E2B and E4B, the share of
examples whose predicted label matches the L4 run in ../jev with the same checkpoint and
prompts. Writes results/<prefix>-SWEEP.md and .json.
"""

import argparse
import json
import math
import os
import random

import score as sc

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ("sst2", "ag_news", "emotion", "irony")
L4 = {"e2b": "2026-09-23-l4-e2b", "e4b": "2026-09-23-l4-e4b"}


def boot_range(correct, seed=0, n=2000):
    rng = random.Random(seed)
    k = len(correct)
    draws = sorted(sum(correct[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return draws[int(0.025 * n)], draws[int(0.975 * n) - 1]


def at_bound(rec):
    """The record with its missing labels raised from the proxy's floor (lowest returned
    logprob minus 5) to the top-32 bound (the lowest returned logprob): the most probability
    they could have had. Missing labels are the smallest labels_returned-complement of the
    read's probabilities, which the floor made identical."""
    r = dict(rec)
    reads = []
    for rd in rec["reads"]:
        p = list(rd["probs"])
        k = len(p) - rd.get("labels_returned", len(p))
        if k > 0:
            for i in sorted(range(len(p)), key=lambda i: p[i])[:k]:
                p[i] *= math.e ** 5
            z = sum(p)
            p = [x / z for x in p]
        reads.append(dict(rd, probs=p))
    r["reads"] = reads
    return r


def load(path):
    return sc.load(path) if os.path.exists(path) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    args = ap.parse_args()
    out, md = {}, [f"# v6e-1 sweep, run `{args.prefix}`", "", "Computed by `sweep_score.py` with `score.py`'s functions. Ranges are 95% over 2000 resamples of examples; ECE over 15 bins.", ""]
    md += ["| Arm | Task | n | Accuracy (95% range) | Majority | ECE | Brier | ECE after 50 labels, single split | 20-split mean | Fitted T at 50 | Reversed-order changes | All labels in top 32 | ECE raw / 50 labels, missing labels at the top-32 bound | Median ms | p90 ms | Same label as L4 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in args.arms:
        run = os.path.join(HERE, "results", f"{args.prefix}-{arm}")
        lat_run = run + "-latency"
        for task in TASKS:
            recs = load(os.path.join(run, f"autoregressive-{task}.jsonl"))
            if not recs:
                continue
            ids = sorted(recs)
            rl = [recs[i] for i in ids]
            s, correct, *_ = sc.score(rl, "ar")
            rb = [at_bound(x) for x in rl]
            sb, *_ = sc.score(rb, "ar")
            multib = next(r for r in sc.label_curve_multi(rb, "ar") if r["n_labels"] == 50)
            lo, hi = boot_range(correct)
            single = next(r for r in sc.label_curve(rl, "ar") if r["n_labels"] == 50)
            multi = next(r for r in sc.label_curve_multi(rl, "ar") if r["n_labels"] == 50)
            rev = load(os.path.join(run, f"autoregressive-{task}--reversed.jsonl"))
            flips = sc.flip_rate(recs, rev, "ar") if rev else None
            lat = load(os.path.join(lat_run, f"autoregressive-{task}.jsonl"))
            lt = sc.latency(list(lat.values()), "ar") if lat else None
            agree = None
            if arm in L4:
                l4 = load(os.path.join(HERE, "..", "jev", "results", L4[arm], f"autoregressive-{task}.jsonl"))
                both = sorted(set(l4) & set(recs))
                if both:
                    agree = {"n": len(both), "same": sum(sc.predicted_name(recs[i], "ar") == sc.predicted_name(l4[i], "ar") for i in both)}
            row = {"n": s["n"], "accuracy": s["accuracy"], "range": [lo, hi], "majority": s["majority_accuracy"], "ece": s["ece"], "brier": s["brier"],
                   "ece_fit50_single": single["ece"], "ece_fit50_mean20": multi["ece"]["mean"], "t_fit50_mean20": multi["temperature"]["mean"],
                   "flips": flips, "all_labels_returned": s["all_labels_returned"],
                   "at_bound": {"accuracy": sb["accuracy"], "ece": sb["ece"], "ece_fit50_mean20": multib["ece"]["mean"]}, "latency": lt, "agree_l4": agree}
            out.setdefault(arm, {})[task] = row
            pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
            cells = [arm, task, str(s["n"]), f"{pct(s['accuracy'])} ({pct(lo)}–{pct(hi)})", pct(s["majority_accuracy"]),
                     f"{s['ece']:.3f}", f"{s['brier']:.3f}", f"{single['ece']:.3f}", f"{multi['ece']['mean']:.3f}",
                     f"{multi['temperature']['mean']:.2f}", f"{flips['flips']} of {flips['n']}" if flips else "—",
                     pct(s["all_labels_returned"]), f"{sb['ece']:.3f} / {multib['ece']['mean']:.3f}", f"{lt['median_ms']:.0f}" if lt else "—", f"{lt['p90_ms']:.0f}" if lt else "—",
                     f"{agree['same']} of {agree['n']}" if agree else "—"]
            md.append("| " + " | ".join(cells) + " |")
    json.dump(out, open(os.path.join(HERE, "results", f"{args.prefix}-SWEEP.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", f"{args.prefix}-SWEEP.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
