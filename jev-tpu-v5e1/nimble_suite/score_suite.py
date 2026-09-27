"""Score run_suite.py records against the human labels, beside Bespoke Labs' published Jev and Nimble numbers.

    python3 nimble_suite/score_suite.py --run RUN [--run RUN2 ...]

Definitions follow Bespoke Labs' public-suite report so the columns are comparable:
accuracy is the top option equal to the reference; ECE is equal-width over 10
bins on the top option's probability; Brier is the multiclass sum of squared
errors against the one-hot reference, averaged per record. The same functions
are checked against Bespoke Labs' own implementation in tests/test_suite_scoring.py.

Added beside the published columns: ECE on a held-out half after one
temperature fitted on 50 labels, mean over 20 random splits (seed 20260923).
Writes results/RUN/SUITE.md and suite.json.
"""

import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINS = 10
T_GRID = [round(0.25 * 1.05 ** i, 4) for i in range(0, 72)]


def ece10(pairs, bins=BINS):
    n = len(pairs)
    err = 0.0
    for i in range(bins):
        idx = [j for j, (x, _) in enumerate(pairs) if i / bins <= x and (x < (i + 1) / bins or i == bins - 1 and x <= 1)]
        if idx:
            err += abs(sum(pairs[j][0] - pairs[j][1] for j in idx)) / n
    return err


def gold_key(rec):
    t = rec["target"]
    if rec["type"] == "noul":
        return "true" if t else "false"
    return str(t)


def dist(rec, kind):
    reads = rec["reads"]
    if kind in ("ar", "dg1"):
        p = reads[0]["probs"]
    else:
        p = [sum(r["probs"][i] for r in reads) / len(reads) for i in range(len(reads[0]["probs"]))]
    return dict(zip(rec["keys"], p))


def assess(p, gold):
    best = max(p, key=p.get)
    return {
        "correct": best == gold,
        "top": p[best],
        "brier": sum((v - (1.0 if k == gold else 0.0)) ** 2 for k, v in p.items()),
        "nll": -math.log(max(p[gold], 1e-15)),
    }


def temper(p, t):
    w = {k: max(v, 1e-12) ** (1.0 / t) for k, v in p.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def fit_t(ps, golds):
    return min(T_GRID, key=lambda t: sum(-math.log(max(temper(p, t)[g], 1e-15)) for p, g in zip(ps, golds)))


def fitted_ece(ps, golds, n=50, splits=20, seed=20260923):
    vals = []
    for k in range(splits):
        idx = list(range(len(ps)))
        random.Random(seed + k).shuffle(idx)
        half = len(idx) // 2
        fit, held = idx[:half][:n], idx[half:]
        t = fit_t([ps[i] for i in fit], [golds[i] for i in fit])
        pairs = []
        for i in held:
            q = temper(ps[i], t)
            b = max(q, key=q.get)
            pairs.append((q[b], b == golds[i]))
        vals.append(ece10(pairs))
    return sum(vals) / len(vals)


def score(recs, kind):
    rows = [assess(dist(r, kind), gold_key(r)) for r in recs]
    ps = [dist(r, kind) for r in recs]
    golds = [gold_key(r) for r in recs]
    return {
        "n": len(rows),
        "acc": sum(r["correct"] for r in rows) / len(rows),
        "ece": ece10([(r["top"], r["correct"]) for r in rows]),
        "brier": sum(r["brier"] for r in rows) / len(rows),
        "ece_fit50": fitted_ece(ps, golds),
        "label_mass": sorted(x["label_mass"] for r in recs for x in r["reads"])[len(recs) * len(recs[0]["reads"]) // 2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    args = ap.parse_args()
    pub = json.load(open(os.path.join(HERE, "nimble_suite", "published_jev_nimble.json")))["subsets"]
    arms = {}
    for run in args.run:
        d = os.path.join(HERE, "results", run)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".jsonl"):
                continue
            arm, sub = f[:-6].split("-", 1)
            recs = [json.loads(line) for line in open(os.path.join(d, f))]
            model = recs[0]["model"].split("/")[-1]
            kinds = ("ar",) if arm == "autoregressive" else ("dg1", "dg4")
            for k in kinds:
                arms.setdefault(f"{model} {k}", {})[sub] = score(recs, k)
    subs = sorted(pub)
    out = {"published": pub, "measured": arms}
    md = ["# Bespoke Labs' 13-subset public suite: Gemma 4 read by label probabilities", "",
          "Jev 1.13.0 and Nimble-9B columns are as published by Bespoke Labs on the same records; Gemma columns computed by `score_suite.py`. ECE: 10 equal-width bins; Brier: multiclass mean.", ""]
    names = sorted(arms)
    md.append("| Subset | type | n | Jev acc | Nimble acc | " + " | ".join(f"{a} acc" for a in names) + " |")
    md.append("|---|---|---|---|---|" + "---|" * len(names))
    for s in subs:
        md.append(f"| {s} | {pub[s]['type']} | {pub[s]['n']} | {pub[s]['jev_acc']:.3f} | {pub[s]['nimble_acc']:.3f} | " +
                  " | ".join(f"{arms[a][s]['acc']:.3f}" if s in arms[a] else "—" for a in names) + " |")
    md += ["", "| Subset | Jev ECE | Nimble ECE | " + " | ".join(f"{a} ECE (after 50 labels)" for a in names) + " |",
           "|---|---|---|" + "---|" * len(names)]
    for s in subs:
        md.append(f"| {s} | {pub[s].get('jev_ece', float('nan')):.3f} | {pub[s].get('nimble_ece', float('nan')):.3f} | " +
                  " | ".join(f"{arms[a][s]['ece']:.3f} ({arms[a][s]['ece_fit50']:.3f})" if s in arms[a] else "—" for a in names) + " |")
    md += ["", "Pooled and macro accuracy, by question type:", "", "| Group | Jev micro | Jev macro | " + " | ".join(f"{a} micro | {a} macro" for a in names) + " |",
           "|---|---|---|" + "---|---|" * len(names)]
    for g in ("all", "noul", "choice", "score"):
        ss_ = [s for s in subs if g == "all" or pub[s]["type"] == g]
        def micro(get_acc, get_n):
            return sum(get_acc(s) * get_n(s) for s in ss_) / sum(get_n(s) for s in ss_)
        def macro(get_acc):
            return sum(get_acc(s) for s in ss_) / len(ss_)
        cells = [f"{micro(lambda s: pub[s]['jev_acc'], lambda s: pub[s]['n']):.3f}", f"{macro(lambda s: pub[s]['jev_acc']):.3f}"]
        for a in names:
            if all(s in arms[a] for s in ss_):
                cells += [f"{micro(lambda s: arms[a][s]['acc'], lambda s: arms[a][s]['n']):.3f}", f"{macro(lambda s: arms[a][s]['acc']):.3f}"]
            else:
                cells += ["—", "—"]
        md.append(f"| {g} | " + " | ".join(cells) + " |")
    target = os.path.join(HERE, "results", args.run[0])
    json.dump(out, open(os.path.join(target, "suite.json"), "w"), indent=1)
    open(os.path.join(target, "SUITE.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
