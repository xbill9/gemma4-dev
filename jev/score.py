"""Score a run: accuracy, calibration and the escalation rule, per task and readout.

    python3 score.py --run RUN

Reads results/RUN/<arm>-<task>.jsonl and writes results/RUN/summary.json and
results/RUN/SUMMARY.md. Every number in them is computed here; nothing is
counted by hand.

Readouts compared on the same examples:
  ar      Gemma 4 26B, one left-to-right read
  dg1     DiffusionGemma, read 0 alone (one noise draw)
  dg4     DiffusionGemma, the mean of all reads
  dgauto  what structured_server's samples="auto" returns: read 0 when its
          entropy is at or under the threshold, otherwise the mean of all reads

Calibration is measured on the top answer: ECE bins the top probability into
equal-width bins and averages |accuracy - confidence| weighted by bin size.
Brier is the multi-class squared error against the one-hot gold label.
"""

import argparse
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO_THRESHOLD = 0.1  # structured_server's default auto_threshold
ECE_BINS = 15
BOOT = 2000


# ---------------------------------------------------------------------------
# Metrics. Pure functions over lists, so the tests can check them exactly.
# ---------------------------------------------------------------------------


def ece(conf, correct, bins=ECE_BINS):
    n = len(conf)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - avg)
    return total


def brier(probs, gold_idx):
    return sum(
        sum((p - (1.0 if k == g else 0.0)) ** 2 for k, p in enumerate(ps))
        for ps, g in zip(probs, gold_idx)
    ) / len(probs)


def nll(probs, gold_idx):
    return -sum(math.log(max(ps[g], 1e-12)) for ps, g in zip(probs, gold_idx)) / len(probs)


def auroc(scores, positive):
    """Probability that a random positive outranks a random negative (ties
    count half). None when either class is empty."""
    pos = [s for s, y in zip(scores, positive) if y]
    neg = [s for s, y in zip(scores, positive) if not y]
    if not pos or not neg:
        return None
    ranked = sorted((s, i) for i, s in enumerate(scores))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(r for r, y in zip(ranks, positive) if y)
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def avg_dist(reads):
    k = len(reads[0]["probs"])
    return [sum(r["probs"][i] for r in reads) / len(reads) for i in range(k)]


def readout(rec, kind):
    """(probs, gate_entropy, label_mass, argmax_is_label, all_labels_returned)
    for one record under one readout."""
    reads = rec["reads"]
    r0 = reads[0]
    ret = all(r.get("labels_returned") in (None, len(rec["labels"])) for r in reads)
    if kind in ("ar", "dg1"):
        return r0["probs"], r0["entropy"], r0["label_mass"], r0["argmax_is_label"], ret
    if kind == "dg4":
        return (
            avg_dist(reads),
            r0["entropy"],
            mean([r["label_mass"] for r in reads]),
            all(r["argmax_is_label"] for r in reads),
            ret,
        )
    if kind == "dgauto":
        if r0["entropy"] <= AUTO_THRESHOLD:
            return readout(rec, "dg1")
        return readout(rec, "dg4")
    raise ValueError(kind)


def spread(rec):
    """Standard error of the predicted label's probability across noise draws,
    the proxy's `stderr`. None for a single read."""
    reads = rec["reads"]
    n = len(reads)
    if n < 2:
        return None
    m = avg_dist(reads)
    top = max(range(len(m)), key=lambda i: m[i])
    var = sum((r["probs"][top] - m[top]) ** 2 for r in reads) / (n - 1)
    return (var / n) ** 0.5


def score(records, kind):
    gold_idx, probs, conf, correct, ent, mass, arg_ok, ret = [], [], [], [], [], [], [], []
    for rec in records:
        p, e, lm, al, rt = readout(rec, kind)
        g = rec["names"].index(rec["gold"])
        top = max(range(len(p)), key=lambda i: p[i])
        gold_idx.append(g)
        probs.append(p)
        conf.append(p[top])
        correct.append(top == g)
        ent.append(e)
        mass.append(lm)
        arg_ok.append(al)
        ret.append(rt)
    kept = [i for i, e in enumerate(ent) if e <= AUTO_THRESHOLD]
    esc = [i for i, e in enumerate(ent) if e > AUTO_THRESHOLD]
    counts = {}
    for g in gold_idx:
        counts[g] = counts.get(g, 0) + 1
    out = {
        "n": len(records),
        "accuracy": mean(correct),
        "majority_accuracy": max(counts.values()) / len(records),
        "mean_confidence": mean(conf),
        "ece": ece(conf, correct),
        "brier": brier(probs, gold_idx),
        "nll": nll(probs, gold_idx),
        "auroc_conf_vs_correct": auroc(conf, correct),
        "label_mass_median": median(mass),
        "label_mass_under_half": mean([m < 0.5 for m in mass]),
        "argmax_is_label": mean(arg_ok),
        "all_labels_returned": mean(ret),
        "gate_escalated": len(esc) / len(records),
        "gate_acc_kept": mean([correct[i] for i in kept]),
        "gate_acc_escalated": mean([correct[i] for i in esc]),
        "auroc_low_entropy_vs_correct": auroc([-e for e in ent], correct),
    }
    return out, correct, conf, probs, gold_idx


def spread_stats(records):
    sp = [spread(r) for r in records]
    if any(s is None for s in sp):
        return {}
    _, correct, _, _, _ = score(records, "dg4")
    wrong = [s for s, c in zip(sp, correct) if not c]
    return {
        "spread_median": median(sp),
        "auroc_low_spread_vs_correct": auroc([-s for s in sp], correct),
        "wrong_with_spread_under_0.02": mean([s < 0.02 for s in wrong]) if wrong else None,
    }


def paired_diff(a, b, stat, seed=0):
    """stat(b) - stat(a) over the same examples, with a 95% range from
    resampling examples with replacement."""
    n = len(a[1])  # a is score()'s tuple; [1] is the per-example correctness
    rng = random.Random(seed)
    point = stat(b, range(n)) - stat(a, range(n))
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(stat(b, idx) - stat(a, idx))
    draws.sort()
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


def acc_stat(res, idx):
    correct = res[1]
    return sum(correct[i] for i in idx) / len(idx)


def ece_stat(res, idx):
    return ece([res[2][i] for i in idx], [res[1][i] for i in idx])


def brier_stat(res, idx):
    return brier([res[3][i] for i in idx], [res[4][i] for i in idx])


def latency(records, kind):
    first = [r["latency"]["first_ms"] for r in records]
    if kind != "dgauto":
        return {"median_ms": median(first), "p90_ms": sorted(first)[int(0.9 * len(first)) - 1]}
    total = []
    for r in records:
        t = r["latency"]["first_ms"]
        if r["reads"][0]["entropy"] > AUTO_THRESHOLD and r["latency"]["extra_ms"] is not None:
            t += r["latency"]["extra_ms"]
        total.append(t)
    return {"median_ms": median(total), "p90_ms": sorted(total)[int(0.9 * len(total)) - 1]}


# ---------------------------------------------------------------------------
# Calibration against labels used for tuning, and option-order sensitivity.
# ---------------------------------------------------------------------------

CURVE_NS = (0, 25, 50, 100, 150)
T_GRID = [round(0.25 * 1.05 ** i, 4) for i in range(0, 72)]  # 0.25 .. ~7.9


def temper(p, t):
    """Rescale a label distribution by temperature t: p_i^(1/t), renormalised."""
    w = [max(x, 1e-12) ** (1.0 / t) for x in p]
    z = sum(w)
    return [x / z for x in w]


def fit_temperature(probs, gold_idx):
    """The grid temperature with the lowest log loss on the given examples."""
    return min(T_GRID, key=lambda t: nll([temper(p, t) for p in probs], gold_idx))


def label_curve(records, kind, seed=20260923):
    """Split the examples once with a fixed seed: a fitting pool and a held-out
    half. For each N, fit one temperature on the first N of the pool (N = 0 is
    no fitting) and score the held-out half."""
    _, correct, _, probs, gold = score(records, kind)
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    half = len(idx) // 2
    pool, held = idx[:half], idx[half:]
    out = []
    for n in CURVE_NS:
        if n > len(pool):
            continue
        t = 1.0 if n == 0 else fit_temperature([probs[i] for i in pool[:n]], [gold[i] for i in pool[:n]])
        hp = [temper(probs[i], t) for i in held]
        hg = [gold[i] for i in held]
        conf = [max(p) for p in hp]
        corr = [max(range(len(p)), key=lambda k: p[k]) == g for p, g in zip(hp, hg)]
        out.append({"n_labels": n, "temperature": t, "ece": ece(conf, corr), "nll": nll(hp, hg), "brier": brier(hp, hg), "n_heldout": len(held)})
    return out


def predicted_name(rec, kind):
    p = readout(rec, kind)[0]
    return rec["names"][max(range(len(p)), key=lambda i: p[i])]


def flip_rate(base, variant, kind):
    """Share of examples whose predicted option name changes when the options
    are listed in reverse order."""
    ids = sorted(set(base) & set(variant))
    flips = sum(predicted_name(base[i], kind) != predicted_name(variant[i], kind) for i in ids)
    return {"n": len(ids), "flips": flips, "rate": flips / len(ids) if ids else None}


# ---------------------------------------------------------------------------


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def fmt(x, pct=False):
    if x is None:
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    rundir = os.path.join(HERE, "results", args.run)
    tasks = sorted(
        {f.split("-", 1)[1][:-6] for f in os.listdir(rundir) if f.endswith(".jsonl") and "--" not in f}
    )
    summary = {}
    md = [f"# Jev-style reads: DiffusionGemma vs Gemma 4 26B — run `{args.run}`", ""]
    md.append(
        "Computed by `score.py` from the records in this directory. "
        f"Escalation threshold {AUTO_THRESHOLD} (structured_server default); "
        f"ECE over {ECE_BINS} equal-width bins; ranges are 95% over {BOOT} resamples of examples."
    )
    for task in tasks:
        paths = {
            arm: os.path.join(rundir, f"{arm}-{task}.jsonl")
            for arm in ("autoregressive", "diffusion")
        }
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        ar, dg = load(paths["autoregressive"]), load(paths["diffusion"])
        ids = sorted(set(ar) & set(dg))
        ar_recs, dg_recs = [ar[i] for i in ids], [dg[i] for i in ids]
        rows = {}
        res = {}
        for kind, recs in (("ar", ar_recs), ("dg1", dg_recs), ("dg4", dg_recs), ("dgauto", dg_recs)):
            s, *rest = score(recs, kind)
            s |= latency(recs, kind)
            rows[kind] = s
            res[kind] = (s, *rest)
        rows["dg_spread"] = spread_stats(dg_recs)
        rows["diff_dgauto_minus_ar"] = {
            name: paired_diff(res["ar"], res["dgauto"], f)
            for name, f in (("accuracy", acc_stat), ("ece", ece_stat), ("brier", brier_stat))
        }
        summary[task] = rows
        md += ["", f"## {task} (n={len(ids)})", ""]
        md.append("| readout | accuracy | ECE | Brier | AUROC conf→correct | median label mass | escalated | acc kept / escalated | median ms |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for kind in ("ar", "dg1", "dg4", "dgauto"):
            s = rows[kind]
            md.append(
                f"| {kind} | {fmt(s['accuracy'], True)} | {fmt(s['ece'])} | {fmt(s['brier'])} | "
                f"{fmt(s['auroc_conf_vs_correct'])} | {fmt(s['label_mass_median'])} | "
                f"{fmt(s['gate_escalated'], True)} | {fmt(s['gate_acc_kept'], True)} / "
                f"{fmt(s['gate_acc_escalated'], True)} | {s['median_ms']:.0f} |"
            )
        md.append("")
        md.append(f"Majority-class accuracy: {fmt(rows['ar']['majority_accuracy'], True)}.")
        for name, (pt, lo, hi) in rows["diff_dgauto_minus_ar"].items():
            md.append(f"- dgauto − ar, {name}: {pt:+.3f} (95% range {lo:+.3f} to {hi:+.3f})")
        sp = rows["dg_spread"]
        if sp:
            md.append(
                f"- DiffusionGemma spread across noise draws: median {fmt(sp['spread_median'])}; "
                f"low spread predicts correct with AUROC {fmt(sp['auroc_low_spread_vs_correct'])}; "
                f"{fmt(sp['wrong_with_spread_under_0.02'], True)} of wrong answers had spread under 0.02"
            )
        rows["label_curve"] = {k: label_curve(r, k) for k, r in (("ar", ar_recs), ("dg1", dg_recs), ("dg4", dg_recs))}
        md += ["", "Calibration on the held-out half after fitting one temperature on N labels:", "",
               "| N labels | ar T | ar ECE | dg1 T | dg1 ECE | dg4 T | dg4 ECE |", "|---|---|---|---|---|---|---|"]
        for j, pt in enumerate(rows["label_curve"]["ar"]):
            c = [rows["label_curve"][k][j] for k in ("ar", "dg1", "dg4")]
            md.append(f"| {pt['n_labels']} | " + " | ".join(f"{x['temperature']:.2f} | {x['ece']:.3f}" for x in c) + " |")
        flips = {}
        for arm, kind in (("autoregressive", "ar"), ("diffusion", "dg1")):
            vpath = os.path.join(rundir, f"{arm}-{task}--reversed.jsonl")
            if os.path.exists(vpath):
                flips[kind] = flip_rate(load(paths[arm]), load(vpath), kind)
        if flips:
            rows["option_order_flips"] = flips
            md.append("")
            md.append("Options listed in reverse order: " + "; ".join(
                f"{k} changed {v['flips']} of {v['n']} answers ({fmt(v['rate'], True)})" for k, v in flips.items()))
        for kind in ("ar", "dg1"):
            s = rows[kind]
            if s["all_labels_returned"] is not None and s["all_labels_returned"] < 1:
                md.append(
                    f"- ⚠️ {kind}: only {fmt(s['all_labels_returned'], True)} of examples returned a "
                    "logprob for every label; the rest were scored at the floor"
                )
    with open(os.path.join(rundir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(rundir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
