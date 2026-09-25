"""Paired accuracy comparison: 4-bit arms against bf16 arms on the same records.

    python3 quant_compare.py --prefix 2026-09-25-w4a16 \
        --pair e2b-w4a16=../jev-tpu/results/2026-09-24-v6e1-e2b \
        --pair e4b-w4a16=../jev-tpu/results/2026-09-24-v6e1-e4b \
        --pair 12b-w4a16=results/2026-09-25-w4a16-12b-bf16 \
        --solo 31b-w4a16

Each --pair ARM=REF matches records of results/<prefix>-<arm> with those of REF by id (and
variant for the four tasks), then reports per task and for the public suite: n, accuracy of
each, the paired difference with a 95% bootstrap range over examples, and how many examples
flip each way. A --solo arm has no bf16 reference on one chip and gets accuracies only.
Correctness uses the harness's own definitions: argmax of the read's probabilities against
the gold label for the tasks (score.py), score_suite.assess for the suite.

The read takes labels from the top 32 log-probabilities and scores a missing label at a
floor, and two arms can differ in how often that happens. So each group also reports the
share of records where every label came back, per arm, and the same paired difference on
only the records where both arms returned every label. Suite records carry that count
only from runs made after it was added to run_suite.py; older ones show a dash.
Only the suite difference is pre-registered as primary; the per-task rows are secondary
and their ranges are not corrected for multiple comparisons.
Writes results/<prefix>-QUANT.md and .json.
"""

import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "nimble_suite"))
from score_suite import assess, dist, gold_key  # noqa: E402

TASKS = ("sst2", "ag_news", "emotion", "irony")


def full_coverage(read, n_labels):
    """True when every label came back with a real logprob; None when not recorded."""
    k = read.get("labels_returned")
    return None if k is None else k == n_labels


def task_correct(run_dir):
    """{(task, variant, id): (correct, full_coverage)} over the four tasks, both orders."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "autoregressive-*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "reads" not in r:
                continue
            p = r["reads"][0]["probs"]
            ok = max(range(len(p)), key=p.__getitem__) == r["names"].index(r["gold"])
            out[(r["task"], r["variant"], r["id"])] = (ok, full_coverage(r["reads"][0], len(r["labels"])))
    return out


def suite_correct(run_dir):
    """{(subset, id): (correct, full_coverage)} over the public suite."""
    out = {}
    for f in glob.glob(os.path.join(run_dir + "-suite", "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "reads" not in r:
                continue
            ok = assess(dist(r, "ar"), gold_key(r))["correct"]
            out[(r["subset"], r["id"])] = (ok, full_coverage(r["reads"][0], len(r["keys"])))
    return out


def coverage(recs):
    flags = [c for _, c in recs.values()]
    return None if any(c is None for c in flags) else sum(flags) / len(flags)


def paired(ref, test, seed=0, n=2000):
    keys = sorted(set(ref) & set(test))
    a = [ref[k][0] for k in keys]
    b = [test[k][0] for k in keys]
    diffs = [int(y) - int(x) for x, y in zip(a, b)]
    rng = random.Random(seed)
    k = len(diffs)
    draws = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return {
        "n": k,
        "only_in_ref": len(set(ref) - set(test)),
        "only_in_test": len(set(test) - set(ref)),
        "acc_ref": sum(a) / k,
        "acc_test": sum(b) / k,
        "diff": sum(diffs) / k,
        "diff_lo": draws[int(0.025 * n)],
        "diff_hi": draws[int(0.975 * n) - 1],
        "ref_right_test_wrong": sum(x and not y for x, y in zip(a, b)),
        "ref_wrong_test_right": sum(y and not x for x, y in zip(a, b)),
    }


def solo(test):
    return {"n": len(test), "acc_test": sum(c for c, _ in test.values()) / len(test),
            "coverage": coverage(test)}


def both_full(ref, test):
    """The records where both arms returned every label, or None if either lacks the count."""
    keys = set(ref) & set(test)
    if any(ref[k][1] is None or test[k][1] is None for k in keys):
        return None
    keep = {k for k in keys if ref[k][1] and test[k][1]}
    return {k: ref[k] for k in keep}, {k: test[k] for k in keep}


def pct(x):
    return "-" if x is None else f"{x:.1%}"


def groups(correct, suite):
    """Split a correctness map into the reported groups."""
    if suite:
        return {"suite (all subsets)": correct}
    g = {t: {k: v for k, v in correct.items() if k[0] == t and k[1] == "none"} for t in TASKS}
    g["reversed options (3 choice tasks)"] = {k: v for k, v in correct.items() if k[1] != "none"}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--pair", action="append", default=[], help="ARM=REF_RUN_DIR")
    ap.add_argument("--solo", action="append", default=[])
    args = ap.parse_args()

    out, md = {}, [f"# {args.prefix}: 4-bit against bf16, paired on the same records", ""]
    for spec in args.pair:
        arm, ref_dir = spec.split("=", 1)
        test_dir = os.path.join(HERE, "results", f"{args.prefix}-{arm}")
        ref_dir = os.path.join(HERE, ref_dir) if not os.path.isabs(ref_dir) else ref_dir
        md += [f"## {arm} against {os.path.relpath(ref_dir, HERE)}", "",
               "| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right "
               "| all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |",
               "|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|"]
        out[arm] = {}
        for suite in (False, True):
            ref_g = groups(suite_correct(ref_dir) if suite else task_correct(ref_dir), suite)
            test_g = groups(suite_correct(test_dir) if suite else task_correct(test_dir), suite)
            for name in ref_g:
                if not test_g.get(name) or not ref_g[name]:
                    continue
                s = paired(ref_g[name], test_g[name])
                s["coverage_ref"], s["coverage_test"] = coverage(ref_g[name]), coverage(test_g[name])
                sub = both_full(ref_g[name], test_g[name])
                s["both_full"] = paired(*sub) if sub and sub[0] else None
                out[arm][name] = s
                bf = s["both_full"]
                tail = (f"{bf['n']} | {bf['diff']:+.3f} | {bf['diff_lo']:+.3f} to {bf['diff_hi']:+.3f}"
                        if bf else "- | - | -")
                md.append(f"| {name} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | {s['diff']:+.3f} | "
                          f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} | {s['ref_right_test_wrong']} | {s['ref_wrong_test_right']} | "
                          f"{pct(s['coverage_ref'])} / {pct(s['coverage_test'])} | {tail} |")
        md.append("")
    for arm in args.solo:
        test_dir = os.path.join(HERE, "results", f"{args.prefix}-{arm}")
        md += [f"## {arm} (no bf16 reference fits one chip)", "", "| group | n | 4-bit | all labels returned |",
               "|---|---:|---:|---:|"]
        out[arm] = {}
        for suite in (False, True):
            for name, c in groups(suite_correct(test_dir) if suite else task_correct(test_dir), suite).items():
                if c:
                    out[arm][name] = solo(c)
                    md.append(f"| {name} | {len(c)} | {out[arm][name]['acc_test']:.3f} | {pct(out[arm][name]['coverage'])} |")
        md.append("")
    base = os.path.join(HERE, "results", f"{args.prefix}-QUANT")
    json.dump(out, open(base + ".json", "w"), indent=1)
    open(base + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
