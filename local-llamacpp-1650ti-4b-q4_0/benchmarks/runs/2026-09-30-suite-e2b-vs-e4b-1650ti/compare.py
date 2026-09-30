"""Paired suite accuracy: E2B vs E4B exact Q4_0 on the GTX 1650 Ti, and each against its TPU bf16 run.

    python3 compare.py            # writes COMPARE.md and compare.json beside this file

Correctness is the harness's own (score_suite.assess on score_suite.dist, gold_key); the paired
difference and its 2,000-draw bootstrap 95% range are quant_compare.paired, unchanged.
"""
import collections
import glob
import json
import os
import sys

J = "/home/xbill/gemma4-dev/jev-tpu-v5e1"
sys.path[:0] = [J, os.path.join(J, "nimble_suite")]
from quant_compare import paired  # noqa: E402
from score_suite import assess, dist, gold_key  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {
    "E2B Q4 (1650 Ti)": f"{J}/results/2026-09-30-suite-1650ti-e2b-q4exact",
    "E4B Q4 (1650 Ti)": f"{J}/results/2026-09-30-suite-1650ti-e4b-q4exact",
    "E2B bf16 (TPU v5e)": f"{J}/results/2026-09-29-w8a8-v5e1-e2b-bf16-suite",
    "E4B bf16 (TPU v6e)": "/home/xbill/gemma4-dev/jev-tpu-31b/results/2026-09-25-followup-e4b-bf16-suite",
}


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "reads" in r:
                full = r["reads"][0].get("labels_returned")
                out[(r["subset"], r["id"])] = (assess(dist(r, "ar"), gold_key(r))["correct"],
                                               None if full is None else full == len(r["keys"]))
    return out


C = {k: load(v) for k, v in RUNS.items()}
PAIRS = [("E2B bf16 (TPU v5e)", "E2B Q4 (1650 Ti)"), ("E4B bf16 (TPU v6e)", "E4B Q4 (1650 Ti)"),
         ("E2B Q4 (1650 Ti)", "E4B Q4 (1650 Ti)"), ("E2B bf16 (TPU v5e)", "E4B bf16 (TPU v6e)")]
md = ["# Nimble suite: E2B vs E4B, exact Q4_0 on the GTX 1650 Ti, with TPU bf16 anchors", "",
      "| run | records | accuracy | all labels returned |", "|---|---:|---:|---:|"]
res = {"runs": {}, "pairs": [], "subsets": {}}
for k, c in C.items():
    cov = [f for _, f in c.values()]
    cv = None if any(f is None for f in cov) else sum(cov) / len(cov)
    acc = sum(ok for ok, _ in c.values()) / len(c)
    res["runs"][k] = {"n": len(c), "acc": acc, "coverage": cv}
    md.append(f"| {k} | {len(c)} | {acc:.4f} | {'-' if cv is None else f'{cv:.1%}'} |")
md += ["", "## Paired on the same records (difference = B - A)", "",
       "| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |",
       "|---|---|---:|---:|---:|---:|---|---:|---:|"]
for a, b in PAIRS:
    s = paired(C[a], C[b])
    res["pairs"].append({"a": a, "b": b, **s})
    md.append(f"| {a} | {b} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | **{s['diff']:+.3f}** | "
              f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} | {s['ref_right_test_wrong']} | {s['ref_wrong_test_right']} |")
md += ["", "## Per subset (secondary; ranges not corrected for 13 comparisons)", "",
       "| subset | n | E2B Q4 | E4B Q4 | E4B - E2B | 95% range | E2B bf16 | E4B bf16 |", "|---|---:|---:|---:|---:|---|---:|---:|"]
subs = sorted({s for s, _ in C["E2B Q4 (1650 Ti)"]})
for sub in subs:
    g = {k: {i: v for i, v in c.items() if i[0] == sub} for k, c in C.items()}
    s = paired(g["E2B Q4 (1650 Ti)"], g["E4B Q4 (1650 Ti)"])
    a2 = sum(ok for ok, _ in g["E2B bf16 (TPU v5e)"].values()) / max(1, len(g["E2B bf16 (TPU v5e)"]))
    a4 = sum(ok for ok, _ in g["E4B bf16 (TPU v6e)"].values()) / max(1, len(g["E4B bf16 (TPU v6e)"]))
    res["subsets"][sub] = {**s, "e2b_bf16": a2, "e4b_bf16": a4}
    md.append(f"| {sub} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | {s['diff']:+.3f} | "
              f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} | {a2:.3f} | {a4:.3f} |")
open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(md) + "\n")
json.dump(res, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
print("\n".join(md))
