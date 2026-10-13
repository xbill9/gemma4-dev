"""E2B vs E4B answer quality, exact Q4_0 GGUFs on the RTX 3050 OEM, with TPU anchors.

    python3 compare.py      # writes COMPARE.md and compare.json beside this file

Suite correctness is the harness's own (score_suite.assess on score_suite.dist, gold_key) and the
paired difference with its 2,000-draw bootstrap range is quant_compare.paired. GSM8K and BFCL are
gen_eval.py's own `right` field, paired by gen_compare.compare (10,000 resamples, seed 0).
Nothing is counted by hand.
"""
import glob
import json
import os
import sys

J = "/home/xbill/gemma4-dev/jev-tpu-v5e1"
sys.path[:0] = [J, os.path.join(J, "nimble_suite")]
from gen_compare import compare as gpair, load as gload  # noqa: E402
from quant_compare import paired  # noqa: E402
from score_suite import assess, dist, gold_key  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = {
    "E2B Q4 (3050)": HERE + "/2026-10-13-suite-3050oem-e2b-q4exact",
    "E4B Q4 (3050)": HERE + "/2026-10-13-suite-3050oem-e4b-q4exact",
    "E2B bf16 (TPU v5e)": f"{J}/results/2026-09-29-w8a8-v5e1-e2b-bf16-suite",
    "E4B bf16 (TPU v6e)": "/home/xbill/gemma4-dev/jev-tpu-31b/results/2026-09-25-followup-e4b-bf16-suite",
}
GEN = {
    "E2B Q4 (3050)": HERE + "/{local}-e2b.jsonl",
    "E4B Q4 (3050)": HERE + "/{local}-e4b.jsonl",
    "E2B bf16 (TPU v5e)": J + "/results/2026-09-30-gen2048a-v5e1-e2b-bf16-gen/{t}.jsonl",
    "E4B QAT repack W4A16 (TPU v5e)": J + "/results/2026-09-30-gen2048b2-v5e1-e4b-repack-gen/{t}.jsonl",
}


def suite_load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "reads" in r:
                full = r["reads"][0].get("labels_returned")
                out[(r["subset"], r["id"])] = (assess(dist(r, "ar"), gold_key(r))["correct"],
                                               None if full is None else full == len(r["keys"]))
    return out


res = {"suite": {}, "suite_pairs": [], "subsets": {}, "gen": {}}
md = ["# Answer quality: E2B vs E4B, exact Q4_0 on the RTX 3050 OEM", ""]

S = {k: suite_load(v) for k, v in SUITE.items()}
md += ["## Nimble 13-subset suite", "", "| run | records | accuracy | all labels returned |", "|---|---:|---:|---:|"]
for k, c in S.items():
    cov = [f for _, f in c.values()]
    cv = None if any(f is None for f in cov) else sum(cov) / len(cov)
    acc = sum(ok for ok, _ in c.values()) / len(c)
    res["suite"][k] = {"n": len(c), "acc": acc, "coverage": cv}
    md.append(f"| {k} | {len(c)} | {acc:.4f} | {'-' if cv is None else f'{cv:.1%}'} |")
md += ["", "Paired on the same records (difference = B - A; 95% range from 2,000 bootstrap draws):", "",
       "| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |",
       "|---|---|---:|---:|---:|---:|---|---:|---:|"]
for a, b in [("E2B Q4 (3050)", "E4B Q4 (3050)"), ("E2B bf16 (TPU v5e)", "E2B Q4 (3050)"),
             ("E4B bf16 (TPU v6e)", "E4B Q4 (3050)"), ("E2B bf16 (TPU v5e)", "E4B bf16 (TPU v6e)")]:
    s = paired(S[a], S[b])
    res["suite_pairs"].append({"a": a, "b": b, **s})
    md.append(f"| {a} | {b} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | **{s['diff']:+.3f}** | "
              f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} | {s['ref_right_test_wrong']} | {s['ref_wrong_test_right']} |")
md += ["", "Per subset, E4B - E2B on the 3050 (secondary; ranges not corrected for 13 comparisons):", "",
       "| subset | n | E2B | E4B | E4B - E2B | 95% range |", "|---|---:|---:|---:|---:|---|"]
for sub in sorted({s for s, _ in S["E2B Q4 (3050)"]}):
    g = {k: {i: v for i, v in S[k].items() if i[0] == sub} for k in ("E2B Q4 (3050)", "E4B Q4 (3050)")}
    s = paired(g["E2B Q4 (3050)"], g["E4B Q4 (3050)"])
    res["subsets"][sub] = s
    md.append(f"| {sub} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | {s['diff']:+.3f} | "
              f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} |")

md += ["", "## GSM8K (2,048-token limit) and BFCL simple", "",
       "| task | A | B | n | A | B | difference | 95% range | A only | B only |",
       "|---|---|---|---:|---:|---:|---:|---|---:|---:|"]
for t in ("gsm8k", "bfcl_simple"):
    for a, b in [("E2B Q4 (3050)", "E4B Q4 (3050)"), ("E2B bf16 (TPU v5e)", "E2B Q4 (3050)"),
                 ("E4B QAT repack W4A16 (TPU v5e)", "E4B Q4 (3050)")]:
        local = {"gsm8k": "gsm8k", "bfcl_simple": "bfcl"}[t]  # run_quality.sh's file names
        fa, fb = GEN[a].format(t=t, local=local), GEN[b].format(t=t, local=local)
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        s = gpair(gload(fa), gload(fb))
        res["gen"].setdefault(t, []).append({"a": a, "b": b, **s})
        md.append(f"| {t} | {a} | {b} | {s['n']} | {s['ref']:.3f} | {s['test']:.3f} | **{s['diff']:+.3f}** | "
                  f"{s['lo']:+.3f} to {s['hi']:+.3f} | {s['ref_only']} | {s['test_only']} |")
open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(md) + "\n")
json.dump(res, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
print("\n".join(md))
