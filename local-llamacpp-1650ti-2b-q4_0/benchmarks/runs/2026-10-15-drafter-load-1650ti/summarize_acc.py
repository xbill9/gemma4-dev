"""Part B: drafter against no drafter, paired per task (jev-tpu-v5e1/gen_compare.compare), with wall time."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../../jev-tpu-v5e1"))
from gen_compare import compare, load
print("| | no drafter | drafter | difference (95% range) | only none right | only drafter right | wall s none -> drafter | speed x |")
print("|---|---:|---:|---|---:|---:|---|---:|")
for m in ("e2b", "e4b"):
    for t in ("gsm8k", "bfcl_simple"):
        a, b = (os.path.join(HERE, "acc", f"{t}-{m}-{x}") for x in ("none", "q4x"))
        c = compare(load(a + ".jsonl"), load(b + ".jsonl"))
        wa, wb = (json.loads(open(p + ".log").read().strip().splitlines()[-1])["wall_s"] for p in (a, b))
        print(f"| {m} {t} | {c['ref']:.4f} | {c['test']:.4f} | {c['diff']:+.4f} ({c['lo']:+.4f} to {c['hi']:+.4f}) "
              f"| {c['ref_only']} | {c['test_only']} | {wa:,.0f} -> {wb:,.0f} | {wa / wb:.2f} |")
