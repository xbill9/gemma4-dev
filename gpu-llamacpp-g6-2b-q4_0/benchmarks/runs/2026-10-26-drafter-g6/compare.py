"""GSM8K and BFCL simple with the drafter (2 tokens) against without it, on the L4.

    python3 compare.py      # writes accuracy/COMPARE.md and accuracy/compare.json

Correctness is gen_eval.py's own `right`; pairing and 95% ranges are gen_compare.compare (10,000 resamples, seed 0).
Acceptance comes from llama-server's /metrics counters before and after each arm. Nothing is counted by hand.
"""
import json, os, re, sys

sys.path.insert(0, "/home/xbill/gemma4-dev/jev-tpu-v5e1")
from gen_compare import compare, load  # noqa: E402

A = os.path.dirname(os.path.abspath(__file__)) + "/accuracy"


def metrics(p):
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^llamacpp:(\w+) ([\d.e+]+)$", open(p).read(), re.M)}


def wall(t, arm):
    return json.loads(open(f"{A}/{t}-{arm}.log").read().strip().splitlines()[-1])["wall_s"]


res = {"pairs": []}
md = ["# GSM8K and BFCL simple with and without the drafter, NVIDIA L4", "",
      "| task | n | no drafter | drafter, 2 tokens | difference | 95% range | none only | drafter only | same text | wall none, s | wall drafter, s | speed-up |",
      "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
for t in ("gsm8k", "bfcl_simple"):
    pa, pb = f"{A}/{t}-none.jsonl", f"{A}/{t}-q4x-n2.jsonl"
    s = compare(load(pa), load(pb))
    ta = {r["id"]: json.dumps([r.get("text"), r.get("calls")], sort_keys=True) for r in map(json.loads, open(pa))}
    tb = {r["id"]: json.dumps([r.get("text"), r.get("calls")], sort_keys=True) for r in map(json.loads, open(pb))}
    same = sum(ta[i] == tb[i] for i in ta if i in tb)
    wa, wb = wall(t, "none"), wall(t, "q4x-n2")
    res["pairs"].append({"task": t, "same_text": same, "wall_none": wa, "wall_q4x": wb, **s})
    md.append(f"| {t} | {s['n']} | {s['ref']:.4f} | {s['test']:.4f} | **{s['diff']:+.4f}** | {s['lo']:+.4f} to {s['hi']:+.4f} | "
              f"{s['ref_only']} | {s['test_only']} | {same} | {wa:.0f} | {wb:.0f} | {wa / wb:.2f}x |")
b, a = metrics(f"{A}/metrics-before-q4x-n2.prom"), metrics(f"{A}/metrics-after-q4x-n2.prom")
keys = sorted(k for k in a if "draft" in k)
res["draft_metrics"] = {k: a[k] - b.get(k, 0) for k in keys}
md += ["", "Drafter counters over both tasks (after − before): " +
       ", ".join(f"`{k}` {v:g}" for k, v in res["draft_metrics"].items())]
json.dump(res, open(f"{A}/compare.json", "w"), indent=1)
open(f"{A}/COMPARE.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
