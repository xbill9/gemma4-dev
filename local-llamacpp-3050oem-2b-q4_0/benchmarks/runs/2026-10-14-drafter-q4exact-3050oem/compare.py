"""Answer quality with each drafter against no drafter, E2B and E4B on the RTX 3050 OEM.

    python3 compare.py      # writes accuracy/COMPARE.md and accuracy/compare.json

Two no-drafter arms: `none` is run_control.sh, the same day as the drafted arms, and
`none-1013` is the 2026-10-13 quality run (identical flags) — paired with `none` it tests
whether that baseline reproduces.
Correctness is gen_eval.py's own `right`; pairing and 95% ranges are gen_compare.compare
(10,000 resamples, seed 0). Drafter acceptance is read off llama-server's /metrics counters
before and after each arm. Greedy texts are compared by exact string. Nothing is counted by hand.
"""
import json, os, re, sys

J = "/home/xbill/gemma4-dev/jev-tpu-v5e1"
sys.path.insert(0, J)
from gen_compare import compare, load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A = HERE + "/accuracy"
BASE = "/home/xbill/gemma4-dev/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-10-13-quality-e2b-vs-e4b-3050oem"
LOCAL = {"gsm8k": "gsm8k", "bfcl_simple": "bfcl"}


def path(model, arm, t):
    return f"{BASE}/{LOCAL[t]}-{model}.jsonl" if arm == "none-1013" else f"{A}/{t}-{model}-{arm}.jsonl"


def texts(p):
    return {r["id"]: json.dumps([r.get("text"), r.get("calls")], sort_keys=True) for r in map(json.loads, open(p))}


def wall(model, arm, t):
    """wall_s from gen_eval.py's closing summary line."""
    p = f"{BASE}/{LOCAL[t]}-{model}.log" if arm == "none-1013" else f"{A}/{t}-{model}-{arm}.log"
    return json.loads(open(p).read().strip().splitlines()[-1])["wall_s"]


def metrics(p):
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^llamacpp:(\w+) ([\d.e+]+)$", open(p).read(), re.M)}


res, md = {"pairs": [], "accept": {}}, ["# Answer quality with each drafter, RTX 3050 OEM", "",
     "| model | task | A | B | n | A | B | B - A | 95% range | A only | B only | same text |",
     "|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
for model in ("e2b", "e4b"):
    for t in LOCAL:
        for a, b in [("none-1013", "none"), ("none", "q8"), ("none", "q4x"), ("q8", "q4x")]:
            pa, pb = path(model, a, t), path(model, b, t)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            s = compare(load(pa), load(pb))
            ta, tb = texts(pa), texts(pb)
            same = sum(ta[i] == tb[i] for i in ta if i in tb)
            res["pairs"].append({"model": model, "task": t, "a": a, "b": b, "same_text": same, **s})
            md.append(f"| {model} | {t} | {a} | {b} | {s['n']} | {s['ref']:.3f} | {s['test']:.3f} | **{s['diff']:+.3f}** | "
                      f"{s['lo']:+.3f} to {s['hi']:+.3f} | {s['ref_only']} | {s['test_only']} | {same} |")
md += ["", "Wall time per task at 8 requests at once (gen_eval.py's wall_s, seconds):", "",
       "| model | task | none | q8 | q4x | q8 / none | q4x / none |", "|---|---|---:|---:|---:|---:|---:|"]
for model in ("e2b", "e4b"):
    for t in LOCAL:
        try:
            w = {arm: wall(model, arm, t) for arm in ("none", "q8", "q4x")}
        except (FileNotFoundError, KeyError, ValueError, IndexError):
            continue
        res.setdefault("wall", {})[f"{model}-{t}"] = w
        md.append(f"| {model} | {t} | {w['none']:.0f} | {w['q8']:.0f} | {w['q4x']:.0f} | "
                  f"{w['none'] / w['q8']:.2f}x faster | {w['none'] / w['q4x']:.2f}x faster |")
md += ["", "Drafter acceptance over both tasks (from /metrics):", "",
       "| arm | drafted | accepted | acceptance | tokens generated |", "|---|---:|---:|---:|---:|"]
for model in ("e2b", "e4b"):
    for arm in ("none", "q8", "q4x"):
        f0, f1 = f"{A}/metrics-before-{model}-{arm}.prom", f"{A}/metrics-after-{model}-{arm}.prom"
        if not os.path.exists(f1):
            continue
        m0, m1 = metrics(f0), metrics(f1)
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in ("spec_decode_num_draft_tokens_total",
             "spec_decode_num_accepted_tokens_total", "tokens_predicted_total")}
        acc = d["spec_decode_num_accepted_tokens_total"] / max(1, d["spec_decode_num_draft_tokens_total"])
        res["accept"][f"{model}-{arm}"] = {**d, "acceptance": acc}
        md.append(f"| {model}-{arm} | {d['spec_decode_num_draft_tokens_total']:.0f} | "
                  f"{d['spec_decode_num_accepted_tokens_total']:.0f} | {acc:.4f} | {d['tokens_predicted_total']:.0f} |")
open(A + "/COMPARE.md", "w").write("\n".join(md) + "\n")
json.dump(res, open(A + "/compare.json", "w"), indent=1)
print("\n".join(md))
