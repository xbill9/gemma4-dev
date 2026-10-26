"""Split each card's drafter speed-up into tokens per step and the cost of one step, from the speed.py results
already recorded (greedy, six prompts, 256 tokens, llama.cpp 77c4540ce on all three cards).

A step is one pass of the target over the last token plus the drafted ones, after the drafter has proposed them.
Steps per reply = tokens - accepted drafted tokens. Step time = reply decode time / steps. Pooled over every greedy
reply of an arm; the speed-up here is pooled tokens over pooled time, so it can differ from the reports' geometric
mean by a little.

    python3 step_cost.py > step_cost.txt; also writes step_cost.json
"""
import json, os

R = os.path.dirname(os.path.abspath(__file__)) + "/../../../.."
SRC = {
    ("GTX 1650 Ti", "E2B"): "local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-1650ti/e2b/speed/results.jsonl",
    ("GTX 1650 Ti", "E4B"): "local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-1650ti/e4b/speed/results.jsonl",
    ("RTX 3050 OEM", "E2B"): "local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem/e2b/speed/results.jsonl",
    ("RTX 3050 OEM", "E4B"): "local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem/e4b/speed/results.jsonl",
    ("L4", "E2B"): "gpu-llamacpp-g6-2b-q4_0/benchmarks/runs/2026-10-26-drafter-g6/speed/results.jsonl",
    ("L4", "E4B"): "gpu-llamacpp-g6-4b-q4_0/benchmarks/runs/2026-10-26-drafter-g6/speed/results.jsonl",
}
ARMS = ["none", "q4x-n1", "q4x-n2", "q4x-n3", "q4x-n4"]
out = []
print("card          model  arm      tok/step  step ms  step / plain step  speed-up = tok/step / step cost   best")
for (card, model), p in SRC.items():
    rows = [json.loads(l) for l in open(f"{R}/{p}")]
    agg = {}
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a and r["mode"] == "greedy"]
        if not rs:
            continue
        n = sum(r["n"] for r in rs); acc = sum(r["draft_accepted"] for r in rs)
        t = sum(r["n"] / r["decode_tps"] for r in rs)
        agg[a] = {"tokens": n, "accepted": acc, "steps": n - acc, "time_s": t,
                  "tok_per_step": n / (n - acc), "step_ms": 1000 * t / (n - acc)}
    base = agg["none"]["step_ms"]
    for a, x in agg.items():
        x["step_cost"] = x["step_ms"] / base
        x["speedup"] = x["tok_per_step"] / x["step_cost"]
    best = max(agg, key=lambda a: agg[a]["speedup"])
    for a, x in agg.items():
        print(f"{card:13s} {model:5s}  {a:8s} {x['tok_per_step']:8.3f} {x['step_ms']:8.2f} {x['step_cost']:17.3f} "
              f"{x['speedup']:12.3f}{'   <-' if a == best else ''}")
        out.append({"card": card, "model": model, "arm": a, **x})
    print()
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/step_cost.json", "w"), indent=1)
