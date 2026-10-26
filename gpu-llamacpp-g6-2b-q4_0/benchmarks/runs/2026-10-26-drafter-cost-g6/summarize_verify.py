"""verify/<model>.jsonl (llama-bench) -> time for one pass over k tokens, against k = 1, per depth; then the L4
step of step_cost.json split into the target's pass over 1 + n tokens (depth 0) and the rest (the drafter and
bookkeeping).

    python3 summarize_verify.py > verify/summary.txt
"""
import json, os

H = os.path.dirname(os.path.abspath(__file__))
steps = {(r["model"], r["arm"]): r for r in json.load(open(f"{H}/step_cost.json")) if r["card"] == "L4"}
for m in ("e2b", "e4b"):
    rows = [json.loads(l) for l in open(f"{H}/verify/{m}.jsonl") if l.startswith("{")]
    ms = {(r["n_depth"], r["n_prompt"]): r["avg_ns"] / 1e6 for r in rows}
    sd = {(r["n_depth"], r["n_prompt"]): r["stddev_ns"] / 1e6 for r in rows}
    M = m.upper()
    print(f"{M}: one target pass over k tokens, ms (x vs k = 1)")
    depths = sorted({d for d, _ in ms}); ks = sorted({k for _, k in ms})
    print("depth  " + "".join(f"{'k=' + str(k):>18s}" for k in ks))
    for d in depths:
        print(f"{d:6d} " + "".join(f"{ms[(d, k)]:7.2f}±{sd[(d, k)]:4.2f} ({ms[(d, k)] / ms[(d, 1)]:.2f}x)" for k in ks))
    print(f"\n{M} on the L4, a drafted step at 0 depth: target pass over 1 + n tokens + the rest")
    print("arm      step ms  target pass ms  rest ms  rest per drafted token ms  rest / plain step")
    plain = steps[(M, "none")]["step_ms"]
    for n in range(1, 5):
        s = steps.get((M, f"q4x-n{n}"))
        if not s or (0, n + 1) not in ms:
            continue
        v = ms[(0, n + 1)]; rest = s["step_ms"] - v
        print(f"q4x-n{n}  {s['step_ms']:7.2f} {v:15.2f} {rest:8.2f} {rest / n:26.2f} {rest / plain:18.3f}")
    print(f"plain step {plain:.2f} ms (speed.py), llama-bench k=1 {ms[(0, 1)]:.2f} ms\n")
