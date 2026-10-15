"""1 drafted token against no drafter and 2 drafted tokens (the latter two from
2026-10-15-concurrency-mixed-3050oem, same server settings, same day): aggregate output tok/s
(median of 5 repeats), ratios to no drafter, acceptance at 1 token."""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
MIX = os.path.join(HERE, "..", "2026-10-15-concurrency-mixed-3050oem")

def load(p):
    cells = {}
    for line in open(p):
        r = json.loads(line)
        cells.setdefault((r["mode"], r["c"]), []).append(r)
    out = {}
    for k, reps in cells.items():
        runs = [x for r in reps for x in r["runs"]]
        dn = sum(x["draft_n"] for x in runs)
        out[k] = {"tps": st.median(sum(x["n"] for x in r["runs"]) / r["wall_s"] for r in reps),
                  "acc": sum(x["draft_accepted"] for x in runs) / dn if dn else None,
                  "ttft": st.median(x["ttft_ms"] for x in runs)}
    return out

for m in ("e2b", "e4b"):
    n0 = load(os.path.join(MIX, m, "none", "results.jsonl"))
    n2 = load(os.path.join(MIX, m, "q4x-n2", "results.jsonl"))
    n1 = load(os.path.join(HERE, m, "q4x-n1", "results.jsonl"))
    for mode in ("greedy", "sampled"):
        print(f"\n{m.upper()} {mode}: aggregate output tok/s; ratios to no drafter; acceptance at 1 token; median TTFT ms")
        print(f"{'c':>3} {'none':>7} {'1 tok':>7} {'2 tok':>7} {'1/none':>7} {'2/none':>7} {'acc1':>6} {'ttft0':>6} {'ttft1':>6} {'ttft2':>6}")
        for c in (1, 2, 4, 8, 16, 32, 64):
            a, b, d = n0[(mode, c)], n1[(mode, c)], n2[(mode, c)]
            print(f"{c:>3} {a['tps']:7.1f} {b['tps']:7.1f} {d['tps']:7.1f} {b['tps']/a['tps']:7.2f} {d['tps']/a['tps']:7.2f} "
                  f"{b['acc']:6.3f} {a['ttft']:6.0f} {b['ttft']:6.0f} {d['ttft']:6.0f}")
