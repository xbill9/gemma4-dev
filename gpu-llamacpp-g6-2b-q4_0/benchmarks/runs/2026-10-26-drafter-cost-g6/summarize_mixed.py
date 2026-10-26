"""mixed/<model>/<arm>/results.jsonl -> aggregate output tok/s per level (median of repeats, spread (max-min)/median),
x vs no drafter, acceptance, median time to first token. From the 3050's concurrency-mixed summarize.py, with a
third arm.

    python3 summarize_mixed.py > mixed/summary.txt
"""
import json, os, statistics as st

H = os.path.dirname(os.path.abspath(__file__))

def load(m, arm):
    p = f"{H}/mixed/{m}/{arm}/results.jsonl"
    if not os.path.exists(p):
        return {}
    cells = {}
    for line in open(p):
        r = json.loads(line)
        cells.setdefault(r["c"], []).append(r)
    out = {}
    for c, reps in cells.items():
        rates = [sum(x["n"] for x in r["runs"]) / r["wall_s"] for r in reps]
        runs = [x for r in reps for x in r["runs"]]
        med = st.median(rates); dn = sum(x["draft_n"] for x in runs)
        out[c] = {"tps": med, "spread": (max(rates) - min(rates)) / med,
                  "acc": sum(x["draft_accepted"] for x in runs) / dn if dn else None,
                  "ttft": st.median(x["ttft_ms"] for x in runs)}
    return out

for m in ("e2b", "e4b"):
    a = {arm: load(m, arm) for arm in ("none", "q4x-n2", "q4x-n3")}
    print(f"\n{m.upper()} greedy: aggregate output tok/s (spread), x vs none [acceptance], median TTFT ms")
    print(f"{'c':>3} {'none':>15} {'2 tokens':>30} {'3 tokens':>30}  ttft none / 2 / 3")
    for c in (1, 2, 4, 8, 16, 32, 64):
        x = a["none"].get(c)
        if not x:
            continue
        row = f"{c:>3} {x['tps']:7.1f} ({x['spread']*100:4.1f}%)"
        for arm in ("q4x-n2", "q4x-n3"):
            y = a[arm].get(c)
            row += f" {y['tps']:7.1f} ({y['spread']*100:4.1f}%) {y['tps']/x['tps']:5.2f}x [{y['acc']:.2f}]" if y else f"{'—':>30}"
        row += "  " + " / ".join(f"{a[k][c]['ttft']:.0f}" if c in a[k] else "—" for k in ("none", "q4x-n2", "q4x-n3"))
        print(row)
