"""results.jsonl per (model, arm) -> aggregate output tok/s per level (median of repeats, spread
(max-min)/median), drafter/none ratio, acceptance and median TTFT."""
import json, os, statistics as st

def load(m, arm):
    p = f"{m}/{arm}/results.jsonl"
    if not os.path.exists(p):
        return {}
    cells = {}
    for line in open(p):
        r = json.loads(line)
        cells.setdefault((r["mode"], r["c"]), []).append(r)
    out = {}
    for k, reps in cells.items():
        rates = [sum(x["n"] for x in r["runs"]) / r["wall_s"] for r in reps]
        runs = [x for r in reps for x in r["runs"]]
        med = st.median(rates)
        dn = sum(x["draft_n"] for x in runs)
        out[k] = {"tps": med, "spread": (max(rates) - min(rates)) / med, "reps": len(reps),
                  "acc": sum(x["draft_accepted"] for x in runs) / dn if dn else None,
                  "ttft": st.median(x["ttft_ms"] for x in runs), "n": st.mean(x["n"] for x in runs)}
    return out

for m in ("e2b", "e4b"):
    a, b = load(m, "none"), load(m, "q4x-n2")
    for mode in ("greedy", "sampled"):
        print(f"\n{m.upper()} {mode}: aggregate output tok/s (spread), drafter/none, acceptance, median TTFT ms, mean tokens")
        print(f"{'c':>3} {'none':>14} {'drafter':>14} {'ratio':>6} {'acc':>6} {'ttft none':>9} {'ttft dr':>8} {'tok none':>8} {'tok dr':>7}")
        for c in (1, 2, 4, 8, 16, 32, 64):
            x, y = a.get((mode, c)), b.get((mode, c))
            if not x and not y:
                continue
            f = lambda z: f"{z['tps']:7.1f} ({z['spread']*100:4.1f}%)" if z else f"{'—':>14}"
            ratio = f"{y['tps']/x['tps']:6.2f}" if x and y else f"{'':>6}"
            acc = f"{y['acc']:6.3f}" if y and y["acc"] is not None else f"{'':>6}"
            t = lambda z, k, w, p: f"{z[k]:{w}.{p}f}" if z else f"{'—':>{w}}"
            print(f"{c:>3} {f(x)} {f(y)} {ratio} {acc} {t(x,'ttft',9,0)} {t(y,'ttft',8,0)} {t(x,'n',8,1)} {t(y,'n',7,1)}")
