"""<model>/<kv>-<drafter>/results.jsonl -> per pass: aggregate output tok/s (median of repeats,
spread (max-min)/median), drafter/none and shared/per-slot ratios, pass 2 / pass 1, acceptance,
median TTFT and median per-request decode rate."""
import json, os, statistics as st

LEVELS = (1, 2, 4, 8, 16, 32)

def load(m, arm):
    p = f"{m}/{arm}/results.jsonl"
    cells = {}
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            cells.setdefault((r["pass"], r["c"]), []).append(r)
    out = {}
    for k, reps in cells.items():
        rates = [sum(x["n"] for x in r["runs"]) / r["wall_s"] for r in reps]
        runs = [x for r in reps for x in r["runs"]]
        med = st.median(rates); dn = sum(x["draft_n"] for x in runs)
        out[k] = {"tps": med, "spread": (max(rates) - min(rates)) / med, "reps": len(reps),
                  "acc": sum(x["draft_accepted"] for x in runs) / dn if dn else None,
                  "ttft": st.median(x["ttft_ms"] for x in runs),
                  "dec": st.median(x["decode_tps"] for x in runs), "n": st.mean(x["n"] for x in runs)}
    return out

D = {(m, kv, d): load(m, f"{kv}-{d}") for m in ("e2b", "e4b") for kv in ("perslot", "kvu") for d in ("none", "q4x-n2")}
fmt = lambda z: f"{z['tps']:7.1f} ({z['spread']*100:4.1f}%)" if z else f"{'—':>15}"
rat = lambda a, b: f"{a['tps']/b['tps']:6.2f}" if a and b else f"{'—':>6}"

for m in ("e2b", "e4b"):
    for kv in ("perslot", "kvu"):
        a, b = D[(m, kv, "none")], D[(m, kv, "q4x-n2")]
        print(f"\n{m.upper()} {kv}: aggregate output tok/s (spread), drafter/none, acceptance, median TTFT ms, median per-request decode tok/s")
        print(f"{'pass':>4} {'c':>3} {'none':>15} {'drafter':>15} {'ratio':>6} {'acc':>6} {'ttft no':>8} {'ttft dr':>8} {'dec no':>7} {'dec dr':>7}")
        for p in (1, 2):
            for c in LEVELS:
                x, y = a.get((p, c)), b.get((p, c))
                if not x and not y:
                    continue
                acc = f"{y['acc']:6.3f}" if y and y["acc"] is not None else f"{'':>6}"
                t = lambda z, k: f"{z[k]:8.0f}" if z else f"{'—':>8}"
                v = lambda z: f"{z['dec']:7.1f}" if z else f"{'—':>7}"
                print(f"{p:>4} {c:>3} {fmt(x)} {fmt(y)} {rat(y, x)} {acc} {t(x,'ttft')} {t(y,'ttft')} {v(x)} {v(y)}")

print("\nShared / per-slot aggregate output, per pass (p1 p2)")
print(f"{'arm':>14} " + " ".join(f"{'c='+str(c):>11}" for c in LEVELS))
for m in ("e2b", "e4b"):
    for d in ("none", "q4x-n2"):
        a, b = D[(m, "kvu", d)], D[(m, "perslot", d)]
        print(f"{m+' '+d:>14} " + " ".join(f"{rat(a.get((1,c)), b.get((1,c)))}{rat(a.get((2,c)), b.get((2,c)))[1:]}" for c in LEVELS))

print("\nPass 2 / pass 1 aggregate output")
print(f"{'arm':>22} " + " ".join(f"{'c='+str(c):>6}" for c in LEVELS))
for k, v in D.items():
    print(f"{' '.join(k):>22} " + " ".join(rat(v.get((2, c)), v.get((1, c))) for c in LEVELS))
