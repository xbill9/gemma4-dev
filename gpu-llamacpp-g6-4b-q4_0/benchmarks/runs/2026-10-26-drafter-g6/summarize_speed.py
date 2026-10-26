"""Tables from speed.py results: geometric-mean decode tok/s (median of repeats), x vs none,
acceptance, greedy replies identical to none, VRAM."""
import json, math, sys, collections
R = [json.loads(l) for l in open(sys.argv[1])]
for mode in ("greedy", "sampled"):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        if r["mode"] == mode:
            by[r["arm"]][r["prompt"]].append(r)
    gm = lambda d: math.exp(sum(math.log(v) for v in d.values()) / len(d))
    med = {a: {p: sorted(x["decode_tps"] for x in v)[len(v) // 2] for p, v in ps.items()} for a, ps in by.items()}
    base = gm(med["none"])
    print(mode)
    for a, ps in by.items():
        acc = sum(x["draft_accepted"] for v in ps.values() for x in v)
        dn = sum(x["draft_n"] for v in ps.values() for x in v)
        same = sum(x["sha"] == by["none"][x["prompt"]][x["rep"]]["sha"] for v in ps.values() for x in v)
        tot = sum(len(v) for v in ps.values())
        prompts = " ".join(f"{p}={med[a][p] / med['none'][p]:.2f}" for p in ps)
        print(f"  {a:14s} {gm(med[a]):6.1f} {gm(med[a]) / base:.3f}x acc {acc / dn if dn else 0:.3f} "
              f"same-as-none {same}/{tot}  {prompts}  [{list(ps.values())[0][0]['gpu']}]")
