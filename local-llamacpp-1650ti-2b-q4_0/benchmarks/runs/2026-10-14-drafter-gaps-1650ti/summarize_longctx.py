"""Tables from longctx.py results: decode tok/s per depth and question, x vs no drafter, acceptance."""
import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1])]
arms = list(dict.fromkeys(r["arm"] for r in R))
cell = {(r["arm"], r["depth"], r["question"]): r for r in R}
depths = sorted({r["depth"] for r in R})
qs = list(dict.fromkeys(r["question"] for r in R))
print("decode tok/s (x vs none) [acceptance]")
print("depth  question " + "".join(f"{a:>24s}" for a in arms))
for d in depths:
    for q in qs:
        if ("none", d, q) not in cell: continue
        b = cell[("none", d, q)]["decode_tps"]; row = f"{d:6d} {q:8s} "
        for a in arms:
            r = cell[(a, d, q)]; acc = r["draft_accepted"] / r["draft_n"] if r["draft_n"] else 0
            row += f"{r['decode_tps']:8.1f} ({r['decode_tps']/b:.2f}x) [{acc:.2f}]"
        print(row)
print("\nper depth, mean of x vs none over questions")
for d in depths:
    xs = {a: [cell[(a, d, q)]["decode_tps"] / cell[("none", d, q)]["decode_tps"] for q in qs if ("none", d, q) in cell] for a in arms}
    print(f"{d:6d} " + "  ".join(f"{a} {sum(v)/len(v):.3f}" for a, v in xs.items()))
print("\nreplies identical to none (sha):", {a: sum(cell[(a, d, q)]["sha"] == cell[("none", d, q)]["sha"] for d in depths for q in qs if ("none", d, q) in cell) for a in arms}, "of", sum(1 for k in cell if k[0] == "none"))
print("VRAM after last request:", {a: [r["gpu"] for r in R if r["arm"] == a][-1] for a in arms})
