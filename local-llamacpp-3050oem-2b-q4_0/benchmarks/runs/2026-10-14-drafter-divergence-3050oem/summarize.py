"""Replies identical to the first no-drafter arm, by server context, cache and prompt length.
Also reports the first differing word position for drafted replies."""
import json
import sys
from collections import defaultdict
R = [json.loads(line) for line in open(sys.argv[1])]
ref = {(r["ctx"], r["cache"], r["depth"], r["question"]): r for r in R if r["arm"] == "none"}
tab = defaultdict(lambda: [0, 0])
first = defaultdict(list)
for r in R:
    if r["arm"] == "none":
        continue
    b = ref[(r["ctx"], r["cache"], r["depth"], r["question"])]
    k = (r["arm"], r["ctx"], r["cache"], r["depth"])
    tab[k][1] += 1
    if r["sha"] == b["sha"]:
        tab[k][0] += 1
    else:
        a, c = r["text"].split(), b["text"].split()
        i = next((i for i, (x, y) in enumerate(zip(a, c)) if x != y), min(len(a), len(c)))
        first[k].append(i)
depths = sorted({r["depth"] for r in R})
print("same as first no-drafter arm (of questions): rows arm/ctx/cache, columns tokens of text ahead")
print(f"{'':28s}" + "".join(f"{d:>8d}" for d in depths))
for arm in ("none-again", "q4x-n2"):
    for ctx in (16384, 131072):
        for cache in (False, True):
            row = f"{arm:11s} c={ctx:<6d} cache={'on ' if cache else 'off'} "
            row += "".join(f"{tab[(arm, ctx, cache, d)][0]:>5d}/{tab[(arm, ctx, cache, d)][1]:<2d}" for d in depths)
            print(row)
print("\nfirst differing word index, drafted replies:", {f"c{k[1]} cache={k[2]} d={k[3]}": v for k, v in first.items() if k[0] == "q4x-n2"})
c16 = {(r["arm"], r["cache"], r["depth"], r["question"]): r["sha"] for r in R if r["ctx"] == 16384}
same = sum(r["sha"] == c16[(r["arm"], r["cache"], r["depth"], r["question"])] for r in R if r["ctx"] == 131072)
print(f"replies identical between -c 16384 and -c 131072 (same arm, cache, prompt): {same} of {sum(1 for r in R if r['ctx'] == 131072)}")
