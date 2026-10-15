"""Distinct greedy texts per arm and prompt over 10 repeats, and where repeats first differ."""
import json
import sys
from collections import defaultdict
R = [json.loads(line) for line in open(sys.argv[1])]
by = defaultdict(list)
for r in R:
    by[(r["arm"], r["prompt"])].append(r)
for (arm, p), rs in by.items():
    shas = [r["sha"] for r in rs]
    base = rs[0]["text"].split()
    firsts = []
    for r in rs[1:]:
        w = r["text"].split()
        if w != base:
            firsts.append(next((i for i, (a, b) in enumerate(zip(w, base)) if a != b), min(len(w), len(base))))
    print(f"{arm:18s} {p:6s} distinct {len(set(shas)):2d} of {len(shas)}  sequence {''.join(chr(65 + sorted(set(shas), key=shas.index).index(s)) for s in shas)}  first-diff-word {sorted(firsts)}  tok/s {sum(r['decode_tps'] for r in rs) / len(rs):.1f}")
