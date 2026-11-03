"""Summarize results.jsonl (drafters.py) per engine and arm: geometric mean over the six prompts of each
prompt's median greedy decode tok/s, ratio to the same engine's first `none` arm, per-prompt ratios,
the sampled pass, acceptance (accepted / drafted tokens over every request of the arm), mean accepted
tokens per drafted step where known, and how many greedy prompts give the same text as `none`."""

import json
import math
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(x) for x in open(os.path.join(HERE, "results.jsonl"))]
if len(sys.argv) > 1:
    rows += [json.loads(x) for f in sys.argv[1:] for x in open(f)]


def gmean(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v and all(x > 0 for x in v) else float("nan")


by = defaultdict(list)
for r in rows:
    by[(r["engine"], r["arm"])].append(r)

for engine in sorted({e for e, _ in by}):
    arms = [a for e, a in by if e == engine]
    base = by[(engine, "none")]

    def per_prompt(rs, mode="greedy", key="decode_tps"):
        d = defaultdict(list)
        for r in rs:
            if r["mode"] == mode and r.get(key):
                d[r["prompt"]].append(r[key])
        return {p: statistics.median(v) for p, v in d.items()}

    def texts(rs):
        d = defaultdict(set)
        for r in rs:
            if r["mode"] == "greedy":
                d[r["prompt"]].add(r["sha"])
        return d

    b_pp, b_txt = per_prompt(base), texts(base)
    b_g = gmean(list(b_pp.values()))
    b_s = gmean(list(per_prompt(base, "sampled").values()))
    prompts = sorted(b_pp)
    print(f"== {engine}  (ratios to `none`, greedy geometric mean {b_g:.1f} tok/s)")
    print(f"{'arm':14s} {'greedy':>7s} {'ratio':>6s} {'sampled':>8s} {'ratio':>6s} {'server':>7s} {'accept':>7s} "
          f"{'same/6':>6s} {'repeat':>6s}  " + " ".join(f"{p:>7s}" for p in prompts))
    for arm in arms:
        rs = by[(engine, arm)]
        pp = per_prompt(rs)
        g = gmean(list(pp.values()))
        s = gmean(list(per_prompt(rs, "sampled").values()))
        srv = gmean(list(per_prompt(rs, key="server_tps").values())) if any(r.get("server_tps") for r in rs) else None
        dn = sum(r["draft_n"] or 0 for r in rs)
        da = sum(r["draft_accepted"] or 0 for r in rs)
        txt = texts(rs)
        same = sum(txt[p] == b_txt[p] for p in prompts)
        repeat = sum(len(txt[p]) == 1 for p in prompts)
        acc = f"{da / dn:.3f}" if dn else "-"
        srv_s = f"{srv:.1f}" if srv else "-"
        print(f"{arm:14s} {g:7.1f} {g / b_g:6.3f} {s:8.1f} {s / b_s:6.3f} {srv_s:>7s} {acc:>7s} {same:>6d} {repeat:>6d}  "
              + " ".join(f"{pp[p] / b_pp[p]:7.3f}" for p in prompts))
    print()
