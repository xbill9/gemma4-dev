"""Aggregate output tok/s per arm and concurrency (mean of the repeats), x against none, median
per-request decode tok/s and acceptance. python3 summarize_conc.py OUT_DIR"""
import collections, json, statistics, sys
C = [json.loads(l) for l in open(sys.argv[1] + "/cells.jsonl")]
R = [json.loads(l) for l in open(sys.argv[1] + "/results.jsonl")]
arms = list(dict.fromkeys(c["arm"] for c in C)); concs = sorted({c["conc"] for c in C})
agg = collections.defaultdict(list); acc = collections.defaultdict(lambda: [0, 0]); dec = collections.defaultdict(list)
for c in C:
    agg[(c["arm"], c["conc"])].append(c["out_tps"])
    acc[(c["arm"], c["conc"])][0] += c["draft_accepted"]; acc[(c["arm"], c["conc"])][1] += c["draft_n"]
for r in R:
    if r["decode_tps"]:
        dec[(r["arm"], r["conc"])].append(r["decode_tps"])
print("aggregate output tok/s, mean of repeats (x vs none) [acceptance] {median per-request decode tok/s} <rep spread %>")
print(f"{'arm':14s}" + "".join(f"{'c=' + str(c):>40s}" for c in concs))
for a in arms:
    row = f"{a:14s}"
    for c in concs:
        m = statistics.mean(agg[(a, c)]); base = statistics.mean(agg[("none", c)]) if ("none", c) in agg else m
        sp = (max(agg[(a, c)]) / min(agg[(a, c)]) - 1) * 100
        ac = acc[(a, c)][0] / acc[(a, c)][1] if acc[(a, c)][1] else 0
        row += f"{m:9.1f} ({m / base:.2f}x) [{ac:.2f}] {{{statistics.median(dec[(a, c)]):6.1f}}} <{sp:3.1f}>"
    print(row)
