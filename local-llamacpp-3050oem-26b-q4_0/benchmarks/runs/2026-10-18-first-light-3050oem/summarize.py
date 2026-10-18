"""Tables from results.jsonl: decode tok/s (median of 3 greedy), x vs none, acceptance, and
greedy replies that differ from the none arm. python3 summarize.py OUT_DIR"""
import collections, json, math, statistics, sys
R = [json.loads(l) for l in open(sys.argv[1] + "/results.jsonl")]
arms = list(dict.fromkeys(r["arm"] for r in R))
prompts = list(dict.fromkeys(r["prompt"] for r in R))
g = collections.defaultdict(list)
for r in R:
    g[(r["arm"], r["mode"], r["prompt"])].append(r)
def med(a, m, p): return statistics.median(x["decode_tps"] for x in g[(a, m, p)])
def acc(a, m, p):
    n = sum(x["draft_n"] for x in g[(a, m, p)]); return sum(x["draft_accepted"] for x in g[(a, m, p)]) / n if n else 0
print("GREEDY decode tok/s, median of 3 (x vs none) [acceptance]")
print(f"{'':9s}" + "".join(f"{a:>24s}" for a in arms))
for p in prompts:
    print(f"{p:9s}" + "".join(f"{med(a,'greedy',p):8.1f} ({med(a,'greedy',p)/med('none','greedy',p):.2f}x) [{acc(a,'greedy',p):.2f}]" for a in arms))
for mode in ("greedy", "sampled"):
    gm = {a: math.exp(statistics.mean(math.log(med(a, mode, p) / med("none", mode, p)) for p in prompts)) for a in arms}
    tps = {a: math.exp(statistics.mean(math.log(med(a, mode, p)) for p in prompts)) for a in arms}
    pooled = {a: (sum(x["draft_accepted"] for p in prompts for x in g[(a, mode, p)]) /
                  max(1, sum(x["draft_n"] for p in prompts for x in g[(a, mode, p)]))) for a in arms}
    print(f"{mode.upper()} geomean x vs none:", {a: round(v, 3) for a, v in gm.items()})
    print(f"{mode.upper()} geomean tok/s:", {a: round(v, 1) for a, v in tps.items()})
    print(f"{mode.upper()} pooled acceptance:", {a: round(v, 3) for a, v in pooled.items()})
spread = {a: round(max((max(x["decode_tps"] for x in g[(a, "greedy", p)]) / min(x["decode_tps"] for x in g[(a, "greedy", p)]) - 1) * 100 for p in prompts), 1) for a in arms}
print("max within-cell spread %:", spread)
ref = {(r["prompt"], r["rep"]): r["sha"] for r in R if r["arm"] == "none" and r["mode"] == "greedy"}
for a in arms[1:]:
    rs = [r for r in R if r["arm"] == a and r["mode"] == "greedy"]
    print(f"{a} greedy replies differing from none: {sum(r['sha'] != ref[(r['prompt'], r['rep'])] for r in rs)} of {len(rs)}")
