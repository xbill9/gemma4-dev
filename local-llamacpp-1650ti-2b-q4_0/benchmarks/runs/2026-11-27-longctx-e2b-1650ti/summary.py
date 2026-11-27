"""Tables for the E2B long-context run (longctx.jsonl), plus the 12B's recall test at the same depths."""
import json, math, random, re, statistics as st
src = open("longctx.py").read()
ns = {"random": random, "re": re}
exec(src[src.index("PROJECTS ="):src.index("TRIALS, N")], ns)
N = 8
def truth(task, depth, trial):
    rng = random.Random(f"{task}-{depth}-{trial}")
    names = rng.sample(ns["PROJECTS"], N)
    if task == "recall":
        return {n: f"{rng.randrange(100000, 1000000)}" for n in names}
    people, cities = rng.sample(ns["PEOPLE"], N), rng.sample(ns["CITIES"], N)
    return dict(zip(names, cities))
def per_fact(r):
    pat = (r'"?(?:[Pp]roject )?([A-Z][a-z]+)"?\s*:\s*"?(\d{6})' if r["task"] == "recall"
           else r'"(?:[Pp]roject )?([A-Z][a-z]+)"\s*:\s*"(?:the )?([A-Z][a-z]+)')
    got = dict(re.findall(pat, r["reply"])); t = truth(r["task"], r["depth"], r["trial"])
    return [got.get(n) == v for n, v in t.items()]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
R = [json.loads(l) for l in open("longctx.jsonl")]
for r in R:
    f = per_fact(r); assert sum(f) == r["right"], (r["arm"], r["task"], r["depth"], r["trial"])
    r["facts"] = f
ARMS = ["f16-draft", "q8-draft", "q4-draft", "f16-sampled-off", "f16-sampled-on"]
DEPTHS = sorted({r["depth"] for r in R})
idx = {(r["arm"], r["task"], r["depth"], r["trial"]): r for r in R}
print("Facts right of 80 per cell (10 trials x 8), Wilson 95% range")
for task in ("recall", "link"):
    print(f"\n{task}:  " + "  ".join(f"{d:>16}" for d in DEPTHS))
    for a in ARMS:
        row = []
        for d in DEPTHS:
            rs = [r for r in R if r["arm"] == a and r["task"] == task and r["depth"] == d]
            if not rs: row.append(f"{'-':>16}"); continue
            k = sum(r["right"] for r in rs); lo, hi = wilson(k, 80)
            row.append(f"{k:>3} ({lo:.2f}-{hi:.2f})")
        print(f"  {a:<16}" + "  ".join(f"{x:>16}" for x in row))
print("\nPaired against f16-draft, per fact (only that arm right / only f16-draft right, exact McNemar p)")
for a in ARMS[1:]:
    for task in ("recall", "link"):
        cells = []
        for d in DEPTHS:
            b = c = 0; seen = False
            for t in range(10):
                x, y = idx.get((a, task, d, t)), idx.get(("f16-draft", task, d, t))
                if not x or not y: continue
                seen = True
                for fx, fy in zip(x["facts"], y["facts"]):
                    b += fx and not fy; c += fy and not fx
            if seen: cells.append(f"{d}: {b}/{c} p={mcnemar(b, c):.3g}")
        print(f"  {a} {task}: " + "; ".join(cells))
for a in ARMS[1:]:
    b = c = 0
    for k, y in idx.items():
        if k[0] != "f16-draft": continue
        x = idx.get((a,) + k[1:])
        if not x: continue
        for fx, fy in zip(x["facts"], y["facts"]):
            b += fx and not fy; c += fy and not fx
    print(f"  {a} all depths and tasks it shares: {b}/{c} p={mcnemar(b, c):.3g}")
print("\nTime to read the prompt (s, median of 20) / reading speed (tok/s) / generation (tok/s, median), per arm")
for a in ARMS:
    print(f"  {a}")
    for d in DEPTHS:
        rs = [r for r in R if r["arm"] == a and r["depth"] == d]
        if not rs: continue
        print(f"    {d:>6}: read {st.median(r['prompt_ms'] for r in rs)/1000:6.1f} s  {st.median(r['prompt_tps'] for r in rs):6.0f} tok/s"
              f"  generate {st.median(r['decode_tps'] for r in rs):6.1f} tok/s  reply {st.median(r['completion_n'] for r in rs):5.0f} tok"
              f"  cut off {sum(r['finish'] == 'length' for r in rs)}  wall {st.median(r['wall_s'] for r in rs):6.1f} s")
print("\nGPU memory and power at the last request of each arm (nvidia-smi: temp, clock, memory, power)")
for a in ARMS:
    print(f"  {a}: {[r for r in R if r['arm'] == a][-1]['gpu']}")
print("\nTotal facts right per arm, depths 16,384-98,304 (the depths all five arms share)")
for a in ARMS:
    for task in ("recall", "link"):
        rs = [r for r in R if r["arm"] == a and r["task"] == task and r["depth"] <= 98304]
        k = sum(r["right"] for r in rs); n = 8 * len(rs); lo, hi = wilson(k, n)
        print(f"  {a:<16} {task:<6} {k:>3}/{n} = {k/n:.3f} ({lo:.3f}-{hi:.3f})")
print("\n12B (2026-10-17, q4_0 cache, greedy, prompt cache on) recall at the depths it shares")
T = "../../../../local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem/recall.jsonl"
B = [json.loads(l) for l in open(T)]
for arm in sorted({r["arm"] for r in B}):
    for d in sorted({r["depth"] for r in B if r["arm"] == arm}):
        rs = [r for r in B if r["arm"] == arm and r["depth"] == d]
        k = sum(r["right"] for r in rs)
        if d in DEPTHS:
            e = sum(r["right"] for r in R if r["arm"] == "f16-draft" and r["task"] == "recall" and r["depth"] == d)
            print(f"  12B {arm} {d}: {k}/{len(rs)*8}   E2B f16-draft {e}/80")
