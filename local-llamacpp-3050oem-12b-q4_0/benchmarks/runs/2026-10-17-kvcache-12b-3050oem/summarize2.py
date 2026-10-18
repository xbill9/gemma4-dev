"""Tables from tune.py (tune.jsonl, tune-fit.jsonl) and recall.py (recall.jsonl). Tuning arms are
printed beside kv_sweep.py's q4_0 rows (-ub 512, 2 drafted tokens) at the same depths: generation
tok/s as the geometric mean over the questions, prompt reading for the first question at each depth,
drafted tokens kept. Recall: codes returned right out of 80 per depth (8 per trial, 10 trials)."""

import collections, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(name):
    return [json.loads(line) for line in open(os.path.join(HERE, name))]


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v))


print("## Tuning: fit\n")
for f in rows("tune-fit.jsonl"):
    print(f"{f['arm']:16s} -ub {f['ub']:>4d} -c {f['ctx']:>6d} {'loads' if f['loaded'] else 'fails'}  {f['gpu']}")

base = [dict(r, arm=r["arm"] + " (sweep)") for r in rows("results.jsonl") if r["ctk"] == "q4_0"]
tune = rows("tune.jsonl")
g = collections.defaultdict(list)
for r in base + tune:
    g[(r["arm"], r["depth"])].append(r)
depths = sorted({r["depth"] for r in tune})
arms = list(dict.fromkeys(r["arm"] for r in base + tune))

print("\n## Tuning: generation tok/s (geomean) / prompt t/s / drafted kept, by depth\n")
print("| arm | -c | " + " | ".join(str(d) for d in depths) + " |")
print("| --- | ---: | " + " | ".join("---" for _ in depths) + " |")
for a in arms:
    ctx = next(r["ctx"] for r in base + tune if r["arm"] == a)
    cells = []
    for d in depths:
        v = g.get((a, d), [])
        if not v:
            cells.append("-")
            continue
        p = [r["prompt_tps"] for r in v if r["prompt_new"] > 1000]
        dn, da = sum(r["draft_n"] for r in v), sum(r["draft_accepted"] for r in v)
        cell = f"{gm([r['decode_tps'] for r in v]):.1f}"
        cell += f" / pp {p[0]:.0f}" if p else ""
        cell += f" / kept {da / dn:.2f}" if dn else ""
        cells.append(cell)
    print(f"| {a} | {ctx} | " + " | ".join(cells) + " |")

print("\n## Recall: codes right (8 per trial, 10 trials) and generation tok/s\n")
print("| arm | -c | depth | right | of | trials all 8 | tok/s (geomean) |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
rg = collections.defaultdict(list)
for r in rows("recall.jsonl"):
    rg[(r["arm"], r["ctx"], r["depth"])].append(r)
for (a, c, d), v in rg.items():
    print(
        f"| {a} | {c} | {d} | {sum(r['right'] for r in v)} | {sum(r['of'] for r in v)} | "
        f"{sum(r['right'] == r['of'] for r in v)} / {len(v)} | {gm([r['decode_tps'] for r in v]):.1f} |"
    )
