"""Tables from kv_sweep.py's results.jsonl and fit.jsonl: per arm and depth, generation tok/s
(geometric mean over the questions, with min-max), prompt-reading t/s for the first question at
each depth (the only one that reads the whole document), and drafted tokens kept."""
import json, math, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(line) for line in open(os.path.join(HERE, "results.jsonl"))]
fits = [json.loads(line) for line in open(os.path.join(HERE, "fit.jsonl"))]
arms = list(dict.fromkeys(r["arm"] for r in rows))
depths = sorted({r["depth"] for r in rows})

print("## Fit\n")
for f in fits:
    print(f"{f['arm']:10s} -c {f['ctx']:>6d} {'loads' if f['loaded'] else 'fails'}  {f['gpu']}")

print("\n## Generation tok/s by tokens already in context (geomean of questions, min-max)\n")
print("| arm | max -c | " + " | ".join(str(d) for d in depths) + " |")
print("| --- | ---: | " + " | ".join("---:" for _ in depths) + " |")
g = collections.defaultdict(list)
for r in rows:
    g[(r["arm"], r["depth"])].append(r)
for a in arms:
    ctx = next(r["ctx"] for r in rows if r["arm"] == a)
    cells = []
    for d in depths:
        v = [r["decode_tps"] for r in g.get((a, d), [])]
        cells.append(f"{math.exp(sum(map(math.log, v)) / len(v)):.1f} ({min(v):.1f}-{max(v):.1f})" if v else "-")
    print(f"| {a} | {ctx} | " + " | ".join(cells) + " |")

print("\n## Prompt reading t/s (first question at each depth)\n")
for a in arms:
    cells = []
    for d in depths:
        v = [r for r in g.get((a, d), []) if r["prompt_new"] > 1000]
        cells.append(f"{d}:{v[0]['prompt_tps']:.0f}" if v else "")
    print(f"{a:10s} " + "  ".join(c for c in cells if c))

print("\n## Drafted tokens kept (all questions at each depth)\n")
for a in arms:
    cells = []
    for d in depths:
        v = g.get((a, d), [])
        n = sum(r["draft_n"] for r in v)
        if n:
            cells.append(f"{d}:{sum(r['draft_accepted'] for r in v) / n:.3f}")
    if cells:
        print(f"{a:10s} " + "  ".join(cells))
