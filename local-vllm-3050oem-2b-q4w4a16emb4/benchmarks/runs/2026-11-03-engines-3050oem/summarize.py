"""Per arm of results.jsonl (engines.py): geometric mean over the six prompts of each prompt's median, for
end-to-end tok/s (non-streaming, all engines), stream decode tok/s and TTFT; the median aggregate tok/s of four
requests at once and its ratio to one request end-to-end; GPU memory after warm-up; greedy repeatability
(prompts whose three e2e replies are identical) and ratio to llama.cpp without a drafter."""

import sys
import json
import math
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(x) for x in open(os.path.join(sys.argv[1] if len(sys.argv) > 1 else HERE, "results.jsonl"))]
seen, uniq = set(), []
for r in rows:  # engines.py printed and wrote each row once; guard against a doubled line
    k = (r["arm"], r["kind"], r.get("prompt"), r["rep"])
    if k not in seen:
        seen.add(k)
        uniq.append(r)
by = defaultdict(list)
for r in uniq:
    by[r["arm"]].append(r)


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v and all(x > 0 for x in v) else None


def per_prompt(rs, kind, key):
    d = defaultdict(list)
    for r in rs:
        if r["kind"] == kind:
            d[r["prompt"]].append(r[key])
    return {p: statistics.median(v) for p, v in d.items()}


table = {}
for arm, rs in by.items():
    e2e = gm(list(per_prompt(rs, "e2e", "tps").values()))
    dec = gm(list(per_prompt(rs, "stream", "tps").values()))
    ttft = gm(list(per_prompt(rs, "stream", "ttft_s").values()))
    b4 = [r["tps"] for r in rs if r["kind"] == "batch4"]
    shas = defaultdict(set)
    for r in rs:
        if r["kind"] == "e2e":
            shas[r["prompt"]].add(r["sha"])
    table[arm] = dict(e2e=e2e, decode=dec, ttft_ms=ttft * 1000 if ttft else None,
                      batch4=statistics.median(b4) if b4 else None, mem=rs[0]["gpu"].split(",")[2].strip(),
                      ready_s=rs[0]["ready_s"], repeat=sum(len(s) == 1 for s in shas.values()), n=len(shas))
ref = table.get("llama", {}).get("e2e")
print(f"{'arm':16s} {'e2e tok/s':>9s} {'vs llama':>8s} {'decode':>7s} {'TTFT ms':>7s} {'4 at once':>9s} "
      f"{'x one':>6s} {'memory':>9s} {'ready s':>7s} {'repeat':>6s}")
for arm, t in table.items():
    f = lambda v, p=1: f"{v:.{p}f}" if v else "-"  # noqa: E731
    print(f"{arm:16s} {f(t['e2e']):>9s} {f(t['e2e'] / ref if ref and t['e2e'] else None, 3):>8s} {f(t['decode']):>7s} "
          f"{f(t['ttft_ms'], 0):>7s} {f(t['batch4']):>9s} {f(t['batch4'] / t['e2e'] if t['batch4'] else None, 2):>6s} "
          f"{t['mem']:>9s} {t['ready_s']:7.1f} {t['repeat']:>3d}/{t['n']}")
