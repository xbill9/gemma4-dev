"""Paired comparisons of this run's cells against wave 3 (../2026-11-17-thinking-12b-3050oem), same problems."""
import json, math, os
W3 = "../2026-11-17-thinking-12b-3050oem"
def load(p):
    if not os.path.exists(p): p += ".partial"
    return {r["id"]: r for r in map(json.loads, open(p))}
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def pair(label, a, b):
    ids = [i for i in a if i in b]
    ra = sum(a[i]["right"] for i in ids); rb = sum(b[i]["right"] for i in ids)
    only_a = sum(a[i]["right"] and not b[i]["right"] for i in ids); only_b = sum(b[i]["right"] and not a[i]["right"] for i in ids)
    ta = sum(a[i]["finish"] == "length" for i in ids); tb = sum(b[i]["finish"] == "length" for i in ids)
    sa = sum(a[i]["seconds"] for i in ids) / 3600; sb = sum(b[i]["seconds"] for i in ids) / 3600
    print(f"{label}: n={len(ids)}  A {ra} right ({ta} cut off, {sa:.2f} h)  B {rb} right ({tb} cut off, {sb:.2f} h)  "
          f"A only {only_a}, B only {only_b}, McNemar p={mcnemar(only_a, only_b):.4g}  B/A time {sb/sa:.2f}x")
cells = {
  "gsm off greedy": load(f"{W3}/gsm8k-repack-none-thinkoff.jsonl"),
  "gsm on greedy": load(f"{W3}/gsm8k-repack-none-thinkon.jsonl"),
  "aime off greedy": load(f"{W3}/aime2025-repack-none-thinkoff.jsonl"),
  "aime on greedy": load(f"{W3}/aime2025-repack-none-thinkon.jsonl"),
  "gsm on sampled": load("gsm8k-sampled-none-thinkon.jsonl"),
  "aime on sampled": load("aime2025-sampled-none-thinkon.jsonl"),
  "gsm off greedy drafter": load("gsm8k-greedy-q4x-thinkoff.jsonl"),
  "aime off greedy drafter": load("aime2025-greedy-q4x-thinkoff.jsonl"),
  "gsm on greedy drafter": load("gsm8k-greedy-q4x-thinkon.jsonl"),
}
for a, b in [("gsm off greedy", "gsm on sampled"), ("gsm on greedy", "gsm on sampled"),
             ("aime off greedy", "aime on sampled"), ("aime on greedy", "aime on sampled"),
             ("gsm off greedy", "gsm off greedy drafter"), ("aime off greedy", "aime off greedy drafter"),
             ("gsm on greedy", "gsm on greedy drafter")]:
    pair(f"A={a} | B={b}", cells[a], cells[b])
for k in ("gsm on sampled", "aime on sampled"):
    toks = sorted(r["completion_tokens"] for r in cells[k].values())
    print(f"{k}: reply tokens median {toks[len(toks)//2]}, max {toks[-1]}, cut off {sum(r['finish']=='length' for r in cells[k].values())}")
