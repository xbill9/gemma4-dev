"""Wave 3 (12B) as it stands: repack, no drafter. GSM8K thinking on stopped at the rows in its .partial."""
import json, statistics as st
from math import comb
def load(f): return {json.loads(l)["id"]: json.loads(l) for l in open(f)}
def mc(b, c):
    n = b + c; return min(1, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1
C = {"gsm8k off": "gsm8k-repack-none-thinkoff.jsonl", "gsm8k on (stopped)": "gsm8k-repack-none-thinkon.jsonl.partial",
     "aime2025 off": "aime2025-repack-none-thinkoff.jsonl", "aime2025 on": "aime2025-repack-none-thinkon.jsonl"}
R = {k: load(v) for k, v in C.items()}
print("| Cell | Problems | Right | Hit 32,768 limit | Median s | Total h |\n|---|---|---|---|---|---|")
for k, r in R.items():
    v = list(r.values())
    print(f"| {k} | {len(v)} | {sum(x['right'] is True for x in v)} | {sum(x['finish']=='length' for x in v)} | "
          f"{st.median(x['seconds'] for x in v):.1f} | {sum(x['seconds'] for x in v)/3600:.2f} |")
for t, on in [("gsm8k", "gsm8k on (stopped)"), ("aime2025", "aime2025 on")]:
    off = R[f"{t} off"]; o = R[on]; ids = list(o)
    b = sum(off[i]["right"] is True and o[i]["right"] is not True for i in ids)
    c = sum(o[i]["right"] is True and off[i]["right"] is not True for i in ids)
    so = sum(off[i]["seconds"] for i in ids); sn = sum(o[i]["seconds"] for i in ids)
    print(f"\n{t}, same {len(ids)} problems: off right {sum(off[i]['right'] is True for i in ids)}, on right {sum(o[i]['right'] is True for i in ids)}; "
          f"off only {b}, on only {c}, exact McNemar p = {mc(b, c):.3g}; on took {sn/so:.2f}x the time")
v = list(R["gsm8k on (stopped)"].values())
fin = [x for x in v if x["finish"] != "length"]
print(f"\ngsm8k on: replies that finished on their own {len(fin)}, right {sum(x['right'] is True for x in fin)}")
