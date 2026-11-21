"""Roll-up of the three 3050 thinking runs (E2B, E4B, 12B), computed from the saved replies.
Common setting: exact Q4_0 repack, no drafter, greedy (temperature 0), 32,768-token reply limit.
E2B/E4B: f16 cache, -c 131072. 12B: q4_0 cache, -c 81920. 12B GSM8K thinking on stopped at 157 problems."""
import json, os, statistics as st
from math import comb, sqrt
R = "/home/xbill/gemma4-dev"
RUN = {"E2B": f"{R}/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem",
       "E4B": f"{R}/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem",
       "12B": f"{R}/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-17-thinking-12b-3050oem"}
NEW = f"{R}/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-21-drafter-sampling-12b-3050oem"
def load(p):
    if not os.path.exists(p): p += ".partial"
    if not os.path.exists(p): return None
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
def mc(b, c):
    n = b + c; return min(1, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def right(r): return r["right"] is True
def gen_tps(rows):
    t = [r["timings"] for r in rows if r.get("timings")]
    return sum(x["predicted_n"] for x in t) / sum(x["predicted_ms"] for x in t) * 1000 if t else None
def stats(rows, label):
    v = list(rows.values()); k = sum(map(right, v)); lo, hi = wilson(k, len(v))
    tps = gen_tps(v)
    print(f"{label}: {k}/{len(v)} = {k/len(v):.4f} [95% {lo:.4f}-{hi:.4f}], hit limit {sum(r['finish']=='length' for r in v)}, "
          f"{sum(r['seconds'] for r in v)/3600:.2f} h, {sum(r['seconds'] for r in v)/len(v):.2f} s/problem, "
          f"median {st.median(r['completion_tokens'] for r in v)} tokens, gen {tps:.1f} tok/s" if tps else "")
    return v
C = {}
print("== exact Q4_0 repack, no drafter, greedy ==")
for m, d in RUN.items():
    for t in ("gsm8k", "aime2025"):
        for th in ("off", "on"):
            rows = load(f"{d}/{t}-repack-none-think{th}.jsonl")
            C[m, t, th] = rows; stats(rows, f"{m} {t} thinking {th}")
def paired(a, b, la, lb, ids=None):
    ids = ids or [i for i in a if i in b]
    x = sum(right(a[i]) and not right(b[i]) for i in ids); y = sum(right(b[i]) and not right(a[i]) for i in ids)
    print(f"{la} vs {lb} on {len(ids)}: {sum(right(a[i]) for i in ids)} vs {sum(right(b[i]) for i in ids)}; only {la} {x}, only {lb} {y}, p = {mc(x, y):.3g}")
print("\n== paired, same problems ==")
for t in ("gsm8k", "aime2025"):
    paired(C["E2B", t, "off"], C["E4B", t, "off"], f"E2B {t} off", "E4B")
    paired(C["E4B", t, "off"], C["12B", t, "off"], f"E4B {t} off", "12B")
    paired(C["E2B", t, "off"], C["12B", t, "off"], f"E2B {t} off", "12B")
ids157 = list(C["12B", "gsm8k", "on"])
paired(C["E4B", "gsm8k", "on"], C["12B", "gsm8k", "on"], "E4B gsm8k on", "12B", ids157)
paired(C["E2B", "gsm8k", "on"], C["12B", "gsm8k", "on"], "E2B gsm8k on", "12B", ids157)
for m in RUN:
    paired(C[m, "gsm8k", "off"], C[m, "gsm8k", "on"], f"{m} gsm8k off", "on", ids157 if m == "12B" else None)
    paired(C[m, "aime2025", "off"], C[m, "aime2025", "on"], f"{m} aime off", "on")
print("\n== replies that hit the 32,768-token limit, all saved replies of each run ==")
import glob
for m, d in RUN.items():
    n = cut = 0
    for f in glob.glob(f"{d}/*.jsonl") + glob.glob(f"{d}/*.jsonl.partial"):
        for l in open(f):
            r = json.loads(l); n += 1; cut += r.get("finish") == "length"
    print(f"{m}: {cut} of {n}")
print("\n== time ratios, GSM8K thinking off, repack no drafter ==")
h = {m: sum(r["seconds"] for r in C[m, "gsm8k", "off"].values()) for m in RUN}
print(f"E4B/E2B {h['E4B']/h['E2B']:.2f}x, 12B/E4B {h['12B']/h['E4B']:.2f}x, 12B/E2B {h['12B']/h['E2B']:.2f}x")
print("\n== 12B follow-up run (", NEW.split('/')[-1], ") ==")
for name in ("gsm8k-sampled-none-thinkon", "gsm8k-greedy-q4x-thinkoff", "aime2025-greedy-q4x-thinkoff", "gsm8k-greedy-q4x-thinkon", "aime2025-sampled-none-thinkon"):
    rows = load(f"{NEW}/{name}.jsonl")
    if not rows: print(f"{name}: not run yet"); continue
    stats(rows, f"12B {name}{' (in progress)' if not os.path.exists(f'{NEW}/{name}.jsonl') else ''}")
    t = name.split("-")[0]; th = name.split("think")[-1]
    base = C["12B", t, th]
    paired(base, rows, f"12B greedy no-drafter {t} {th}", name, [i for i in rows if i in base])
    if "q4x" in name:
        acc = [r["timings"].get("draft_n_accepted", 0) / r["timings"]["draft_n"] for r in rows.values() if r.get("timings", {}).get("draft_n")]
        ids = [i for i in rows if i in base]
        print(f"  drafter: time {sum(base[i]['seconds'] for i in ids)/sum(rows[i]['seconds'] for i in ids):.2f}x sooner on {len(ids)}, "
              f"gen {gen_tps([base[i] for i in ids]):.1f} -> {gen_tps([rows[i] for i in ids]):.1f} tok/s, acceptance mean {st.mean(acc):.3f}" if acc else "")
