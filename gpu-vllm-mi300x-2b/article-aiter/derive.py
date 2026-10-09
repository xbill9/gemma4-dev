"""Every derived figure in the AITER article, from evidence/."""
import json, collections, re, statistics
def cells(p):
    r = json.load(open(p))
    return {(c["concurrency"], c["input_len"]): c for c in r["throughput"]["sweep"] if c.get("status") == "ok"}
A, S, O = cells("evidence/report-aiter.json"), cells("evidence/report-stock.json"), cells("evidence/report-sweep-2026-10-07.json")
cv = lambda c: (c.get("raw", {}).get("repeats") or {}).get("cv_pct")
print("| requests | prompt tokens | AITER tok/s | stock tok/s | AITER / stock | AITER TPOT ms | stock TPOT ms | stock / 2026-10-07 | cv AITER % | cv stock % |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
r, d = [], []
for k in sorted(A):
    a, s, o = A[k]["output_tok_per_s"], S[k]["output_tok_per_s"], O[k]["output_tok_per_s"]
    r.append(a / s); d.append(s / o)
    print(f"| {k[0]} | {k[1]:,} | {a:,.1f} | {s:,.1f} | {a/s:.3f} | {A[k]['tpot_ms']['median']} | {S[k]['tpot_ms']['median']} | {s/o:.3f} | {cv(A[k])} | {cv(S[k])} |")
print(f"\nAITER/stock: min {min(r):.3f} max {max(r):.3f} median {statistics.median(r):.3f}; cells slower: {sum(x < 1 for x in r)} of {len(r)}")
print(f"loss: {100*(1-max(r)):.1f}% to {100*(1-min(r)):.1f}%")
print(f"stock vs 2026-10-07 sweep: {min(d):.3f}-{max(d):.3f}, max deviation {100*max(abs(1-x) for x in d):.1f}%")
print(f"max repeat cv: AITER {max(cv(c) for c in A.values())}%, stock {max(cv(c) for c in S.values())}%")
sh = [l.strip() for l in open("evidence/aiter-untuned-shapes.txt")]
nk = collections.Counter(re.sub(r"M:\d+, ", "", l) for l in sh)
ms = {int(re.search(r"M:(\d+)", l).group(1)) for l in sh}
print(f"\nuntuned AITER GEMM lines: {len(sh)}; distinct (N, K): {len(nk)}; distinct M per shape: {len(ms)}, M from {min(ms)} to {max(ms)}")
for k, v in nk.items(): print(f"- {k.replace('shape is ', '')}: {v}")
