"""Every derived figure in the cross-size article, from evidence/dtype-summary.<size>.json."""
import json, re, statistics
SIZES = ["2b", "4b", "12b", "26b", "31b"]
NAME = {"2b": "E2B", "4b": "E4B", "12b": "12B", "26b": "26B-A4B", "31b": "31B"}
D = {s: {a["encoding"]: a for a in json.load(open(f"evidence/dtype-summary.{s}.json"))["arms"]} for s in SIZES}
boot = open("evidence/boot-bf16-memory.2b.txt").read()
D["2b"]["bf16"]["model_loading_gib"] = float(re.search(r"took ([\d.]+) GiB", boot).group(1))
D["2b"]["bf16"]["kv_tokens"] = int(re.search(r"KV cache size: ([\d,]+)", boot).group(1).replace(",", ""))
CELLS = [(c, i) for c in (1, 8, 64) for i in (128, 1024, 8192)]
def cell(s, e, c, i): return next(x for x in D[s][e]["cells"] if x["concurrency"] == c and x["input_len"] == i)
def tok(s, e, c, i): return cell(s, e, c, i)["output_tok_s"]
def ratios(s, e, base="bf16"): return [tok(s, e, c, i) / tok(s, base, c, i) for c, i in CELLS]

print("## bf16 per size\n\n| size | GiB | KV tokens | c1/128 | c8/1024 | c64/1024 | TPOT c1/128 ms |\n|---|---:|---:|---:|---:|---:|---:|")
for s in SIZES:
    b = D[s]["bf16"]
    print(f"| {NAME[s]} | {b['model_loading_gib']:.2f} | {b['kv_tokens']:,} | {tok(s,'bf16',1,128):,.0f} | {tok(s,'bf16',8,1024):,.0f} | {tok(s,'bf16',64,1024):,.0f} | {cell(s,'bf16',1,128)['tpot_ms_median']} |")

print("\n## fp8 and q4w4a16 against bf16 per size\n\n| size | fp8 c1/128 x | fp8 min | fp8 max | fp8 cells >= 1.00 | fp8 median | q4 c1/128 x | q4 min | q4 max | w8a8 min | w8a8 max |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for s in SIZES:
    f, q = ratios(s, "fp8"), ratios(s, "q4w4a16")
    w = ratios(s, "w8a8") if "w8a8" in D[s] else None
    print(f"| {NAME[s]} | {f[0]:.2f} | {min(f):.2f} | {max(f):.2f} | {sum(x >= 1.0 for x in f)} of 9 | {statistics.median(f):.2f} | {q[0]:.2f} | {min(q):.2f} | {max(q):.2f} | "
          + (f"{min(w):.2f} | {max(w):.2f} |" if w else "- | - |"))

print("\n## Every ok arm per size: weights, KV, c1/128, c8/1024, c64/1024, ratio range, weights as % of bf16, KV x bf16\n")
print("| size | arm | GiB | % bf16 | KV tokens | KV x bf16 | c1/128 | c8/1024 | c64/1024 | min | max | kernels |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for s in SIZES:
    b = D[s]["bf16"]
    for e, a in D[s].items():
        if a["status"] != "ok":
            print(f"| {NAME[s]} | {e} | failed | | | | | | | | | |"); continue
        r = ratios(s, e)
        print(f"| {NAME[s]} | {e} | {a['model_loading_gib']:.2f} | {100*a['model_loading_gib']/b['model_loading_gib']:.0f}% | {a['kv_tokens']:,} | {a['kv_tokens']/b['kv_tokens']:.2f} | "
              f"{tok(s,e,1,128):,.0f} | {tok(s,e,8,1024):,.0f} | {tok(s,e,64,1024):,.0f} | {min(r):.2f} | {max(r):.2f} | {', '.join(a['kernels']) or '-'} |")

print("\n## fp8 / bf16 per cell, every size\n")
print("| size | " + " | ".join(f"c{c}/{i}" for c, i in CELLS) + " |\n|---|" + "---:|" * 9)
for s in SIZES:
    print(f"| {NAME[s]} | " + " | ".join(f"{x:.2f}" for x in ratios(s, "fp8")) + " |")

print("\n## emb4 against the same format with bf16 tables, per-cell ratio range\n")
for s in SIZES:
    for x, y in [("fp8emb4", "fp8"), ("q4w4a16emb4", "q4w4a16"), ("w8a8emb4", "w8a8")]:
        if x in D[s] and D[s][x]["status"] == "ok" and y in D[s]:
            r = [tok(s, x, c, i) / tok(s, y, c, i) for c, i in CELLS]
            print(f"- {NAME[s]} {x}/{y}: {min(r):.3f}-{max(r):.3f}, median {statistics.median(r):.3f}")

print("\n## Speed per GiB loaded at one request (c1/128 tok/s per GiB of weights)\n")
for s in SIZES:
    print(f"- {NAME[s]}: " + ", ".join(f"{e} {tok(s,e,1,128)/D[s][e]['model_loading_gib']:.1f}" for e in ("bf16", "fp8", "q4w4a16")))

print("\n## Boot seconds per arm\n")
for s in SIZES:
    print(f"- {NAME[s]}: " + ", ".join(f"{e} {a.get('boot_seconds')}" for e, a in D[s].items()))

print("\n## Repeat spread (cv %) per size, every ok arm and cell\n")
import os
for s in SIZES:
    cvs = []
    for e, a in D[s].items():
        if a["status"] != "ok": continue
        rep = json.load(open(os.path.join("..", "..", a["report"])))
        for c in rep["throughput"]["sweep"]:
            v = (c.get("raw", {}).get("repeats") or {}).get("cv_pct")
            if v is not None: cvs.append((v, e, c["concurrency"], c["input_len"]))
    cvs.sort(reverse=True)
    top = cvs[0]
    over5 = sum(v > 5 for v, *_ in cvs)
    print(f"- {NAME[s]}: max cv {top[0]}% ({top[1]} c{top[2]}/{top[3]}), cells over 5%: {over5} of {len(cvs)}")

print("\n## q4w4a16 / bf16 by request count (mean over prompt lengths) and int4-table savings\n")
for s in SIZES:
    by = {c: statistics.mean(tok(s,"q4w4a16",c,i)/tok(s,"bf16",c,i) for i in (128,1024,8192)) for c in (1,8,64)}
    byi = {i: statistics.mean(tok(s,"q4w4a16",c,i)/tok(s,"bf16",c,i) for c in (1,8,64)) for i in (128,1024,8192)}
    print(f"- {NAME[s]}: by requests " + ", ".join(f"{c}: {v:.2f}" for c,v in by.items()) + " | by prompt " + ", ".join(f"{i}: {v:.2f}" for i,v in byi.items()))
for s in ("4b","12b"):
    print(f"- {NAME[s]} fp8 - fp8emb4 = {D[s]['fp8']['model_loading_gib']-D[s]['fp8emb4']['model_loading_gib']:.2f} GiB")
w_vs_f = [(s, c, i) for s in SIZES if "w8a8" in D[s] and D[s]["w8a8"]["status"]=="ok" for c, i in CELLS if tok(s,"w8a8",c,i) > tok(s,"fp8",c,i)]
print(f"- cells where w8a8 beats fp8: {w_vs_f}")
