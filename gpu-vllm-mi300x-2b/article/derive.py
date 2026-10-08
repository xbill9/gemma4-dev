"""Every derived figure in the article, computed from evidence/dtype-summary.json and the boot log."""
import json, re, statistics
d = {a["encoding"]: a for a in json.load(open("evidence/dtype-summary.json"))["arms"]}
boot = open("evidence/boot-bf16-memory.txt").read()
d["bf16"]["model_loading_gib"] = float(re.search(r"took ([\d.]+) GiB", boot).group(1))
d["bf16"]["kv_tokens"] = int(re.search(r"KV cache size: ([\d,]+)", boot).group(1).replace(",", ""))
cell = lambda e, c, i: next(x for x in d[e]["cells"] if x["concurrency"] == c and x["input_len"] == i)
order = ["bf16", "fp8", "fp8fnuz", "fp8emb4", "fp8fnuzemb4", "w8a8", "w8a8emb4", "q4w4a16", "q4w4a16ple4", "q4w4a16emb4"]
print("## Per arm: weights, KV, output tok/s at c1/in128, c8/in1024, c64/in1024, ratio range vs bf16\n")
print("| arm | GiB | % bf16 GiB | KV tokens | KV x bf16 | c1/128 | c8/1024 | c64/1024 | ratio min | ratio max | boot s | kernels |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
b = d["bf16"]
for e in order:
    a = d[e]
    print(f"| {e} | {a['model_loading_gib']:.2f} | {100*a['model_loading_gib']/b['model_loading_gib']:.0f}% | {a['kv_tokens']:,} | "
          f"{a['kv_tokens']/b['kv_tokens']:.3f} | {cell(e,1,128)['output_tok_s']:,.0f} | {cell(e,8,1024)['output_tok_s']:,.0f} | "
          f"{cell(e,64,1024)['output_tok_s']:,.0f} | {a['ratio_min']:.2f} | {a['ratio_max']:.2f} | {a['boot_seconds']:.0f} | {', '.join(a['kernels']) or '-'} |")
print("\n## Full grid, output tok/s (ratio to bf16)\n")
cells = [(c, i) for c in (1, 8, 64) for i in (128, 1024, 8192)]
print("| arm | " + " | ".join(f"c{c}/{i}" for c, i in cells) + " |\n|---|" + "---:|" * len(cells))
for e in order:
    print(f"| {e} | " + " | ".join(f"{cell(e,c,i)['output_tok_s']:,.0f} ({cell(e,c,i)['output_ratio_vs_bf16']:.2f})" for c, i in cells) + " |")
def pair(x, y):
    r = [cell(x, c, i)["output_tok_s"] / cell(y, c, i)["output_tok_s"] for c, i in cells]
    rr = [v for (c, i), v in zip(cells, r) if (c, i) != (64, 8192)]
    return f"{x} / {y}: all 9 cells {min(r):.3f}-{max(r):.3f}; 8 cells without c64/8192 {min(rr):.3f}-{max(rr):.3f}, median {statistics.median(r):.3f}"
print("\n## Pairs, per-cell ratio of output tok/s\n")
for x, y in [("fp8fnuz", "fp8"), ("fp8fnuzemb4", "fp8emb4"), ("fp8emb4", "fp8"), ("w8a8emb4", "w8a8"),
             ("q4w4a16emb4", "q4w4a16"), ("q4w4a16ple4", "q4w4a16"), ("fp8", "w8a8"), ("fp8", "q4w4a16"), ("w8a8", "q4w4a16")]:
    print("- " + pair(x, y))
print("\n## bf16 c64/8192 across arms (the noisiest cell)\n")
print(", ".join(f"{e} {cell(e,64,8192)['output_tok_s']:,.0f}" for e in order))
print("\n## TPOT median ms at c1/in128\n")
print(", ".join(f"{e} {cell(e,1,128)['tpot_ms_median']}" for e in order))
w = json.load(open("evidence/weights-w8a8-w8a8_report.json"))["int8_error"]
q = json.load(open("evidence/weights-q4w4a16-verify_report.json"))["quantized"]
vals = sum(x["values"] for x in q.values())
print("\n## Weights against the QAT values\n")
print(f"- q4w4a16: {sum(x['groups'] for x in q.values()):,} groups, {sum(x['groups_level_mismatch'] for x in q.values())} off the grid, "
      f"{100*sum(x['values_bit_identical'] for x in q.values())/vals:.1f}% bit-identical, max rel err {max(x['max_rel_err'] for x in q.values()):.4f}")
print(f"- w8a8: {sum(x['tensors'] for x in w.values())} tensors, rel err {100*min(x['rel_err_vs_qat_min'] for x in w.values()):.2f}%-"
      f"{100*max(x['rel_err_vs_qat_max'] for x in w.values()):.2f}%, mean of means {100*statistics.mean(x['rel_err_vs_qat_mean'] for x in w.values()):.2f}%")
for k in ("fp8", "fp8fnuz"):
    v = json.load(open(f"evidence/weights-{k}-verify_report.json"))
    print(f"- {k}: {v['modules']} modules, rel RMS {100*v['relative_rms_error']:.3f}%, max row-relative {100*v['max_rel_err']:.2f}%")
print("\n## Repeat spread (cv %) per cell, from each arm's report\n")
import os
for e in order:
    rep = json.load(open(os.path.join("..", "..", d[e]["report"])))
    cv = {(c["concurrency"], c["input_len"]): (c.get("raw", {}).get("repeats") or {}).get("cv_pct") for c in rep["throughput"]["sweep"]}
    print(f"- {e}: c64/8192 cv {cv[(64, 8192)]}%, other cells max cv {max(v for k, v in cv.items() if k != (64, 8192) and v is not None)}%")
