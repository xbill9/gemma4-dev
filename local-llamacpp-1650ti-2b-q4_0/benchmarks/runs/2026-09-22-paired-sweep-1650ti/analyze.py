"""Pair the 2026-09-22 ABBA sweep: CPU p1, GPU p1, GPU p2, CPU p2."""
import json
import statistics as st
import sys

R = "/home/xbill/gemma4-dev"
CPU = f"{R}/local-llamacpp-cpu-2b-q4_0/benchmarks/runs/2026-09-22-paired-sweep-cpu"
GPU = f"{R}/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-09-22-paired-sweep-1650ti"


def load(d, p):
    j = json.load(open(f"{d}/pass{p}/sweep.json"))
    out = {}
    for c in j["cells"]:
        if c.get("status") != "ok":
            continue
        runs = c["runs"]
        # tokens over the measured decode span, not chunks (CLAUDE.md: chunks undercount)
        tok = st.median((r["completion_tokens"] - 1) / ((r["wall_s"] * 1000 - r["ttft_ms"]) / 1000) for r in runs)
        spread = (max(r["decode_tps"] for r in runs) - min(r["decode_tps"] for r in runs)) / c["decode_tps_median"]
        out[(c["input_len"], c["output_len"])] = dict(
            dec=c["decode_tps_median"], tok=tok, ttft=c["ttft_ms_median"], e2e=c["end_to_end_tps_median"], spread=spread)
    return j, out


jc1, c1 = load(CPU, 1)
jg1, g1 = load(GPU, 1)
jg2, g2 = load(GPU, 2)
jc2, c2 = load(CPU, 2)
for j in (jc1, jg1, jg2, jc2):
    a = j["attestation"]
    print(a["device"], a["exe_sha256"][:16], "-ngl", a["n_gpu_layers"], "-t", a["threads"], "-tb", a["threads_batch"],
          "cells", j["summary"]["cells_ok"], "/", j["summary"]["cells_total"])
keys = sorted(set(c1) & set(g1) & set(g2) & set(c2))
print("paired cells:", len(keys), "input_lens equal:", all(k in c1 for k in keys))


def m(a, b, f):
    return (a[f] + b[f]) / 2


rows, rd, rp, re, rt = [], [], [], [], []
for k in keys:
    cd, gd = m(c1[k], c2[k], "dec"), m(g1[k], g2[k], "dec")
    ct, gt = m(c1[k], c2[k], "ttft"), m(g1[k], g2[k], "ttft")
    ce, ge = m(c1[k], c2[k], "e2e"), m(g1[k], g2[k], "e2e")
    ctok, gtok = m(c1[k], c2[k], "tok"), m(g1[k], g2[k], "tok")
    rd.append(gd / cd); rp.append(ct / gt); re.append(ge / ce); rt.append(gtok / ctok)
    rows.append((k, cd, gd, gd / cd, ct, gt, ct / gt, ce, ge, ge / ce, ctok, gtok, gtok / ctok))
print("\n| in tok | out tok | CPU decode | GPU decode | **x** | CPU TTFT ms | GPU TTFT ms | **x** | CPU e2e | GPU e2e | **x** |")
print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for r in rows:
    (i, o), cd, gd, xd, ct, gt, xt, ce, ge, xe = r[:10]
    print(f"| {i} | {o} | {cd:.2f} | {gd:.2f} | **{xd:.2f}** | {ct:.0f} | {gt:.0f} | **{xt:.2f}** | {ce:.2f} | {ge:.2f} | **{xe:.2f}** |")
print(f"\nmedian ratios: decode {st.median(rd):.2f} prefill(TTFT) {st.median(rp):.2f} e2e {st.median(re):.2f}"
      f" token-decode {st.median(rt):.2f}")
print(f"ranges: decode {min(rd):.2f}-{max(rd):.2f} prefill {min(rp):.2f}-{max(rp):.2f} e2e {min(re):.2f}-{max(re):.2f}")
print("token-based decode: CPU %.2f-%.2f GPU %.2f-%.2f" % (min(r[10] for r in rows), max(r[10] for r in rows),
                                                          min(r[11] for r in rows), max(r[11] for r in rows)))

print("\n| in | out | CPU p1 | CPU p2 | drift | GPU p1 | GPU p2 | drift | CPU TTFT p1 | p2 | GPU TTFT p1 | p2 |")
print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
dc, dg = [], []
for k in keys:
    a = (c2[k]["dec"] - c1[k]["dec"]) / c1[k]["dec"]; b = (g2[k]["dec"] - g1[k]["dec"]) / g1[k]["dec"]
    dc.append(a); dg.append(b)
    print(f"| {k[0]} | {k[1]} | {c1[k]['dec']:.2f} | {c2[k]['dec']:.2f} | {a:+.1%} | {g1[k]['dec']:.2f} | {g2[k]['dec']:.2f} | {b:+.1%}"
          f" | {c1[k]['ttft']:.0f} | {c2[k]['ttft']:.0f} | {g1[k]['ttft']:.0f} | {g2[k]['ttft']:.0f} |")
print(f"\nCPU pass drift median {st.median(dc):+.1%} range {min(dc):+.1%}..{max(dc):+.1%}; "
      f"GPU {st.median(dg):+.1%} range {min(dg):+.1%}..{max(dg):+.1%}")
for name, d in (("CPU p1", c1), ("CPU p2", c2), ("GPU p1", g1), ("GPU p2", g2)):
    s = [v["spread"] for v in d.values()]
    print(f"{name} within-cell spread max {max(s):.2%} median {st.median(s):.2%}")
# single-pass ratios, to show what a non-ABBA run would have reported
for lab, cc, gg in (("CPU1->GPU1 (the 09-16 order)", c1, g1), ("GPU2->CPU2", c2, g2)):
    print(lab, "decode x", round(st.median(gg[k]["dec"] / cc[k]["dec"] for k in keys), 2),
          "prefill x", round(st.median(cc[k]["ttft"] / gg[k]["ttft"] for k in keys), 2))
