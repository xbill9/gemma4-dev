#!/usr/bin/env python3
"""Every ratio and total the article quotes, from the files beside this one (1650 Ti run) and the
3050 run's results-3050.json. Per-problem agreement reads the two runs' .jsonl replies (paths below)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
c = json.load(open(f"{H}/results.json"))["cells"]; b = json.load(open(f"{H}/results-3050.json"))["cells"]
k = json.load(open(f"{H}/tokens.json"))
A = ["repack-none", "repack-q4x", "google-none", "google-q4x"]
TH = ("off", "on")
tot, tot3 = sum(x["wall_h"] for x in c.values()), sum(x["wall_h"] for x in b.values())
print(f"total hours 1650 Ti {tot:.2f}, 3050 {tot3:.2f}, ratio x{tot/tot3:.2f}")
for t in ("gsm8k", "aime2025"):
    for a in A:
        off, on = c[f"{t} {a} off"], c[f"{t} {a} on"]
        print(f"{t} {a}: thinking time x{on['wall_h']/off['wall_h']:.2f}, generated tokens x{on['tokens_mean']/off['tokens_mean']:.2f}, "
              f"score {off['acc']:.4f} -> {on['acc']:.4f} ({on['right']-off['right']:+d}), s/problem off {off['wall_h']*3600/off['n']:.2f} on {on['wall_h']*3600/on['n']:.2f}")
    for w in ("repack", "google"):
        for th in TH:
            n, q = c[f"{t} {w}-none {th}"], c[f"{t} {w}-q4x {th}"]; n3, q3 = b[f"{t} {w}-none {th}"], b[f"{t} {w}-q4x {th}"]
            print(f"{t} {w} {th}: drafter time x{n['wall_h']/q['wall_h']:.2f} sooner (3050 x{n3['wall_h']/q3['wall_h']:.2f}), gen tok/s {n['gen_tok_s']:.1f} -> {q['gen_tok_s']:.1f} "
                  f"(x{q['gen_tok_s']/n['gen_tok_s']:.2f}; 3050 x{q3['gen_tok_s']/n3['gen_tok_s']:.2f}), acceptance {q['accept']:.3f} (3050 {q3['accept']:.3f})")
    for d in ("none", "q4x"):
        for th in TH:
            r, g = c[f"{t} repack-{d} {th}"], c[f"{t} google-{d} {th}"]
            print(f"{t} {d} {th}: repack/google gen tok/s x{r['gen_tok_s']/g['gen_tok_s']:.3f}")
    for a in A:
        for th in TH:
            x, y = c[f"{t} {a} {th}"], b[f"{t} {a} {th}"]
            print(f"{t} {a} {th}: 1650/3050 gen tok/s x{x['gen_tok_s']/y['gen_tok_s']:.2f}, time x{x['wall_h']/y['wall_h']:.2f}, right {x['right']} vs {y['right']} ({x['right']-y['right']:+d})")
for t in ("gsm8k", "aime2025"):
    v = [c[f"{t} {a} {th}"] for a in A for th in TH]
    print(f"{t} 1650 range {min(x['acc'] for x in v):.4f}-{max(x['acc'] for x in v):.4f} right {min(x['right'] for x in v)}-{max(x['right'] for x in v)}, "
          f"hours {min(x['wall_h'] for x in v):.2f}-{max(x['wall_h'] for x in v):.2f}, cut off {min(x['truncated'] for x in v)}-{max(x['truncated'] for x in v)}")
    print(f"{t} gen tok/s no drafter {min(c[f'{t} {a} {th}']['gen_tok_s'] for a in A if 'none' in a for th in TH):.1f}-{max(c[f'{t} {a} {th}']['gen_tok_s'] for a in A if 'none' in a for th in TH):.1f}, "
          f"drafter {min(c[f'{t} {a} {th}']['gen_tok_s'] for a in A if 'q4x' in a for th in TH):.1f}-{max(c[f'{t} {a} {th}']['gen_tok_s'] for a in A if 'q4x' in a for th in TH):.1f}")
ps = json.load(open(f"{H}/results.json"))["tests"]
print("min p over all 24 tests:", min(x["p"] for x in ps)); print("tests below 0.05:", [(x["test"], round(x["p"], 4)) for x in ps if x["p"] < 0.05])
mem = {"repack-none": 3355, "repack-q4x": 3457, "google-none": 3559, "google-q4x": 3661}  # driver.log, server up
mem3 = {"repack-none": 6430, "repack-q4x": 6624, "google-none": 7326, "google-q4x": 7520}  # driver-3050.log, server up
print("memory MiB 1650:", mem, "drafter adds", mem["repack-q4x"] - mem["repack-none"], mem["google-q4x"] - mem["google-none"], "google over repack", mem["google-none"] - mem["repack-none"], "free of 4096 at largest", 4096 - max(mem.values()))
print("memory MiB 3050:", mem3)
print("repack smaller by GB", round((5154941280 - 4215695712) / 1e9, 2))
g = [c[f"gsm8k {a} {th}"] for a in A for th in TH]; g3 = [b[f"gsm8k {a} {th}"] for a in A for th in TH]
print(f"3050 gsm8k range {min(x['acc'] for x in g3):.4f}-{max(x['acc'] for x in g3):.4f}")
d = [c[f"gsm8k {a} {th}"]['right'] - b[f"gsm8k {a} {th}"]['right'] for a in A for th in TH]
d2 = [c[f"aime2025 {a} {th}"]['right'] - b[f"aime2025 {a} {th}"]['right'] for a in A for th in TH]
print("1650 minus 3050 right: gsm8k", min(d), max(d), "aime", min(d2), max(d2))
a3 = [b[f"aime2025 {a} {th}"]['right'] for a in A for th in TH]; print("3050 aime right", min(a3), max(a3))
for d in ("none", "q4x"):
    v = [b[f"gsm8k {w}-{d} {t}"]["gen_tok_s"] for w in ("repack", "google") for t in TH]
    print(f"3050 gsm8k gen tok/s drafter {d}: {min(v):.1f}-{max(v):.1f}")
print("per-layer table GB (Q4_0, 18 bytes per 32 values, [10752, 262144]):", round(10752 * 262144 / 32 * 18 / 1e9, 3))
print("errors 1650:", sorted({x["errors"] for x in c.values()}), "truncated 1650:", sorted({x["truncated"] for x in c.values()}))
print("3050 best gsm8k cell:", max((v["right"], k) for k, v in b.items() if k.startswith("gsm8k")))
print("gsm8k answer tokens on/off:", round(263 / 342, 3), "(tokens.md means)")
import re
ag = [float(m) for m in re.findall(r"McNemar p ([\d.]+)", open(f"{H}/agreement.txt").read())]
print(f"settings within chance across cards (p >= 0.05): {sum(p >= 0.05 for p in ag)} of {len(ag)}")
x = c["gsm8k google-q4x on"]; print(f"google-q4x thinking on: {x['right']:,} of {x['n']:,}")
