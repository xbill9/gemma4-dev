#!/usr/bin/env python3
"""Every ratio and total the article quotes, from results.json and tokens.json beside this file."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
c = json.load(open(f"{H}/results.json"))["cells"]; k = json.load(open(f"{H}/tokens.json"))
A = ["repack-none", "repack-q4x", "google-none", "google-q4x"]
print("total hours, 16 cells:", round(sum(x["wall_h"] for x in c.values()), 2))
for t in ("gsm8k", "aime2025"):
    for a in A:
        off, on = c[f"{t} {a} off"], c[f"{t} {a} on"]
        print(f"{t} {a}: thinking time x{on['wall_h']/off['wall_h']:.2f}, generated tokens x{on['tokens_mean']/off['tokens_mean']:.2f}, "
              f"score {off['acc']:.4f} -> {on['acc']:.4f} ({on['right']-off['right']:+d} problems), "
              f"seconds per problem off {off['wall_h']*3600/off['n']:.2f} on {on['wall_h']*3600/on['n']:.2f}")
    for w in ("repack", "google"):
        for th in ("off", "on"):
            n, q = c[f"{t} {w}-none {th}"], c[f"{t} {w}-q4x {th}"]
            print(f"{t} {w} thinking {th}: drafter time x{n['wall_h']/q['wall_h']:.2f} sooner, gen tok/s {n['gen_tok_s']:.1f} -> {q['gen_tok_s']:.1f} (x{q['gen_tok_s']/n['gen_tok_s']:.2f}), acceptance {q['accept']:.3f}")
    for d in ("none", "q4x"):
        for th in ("off", "on"):
            r, g = c[f"{t} repack-{d} {th}"], c[f"{t} google-{d} {th}"]
            print(f"{t} drafter {d} thinking {th}: repack/google gen tok/s x{r['gen_tok_s']/g['gen_tok_s']:.3f}, time google/repack x{g['wall_h']/r['wall_h']:.3f}")
g = [c[f"gsm8k {a} {th}"]["acc"] for a in A for th in ("off", "on")]
print(f"gsm8k score range all 8 cells: {min(g):.4f} to {max(g):.4f}")
print(f"gsm8k fastest cell {min(c[f'gsm8k {a} {th}']['wall_h'] for a in A for th in ('off','on')):.2f} h, slowest {max(c[f'gsm8k {a} {th}']['wall_h'] for a in A for th in ('off','on')):.2f} h")
R, G = 4215695712, 5154941280  # ls -l in host.txt
print(f"weights file bytes repack {R} google {G}: google/repack", round(G/R, 3))
print("repack smaller by GB", round((G-R)/1e9, 2)); print("drafter memory MiB (server up lines in driver.log)", 6624-6430, 7520-7326, "; google-repack MiB", 7326-6430)
# E4B against E2B, the same cell on the same problems (exact McNemar), and the multiple-test odds
from scipy.stats import binomtest
R4 = os.path.normpath(os.path.join(H, "../../../local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem"))
R2 = os.path.normpath(os.path.join(H, "../../../local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem"))
def rows(d, f): return {r["id"]: r for r in map(json.loads, open(f"{d}/{f}.jsonl"))}
print("chance at least one of 4 null tests has p<0.05:", round(1 - 0.95**4, 3))
c2 = json.load(open(f"{R2}/results.json"))["cells"]
for t in ("gsm8k", "aime2025"):
    for a in A:
        for th in ("off", "on"):
            f = f"{t}-{a}-think{th}"; x, y = rows(R2, f), rows(R4, f)
            b = sum(x[i]["right"] and not y[i]["right"] for i in x); cc = sum(y[i]["right"] and not x[i]["right"] for i in x)
            p = binomtest(min(b, cc), b + cc, 0.5).pvalue if b + cc else 1.0
            k2 = c2[f"{t} {a} {th}"]; k4 = c[f"{t} {a} {th}"]
            print(f"E2B->E4B {t} {a} {th}: right {k2['right']}->{k4['right']} only-E2B {b} only-E4B {cc} p {p:.2e} time x{k4['wall_h']/k2['wall_h']:.2f} gen tok/s {k2['gen_tok_s']:.1f}->{k4['gen_tok_s']:.1f}")
sp, ln = [], []
for key, v in c.items():
    sp.append(v["gen_tok_s"] / c2[key]["gen_tok_s"]); ln.append((key, v["tokens_mean"] / c2[key]["tokens_mean"]))
print(f"E4B/E2B generation speed, 16 cells: {min(sp):.2f} to {max(sp):.2f}")
print("E4B/E2B mean generated tokens per reply:", ", ".join(f"{k} x{r:.2f}" for k, r in ln))
tk = json.load(open(f"{H}/tokens.json")) if os.path.exists(f"{H}/tokens.json") else None
import re
ctl = re.search(r"Control\n\n(.*)", open(f"{H}/tokens.md").read()).group(1)
print("token count control:", ctl)
for th in ("off", "on"):
    nums = [(int(d), int(n)) for d, n in re.findall(rf"thinking {th}: (\d+) on (\d+) replies", ctl)]
    print(f"thinking {th}: {nums[0][1]} of {sum(n for _, n in nums)} replies at the fixed offset")
ps = []
for t in ("gsm8k",):
    for a in A:
        for th in ("off", "on"):
            f = f"{t}-{a}-think{th}"; x, y = rows(R2, f), rows(R4, f)
            b = sum(x[i]["right"] and not y[i]["right"] for i in x); cc = sum(y[i]["right"] and not x[i]["right"] for i in x)
            ps.append(binomtest(min(b, cc), b + cc, 0.5).pvalue)
print(f"E2B->E4B GSM8K largest p over 8 cells: {max(ps):.5f}")
