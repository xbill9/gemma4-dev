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
print("weights file bytes repack 2620370912 google 3349516256: google/repack", round(3349516256/2620370912, 3))
print("repack smaller by GB", round((3349516256-2620370912)/1e9, 2)); print("drafter memory MiB", 3802-3610, 4498-4306)
