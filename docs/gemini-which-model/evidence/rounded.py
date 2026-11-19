#!/usr/bin/env python3
"""Rounded figures the article quotes, read from the evidence files and rounded here.

derived.txt (pooled and AIME II per-problem costs) and sources/gemini-vs-gemma-roadmap.txt plus
sources/neon-accuracy-compare.txt (cost per 1,000 short problems). Writes rounded.txt.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
out = []
d = open(os.path.join(HERE, "derived.txt")).read()
ii, pooled = d.split("== AIME 2025 I + II pooled")[0], d.split("== AIME 2025 I + II pooled")[1].split("==")[0]


def per_problem(block, name, col):
    line = next(l for l in block.splitlines() if l.strip().startswith(name + " "))
    vals = re.findall(r"\$\s*(\d+\.\d+)", line)  # cost, per problem, per right, Dec per problem
    return float(vals[col])


for label, block, name, col in [
    ("pooled lowest, per problem", pooled, "neon-4-low", 1),
    ("pooled lowest, December per problem", pooled, "carbon-4-low", 3),
    ("AIME II high, per problem", ii, "neon-4-high", 1),
    ("AIME II high, per problem", ii, "carbon-4-high", 1),
    ("AIME II high, December per problem", ii, "carbon-4-high", 3),
]:
    v = per_problem(block, name, col)
    out.append(f"{label:38s} {name:16s} {v:.4f} -> ${v:.3f}")
r = open(os.path.join(HERE, "sources", "gemini-vs-gemma-roadmap.txt")).read()
for name in ("3.5 Flash-Lite", "3.8 Flash", "Gemini 4 Hydrogen", "Gemini 4 Carbon", "Gemini 4 Argon", "Gemini 4 Neon"):
    for t in ("gsm8k", "bfcl_simple"):
        m = re.search(rf"{t}\s+{re.escape(name)}\s+\S+\s+per 1k \$\s*([\d.]+)", r)
        v = float(m.group(1))
        out.append(f"{t:12s} per 1,000 {name:18s} {v:.3f} -> ${v:.2f}")
open(os.path.join(HERE, "rounded.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
