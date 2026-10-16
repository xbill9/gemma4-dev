"""Aggregate output with one shared KV cache (conc-kvu/) divided by per-slot caches (conc/), per cell."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
def rd(p):
    out, k = {}, None
    for l in open(os.path.join(HERE, p)):
        if l.startswith("=="): k = " ".join(l.split()[1:3]).rstrip(":")
        m = re.match(r"c=\s*(\d+)\s+none\s+([\d.]+)\s+\(\s*([\d.]+)%\).*drafter\s+([\d.]+)\s+\(\s*([\d.]+)%\)", l)
        if m: out[(k, int(m[1]))] = tuple(float(x) for x in m.groups()[1:])
    return out
a, b = rd("conc/summary.txt"), rd("conc-kvu/summary.txt")
print("cell  c  shared/per-slot: none  drafter   spread% per-slot -> shared: none  drafter")
for k in a:
    print(f"{k[0]:<15} {k[1]:>2}  {b[k][0] / a[k][0]:.2f}  {b[k][2] / a[k][2]:.2f}   {a[k][1]:5.1f} -> {b[k][1]:4.1f}  {a[k][3]:5.1f} -> {b[k][3]:4.1f}")
