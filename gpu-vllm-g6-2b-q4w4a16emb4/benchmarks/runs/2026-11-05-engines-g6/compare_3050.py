"""L4 (this run) against the 3050 flight, arm by arm, both summarized by summarize.py.

python3 compare_3050.py  (from this directory; reads the three 3050 run directories)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
RUNS_3050 = [  # first directory holding an arm wins
    "local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem",
    "local-mistralrs-3050oem-2b-q4_0/benchmarks/runs/2026-11-04-engines-3050oem",
    "local-sglang-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-04-engines-3050oem",
]


def table(d):
    out = subprocess.run([sys.executable, os.path.join(HERE, "summarize.py"), d], capture_output=True, text=True,
                         check=True).stdout.splitlines()[1:]
    rows = {}
    for line in out:
        f = line.split()
        rows[f[0]] = {"e2e": float(f[1]), "four": float(f[5])}
    return rows


g6 = table(HERE)
ref = {}
for d in RUNS_3050:
    for arm, r in table(os.path.join(ROOT, d)).items():
        ref.setdefault(arm, r)
print(f"{'arm':16} {'L4 one':>8} {'3050 one':>9} {'L4/3050':>8} {'L4 four':>8} {'3050 four':>10} {'L4/3050':>8}")
for arm, r in g6.items():
    b = ref.get(arm)
    if b is None:
        print(f"{arm:16} {r['e2e']:8.1f} {'-':>9} {'-':>8} {r['four']:8.1f} {'-':>10} {'-':>8}")
        continue
    print(f"{arm:16} {r['e2e']:8.1f} {b['e2e']:9.1f} {r['e2e'] / b['e2e']:8.3f} "
          f"{r['four']:8.1f} {b['four']:10.1f} {r['four'] / b['four']:8.3f}")
