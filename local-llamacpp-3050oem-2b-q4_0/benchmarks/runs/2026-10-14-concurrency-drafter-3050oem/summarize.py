"""Aggregate output tok/s by concurrency, drafter off and on, from each cell's concurrency.json
(sweep.py: output_tokens_per_second is the median over 5 repeats of total tokens / wall time)."""
import json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
for m in ("e2b", "e4b"):
    for shape in ("in512-out128", "in128-out512"):
        rows = {}
        for arm in ("none", "q4x-n2"):
            f = os.path.join(HERE, m, arm, shape, "concurrency.json")
            if os.path.exists(f):
                for lv in json.load(open(f))["concurrency"]:
                    rows.setdefault(lv["concurrency"], {})[arm] = lv
        if not rows:
            continue
        print(f"== {m} {shape}: aggregate output tok/s (spread %), TTFT ms median")
        for c in sorted(rows):
            r = rows[c]
            a, b = r.get("none"), r.get("q4x-n2")
            fmt = lambda x: f"{x['output_tokens_per_second']:8.1f} ({x['output_tps_spread_pct']:4.1f}%) ttft {x['ttft_ms_median']:7.0f}" if x and x.get("status") == "ok" else f"{'—':>34s}"
            ratio = f"{b['output_tokens_per_second'] / a['output_tokens_per_second']:.3f}x" if a and b and a.get("status") == b.get("status") == "ok" else "—"
            print(f"c={c:3d}  none {fmt(a)}  drafter {fmt(b)}  drafter/none {ratio}")
        print()
