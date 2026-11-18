#!/usr/bin/env python3
"""AIME, repack with drafter: reply wall time against the server's own generation time, per thinking setting."""
import json, ast, os
H = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.normpath(os.path.join(H, "../../../local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem"))
g = {}
for th in ("off", "on"):
    rows = [json.loads(l) for l in open(f"{RUN}/aime2025-repack-q4x-think{th}.jsonl")]
    tm = lambda r: ast.literal_eval(r["timings"]) if isinstance(r["timings"], str) else r["timings"]
    s = sum(r["seconds"] for r in rows); g[th] = sum(tm(r)["predicted_ms"] for r in rows) / 1000
    big = sorted(((r["seconds"] - tm(r)["predicted_ms"] / 1000, r["id"]) for r in rows))[-3:]
    print(f"thinking {th}: reply seconds {s:.0f}, generation seconds {g[th]:.0f}, outside generation {s-g[th]:.0f}, largest {[(round(x), i) for x, i in big]}")
print(f"thinking on / off by generation time: x{g['on']/g['off']:.2f}")
