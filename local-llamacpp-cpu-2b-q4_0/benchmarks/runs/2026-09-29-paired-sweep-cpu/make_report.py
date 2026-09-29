"""Build one arm's schema-1.1 report from an ABBA run's two passes.

    python3 make_report.py RUN_DIR TEMPLATE_JSON OUT_JSON

Per cell: throughput, decode and TTFT/TPOT medians are the mean of the two passes'
medians; p99 TTFT and raw min/max span all repeats of both passes; spread_pct is
(max - min) / per-stream mean. Everything outside `throughput.sweep` is copied from
the template (the caller edits notes, model and memory). Verified by regenerating
2026-09-22's sweep entries from its own passes.
"""
import json
import statistics as st
import sys

run, tmpl, out = sys.argv[1:4]
passes = [json.load(open(f"{run}/pass{p}/sweep.json")) for p in (1, 2)]
rep = json.load(open(tmpl))


def cells(j):
    return {(c["input_len"], c["output_len"]): c for c in j["cells"] if c.get("status") == "ok"}


a, b = (cells(j) for j in passes)
sweep = []
for k in sorted(a):
    ca, cb = a[k], b[k]
    runs = ca["runs"] + cb["runs"]
    dec = [r["decode_tps"] for r in runs]
    ttft = sorted(r["ttft_ms"] for r in runs)
    per = (ca["decode_tps_median"] + cb["decode_tps_median"]) / 2
    sweep.append({
        "concurrency": 1, "input_len": k[0], "output_len": k[1], "status": "ok",
        "output_tok_per_s": round((ca["end_to_end_tps_median"] + cb["end_to_end_tps_median"]) / 2, 3),
        "per_stream_tok_per_s": round(per, 3),
        "ttft_ms": {"median": round((ca["ttft_ms_median"] + cb["ttft_ms_median"]) / 2, 3),
                    "p99": round(ttft[min(len(ttft) - 1, round(0.99 * (len(ttft) - 1)))], 3)},
        "tpot_ms": {"median": round((ca["tpot_ms_median"] + cb["tpot_ms_median"]) / 2, 4)},
        "raw": {"repeats": len(runs), "spread_pct": round((max(dec) - min(dec)) / per * 100, 2),
                "min": round(min(dec), 3), "max": round(max(dec), 3)},
    })
rep["throughput"]["sweep"] = sweep
json.dump(rep, open(out, "w"), indent=2)
print(out, len(sweep), "cells")
