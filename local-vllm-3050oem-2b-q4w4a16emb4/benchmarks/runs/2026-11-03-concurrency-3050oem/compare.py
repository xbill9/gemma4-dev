"""vLLM against llama.cpp on the RTX 3050 OEM, per concurrency level and shape: aggregate output
tok/s (median of 5 repeats), TTFT and TPOT medians, and the vLLM / llama.cpp ratio. llama.cpp is
local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-03-concurrency-3050oem, run the same day (same sweep.py
arguments; 7182ef71e, exact Q4_0 GGUF, 64 slots x 1024, no drafter)."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "local-llamacpp-3050oem-2b-q4_0", "benchmarks", "runs",
                                  "2026-11-03-concurrency-3050oem"))


def cells(path):
    return {c["concurrency"]: c for c in json.load(open(path))["concurrency"] if c.get("status", "ok") == "ok"}


for shape in ("in512-out128", "in128-out512"):
    v, ll = cells(os.path.join(HERE, shape, "concurrency.json")), cells(os.path.join(L, shape, "concurrency.json"))
    print(f"\n{shape}")
    print(f"{'c':>3} {'vLLM tok/s':>11} {'llama tok/s':>11} {'ratio':>6} {'vLLM TTFT':>10} {'llama TTFT':>10} "
          f"{'vLLM TPOT':>9} {'llama TPOT':>10} {'vLLM spread':>11}")
    for c in sorted(v):
        a, b = v[c], ll[c]
        r = a["output_tokens_per_second"] / b["output_tokens_per_second"]
        print(f"{c:3d} {a['output_tokens_per_second']:11.1f} {b['output_tokens_per_second']:11.1f} {r:6.2f} "
              f"{a['ttft_ms_median']:10.0f} {b['ttft_ms_median']:10.0f} {a['tpot_ms_median']:9.2f} "
              f"{b['tpot_ms_median']:10.2f} {a['output_tps_spread_pct']:10.1f}%")
