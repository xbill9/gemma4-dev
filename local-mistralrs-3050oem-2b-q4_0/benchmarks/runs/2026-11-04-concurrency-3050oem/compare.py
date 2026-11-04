"""mistral.rs against llama.cpp and vLLM on the RTX 3050 OEM, per concurrency level and shape: aggregate
output tok/s (median of 5 repeats), TTFT and TPOT medians, and mistral.rs / llama.cpp and mistral.rs / vLLM.
llama.cpp is local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-03-concurrency-3050oem (7182ef71e, the
same GGUF, 64 slots x 1024, no drafter); vLLM is local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/
2026-11-03-concurrency-3050oem (64 requests, no drafter). Same sweep.py arguments; the day before."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
L = os.path.join(ROOT, "local-llamacpp-3050oem-2b-q4_0", "benchmarks", "runs", "2026-11-03-concurrency-3050oem")
V = os.path.join(ROOT, "local-vllm-3050oem-2b-q4w4a16emb4", "benchmarks", "runs", "2026-11-03-concurrency-3050oem")


def cells(path):
    return {c["concurrency"]: c for c in json.load(open(path))["concurrency"] if c.get("status", "ok") == "ok"}


for shape in ("in512-out128", "in128-out512"):
    m, ll, v = (cells(os.path.join(d, shape, "concurrency.json")) for d in (HERE, L, V))
    print(f"\n{shape}")
    print(f"{'c':>3} {'mistral.rs':>10} {'llama.cpp':>9} {'vLLM':>7} {'m/llama':>7} {'m/vLLM':>6} "
          f"{'m TTFT':>7} {'l TTFT':>7} {'m TPOT':>7} {'l TPOT':>7} {'m spread':>8}")
    for c in sorted(m):
        a, b, w = m[c], ll[c], v[c]
        x = a["output_tokens_per_second"]
        print(f"{c:3d} {x:10.1f} {b['output_tokens_per_second']:9.1f} {w['output_tokens_per_second']:7.1f} "
              f"{x / b['output_tokens_per_second']:7.2f} {x / w['output_tokens_per_second']:6.2f} "
              f"{a['ttft_ms_median']:7.0f} {b['ttft_ms_median']:7.0f} {a['tpot_ms_median']:7.2f} "
              f"{b['tpot_ms_median']:7.2f} {a['output_tps_spread_pct']:7.1f}%")
