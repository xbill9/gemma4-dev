"""SGLang against vLLM, llama.cpp and mistral.rs on the RTX 3050 OEM, per concurrency level and shape: aggregate
output tok/s (median of 5 repeats), TTFT and TPOT medians, and SGLang / vLLM and SGLang / llama.cpp.
vLLM is local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-concurrency-3050oem (the same checkpoint
and revision, 64 requests, no drafter); llama.cpp local-llamacpp-3050oem-2b-q4_0/.../2026-11-03-concurrency-3050oem
(7182ef71e, 64 slots x 1024, no drafter); mistral.rs local-mistralrs-3050oem-2b-q4_0/.../
2026-11-04-concurrency-3050oem (4 running requests). Same sweep.py arguments."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
RUNS = {
    "vLLM": ("local-vllm-3050oem-2b-q4w4a16emb4", "2026-11-03-concurrency-3050oem"),
    "llama.cpp": ("local-llamacpp-3050oem-2b-q4_0", "2026-11-03-concurrency-3050oem"),
    "mistral.rs": ("local-mistralrs-3050oem-2b-q4_0", "2026-11-04-concurrency-3050oem"),
}


def cells(path):
    return {c["concurrency"]: c for c in json.load(open(path))["concurrency"] if c.get("status", "ok") == "ok"}


for shape in ("in512-out128", "in128-out512"):
    s = cells(os.path.join(HERE, shape, "concurrency.json"))
    o = {
        k: cells(os.path.join(ROOT, r, "benchmarks", "runs", d, shape, "concurrency.json"))
        for k, (r, d) in RUNS.items()
    }
    print(f"\n{shape}")
    print(
        f"{'c':>3} {'SGLang':>7} {'vLLM':>7} {'llama':>7} {'mistral':>7} {'s/vLLM':>6} {'s/llama':>7} "
        f"{'s TTFT':>7} {'v TTFT':>7} {'s TPOT':>7} {'v TPOT':>7} {'s spread':>8}"
    )
    for c in sorted(s):
        a, v, ll, m = s[c], o["vLLM"][c], o["llama.cpp"][c], o["mistral.rs"][c]
        x = a["output_tokens_per_second"]
        print(
            f"{c:3d} {x:7.1f} {v['output_tokens_per_second']:7.1f} {ll['output_tokens_per_second']:7.1f} "
            f"{m['output_tokens_per_second']:7.1f} {x / v['output_tokens_per_second']:6.2f} "
            f"{x / ll['output_tokens_per_second']:7.2f} {a['ttft_ms_median']:7.0f} {v['ttft_ms_median']:7.0f} "
            f"{a['tpot_ms_median']:7.2f} {v['tpot_ms_median']:7.2f} {a['output_tps_spread_pct']:7.1f}%"
        )
    print(
        f"peak: SGLang {max(c['output_tokens_per_second'] for c in s.values()):.1f}, "
        + ", ".join(f"{k} {max(c['output_tokens_per_second'] for c in d.values()):.1f}" for k, d in o.items())
    )
