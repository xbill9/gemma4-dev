"""T4 load sweeps: output tok/s per concurrency level (median of 5 repeats, as sweep.py reports), SGLang int4
against bf16 SGLang, vLLM and llama.cpp from 2026-11-06-concurrency-t4, no drafters."""
import json, os
R = os.path.expanduser("~/gemma4-dev")
ARMS = {"sglang4": "gpu-sglang-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-10-int4-t4/concurrency/sglang4",
        "sglang bf16": "gpu-sglang-t4-2b-q4_0/benchmarks/runs/2026-11-06-concurrency-t4/sglang",
        "vllm": "gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-concurrency-t4/vllm",
        "llama": "gpu-llamacpp-t4-2b-q4_0/benchmarks/runs/2026-11-06-concurrency-t4/llama"}
def levels(p):
    try: d = json.load(open(os.path.join(R, p, "concurrency.json")))
    except OSError: return {}
    return {c["concurrency"]: c["output_tokens_per_second"] for c in d["concurrency"] if c["status"] == "ok"}
for shape in ("in512-out128", "in128-out512"):
    L = {a: levels(os.path.join(p, shape)) for a, p in ARMS.items()}
    lv = sorted(set().union(*L.values()))
    print(f"\n{shape}: output tok/s")
    print(f"{'arm':12s}" + "".join(f"{c:>8d}" for c in lv))
    for a, x in L.items():
        print(f"{a:12s}" + "".join(f"{x[c]:8.1f}" if c in x else "       -" for c in lv))
    for b in ("sglang bf16", "vllm", "llama"):
        print(f"{'sglang4/' + b:12s}"[:22].ljust(12) + "".join(
            f"{L['sglang4'][c] / L[b][c]:8.2f}" if c in L['sglang4'] and c in L[b] else "       -" for c in lv))
