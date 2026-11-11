"""T4 against the L4 (EC2 G6, 2026-11-05) for one request, end-to-end tok/s, same engines.py prompts.
SGLang on the T4 is the int4 build from 2026-11-10-int4-t4; LiteRT is Google's int248 file on both."""
import os
R = os.path.expanduser("~/gemma4-dev")
def table(p):
    out = {}
    for line in open(os.path.join(R, p)).read().splitlines()[1:]:
        f = line.split()
        if f: out[f[0]] = float(f[1])
    return out
T4 = table("gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-05-engines-t4/SUMMARY.txt")
T4.update({k.replace("sglang4", "sglang"): v for k, v in
           table("gpu-sglang-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-10-int4-t4/SUMMARY.txt").items() if k.startswith("sglang4")})
L4 = table("gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-05-engines-g6/SUMMARY.txt")
print(f"{'arm':15s} {'T4':>7s} {'L4':>7s} {'T4/L4':>6s}")
for a in ("vllm", "sglang", "mistralrs", "pytorch", "llama", "litert-int248", "vllm-d4", "llama-d2", "sglang-d4", "litert-int248-d"):
    if a in T4 and a in L4: print(f"{a:15s} {T4[a]:7.1f} {L4[a]:7.1f} {T4[a]/L4[a]:6.3f}")
