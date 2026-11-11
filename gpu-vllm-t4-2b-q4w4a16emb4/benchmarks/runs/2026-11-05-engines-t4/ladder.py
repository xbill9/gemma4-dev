"""T4 load sweeps for every engine arm: output tok/s per level (median of 5 repeats, as sweep.py reports),
then each engine's fastest setting per level. Run from anywhere; reads the run directories under ~/gemma4-dev."""
import json, os
R = os.path.expanduser("~/gemma4-dev")
V = "gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-concurrency-t4/"
L = "gpu-llamacpp-t4-2b-q4_0/benchmarks/runs/2026-11-06-concurrency-t4/"
S = "gpu-sglang-t4-2b-q4_0/benchmarks/runs/2026-11-06-concurrency-t4/"
ARMS = {"vllm": V + "vllm", **{f"vllm-d{k}": V + f"vllm-d{k}" for k in range(1, 5)},
        "llama": L + "llama", **{f"llama-d{k}": L + f"llama-d{k}" for k in range(1, 5)},
        "sglang4": "gpu-sglang-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-10-int4-t4/concurrency/sglang4",
        "sglang-bf16": S + "sglang", "sglang-bf16-d1": S + "sglang-d1",
        "mistralrs": "gpu-mistralrs-t4-2b-q4_0/benchmarks/runs/2026-11-06-concurrency-t4/mistralrs"}
ENGINE = lambda a: a.split("-")[0]
def levels(p):
    try: d = json.load(open(os.path.join(R, p, "concurrency.json")))
    except OSError: return {}
    return {c["concurrency"]: c for c in d["concurrency"] if c["status"] == "ok"}
for shape in ("in128-out512", "in512-out128"):
    D = {a: levels(os.path.join(p, shape)) for a, p in ARMS.items()}
    lv = sorted(set().union(*D.values()))
    print(f"\n{shape}: output tok/s")
    print(f"{'arm':15s}" + "".join(f"{c:>8d}" for c in lv))
    for a, x in D.items():
        print(f"{a:15s}" + "".join(f"{x[c]['output_tokens_per_second']:8.1f}" if c in x else "       -" for c in lv))
    print(f"\n{shape}: each engine's fastest arm per level (sglang = int4 build)")
    eng = ["vllm", "sglang4", "llama", "mistralrs"]
    print(f"{'level':>5s}  " + "".join(f"{e:>22s}" for e in eng))
    for c in lv:
        row = []
        for e in eng:
            cand = [(D[a][c]["output_tokens_per_second"], a) for a in D if ENGINE(a) == ENGINE(e) and c in D[a]
                    and not a.startswith("sglang-bf16")]
            t, a = max(cand) if cand else (None, "-")
            row.append(f"{t:8.1f} ({a})" if t else "-")
        print(f"{c:5d}  " + "".join(f"{x:>22s}" for x in row))
    print(f"\n{shape}: time per output token, ms (median), no drafter")
    for a in ("vllm", "sglang4", "llama", "mistralrs"):
        x = D[a]; print(f"{a:15s}" + "".join(f"{x[c].get('tpot_ms_median', x[c].get('tpot_ms', float('nan'))):8.1f}" if c in x else "       -" for c in lv))
