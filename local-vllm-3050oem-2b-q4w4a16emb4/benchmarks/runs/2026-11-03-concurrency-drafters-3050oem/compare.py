"""Aggregate output tok/s (median of 5 repeats) per concurrency level and shape, with and without drafters, on
both engines the same day: no drafter from each rig's 2026-11-03-concurrency-3050oem, drafter arms from each
rig's 2026-11-03-concurrency-drafters-3050oem (vLLM W4A16 drafter at 1-4 tokens, llama.cpp exact Q4_0 at 1-2).
Prints each arm's ratio to its own engine without a drafter, then the best of each engine and vLLM / llama.cpp.
vLLM acceptance per shape from its spec-decode counters (metrics after - before)."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
VR = os.path.join(ROOT, "local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs")
LR = os.path.join(ROOT, "local-llamacpp-3050oem-2b-q4_0/benchmarks/runs")
ARMS = {
    "vllm": [("none", os.path.join(VR, "2026-11-03-concurrency-3050oem"))]
    + [(f"w4a16-n{n}", os.path.join(VR, "2026-11-03-concurrency-drafters-3050oem", f"w4a16-n{n}")) for n in (1, 2, 3, 4)],
    "llama": [("none", os.path.join(LR, "2026-11-03-concurrency-3050oem"))]
    + [(f"q4x-n{n}", os.path.join(LR, "2026-11-03-concurrency-drafters-3050oem", f"q4x-n{n}")) for n in (1, 2)],
}


def cells(d, shape):
    f = os.path.join(d, shape, "concurrency.json")
    if not os.path.exists(f):
        return {}
    return {c["concurrency"]: c for c in json.load(open(f))["concurrency"] if c.get("status", "ok") == "ok"}


def counters(f):
    out = {}
    for line in open(f):
        if line.startswith("vllm:spec_decode_num_") and "_total" in line:
            name, val = line.rsplit(" ", 1)
            k = name.split("{")[0]
            out[k] = out.get(k, 0.0) + float(val)
    return out


for shape in ("in512-out128", "in128-out512"):
    data = {e: {a: cells(d, shape) for a, d in arms} for e, arms in ARMS.items()}
    levels = sorted(data["vllm"]["none"])
    print(f"\n{shape}: output tok/s (ratio to the same engine without a drafter)")
    hdr = [f"{e}:{a}" for e in data for a in data[e]]
    print(f"{'c':>3} " + " ".join(f"{h:>20s}" for h in hdr) + f" {'best vllm/llama':>16s}")
    for c in levels:
        row, best = [], {}
        for e in data:
            base = data[e]["none"].get(c, {}).get("output_tokens_per_second")
            for a, cs in data[e].items():
                v = cs.get(c, {}).get("output_tokens_per_second")
                if v and base:
                    row.append(f"{v:9.1f} ({v / base:5.3f})")
                    if v > best.get(e, (0, ""))[0]:
                        best[e] = (v, a)
                else:
                    row.append(f"{'-':>20s}")
        ratio = (f"{best['vllm'][0] / best['llama'][0]:5.2f} {best['vllm'][1]}/{best['llama'][1]}"
                 if "vllm" in best and "llama" in best else "-")
        print(f"{c:3d} " + " ".join(f"{x:>20s}" for x in row) + f" {ratio:>16s}")
    for a, d in ARMS["vllm"][1:]:
        b, f = os.path.join(d, f"metrics-before-{shape}.prom"), os.path.join(d, f"metrics-after-{shape}.prom")
        if os.path.exists(b) and os.path.exists(f):
            x, y = counters(b), counters(f)
            dn = y["vllm:spec_decode_num_draft_tokens_total"] - x["vllm:spec_decode_num_draft_tokens_total"]
            da = y["vllm:spec_decode_num_accepted_tokens_total"] - x["vllm:spec_decode_num_accepted_tokens_total"]
            print(f"  vllm {a} acceptance {da / dn:.3f} ({da:.0f} / {dn:.0f})")
