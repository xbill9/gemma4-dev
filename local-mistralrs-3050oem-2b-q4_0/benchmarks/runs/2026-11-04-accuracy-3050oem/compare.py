"""GSM8K and BFCL simple for mistral.rs v0.9.5 on the RTX 3050 OEM (no drafter, thinking off, one request at a
time) against the other engines on the same card, each one request at a time with no drafter: llama.cpp on the
same exact Q4_0 GGUF (-c 131072; local-llamacpp-3050oem-2b-q4_0/.../2026-10-14-accuracy-fullctx-3050oem),
vLLM 0.33.0 on the W4A16 QAT build (local-vllm-3050oem-2b-q4w4a16emb4/.../2026-11-03-accuracy-3050oem), the int4
PyTorch engine (local-pytorch-3050oem-2b-q4w4a16/.../2026-10-31-accuracy-3050oem) and LiteRT-LM's exact Q4_0
build (GSM8K stock server, BFCL constrained decoding off). Also mistral.rs BFCL at its default (thinking on).
Paired by question, 95% paired bootstrap range (2,000 resamples, seed 0)."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
LC = os.path.join(ROOT, "local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem")
PT = os.path.join(ROOT, "local-pytorch-3050oem-2b-q4w4a16/benchmarks/runs/2026-10-31-accuracy-3050oem")
LG = os.path.join(ROOT, "local-litertlm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-02-accuracy-3050oem")
VL = os.path.join(ROOT, "local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-accuracy-3050oem")
LB = os.path.join(ROOT, "local-litertlm-3050oem-2b-int248/benchmarks/runs/2026-11-03-tools-constraint-3050oem")
ARMS = {
    "gsm8k": {"mistralrs": (HERE, "gsm8k-mistralrs"), "vllm": (VL, "gsm8k-vllm"), "llama.cpp": (LC, "gsm8k-e2b-none"), "pytorch": (PT, "gsm8k-pytorch"),
              "litert": (LG, "gsm8k-q4exact")},
    "bfcl_simple": {"mistralrs": (HERE, "bfcl_simple-mistralrs"),
                    "mistralrs-think": (HERE, "bfcl_simple-mistralrs-thinkdefault"), "vllm": (VL, "bfcl_simple-vllm"), "llama.cpp": (LC, "bfcl_simple-e2b-none"),
                    "pytorch": (PT, "bfcl_simple-pytorch"), "litert": (LB, "bfcl_simple-q4exact-off")},
}


def load(d, s):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(d, s + ".jsonl")))}


def wall(d, s):
    try:
        return json.loads(open(os.path.join(d, s + ".log")).read().strip().splitlines()[-1])["wall_s"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]


for t, arms in ARMS.items():
    rows = {n: load(*p) for n, p in arms.items()}
    walls = {n: wall(*p) for n, p in arms.items()}
    for n, r in rows.items():
        right = sum(bool(x.get("right")) for x in r.values())
        trunc = sum(x.get("finish") == "length" for x in r.values())
        w = walls[n]
        rel = f" ({w / walls['llama.cpp']:.2f}x llama.cpp)" if w and walls["llama.cpp"] else ""
        print(f"{t:12s} {n:10s} n={len(r)} right {right} acc {right / len(r):.4f} truncated {trunc} wall {w} s{rel}")
    for b in [n for n in arms if n != "mistralrs"]:
        x, y = rows["mistralrs"], rows[b]
        ids = sorted(set(x) & set(y))
        d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
        lo, hi = boot(d)
        same = sum((x[i].get("text") or "", json.dumps(x[i].get("calls"), sort_keys=True))
                   == (y[i].get("text") or "", json.dumps(y[i].get("calls"), sort_keys=True)) for i in ids)
        print(f"{t:12s} mistralrs - {b}: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] "
              f"(+{d.count(1)} / -{d.count(-1)}) identical replies {same} of {len(ids)}")
