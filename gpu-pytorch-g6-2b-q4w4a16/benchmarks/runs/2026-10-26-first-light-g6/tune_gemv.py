"""Autotune the int4 GEMV on this GPU for every shape the engine runs at decode (w4gemv.CONFIGS' keys).

    python3 tune_gemv.py OUT.jsonl

For each (rows, packed cols): GPU time per call (50 calls in one CUDA graph, as in kbench.py) for the pinned v5
configuration and for v4's autotuned best, and the configuration v4 chose.
"""

import json
import sys
import time

import torch

sys.path.insert(0, "/opt/torch-g6-q4/app")
import w4gemv  # noqa: E402
import w4gemv_v4  # noqa: E402

fh = open(sys.argv[1], "w")
SHIFTS = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)


def ref(x, packed, scale):
    o, g = scale.shape
    q = ((packed.unsqueeze(-1) >> SHIFTS) & 15).view(o, g, 32).float() - 8
    return torch.nn.functional.linear(x.float(), (q * scale.float().unsqueeze(-1)).view(o, g * 32))


def t(f, n=50):
    f()
    f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            f()
    g.replay()
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / (5 * n) * 1e3


torch.manual_seed(0)
for O, KP in sorted(w4gemv.CONFIGS):
    packed = torch.randint(-(2**31), 2**31 - 1, (O, KP), dtype=torch.int32, device="cuda")
    scale = (torch.rand(O, KP // 4, device="cuda") * 0.02).half()
    x = torch.randn(1, 1, KP * 8, device="cuda").half()
    r = ref(x, packed, scale)
    nbytes = packed.numel() * 4 + scale.numel() * 2
    row = {"shape": [O, KP], "device": torch.cuda.get_device_name(0)}
    for name, fn in (("v5", w4gemv.w4_gemv), ("v4", w4gemv_v4.w4_gemv)):
        err = float((fn(x, packed, scale).float() - r).abs().max() / r.abs().max())
        ms = t(lambda fn=fn: fn(x, packed, scale))
        row[name] = {"ms": round(ms, 4), "GBps": round(nbytes / ms / 1e6, 1), "rel_err": float(f"{err:.1e}")}
    for obj in vars(w4gemv_v4).values():
        if hasattr(obj, "cache") and isinstance(obj.cache, dict):
            for k, v in obj.cache.items():
                if tuple(k[:2]) == (O, KP):
                    row["v4_best"] = str(v)
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    print(json.dumps(row), flush=True)
