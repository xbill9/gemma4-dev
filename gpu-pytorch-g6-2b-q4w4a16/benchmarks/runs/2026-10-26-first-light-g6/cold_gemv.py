"""DRAM-honest numbers: device copy bandwidth, and the int4 GEMV per engine shape with weights rotated through enough
copies (>= 256 MiB) that L2 (48 MiB on the L4) cannot hold them.

    python3 cold_gemv.py OUT.jsonl
"""

import json
import sys
import time

import torch

sys.path.insert(0, "/opt/torch-g6-q4/app")
import w4gemv  # noqa: E402

fh = open(sys.argv[1], "w")


def emit(**kw):
    fh.write(json.dumps(kw) + "\n")
    fh.flush()
    print(json.dumps(kw), flush=True)


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / reps


a = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
b = torch.empty_like(a)
s = timed(lambda: b.copy_(a), 20)
emit(test="copy_1GiB", read_plus_write_GBps=round(2 * a.numel() / s / 1e9, 1))
s = timed(lambda: a.view(torch.int32).sum(), 20)
emit(test="sum_1GiB", read_GBps=round(a.numel() / s / 1e9, 1))
del a, b

torch.manual_seed(0)
for O, KP in sorted(w4gemv.CONFIGS):
    nbytes = O * KP * 4 + O * KP // 4 * 2
    copies = max(2, (256 << 20) // nbytes)
    ps = [torch.randint(-(2**31), 2**31 - 1, (O, KP), dtype=torch.int32, device="cuda") for _ in range(copies)]
    ss = [(torch.rand(O, KP // 4, device="cuda") * 0.02).half() for _ in range(copies)]
    x = torch.randn(1, 1, KP * 8, device="cuda").half()

    def run():
        for p, sc in zip(ps, ss):
            w4gemv.w4_gemv(x, p, sc)

    per = timed(run, 5) / copies
    emit(shape=[O, KP], copies=copies, us=round(per * 1e6, 2), GBps=round(nbytes / per / 1e9, 1))
    del ps, ss
