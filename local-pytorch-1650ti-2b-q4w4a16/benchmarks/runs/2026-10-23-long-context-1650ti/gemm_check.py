"""Prompt-read matmul throughput on the 1650 Ti: [tokens, 1536] x [1536, out] in fp16 vs fp32, GPU-timed."""
import json, time, torch
def gtime(f, n=10):
    f(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
rows = []
for M in (256, 512):
    for O, I in ((24576, 1536), (1536, 12288), (2560, 1536)):
        r = {"tokens": M, "shape": [O, I]}
        for name, dt, red in (("fp16", torch.float16, True), ("fp16_noreduce", torch.float16, False), ("fp32", torch.float32, None), ("tf32_off_fp32", torch.float32, None)):
            x = torch.randn(M, I, device="cuda", dtype=dt); w = torch.randn(O, I, device="cuda", dtype=dt)
            if dt == torch.float16:
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = red
            ms = gtime(lambda: torch.nn.functional.linear(x, w))
            r[name] = {"ms": round(ms, 3), "TFLOPs": round(2 * M * O * I / ms / 1e9, 2)}
        rows.append(r); print(json.dumps(r), flush=True)
