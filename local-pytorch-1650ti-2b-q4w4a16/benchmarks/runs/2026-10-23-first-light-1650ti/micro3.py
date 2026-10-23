import json, time, torch
from w4gemv import w4_gemv
torch.manual_seed(0)
SHIFTS = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)
def deq_mm(x, packed, scale):
    o, g = scale.shape
    q = ((packed.unsqueeze(-1) >> SHIFTS) & 15).view(o, g, 32).to(x.dtype) - 8
    return torch.nn.functional.linear(x, (q * scale.unsqueeze(-1)).view(o, g * 32))
compiled = torch.compile(deq_mm, dynamic=False)
def t(f, n=100):
    f(); f(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
res = []
for O, I in [(2048, 1536), (6144, 1536), (12288, 1536), (1536, 6144), (1536, 12288), (4096, 1536)]:
    packed = torch.randint(-2**31, 2**31 - 1, (O, I // 8), dtype=torch.int32, device="cuda")
    scale = (torch.rand(O, I // 32, device="cuda") * 0.02).half()
    x = torch.randn(1, 1, I, device="cuda").half()
    ref = deq_mm(x.float(), packed, scale.float())
    y = w4_gemv(x, packed, scale)
    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    ms_t, ms_c = t(lambda: w4_gemv(x, packed, scale)), t(lambda: compiled(x, packed, scale))
    gbps = (packed.numel() * 4 + scale.numel() * 2) / ms_t / 1e6
    res.append({"shape": [O, I], "rel_err": float(f"{err:.2e}"), "triton_ms": round(ms_t, 3), "compiled_ms": round(ms_c, 3),
                "speedup": round(ms_c / ms_t, 2), "triton_GBps_read": round(gbps, 1)})
for r in res: print(json.dumps(r))
