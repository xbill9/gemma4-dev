"""Sweep the split decode kernel's launch shape on full-attention layers (D=512) at 32k and 128k tokens held."""
import itertools, json, time, torch
from w4attn_long import decode_attention
torch.manual_seed(0)
L = 131072
K = (torch.randn(L, 512, device="cuda") * 0.3).half(); V = torch.randn(L, 512, device="cuda").half()
q = torch.randn(1, 1, 8, 512, device="cuda").half()
def gtime(f, n=10):
    f(); f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): f()
    g.replay(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / (3 * n) * 1e3
rows = []
for S, bn, w in itertools.product((32, 64, 128, 256), (16, 32, 64), (2, 4, 8)):
    try:
        r = {"splits": S, "bn": bn, "warps": w}
        for p in (32767, 131071):
            pos = torch.tensor([p], device="cuda")
            ms = gtime(lambda: decode_attention(q, K, V, None, pos, False, None, 1.0, splits=S, bn=bn, warps=w))
            r[f"ms_{p + 1}"] = round(ms, 3)
            r[f"GBps_{p + 1}"] = round((p + 1) * 512 * 2 * 2 / ms / 1e6, 1)
        rows.append(r); print(json.dumps(r), flush=True)
    except Exception as e:
        print(json.dumps({"splits": S, "bn": bn, "warps": w, "error": str(e)[:120]}), flush=True)
best = min(rows, key=lambda r: r["ms_131072"])
print(json.dumps({"best": best}))
