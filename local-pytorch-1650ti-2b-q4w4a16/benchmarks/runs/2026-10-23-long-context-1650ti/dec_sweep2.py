"""Decode kernel variants (fp16 dot / fp32 dot / broadcast-sum) on full layers, D=512, at 32k and 128k held."""
import json, time, torch
from w4attn_long import decode_attention
torch.manual_seed(0)
L = 131072
K = (torch.randn(L, 512, device="cuda") * 0.3).half(); V = torch.randn(L, 512, device="cuda").half()
q = torch.randn(1, 1, 8, 512, device="cuda").half()
def ref(p):
    s = q[0, 0].float() @ K[: p + 1].float().T
    return (s.softmax(-1) @ V[: p + 1].float())
def gtime(f, n=10):
    f(); f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): f()
    g.replay(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / (3 * n) * 1e3
for mode, hp, S, bn, w in [(m, hp, S, bn, w) for m, hp in ((1, 16), (2, 8)) for S in (32, 128) for bn in (4, 8, 16) for w in (4, 8)]:
    r = {"mode": mode, "hp": hp, "splits": S, "bn": bn, "warps": w}
    try:
        pos = torch.tensor([131071], device="cuda")
        o = decode_attention(q, K, V, None, pos, False, None, 1.0, splits=S, bn=bn, warps=w, mode=mode, hp=hp)
        rr = ref(131071)
        r["rel_err"] = float(f"{float((o[0, 0].float() - rr).abs().max() / rr.abs().max()):.1e}")
        for p in (32767, 131071):
            pos = torch.tensor([p], device="cuda")
            ms = gtime(lambda: decode_attention(q, K, V, None, pos, False, None, 1.0, splits=S, bn=bn, warps=w, mode=mode, hp=hp))
            r[f"ms_{p + 1}"] = round(ms, 3); r[f"GBps_{p + 1}"] = round((p + 1) * 2048 / ms / 1e6, 1)
    except Exception as e:
        r["error"] = str(e)[:100]
    print(json.dumps(r), flush=True)
