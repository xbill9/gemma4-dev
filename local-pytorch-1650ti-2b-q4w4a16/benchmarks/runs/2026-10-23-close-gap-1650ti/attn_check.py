"""w4_decode_attention against torch SDPA (math) on random fp16 data, Gemma 4 E2B decode shapes, and its GPU time."""
import json, time, torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from w4attn import w4_decode_attention
torch.manual_seed(0)
res = []
for D in (256, 512):
    for filled in (20, 150, 384):
        H, KVH, L = 8, 1, 384
        q = torch.randn(1, H, 1, D, device="cuda").half()
        k = torch.randn(1, KVH, L, D, device="cuda").half() * 0.3
        v = torch.randn(1, KVH, L, D, device="cuda").half()
        mask = torch.zeros(1, 1, 1, L, dtype=torch.bool, device="cuda"); mask[..., :filled] = True
        with sdpa_kernel(SDPBackend.MATH):
            ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float().expand(1, H, L, D), v.float().expand(1, H, L, D),
                                                                 attn_mask=mask, scale=1.0).transpose(1, 2)
        out, _ = w4_decode_attention(None, q, k, v, mask, scaling=1.0)
        err = float((out.float() - ref).abs().max() / ref.abs().max())
        f = lambda: w4_decode_attention(None, q, k, v, mask, scaling=1.0)
        f(); f(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(50): f()
        g.replay(); torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(5): g.replay()
        torch.cuda.synchronize()
        res.append({"D": D, "filled": filled, "rel_err": float(f"{err:.1e}"), "us_per_call": round((time.perf_counter() - s) / 250 * 1e6, 1)})
        print(json.dumps(res[-1]))
