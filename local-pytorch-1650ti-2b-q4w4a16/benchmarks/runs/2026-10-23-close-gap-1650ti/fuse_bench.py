"""GPU time of separate vs merged int4 GEMVs for E2B's per-layer projection groups (CUDA-graph timed)."""
import json, time, torch
from w4gemv_v2 import w4_gemv
torch.manual_seed(0)
def mk(O, I):
    return (torch.randint(-2**31, 2**31 - 1, (O, I // 8), dtype=torch.int32, device="cuda"),
            (torch.rand(O, I // 32, device="cuda") * 0.02).half())
def t(f, n=50):
    f(); f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): f()
    g.replay(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / (5 * n) * 1e3
x = torch.randn(1, 1, 1536, device="cuda").half()
groups = {  # name: (row counts merged, how many layers per token use it)
    "qkv sliding (2048+256+256)": ([2048, 256, 256], 12),
    "qkv full (4096+512+512)": ([4096, 512, 512], 3),
    "gate+up narrow (6144+6144)": ([6144, 6144], 15),
    "gate+up wide (12288+12288)": ([12288, 12288], 20),
}
total_sep = total_cat = 0.0
for name, (rows, per_tok) in groups.items():
    parts = [mk(o, 1536) for o in rows]
    cat = (torch.cat([p for p, _ in parts]), torch.cat([s for _, s in parts]))
    sep = t(lambda: [w4_gemv(x, p, s) for p, s in parts])
    merged = t(lambda: w4_gemv(x, *cat))
    total_sep += sep * per_tok; total_cat += merged * per_tok
    print(json.dumps({"group": name, "per_token": per_tok, "separate_ms": round(sep, 4), "merged_ms": round(merged, 4)}))
print(json.dumps({"per_token_separate_ms": round(total_sep, 3), "per_token_merged_ms": round(total_cat, 3)}))
