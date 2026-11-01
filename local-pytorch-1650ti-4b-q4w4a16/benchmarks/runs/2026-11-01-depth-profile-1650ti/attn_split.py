"""One full-attention layer's decode read at 32,824 keys, no model: 8 query heads on one K/V, the same 8 split 4+4 on the
same K/V (E2B data, E4B's call pattern), and 4+4 on two K/V pairs (E4B). Random fp16, S=64, kernel as deployed."""
import json, sys, time
import torch
sys.path.insert(0, "/home/xbill/gemma4-dev/local-pytorch-1650ti-4b-q4w4a16")
from w4attn import decode_attention
N, D, CAP = 32824, 512, 40960
torch.manual_seed(0)
mk = lambda: torch.randn(CAP, D, device=0, dtype=torch.float16) * 0.3
k1, v1, k2, v2 = mk(), mk(), mk(), mk()
pos = torch.tensor([N], device=0)
q = torch.randn(1, 1, 8, D, device=0, dtype=torch.float16) * 0.3
sc = D ** -0.5
def one(): return decode_attention(q, k1, v1, None, pos, False, None, sc)
def split_same():
    return torch.cat([decode_attention(q[:, :, :4], k1, v1, None, pos, False, None, sc), decode_attention(q[:, :, 4:], k1, v1, None, pos, False, None, sc)], 2)
def split_two():
    return torch.cat([decode_attention(q[:, :, :4], k1, v1, None, pos, False, None, sc), decode_attention(q[:, :, 4:], k2, v2, None, pos, False, None, sc)], 2)
def bench(f, n=200):
    for _ in range(20): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return round((time.perf_counter() - t) / n * 1e3, 3)
res = {}
with torch.inference_mode():
    a, b = one(), split_same()
    res["max_abs_diff_8_vs_4+4"] = float((a.float() - b.float()).abs().max()); res["out_abs_max"] = float(a.float().abs().max())
    for r in range(3):
        for name, f in (("h8_one_kv", one), ("h4h4_same_kv", split_same), ("h4h4_two_kv", split_two)):
            res.setdefault(name, []).append(bench(f))
print(json.dumps(res)); json.dump(res, open("attn_split.json", "w"), indent=1)
