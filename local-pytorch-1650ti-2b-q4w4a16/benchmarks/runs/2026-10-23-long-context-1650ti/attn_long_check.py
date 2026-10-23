"""decode_attention and prefill_attention against an fp32 reference; decode GPU time per call vs context held."""
import json, time, torch
from w4attn_long import decode_attention, prefill_attention
torch.manual_seed(0)
H, W = 8, 512

def ref(q, K, V, mask):  # q [n,H,D], K/V [L,D], mask [n,L] bool
    s = torch.einsum("nhd,ld->nhl", q.float(), K.float())
    s = s.masked_fill(~mask[:, None, :], float("-inf"))
    return torch.einsum("nhl,ld->nhd", s.softmax(-1), V.float())

def gtime(f, n=20):
    f(); f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): f()
    g.replay(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / (3 * n) * 1e3

out = []
# full layers: slot == position, cache sized for 131072
L = 131072
K = (torch.randn(L, 512, device="cuda") * 0.3).half(); V = torch.randn(L, 512, device="cuda").half()
for p in (0, 100, 4095, 32767, 131071):
    q = torch.randn(1, 1, H, 512, device="cuda").half()
    pos = torch.tensor([p], device="cuda")
    o = decode_attention(q, K, V, None, pos, False, None, 1.0)
    r = ref(q[0], K[: p + 1], V[: p + 1], torch.ones(1, p + 1, dtype=torch.bool, device="cuda"))
    err = float((o[0].float() - r).abs().max() / r.abs().max())
    out.append({"kind": "decode full D512", "pos": p, "rel_err": float(f"{err:.1e}"), "ms": round(gtime(lambda: decode_attention(q, K, V, None, pos, False, None, 1.0)), 4)})
# sliding layers: ring of 512 holding positions p-511..p in slots pos % 512
for p in (10, 600, 5000):
    Kr = (torch.randn(W, 256, device="cuda") * 0.3).half(); Vr = torch.randn(W, 256, device="cuda").half()
    kpos = torch.full((W,), -1, dtype=torch.int32, device="cuda")
    for t in range(max(0, p - W + 1), p + 1): kpos[t % W] = t
    q = torch.randn(1, 1, H, 256, device="cuda").half(); pos = torch.tensor([p], device="cuda")
    o = decode_attention(q, Kr, Vr, kpos, pos, True, W, 1.0)
    keep = (kpos >= 0) & (kpos <= p) & (kpos > p - W)
    r = ref(q[0], Kr, Vr, keep[None, :])
    err = float((o[0].float() - r).abs().max() / r.abs().max())
    out.append({"kind": "decode sliding D256", "pos": p, "rel_err": float(f"{err:.1e}"), "ms": round(gtime(lambda: decode_attention(q, Kr, Vr, kpos, pos, True, W, 1.0)), 4)})
# prefill chunk, full layer: 512 queries at positions p0.. over keys 0..p0+511
for p0 in (0, 8192):
    n = 512; Lk = p0 + n
    q = torch.randn(1, n, H, 512, device="cuda").half()
    o = prefill_attention(q, K[:Lk], V[:Lk], None, p0, False, None, 1.0)
    qp = p0 + torch.arange(n, device="cuda"); kp = torch.arange(Lk, device="cuda")
    r = ref(q[0], K[:Lk], V[:Lk], kp[None, :] <= qp[:, None])
    out.append({"kind": "prefill full D512", "p0": p0, "rel_err": float(f"{float((o[0].float() - r).abs().max() / r.abs().max()):.1e}")})
# prefill chunk, sliding: ring (positions p0-512..p0-1) then chunk
p0, n = 1000, 300
kr = torch.arange(p0 - W, p0, device="cuda"); keys = (torch.randn(W + n, 256, device="cuda") * 0.3).half(); vals = torch.randn(W + n, 256, device="cuda").half()
kpos = torch.cat([kr, p0 + torch.arange(n, device="cuda")])
q = torch.randn(1, n, H, 256, device="cuda").half()
o = prefill_attention(q, keys, vals, kpos, p0, True, W, 1.0)
qp = p0 + torch.arange(n, device="cuda")
r = ref(q[0], keys, vals, (kpos[None, :] <= qp[:, None]) & (kpos[None, :] > qp[:, None] - W))
out.append({"kind": "prefill sliding D256", "p0": p0, "rel_err": float(f"{float((o[0].float() - r).abs().max() / r.abs().max()):.1e}")})
for r in out: print(json.dumps(r))
