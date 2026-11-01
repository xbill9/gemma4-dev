"""Time the int4 GEMV kernel on E4B's decode shapes over a tiling grid; report the best per shape and the token total."""
import itertools, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch, triton
from w4gemv import _w4_gemv

# (rows, packed int32 cols): count per generated token, E4B with merged gate+up and q+k+v.
SHAPES = {(20480, 320): 42, (2560, 1280): 42, (3072, 320): 20, (6144, 320): 4, (2048, 320): 15, (4096, 320): 3,
          (2560, 256): 35, (2560, 512): 7, (262144, 320): 1}
GRID = list(itertools.product((8, 16, 32), (8, 16, 32), (1, 2, 4), (1, 2, 3, 4)))
def timed(f, n):
    f(); f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): f()
    g.replay(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / (5 * n) * 1e3
torch.manual_seed(0)
best, total = {}, 0.0
for (O, KP), cnt in SHAPES.items():
    packed = torch.randint(-2**31, 2**31 - 1, (O, KP), dtype=torch.int32, device="cuda")
    scale = (torch.rand(O, KP // 4, device="cuda") * 0.02).half()
    x = torch.randn(KP * 8, device="cuda").half()
    y = torch.empty(O, device="cuda", dtype=torch.float16)
    res = []
    for bo, bg, w, st in GRID:
        if bo * bg * 4 * 8 > 32768: continue
        f = lambda: _w4_gemv[(triton.cdiv(O, bo),)](x, packed, scale, y, O, KP, scale.shape[1], BLOCK_O=bo, BLOCK_G=bg, num_warps=w, num_stages=st)
        try: ms = timed(f, 20 if O < 100000 else 5)
        except Exception as e: continue
        res.append((ms, bo, bg, w, st))
    res.sort(); default = next(r[0] for r in res if r[1:] == (8, 16, 1, 1))
    best[(O, KP)] = res[0]; total += res[0][0] * cnt
    print(json.dumps({"shape": [O, KP], "per_token": cnt, "best": res[:3], "default_ms": round(default, 4), "n": len(res)}), flush=True)
print(json.dumps({"ms_per_token_gemv": round(total, 3)}))
print("CONFIGS", {k: v[1:] for k, v in best.items()})
