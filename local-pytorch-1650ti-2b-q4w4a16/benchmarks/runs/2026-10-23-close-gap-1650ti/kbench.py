"""Time int4 GEMV kernels on E2B's real decode shapes, weighted by how many of each run per token."""
import collections, importlib, json, pathlib, sys, time, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

d = pathlib.Path(snapshot_download("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4", allow_patterns=["*.safetensors", "*.json"]))
shapes = collections.Counter()
for p in d.glob("*.safetensors"):
    with safe_open(str(p), "pt") as f:
        for k in f.keys():
            if k.endswith(".weight_shape") and "embed_tokens" not in k:
                shapes[tuple(f.get_tensor(k).tolist())] += 1
SHIFTS = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)
def ref(x, packed, scale):
    o, g = scale.shape
    q = ((packed.unsqueeze(-1) >> SHIFTS) & 15).view(o, g, 32).float() - 8
    return torch.nn.functional.linear(x.float(), (q * scale.float().unsqueeze(-1)).view(o, g * 32))
def t(f, n=50):
    """GPU time per call: n calls captured in one CUDA graph, so Python launch cost is excluded as in the model."""
    f(); f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): f()
    g.replay(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / (5 * n) * 1e3
kernels = {name: getattr(importlib.import_module(mod), fn) for name, mod, fn in
           [a.split(":") for a in sys.argv[1:]]}
torch.manual_seed(0)
rows, totals, bytes_tok = [], collections.Counter(), 0
for (O, I), cnt in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[1]):
    packed = torch.randint(-2**31, 2**31 - 1, (O, I // 8), dtype=torch.int32, device="cuda")
    scale = (torch.rand(O, I // 32, device="cuda") * 0.02).half()
    x = torch.randn(1, 1, I, device="cuda").half()
    r = ref(x, packed, scale)
    nbytes = packed.numel() * 4 + scale.numel() * 2
    bytes_tok += nbytes * cnt
    row = {"shape": [O, I], "per_token": cnt}
    for name, fn in kernels.items():
        err = float((fn(x, packed, scale).float() - r).abs().max() / r.abs().max())
        ms = t(lambda: fn(x, packed, scale))
        totals[name] += ms * cnt
        row[name] = {"ms": round(ms, 4), "GBps": round(nbytes / ms / 1e6, 1), "rel_err": float(f"{err:.1e}")}
    rows.append(row)
    print(json.dumps(row), flush=True)
print(json.dumps({"summary": {k: {"ms_per_token": round(v, 2), "GBps_effective": round(bytes_tok / v / 1e6, 1)} for k, v in totals.items()},
                  "bytes_per_token_MiB": round(bytes_tok / 2**20, 1), "distinct_shapes": len(shapes)}))

# Which configuration the autotuner settled on, per shape, for each kernel that has one.
for name, fn in kernels.items():
    mod = importlib.import_module(fn.__module__)
    for obj in vars(mod).values():
        if hasattr(obj, "cache") and isinstance(getattr(obj, "cache"), dict) and obj.cache:
            print(json.dumps({"kernel": name, "best": {str(k[:2]): str(v) for k, v in obj.cache.items()}}))
