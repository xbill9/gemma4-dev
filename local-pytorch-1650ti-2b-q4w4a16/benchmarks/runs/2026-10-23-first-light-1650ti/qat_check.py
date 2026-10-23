"""Dequantize the repack's int4 weights and compare them with Google's QAT bf16 weights, tensor by tensor."""
import json, pathlib, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

REPACK = pathlib.Path(snapshot_download("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text", local_files_only=True))
GOOGLE = pathlib.Path(snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", allow_patterns=["*.safetensors", "*.json"]))
SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)

def index(d):
    f = d / "model.safetensors.index.json"
    if f.exists():
        return {k: d / v for k, v in json.load(open(f))["weight_map"].items()}
    out = {}
    for p in d.glob("*.safetensors"):
        with safe_open(str(p), "pt") as h:
            out.update({k: p for k in h.keys()})
    return out

def get(ix, k):
    with safe_open(str(ix[k]), "pt") as h:
        return h.get_tensor(k)

r, g = index(REPACK), index(GOOGLE)
if __name__ == "__main__":
    packed = sorted(k[: -len(".weight_packed")] for k in r if k.endswith(".weight_packed"))
    exact = maxdiff = 0
    missing = []
    for name in packed:
        p, s = get(r, name + ".weight_packed"), get(r, name + ".weight_scale").float()
        o, gr = s.shape
        w = ((((p.unsqueeze(-1) >> SHIFTS) & 15).view(o, gr, 32).float() - 8) * s.unsqueeze(-1)).view(o, gr * 32)
        if name + ".weight" not in g:
            missing.append(name); continue
        ref = get(g, name + ".weight").float()
        d = (w.to(torch.bfloat16).float() - ref).abs().max().item()
        exact += d == 0
        maxdiff = max(maxdiff, d)
    print(json.dumps({"packed_linears": len(packed), "compared": len(packed) - len(missing), "bit_identical_to_google_qat": int(exact),
                      "max_abs_diff": maxdiff, "missing_in_google": missing[:3]}))
