"""Is the QAT embed_tokens (tied lm_head) on a Q4_0 grid? Q4_0: groups of 32 along the row, d = absmax-signed / -8,
q = round(x / d) in [-8, 7]. Also: does the repack's embed_tokens equal Google's?"""
import json, pathlib, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

def tensor(repo, key):
    d = pathlib.Path(snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"]))
    for p in sorted(d.glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(key)

KEY = "model.language_model.embed_tokens.weight"
g = tensor("google/gemma-4-E2B-it-qat-q4_0-unquantized", KEY)
r = tensor("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text", KEY)
x = g.float().view(g.shape[0], -1, 32)
amax_idx = x.abs().argmax(-1, keepdim=True)
signed_max = x.gather(-1, amax_idx)
d = signed_max / -8
d = torch.where(d == 0, torch.ones_like(d), d)
q = (x / d).round().clamp(-8, 7)
recon = (q * d).to(torch.bfloat16).float()
# Looser test: per group, does some step s put every value on an integer multiple within [-8, 7]?
step = x.abs().amax(-1, keepdim=True) / 8
q2 = (x / step).round()
recon2 = (q2 * step).to(torch.bfloat16).float()
print(json.dumps({
    "shape": list(g.shape), "dtype": str(g.dtype),
    "repack_equals_google": bool(torch.equal(g, r)),
    "q4_0_exact_fraction": float((recon == x).float().mean()),
    "q4_0_groups_exact": float((recon == x).all(-1).float().mean()),
    "absmax_over_8_groups_exact": float((recon2 == x).all(-1).float().mean()),
    "distinct_values_per_group_max": int(max(len(torch.unique(row)) for row in x[:2048].reshape(-1, 32))),
}))
