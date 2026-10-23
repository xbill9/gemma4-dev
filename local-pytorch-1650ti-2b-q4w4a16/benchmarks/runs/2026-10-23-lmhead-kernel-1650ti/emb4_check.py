"""How closely the -text-emb4 int4 tables (fp16 steps) reproduce the bf16 QAT tables they were packed from."""
import json, pathlib, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)
E4 = pathlib.Path(snapshot_download("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4", allow_patterns=["model-rest-00[01].safetensors", "*.json"]))
TX = pathlib.Path(snapshot_download("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text", allow_patterns=["*.safetensors", "*.json"]))

def find(d, key):
    for p in sorted(d.glob("*.safetensors")):
        f = safe_open(str(p), "pt")
        if key in f.keys():
            return f
    raise KeyError(key)

out = {}
for name, packed_key, ref_key in [
    ("lm_head", "lm_head", "model.language_model.embed_tokens.weight"),
    ("embed_tokens", "model.language_model.embed_tokens", "model.language_model.embed_tokens.weight"),
    ("embed_tokens_per_layer", "model.language_model.embed_tokens_per_layer", "model.language_model.embed_tokens_per_layer.weight"),
]:
    fp, fr = find(E4, packed_key + ".weight_packed"), find(TX, ref_key)
    P, S, R = fp.get_slice(packed_key + ".weight_packed"), fp.get_slice(packed_key + ".weight_scale"), fr.get_slice(ref_key)
    rows = P.get_shape()[0]
    tot = exact = 0
    worst = 0.0
    sdtype = None
    for a in range(0, rows, 8192):
        p, s, ref = P[a:a + 8192], S[a:a + 8192], R[a:a + 8192].float()
        sdtype = str(s.dtype)
        o, g = s.shape
        q = (((p.unsqueeze(-1) >> SHIFTS) & 15).view(o, g, 32).float() - 8)
        w = (q * s.float().unsqueeze(-1)).view(o, g * 32)
        tot += ref.numel(); exact += int((w.to(torch.bfloat16).float() == ref).sum())
        worst = max(worst, float(((w - ref).abs() / s.float().repeat_interleave(32, 1)).max()))
    out[name] = {"shape": [rows, R.get_shape()[1]], "scale_dtype": sdtype, "values_exact_after_bf16_round": exact / tot,
                 "worst_diff_in_group_steps": round(worst, 4)}
print(json.dumps(out))
