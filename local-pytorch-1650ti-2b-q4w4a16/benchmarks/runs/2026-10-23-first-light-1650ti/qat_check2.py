"""Where the repack and Google's QAT bf16 weights differ: how many elements, by how much, and whether the int4 codes agree."""
import json, torch
from qat_check import SHIFTS, g, get, r
packed = sorted(k[: -len(".weight_packed")] for k in r if k.endswith(".weight_packed"))
tot = diff = code_mismatch = 0
worst_rel = 0.0
for name in packed:
    p, s = get(r, name + ".weight_packed"), get(r, name + ".weight_scale").float()
    o, gr = s.shape
    q = (((p.unsqueeze(-1) >> SHIFTS) & 15).view(o, gr, 32).float() - 8)
    w = (q * s.unsqueeze(-1)).view(o, gr * 32).to(torch.bfloat16).float()
    ref = get(g, name + ".weight").float()
    d = (w != ref)
    tot += ref.numel(); diff += int(d.sum())
    # int4 code Google's value implies under the repack's scale
    qref = (ref.view(o, gr, 32) / s.unsqueeze(-1)).round()
    code_mismatch += int((qref != q).sum())
    rel = ((w - ref).abs() / s.repeat_interleave(32, 1)).max().item()
    worst_rel = max(worst_rel, rel)
print(json.dumps({"elements": tot, "elements_differing": diff, "fraction_differing": diff / tot,
                  "int4_codes_disagreeing": code_mismatch, "worst_diff_in_units_of_group_step": worst_rel}))
