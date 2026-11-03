"""Repack a Gemma 4 QAT assistant drafter (`google/gemma-4-E?B-it-qat-q4_0-unquantized-assistant`)
as compressed-tensors W4A16, the format the rigs' targets already serve in.

    python3 repack_drafter.py SRC_DIR OUT_DIR

The drafter (`Gemma4AssistantForCausalLM`, vLLM `gemma4_mtp.py`) is four decoder layers at width
256 plus two projections to and from the target's width. Its names have no `language_model.`
level, so `repack_q4_0.py`'s LINEAR pattern matches none of them; the grid recovery and packing
(`quantize`, `pack`, `quantized_entries`) are imported from it unchanged.

What is quantized: every `self_attn.{q,o}_proj`, `mlp.{gate,up,down}_proj`, `pre_projection` and
`post_projection`, each only when every 32-wide group recovers onto a 4-bit grid; a tensor that
does not stays bf16 and goes on the ignore list. What stays bf16, byte for byte:
`model.embed_tokens` (vLLM ties the drafter's lm_head to it and indexes its rows directly in the
centroid-masked logits, so it must stay a dense weight), `masked_embedding.*`, norms and scalars.

Every quantized tensor is checked against its source on the way out: the levels times the stored
bf16 step, rounded to bf16, against the source bf16 values. The count of values that differ and
the worst relative error go to OUT_DIR/repack_report.json.
"""

import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_q4_0 import (
    GROUP,
    bf16_to_f32,
    f32_to_bf16,
    open_checkpoint,  # noqa: E402
    quantized_entries,
    unpack,
    write_safetensors,
)

LINEAR = re.compile(
    r"^(?:model\.layers\.\d+\.(?:self_attn\.[qo]_proj|mlp\.(?:gate|up|down)_proj)"
    r"|pre_projection|post_projection)\.weight$"
)


def main(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ck = open_checkpoint(src)
    tensors, ignored, report = [], [], {}
    for name in sorted(ck):
        tag = ck[name].header[name]["dtype"]
        w = np.asarray(ck[name].raw(name))
        if LINEAR.match(name) and tag == "BF16":
            module = name[: -len(".weight")]
            ents, bad = quantized_entries(module, w)
            if ents:
                q = unpack(ents[0][2]).astype(np.float32)
                s = np.repeat(bf16_to_f32(ents[1][2]), GROUP, axis=1)
                back = f32_to_bf16(q * s)
                src_f, back_f = bf16_to_f32(w), bf16_to_f32(back)
                rel = np.abs(back_f - src_f) / np.maximum(np.abs(src_f), 1e-30)
                report[module] = {
                    "shape": list(w.shape),
                    "values_differing": int((back != w).sum()),
                    "max_rel_err_of_differing": float(rel[back != w].max()) if (back != w).any() else 0.0,
                }
                tensors += ents
                continue
            report[module] = {"shape": list(w.shape), "kept_bf16_unrecovered_groups": bad}
            ignored.append(module)
        tensors.append((name, tag, w))
    write_safetensors(os.path.join(out_dir, "model.safetensors"), tensors)

    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": GROUP,
                    "dynamic": False,
                    "actorder": None,
                    "block_structure": None,
                    "observer": None,
                    "observer_kwargs": {},
                },
            }
        },
        "ignore": sorted(ignored) + ["lm_head"],
        "kv_cache_scheme": None,
        "sparsity_config": {},
    }
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if f != "config.json" and not f.endswith(".safetensors") and os.path.isfile(p):
            shutil.copy2(p, out_dir)
    total = os.path.getsize(os.path.join(out_dir, "model.safetensors"))
    json.dump(
        {
            "source": os.path.realpath(src),
            "total_bytes": total,
            "ignore": cfg["quantization_config"]["ignore"],
            "modules": report,
        },
        open(os.path.join(out_dir, "repack_report.json"), "w"),
        indent=2,
    )
    nq = sum(1 for v in report.values() if "values_differing" in v)
    diff = sum(v.get("values_differing", 0) for v in report.values())
    print(
        f"{total:,} B; {nq} of {len(report)} linears W4A16; {len(ignored)} kept bf16; "
        f"{diff:,} quantized values differ from the source after rounding to bf16"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
