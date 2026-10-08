"""Build a compressed-tensors int8 W8A8 checkpoint from a Gemma 4 `-qat-q4_0-unquantized` export.

    python3 w8a8_from_qat.py SRC_DIR OUT_DIR

The same scheme as an llm-compressor W8A8 export (for example glenic/gemma-4-E2B-it-W8A8-INT8):
int8 weights with one bf16 scale per output channel (`weight_scale` [out, 1]) and activations
quantized to int8 per token at run time. The difference is the source: the QAT weights, which
already sit on the Q4_0 grid, rather than the original bf16 model.

- Quantized: the same language-model linears repack_q4_0.py quantizes (attention, MLP, the
  per-layer-embedding projections), each row scaled by max|w| / 127 and rounded to nearest.
- Mixture-of-experts banks (26B-A4B): each fused `experts.gate_up_proj` / `experts.down_proj`
  is split into per-expert `experts.{i}.{gate,up,down}_proj` modules, the layout the W4A16
  repack uses and vLLM's FusedMoE loader reads, and each is quantized as above. The routers
  (`router.proj`) stay bf16 and go on the ignore list.
- Dropped: the vision and audio towers (text only, `Gemma4ForCausalLM`).
- Copied byte for byte: everything else (embeddings, norms, scalars).

Prints, per kind of tensor, the relative error of the int8 values against the QAT values.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_q4_0 import (EXPERTS, KEEP_LINEAR, LINEAR, TOWERS, bf16_to_f32,  # noqa: E402
                         f32_to_bf16, open_checkpoint, write_safetensors, _kind)

SHARD_BYTES = 2 << 30


def quantize_int8(w_u16):
    """bf16 bits [out, in] -> (int8 [out, in], bf16-bits scale [out, 1]) and relative error."""
    w = bf16_to_f32(w_u16)
    amax = np.abs(w).max(axis=1, keepdims=True)
    scale = f32_to_bf16(np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32))
    s = bf16_to_f32(scale)
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    err = np.linalg.norm(q.astype(np.float32) * s - w) / max(np.linalg.norm(w), 1e-30)
    return q, scale, float(err)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = open_checkpoint(a.src)

    shard, size, n, weight_map, errs, ignore = [], 0, 0, {}, {}, ["lm_head"]

    def flush():
        nonlocal shard, size, n
        if shard:
            fname = f"model-{n:05d}.safetensors"
            write_safetensors(os.path.join(a.out, fname), shard)
            weight_map.update({t[0]: fname for t in shard})
            shard, size, n = [], 0, n + 1

    for name in sorted(ck):
        if TOWERS.match(name):
            continue
        tag = ck[name].header[name]["dtype"]
        raw = np.asarray(ck[name].raw(name))
        e = EXPERTS.match(name)
        if LINEAR.match(name) and tag == "BF16":
            q, scale, err = quantize_int8(raw)
            errs.setdefault(_kind(name), []).append(err)
            entries = [(name, "I8", q), (name[:-len(".weight")] + ".weight_scale", "BF16", scale)]
        elif e and tag == "BF16":
            base = name[: -len(e.group(2))]
            if e.group(2) == "gate_up_proj":
                f = raw.shape[1] // 2
                parts = [("gate_proj", slice(0, f)), ("up_proj", slice(f, 2 * f))]
            else:
                parts = [("down_proj", slice(None))]
            entries = []
            for i in range(raw.shape[0]):
                for proj, rows in parts:
                    q, scale, err = quantize_int8(np.ascontiguousarray(raw[i, rows]))
                    errs.setdefault(f"experts.{proj}", []).append(err)
                    entries += [(f"{base}{i}.{proj}.weight", "I8", q), (f"{base}{i}.{proj}.weight_scale", "BF16", scale)]
        else:
            if KEEP_LINEAR.search(name) and tag == "BF16":
                ignore.append(name[: -len(".weight")])
            entries = [(name, tag, raw)]
        nbytes = sum(e[2].nbytes for e in entries)
        if size + nbytes > SHARD_BYTES:
            flush()
        shard += entries
        size += nbytes
    flush()

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg["architectures"] = ["Gemma4ForCausalLM"]
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "format": "int-quantized",
            "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel",
                        "group_size": None, "dynamic": False, "actorder": None,
                        "block_structure": None, "observer": None, "observer_kwargs": {}},
            "input_activations": {"num_bits": 8, "type": "int", "symmetric": True,
                                  "strategy": "token", "dynamic": True, "group_size": None,
                                  "actorder": None, "block_structure": None, "observer": None,
                                  "observer_kwargs": {}},
            "output_activations": None,
        }},
        "ignore": ignore,
        "kv_cache_scheme": None,
        "sparsity_config": {},
    }
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in os.listdir(a.src):
        path = os.path.join(a.src, f)
        if (f not in ("config.json", "model.safetensors.index.json", "processor_config.json")
                and not f.endswith(".safetensors") and os.path.isfile(path)):
            shutil.copy2(path, a.out)

    report = {kind: {"tensors": len(v), "rel_err_vs_qat_min": min(v), "rel_err_vs_qat_max": max(v),
                     "rel_err_vs_qat_mean": sum(v) / len(v)} for kind, v in sorted(errs.items())}
    json.dump({"source": os.path.abspath(a.src), "int8_error": report, "total_bytes": total},
              open(os.path.join(a.out, "w8a8_report.json"), "w"), indent=2)
    for kind, r in report.items():
        print(f"{kind:28s} {r['tensors']:4d} tensors, int8 vs QAT rel err "
              f"{r['rel_err_vs_qat_min']:.4f}-{r['rel_err_vs_qat_max']:.4f} (mean {r['rel_err_vs_qat_mean']:.4f})")
    print(f"total {total / 2**30:.2f} GiB in {n} shards")


if __name__ == "__main__":
    main()
