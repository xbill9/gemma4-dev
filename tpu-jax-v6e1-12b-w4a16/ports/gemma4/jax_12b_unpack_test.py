"""Exploratory test: test W4A16 nibble unpack variations on Gemma 4 12B.

Runs eager prefill for prompt "The capital of France is" under different
W4A16 unpack bit-shift orders to test if compressed-tensors 0.17.1 changed
the nibble order or packing orientation.

Usage on TPU VM:
    python3.13 ports/gemma4/jax_12b_unpack_test.py --order little
    python3.13 ports/gemma4/jax_12b_unpack_test.py --order big
    python3.13 ports/gemma4/jax_12b_unpack_test.py --order swap16
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from jax_engine import config_from_hf
from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params
from ports.gemma4.jax_e_model import (
    Gemma4EConfig,
    Gemma4EModelJAX,
    qat_w4a16_unpack_dequant_jax,
)

DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"


def unpack_custom(packed_int4: jax.Array, scale: jax.Array, group_size: int = 32, order: str = "little") -> jax.Array:
    out_features, packed_k = packed_int4.shape
    in_features = packed_k * 8

    if order == "little":
        shifts = (jnp.arange(8, dtype=jnp.int32) * 4)[None, None, :]
    elif order == "big":
        shifts = ((7 - jnp.arange(8, dtype=jnp.int32)) * 4)[None, None, :]
    elif order == "swap16":
        # 16-bit word swap: [4, 0, 12, 8, 20, 16, 28, 24]
        pattern = jnp.array([1, 0, 3, 2, 5, 4, 7, 6], dtype=jnp.int32)
        shifts = (pattern * 4)[None, None, :]
    elif order == "swap_pair":
        # Nibble pair swap: [4, 0, 12, 8, 20, 16, 28, 24]
        pattern = jnp.array([4, 0, 12, 8, 20, 16, 28, 24], dtype=jnp.int32)
        shifts = pattern[None, None, :]
    else:
        raise ValueError(f"Unknown order: {order}")

    words = packed_int4[:, :, None]
    q = ((words >> shifts) & jnp.int32(0xF)).reshape(out_features, in_features)
    q = q.astype(jnp.bfloat16) - jnp.bfloat16(8)

    grouped = q.reshape(out_features, in_features // group_size, group_size)
    scaled = grouped * scale.astype(jnp.bfloat16)[:, :, None]
    return scaled.reshape(out_features, in_features)


def dequantize_params(params: dict, order: str) -> dict:
    """Recursively replaces packed W4A16 tensors with custom-unpacked dense BF16 weights."""
    out = {}
    packed_names = set()
    for k in params:
        if k.endswith("_packed"):
            packed_names.add(k[:-len("_packed")])

    for k, v in params.items():
        if isinstance(v, dict):
            out[k] = dequantize_params(v, order)
        elif k.endswith("_packed") or k.endswith("_scale"):
            continue
        else:
            out[k] = v

    for name in packed_names:
        packed = params[f"{name}_packed"]
        scale = params.get(f"{name}_scale")
        if scale is None:
            raise ValueError(f"Missing scale for {name}_packed")
        out[name] = unpack_custom(packed, scale, order=order).T

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--order", choices=["little", "big", "swap16", "swap_pair"], default="little")
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Testing unpack order: {args.order}")

    import json
    cfg_path = hf_hub_download(args.model, "config.json")
    with open(cfg_path) as f:
        hf_cfg = json.load(f)
    config = config_from_hf(hf_cfg)
    st_path = hf_hub_download(args.model, "model.safetensors")

    print(f"Loading weights from {st_path}...")
    raw_weights = {}
    with safe_open(st_path, framework="np", device="cpu") as f:
        for k in f.keys():
            raw_weights[k] = f.get_tensor(k)

    params = convert_safetensors_to_jax_params(
        raw_weights,
        num_layers=config.num_hidden_layers,
        first_kv_shared_idx=config.first_kv_shared_layer_idx,
        attention_k_eq_v=config.attention_k_eq_v,
    )

    print(f"Dequantizing with order '{args.order}'...")
    dense_params = dequantize_params(params, args.order)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = jnp.array(tokenizer(args.prompt, return_tensors="np")["input_ids"][0])

    model = Gemma4EModelJAX(config)

    print("Running eager prefill...")
    t0 = time.perf_counter()
    position_ids = jnp.arange(ids.shape[0], dtype=jnp.int32)[None, :]
    logits = model(ids[None, :], dense_params, position_ids)
    t_prefill = time.perf_counter() - t0

    last_logits = logits[0, -1]
    top_indices = jnp.argsort(last_logits)[::-1][:5]

    print(f"\nResults for order '{args.order}' (prefill time: {t_prefill:.2f}s):")
    print(f"Logits RMS: {jnp.sqrt(jnp.mean(last_logits**2)):.2f}")
    print("Top 5 predicted tokens:")
    for idx in top_indices:
        token_str = tokenizer.decode([int(idx)])
        print(f"  ID {int(idx):6d} | score {float(last_logits[idx]):6.2f} | token {token_str!r}")


if __name__ == "__main__":
    main()
