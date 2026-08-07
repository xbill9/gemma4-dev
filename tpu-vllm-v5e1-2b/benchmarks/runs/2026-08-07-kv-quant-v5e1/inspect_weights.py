"""Read safetensors headers and report the real per-layer attention geometry.

Does NOT load weights. The safetensors format puts an 8-byte little-endian header length at the
start of the file followed by a JSON header mapping tensor name -> {dtype, shape, data_offsets},
so every shape is readable with two small reads per shard.

Answers three questions the config alone cannot:
  1. Which layers actually carry k_proj / v_proj / k_norm — i.e. is the 0-14 vs 15-34 KV-sharing
     split real in the checkpoint, or only in the runtime?
  2. What are the true projection output dims, and do they match num_heads * head_dim?
  3. Do full_attention layers differ from sliding_attention layers (the global_head_dim=512
     question, which decides whether KV is 15 or 18 KiB/token)?

stdlib only; runs on the TPU VM host against the HF cache in /dev/shm.
"""

import argparse
import glob
import json
import os
import struct
from typing import Any, Dict, List, Optional


def read_header(path: str) -> Dict[str, Any]:
    """Return the safetensors JSON header without reading tensor data."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def collect(model_dir: str) -> Dict[str, List[int]]:
    """tensor name -> shape, across every shard."""
    shapes: Dict[str, List[int]] = {}
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise SystemExit(f"no .safetensors under {model_dir}")
    for path in files:
        for name, meta in read_header(path).items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            if "shape" in meta:
                shapes[name] = meta["shape"]
    return shapes


def layer_index(name: str) -> Optional[int]:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--config", default=None, help="config.json, to cross-check layer_types")
    args = ap.parse_args()

    shapes = collect(args.model_dir)

    layer_types: List[str] = []
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        text = cfg.get("text_config", cfg)
        layer_types = list(text.get("layer_types") or [])

    # Group attention-ish tensors by layer.
    by_layer: Dict[int, Dict[str, List[int]]] = {}
    for name, shape in shapes.items():
        idx = layer_index(name)
        if idx is None:
            continue
        leaf = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
        for key in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            if f".{key}." in name or name.endswith(f".{key}.weight"):
                by_layer.setdefault(idx, {})[key] = shape
                break
        else:
            if "self_attn" in name:
                by_layer.setdefault(idx, {})[leaf] = shape

    if not by_layer:
        print("No per-layer attention tensors found. Top-level names sample:")
        for n in sorted(shapes)[:40]:
            print("   ", n, shapes[n])
        return

    print(f"{'layer':>5} {'type':>18} {'q_proj':>16} {'k_proj':>16} {'v_proj':>16} {'q_norm':>10} {'k_norm':>10}")
    have_kv: List[int] = []
    missing_kv: List[int] = []
    for idx in sorted(by_layer):
        d = by_layer[idx]
        t = layer_types[idx] if idx < len(layer_types) else "?"

        def fmt(key: str, layer: Dict[str, List[int]] = d) -> str:
            return "x".join(str(v) for v in layer[key]) if key in layer else "-"

        if "k_proj" in d:
            have_kv.append(idx)
        else:
            missing_kv.append(idx)
        print(
            f"{idx:>5} {t:>18} {fmt('q_proj'):>16} {fmt('k_proj'):>16} "
            f"{fmt('v_proj'):>16} {fmt('q_norm'):>10} {fmt('k_norm'):>10}"
        )

    print()
    print(f"layers WITH k_proj    ({len(have_kv):>2}): {have_kv}")
    print(f"layers WITHOUT k_proj ({len(missing_kv):>2}): {missing_kv}")

    # Do full_attention layers differ from sliding ones?
    if layer_types:
        dims: Dict[str, set] = {}
        for idx in have_kv:
            t = layer_types[idx] if idx < len(layer_types) else "?"
            k = by_layer[idx].get("k_proj")
            if k:
                dims.setdefault(t, set()).add(tuple(k))
        print()
        for t, s in sorted(dims.items()):
            print(f"k_proj shapes for {t}: {sorted(s)}")


if __name__ == "__main__":
    main()
