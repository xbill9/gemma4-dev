#!/usr/bin/env python3
"""Torch-free int8 requantizer for QAT checkpoints (inf2 phase-2 path).

Reads a Hugging Face safetensors checkpoint (single file, sharded dir with
model.safetensors.index.json, or a bare .safetensors path) and emits ONE
output .safetensors file where every eligible 2D projection weight is
replaced by:

    {base}.weight_i8     int8  [out, in]         symmetric, range [-127, 127]
    {base}.weight_scale  fp32  [out, in/GROUP]   per-group scales

plus a byte-identical passthrough of everything else (norms, biases,
embeddings). A quant_config.json with the convention and per-tensor
round-trip error stats is written next to the output.

Convention (mirrors the port repo's w4a16 compressed-tensors scheme at
int8 width — no bit-packing needed):
  - symmetric int8: scale = group_absmax / 127, q = clip(rint(w/scale))
  - group_size 32 along the LAST dim (the contraction/in_features dim)
  - fp32 scales so dequant is bit-exact reproducible; cast in-graph
  - dequant for the traced graph is just q.to(bf16) * scale (repeat 32)

Dependencies: numpy + stdlib only. No torch, no safetensors package —
the safetensors container format is parsed/written directly.

Usage:
  python3 quantize_int8.py CHECKPOINT_DIR_OR_FILE -o model_int8_g32.safetensors
  python3 quantize_int8.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import tempfile

import numpy as np

GROUP_SIZE_DEFAULT = 32
QMAX = 127.0
COPY_CHUNK = 64 * 1024 * 1024

# safetensors dtype tag -> (numpy dtype or None for BF16, itemsize)
DTYPES = {
    "F64": (np.float64, 8),
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "BF16": (None, 2),
    "I64": (np.int64, 8),
    "I32": (np.int32, 4),
    "I16": (np.int16, 2),
    "I8": (np.int8, 1),
    "U8": (np.uint8, 1),
    "BOOL": (np.bool_, 1),
}
FLOAT_TAGS = ("F64", "F32", "F16", "BF16")


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32)


class ShardReader:
    """Minimal safetensors reader: header parse + per-tensor seek/read."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        self.data_start = 8 + hlen
        header.pop("__metadata__", None)
        self.tensors = header  # name -> {dtype, shape, data_offsets}

    def info(self, name: str):
        t = self.tensors[name]
        return t["dtype"], tuple(t["shape"])

    def read_f32(self, name: str) -> np.ndarray:
        """Read a float tensor upcast to fp32 (handles BF16)."""
        t = self.tensors[name]
        raw = self._raw(name)
        if t["dtype"] == "BF16":
            arr = bf16_bytes_to_f32(raw)
        else:
            arr = np.frombuffer(raw, dtype=DTYPES[t["dtype"]][0]).astype(np.float32)
        return arr.reshape(t["shape"])

    def _raw(self, name: str) -> bytes:
        start, end = self.tensors[name]["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + start)
            return f.read(end - start)

    def copy_raw(self, name: str, out_f) -> None:
        start, end = self.tensors[name]["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + start)
            remaining = end - start
            while remaining:
                chunk = f.read(min(COPY_CHUNK, remaining))
                out_f.write(chunk)
                remaining -= len(chunk)


def discover_shards(path: str) -> list[tuple[str, list[str]]]:
    """Return [(shard_path, [tensor names in shard])] in deterministic order."""
    if os.path.isfile(path):
        r = ShardReader(path)
        return [(path, sorted(r.tensors))]
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        by_shard: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        return [
            (os.path.join(path, shard), sorted(names))
            for shard, names in sorted(by_shard.items())
        ]
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        r = ShardReader(single)
        return [(single, sorted(r.tensors))]
    raise FileNotFoundError(f"no safetensors checkpoint found at {path}")


def quantize_int8(w: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """[out, in] fp32 -> (int8 [out, in], fp32 scales [out, in/group_size])."""
    out_dim, in_dim = w.shape
    g = w.reshape(out_dim, in_dim // group_size, group_size)
    scale = np.abs(g).max(axis=-1, keepdims=True) / QMAX
    scale = np.maximum(scale, np.finfo(np.float32).tiny)  # all-zero groups
    q = np.clip(np.rint(g / scale), -QMAX, QMAX).astype(np.int8)
    return q.reshape(out_dim, in_dim), scale.reshape(out_dim, in_dim // group_size).astype(np.float32)


def dequantize_int8(q: np.ndarray, scale: np.ndarray, group_size: int) -> np.ndarray:
    out_dim, in_dim = q.shape
    g = q.reshape(out_dim, in_dim // group_size, group_size).astype(np.float32)
    return (g * scale.reshape(out_dim, in_dim // group_size, 1)).reshape(out_dim, in_dim)


def plan_tensor(name: str, dtype: str, shape: tuple, args) -> bool:
    """True if this tensor gets quantized."""
    if dtype not in FLOAT_TAGS or len(shape) != 2:
        return False
    if shape[-1] % args.group_size or min(shape) < args.min_dim:
        return False
    return not (args.exclude and re.search(args.exclude, name))


def out_names(name: str) -> tuple[str, str]:
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    return base + ".weight_i8", base + ".weight_scale"


def run(args) -> dict:
    shards = discover_shards(args.checkpoint)
    readers = {p: ShardReader(p) for p, _ in shards}

    # Pass 1: plan the output header (shapes/offsets only, no tensor data).
    entries = []  # (out_name, out_dtype, out_shape, nbytes, src_shard, src_name, kind)
    seen = set()
    for shard_path, names in shards:
        r = readers[shard_path]
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            dtype, shape = r.info(name)
            if plan_tensor(name, dtype, shape, args):
                qn, sn = out_names(name)
                out_dim, in_dim = shape
                entries.append((qn, "I8", shape, out_dim * in_dim, shard_path, name, "quant_q"))
                sshape = (out_dim, in_dim // args.group_size)
                entries.append((sn, "F32", sshape, sshape[0] * sshape[1] * 4, shard_path, name, "quant_s"))
            else:
                start, end = r.tensors[name]["data_offsets"]
                entries.append((name, dtype, shape, end - start, shard_path, name, "copy"))

    header: dict = {
        "__metadata__": {
            "format": f"int8-sym-g{args.group_size}",
            "group_size": str(args.group_size),
            "quantized_along": "last dim (in_features)",
            "range": "[-127, 127]",
            "scale_dtype": "F32",
            "producer": "tpu-pytorch-inf2/quant/quantize_int8.py",
        }
    }
    offset = 0
    for out_name, out_dtype, out_shape, nbytes, *_ in entries:
        header[out_name] = {
            "dtype": out_dtype,
            "shape": list(out_shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    hjson = json.dumps(header, separators=(",", ":")).encode()

    # Pass 2: stream data in header order; quantize each source tensor once.
    stats: dict[str, dict] = {}
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with open(args.output, "wb") as out_f:
        out_f.write(struct.pack("<Q", len(hjson)))
        out_f.write(hjson)
        for out_name, _dt, _shape, _nb, shard_path, src_name, kind in entries:
            r = readers[shard_path]
            if kind == "copy":
                r.copy_raw(src_name, out_f)
                continue
            if src_name not in cache:
                w = r.read_f32(src_name)
                q, scale = quantize_int8(w, args.group_size)
                dq = dequantize_int8(q, scale, args.group_size)
                denom = float(np.abs(w).mean()) or 1.0
                stats[src_name] = {
                    "shape": list(w.shape),
                    "mean_abs_err_rel": float(np.abs(dq - w).mean() / denom),
                    "max_abs_err": float(np.abs(dq - w).max()),
                }
                cache[src_name] = (q, scale)
            q, scale = cache[src_name]
            out_f.write(q.tobytes() if kind == "quant_q" else scale.tobytes())
            if kind == "quant_s":
                del cache[src_name]

    config = {
        "format": f"int8-sym-g{args.group_size}",
        "group_size": args.group_size,
        "range": [-127, 127],
        "scale_dtype": "float32",
        "suffixes": {"qweight": ".weight_i8", "scale": ".weight_scale"},
        "dequant": "w = q.astype(compute_dtype) * scale.repeat(group_size, axis=-1)",
        "source_checkpoint": args.checkpoint,
        "exclude_pattern": args.exclude,
        "quantized_tensors": stats,
    }
    cfg_path = os.path.splitext(args.output)[0] + ".quant_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    total_in = sum(v["shape"][0] * v["shape"][1] for v in stats.values())
    print(f"quantized {len(stats)} tensors ({total_in / 1e9:.2f} B params) -> {args.output}")
    if stats:
        worst = max(stats.items(), key=lambda kv: kv[1]["mean_abs_err_rel"])
        print(f"worst mean-abs relative error: {worst[1]['mean_abs_err_rel']:.4%} ({worst[0]})")
    print(f"config + per-tensor stats: {cfg_path}")
    return config


# ---------------------------------------------------------------------------
# selftest


def _write_safetensors(path: str, tensors: dict) -> None:
    """tensors: name -> (dtype_tag, np array or raw bytes, shape)."""
    header, blobs, offset = {}, [], 0
    for name, (tag, data, shape) in tensors.items():
        raw = data if isinstance(data, bytes) else data.tobytes()
        header[name] = {"dtype": tag, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    hjson = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for raw in blobs:
            f.write(raw)


def f32_to_bf16_bytes(a: np.ndarray) -> bytes:
    return (a.astype(np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()


def selftest() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        proj = rng.standard_normal((512, 256), dtype=np.float32)
        norm = np.ones(256, dtype=np.float32)
        embed = rng.standard_normal((300, 256), dtype=np.float32)
        _write_safetensors(
            os.path.join(d, "model-00001-of-00002.safetensors"),
            {
                "model.layers.0.self_attn.q_proj.weight": ("BF16", f32_to_bf16_bytes(proj), proj.shape),
                "model.layers.0.input_layernorm.weight": ("F32", norm, norm.shape),
            },
        )
        _write_safetensors(
            os.path.join(d, "model-00002-of-00002.safetensors"),
            {"model.embed_tokens.weight": ("F32", embed, embed.shape)},
        )
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump(
                {
                    "weight_map": {
                        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                        "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
                        "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
                    }
                },
                f,
            )

        out = os.path.join(d, "out.safetensors")
        args = argparse.Namespace(
            checkpoint=d, output=out, group_size=32, min_dim=64, exclude=r"embed|lm_head"
        )
        run(args)

        r = ShardReader(out)
        expect = {
            "model.layers.0.self_attn.q_proj.weight_i8": ("I8", (512, 256)),
            "model.layers.0.self_attn.q_proj.weight_scale": ("F32", (512, 8)),
            "model.layers.0.input_layernorm.weight": ("F32", (256,)),
            "model.embed_tokens.weight": ("F32", (300, 256)),
        }
        assert set(r.tensors) == set(expect), sorted(r.tensors)
        for name, (tag, shape) in expect.items():
            assert r.info(name) == (tag, shape), (name, r.info(name))

        # passthrough must be byte-identical
        assert np.array_equal(r.read_f32("model.embed_tokens.weight"), embed)

        # round trip: int8 g32 on ~N(0,1) should sit well under 1% mean error
        q = np.frombuffer(r._raw("model.layers.0.self_attn.q_proj.weight_i8"), dtype=np.int8).reshape(512, 256)
        scale = r.read_f32("model.layers.0.self_attn.q_proj.weight_scale")
        proj_bf16 = bf16_bytes_to_f32(f32_to_bf16_bytes(proj)).reshape(proj.shape)  # what the file held
        dq = dequantize_int8(q, scale, 32)
        rel = np.abs(dq - proj_bf16).mean() / np.abs(proj_bf16).mean()
        assert rel < 0.01, f"round-trip rel error {rel:.4%}"
        assert np.abs(q).max() <= 127

        # dequant convention doc string must match reality
        dq2 = q.astype(np.float32) * np.repeat(scale, 32, axis=-1)
        assert np.array_equal(dq, dq2)

        cfg = json.load(open(os.path.join(d, "out.quant_config.json")))
        assert list(cfg["quantized_tensors"]) == ["model.layers.0.self_attn.q_proj.weight"]
        print(f"SELFTEST PASS (round-trip mean rel error {rel:.4%})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", help="HF checkpoint dir or .safetensors file")
    ap.add_argument("-o", "--output", help="output .safetensors path")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE_DEFAULT)
    ap.add_argument("--min-dim", type=int, default=256, help="skip 2D tensors smaller than this")
    ap.add_argument("--exclude", default=r"embed|lm_head", help="regex of tensor names to pass through")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.checkpoint or not args.output:
        ap.error("checkpoint and -o/--output are required (or use --selftest)")
    run(args)


if __name__ == "__main__":
    main()
