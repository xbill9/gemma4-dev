"""Repack a Gemma 4 `-qat-q4_0-unquantized` checkpoint as compressed-tensors W4A16.

    python3 repack_q4_0.py repack SRC_DIR OUT_DIR [--workers 6]
    python3 repack_q4_0.py verify SRC_DIR OUT_DIR

The `-qat-q4_0-unquantized` exports hold QAT weights that already sit on a
4-bit grid in groups of 32 along the input dimension, stored as bf16. This
recovers each group's step and integer levels and writes them in the layout
the JAX-path W4A16 method (vllm-project/tpu-inference#3653) loads, and that
cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit uses for a 26B MoE:

    <module>.weight_packed  int32 [out, in // 8]   nibble i of word j = q + 8 for column 8j + i
    <module>.weight_scale   bf16  [out, in // 32]  one step per 32-wide group
    <module>.weight_shape   int64 [2]              [out, in]

Fused experts (`experts.gate_up_proj [E, 2F, D]`, `experts.down_proj [E, D, F]`)
are split into `experts.{i}.{gate,up,down}_proj`, gate being the first F rows
of gate_up (tpu_inference/models/jax/gemma4.py, Gemma4MoE.load_weights).

What is quantized: language-model attention and dense-MLP Linear weights, and
the experts, each only when every one of its groups recovers onto a 4-bit grid.
A tensor with any unrecovered group stays bf16 and is added to the ignore list,
never re-gridded. Everything else (router, embeddings, norms, scalars, vision
tower) is copied byte for byte.

Step recovery follows ~/tpu-jax-26b/ports/gemma4/jax_q4_0.py: the textbook
`d = amax / 8` is wrong for any group whose peak is not at level 8, so the step
is `amax / m` for the first m in 1..8 that reproduces the whole group, refined
by least squares over the group, then rounded to the bf16 scale that is stored
(#3653 loads weight_scale as bf16 whatever the checkpoint holds). The levels
are derived against that stored scale.

`verify` rereads both checkpoints from disk and reports, per kind of tensor:
groups whose levels differ from the source grid, values that reconstruct
bit-identically, and the worst relative error of the rest; and it checks that
every copied tensor is byte-identical and that no source tensor is missing.
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

GROUP = 32
PACK = 8
REL_TOL = 2.0 ** -7

LINEAR = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$")
EXPERTS = re.compile(r"^model\.language_model\.layers\.(\d+)\.experts\.(gate_up_proj|down_proj)$")
LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
# 2-D weights of modules that compressed-tensors' "Linear" target would match but
# that stay bf16 on purpose; they go on the ignore list by module name.
KEEP_LINEAR = re.compile(r"(router\.proj|vision_tower\..*|embed_vision\..*)\.weight$")

_DTYPES = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4),
           "I32": (np.int32, 4), "I64": (np.int64, 8), "U8": (np.uint8, 1)}


# ---- bf16 as raw uint16 bits (numpy has no bfloat16) -----------------------

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    """Round to nearest even, as JAX and torch do."""
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))
    return (u >> 16).astype(np.uint16)


# ---- safetensors, by hand (no torch here) ----------------------------------

class Reader:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.meta = self.header.pop("__metadata__", None)

    def names(self):
        return list(self.header)

    def raw(self, name):
        """The tensor's bytes as a read-only memmap of its storage dtype."""
        h = self.header[name]
        dt, size = _DTYPES[h["dtype"]]
        a, b = h["data_offsets"]
        n = (b - a) // size
        return np.memmap(self.path, dtype=dt, mode="r", offset=self.base + a, shape=(n,)).reshape(h["shape"])


def open_checkpoint(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        files = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    readers = [Reader(os.path.join(d, f)) for f in files]
    return {n: r for r in readers for n in r.names()}


def write_safetensors(path, tensors):
    """tensors: list of (name, dtype_tag, ndarray). Writes in the given order."""
    header, off = {}, 0
    for name, tag, arr in tensors:
        n = arr.nbytes
        header[name] = {"dtype": tag, "shape": list(arr.shape), "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, _, arr in tensors:
            f.write(np.ascontiguousarray(arr).tobytes())
    os.replace(path + ".tmp", path)


# ---- the grid --------------------------------------------------------------

def quantize(w_u16):
    """bf16 bits [out, in] -> (q int8 [out, in], scale bf16 bits [out, in/32], n_unrecovered).

    n_unrecovered counts groups that land on no 4-bit grid at any level 1..8.
    """
    out, inn = w_u16.shape
    g = bf16_to_f32(w_u16).reshape(out, inn // GROUP, GROUP)
    amax = np.abs(g).max(axis=-1, keepdims=True)
    nonzero = amax > 0
    tol = amax * REL_TOL
    d = np.where(nonzero, amax / 8.0, 0.0).astype(np.float32)
    ok = ~nonzero
    for m in range(1, 9):
        cand = np.where(nonzero, amax / m, 1.0).astype(np.float32)
        q = np.clip(np.rint(g / cand), -8, 7)
        err = np.abs(g - q * cand).max(axis=-1, keepdims=True)
        good = (err <= tol) & nonzero & ~ok
        d = np.where(good, cand, d)
        ok |= good
    safe = np.where(d > 0, d, 1.0)
    q = np.clip(np.rint(g / safe), -8, 7)
    den = (q * q).sum(axis=-1, keepdims=True)
    num = (q * g).sum(axis=-1, keepdims=True)
    d = np.where(den > 0, num / np.where(den > 0, den, 1.0), d).astype(np.float32)
    scale = f32_to_bf16(d)
    s = bf16_to_f32(scale)
    q = np.clip(np.rint(g / np.where(s > 0, s, 1.0)), -8, 7).astype(np.int8)
    return q.reshape(out, inn), scale.reshape(out, inn // GROUP), int((~ok).sum())


def pack(q):
    """int levels [-8, 7] [out, in] -> int32 [out, in / 8], nibble i of word j = column 8j + i."""
    out, inn = q.shape
    u = (q.astype(np.int32) + 8).astype(np.uint32).reshape(out, inn // PACK, PACK)
    word = np.zeros((out, inn // PACK), dtype=np.uint32)
    for i in range(PACK):
        word |= u[:, :, i] << np.uint32(4 * i)
    return word.view(np.int32)


def unpack(packed):
    """Inverse of pack: int32 [out, in / 8] -> int8 [out, in]."""
    w = packed.view(np.uint32)
    cols = [((w >> np.uint32(4 * i)) & np.uint32(0xF)).astype(np.int8) - 8 for i in range(PACK)]
    return np.stack(cols, axis=-1).reshape(packed.shape[0], packed.shape[1] * PACK)


def quantized_entries(module, w_u16):
    """The three compressed-tensors tensors for one module, or None if any group is off-grid."""
    q, scale, bad = quantize(w_u16)
    if bad:
        return None, bad
    out, inn = w_u16.shape
    return [(module + ".weight_packed", "I32", pack(q)),
            (module + ".weight_scale", "BF16", scale),
            (module + ".weight_shape", "I64", np.array([out, inn], dtype=np.int64))], 0


# ---- repack ----------------------------------------------------------------

def _layer_job(args):
    """Repack every tensor of one decoder layer into its own shard. Returns a summary."""
    src, out_dir, layer, names = args
    ck = open_checkpoint(src)
    tensors, ignored, kept = [], [], {}
    for name in sorted(names):
        r = ck[name]
        tag = r.header[name]["dtype"]
        m = LINEAR.match(name)
        e = EXPERTS.match(name)
        if m and tag == "BF16":
            ents, bad = quantized_entries(name[: -len(".weight")], np.asarray(r.raw(name)))
            if ents:
                tensors += ents
                continue
            kept[name] = bad
            ignored.append(name[: -len(".weight")])
        elif e and tag == "BF16":
            base = name[: -len(e.group(2))]
            w = r.raw(name)
            if e.group(2) == "gate_up_proj":
                f = w.shape[1] // 2
                parts = [("gate_proj", slice(0, f)), ("up_proj", slice(f, 2 * f))]
            else:
                parts = [("down_proj", slice(None))]
            per_expert, bad_total = [], 0
            for i in range(w.shape[0]):
                for proj, rows in parts:
                    ents, bad = quantized_entries(f"{base}{i}.{proj}", np.asarray(w[i, rows]))
                    bad_total += bad
                    if ents:
                        per_expert += ents
            if bad_total == 0:
                tensors += per_expert
                continue
            # An off-grid expert bank stays fused bf16; the loader's fused path reads it.
            kept[name] = bad_total
        tensors.append((name, tag, np.asarray(r.raw(name))))
    fname = f"model-layer-{layer:03d}.safetensors"
    write_safetensors(os.path.join(out_dir, fname), tensors)
    return fname, [t[0] for t in tensors], ignored, kept


def repack(src, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    ck = open_checkpoint(src)
    by_layer, rest = {}, []
    for name in ck:
        m = LAYER.match(name)
        (by_layer.setdefault(int(m.group(1)), []) if m else rest).append(name)

    weight_map, ignored, kept = {}, [], {}
    with ProcessPoolExecutor(workers) as pool:
        jobs = [(src, out_dir, layer, names) for layer, names in sorted(by_layer.items())]
        for fname, names, ign, kp in pool.map(_layer_job, jobs):
            weight_map.update({n: fname for n in names})
            ignored += ign
            kept.update(kp)
            print(f"{fname}: {len(names)} tensors, {len(kp)} kept bf16", flush=True)

    # Everything outside the decoder layers is copied, split so no shard passes ~4 GB.
    shard, size, n = [], 0, 0
    def flush():
        nonlocal shard, size, n
        if shard:
            fname = f"model-rest-{n:03d}.safetensors"
            write_safetensors(os.path.join(out_dir, fname), shard)
            weight_map.update({t[0]: fname for t in shard})
            shard, size, n = [], 0, n + 1
    for name in sorted(rest):
        arr = np.asarray(ck[name].raw(name))
        if size + arr.nbytes > 4 << 30:
            flush()
        shard.append((name, ck[name].header[name]["dtype"], arr))
        size += arr.nbytes
    flush()

    # Modules compressed-tensors would target as Linear but that stay bf16.
    for name in ck:
        if len(ck[name].header[name]["shape"]) == 2 and KEEP_LINEAR.search(name):
            ignored.append(name[: -len(".weight")])
    ignored = sorted(set(ignored)) + ["lm_head"]

    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
                        "group_size": GROUP, "dynamic": False, "actorder": None,
                        "block_structure": None, "observer": None, "observer_kwargs": {}},
        }},
        "ignore": ignored,
        "kv_cache_scheme": None,
        "sparsity_config": {},
    }
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    for f in os.listdir(src):
        if f != "config.json" and not f.endswith(".safetensors") and f != "model.safetensors.index.json" \
                and os.path.isfile(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), out_dir)
    json.dump({"source": os.path.abspath(src), "kept_bf16_unrecovered_groups": kept,
               "ignore": ignored, "total_bytes": total},
              open(os.path.join(out_dir, "repack_report.json"), "w"), indent=2)
    print(f"total {total / 2**30:.2f} GiB in {len(set(weight_map.values()))} shards; "
          f"{len(kept)} tensors kept bf16 for off-grid groups; {len(ignored)} ignored modules")


# ---- verify ----------------------------------------------------------------

def _kind(name):
    m = LINEAR.match(name)
    if m:
        return m.group(2)
    m = EXPERTS.match(name)
    return "experts." + m.group(2) if m else None


def _check_module(src_u16, out, module, acc):
    q = unpack(np.asarray(out[module + ".weight_packed"].raw(module + ".weight_packed")))
    scale = np.asarray(out[module + ".weight_scale"].raw(module + ".weight_scale"))
    shape = np.asarray(out[module + ".weight_shape"].raw(module + ".weight_shape"))
    rows, cols = src_u16.shape
    assert list(shape) == [rows, cols], (module, shape, src_u16.shape)
    s = np.repeat(bf16_to_f32(scale), GROUP, axis=1)
    x = bf16_to_f32(src_u16)
    # Levels: the stored level must be the source value's rank on the stored grid.
    level = np.clip(np.rint(x / np.where(s > 0, s, 1.0)), -8, 7)
    lvl_bad = (level != q).reshape(rows, cols // GROUP, GROUP).any(axis=-1)
    recon = f32_to_bf16(q.astype(np.float32) * s)
    same = recon == src_u16
    rel = np.abs(bf16_to_f32(recon) - x) / np.maximum(np.abs(x), 1e-30)
    rel[x == 0] = np.where(bf16_to_f32(recon)[x == 0] == 0, 0.0, np.inf)
    acc["groups"] += lvl_bad.size
    acc["groups_level_mismatch"] += int(lvl_bad.sum())
    acc["values"] += same.size
    acc["values_bit_identical"] += int(same.sum())
    acc["max_rel_err"] = max(acc["max_rel_err"], float(rel.max()))


def verify(src, out_dir):
    ck, out = open_checkpoint(src), open_checkpoint(out_dir)
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    ignored = set(cfg["quantization_config"]["ignore"])
    stats, copied, problems = {}, {"tensors": 0, "byte_identical": 0}, []
    for name in sorted(ck):
        r = ck[name]
        kind = _kind(name)
        quantized_here = kind and name not in out
        if not quantized_here:
            if name not in out:
                problems.append(f"missing from output: {name}")
                continue
            a, b = np.asarray(r.raw(name)), np.asarray(out[name].raw(name))
            copied["tensors"] += 1
            if a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8)):
                copied["byte_identical"] += 1
            else:
                problems.append(f"copied tensor differs: {name}")
            if kind and name.endswith(".weight") and name[:-7] not in ignored:
                problems.append(f"bf16 Linear not on the ignore list: {name}")
            continue
        acc = stats.setdefault(kind, {"tensors": 0, "groups": 0, "groups_level_mismatch": 0,
                                      "values": 0, "values_bit_identical": 0, "max_rel_err": 0.0})
        acc["tensors"] += 1
        w = r.raw(name)
        if kind.startswith("experts."):
            base = name[: -len(kind[len("experts."):])]
            f = w.shape[1] // 2
            for i in range(w.shape[0]):
                if kind == "experts.gate_up_proj":
                    _check_module(np.asarray(w[i, :f]), out, f"{base}{i}.gate_proj", acc)
                    _check_module(np.asarray(w[i, f:]), out, f"{base}{i}.up_proj", acc)
                else:
                    _check_module(np.asarray(w[i]), out, f"{base}{i}.down_proj", acc)
        else:
            _check_module(np.asarray(w), out, name[: -len(".weight")], acc)
        print(f"checked {name}", file=sys.stderr, flush=True)
    for kind, a in stats.items():
        a["share_bit_identical"] = a["values_bit_identical"] / a["values"]
    report = {"quantized": stats, "copied": copied, "problems": problems,
              "src_bytes": sum(os.path.getsize(r.path) for r in {id(r): r for r in ck.values()}.values()),
              "out_bytes": sum(os.path.getsize(r.path) for r in {id(r): r for r in out.values()}.values())}
    json.dump(report, open(os.path.join(out_dir, "verify_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    return 0 if not problems and all(a["groups_level_mismatch"] == 0 for a in stats.values()) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=["repack", "verify"])
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.mode == "repack":
        repack(a.src, a.out, a.workers)
        return 0
    return verify(a.src, a.out)


if __name__ == "__main__":
    sys.exit(main())
