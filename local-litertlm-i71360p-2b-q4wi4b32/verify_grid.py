"""Check a built .litertlm against the QAT source, tensor by tensor, from the file itself.

    python3 -I verify_grid.py SRC_HF_DIR UNPACKED_LITERTLM_DIR > verify_grid.json

The tflite sections name their weights `arith.constantN`, so tensors are matched by
CONTENT: each int4 blockwise tensor's levels are hashed and looked up among the QAT
source tensors' levels (recovered with build_litertlm.grid_step), the per-layer
embedding table as 256-column slices. For every match it counts
  - levels identical to the QAT levels (a match means all of them are),
  - blocks whose stored scale equals the trained step rounded as LiteRT rounds it
    (float32 -> bfloat16 -> float16),
  - values whose dequantized weight, rounded to bf16, is the source bf16 value.
A source tensor may match more than one stored copy (the embedding table is also the tied
output head). An int4 tensor with no match, or a source tensor never matched, is reported.
"""

import hashlib
import json
import mmap
import os
import struct
import sys

import ml_dtypes
import numpy as np
from ai_edge_litert import schema_py_generated as fb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_litertlm import QK, grid_step  # noqa: E402

PLE = "model.language_model.embed_tokens_per_layer.weight"
PLE_SLICE = 256


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))
    return (u >> 16).astype(np.uint16)


def source_tensors(d):
    p = os.path.join(d, "model.safetensors")
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    for name, v in sorted(h.items()):
        if v["dtype"] == "BF16" and len(v["shape"]) == 2 and "language_model" in name and v["shape"][1] % QK == 0:
            a, b = v["data_offsets"]
            w = np.memmap(p, np.uint16, "r", offset=8 + n + a, shape=((b - a) // 2,)).reshape(v["shape"])
            if name == PLE:
                for i in range(w.shape[1] // PLE_SLICE):
                    yield f"{name}[:, {i * PLE_SLICE}:{(i + 1) * PLE_SLICE}]", w[:, i * PLE_SLICE : (i + 1) * PLE_SLICE]
            else:
                yield name, w


def levels_and_step(w_u16):
    out_q, out_s = [], []
    for i in range(0, w_u16.shape[0], 4096):
        g = bf16_to_f32(np.asarray(w_u16[i : i + 4096])).reshape(-1, w_u16.shape[1] // QK, QK)
        s, _ = grid_step(g)
        out_q.append(np.clip(np.rint(g / np.where(s > 0, s, 1)), -8, 7).astype(np.int8).reshape(-1, w_u16.shape[1]))
        out_s.append(s[..., 0])
    return np.concatenate(out_q), np.concatenate(out_s)


def tflite_int4(path):
    """name -> (levels int8 [rows, cols], scales float32 [rows, cols/32]) for blockwise int4 tensors."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = fb.Model.GetRootAs(mm, 0)

    def data(bi):
        b = m.Buffers(bi)
        if b.Offset() > 1:
            return np.frombuffer(mm, np.uint8, count=b.Size(), offset=b.Offset())
        return b.DataAsNumpy()

    seen, out = set(), {}
    for si in range(m.SubgraphsLength()):
        sg = m.Subgraphs(si)
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            q = t.Quantization()
            if t.Type() != fb.TensorType.INT4 or q is None or t.Buffer() in seen:
                continue
            if q.DetailsType() != fb.QuantizationDetails.BlockwiseQuantization:
                continue
            seen.add(t.Buffer())
            d = fb.BlockwiseQuantization()
            d.Init(q.Details().Bytes, q.Details().Pos)
            assert d.BlockSize() == QK
            st = sg.Tensors(d.Scales())
            assert st.Type() == fb.TensorType.FLOAT16, st.Type()
            shape = [t.Shape(k) for k in range(t.ShapeLength())]
            raw = data(t.Buffer())
            lo = (raw & 0x0F).astype(np.int8)
            hi = (raw >> 4).astype(np.int8)
            nib = np.empty(raw.size * 2, np.int8)
            nib[0::2], nib[1::2] = lo, hi
            nib = np.where(nib > 7, nib - 16, nib).reshape(shape)
            scales = data(st.Buffer()).view(np.float16).astype(np.float32).reshape(shape[0], -1)
            out[f"{os.path.basename(path)}:{t.Name().decode()}"] = (nib, scales)
    return out


def main(src, unpacked):
    by_hash, used = {}, set()
    for name, w in source_tensors(src):
        q, s = levels_and_step(w)
        by_hash[hashlib.sha256(q.tobytes() + str(q.shape).encode()).hexdigest()] = (name, w, s)
    files = sorted(os.path.join(unpacked, f) for f in os.listdir(unpacked) if f.endswith(".tflite"))
    rows, unmatched = [], []

    def pieces(p):
        # A per-layer table stored whole is compared in the source's 256-column slices.
        for tname, (q, scales) in tflite_int4(p).items():
            if q.ndim == 2 and q.shape[1] > PLE_SLICE and q.shape[1] % PLE_SLICE == 0 and "per_layer" in p:
                k = PLE_SLICE // QK
                for i in range(q.shape[1] // PLE_SLICE):
                    yield (
                        f"{tname}[:, {i * PLE_SLICE}:{(i + 1) * PLE_SLICE}]",
                        np.ascontiguousarray(q[:, i * PLE_SLICE : (i + 1) * PLE_SLICE]),
                        scales[:, i * k : (i + 1) * k],
                    )
            else:
                yield tname, q, scales

    for p in files:
        for tname, q, scales in pieces(p):
            key = hashlib.sha256(q.tobytes() + str(q.shape).encode()).hexdigest()
            hit = by_hash.get(key)
            if hit is not None:
                used.add(key)
            if hit is None:
                unmatched.append({"tensor": tname, "shape": list(q.shape)})
                continue
            name, w, step = hit
            want = step.astype(ml_dtypes.bfloat16).astype(np.float16).astype(np.float32)
            deq = (q.reshape(q.shape[0], -1, QK) * scales[..., None]).reshape(q.shape)
            rows.append(
                {
                    "tensor": tname,
                    "source": name,
                    "values": int(q.size),
                    "blocks": int(scales.size),
                    "scales_equal_bf16_step": int((scales == want).sum()),
                    "values_exact": int((f32_to_bf16(deq) == np.asarray(w)).sum()),
                }
            )
            print(
                f"{name}: scales {rows[-1]['scales_equal_bf16_step'] / scales.size:.4%} "
                f"values {rows[-1]['values_exact'] / q.size:.4%}",
                file=sys.stderr,
                flush=True,
            )
    tot = {k: sum(r[k] for r in rows) for k in ("values", "blocks", "scales_equal_bf16_step", "values_exact")}
    json.dump(
        {
            "matched": len(rows),
            "unmatched_int4": unmatched,
            "source_never_matched": sorted(v[0] for k, v in by_hash.items() if k not in used),
            "total": tot,
            "tensors": rows,
        },
        sys.stdout,
        indent=1,
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
