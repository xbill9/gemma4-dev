"""Summarize how a .tflite (or every .tflite in an unpacked .litertlm) stores its weights.

    python3 inspect_tflite.py PATH [PATH ...] [--json]

For each model: signatures with their input shapes, then constant tensors grouped by
(storage type, quantization), with counts, values and bytes. Blockwise quantization
reports its block size; per-channel and per-tensor are told apart by the scale count.
Reads the flatbuffer in place (mmap), so multi-GB sections cost no RAM.
"""

import json
import mmap
import os
import sys
from collections import defaultdict

from ai_edge_litert import schema_py_generated as fb

TYPE = {v: k for k, v in vars(fb.TensorType).items() if k.isupper()}
BITS = {
    "INT2": 2,
    "INT4": 4,
    "UINT4": 4,
    "INT8": 8,
    "UINT8": 8,
    "INT16": 16,
    "FLOAT16": 16,
    "BFLOAT16": 16,
    "FLOAT32": 32,
    "INT32": 32,
    "INT64": 64,
}


def _buf_bytes(model, idx):
    if idx == 0 or idx >= model.BuffersLength():
        return 0
    b = model.Buffers(idx)
    return b.Size() if b.Offset() > 1 else b.DataLength()


def _quant(t):
    q = t.Quantization()
    if q is None:
        return "float"
    if q.DetailsType() == fb.QuantizationDetails.BlockwiseQuantization:
        d = fb.BlockwiseQuantization()
        d.Init(q.Details().Bytes, q.Details().Pos)
        return f"blockwise{d.BlockSize()}"
    n = q.ScaleLength()
    if n == 0:
        return "float"
    return "per-tensor" if n == 1 else "per-channel"


def summarize(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = fb.Model.GetRootAs(mm, 0)
    sigs = []
    for i in range(model.SignatureDefsLength()):
        s = model.SignatureDefs(i)
        sg = model.Subgraphs(s.SubgraphIndex())
        ins = {}
        for j in range(s.InputsLength()):
            t = sg.Tensors(s.Inputs(j).TensorIndex())
            ins[s.Inputs(j).Name().decode()] = [t.Shape(k) for k in range(t.ShapeLength())]
        sigs.append({"name": s.SignatureKey().decode(), "inputs": ins})
    groups = defaultdict(lambda: {"tensors": 0, "values": 0, "bytes": 0, "examples": []})
    seen = set()
    for si in range(model.SubgraphsLength()):
        sg = model.Subgraphs(si)
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            bi = t.Buffer()
            nbytes = _buf_bytes(model, bi)
            if not nbytes or bi in seen:
                continue
            seen.add(bi)
            n = 1
            for k in range(t.ShapeLength()):
                n *= t.Shape(k)
            key = f"{TYPE.get(t.Type(), t.Type())} {_quant(t)}"
            g = groups[key]
            g["tensors"] += 1
            g["values"] += n
            g["bytes"] += nbytes
            if len(g["examples"]) < 3 and n >= 1 << 16:
                g["examples"].append(f"{t.Name().decode()} {[t.Shape(k) for k in range(t.ShapeLength())]}")
    return {
        "file": path,
        "file_bytes": os.path.getsize(path),
        "signatures": sigs,
        "weights": dict(sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])),
    }


def main(argv):
    as_json = "--json" in argv
    paths = []
    for p in (a for a in argv if a != "--json"):
        if os.path.isdir(p):
            paths += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".tflite"))
        else:
            paths.append(p)
    out = [summarize(p) for p in paths]
    if as_json:
        json.dump(out, sys.stdout, indent=1)
        return
    for m in out:
        print(f"== {os.path.basename(m['file'])}  {m['file_bytes']:,} B")
        for s in m["signatures"]:
            print(f"   sig {s['name']}: " + ", ".join(f"{k}{v}" for k, v in s["inputs"].items()))
        for k, g in m["weights"].items():
            bits = 8 * g["bytes"] / g["values"] if g["values"] else 0
            print(
                f"   {k:28s} {g['tensors']:5d} tensors {g['values']:>14,} values {g['bytes']:>14,} B"
                f"  {bits:5.2f} bit/value  e.g. {'; '.join(g['examples'][:2])}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
