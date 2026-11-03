"""Weight encoding of a .tflite section dumped from a .litertlm bundle.

    python3 -P tflite_encoding.py mix   TFLITE_SCHEMA_DIR FILE.tflite
    python3 -P tflite_encoding.py quant TFLITE_SCHEMA_DIR FILE.tflite

TFLITE_SCHEMA_DIR holds the PyPI `tflite` 2.18.0 package (unzipped wheel).
`mix` counts 2-D weight tensors by element type, role and shape, with buffers
deduplicated across subgraphs (prefill and decode share them). `quant` prints
the quantization parameters of the first three tensors of each integer type.
TensorType 19 is newer than that schema; it is INT2 (a 262144 x 1536 table
occupies 100,663,296 bytes, 2 bits per element).
"""

import collections
import mmap
import sys

ROLES = ("gating", "linear", "q_einsum", "kv_einsum", "attn_vec", "embedding", "per_layer")


def load(schema_dir, path):
    sys.path.insert(0, schema_dir)
    import tflite
    from tflite.TensorType import TensorType

    names = {v: k for k, v in vars(TensorType).items() if not k.startswith("_")}
    names[19] = "INT2"
    handle = open(path, "rb")
    data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    return tflite.Model.GetRootAsModel(data, 0), names, (handle, data)


def unique_tensors(model):
    seen = set()
    for si in range(model.SubgraphsLength()):
        graph = model.Subgraphs(si)
        for ti in range(graph.TensorsLength()):
            tensor = graph.Tensors(ti)
            buf = tensor.Buffer()
            if buf == 0 or buf in seen:
                continue
            size = model.Buffers(buf).Size() or model.Buffers(buf).DataLength()
            if not size:
                continue
            seen.add(buf)
            shape = tuple(int(x) for x in tensor.ShapeAsNumpy()) if tensor.ShapeLength() else ()
            yield tensor, size, shape


def mix(model, names):
    counts = collections.Counter()
    total = 0
    for tensor, size, shape in unique_tensors(model):
        total += size
        if len(shape) != 2:
            continue
        name = tensor.Name().decode()
        role = next((r for r in ROLES if r in name), "other")
        counts[(names.get(tensor.Type(), tensor.Type()), role, shape)] += 1
    print("unique const bytes", total)
    for (kind, role, shape), n in sorted(counts.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        print(f"{kind:6s} {role:10s} {str(shape):18s} x{n}")


def quant(model, names):
    shown = collections.Counter()
    for tensor, _size, shape in unique_tensors(model):
        kind = tensor.Type()
        if len(shape) != 2 or kind not in (3, 9, 17, 19) or shown[kind] >= 3:
            continue
        shown[kind] += 1
        q = tensor.Quantization()
        if q is None:
            print(names.get(kind, kind), shape, "no quantization")
            continue
        zp = q.ZeroPointAsNumpy() if q.ZeroPointLength() else []
        nonzero = int((zp != 0).sum()) if len(zp) else 0
        print(
            f"{names.get(kind, kind)} {shape} scales={q.ScaleLength()} zp={q.ZeroPointLength()}"
            f" zp_nonzero={nonzero} qdim={q.QuantizedDimension()} details_type={q.DetailsType()}"
            f" name={tensor.Name().decode()[-60:]}"
        )
        if q.ScaleLength():
            print("    scale[0:4]", q.ScaleAsNumpy()[:4])


def main():
    mode, schema_dir, path = sys.argv[1:4]
    model, names, _keep = load(schema_dir, path)
    {"mix": mix, "quant": quant}[mode](model, names)


if __name__ == "__main__":
    main()
