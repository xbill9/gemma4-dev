"""Write a QAT Q4_0 repack's exact int4 codes and scales into converted LiteRT graphs.

litert-torch's `dynamic_wi4b32_afp32` recipe quantizes each block of 32 weights
itself: codes -7..7 and scale = max|w| / 7, which moves values off the trained
QAT grid. This rewrites every blockwise-int4 tensor in the given .tflite files
with the codes (-8..7) and fp16 scales of a compressed-tensors pack-quantized
group-32 repack (xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4), then
re-reads the output and checks every code and scale.

Tensors are matched to repack weights by value, not name: for each tflite
tensor, among repack weights of the same shape (the per-layer embedding table
is split into 256-column slices), the one whose dequantized rows are closest.
The match must be unambiguous (best error well under the runner-up) or the
script stops.

    python3 -P repack_litertlm.py TFLITE_SCHEMA_DIR REPACK_DIR OUT_DIR IN.tflite...

TFLITE_SCHEMA_DIR holds the PyPI `tflite` 2.18.0 package. Each IN.tflite is
copied to OUT_DIR under its own name and edited there.
"""

import json
import mmap
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

BLOCK = 32
SAMPLE_ROWS = 64
PER_LAYER_KEY = "model.language_model.embed_tokens_per_layer.weight"
PER_LAYER_DIM = 256


def load_schema(schema_dir):
    sys.path.insert(0, schema_dir)
    import flatbuffers
    import tflite
    from flatbuffers.table import Table

    return tflite, flatbuffers, Table


class Repack:
    """compressed-tensors pack-quantized int4, group 32: nibbles offset by 8."""

    def __init__(self, path):
        self.dir = Path(path)
        self.fused = {}
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.keys = sorted(k[: -len("_packed")] for k in self.map if k.endswith(".weight_packed"))
        self.shapes = {k: tuple(self._get(k + "_shape").tolist()) for k in self.keys}
        # litert-torch's GPU composites fuse q/k/v and gate/up into one weight,
        # rows concatenated in this order; offer those concatenations as sources.
        for k in list(self.keys):
            for parts in (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")):
                if k.endswith(f".{parts[0]}.weight"):
                    members = [k.replace(parts[0], x) for x in parts]
                    if all(m in self.shapes for m in members):
                        fused = k.replace(parts[0], "+".join(parts))
                        rows = sum(self.shapes[m][0] for m in members)
                        self.fused[fused] = members
                        self.shapes[fused] = (rows, self.shapes[k][1])
        self._cache = {}

    def _get(self, name):
        import torch  # numpy cannot read bfloat16 scales

        with safe_open(self.dir / self.map[name], "pt") as f:
            t = f.get_tensor(name)
        return (t.float() if t.dtype == torch.bfloat16 else t).numpy()

    def codes_scales(self, key):
        if key in self.fused:
            parts = [self.codes_scales(m) for m in self.fused[key]]
            return np.concatenate([c for c, _ in parts]), np.concatenate([sc for _, sc in parts])
        if key not in self._cache:
            # int32 little-endian, nibble i at bits 4i: byte j holds nibbles 2j (low), 2j+1 (high).
            packed = self._get(key + "_packed")
            raw = packed.view(np.uint8).reshape(packed.shape[0], -1)
            codes = np.empty((raw.shape[0], raw.shape[1] * 2), np.int8)
            codes[:, 0::2] = (raw & 0xF).astype(np.int8) - 8
            codes[:, 1::2] = (raw >> 4).astype(np.int8) - 8
            del raw, packed
            scales = self._get(key + "_scale").astype(np.float32).astype(np.float16)
            self._cache = {key: (codes, scales)}
        return self._cache[key]

    def candidates(self, shape):
        """(label, key, column_slice) for every repack weight of this shape."""
        out = []
        for k, s in self.shapes.items():
            if s == shape:
                out.append((k, k, None))
            elif k == PER_LAYER_KEY and shape == (s[0], PER_LAYER_DIM):
                for layer in range(s[1] // PER_LAYER_DIM):
                    cols = slice(layer * PER_LAYER_DIM, (layer + 1) * PER_LAYER_DIM)
                    out.append((f"{k}[layer {layer}]", k, cols))
        return out

    def slice(self, key, cols, rows=None):
        codes, scales = self.codes_scales(key)
        if rows is not None:
            codes, scales = codes[rows], scales[rows]
        if cols is not None:
            codes = codes[:, cols]
            scales = scales[:, cols.start // BLOCK : cols.stop // BLOCK]
        return codes, scales


def dequant(codes, scales):
    return codes.astype(np.float32) * np.repeat(scales.astype(np.float32), BLOCK, axis=1)


class TfliteFile:
    def __init__(self, schema, path, writable=False):
        self.tflite, self.fb, self.Table = schema
        self.handle = open(path, "r+b" if writable else "rb")
        access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
        self.mm = mmap.mmap(self.handle.fileno(), 0, access=access)
        self.model = self.tflite.Model.GetRootAsModel(self.mm, 0)

    def close(self):
        self.mm.flush()
        self.model = None
        try:
            self.mm.close()
        except BufferError:
            pass  # numpy views still alive; data is flushed, the map goes with them
        self.handle.close()

    def buffer_view(self, index):
        """Writable/readable uint8 view of a buffer's bytes, inline or offset."""
        buf = self.model.Buffers(index)
        if buf.Offset() > 1:
            start, size = buf.Offset(), buf.Size()
        else:
            o = buf._tab.Offset(4)
            if not o:
                return None
            start = buf._tab.Vector(o)
            size = buf._tab.VectorLen(o)
        return np.frombuffer(self.mm, np.uint8, size, start)

    def blockwise_tensors(self):
        """Unique (by buffer) blockwise-int4 weights: (name, shape, weight_buffer, scale_buffer)."""
        seen = set()
        for si in range(self.model.SubgraphsLength()):
            g = self.model.Subgraphs(si)
            for ti in range(g.TensorsLength()):
                t = g.Tensors(ti)
                q = t.Quantization()
                if q is None or q.DetailsType() != 2 or t.Type() != 17 or t.Buffer() in seen:
                    continue
                details = q.Details()
                table = self.Table(details.Bytes, details.Pos)
                off = table.Offset(4)
                block = table.Get(self.fb.number_types.Int32Flags, table.Offset(8) + table.Pos)
                if block != BLOCK:
                    raise SystemExit(f"{t.Name().decode()[:80]}: block size {block}, expected {BLOCK}")
                scale_tensor = g.Tensors(table.Get(self.fb.number_types.Int32Flags, off + table.Pos))
                if scale_tensor.Type() != 1:
                    raise SystemExit("scale tensor is not float16")
                seen.add(t.Buffer())
                shape = tuple(int(x) for x in t.ShapeAsNumpy())
                yield t.Name().decode(), shape, t.Buffer(), scale_tensor.Buffer()


def unpack_tfl(raw, shape):
    lo = (raw & 0xF).astype(np.int8)
    hi = (raw >> 4).astype(np.int8)
    nib = np.stack([lo, hi], -1).reshape(shape)
    return np.where(nib > 7, nib - 16, nib).astype(np.int8)


def pack_tfl(codes):
    u = (codes.astype(np.int16) & 0xF).astype(np.uint8).reshape(-1, 2)
    return (u[:, 0] | (u[:, 1] << 4)).astype(np.uint8)


def match(repack, tf, shape, wbuf, sbuf, used):
    rows = np.linspace(0, shape[0] - 1, SAMPLE_ROWS).astype(int)
    row_bytes = shape[1] // 2
    raw = tf.buffer_view(wbuf).reshape(shape[0], row_bytes)[rows]
    sc = tf.buffer_view(sbuf).view(np.float16).reshape(shape[0], -1)[rows]
    mine = dequant(unpack_tfl(raw, (len(rows), shape[1])), sc)
    scored = []
    for label, key, cols in repack.candidates(shape):
        codes, scales = repack.slice(key, cols, rows)
        scored.append((float(np.abs(dequant(codes, scales) - mine).mean()), label, key, cols))
    if not scored:
        raise SystemExit(f"no repack weight of shape {shape}")
    scored.sort(key=lambda x: x[0])
    if len(scored) > 1 and not scored[0][0] < 0.5 * scored[1][0]:
        # Tied tables (embed_tokens / lm_head) can hold identical values; then
        # either is the right source. Anything else is a real ambiguity.
        a, b = scored[0], scored[1]
        same = np.array_equal(repack.slice(a[2], a[3], rows)[0], repack.slice(b[2], b[3], rows)[0])
        if not same:
            raise SystemExit(f"ambiguous match for {shape}: {scored[:2]}")
        unused = [s for s in scored[:2] if s[1] not in used]
        return unused[0] if unused else scored[0]
    return scored[0]


def main(argv):
    schema = load_schema(argv[1])
    repack = Repack(argv[2])
    out_dir = Path(argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    used = {}
    for src in argv[4:]:
        dst = out_dir / Path(src).name
        shutil.copyfile(src, dst)
        tf = TfliteFile(schema, dst, writable=True)
        tensors = list(tf.blockwise_tensors())
        for name, shape, wbuf, sbuf in tensors:
            err, label, key, cols = match(repack, tf, shape, wbuf, sbuf, used)
            if label in used:
                raise SystemExit(f"{label} matched twice ({used[label]} and {name[:60]})")
            used[label] = name
            codes, scales = repack.slice(key, cols)
            old = unpack_tfl(tf.buffer_view(wbuf), shape)
            changed = float((old != codes).mean())
            if changed > 0.40:
                raise SystemExit(f"{label}: {changed:.1%} of codes differ from the converter's; wrong match?")
            tf.buffer_view(wbuf)[:] = pack_tfl(codes)
            tf.buffer_view(sbuf).view(np.float16)[:] = scales.reshape(-1)
            report.append(
                {
                    "file": dst.name,
                    "shape": shape,
                    "repack": label,
                    "match_err": err,
                    "codes_changed": changed,
                    "codes_minus8": int((codes == -8).sum()),
                }
            )
            print(f"{dst.name} {shape} <- {label}  match {err:.2e}  codes changed {changed:.1%}", flush=True)
        tf.close()
        # Verify from a fresh read.
        tf = TfliteFile(schema, dst)
        for name, shape, wbuf, sbuf in tf.blockwise_tensors():
            label = next(r["repack"] for r in report if r["file"] == dst.name and used[r["repack"]] == name)
            key = label.split("[")[0]
            cols = None
            if "[layer " in label:
                layer = int(label.split("[layer ")[1].rstrip("]"))
                cols = slice(layer * PER_LAYER_DIM, (layer + 1) * PER_LAYER_DIM)
            codes, scales = repack.slice(key, cols)
            got_c = unpack_tfl(tf.buffer_view(wbuf), shape)
            got_s = tf.buffer_view(sbuf).view(np.float16).reshape(scales.shape)
            if not (np.array_equal(got_c, codes) and np.array_equal(got_s.view(np.uint16), scales.view(np.uint16))):
                raise SystemExit(f"verify failed: {label}")
        tf.close()
        print(f"{dst.name}: {len(tensors)} tensors rewritten and verified", flush=True)
    covered = set(used)
    for label in used:
        covered.update(repack.fused.get(label, []))
    unused = [k for k in repack.keys if k not in covered and k != PER_LAYER_KEY]
    (out_dir / "repack_report.json").write_text(
        json.dumps({"tensors": report, "repack_weights_not_in_graphs": unused}, indent=1) + "\n"
    )
    print(f"done: {len(report)} tensors; repack weights with no graph tensor: {len(unused)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
