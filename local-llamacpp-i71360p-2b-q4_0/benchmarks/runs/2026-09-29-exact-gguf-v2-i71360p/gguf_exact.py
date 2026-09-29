"""Rebuild Google's Gemma 4 E2B q4_0 GGUF with every Q4_0 tensor, and both embedding
tables, taken exactly from the QAT `-qat-q4_0-unquantized` source.

    python3 gguf_exact.py SRC_HF_DIR GOOGLE_GGUF OUT_GGUF [--bf16]

--bf16 writes the source bytes unchanged instead (a reference with identical metadata).

Metadata (tokenizer, chat template, hparams) is copied byte for byte from Google's
file; tensor order and every other tensor are unchanged. A tensor with any group off
the 4-bit grid stops the build (nothing is re-gridded). Each 32-value group's step is
amax/m for the first m in 1..8 that puts every value on an integer level, refined by
least squares and rounded to fp16 (Q4_0's scale type).
"""
import json, os, struct, sys
import numpy as np

REL_TOL = 2.0 ** -7  # a group is on the grid when every value is within this of amax


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    """Round to nearest even."""
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))
    return (u >> 16).astype(np.uint16)


def open_safetensors(d):
    """name -> read-only uint16 memmap of a bf16 tensor, for every bf16 tensor under d."""
    idx = os.path.join(d, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    out = {}
    for fn in files:
        p = os.path.join(d, fn)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for name, v in h.items():
            if v["dtype"] == "BF16":
                a, b = v["data_offsets"]
                out[name] = np.memmap(p, np.uint16, "r", offset=8 + n + a, shape=((b - a) // 2,)).reshape(v["shape"])
    return out

Q4_0, BF16, QK, BLOCK_BYTES = 2, 30, 32, 18
TYPE_SIZE = {0: (1, 4), 1: (1, 2), 2: (32, 18), 14: (256, 210), 30: (1, 2)}  # type -> (block elems, bytes)

HF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
      "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
      "ffn_down": "mlp.down_proj", "inp_gate": "per_layer_input_gate", "proj": "per_layer_projection"}
TOP = {"token_embd.weight": "model.language_model.embed_tokens.weight",
       "per_layer_token_embd.weight": "model.language_model.embed_tokens_per_layer.weight",
       # F16 in Google's file; QAT data like the rest (0 of 430,080 groups off the grid).
       "per_layer_model_proj.weight": "model.language_model.per_layer_model_projection.weight"}


def hf_name(g):
    if g in TOP:
        return TOP[g]
    p = g.split(".")
    if p[0] == "blk" and p[2] in HF:
        return f"model.language_model.layers.{p[1]}.{HF[p[2]]}.weight"
    return None


def quantize_fp16(w_u16):
    """bf16 bits [rows, cols] -> (q int8, d fp16 [rows, cols/32], n_off_grid, n_values_exact)."""
    rows, cols = w_u16.shape
    g = bf16_to_f32(w_u16).reshape(rows, cols // QK, QK)
    amax = np.abs(g).max(-1, keepdims=True)
    nz = amax > 0
    tol = amax * REL_TOL
    d = np.where(nz, amax / 8, 0).astype(np.float32)
    ok = ~nz
    for m in range(1, 9):
        cand = np.where(nz, amax / m, 1).astype(np.float32)
        q = np.clip(np.rint(g / cand), -8, 7)
        good = (np.abs(g - q * cand).max(-1, keepdims=True) <= tol) & nz & ~ok
        d = np.where(good, cand, d)
        ok |= good
    q = np.clip(np.rint(g / np.where(d > 0, d, 1)), -8, 7)
    den, num = (q * q).sum(-1, keepdims=True), (q * g).sum(-1, keepdims=True)
    d = np.where(den > 0, num / np.where(den > 0, den, 1), d).astype(np.float16)
    s = d.astype(np.float32)
    q = np.clip(np.rint(g / np.where(s > 0, s, 1)), -8, 7).astype(np.int8)
    # Exact = the reconstruction, rounded to bf16 as the source was, is the source value.
    exact = int((f32_to_bf16((q * s).reshape(rows, cols)) == w_u16).sum())
    return q, d[..., 0], int((~ok).sum()), exact


def q4_0_bytes(q, d):
    """q int8 [rows, nb, 32], d fp16 [rows, nb] -> uint8 [rows, nb*18] in ggml block_q4_0 layout."""
    u = (q.astype(np.int16) + 8).astype(np.uint8)
    qs = u[..., :16] | (u[..., 16:] << 4)
    out = np.empty(q.shape[:2] + (BLOCK_BYTES,), np.uint8)
    out[..., :2] = d[..., None].view(np.uint8).reshape(q.shape[:2] + (2,))
    out[..., 2:] = qs
    return out.reshape(q.shape[0], -1)


class GGUF:
    """Just enough of the format: raw header+KV bytes, tensor infos, data offset."""
    def __init__(self, path):
        self.path = path
        f = self.f = open(path, "rb")
        r = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
        assert f.read(4) == b"GGUF"
        r("I")
        n_t, n_kv = r("QQ")
        self.align = 32

        def rs():
            return f.read(r("Q")[0]).decode()

        def rv(t):
            if t == 8:
                return rs()
            if t == 9:
                at, n = r("IQ")
                return [rv(at) for _ in range(n)]
            return r({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t])[0]
        for _ in range(n_kv):
            k = rs()
            v = rv(r("I")[0])
            if k == "general.alignment":
                self.align = v
        self.kv_end = f.tell()
        self.tensors = []
        for _ in range(n_t):
            name = rs()
            nd = r("I")[0]
            dims = r("Q" * nd)
            t, off = r("IQ")
            self.tensors.append([name, dims, t, off])
        p = f.tell()
        self.data_start = p + (-p % self.align)

    def nbytes(self, dims, t):
        be, bb = TYPE_SIZE[t]
        return int(np.prod(dims)) // be * bb

    def read(self, name, dims, t, off):
        self.f.seek(self.data_start + off)
        return self.f.read(self.nbytes(dims, t))


def main(src, gpath, out, bf16=False):
    ck = open_safetensors(src)
    g = GGUF(gpath)
    plan, report = [], {}
    for name, dims, t, off in g.tensors:
        h = hf_name(name)
        plan.append((name, dims, (BF16 if bf16 else Q4_0) if h else t, off, t, h))

    with open(out + ".tmp", "wb") as o:
        g.f.seek(0)
        o.write(g.f.read(g.kv_end))
        off, offs = 0, []
        for name, dims, t, _, _, _ in plan:
            offs.append(off)
            n = g.nbytes(dims, t)
            off += n + (-n % g.align)
        for (name, dims, t, _, _, _), o_ in zip(plan, offs):
            nb = name.encode()
            o.write(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)))
            o.write(struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", t, o_))
        o.write(b"\0" * (-o.tell() % g.align))
        for name, dims, t, goff, gt, h in plan:
            if h is None:
                o.write(g.read(name, dims, gt, goff))
            else:
                w = ck[h]
                assert tuple(w.shape) == tuple(reversed(dims)), (name, w.shape, dims)
                if bf16:
                    o.write(np.asarray(w).tobytes())
                    o.write(b"\0" * (-o.tell() % g.align))
                    continue
                chunk, parts, off_grid, exact = 4096, [], 0, 0
                for i in range(0, w.shape[0], chunk):
                    q, d, bad, ex = quantize_fp16(np.asarray(w[i:i + chunk]))
                    off_grid += bad
                    exact += ex
                    parts.append(q4_0_bytes(q, d).tobytes())
                if off_grid:
                    raise SystemExit(f"{name}: {off_grid} groups off the 4-bit grid; not handled")
                o.write(b"".join(parts))
                kind = name.split(".")[2] if name.startswith("blk") else name
                a = report.setdefault(kind, {"tensors": 0, "values": 0, "exact": 0, "was": TYPE_NAMES.get(gt, gt)})
                a["tensors"] += 1
                a["values"] += w.size
                a["exact"] += exact
                print(f"{name}: {exact / w.size:.6%} exact", file=sys.stderr, flush=True)
            o.write(b"\0" * (-o.tell() % g.align))
    os.replace(out + ".tmp", out)
    for a in report.values():
        a["share_exact"] = a["exact"] / a["values"]
    print(json.dumps({"out_bytes": os.path.getsize(out), "google_bytes": os.path.getsize(gpath),
                      "rebuilt": report}, indent=2))


TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 14: "Q6_K", 30: "BF16"}

if __name__ == "__main__":
    main(*sys.argv[1:4], bf16="--bf16" in sys.argv)
