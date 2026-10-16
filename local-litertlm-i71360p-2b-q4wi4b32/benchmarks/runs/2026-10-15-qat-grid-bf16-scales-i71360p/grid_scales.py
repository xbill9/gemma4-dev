"""How exactly can LiteRT's int4 blockwise-32 format hold Gemma 4 E2B's QAT weights?

    python3 -I grid_scales.py SRC_HF_DIR > grid_scales.json

For every 2-D bf16 language-model tensor in google/gemma-4-E2B-it-qat-q4_0-unquantized:
- groups of 32 off the 4-bit grid (levels -8..7 times one step; same test as gguf_exact.py),
- the share of values reproduced exactly when the step is stored as
  (a) fp16, as Q4_0 does, (b) bf16 rounded the way ai_edge_quantizer rounds blockwise scales
      (float32 -> bfloat16 -> float16), and (c) ai_edge_quantizer's own default
      (scale = amax/7, then the same bf16 rounding, levels clipped to -8..7).
"Exact" = the dequantized value, rounded to bf16 as the source was, equals the source bits.
"""
import json, os, struct, sys
import numpy as np
import ml_dtypes

QK, REL_TOL = 32, 2.0 ** -7


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))
    return (u >> 16).astype(np.uint16)


def tensors(d):
    p = os.path.join(d, "model.safetensors")
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    for name, v in sorted(h.items()):
        if v["dtype"] == "BF16" and len(v["shape"]) == 2 and "language_model" in name and v["shape"][1] % QK == 0:
            a, b = v["data_offsets"]
            yield name, np.memmap(p, np.uint16, "r", offset=8 + n + a, shape=((b - a) // 2,)).reshape(v["shape"])


def grid_step(g):
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
    d = np.where(den > 0, num / np.where(den > 0, den, 1), d).astype(np.float32)
    return d, int((~ok).sum())


def exact(g, w_u16, s):
    q = np.clip(np.rint(g / np.where(s > 0, s, 1)), -8, 7)
    return int((f32_to_bf16((q * s).reshape(w_u16.shape)) == w_u16).sum())


def to_bf16_scale(x):
    return x.astype(ml_dtypes.bfloat16).astype(np.float16).astype(np.float32)


def main(src):
    out = {}
    for name, w in tensors(src):
        acc = dict(values=0, groups=0, off_grid=0, exact_fp16=0, exact_bf16=0, exact_aeq_default=0,
                   groups_step_bf16_exact=0)
        for i in range(0, w.shape[0], 2048):
            wu = np.asarray(w[i:i + 2048])
            g = bf16_to_f32(wu).reshape(wu.shape[0], -1, QK)
            d, bad = grid_step(g)
            acc["values"] += wu.size
            acc["groups"] += d.size
            acc["off_grid"] += bad
            s16 = d.astype(np.float16).astype(np.float32)
            sbf = to_bf16_scale(d)
            acc["groups_step_bf16_exact"] += int((sbf == s16).sum())
            acc["exact_fp16"] += exact(g, wu, s16)
            acc["exact_bf16"] += exact(g, wu, sbf)
            amax = np.maximum(np.abs(g).max(-1, keepdims=True), 1e-9)
            acc["exact_aeq_default"] += exact(g, wu, to_bf16_scale(amax / 7))
        out[name] = acc
        print(f"{name} {list(w.shape)} off_grid={acc['off_grid']} "
              f"fp16={acc['exact_fp16'] / acc['values']:.4%} bf16={acc['exact_bf16'] / acc['values']:.4%} "
              f"default={acc['exact_aeq_default'] / acc['values']:.4%}", file=sys.stderr, flush=True)
    tot = {k: sum(v[k] for v in out.values()) for k in next(iter(out.values()))}
    json.dump({"source": src, "tensors": out, "total": tot}, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
