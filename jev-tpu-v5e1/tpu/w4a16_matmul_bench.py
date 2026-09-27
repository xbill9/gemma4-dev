#!/usr/bin/env python3
"""Per-layer W4A16 matmul timing on one TPU chip, inside the patched vLLM TPU image.

For Gemma 4 projection shapes and a few token counts, times three ways to compute
x @ W with the same weights:
  bf16    dense bf16 weight (the speed a 4-bit path should beat: it reads 4x the bytes)
  xla     int4 weight, xla_quantized_matmul with a 2D group scale (dequantize, then matmul)
  kernel  int4 weight, gmm_v2 with a 3D group scale (dequantize per tile in VMEM)
and checks kernel output against xla output (relative error, NaN/inf). Prints one JSON
line per case and a summary table; every figure is computed here.
"""
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.gmm_v2.gmm_v2 import gmm_v2

from tpu_inference.layers.common.linear import xla_quantized_matmul

GROUP = 32
SHAPES = [  # (label, in_features, out_features)
    ("31B gate_up", 5376, 43008),
    ("31B down", 21504, 5376),
    ("31B q (full attn)", 5376, 16384),
    ("31B o (sliding)", 8192, 5376),
    ("12B gate_up", 3840, 30720),
    ("12B down", 15360, 3840),
    ("E4B gate_up", 2560, 20480),
    ("E4B down", 10240, 2560),
]
TOKENS = [1, 16, 64]
ITERS = 50
REPS = 5


def timed(fn, *args):
    """Device time per call: ITERS calls enqueued back to back, one wait at the end.

    Timing each call on its own adds host dispatch (about 150 us here), which
    swamps the small shapes; pipelining keeps the device busy so the host cost
    overlaps. The median of REPS such batches is reported.
    """
    out = fn(*args)
    out.block_until_ready()  # compile + warm
    for _ in range(3):
        fn(*args).block_until_ready()
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            last = fn(*args)
        last.block_until_ready()
        samples.append((time.perf_counter() - t0) / ITERS)
    return out, statistics.median(samples)


@jax.jit
def bf16_mm(x, w):
    return jnp.dot(x, w, preferred_element_type=jnp.float32).astype(x.dtype)


@jax.jit
def xla_mm(x, w_q, s2):
    return xla_quantized_matmul(x, w_q, s2, quantize_activation=False)


@jax.jit
def kernel_mm(x, w_q, s3):
    return gmm_v2(lhs=x,
                  rhs=jnp.expand_dims(w_q, 0),
                  group_sizes=jnp.array([x.shape[0]], dtype=jnp.int32),
                  rhs_scale=jnp.expand_dims(s3, 0),
                  rhs_bias=None,
                  group_offset=jnp.array([0], dtype=jnp.int32),
                  zero_initialize=False,
                  preferred_element_type=x.dtype,
                  maybe_quantize_lhs=False)


def main():
    print(json.dumps({"device": str(jax.devices()[0]), "jax": jax.__version__}))
    rng = np.random.default_rng(0)
    rows = []
    for label, n_in, n_out in SHAPES:
        q = rng.integers(-8, 8, size=(n_in, n_out), dtype=np.int8)
        s = rng.uniform(0.001, 0.02, size=(n_in // GROUP, n_out)).astype(np.float32)
        w_q = jnp.asarray(q).astype(jnp.int4)
        s2 = jnp.asarray(s, jnp.bfloat16)
        s3 = s2[:, None, :]
        w_bf16 = (jnp.asarray(q, jnp.float32).reshape(n_in // GROUP, GROUP, n_out)
                  * s2.astype(jnp.float32)[:, None, :]).reshape(n_in, n_out).astype(jnp.bfloat16)
        for m in TOKENS:
            x = jnp.asarray(rng.standard_normal((m, n_in)), jnp.bfloat16)
            _, t_bf16 = timed(bf16_mm, x, w_bf16)
            y_xla, t_xla = timed(xla_mm, x, w_q, s2)
            y_k, t_k = timed(kernel_mm, x, w_q, s3)
            a = np.asarray(y_k, np.float32)
            b = np.asarray(y_xla, np.float32)
            row = {"shape": label, "in": n_in, "out": n_out, "tokens": m,
                   "bf16_us": round(t_bf16 * 1e6, 1), "xla_us": round(t_xla * 1e6, 1),
                   "kernel_us": round(t_k * 1e6, 1),
                   "kernel_vs_bf16": round(t_bf16 / t_k, 2),
                   "kernel_vs_xla": round(t_xla / t_k, 2),
                   "xla_vs_bf16": round(t_bf16 / t_xla, 2),
                   "kernel_rel_err_vs_xla": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                   "kernel_nonfinite": int((~np.isfinite(a)).sum())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    print("\n| shape | tokens | bf16 us | xla us | kernel us | kernel speed vs bf16 | vs xla | rel err | non-finite |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['shape']} | {r['tokens']} | {r['bf16_us']} | {r['xla_us']} | {r['kernel_us']} | "
              f"{r['kernel_vs_bf16']}x | {r['kernel_vs_xla']}x | {r['kernel_rel_err_vs_xla']:.1e} | {r['kernel_nonfinite']} |")
    worst = max(r["kernel_rel_err_vs_xla"] for r in rows)
    bad = sum(r["kernel_nonfinite"] for r in rows)
    print(f"\nkernel faster than bf16 in {sum(r['kernel_vs_bf16'] > 1 for r in rows)} of {len(rows)} cases; "
          f"faster than xla in {sum(r['kernel_vs_xla'] > 1 for r in rows)} of {len(rows)}; "
          f"worst rel err vs xla {worst:.1e}; non-finite outputs {bad}")


if __name__ == "__main__":
    main()
