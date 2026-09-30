#!/usr/bin/env python3
"""E2B vocabulary-layer timing on one TPU chip, inside the patched vLLM TPU image.

The emb4 checkpoint (int4 embedding tables and lm_head) decoded at 0.63x the W4A16 repack
whose only difference is those tables in bf16. This times each vocabulary layer both ways,
with the patched tpu_inference code where it exists:

lm_head, [tokens, 1536] @ [1536, 262144]:
  bf16          dense bf16 weight (the repack's tied head)
  kernel_f32    int4, gmm_v2 with a float32 [groups, 1, out] scale (what emb4 served)
  kernel_bf16   int4, gmm_v2 with a bf16 scale (what the W4A16 linears use)
  xla           int4, xla_quantized_matmul with a 2D bf16 scale
  chunked       WNA16EmbedMethod.decode: packed table unpacked 8192 rows at a time
lookup, [tokens] ids into embed_tokens (262144 x 1536) and embed_tokens_per_layer (262144 x 8960):
  bf16          jnp.take on the bf16 table
  int4          WNA16EmbedMethod.apply_jax on the packed table with fp16 scales

Prints one JSON line per case and a table; every ratio is computed here.
"""
import json
import statistics
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.gmm_v2.gmm_v2 import gmm_v2

from tpu_inference.layers.common.linear import xla_quantized_matmul
from tpu_inference.layers.jax.quantization.wna16 import WNA16EmbedMethod

V, H, PLE, G = 262144, 1536, 8960, 32
TOKENS = [1, 4, 16, 64]
ITERS = 30
REPS = 5


def timed(fn, *args):
    """Median device time per call over REPS batches of ITERS back-to-back calls."""
    out = fn(*args)
    out.block_until_ready()
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


def kernel(x, w_q, s3):
    return gmm_v2(lhs=x, rhs=jnp.expand_dims(w_q, 0),
                  group_sizes=jnp.array([x.shape[0]], dtype=jnp.int32),
                  rhs_scale=jnp.expand_dims(s3, 0), rhs_bias=None,
                  group_offset=jnp.array([0], dtype=jnp.int32), zero_initialize=False,
                  preferred_element_type=x.dtype, maybe_quantize_lhs=False)


def emb_method(features):
    return WNA16EmbedMethod(SimpleNamespace(num_embeddings=V, features=features,
                                            param_dtype=jnp.bfloat16), G, "bench")


def rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    print(json.dumps({"device": str(jax.devices()[0]), "jax": jax.__version__}), flush=True)
    rows = []
    key = jax.random.key(0)

    # lm_head. Levels and scales are random; the same values feed every path.
    q = jax.random.randint(key, (V, H), -8, 8, jnp.int32)             # [vocab, hidden] levels
    s = jax.random.uniform(key, (V, H // G), jnp.float32, 1e-3, 2e-2)  # [vocab, groups]
    s16 = s.astype(jnp.float16)
    sf = s16.astype(jnp.float32)
    shifts = jnp.arange(8, dtype=jnp.int32) * 4
    packed = jnp.sum(((q + 8).reshape(V, H // 8, 8) << shifts), axis=-1, dtype=jnp.int32)
    w_q = q.T.astype(jnp.int4)                                           # [hidden, vocab]
    s3_f32 = sf.T[:, None, :]
    s3_bf16 = sf.T.astype(jnp.bfloat16)[:, None, :]
    s2_bf16 = sf.T.astype(jnp.bfloat16)
    w_bf16 = (q.astype(jnp.float32).reshape(V, H // G, G) * sf[..., None]).reshape(V, H).T.astype(jnp.bfloat16)
    del q, s
    head = emb_method(H)
    table = SimpleNamespace(weight_packed=packed, weight_scale=s16)
    paths = {
        "bf16": (jax.jit(lambda x, w: jnp.dot(x, w, preferred_element_type=jnp.float32).astype(x.dtype)), (w_bf16,)),
        "kernel_f32": (jax.jit(kernel), (w_q, s3_f32)),
        "kernel_bf16": (jax.jit(kernel), (w_q, s3_bf16)),
        "xla": (jax.jit(lambda x, w, s2: xla_quantized_matmul(x, w, s2, quantize_activation=False)), (w_q, s2_bf16)),
        "chunked": (jax.jit(lambda x, p, sc: head.decode(SimpleNamespace(weight_packed=p, weight_scale=sc), x)),
                    (table.weight_packed, table.weight_scale)),
    }
    rng = np.random.default_rng(0)
    for m in TOKENS:
        x = jnp.asarray(rng.standard_normal((m, H)), jnp.bfloat16)
        ref, t_ref = timed(paths["bf16"][0], x, *paths["bf16"][1])
        for name, (fn, args) in paths.items():
            try:
                out, t = (ref, t_ref) if name == "bf16" else timed(fn, x, *args)
                row = {"layer": "lm_head", "path": name, "tokens": m, "us": round(t * 1e6, 1),
                       "vs_bf16": round(t_ref / t, 2), "rel_err_vs_bf16": rel(out, ref)}
            except Exception as e:  # noqa: BLE001 - one failing path must not hide the rest
                row = {"layer": "lm_head", "path": name, "tokens": m, "error": f"{type(e).__name__}: {e}"[:300]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    del w_bf16, w_q, s3_f32, s3_bf16, s2_bf16, packed, s16, sf, table, paths

    # Lookups: values do not change the work, so bf16 tables are filled cheaply.
    for label, d in (("embed_tokens", H), ("embed_tokens_per_layer", PLE)):
        dense = jnp.ones((V, d), jnp.bfloat16)
        pk = jax.random.randint(key, (V, d // 8), 0, 2**31 - 1, jnp.int32)
        sc = jnp.full((V, d // G), 0.01, jnp.float16)
        meth = emb_method(d)
        f_dense = jax.jit(lambda ids, t: jnp.take(t, ids, axis=0))
        f_int4 = jax.jit(lambda ids, p, c: meth.apply_jax(SimpleNamespace(weight_packed=p, weight_scale=c), ids))
        for m in TOKENS:
            ids = jnp.asarray(rng.integers(0, V, m), jnp.int32)
            _, t_d = timed(f_dense, ids, dense)
            _, t_i = timed(f_int4, ids, pk, sc)
            for name, t in (("bf16", t_d), ("int4", t_i)):
                row = {"layer": label, "path": name, "tokens": m, "us": round(t * 1e6, 1),
                       "vs_bf16": round(t_d / t, 2)}
                rows.append(row)
                print(json.dumps(row), flush=True)
        del dense, pk, sc

    print("\n| layer | path | tokens | us | speed vs bf16 | rel err vs bf16 |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows:
        if "error" in r:
            print(f"| {r['layer']} | {r['path']} | {r['tokens']} | error | | {r['error'][:80]} |")
        else:
            err = f"{r['rel_err_vs_bf16']:.1e}" if "rel_err_vs_bf16" in r else ""
            print(f"| {r['layer']} | {r['path']} | {r['tokens']} | {r['us']} | {r['vs_bf16']}x | {err} |")
    served = [r for r in rows if r["layer"] == "lm_head" and r["path"] == "kernel_f32" and "us" in r]
    base = {r["tokens"]: r["us"] for r in rows if r["layer"] == "lm_head" and r["path"] == "bf16"}
    print("\nserved int4 lm_head minus bf16, us per step: "
          + ", ".join(f"{r['tokens']} tok {round(r['us'] - base[r['tokens']], 1)}" for r in served))


if __name__ == "__main__":
    main()
