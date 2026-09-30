#!/usr/bin/env python3
"""int4 embedding-lookup layouts on one TPU chip, inside the patched vLLM TPU image.

With more than one id, the int4 lookup costs as much as reading the whole packed table
(v5e: 4.2 ms for the E2B per-layer table, 0.87 ms for embed_tokens, against ~0.06 ms for the
bf16 gather), with or without an optimization barrier before the unpack. The packed rows
(1120 and 192 int32 words, 480 for 12B) are not multiples of 128 lanes; the bf16 rows are.
This times the same lookup with the tables laid out four ways:

  current   packed int32 [V, D/8] and fp16 scales [V, D/32], two gathers (as served)
  padded    both tables padded to a multiple of 128 columns, two gathers
  fused     scales bitcast into the packed row, padded to 128: one int32 gather
  dslice    current layout, one dynamic_slice per id (lax.map)

and checks every layout returns the same rows as `current`. Prints JSON lines and a table.
"""
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

V, G = 262144, 32
TABLES = [("E2B embed_tokens", 1536), ("E2B embed_tokens_per_layer", 8960), ("12B embed_tokens", 3840)]
TOKENS = [1, 4, 16, 64, 512]
ITERS, REPS = 30, 5


def timed(fn, *args):
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


def ceil128(n):
    return -(-n // 128) * 128


def unpack(packed, scale, d):
    """int32 [..., >= d/8] and float [..., >= d/32] -> bf16 [..., d]."""
    words, groups = d // 8, d // G
    shifts = jnp.arange(8, dtype=jnp.int32) * 4
    q = (jnp.right_shift(packed[..., :words, None], shifts) & 0xF) - 8
    w = q.reshape(*packed.shape[:-1], groups, G).astype(jnp.float32)
    w = w * scale[..., :groups, None].astype(jnp.float32)
    return w.reshape(*packed.shape[:-1], d).astype(jnp.bfloat16)


def main():
    print(json.dumps({"device": str(jax.devices()[0]), "jax": jax.__version__}), flush=True)
    rows, key, rng = [], jax.random.key(0), np.random.default_rng(0)
    for label, d in TABLES:
        words, groups = d // 8, d // G
        packed = jax.random.randint(key, (V, words), -2**31, 2**31 - 1, jnp.int32)
        scale = jax.random.uniform(key, (V, groups), jnp.float32, 1e-3, 2e-2).astype(jnp.float16)
        padded_p = jnp.pad(packed, ((0, 0), (0, ceil128(words) - words)))
        padded_s = jnp.pad(scale, ((0, 0), (0, ceil128(groups) - groups)))
        # fp16 pairs -> one int32 word each, appended after the packed words.
        s_pairs = jax.lax.bitcast_convert_type(
            jnp.pad(scale, ((0, 0), (0, groups % 2))).reshape(V, -1, 2), jnp.int32)
        n_pairs = s_pairs.shape[1]
        fused_w = words + n_pairs
        fused = jnp.pad(jnp.concatenate([packed, s_pairs], axis=1), ((0, 0), (0, ceil128(fused_w) - fused_w)))

        f_current = jax.jit(lambda i, p, s: unpack(jnp.take(p, i, axis=0), jnp.take(s, i, axis=0), d))
        f_padded = f_current

        def fused_fn(i, f):
            r = jnp.take(f, i, axis=0)
            s = jax.lax.bitcast_convert_type(r[..., words:words + n_pairs], jnp.float16)
            return unpack(r, s.reshape(*r.shape[:-1], -1), d)
        f_fused = jax.jit(fused_fn)

        def dslice_fn(i, p, s):
            def one(idx):
                return (jax.lax.dynamic_slice_in_dim(p, idx, 1, 0)[0],
                        jax.lax.dynamic_slice_in_dim(s, idx, 1, 0)[0])
            rp, rs = jax.lax.map(one, i)
            return unpack(rp, rs, d)
        f_dslice = jax.jit(dslice_fn)

        mb = {"current": (packed.nbytes + scale.nbytes) / 2**20, "padded": (padded_p.nbytes + padded_s.nbytes) / 2**20,
              "fused": fused.nbytes / 2**20, "dslice": (packed.nbytes + scale.nbytes) / 2**20}
        for m in TOKENS:
            ids = jnp.asarray(rng.integers(0, V, m), jnp.int32)
            ref, t_ref = timed(f_current, ids, packed, scale)
            for name, fn, args in (("current", None, None), ("padded", f_padded, (padded_p, padded_s)),
                                   ("fused", f_fused, (fused,)), ("dslice", f_dslice, (packed, scale))):
                try:
                    out, t = (ref, t_ref) if fn is None else timed(fn, ids, *args)
                    same = bool(jnp.array_equal(out, ref))
                    row = {"table": label, "layout": name, "tokens": m, "us": round(t * 1e6, 1),
                           "vs_current": round(t_ref / t, 2), "table_MiB": round(mb[name], 1), "same_rows": same}
                except Exception as e:  # noqa: BLE001 - one failing layout must not hide the rest
                    row = {"table": label, "layout": name, "tokens": m, "error": f"{type(e).__name__}: {e}"[:300]}
                rows.append(row)
                print(json.dumps(row), flush=True)
        del packed, scale, padded_p, padded_s, fused, s_pairs

    print("\n| table | layout | tokens | us | speed vs current | table MiB | same rows |")
    print("|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['table']} | {r['layout']} | {r['tokens']} | error | | | {r['error'][:60]} |")
        else:
            print(f"| {r['table']} | {r['layout']} | {r['tokens']} | {r['us']} | {r['vs_current']}x | {r['table_MiB']} | {r['same_rows']} |")
    bad = [r for r in rows if r.get("same_rows") is False]
    print(f"\nlayouts that returned different rows: {len(bad)}")


if __name__ == "__main__":
    main()
