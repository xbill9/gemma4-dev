"""Decompose the roofline gap: which term is slow, the weights or the KV?

The aggregate "% of peak" figure divides total bytes by total time, mixing a
FIXED weight read with a KV read that scales with ctx x B. That cannot attribute
the gap. Two slopes can:

  marginal KV bandwidth = d(KV bytes) / d(time)   at fixed batch, varying ctx
      The weight read is identical at every point, so it cancels in the
      difference. What remains is the rate at which attention streams the cache.

  fixed cost at ctx -> 0 = the intercept
      Extrapolating the same line back gives the per-step cost that does NOT
      depend on the cache: the weight read plus everything else.

If the intercept is close to weights/HBM_BW, the weight path is fine and the gap
is attention. If the marginal bandwidth is close to peak, attention is fine and
the gap is the weight path. Both slow means both.

Also A/Bs dequant_at_load, which removes the per-forward W4A16 dequant entirely,
at the same points — a direct measurement of what that path costs.
"""
import os, sys, time, json, statistics
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX, init_kv_cache,
                                      make_cached_decode_step, dequantize_params_to_dense)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

HBM_BW = 1640e9
cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512))
model = Gemma4EModelJAX(cfg)
packed = build_benchmark_params(cfg)


def tree_bytes(p):
    tot, stack = 0, [p]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            stack.extend(n.values())
        elif hasattr(n, "size"):
            tot += n.size * n.dtype.itemsize
    return tot


n_slide = sum(1 for i in range(cfg.first_kv_shared_layer_idx)
              if cfg.layer_types[i] == "sliding_attention")
n_full = cfg.first_kv_shared_layer_idx - n_slide
KV_EL = (n_slide * cfg.num_key_value_heads * cfg.head_dim * 2
         + n_full * cfg.num_global_key_value_heads * cfg.global_head_dim * 2)


def step_ms(params, quant, cache_dt, ctx, B, step):
    total = ctx + 16
    caches = init_kv_cache(cfg, batch_size=B, max_seq_len=total, dtype=cache_dt)
    valid = jnp.zeros((B, total), dtype=jnp.bool_).at[:, :ctx].set(True)
    tok = jnp.ones((B, 1), dtype=jnp.int32)
    lens = jnp.full((B,), ctx, dtype=jnp.int32)
    args = (params, caches, valid, tok, lens, jnp.int32(ctx))
    for _ in range(2):
        jax.block_until_ready(step(*args))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); jax.block_until_ready(step(*args))
        ts.append((time.perf_counter() - t0) * 1000)
    del caches
    return statistics.median(ts)


B = 64
CTXS = [512, 1024, 2048, 4096, 8192]
OUT = []

for label, params, quant in (("W4A16 packed", packed, "w4a16"),
                             ("dense bf16 (dequant at load)",
                              dequantize_params_to_dense(packed), "fp16")):
    W = tree_bytes(params)
    step = jax.jit(make_cached_decode_step(model, quant_mode=quant, window_kv=False))
    print(f"\n### {label} — weights {W/1e9:.2f} GB, B={B} ###", flush=True)
    print(f"{'ctx':>7}{'KV GB':>9}{'step ms':>10}{'marginal GB/s':>16}{'% peak':>9}", flush=True)
    prev = None
    for ctx in CTXS:
        ms = step_ms(params, quant, jnp.bfloat16, ctx, B, step)
        kv = B * ctx * KV_EL * 2
        marg = ""
        if prev:
            d_bytes, d_ms = kv - prev[1], ms - prev[0]
            if d_ms > 0:
                bw = d_bytes / (d_ms / 1000)
                marg = f"{bw/1e9:>15.1f}{bw/HBM_BW*100:>8.1f}%"
                OUT.append(dict(label=label, ctx=ctx, marginal_gbs=bw/1e9,
                                pct=bw/HBM_BW*100))
        print(f"{ctx:>7}{kv/1e9:>9.2f}{ms:>10.2f}{marg}", flush=True)
        prev = (ms, kv)

    # Intercept: fit a line through the two smallest contexts and extrapolate to 0.
    ms0 = step_ms(params, quant, jnp.bfloat16, CTXS[0], B, step)
    ms1 = step_ms(params, quant, jnp.bfloat16, CTXS[1], B, step)
    kv0, kv1 = B * CTXS[0] * KV_EL * 2, B * CTXS[1] * KV_EL * 2
    slope = (ms1 - ms0) / (kv1 - kv0)
    intercept = ms0 - slope * kv0
    print(f"  fixed cost extrapolated to zero KV: {intercept:.2f} ms", flush=True)
    print(f"  weight read at peak bandwidth:      {W/HBM_BW*1000:.2f} ms"
          f"   -> fixed-path efficiency {W/HBM_BW*1000/intercept*100:.1f}%", flush=True)
    OUT.append(dict(label=label, intercept_ms=intercept,
                    weight_roofline_ms=W / HBM_BW * 1000, weights_gb=W / 1e9))

with open(os.path.expanduser("~/kernel_gap.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("\nKERNEL_GAP_DONE", flush=True)
