"""How many streams can we DECODE, if prefill is not done all at once?

The sweep's batch ceiling came from prefilling B sequences simultaneously. A real
server prefills as requests arrive and only ever decodes the full batch, so build
the KV cache directly and measure the decode-only ceiling.
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX, init_kv_cache,
                                      make_cached_decode_step, dequantize_params_to_dense)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512))
model = Gemma4EModelJAX(cfg)
base = build_benchmark_params(cfg)

def run(label, params, quant, ctx, batches, cache_dtype=jnp.bfloat16):
    print(f"\n### {label} (ctx={ctx}, cache={cache_dtype.__name__}) ###")
    step = jax.jit(make_cached_decode_step(model, quant_mode=quant, window_kv=False))
    for B in batches:
        try:
            total = ctx + 16
            caches = init_kv_cache(cfg, batch_size=B, max_seq_len=total, dtype=cache_dtype)
            valid = jnp.zeros((B, total), dtype=jnp.bool_).at[:, :ctx].set(True)
            tok = jnp.ones((B, 1), dtype=jnp.int32)
            lens = jnp.full((B,), ctx, dtype=jnp.int32)
            args = (params, caches, valid, tok, lens, jnp.int32(ctx))
            for _ in range(2):
                jax.block_until_ready(step(*args))
            ts = []
            for _ in range(5):
                t0 = time.perf_counter(); jax.block_until_ready(step(*args))
                ts.append((time.perf_counter()-t0)*1000)
            ms = statistics.median(ts)
            print(f"  B={B:>4}  step {ms:>7.2f} ms  agg {B*1000/ms:>9.1f} tok/s  per-user {1000/ms:>6.1f}")
        except Exception as e:
            print(f"  B={B:>4}  FAILED: {str(e)[:60]}")
            break

run("packed W4A16", base, "w4a16", 512, [32, 64, 128, 256, 512])
dense = dequantize_params_to_dense(base)
run("dequantized at load", dense, "fp16", 512, [32, 64, 128, 256, 512])
run("fp8 KV cache", base, "w4a16", 512, [64, 128, 256], cache_dtype=jnp.float8_e4m3fn)
run("long context, single stream", base, "w4a16", 8192, [1, 2, 4])
