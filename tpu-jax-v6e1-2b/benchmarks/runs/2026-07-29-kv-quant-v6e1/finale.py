"""Final chip run: (2) bisect the decode wall above B=512, (3) long-context ceiling.

decode_ceiling.py stopped at B=512 because that was the top of the ladder, not
because anything failed. And its long-context probe only went to B=4 at ctx 8192
for the same reason. Both walls are still unmeasured; this finds them.

Also probes int8 KV plumbing, since fp8 turned out to be advertised-but-broken.
"""
import os, sys, time, statistics, json
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX, init_kv_cache,
                                      make_cached_decode_step, dequantize_params_to_dense)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512))
model = Gemma4EModelJAX(cfg)
base = build_benchmark_params(cfg)
dense = dequantize_params_to_dense(base)

RESULTS = []


def probe(params, quant, ctx, B, cache_dtype=jnp.bfloat16, step=None):
    """One decode-step measurement. Returns ms, or None if it did not run."""
    if step is None:
        step = jax.jit(make_cached_decode_step(model, quant_mode=quant, window_kv=False))
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
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def ladder(label, params, quant, ctx, batches, cache_dtype=jnp.bfloat16):
    print(f"\n### {label} (ctx={ctx}) ###", flush=True)
    step = jax.jit(make_cached_decode_step(model, quant_mode=quant, window_kv=False))
    last_ok = None
    for B in batches:
        try:
            ms = probe(params, quant, ctx, B, cache_dtype, step)
            agg = B * 1000 / ms
            print(f"  B={B:>5}  step {ms:>8.2f} ms  agg {agg:>9.1f} tok/s  "
                  f"per-user {1000/ms:>6.1f}  {'OK ' if ms <= 50 else 'SLOW'}", flush=True)
            RESULTS.append(dict(label=label, ctx=ctx, B=B, step_ms=ms, agg=agg))
            last_ok = B
        except Exception as e:
            msg = str(e).split("\n")[0][:70]
            print(f"  B={B:>5}  FAILED: {msg}", flush=True)
            RESULTS.append(dict(label=label, ctx=ctx, B=B, failed=msg))
            break
    print(f"  -> max B at ctx={ctx}: {last_ok}", flush=True)
    return last_ok


# ---- (2) bisect the decode wall above B=512 -------------------------------
ladder("packed W4A16 wall", base, "w4a16", 512, [512, 768, 1024, 1536, 2048, 3072])
ladder("dense bf16 wall", dense, "fp16", 512, [512, 768, 1024, 1536, 2048, 3072])

# ---- (3) long-context ceiling, now that decode is known to be cheap -------
for ctx in [2048, 4096, 8192, 16384, 32768]:
    ladder("packed W4A16 long-ctx", base, "w4a16", ctx, [4, 8, 16, 32, 64, 128, 256])

# ---- quantization probe: does an int8 KV cache even plumb? ----------------
print("\n### int8 KV cache plumbing probe ###", flush=True)
for dt in [jnp.int8, jnp.float8_e5m2, jnp.float16]:
    try:
        ms = probe(base, "w4a16", 512, 64, cache_dtype=dt)
        print(f"  {dt.__name__:>16}: step {ms:.2f} ms  (RUNS — accuracy unvalidated)", flush=True)
    except Exception as e:
        print(f"  {dt.__name__:>16}: {str(e).split(chr(10))[0][:70]}", flush=True)

with open(os.path.expanduser("~/finale.json"), "w") as f:
    json.dump(RESULTS, f, indent=1)
print("\nFINALE_DONE", flush=True)
