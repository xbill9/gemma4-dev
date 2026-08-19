"""Last chip run. Three questions, in priority order.

1. Is the KV-token budget still a clean constant under int8? The bf16 grid gave
   exactly 524,288 across six context lengths; if int8 gives a constant too, the
   budget is a real property of the chip and not a coincidence of one dtype.
2. Where does fp8_e4m3 land? It matched int8 on quality; if it also matches on
   capacity the choice is free, and if not int8 wins outright.
3. How much of prefill's peak memory is chunkable? Prefill sets the batch
   ceiling, and compile-time memory_analysis answers this WITHOUT running the
   OOMing configs — the only way to measure a wall you cannot reach.
"""
import os, sys, time, statistics, json
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX, init_kv_cache,
                                      make_cached_decode_step, prefill_with_kv_cache,
                                      pad_to_tpu_v6e_bucket)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512))
model = Gemma4EModelJAX(cfg)
params = build_benchmark_params(cfg)
RESULTS = []


def max_batch(dt, ctx, batches):
    step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16", window_kv=False))
    best, best_ms = None, None
    for B in batches:
        try:
            total = ctx + 16
            caches = init_kv_cache(cfg, batch_size=B, max_seq_len=total, dtype=dt)
            valid = jnp.zeros((B, total), dtype=jnp.bool_).at[:, :ctx].set(True)
            tok = jnp.ones((B, 1), dtype=jnp.int32)
            lens = jnp.full((B,), ctx, dtype=jnp.int32)
            args = (params, caches, valid, tok, lens, jnp.int32(ctx))
            for _ in range(2):
                jax.block_until_ready(step(*args))
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); jax.block_until_ready(step(*args))
                ts.append((time.perf_counter()-t0)*1000)
            best, best_ms = B, statistics.median(ts)
            print(f"    B={B:>5} ok   step {best_ms:>7.2f} ms   KVtok {ctx*B:>10,}", flush=True)
        except Exception as e:
            print(f"    B={B:>5} OOM", flush=True)
            break
    return best, best_ms


# ---- 1 & 2: the invariant, per cache dtype -------------------------------
LADDERS = {
    512:   [256, 512, 768, 1024, 1536, 2048, 3072],
    2048:  [128, 256, 384, 512, 768],
    8192:  [32, 64, 96, 128, 192],
    32768: [8, 16, 24, 32, 48],
}
for dt in [jnp.bfloat16, jnp.int8, jnp.float8_e4m3fn]:
    print(f"\n########## cache dtype = {dt.__name__} ##########", flush=True)
    for ctx, ladder in LADDERS.items():
        print(f"  ctx={ctx}", flush=True)
        B, ms = max_batch(dt, ctx, ladder)
        kvtok = ctx * B if B else 0
        RESULTS.append(dict(dtype=dt.__name__, ctx=ctx, max_B=B, kv_tokens=kvtok, step_ms=ms))
        print(f"  -> ctx={ctx:>6}: max B {B}, KV tokens {kvtok:,}", flush=True)
    got = [r["kv_tokens"] for r in RESULTS if r["dtype"] == dt.__name__ and r["kv_tokens"]]
    if got:
        print(f"  BUDGET {dt.__name__}: min {min(got):,}  max {max(got):,}  "
              f"spread {(max(got)/min(got) - 1)*100:.1f}%", flush=True)

# ---- 3: how much of prefill's peak is chunkable? --------------------------
# Peak temporaries as a function of the prefill chunk length, at fixed batch.
# If temporaries fall roughly linearly with S, chunked prefill converts the
# prefill ceiling into a scheduling parameter rather than a hard wall.
print("\n########## prefill peak memory vs chunk length ##########", flush=True)
print(f"{'B':>5} {'S':>7} {'temps GB':>10} {'total GB':>10}", flush=True)
for B in [8, 32]:
    for S in [128, 256, 512, 1024, 2048]:
        try:
            ids = jnp.ones((B, S), dtype=jnp.int32)
            padded, valid = pad_to_tpu_v6e_bucket(ids)
            lowered = jax.jit(
                prefill_with_kv_cache,
                static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype", "window_kv"),
            ).lower(model=model, prompt_ids=padded, prompt_valid=valid, params=params,
                    max_new_tokens=16, quant_mode="w4a16", window_kv=False)
            m = lowered.compile().memory_analysis()
            print(f"{B:>5} {S:>7} {m.temp_size_in_bytes/1e9:>10.2f} "
                  f"{(m.temp_size_in_bytes + m.argument_size_in_bytes)/1e9:>10.2f}", flush=True)
            RESULTS.append(dict(kind="prefill_mem", B=B, S=S,
                                temp_gb=m.temp_size_in_bytes/1e9))
        except Exception as e:
            print(f"{B:>5} {S:>7}  analysis failed: {str(e).splitlines()[0][:50]}", flush=True)

with open(os.path.expanduser("~/final_run.json"), "w") as f:
    json.dump(RESULTS, f, indent=1)
print("\nFINAL_RUN_DONE", flush=True)
