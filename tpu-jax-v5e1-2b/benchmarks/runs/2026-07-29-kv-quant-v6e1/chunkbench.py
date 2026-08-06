"""Does chunked prefill actually raise the batch ceiling?

Decode capacity is a clean constant (524,288 KV tokens bf16, 1,048,576 int8).
The batch ceiling is a DIFFERENT wall, set by prefill temporaries that scale with
prompt_len x batch. If chunking works, the max batch at a given context should
rise toward the decode budget as chunk_size falls.
"""
import os, sys, time, statistics, json
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX,
                                      chunked_prefill_with_kv_cache,
                                      prefill_with_kv_cache, pad_to_tpu_v6e_bucket)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512))
model = Gemma4EModelJAX(cfg)
params = build_benchmark_params(cfg)
OUT = []

CTX = 2048
LADDER = [8, 16, 32, 64, 128, 256]


def try_prefill(B, chunk, cache_dtype):
    ids = jnp.ones((B, CTX), dtype=jnp.int32)
    valid = jnp.ones((B, CTX), dtype=jnp.bool_)
    if chunk is None:
        fn = lambda: prefill_with_kv_cache(model, ids, valid, params, 16,
                                           quant_mode="w4a16", cache_dtype=cache_dtype,
                                           window_kv=False)
    else:
        fn = lambda: chunked_prefill_with_kv_cache(model, ids, valid, params, 16,
                                                   chunk_size=chunk, quant_mode="w4a16",
                                                   cache_dtype=cache_dtype)
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    return (time.perf_counter() - t0) * 1000


for cache_dtype in (jnp.bfloat16, jnp.int8):
    for chunk in (None, 512, 256, 128):
        label = "one-shot" if chunk is None else f"chunk={chunk}"
        print(f"\n### ctx={CTX} {label} cache={cache_dtype.__name__} ###", flush=True)
        best, best_ms = None, None
        for B in LADDER:
            try:
                ms = try_prefill(B, chunk, cache_dtype)
                print(f"  B={B:>4}  prefill {ms:>8.1f} ms", flush=True)
                best, best_ms = B, ms
            except Exception as e:
                print(f"  B={B:>4}  OOM", flush=True)
                break
        print(f"  -> max B {best}", flush=True)
        OUT.append(dict(cache=cache_dtype.__name__, chunk=chunk, max_B=best,
                        prefill_ms=best_ms, ctx=CTX))

print("\n%-10s %-10s %8s %12s" % ("cache", "mode", "max B", "prefill ms"), flush=True)
for r in OUT:
    print("%-10s %-10s %8s %12s" % (
        r["cache"], "one-shot" if r["chunk"] is None else f"chunk{r['chunk']}",
        r["max_B"], f"{r['prefill_ms']:.1f}" if r["prefill_ms"] else "-"), flush=True)

with open(os.path.expanduser("~/chunkbench.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("\nCHUNKBENCH_DONE", flush=True)
