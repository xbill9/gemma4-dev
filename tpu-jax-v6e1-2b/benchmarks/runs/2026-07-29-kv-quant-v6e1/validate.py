"""Two validations: calibrate the instrument, then test the MQA hypothesis.

A. CALIBRATION. Every "% of peak" figure in this report divides measured bytes by
   a published 1640 GB/s that was never verified on this chip. If a trivially
   memory-bound kernel only reaches 40% of that number, then our attention
   reaching 20% means something completely different than if a trivial kernel
   reaches 95%. This measures the ceiling with an op that has nowhere to hide.

B. THE MQA HYPOTHESIS. Rubens measured vLLM's hand-tuned Pallas attention at
   ~25% of bandwidth; we measure our naive eager attention at 14-23%. Two
   unrelated implementations landing in the same place suggests a structural
   cause, and the obvious candidate is that both models have ONE KV head: the
   cache axis is degenerate, and each request reads its own cache so nothing
   amortizes across the batch.

   Testable directly. Hold everything else fixed and vary num_key_value_heads
   over 1, 2, 4, 8 (E2B has 8 query heads). Marginal bandwidth already divides
   by bytes, so the comparison is fair even though total KV grows. If bandwidth
   climbs with head count, the hypothesis holds and the finding is structural
   rather than a property of either codebase.
"""
import os, sys, time, json, statistics
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX, init_kv_cache,
                                      make_cached_decode_step)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

HBM_BW = 1640e9
OUT = {}


def timed(fn, n=5):
    for _ in range(2):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


# ---------------------------------------------------------------- A. calibrate
print("### A. bandwidth calibration — what does this chip actually reach? ###", flush=True)
cal = []
for gb in (1, 2, 4):
    n = int(gb * 1e9 / 2)
    x = jnp.ones((n,), dtype=jnp.bfloat16)
    # A pure streaming reduction: reads every byte once, writes a scalar, and has
    # no reuse to hide behind.
    f = jax.jit(lambda a: jnp.sum(a, dtype=jnp.float32))
    ms = timed(lambda: f(x))
    bw = (n * 2) / (ms / 1000)
    print(f"  sum over {gb} GB bf16: {ms:7.3f} ms -> {bw/1e9:7.1f} GB/s "
          f"= {bw/HBM_BW*100:5.1f}% of published peak", flush=True)
    cal.append(bw / 1e9)

    # A strided read shaped like the KV cache access: [B, 1, S, D] contracted
    # against a single query position — the degenerate-head layout under test.
    del x, f
OUT["calibration_gbs"] = cal
peak = max(cal)
print(f"  -> best observed streaming bandwidth: {peak:.1f} GB/s "
      f"({peak/(HBM_BW/1e9)*100:.1f}% of the 1640 figure)", flush=True)
print(f"  -> every percentage in this report should arguably be normalized to "
      f"this, not to 1640.", flush=True)

# ---------------------------------------------------------------- B. MQA
print("\n### B. does KV head count explain the attention gap? ###", flush=True)
print(f"{'kv_heads':>9}{'ctx':>7}{'KV GB':>9}{'step ms':>10}{'marginal GB/s':>15}"
      f"{'% pub':>8}{'% cal':>8}", flush=True)
B, CTXS = 32, [1024, 2048, 4096]
mqa = {}
for nkv in (1, 2, 4, 8):
    try:
        cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512,
                                   num_key_value_heads=nkv,
                                   num_global_key_value_heads=nkv))
        model = Gemma4EModelJAX(cfg)
        params = build_benchmark_params(cfg)
        step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16", window_kv=False))
        n_slide = sum(1 for i in range(cfg.first_kv_shared_layer_idx)
                      if cfg.layer_types[i] == "sliding_attention")
        n_full = cfg.first_kv_shared_layer_idx - n_slide
        kv_el = (n_slide * nkv * cfg.head_dim * 2
                 + n_full * nkv * cfg.global_head_dim * 2)
        prev, slopes = None, []
        for ctx in CTXS:
            total = ctx + 16
            caches = init_kv_cache(cfg, batch_size=B, max_seq_len=total, dtype=jnp.bfloat16)
            valid = jnp.zeros((B, total), dtype=jnp.bool_).at[:, :ctx].set(True)
            tok = jnp.ones((B, 1), dtype=jnp.int32)
            lens = jnp.full((B,), ctx, dtype=jnp.int32)
            args = (params, caches, valid, tok, lens, jnp.int32(ctx))
            ms = timed(lambda: step(*args), n=3)
            kv = B * ctx * kv_el * 2
            s = ""
            if prev and ms > prev[0]:
                bw = (kv - prev[1]) / ((ms - prev[0]) / 1000)
                slopes.append(bw / 1e9)
                s = f"{bw/1e9:>14.1f}{bw/HBM_BW*100:>7.1f}%{bw/(peak*1e9)*100:>7.1f}%"
            print(f"{nkv:>9}{ctx:>7}{kv/1e9:>9.2f}{ms:>10.2f}{s}", flush=True)
            prev = (ms, kv)
            del caches
        if slopes:
            mqa[nkv] = sum(slopes) / len(slopes)
        del params, model, step
    except Exception as e:
        print(f"{nkv:>9}  FAILED: {str(e).splitlines()[0][:60]}", flush=True)

OUT["mqa_marginal_gbs"] = mqa
if mqa:
    print("\n  mean marginal KV bandwidth by head count:", flush=True)
    base = mqa.get(1)
    for k in sorted(mqa):
        rel = f" ({mqa[k]/base:.2f}x vs 1 head)" if base else ""
        print(f"    {k} KV head(s): {mqa[k]:7.1f} GB/s = "
              f"{mqa[k]/(peak)*100:5.1f}% of calibrated peak{rel}", flush=True)

with open(os.path.expanduser("~/validate.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("\nVALIDATE_DONE", flush=True)
