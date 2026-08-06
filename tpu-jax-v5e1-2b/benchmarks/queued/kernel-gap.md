# Queued: where does the 5x attention gap go?

**Status:** queued, needs a v6e-1. Run `benchmarks/queued/kernel_gap_suite.py`.

## What we know

Measured 2026-07-29 on a v6e-1 (`benchmarks/runs/2026-07-29-kv-quant-v6e1/`):

| quantity | value |
|---|---|
| calibrated streaming bandwidth (`jnp.sum` over 2 GB) | **1412 GB/s** = 86% of the published 1640 |
| marginal KV bandwidth (slope, weights cancelled) | **240-310 GB/s** = ~20% of calibrated |
| fixed per-step cost, W4A16 packed | 5.97 ms vs 1.17 ms roofline = 20% |
| fixed per-step cost, dense bf16 | 3.61 ms vs 2.78 ms roofline = 77% |

So: the weight path is fine when dense and ~4x off when packed (the reference
W4A16 dequant, ~2.4 ms/step fixed). **The dominant gap is the KV read at ~20% of
achievable bandwidth**, and it is the term that scales with load.

## Hypotheses, one already dead

**H1: MQA / degenerate KV-head axis. REFUTED.** Varying `num_key_value_heads`
over 1, 2, 4, 8 left marginal bandwidth flat at 240-310 GB/s. One head performs
like eight. (The summary row claiming 1 head = 539 GB/s was contaminated by an
841 GB/s outlier from a ctx step where the time delta was mostly fixed-cost
noise; exclude slopes whose time delta is small relative to the fixed cost.)

**H2: `dynamic_update_slice` without buffer donation — LEADING.** Writing one
token into the cache produces a *new* array unless the buffer is donated, so each
decode step plausibly reads the whole cache, writes a whole copy, then reads it
again for attention: ~3x traffic, and **head-count independent**, which matches
what H1 measured. `jax_engine.py:310` donates only when `donate_cache=True`
(default `False`), and every benchmark to date built its own
`jax.jit(make_cached_decode_step(...))` with **no donation at all**.

**H3: eager attention materializes intermediates.** Scores `[B,H,1,T]` in f32,
softmax in f32, mask `[B,1,1,T]` in f32 rebuilt every step. Individually small
against a multi-GB cache, but they may prevent fusion of the cache read.

**H4: the read is not contiguous.** The 5-D grouped-query einsum
(`bgnsd,bgtd->bgnst`) may lower to a layout that defeats sequential prefetch.

## Tests, in priority order

1. **Donation A/B** (tests H2). Same marginal-bandwidth sweep with and without
   `donate_argnums`. Prediction if H2 holds: marginal bandwidth rises toward the
   calibrated ceiling and peak memory falls, with byte-identical outputs.
2. **Profiler attribution.** Rank ops by device time via `jax.profiler.trace`.
   This is the instrument Rubens had and we lack; it converts "the slope implies
   attention" into a named operation with a measured duration. Key on tensor
   SHAPE, not op name — names are reused across compiled buckets.
3. **Attention in isolation.** Time `eager_attention_jax` alone on decode-shaped
   inputs against the pure cache-read roofline, removing every other step cost.
4. **Overhead census.** Cost of mask construction and the f32 softmax, measured
   by ablation.
5. **Re-run the roofline with the corrected byte model.** The committed roofline
   figures counted the 4.70 GB PLE table as streamed; it is a gather and is not.
   Per-step weight traffic is transformer weights + the LM head table only.

## Rule carried forward

Compute a marginal slope only where the time delta is large relative to the fixed
cost. Two-point slopes at small deltas produced the 841 GB/s artifact above, and
the same class of error produced the fake "0.0% spread" invariant earlier in the
session.
