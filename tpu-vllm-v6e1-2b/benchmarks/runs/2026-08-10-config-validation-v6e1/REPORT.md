# 2026-08-10 — v6e-1 config validation

**The first measurement this rig has taken on its own hardware.** Everything else under
`benchmarks/` arrived with the fork and was measured on v5e-1, or on a v6e-era fork that is a
different rig and a different vLLM.

**Provenance.** `google/gemma-4-E2B-it` on TPU **v6e-1**, `europe-west4-a`, **flex-start**,
`vllm/vllm-tpu:nightly`, TP=1, `max_model_len` **32768**, `max_num_batched_tokens` 4096,
`kv_cache_dtype=auto`, `enable_prefix_caching=True`. 24 cells across three sweeps.

---

## Headline results

**1. The KV pool is independent of `max_model_len`.**

| `max_model_len` | KV pool | block_size |
| ---: | ---: | ---: |
| 16,384 | 1,151,776 tokens | 32 |
| 32,768 | 1,151,744 tokens | 64 |
| 65,536 (recorded earlier, `@HARDWARE.md`) | 1,151,744 tokens | 128 |

A 0.003% spread across a 4x range. Longer context costs **block size, not capacity** — the
backend derives `block_size` to hold blocks-per-request at 512. **This is the evidence that
raising `MAX_MODEL_LEN` is free, and it applies equally to 65536.**

**2. fp8 KV does not engage — now measured on v6e, not inferred.**
The engine logs `Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e` **20 times**, and
allocates `regular_attn_dtype=bfloat16`. The arithmetic is not close: 1,151,744 tokens against
19.77 GiB is the bf16 model to **0.01%**; fp8 is **50%** off. The flag was never passed —
`kv_cache_dtype=auto` — so this fires on the default path. Sixth false fp8 signal on record, and
the first on a second silicon generation. See `@QUANTIZATION.md`.

**3. v6e beats v5e by ~2.9x where v5e was memory-constrained, and only ~1.63x where it was not.**

| role | cells | mean vs v5e |
| :--- | :--- | ---: |
| working set < 10% of v5e's pool | 3 | **1.63x** |
| working set ≥ 83% of v5e's pool | 5 | **2.90x** |

Against a **1.907x** bandwidth ratio and a **2.25x** price ratio. On decode throughput alone
this chip is a bad deal and the measurement says so. It pays for capacity, not speed —
exactly the claim in `@HARDWARE.md`.

**4. There is no capacity knee. Anywhere.**

`TTFT = -8542 + 265 x concurrency`, **R² = 1.00000** over 10 points spanning **56% to 157%** of
the KV pool. Throughput is flat at 453-474 tok/s (4.7%) across that entire range.
**`num_preemptions_total` = 0 in every cell**, including 157% occupancy.

The scheduler **admission-controls rather than evicts**: it admits what fits and queues the
rest, so the working set never thrashes and crossing the pool boundary is not an event. A line
fitted entirely *below* 100% predicts 157% occupancy to within 0.13%.

> **This falsifies a derivation that was written into `SERVING-PARAMS.md`.** That file scaled
> v5e's "keep `clients x context` under ~250,000 tokens (78% of pool)" to **~900,000 tokens** on
> v6e, flagged as a derivation rather than a measurement. There is no knee at 78%, no knee
> anywhere in 56-157%, and no event on crossing 100%. The v5e rule does not transfer as a
> fraction of pool, and the file has been corrected.

**Operational rule, now measured:** size by the latency you will accept, not by pool occupancy.
Throughput saturates near concurrency 40 and every further concurrent request buys exactly
**265 ms of TTFT and nothing else**. The pool bounds what is *resident*; it is not a
performance threshold.

---

## Two methodology errors found in this run

Both were found by the run's own instrumentation, and both are recorded because the corrections
are more useful than the numbers.

**1. The config silently did not take effect, and one check caught it while another went blind.**
The node booted at `max_model_len: 16384` — the MCP server process had loaded `tpu.env` before
`MAX_MODEL_LEN=32768` was added, and rendered the startup script from a stale in-memory default.
`verify_allocation.py` ran before any benchmark and stopped the run.

But its **primary** check silently returned UNKNOWN: vLLM prints the value inside a dict repr
(`'max_model_len': 16384`) and the regex did not allow for the quote. What actually caught it was
`block_size` reading 32 instead of 64 — a *different* derived quantity, downstream of the same
setting. **Two independent derivations of one setting is why the run was saved.** Both the regex
and this reasoning are now in the file, so the redundancy is not later tidied away as duplication.

**2. `--seed 0` plus prefix caching silently coupled cells.**
vLLM defaults to `enable_prefix_caching=True` and the random dataset is deterministic in the
seed, so two cells at the same `input_len` draw overlapping prompts and the later one is served
from cache. In the main sweep `16000x32` ran after `16000x64` and took **1,539,904 cache-hit
tokens against 1,536,000 input tokens** — essentially all of it. Its 2,461 ms TTFT is not
comparable to the others.

That contaminated point, plus an unsampled gap from 46% to 89% of pool, is what made the main
sweep *look* like it had a 3.4x cliff. Two points far apart with the lower one artificially fast
reads as a cliff. **The knee and overflow sweeps use a distinct seed per cell and measured 0.0%
prefix hits throughout.** `run_cells.py:run_cell` now takes a `seed` argument, defaulting to 0 so
the archived main-sweep cells stay reproducible as taken.

---

## Why there is no cliff: the Gemma quirk, stated precisely

E2B caches 15 of 35 layers (`num_kv_shared_layers=20`): 12 sliding-window at head_dim 256 with a
512-token window, 3 full-attention at 512. tpu_inference sees two head dims, sets
`disable_sliding_window`, and gives **every** layer full-length blocks (`num_kv_cache_groups=1`,
15 tensors — confirmed in this run's boot log).

Allocation and reads therefore diverge:

| | 12 sliding layers | 3 full layers |
| :--- | :--- | :--- |
| **allocated** | full context — 12,288 B/token | 6,144 B/token |
| **read per decode step** | **6.29 MB, constant** once L > 512 | 98.3 MB at L=16,000 |

At 16,000 context, **94% of KV bytes read come from 3 of 15 layers**. The pool is charged
~18 KiB/token where ~6.4 KiB would do — the 2.9x.

**This does not prevent a cliff; it moves the cliff closer.** It is a capacity tax, not a latency
shield. The reason no cliff appeared is simpler and was measured directly: **nothing was ever
evicted.** Occupancy alone costs nothing; eviction would, and preemption never engaged even at
157%.

---

## Files

| | |
| :--- | :--- |
| `verify_allocation.py` | reads the boot log, checks it against arithmetic from `@MODELS.md`. Gates the benchmark |
| `run_cells.py` | main 10-cell sweep — context x concurrency, v5e references inlined |
| `run_knee.py` | 5 cells, fixed 16,000 context, bisecting 56-84% of pool |
| `run_overflow.py` | 4 cells crossing the pool boundary, 101-157%, with per-cell preemption deltas |
| `run_all.sh` | driver — waits for vLLM, captures the log, verifies, then benchmarks |
| `results/allocation.json` | the five allocation checks |
| `results/cells_v6e1.json`, `knee_v6e1.json`, `overflow_v6e1.json` | 24 cells |
| `logs/boot.log`, `logs/cells_v6e1.log` | evidence |

## Not established here

- **Nothing above 157% of pool**, and nothing that forces preemption. The eviction regime remains
  untested — every cell in this run was admission-controlled.
- **The knee sweep varied concurrency at fixed context only.** A context sweep at fixed
  concurrency would load the 3 full-attention layers differently and is not covered.
- **The 1.63x control figure is not explained.** It sits below the 1.907x bandwidth ratio, and
  the MXU geometry change (4x128x128 -> 2x256x256, so a 4x larger minimum tile) is a plausible
  cause but is **not demonstrated** — separating it from "small batches do not saturate
  bandwidth" needs a batch-size sweep at fixed short context.
- **No v5e reference exists for the two `long_ctx` cells**, correctly: that configuration cannot
  exist on v5e at any setting.
