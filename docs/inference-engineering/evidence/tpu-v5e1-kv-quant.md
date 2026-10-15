# KV cache fp8 vs bf16 — Gemma 4 E2B on TPU v5e-1

**Run:** 2026-08-07 · `tpu-2B-v5e1-devops-agent` (v5litepod-1, 1x1, us-west4-a, `aisprint-491218`)
**Engine:** vLLM `0.26.1rc1.dev125+ga7a204cc6`, image pinned by ID `sha256:2a4a1f82…` (not the `:nightly` tag)
**Arms:** `cache_dtype=auto` (bf16) vs `--kv-cache-dtype fp8_e4m3`. One flag differs; everything else byte-identical.
**Coverage:** 8/8 throughput cells measured in each arm, 9/9 quality probes in each arm. Nothing skipped.

Machine-readable: [`comparison.json`](comparison.json), [`SUMMARY.md`](SUMMARY.md), raw per-arm data in
[`results/`](results/). Regenerate with `python3 aggregate.py --bf16-quality results/quality_bf16.json
--fp8-quality results/quality_fp8.json --bf16-cells results/cells_bf16.json --fp8-cells results/cells_fp8.json
--bf16-kv-tokens 321376 --fp8-kv-tokens 321376`.

## Verdict

**`--kv-cache-dtype fp8_e4m3` produces no capacity gain, no throughput gain, and a consistent small
throughput loss. Do not set it on this build.**

| | bf16 | fp8_e4m3 | ratio |
| :--- | ---: | ---: | ---: |
| `kv_cache_size_tokens` | 321,376 | 321,376 | **1.000x** |
| `num_gpu_blocks` | 10,043 | 10,043 | 1.000x |
| `kv_cache_max_concurrency` | 19.615 | 19.615 | 1.000x |
| bytes per block per layer | 32,768 | 32,768 | 1.000x |

The prediction was 642,752 tokens (exactly 2x). The measurement is 1.000x.

## The flag is accepted everywhere and still does nothing

This is worth recording as a pattern, not just a result. The flag produces **five** independent
confirmations that it took effect:

1. Accepted at the CLI without error.
2. Echoed in the engine's `non-default args`: `'kv_cache_dtype': 'fp8_e4m3'`.
3. An approving log line — *"Using fp8_e4m3 data type to store kv cache. It reduces the GPU memory
   footprint and boosts the performance"* (`cache.py:296`).
4. Reported in `/metrics` as `cache_dtype="fp8_e4m3"`.
5. The allocated tensor really is `regular_attn_dtype=float8_e4m3fn`.

And the capacity is byte-for-byte unchanged. The only line that reveals it is the KV block shape:

```
bf16:  regular_attn_shape=(num_blocks, (32, 1, 2, 256))   regular_attn_dtype=bfloat16
fp8:   regular_attn_shape=(num_blocks, (32, 1, 4, 256))   regular_attn_dtype=float8_e4m3fn
```

The third dimension doubles from 2 to 4 exactly as the element width halves. 16,384 elements x 2 bytes
and 32,768 elements x 1 byte are both 32,768 bytes. The layout is word-aligned — two bf16 or four fp8
fill the same 32-bit word — so narrowing the element buys padding, not capacity.

**Open question, deliberately not answered here:** whether that doubled dimension is inert padding or
something structural such as a hi/lo byte split of the original bf16 value. The near-perfect output
match below is consistent with either a lossless repack *or* with greedy decoding absorbing the
quantization error. Distinguishing them requires reading the tpu_inference KV-cache spec code, not
running another benchmark. No mechanism is claimed.

## Why there was never headroom for a compute win

TPU v5e has **no native fp8 path in the MXU**. Google publishes bf16 (197 TFLOPs) and Int8 (393 TOPs)
peaks for v5e and no fp8 figure; Ironwood/v7 is the first TPU generation with fp8 in the MXU.
`tpu_inference` quantizes on the write path only (`models/jax/gemma4.py:483-490`) and the kernel widens
back to bf16 before the matmul — Q stays bf16 throughout, its `q_scale` line commented out. So fp8 KV
could only ever have changed *bytes stored and bytes moved*, never FLOPS. With the byte count unchanged,
both remaining channels are closed too.

## Throughput — every cell negative

| ctx | conc | role | KV needed | bf16 tok/s | fp8 tok/s | Δ |
|---|---|---|---|---|---|---|
| 128 | 1 | control | 256 | 123.26 | 120.13 | −2.5% |
| 128 | 8 | control | 2,048 | 738.28 | 696.73 | −5.6% |
| 1024 | 16 | control | 18,432 | 896.11 | 855.97 | −4.5% |
| 4096 | 64 | bandwidth | 270,336 | 585.92 | 570.47 | −2.6% |
| 8192 | 32 | bandwidth | 266,240 | 307.76 | 301.33 | −2.1% |
| 8192 | 64 | key | 532,480 | 314.44 | 307.83 | −2.1% |
| 16000 | 32 | key | 516,096 | 166.76 | 163.75 | −1.8% |
| 16000 | 64 | saturated | 1,032,192 | 166.69 | 163.69 | −1.8% |

No cell gained. The KEY cells — the only place a capacity win could have appeared — lost 1.8-2.1%,
and their median TTFT stayed at ~11-12 s in both arms, meaning the bf16 KV wall is exactly as high
under fp8. That is the direct consequence of the 1.000x capacity ratio.

**Arm order is confounded with warm-up, and it inflates the control losses.** The bf16 arm ran on a
container that had been up 20 hours; the fp8 arm ran on one booted ~10 minutes earlier. Cells run in
matrix order, so the three control cells are the most exposed, and they carry the largest losses
(−2.5%, −5.6%, −4.5%) plus the one TTFT outlier (36.55 → 65.8 ms, +80% on small absolute numbers).
The settled later cells cluster tightly at −1.8% to −2.6%. **Best estimate of the real cost is ~2%,
from the settled cells; the control-cell figures should not be read as a larger effect at short
context.** Re-running with the arms in the opposite order would separate this, and is the obvious
next step if the ~2% ever matters.

The `aggregate.py` monotonicity line prints an increasing mean by role (control −4.2%, bandwidth
−2.4%, key −2.0%), but that check was written to detect a *positive* gradient as evidence of a
bytes-moved benefit. With every value negative it is measuring the warm-up gradient above, not a
capacity effect. Do not read it as a partial win.

## Quality — unchanged, and the one divergence is not fp8 damage

8/9 probes byte-identical at temperature 0. **All three needle probes recalled the constant at 2k, 8k
and 14k context in both arms — 0/3 lost.**

The single divergence is `short-regional`, on a **21-token** prompt:

- bf16: *"I need more context to answer your question. Could you please tell me what you are referring to?"*
- fp8: *"Both pork roll and Taylor ham are common deli meats found in New Jersey…"*

This is the *opposite* of the predicted failure signature. Quantization error lands in cached KV and so
accumulates with the number of cached tokens; damage should appear first at 14k, not at 21 tokens where
there is almost no cache to corrupt. A knife-edge prompt flipping between "ask for clarification" and
"answer directly" is prompt sensitivity, and fp8 gave the better answer. Counting it as a quality
regression would be wrong.

Caveat: one sample per prompt at temperature 0 measures reproducibility, not quality. The probe set can
show that fp8 did not *break* anything; it cannot bound a small average quality change.

## What this run does and does not license

- **Does:** rule out `--kv-cache-dtype fp8_e4m3` on this engine build and this TPU path. Capacity is the
  binding claim and it is measured at 1.000x.
- **Does not:** say anything about `fp8_e5m2`. It has the same word-alignment problem by construction
  (also 1 byte), so the capacity result should carry over, but it was not run.
- **Does not:** say anything about weight quantization. That is the larger decode lever on this chip —
  ~6.7 GiB of weight traffic per decode step against the measured 120 tok/s single-stream — and it is
  untested. Note v5e *does* accelerate int8 natively at 2x bf16, unlike fp8.
- **Does not:** generalize to v6e or Ironwood. Ironwood has native fp8 and its KV layout may differ.

## Reproducing / reverting

`swap_arm.sh forward <kv-dtype>` and `swap_arm.sh back` implement the procedure in
[`ROLLBACK.md`](ROLLBACK.md): the baseline container is stopped and renamed, never deleted, so revert is
`docker start` on the untouched original with its warm compile cache and its `HF_TOKEN` never
re-handled. Two pre-flight checks run *before* anything stops — Secret Manager reachability and
image-by-ID presence — so a missing IAM grant cannot strand the rig with no serving container.

fp8 boot took **849.5 s** (770.3 s compilation), a guaranteed compile-cache miss because the KV shapes
change. Budget ~14 minutes per arm swap.
