# 2026-08-25 — context × output sweep, and the first xprof profile

**`g5g.2xlarge` spot (`i-08806efbd4308e970`), `us-east-1a`, AMI `ami-077792d0bb6a000b8`,
jax 0.11.1 / Python 3.14, build id `6852f5680f43`. Instance terminated after the run.**

First sweep on this rig — the two prior runs were a single point each. Also the first
xprof profile taken here, which is what turns the throughput number into a cause.

## Why concurrency is not an axis

It is not available on this hardware, and that is worth stating rather than leaving as a
gap in the grid. `MAX_NUM_SEQS=1`; `Gemma4EModelJAX` raises `NotImplementedError` for
`B > 1`; and the decode step **donates its KV buffers**, so firing concurrent requests
through one engine would measure a correctness hazard, not throughput. The sweep is
therefore context × output length at concurrency 1.

Every cell is warmed **at the shape it measures**. `max_new_tokens` is a `static_argnames`
entry, so `(bucket, max_tokens)` is the compiled shape; warming at a different `max_tokens`
than you measure leaves the measured request cold, previously a 4× error here. Warm-up cost
16–27 s per cell against 2.7–13.8 s measured, so this is not a rounding detail.

## Results — 12/12 cells ok

| in tok | bucket | pad | out | decode tok/s | prefill ms | spread |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 64 | 23 | 32 | 12.80 | 143.0 | 0.0% |
| 137 | 256 | 119 | 32 | 13.03 | 333.7 | 0.8% |
| 521 | 640 | 119 | 32 | 12.93 | 763.3 | 0.8% |
| 1033 | 1152 | 119 | 32 | 12.93 | 1541.4 | 0.8% |
| 2057 | 2176 | 119 | 32 | 12.93 | 2962.3 | 0.8% |
| 4105 | 4224 | 119 | 32 | 12.87 | 6270.0 | 0.8% |
| 41 | 64 | 23 | 128 | 12.80 | 140.7 | 0.0% |
| 137 | 256 | 119 | 128 | 12.63 | 332.7 | 0.8% |
| 521 | 640 | 119 | 128 | 12.67 | 754.4 | 0.8% |
| 1033 | 1152 | 119 | 128 | 12.70 | 1531.9 | 0.0% |
| 2057 | 2176 | 119 | 128 | 12.70 | 2936.4 | 0.0% |
| 4105 | 4224 | 119 | 128 | 12.60 | 6228.7 | 0.0% |

Three findings:

- **Decode does not degrade with context.** 12.80 tok/s mean, **3.4% total spread** across a
  100× context range and a 4× output range. Whatever sets decode speed here is not the KV
  cache — if it were, decode would fall as context grew.
- **Prefill is linear in the PADDED bucket**, not in the real prompt:
  `prefill_ms = 1.478 × bucket − 101`, **R² = 0.997**. TTFT is a bucket property.
- **The dense ceiling was not reached at 4,105 tokens.** `docs/larger-models-on-t4g.md`
  bracketed it at (115, 2015] and recorded 3,515 after the `logits_at` fix. Nothing in this
  grid was `infeasible`, so the ceiling is now above 4,105 and remains unlocated.

Bucket padding is a flat **119 tokens** for every bucket ≥ 256 — the post-fix ladder holding
worst-case padding at ≤ 127 and making `pad_len >= 512` unreachable, confirmed on hardware.

## xprof — where the 12.8 tok/s actually goes

xprof was installed on demand (`requirements-profiling.txt`, 2.23.1) and the trace converted
with `xprof.convert.raw_to_tool_data`. Raw tool output is in `xprof/`.

**No kernel used a TensorCore.** `Kernel uses TensorCore` is `False` for **100.0% of kernel
time**, on a chip with 65.1 TFLOP/s of fp16 tensor-core throughput.

| Kernel class | Time | Share |
| --- | ---: | ---: |
| dtype conversion (`wrapped_convert_*`) | 811.4 ms | **54.4%** |
| fp32 `gemvx` | 486.2 ms | **32.6%** |
| reduce fusions | 167.2 ms | 11.2% |
| everything else | 27.2 ms | 1.8% |

**87% of decode is dtype tax.** The cause is already documented in
`docs/bf16-weights-on-turing.md` and is confirmed here from the other side: the loader stores
all 540 float parameters as `bfloat16` while `COMPUTE_DTYPE` is `float16`, so XLA converts in
front of every use, and the matmuls that remain run as **fp32** GEMV.

This also explains the sweep's flat decode. A per-step conversion cost proportional to the
*weights* — not the context — produces exactly a decode rate that does not move as context
grows 100×. Two independent measurements, one cause.

Device constants from the roofline tool, for sizing any fix:

| | |
| --- | ---: |
| peak HBM bandwidth | 298.1 GiB/s |
| peak FLOP rate | 65,126 GFLOP/s |
| HBM ridge point | 203.5 FLOP/byte |

## Memory

xprof `GPU_0_bfc` at peak: **10.171 GiB in use, 2.937 GiB free, fragmentation 0.661.**

The free memory is two-thirds fragmented. That is the same condition behind the
contiguous-block failures in `docs/larger-models-on-t4g.md`, where `device_put` of the
finished tree could not find a contiguous 4.38 GiB block inside a 14.07 GiB budget. Peak
usage is comfortable; contiguity is not.

## What to fix, in order

1. **Kill the bf16→float16 conversion.** 54.4% of decode, and pure waste — it computes
   nothing. `docs/bf16-weights-on-turing.md` records three placements tried and rejected on
   hardware (on-device OOM, unvectorised host cast, fragmentation). The untried direction
   noted there is a bit-level path, and it is straightforward: **bf16 → float32 is a 16-bit
   left shift** (`u16.astype(uint32) << 16` viewed as `float32`), which is pure NumPy and
   fully vectorised, and `float32 → float16` is a native vectorised cast. That avoids
   `ml_dtypes`' unvectorised element-wise path entirely — the thing that made E2B's 4.7 GB
   table not finish in 10 minutes. Do it per shard at load, so no bf16 source outlives its
   converted copy.

2. **Then re-measure before chasing TensorCores.** At `B=1` decode is a matrix-*vector*
   product, which is bandwidth-bound by nature; tensor cores need matrix-matrix work to pay.
   The win from step 1 is removing 54% of pure overhead and halving weight traffic in the
   remaining GEMV (fp32 → fp16), **not** lighting up the tensor cores. Do not size the
   expected gain off the 65 TFLOP/s peak.

3. **Add finer buckets below 512.** Prefill is linear in the bucket, and at 137 real tokens
   the bucket is 256 — so ~39% of that cell's prefill is spent on padding. The ladder jumps
   128 → 256 → 640. Inserting 192 (and 32-steps below 512) costs one compile per newly seen
   bucket, amortised by the persistent compilation cache, and cuts short-prompt TTFT directly.

4. **Locate the dense ceiling.** It is above 4,105 tokens and unmeasured. `MAX_MODEL_LEN` is
   8,192, so the remaining range is one sweep — extend contexts to 6,144 and 8,000.

5. **Watch fragmentation, not peak.** 0.661 at peak with 2.937 GiB free means headroom
   figures overstate what is allocatable. Any future capacity claim on this rig should quote
   the largest contiguous block, not free bytes.

## Artifacts

| File | What |
| --- | --- |
| `sweep.py` | the harness, warm-up-per-shape and checkpointed per request |
| `sweep_results.json` | per-cell results |
| `sweep_requests.jsonl` | one line per request, written as it went |
| `sweep_console.log` | live console output |
| `xprof/kernel_stats.json` | per-kernel time and TensorCore flags |
| `xprof/memory_profile.json` | allocator peak and fragmentation |
| `xprof/roofline_model.json` | device peak bandwidth / FLOP / ridge point |
| `xprof/op_profile.json`, `hlo_stats.json` | op-level rollups (thin — the trace is decode-only) |

Schema-conforming report: `benchmarks/reports/2026-08-25-context-sweep-g5g.json`.
