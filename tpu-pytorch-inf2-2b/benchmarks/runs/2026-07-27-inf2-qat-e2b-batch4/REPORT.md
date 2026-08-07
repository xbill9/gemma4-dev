# Gemma 4 E2B QAT lockstep batching on AWS Inferentia2 (b4 / b8 / b16)

**Run:** 2026-07-27
**Build box:** `inf2.8xlarge` spot, `us-east-1b` (`i-06ef52a7e61ab4e87`,
terminated after upload; region-wide spot drought delayed launch ~75 min —
25 retry attempts across all AZs)
**Model:** `google/gemma-4-E2B-it-qat-q4_0-unquantized`, bf16 (production
weights — no quantization)
**Recipe:** Option-B two-graph KV-cache path with batch dimension B=4 on
all graph I/O (lockstep: streams share one position, so one-hot/masks stay
batch-1 and broadcast), `KV_BUCKET=32`, `KV_MAX=128`, same compiler args as
production.

## Motivation

The int8 run (`2026-07-27-inf2-qat-e2b-int8`) established that decode is
weight-bandwidth-bound (~160 GB/s effective ≈ inf2 HBM). The TPU repo's
measured law: on a bandwidth-bound decoder, weights are read once per step
regardless of batch, so lockstep streams ride nearly free. This run prices
that law on Inferentia2.

## Results

| Config | ms/step | Aggregate | Per stream | Neff |
| --- | ---: | ---: | ---: | ---: |
| batch-1 bf16 (baseline) | 21.1 | 47.3 tok/s | 47.3 tok/s | 3.42 GB |
| batch-4 bf16 | 25.5 | 157.1 tok/s | 39.3 tok/s | 3.42 GB |
| batch-8 bf16 (follow-up, `i-0c0b0ecc433b19e53`) | 29.1 | 274.9 tok/s | 34.4 tok/s | 3.42 GB |
| **batch-16 bf16 (same box)** | **36.1** | **442.7 tok/s** | 27.7 tok/s | 3.42 GB |

**16 streams for 1.71× the step cost = 9.4× aggregate throughput; no
memory cliff anywhere on the curve** (the TPU repo's b6/b8 VMEM cliff has
no inf2 analog at these sizes). Correctness at every batch:
`STREAMS_EQUAL True` and `SEQ_MATCH True` — all device streams match the
CPU greedy reference token-for-token.

The +4.4 ms/step over batch-1 is consistent with 4× KV-buffer host↔device
traffic (~31 MB/step round trip) plus marginally more compute — the weight
read (the dominant 3.4 GB) is unchanged, exactly as predicted. Compare the
TPU no-cache analog: 4 streams for 2.0× step cost there; inf2's lockstep
scaling is *better* because its fixed cost (weight streaming) is a larger
fraction of the step.

## Cross-platform scoreboard (E2B decode)

| Platform / config | Aggregate tok/s |
| --- | ---: |
| inf2.xlarge b1 (production) | 47.3 |
| **inf2.xlarge b4 (this run)** | **157.1** |
| v6e-1 no-cache b1 bf16 | 227.6 |
| v6e-1 cached port b4 int8 | 809.9 |

## Serving implications

- The graphs are drop-in for the optb container layout (same I/O contract,
  batch-4 shapes). Production use needs a lockstep batching queue in
  `optb_server_qat.py`: collect up to 4 concurrent requests, pad to the
  common position grid, decode in lockstep, release streams as they hit
  EOS (a finished stream keeps burning its slot until the batch drains or
  is refilled — acceptable at 4).
- Single-request latency degrades 17% (39.3 vs 47.3 tok/s per stream);
  worth exposing a `BATCH=1` env fallback if latency-sensitive.
- At batch-4 the KV round trip is ~31 MB/step (~3-4 ms). If batch or
  context grows further, device-resident KV becomes the next lever (the
  TPU repo's cached-port data shows exactly where that goes).

## Cost model (two-point fit, tpu-pytorch methodology)

From b1 (21.1 ms) and b4 (25.5 ms), step time = **F + m·B** with
**F ≈ 19.6 ms fixed** and **m ≈ 1.47 ms per stream**. The b8/b16 follow-up
refined it: marginal cost settles at **m ≈ 0.9 ms/stream** beyond b4
(b4→b8: 0.90; b8→b16: 0.88 — the b1→b4 1.47 included one-off batching
overhead), giving an asymptote near 1,100 tok/s aggregate if nothing else
binds. Both terms match physical budgets:

- F ≈ the weight stream: 3.42 GB ÷ ~190 GB/s HBM ≈ 18 ms, plus graph
  overhead. This is why lockstep scales so well here — 93 % of the step is
  batch-independent (on TPU's much faster HBM the fixed fraction is ~50 %,
  hence their worse-looking 2.0× no-cache scaling).
- m ≈ per-stream host↔device I/O: ~7.9 MB/step KV round trip
  (15 source layers × K+V × [2 heads, 128, 256] bf16, in and out) plus
  ~1 MB logits + embeddings ≈ 1.2–1.5 ms at PCIe rates.

The b8 prediction (31.4 ms / 255 agg) was conservative — measured 29.1 ms
/ 274.9. The TPU repo's b6/b8 memory cliff did NOT appear on inf2 through
b16; the curve is smooth and slightly sublinear in cost. Per-stream
latency is the real trade: 47.3 → 27.7 tok/s from b1 to b16 (−41 %).
Serving sweet spot depends on load: b4 (−17 % latency, 3.3×) or b8
(−27 %, 5.8×); b16 is a throughput configuration.

Two TPU lessons that carry over directly:

1. **KV device-residency attacks m, not F.** Their cached port keeps KV as
   in-place-mutated device tensors (zero host round trip); ours ships all
   KV as graph I/O. Eliminating m raises the batch asymptote ~3× — the NxD
   ModelBuilder's aliased state buffers are the inf2 mechanism for this
   (torch_neuronx.trace alone doesn't expose aliasing; to verify).
2. **Batch and cache interact non-monotonically.** Their cached b1 was
   *slower* than no-cache b1 (1-row matmul tiling waste) but cached b4 was
   the record. Any inf2 cached redesign should be evaluated at the target
   batch, not b1.

## Artifacts (`s3://xbill-gemma4-patches-2b/qat-e2b-b4/`)

`qat_e2b_b4_{prefill,decode}.pt` (3.42 GB each, SEQ_MATCH-validated),
`qat_e2b_b4_trace.py`, `build_b4.sh`.
