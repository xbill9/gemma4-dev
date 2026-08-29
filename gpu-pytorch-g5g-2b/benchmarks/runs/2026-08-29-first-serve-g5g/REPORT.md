# 2026-08-29 — first serve, `gpu-pytorch-g5g-2b`

**The first token this rig has ever produced.** `g5g.2xlarge` spot in `us-east-1d`
(`i-025c02baf3c836e44`, AMI `ami-07a66fa2acbcfea88`, $0.4058/hr), torch 2.12.0+cu132 on
Python 3.13, build id `4ca8039100d7`. Driven by `sweep.py`; artifacts are `sweep.json`,
`metrics.prom` and `REPORT.json` (schema 1.1, validated).

## Result

**Decode 10.88 tok/s (median of 8 cells), 8/8 cells OK, 0 failed, 0 degenerate replies.**

| input tok | output | decode tok/s | end-to-end tok/s | wall s | warm-up tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 92 | 32 | 10.66 | 10.12 | 3.16 | 10.59 |
| 92 | 128 | 10.56 | 10.35 | 7.54 | 10.65 |
| 633 | 32 | 10.87 | 9.60 | 3.33 | 10.84 |
| 633 | 128 | 10.98 | 10.46 | 8.31 | 10.86 |
| 1259 | 32 | 10.92 | 8.60 | 3.72 | 10.95 |
| 1259 | 128 | 10.88 | 9.90 | 8.79 | 10.95 |
| 2501 | 32 | 10.81 | 6.91 | 4.63 | 10.80 |
| 2501 | 128 | 10.95 | 9.08 | 9.80 | 10.90 |

Cumulative over the whole run: 1,792 completion tokens in 165.96 s of decode = **10.80 tok/s**,
which agrees with the per-cell median. Weights **10.209 GB**, device 14,477 of 15,360 MiB.

## The headline: PyTorch is 15% SLOWER than JAX on identical silicon

| | `gpu-jax-g5g-2b` | **this rig** |
| --- | ---: | ---: |
| decode, comparable config | 12.80 tok/s (`ple0`) | **10.88 tok/s** |
| decode, best config | 13.10 tok/s (`ple4+int8head`) | — |
| weights | 9.257 GB | 10.209 GB |

**This is the answer to the question the rig was built to ask, and it is not the expected one.**

`gpu-jax-g5g-2b` spends **54% of decode in `wrapped_convert` kernels** and 0.0% of kernel time
on a TensorCore. Its own 2026-08-28 experiment already falsified the storage-dtype explanation —
converting the whole tree to float16 moved throughput **+0.0%** — and identified the real cause
as cuBLAS dispatching a **fp32 `gemvx`** with no half-precision path at `B=1`.

**PyTorch is the independent confirmation.** It loads directly as float16, has no bf16 tree and
no conversion pass to blame, and it is *slower anyway*. Two rigs, two frameworks, one wall:

```
weights / peak HBM  =  10.209 GB / 320 GB/s  =  31.9 ms/step  ->  31.3 tok/s   (a FLOOR)
measured                                                          10.88 tok/s  ->  35% of it
```

The JAX rig sits at 26% of its own ceiling; this one at 31–35% of its. **Neither framework is
within 3x of the bandwidth bound, and the gap between them (15%) is far smaller than the gap
each has to the hardware (3x).** So the deficit is not a framework artifact — it is `B=1` decode
being a matrix-*vector* product that no dtype and no runtime turns into a GEMM.

**Do not read this as "JAX is the better runtime here" and stop.** The useful reading is that
the remaining 3x is in neither runtime's gift: it needs batching, which both rigs currently
refuse (`MAX_NUM_SEQS=1`).

## Corroborating: decode does not depend on context

**3.9% total spread over a 27x context range** (92 → 2,501 tokens), with no trend. A cost
proportional to the *weights* produces exactly that; if KV traffic were binding, decode would
fall as context grew. **The KV cache is not what sets decode speed here** — and at E2B's
~18 KiB/token the whole 4K cache is tens of MiB against 14.07 GB of budget.

This independently reproduces the JAX rig's 3.4% spread over a 100x range, from a different
framework and a different harness.

## Why the end-to-end column is in the table and must not be quoted

It falls from 10.12 to 6.91 tok/s across the same rows where decode is flat, because it carries
prefill and the HTTP round trip. **The two columns disagree by up to 36% and the disagreement is
the point** — `tpu_jax_decode_tokens_per_second` is what every sibling report compares on.

## Notes and caveats

- **Warm-up matched measurement to within 1% in every cell**, so no cold shape leaked into a
  median. `sweep.py` warms at the exact `(prompt, max_tokens)` pair it then measures.
- **This rig loads 0.952 GB more than the JAX sibling** — 10.209 vs 9.257 GB — which is exactly
  the non-text towers the JAX loader skips. They are resident but not streamed during text
  decode, so they cost memory rather than throughput. Against the text-tower-only ceiling this
  rig is at 31%.
- **Concurrency is not an axis.** `MAX_NUM_SEQS=1` and the server serializes on one lock, so
  every cell is `concurrency: 1`.
- **The comparison is config-fair only against the JAX rig's `ple0` run.** 13.10 tok/s is a
  quantised configuration (`ple4+int8head`) with 6.155 GB of weights and ~0.8% logit error;
  this rig runs the dense reference checkpoint with no quantisation path at all.
- **`apt-daily-upgrade` restarted the serving unit twice during the initial load** and was
  masked before any measurement was taken. No measured request ran against a disturbed process.
