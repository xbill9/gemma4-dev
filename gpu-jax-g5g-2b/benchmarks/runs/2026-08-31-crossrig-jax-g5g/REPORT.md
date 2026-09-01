# 2026-08-31 — JAX leg of the three-runtime controlled comparison

**This rig's contribution to a cross-rig run, not a standalone experiment.** The full analysis
lives in `gpu-pytorch-g5g-2b/benchmarks/runs/2026-08-31-crossrig-torch-g5g/REPORT.md`; this
file records what happened on this rig.

`g5g.2xlarge` spot, `us-east-1a`, `i-0ffdb819663bacc77`, build `53e79b8f40f3`, config
**ple4+int8head** (the shipped default — `/health` confirms `ple_bits: 4`,
`int8_lm_head: true`). Driven by **`gpu-pytorch-g5g-2b/sweep.py`**, not by a sweep of this
rig's own — that is the point of the run.

## Result

**Decode 12.962 tok/s (server gauge) / 12.687 tok/s (client-side stream), 10/12 cells, 0 degenerate.**

| input tok | out | gauge tok/s | stream tok/s | stream/gauge |
| ---: | ---: | ---: | ---: | ---: |
| 92 | 32 | 13.10 | 12.63 | 0.9640 |
| 92 | 63 | 12.86 | 12.80 | 0.9950 |
| 633 | 32 | 13.20 | 12.73 | 0.9648 |
| 633 | 82 | 12.79 | 12.73 | 0.9953 |
| 1259 | 32 | 13.18 | 12.71 | 0.9646 |
| 1259 | 83 | 12.77 | 12.72 | 0.9954 |
| 2501 | 32 | 13.13 | 12.67 | 0.9644 |
| 2501 | 84 | 12.72 | 12.66 | 0.9950 |
| 3746 | 32 | 13.06 | 12.60 | 0.9646 |
| 3746 | 84 | 12.66 | 12.60 | 0.9955 |

The two 3,800-context cells are `infeasible`: 4,630 prompt tokens against `max_model_len=4096`,
refused with an error naming both numbers.

## The gauge is no longer rounded to 0.1 tok/s

**Fixed in this rig on 2026-08-31, and this run is the first to benefit.**
`jax_openai_server.py` emitted the decode gauge as `:.1f` (line 569) and
`round(..., 1)` (line 693). At ~13 tok/s that is **0.78% resolution**, which is why every
earlier sweep here shows all three repeats of a cell as byte-identical — 12.8, 12.8, 12.8 —
and why this rig could not resolve the ~2% effects routinely argued about on it. Both sites now
use 4 decimals.

**One conclusion this immediately changes.** `CLAUDE.md` and `tpu.env` credit `int8_lm_head`
with **+2.3%** (12.80 -> 13.10). Recomputed from the raw runs of
`2026-08-26-quant-levers-fixed-g5g`, 13.10 is that config's **best cell**, not its median; the
run medians are **12.80 -> 13.00, i.e. +1.6%**. Matched per context it is +2.3% / +1.6% /
+1.6%. The honest figure is the median, and +1.6% is small enough that it needs a repeat before
it should be called an improvement.

## stream/gauge is 0.9799, and it is not a universal constant

Measured with `--decode-source both`. **The PyTorch sibling's ratio on the same day and the
same instance shape is 0.9543** — 2.0% against 4.6%. Do not convert one rig's number into
another's statistic with a borrowed ratio.

Note the ratio is bimodal by output length here: ~0.964 at `out=32` and ~0.995 at `out≈83`.
With only 31 gaps, per-token SSE framing weighs more heavily on the short cells.

## Against the other two runtimes

| runtime | stream tok/s | vs this rig |
| --- | ---: | ---: |
| vLLM v0.27.2rc0 | 32.53 | 2.56x |
| **JAX (this rig)** | **12.69** | 1.00x |
| PyTorch + transformers | 10.24 | 0.81x |

JAX remains ~24% ahead of PyTorch on identical silicon and ~2.6x behind vLLM. Unlike vLLM,
decode here is nearly flat in context (3.4% over 41x) — see the cross-rig report for why that
is a symptom rather than a virtue.

## Boot and revision time (3 repeats, 2026-08-31)

Cold boot **242.2 s** [237.7–265.5, 11.5%] · warm redeploy **74.1 s** [4.2%].

**The number that matters here is first-completion, not health 200: 22.9 s cold, 9.2 s warm**,
against PyTorch's 1.0 s / 0.7 s. This rig returns health 200 and *then* compiles XLA per shape
bucket on the first real request, so health alone understates its time-to-serving by 22 seconds.
The compile does not disappear on a warm redeploy either.

**Real revision cycle: 74.1 + 9.2 = 83 s**, against PyTorch's 25 s on the same mechanism.
Full three-runtime table and method in
`gpu-pytorch-g5g-2b/benchmarks/runs/2026-08-31-crossrig-torch-g5g/REPORT.md`.

## Prefill: this rig is 2.4x slower than the PyTorch sibling

**The one prefill result that survived the cross-rig run**, and it is about this rig.

| | TTFT slope | TTFT at 92 tok | at 3,746 tok | growth |
| --- | ---: | ---: | ---: | ---: |
| **JAX (this rig)** | **1.403 ms/token** | 225 ms | **5,352 ms** | 23.8x |
| PyTorch sibling | 0.595 ms/token | 164 ms | 2,339 ms | 14.3x |

Consistent across all five shared context lengths, and **the comparison is fair**: neither
engine has a prefix cache, both saw identical prompts, both ran the same harness on the same
instance shape in the same AZ on the same day.

**Do not compare either against the vLLM leg's TTFT.** That rig ships
`enable_prefix_caching=True` and answered 94.7% of its requests from cache
(97,440 hits / 102,898 queries), because `sweep.py` reused one prompt per cell across the
warm-up and all repeats. Its apparent 0.025 ms/token is a cache-hit workload, not prefill.
`sweep.py --prompt-mode` now defaults to `unique` so this cannot recur; every number in this
run was taken under the old `fixed` behaviour.

**5.35 s to first token at 3,746 input tokens is this rig's weakest measured result** — worse
relative to the sibling than its decode advantage (1.24x in this rig's favour) is good. On any
interactive workload with real context, prefill dominates the user-visible latency and this is
where the JAX port loses.

Note the separate readiness cost that compounds it: **22.9 s of XLA compile on the first
request after a cold start, 9.2 s after a warm redeploy** (see the boot section above).
