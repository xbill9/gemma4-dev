# 2026-08-29 — first profile, and two fixes

`g5g.2xlarge` spot in `us-east-1d` (`i-0dee6eadc33c86ac8`, AMI `ami-07a66fa2acbcfea88`,
$0.4058/hr). Baseline profile taken on build `d1defe28fdd7`; the sweep re-run on
`060a572aeb55` after two fixes. Artifacts: `batch_sweep.json`, `kernels.json`, `sweep.json`,
`metrics.prom`, `REPORT.json` (schema 1.1, validated), `sweep.log`.

**This run exists because the first one measured a number without explaining it.** xprof and
tensorboard were installed on 2026-08-29 and never used; every claim about *where* decode time
went was inherited from the JAX sibling, whose profile explicitly does not transfer.

## The headline: batching is free, and worth 7.8x

`profile_decode.py --mode batch`, 20 steps at a 256-token prompt:

| B | ms/step | tok/s | peak GB | step vs B=1 | tput vs B=1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 93.21 | 10.73 | 10.271 | 1.000x | 1.00x |
| 2 | 94.43 | 21.18 | 10.308 | 1.013x | 1.97x |
| 4 | 94.92 | 42.14 | 10.382 | 1.018x | 3.93x |
| 8 | 95.05 | 84.16 | 10.529 | 1.020x | **7.84x** |

**Per-step time grew 2.0% while the batch grew 8x**, for 0.258 GB of extra memory. This is the
direct confirmation of what the first report could only argue: the decode step is dominated by
costs that do not depend on batch — weight streaming and kernel launches — so every extra
sequence in the batch is very nearly free.

**84.16 tok/s at B=8 beats the vLLM sibling's 43-44 tok/s** and beats the JAX sibling's 13.10
outright. It is measured on the engine directly, NOT through the server: `MAX_NUM_SEQS=1` and
one lock mean the *served* path still cannot reach it. **Continuous batching is now the single
highest-value piece of work in this rig, and it is quantified rather than assumed.**

## Where the 93 ms/step goes

`torch.profiler`, 24 decode steps, `record_shapes=True`. Top CUDA kernels:

| kernel | share | calls | µs/call | ms/step |
| --- | ---: | ---: | ---: | ---: |
| `gemv2T_kernel_val<..., __half, __half, __half, ...>` | 15.43% | 2544 | 95.80 | 10.155 |
| `internal::gemvx::kernel` (3 variants) | 14.65% | 3265 | — | 9.634 |
| `aten::copy_` | 4.23% | 22924 | 2.91 | 2.781 |
| `aten::mul` | 3.13% | 17600 | 2.81 | 2.058 |
| `unrolled_elementwise_kernel` | 2.83% | 13875 | 3.22 | 1.863 |
| `turing_fp16_s1688gemm_*` (tensor core) | ~4% | 106 | 550-810 | 2.67 |
| `aten::_efficient_attention_forward` (SDPA) | 1.03% | 175 | 92.97 | 0.678 |

Four things, and the third is the actionable one:

- **SDPA is already active** (`attn_implementation: sdpa`) and attention is **1.0% of decode**.
  There is no free win there, and FlashAttention-2 needs Ampere anyway.
- **Turing's tensor cores DO get used** — `turing_fp16_s1688gemm_*` is an HMMA kernel — but only
  ~106 times across 24 steps. Those are prefill. **Decode runs on `gemv` kernels**, as the JAX
  sibling predicted, though note this rig's `gemv2T_kernel_val` is templated on `__half`, so
  PyTorch is *not* uniformly promoting to fp32 the way the JAX profile showed.
- **~5,650 kernel launches per step.** `aten::copy_` alone is 955 calls/step at 2.91 µs, `mul`
  733/step at 2.81 µs, `pow` 504/step at 1.49 µs. These are 1-3 µs kernels on a chip whose
  launch overhead is 5-10 µs: **launch-bound, not compute-bound.** This is precisely what
  `torch.compile(mode="reduce-overhead")` plus a `StaticCache` collapses into CUDA graphs, and
  it is the second-ranked piece of work.
- The step's theoretical weight-streaming floor is 10.209 GB / 320 GB/s = **31.9 ms**, so
  **61.3 ms of the 93.2 ms step is not bandwidth** — and, per the batch table, none of it scales
  with B.

## Two fixes, measured

### `logits_to_keep=1` — prefill

`AutoModelForCausalLM.forward` defaults `logits_to_keep=0`, which means *keep all*. Prefill was
running the LM head over every prompt position and building a `[1, S, 262144]` tensor to use one
row of it — 1.311 GB at S=2,501 and 2.147 GB at the configured `--seq 4096`, plus ~2 TFLOP of
discarded matmul. **This is the same bug the JAX sibling fixed as `logits_at`**, reintroduced
here by writing the obvious `out.logits[:, -1, :]`, which slices after the cost is paid.

Least squares over the range both runs share (92-2,501 tokens):

| | ms/token | change |
| --- | ---: | ---: |
| run 1, no `logits_to_keep` | 0.6417 | — |
| run 2, `logits_to_keep=1` | 0.5611 | **−12.6%** |

−12.6% against ~17% predicted from the LM head's share of per-token FLOPs. **Compare on the
common range only**: run 2's slope over its own full 92-3,746 range is 0.6449 ms/token, because
attention is O(S²) and the extra reach is super-linear — differencing that against run 1's
shorter range hides the improvement entirely.

### `device_map` instead of `.to(device)` — load

`from_pretrained(...).to(cuda)` builds the whole tree in host memory and then copies it.
Measured on a 16 GiB `g5g.2xlarge`:

| | host peak | swap peak | observed |
| --- | ---: | ---: | --- |
| `.to(device)` | 14.0 GB | 2.8 GB | minutes at 98% iowait, si/so ~30k blk/s |
| `device_map={"": 0}` | **10.52 GB** | **0** | load 11.3 s, warm-up 1.2 s |

**Quote the memory, not the wall time.** The 11.3 s load also benefited from a warm page cache,
so it is not a clean A/B; host peak and swap peak are, and swap went to zero. The swapfile
remains the right safety net — this stops it being the load path.

## Sweep after the fixes

**10/12 cells OK, 0 failed, 0 degenerate. Decode 10.84 tok/s median** (10.88 in run 1 — the
fixes were prefill and memory, not decode, and the two agree to 0.3%).

| input tok | out 32 | out 128 |
| ---: | ---: | ---: |
| 92 | 10.35 | 10.27 |
| 633 | 10.80 | 10.81 |
| 1259 | 10.81 | 10.90 |
| 2501 | 10.87 | 10.90 |
| **3746** | **10.88** | **10.91** |

**3,746 tokens now serves.** Run 1 stopped at 2,501 and still reported "8/8 cells" — coverage
overstated by never approaching the configured `--seq 4096`. The two `infeasible` cells are the
harness's own prompt-length estimator overshooting to 4,630 tokens; the server returned a clean
400 naming the bound, which is the guard working rather than a failure.

Decode remains flat: **6.2% spread over a 41x context range**.

## What to do next, in order

1. **Continuous batching.** Measured 7.84x at B=8 for 0.258 GB. Nothing else is close.
2. **`torch.compile(mode="reduce-overhead")` + `StaticCache`.** ~5,650 launches/step of 1-3 µs
   kernels on a chip with 5-10 µs launch overhead.
3. **Stop loading the non-text towers** — 0.952 GB resident, never used in text decode. Memory,
   not throughput.

Not worth pursuing, now measured rather than assumed: **the attention backend** (SDPA already
active, 1.0% of decode) and **the weight storage dtype** (the JAX sibling falsified it at +0.0%,
and decode here is already dispatching `__half` GEMV kernels).
