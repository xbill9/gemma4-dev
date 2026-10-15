# 2026-08-28 — first serve on an NVIDIA L4, and the dtype tax is gone

**`g6.2xlarge` spot (`i-00c7f5a16b0f8a429`), `us-east-1d`, AMI `ami-0000d4d67c1d21bb0`
(Deep Learning Base OSS Nvidia Driver GPU, Ubuntu 26.04, **x86_64**), jax 0.11.1 /
Python 3.14, build id `51bc52c9e2e9`.**

**This rig's first measurement of any kind.** Everything in `tpu.env` marked MEASURED before
today was measured on G5g (T4G, Turing) and inherited through the fork.

It answers the question the rig was forked to ask. The 87% dtype tax that dominates decode on
Turing is **0.0%** here, throughput is **3.7x**, and the rig now sits **at its bandwidth
roofline** rather than at 26% of it.

The payload is byte-identical to the T4G runs — same build id `51bc52c9e2e9`, same
`ple4 + int8_lm_head` config, and `tpu_jax_weight_bytes` = **6,155,450,950**, the same integer
the 2026-08-26, -27 and -28 g5g runs report. **Only the chip differs.**

## Headline

| | T4G (SM 7.5) | **L4 (SM 8.9)** | |
| --- | ---: | ---: | ---: |
| decode, gauge | 12.80 tok/s | **48.3–48.5 tok/s** | **3.7x** |
| total kernel time / profile | 1466.0 ms | **362.8 ms** | **4.04x** |
| dtype conversion | 54.0% | **0.0%** | — |
| fp32 `gemvx` | 32.8% | **absent** | — |
| TensorCore | 0.0% | 0.0% | unchanged |
| peak HBM (xprof) | 298.083 GiB/s | **279.441 GiB/s** | **lower** |
| % of bandwidth roofline | 26% | **~100%** | — |

## The device picked bfloat16 by itself

The first line the process emits, and the whole premise of the rig landing on hardware:

```
INFO ports.gemma4.jax_e_model: jax_e_model device policy: platform=gpu
compute_capability=8.9 compute_dtype=bfloat16 pallas_interpret=False
```

No code change and no `tpu.env` change. `jax_e_model.py` reads the live compute capability and
picks `bfloat16` at SM >= 8.0; on Turing the same line reads `compute_dtype=float16`. `/health`
then reports `weights=bfloat16 activations=bfloat16 kv_cache=bfloat16 pre_ampere=false` —
**storage dtype and compute dtype match for the first time on this engine.**

## Sweep — decode is flat across an 88x context range

| in tok | out | bucket | pad | gauge tok/s | end-to-end tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 64 | 64 | 23 | **48.5** | 46.23 |
| 521 | 64 | 640 | 119 | **48.4** | 42.87 |
| 2,057 | 64 | 2,176 | 119 | **48.3** | 34.57 |
| 3,593 | 64 | 3,712 | 119 | **48.3** | 27.55 |

3 repeats per cell, median. 0 degenerate, 0 failed. HBM 6.197 GB of 23,034 MiB.

**Decode does not move** — 0.4% spread across 41 → 3,593 input tokens. That reproduces the
2026-08-25 g5g finding on a different GPU generation: a cost proportional to the *weights*
rather than the context produces exactly this shape, and it is why **KV is still not what sets
decode speed here.**

**End-to-end does move** (46.23 → 27.55) and that is prefill, as on Turing. **Quote the gauge.**

## The kernel table is a 1:1 substitution, not a reshuffle

`profile_decode.py`, 20 decode steps, service stopped:

```
wall 20.193 ms/token -> 49.52 tok/s   (GPU lane 18.140 ms/token)

 ms/token      %   calls  kernel
    5.958  32.8%    40.0  gemm_fusion_dot_419
    3.024  16.7%    20.0  gemm_fusion_dot_479
    2.429  13.4%    15.0  input_reduce_fusion_102
    1.708   9.4%     1.0  gemm_fusion_dot_478
    1.180   6.5%    15.0  input_reduce_fusion_14
```

**The call counts are the tell.** Line them up against the Turing table:

| calls/step | T4G | ms | **L4** | ms |
| ---: | --- | ---: | --- | ---: |
| 40 | `wrapped_convert_1` | 19.90 | **`gemm_fusion_dot_419`** | **5.958** |
| 20 | `wrapped_convert_3` | 9.96 | **`gemm_fusion_dot_479`** | **3.024** |
| 1 | `wrapped_convert_61` (LM head) | 9.31 | **`gemm_fusion_dot_478`** (LM head) | **1.708** |

Same three shapes, same call counts, same LM head singleton. On Turing the top three kernels
were the fp32 **promotion**; here the promotion does not exist and the matmul itself is what
shows up. The conversion was not moved or hidden — **it was never needed.**

Confirmed by direct inspection of `xprof_kernel_stats.json`: **zero `gemvx` kernels** and no
`wrapped_convert_*` at all. The only convert-named kernel left is
`loop_convert_dynamic_slice_fusion`, the int8 LM-head dequantisation, and it is negligible.

## The roofline closes — this rig is now bandwidth-bound

xprof reports the L4's `peak_hbm_bw` as **279.441 GiB/s = 300.1 GB/s**, and `peak_flop_rate`
as **121,160 GFLOP/s** (against the T4G's 298.083 GiB/s / 65,126 GFLOP/s).

**The L4 has LOWER memory bandwidth than the T4G — 300 GB/s against 320 — and is 3.7x faster.**
That is the cleanest available proof that Turing decode was never bandwidth-bound. It was bound
by the fp32 promotion, exactly as `gpu-jax-g5g-2b/docs/bf16-weights-on-turing.md` diagnosed.

```
6.155 GB / 300.1 GB/s = 20.51 ms/token  ->  48.75 tok/s   (weight-streaming floor)
measured                20.193 ms/token ->  49.52 tok/s
```

Measured lands **just past** the naive floor, which is the expected sign rather than a
contradiction: the `ple4` table is a **gather**, never streamed, so not all 6.155 GB moves per
step. Either way the conclusion is the same and it is a change of regime:

- **T4G: 13.6 tok/s against a 52.0 tok/s ceiling — 26%.** Three quarters of the time was waste.
- **L4: 49.5 tok/s against a 48.8 tok/s floor — at the wall.**

**So the optimisation advice inverts.** On Turing the win was removing overhead. Here there is
no overhead left to remove: further speed requires **fewer weight bytes** — a real int8 or int4
matmul — not less waste. `int8_lm_head` today still dequantises to bf16 in full before an
ordinary matmul, so the int8 tensor cores remain unexploited on both rigs.

## TensorCore is still 0.0%, and this is weaker evidence than on Turing

`is_kernel_using_tensor_core` is `False` for **100.0% of kernel time**, and
`is_op_tensor_core_eligible` is **0.0%** too, on a chip with 121 TFLOP/s of bf16 tensor-core
throughput.

**Do not read this as firmly as the Turing result.** There, the kernel was literally cuBLAS
`gemvx::kernel<int, int, float, float, float, float, ...>` — every template parameter fp32, a
direct signature. Here the kernels are XLA-generated `gemm_fusion_dot` fusions, and xprof
reporting them as not-eligible may be a **reporting gap for XLA's own fusions** rather than a
fact about what the SM executed. The `B=1` GEMV argument still applies and is still the most
likely explanation — `MAX_NUM_SEQS=1`, `B > 1` raises `NotImplementedError` — but on this rig
the flag alone does not settle it.

That matters because the rig is at its bandwidth roofline anyway. **Lighting up tensor cores
cannot help a bandwidth-bound decode**, so this is a curiosity here, not a lever.

## `MAX_MODEL_LEN=4096` is inherited from Turing's memory ceiling and is now untested

A 4,096-token cell returned **500**, and the new error logging named it immediately:

```
ValueError: Prompt of 4105 tokens leaves no room within max_model_len=4096
```

**That is the config limit, not an OOM** — and the limit exists because on a 15,360 MiB T4G the
prefill transient made 5,120 tokens infeasible (`2026-08-26`). This card has **23,034 MiB** and
peaked at 6.197 GB. The value was left alone rather than changed mid-benchmark, but **it is now
an inherited assumption with no evidence behind it on this hardware**, and finding the real
ceiling is the obvious next run.

## Cold/warm is still 61x — warm at the shape you measure

| | cold | warm |
| --- | ---: | ---: |
| prefill @ 2,057 tok | **28,927.7 ms** | 467 ms |
| decode | 9.74 tok/s | 48.3 tok/s |

Faster silicon did **not** shrink the compile penalty; if anything the ratio grew, because warm
decode got 3.7x faster while compilation did not. `tune_loop.py` warms at the measured
`(bucket, max_tokens)` shape. Anything hand-rolled must too.

## Install: 66 seconds

Cloud-init `status: done`, no errors, `[stage] INSTALL COMPLETE total 66s`. The x86_64 SSM
parameter resolved Ubuntu 26.04 and `cp314` wheels installed against the system interpreter —
the same path the g5g rig measured at ~80 s on arm64. **No compilation-cache restore was
configured for this run** (`JAX_CACHE_S3_URI` empty), so 66 s is the uncached figure.

## Fixed in this run

**`tune_loop.py:196` hardcoded the `-g5g` suffix** on the run directory, inherited from the
fork. It filed this L4 result under the T4G's hardware short name — the misattribution the
monorepo `CLAUDE.md` calls out by name ("a report's `<hw-short>` is the hardware *measured*").
Now derived from `RIG_NAME`'s hardware slot. The directory was renamed `…-first-serve-g6`.

**The instance role's S3 policy only covered `gpu-jax-g5g-2b/*` prefixes**, so the profile
upload would have failed. Extended with `jax-cache/gpu-jax-g6-2b/*` and
`benchmarks/gpu-jax-g6-2b/*`.

## Not established here

- **Spot durability.** The instance was terminated deliberately after the run; it was not
  reclaimed, so this says nothing about G6 spot lifetime. `g6.xlarge` spot *was* exhausted in
  all five `us-east-1` AZs — `g6.2xlarge` in `us-east-1d` was chosen from
  `get-spot-placement-scores` (score 3 against 1 elsewhere), which is a cheaper way to pick
  than launching until one succeeds.
- **The real `MAX_MODEL_LEN` ceiling** on 23 GB — see above.
- **Cross-instance compilation-cache restore** on this rig. Confirmed on g5g, unconfigured here.
- **Anything about `g6.xlarge`.** Only `g6.2xlarge` was measured. The L4 is the same on both;
  host RAM is not (16 vs 32 GiB), and the swapfile gate turns on at `<= 16`.
