# 2026-08-29 — first serve, `gpu-pytorch-g6-2b`

**Gemma 4 E2B on an NVIDIA L4 (Ada, SM 8.9) under stock PyTorch + HF transformers.**
First serve, first sweep, and this rig's only measurement.

| | |
| --- | --- |
| Instance | `i-0f5ad4013e7265ee4`, `g6.2xlarge`, **spot**, `us-east-1d` |
| AMI | `ami-0d7cd40a7956dd2c4` — Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.13 (Ubuntu 26.04) |
| torch | `2.13.0+cu130`, from the DLAMI's venv (`/opt/pytorch/bin/python`, Python **3.13**) |
| Build id | `a4be4ec1edb6` (served id matched the local payload digest) |
| dtype | **bfloat16**, selected from the live compute capability |
| Weights | **10,208,595,008 B** (10.209 GB), dense — no quantization on this path |
| Result | **20.93 tok/s** median decode, 8/8 cells ok, **0 degenerate**, 0 failed of 64 requests |

## Decode is flat in context; end-to-end is not

| context | in | out=32 | out=128 |
| ---: | ---: | ---: | ---: |
| 64 | 84 | 21.37 | 21.44 |
| 512 | 575 | 20.89 | 20.87 |
| 1024 | 1134 | 21.04 | 20.91 |
| 2048 | 2254 | 20.75 | 20.95 |

Decode spans **20.75–21.44 tok/s — a 3.3% spread over a 27x context range.** A cost
proportional to the *weights* rather than the context produces exactly that, and it is the
same signature both G5g siblings showed. **KV is not what sets decode speed here.**

End-to-end falls with context (19.93 → 16.33 at out=32) because it carries prefill and the
HTTP round trip. The two disagree by design — quote `tpu_jax_decode_tokens_per_second`.

## Against the physical bound

The L4 has ~300 GB/s of GDDR6 (xprof measured 279.441 GiB/s peak on the JAX sibling). A
decode step that streams the weights once cannot beat:

```
10.209 GB / 300.05 GB/s = 34.02 ms/step  ->  29.4 tok/s      (a FLOOR)
measured                  47.78 ms/step  ->  20.93 tok/s     (71% of it)
```

**So ~13.75 ms/step — 29% of every decode step — is not weight streaming.** That is the
eager-mode cost: HF transformers launches hundreds of small kernels per step from Python,
with no fusion and no graph capture. Nothing here is compiled.

The contrast with the JAX rig on **the same chip** is the useful part:

| | weights | tok/s | ms/step | floor | % of roofline | overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **this rig** — PyTorch, L4 | 10.209 GB | **20.93** | 47.78 | 34.02 | **71.2%** | **13.75 ms** |
| `gpu-jax-g6-2b` — JAX, L4 | 6.155 GB | 48.40 | 20.66 | 20.51 | 99.3% | 0.15 ms |
| `gpu-pytorch-g5g-2b` — PyTorch, T4G | 10.209 GB | 10.88 | 91.95 | 31.90 | 34.7% | 60.05 ms |

## The two comparisons, and only one of them is clean

**Clean — chip only.** Against `gpu-pytorch-g5g-2b`: identical runtime, identical dense
checkpoint, byte-identical `tpu_jax_weight_bytes` (10,208,595,008). Only the GPU differs.

> **1.92x** (20.93 vs 10.875 tok/s).

Note the T4G ran at **34.7%** of its own bandwidth roofline while this runs at **71.2%** —
consistent with the Turing dtype tax (bf16 emulated through fp32) that the JAX sibling
measured at 54% of kernel time, and which does not exist on Ada.

**NOT clean — runtime *and* weight bytes.** Against `gpu-jax-g6-2b` on identical silicon:
that rig serves **6.155 GB** (`ple4 + int8_lm_head`) against this rig's **10.209 GB**. Decode
is bandwidth-bound on exactly those bytes, so the raw 2.31x gap is not a runtime result.

Normalising it away — giving this rig JAX's weight footprint at its own measured overhead —
predicts **~29.2 tok/s**, so JAX would still lead by **~1.66x**. **That residual is the real
runtime gap**, and it is the 13.75 ms/step of eager-mode launch overhead, not the dtype.

## What this run does not establish

- **Nothing above 2,254 prompt tokens was tried.** `MAX_MODEL_LEN=4096` is the inherited
  conservative value; the L4's 23034 MiB is 1.5x the T4G's and the real ceiling is unknown.
- **Concurrency was not swept and cannot be** — `MAX_NUM_SEQS=1`. Batching is exactly what
  would amortise the launch overhead identified above, so this is the obvious next question.
- **No profile was taken.** The 13.75 ms/step is arithmetic against the roofline, not a
  kernel table. `torch.profiler` would attribute it directly.
- **The swapfile path is unexercised** — `g6.2xlarge` has 32 GiB and sits above the gate.

## Provenance note

The install aborted once on a **wrong assertion, not a wrong GPU**: the DLAMI's torch carries
`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` and the L4 is `sm_89`, so an
exact-match arch check failed a healthy device. CUDA cubins run on any device of the same
major with a minor >= their own; fp16/bf16/fp32 matmuls all ran correctly off `sm_86`, and
`torch.cuda.is_bf16_supported()` was True. Fixed in `server.py` and pinned by
`CubinCompatibilityTests`. The install was then completed by hand on the same instance, so
**this run's bootstrap was not end-to-end automated** — the corrected path has not itself
been launched from scratch.
