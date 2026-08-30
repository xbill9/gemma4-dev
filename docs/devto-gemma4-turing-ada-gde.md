---
title: "Gemma 4 in Pure JAX: What Changes Between Turing and Ada, and What Doesn't"
published: false
description: "One hand-written Gemma 4 port, no PyTorch and no vLLM, on two NVIDIA GPUs a generation apart. Most of it ports untouched. Two things do not, and one of them was quietly eating 87% of decode."
tags: jax, gemma, cuda, machinelearning
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/devto-cover-gde.jpg
---

This article is a measurement report on running a hand-written **Gemma 4** port in
**pure JAX** across two NVIDIA GPUs a generation apart, and on the two places the
"it's just JAX" abstraction leaks. One of those leaks costs 87% of decode and
nothing in the logs is red.

The code is here:

https://github.com/xbill9/gemma4-dev

## What This Article Measures

One port, one build, one checkpoint, two cards. Everything below comes from two
archived runs, named so you can check them.

| | G5g | G6 |
|---|---|---|
| Chip | NVIDIA **T4G** — Turing, SM 7.5, 15,360 MiB | NVIDIA **L4** — Ada, SM 8.9, 23,034 MiB |
| Host | `g5g.2xlarge` spot — Graviton2, **aarch64** | `g6.2xlarge` spot — **x86_64**, `us-east-1d` |
| Checkpoint | `google/gemma-4-E2B-it`, dense reference | `google/gemma-4-E2B-it`, dense reference |
| Compute dtype | `float16` (device-chosen) | `bfloat16` (device-chosen) |
| Stack | jax 0.11.1, CUDA from pip | jax 0.11.1, Python 3.14 |
| Run cited | `2026-08-28-full-run-cached-g5g` | `2026-08-28-first-serve-g6` |

Build id `51bc52c9e2e9` on both, config `ple4 + int8_lm_head`, and
`tpu_jax_weight_bytes` reads **6,155,450,950** on both cards — the same integer.
Only the chip and its host differ.

## At This Point You Should Have

- An AWS account with spot capacity for `g5g.2xlarge` and `g6.2xlarge`
- A Hugging Face token with access to `google/gemma-4-E2B-it`
- Python 3.13 or newer on the instance, system-wide, with no virtualenv
- No CUDA toolkit, no Rust toolchain, and no compiler — none of them are needed

## Why Pure JAX At All

The port lives in `ports/gemma4/` and is driven by a generation loop behind an
OpenAI-compatible server. No PyTorch, no vLLM, no `torch_xla`.

The premise under test is that the same source runs on both cards with nothing
changed but a config file. It mostly holds. The interesting part is where it does
not.

## Gemma 4 E2B Is Not a Stock Transformer

Any port has to carry four irregularities, and none of them are optional.

1. **Two attention geometries.** Sliding layers use `head_dim=256`, global layers
   use **512**. Most inference stacks assume one head dimension per model.
2. **8:1 MQA**, so the KV budget is nothing like the parameter count suggests.
3. **A KV-share map** that collapses **35 layers onto 15 caches**.
4. **A 512-slot sliding ring**, plus per-layer embeddings held in a **4.70 GB**
   table that is quantized to 4 bits on load.

## The Geometry Is What Breaks Other Stacks

That first irregularity is the expensive one. On the vLLM path the heterogeneous
head dims force the Triton attention backend:

```
Gemma4 model has heterogeneous head dimensions
(sliding=256, global=512); falling back to the Triton attention backend
```

On a Turing GPU that backend then asks for shared memory the hardware does not
have:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 147456, Hardware limit: 65536
```

**JAX never enters that conversation.** Attention is ordinary XLA rather than a
hand-tiled kernel, so there is no per-block shared-memory ceiling in the attention
path at all. The irregular geometry that is a special case everywhere else is just
array shapes here.

## Leak One: The dtype Policy Has To Read the Device

This is the single most expensive lesson in the repository.

**A wrong compute dtype does not raise. It emulates.** `bfloat16` on a pre-Ampere
GPU does not fail — XLA routes it through fp32 and most of decode disappears into
conversion. Nothing in the logs is red.

So the port does not take the dtype from a config file. It reads the live compute
capability off the device:

```python
COMPUTE_DTYPE = float16 if IS_PRE_AMPERE else bfloat16
```

On the SM 8.9 Ada card that resolves to `bfloat16`. On the SM 7.5 Turing card it
resolves to `float16` — Turing's only real 16-bit datapath, since it has neither
bf16 nor fp8.

## The Process States What It Decided

The first line the server emits is the policy, so a misconfiguration is one `grep`
away rather than a mystery in the throughput:

```
INFO ports.gemma4.jax_e_model: jax_e_model device policy: platform=gpu
compute_capability=8.9 compute_dtype=bfloat16 pallas_interpret=False
```

`pallas_interpret=False` matters just as much. It is the difference between
serving and silently running a simulator.

## Leak Two: Pallas Is Not Portable as a Memory Model

Here is the part that does not port, and it is not a bug. It is a real hardware
difference wearing a portable API.

The fused **W4A16 kernel is written in Pallas**, and it was tiled for a device with
16 MB of scratchpad per core. At this model's shapes the tiles want **550 KiB to
1.1 MiB per block**.

On a GPU, Pallas lowers through Triton, and those tiles become **shared memory**.
Turing gives you 64 KiB per block. Ada raises the ceiling, but nowhere near a
megabyte.

So the fast path runs on **neither card**. The engine computes the requirement at
startup and refuses with the arithmetic attached, rather than dying as a cryptic
`OutOfResources` at the first token:

```python
check_w4a16_fits_scoped_memory()
```

The practical consequence is that both GPU rigs serve the **dense reference
checkpoint** at 16-bit. **Pallas is portable as an API and not portable as a
memory model.** That boundary is worth knowing before planning a port around a
fused kernel.

## The Bug That Returns 200 OK

A padding-eviction bug in the KV ring cache cost a week, and it is the kind only
Gemma 4's geometry produces.

The invariant is that **a cache index is an absolute real position, and padding
never occupies an index a real position uses.** A port that right-pads into the
512-slot ring violates it, and the failure mode is not a crash and not a NaN. It is
a token loop — a clean HTTP `200`, `status: "success"`, and output like
`The The The The`.

Nothing in the logs is red. Nothing in the metrics is red. The only thing that
catches it is a degeneracy check on the output itself, which the server now runs on
every response.

The scariest bugs in this project all returned success.

## Install: No Build Step

`jax[cuda13]` supplies CUDA as wheels, so the install needs no CUDA toolkit, no
Rust, and no compiler on the box.

```
Install: 117 s, with the cache restore included
```

XLA's persistent compilation cache ports as-is. On the T4G rig it restores **805
files / 12 MB in 6 seconds** onto a fresh instance, from a box that had already
been terminated.

## Warm Up At the Shape You Measure

`max_new_tokens` is a `static_argnames` entry, so `(bucket, max_tokens)` is the
compiled shape on every backend. A harness that does not warm up misreports the
rig badly.

On the T4G the first request off a fresh engine took **18.06 s against 4.50 s
warm** — a 4.0x whole-request ratio, measured in
`2026-08-21-cuda13-py314-g5g`.

That run also notes something worth repeating: the 56x figure from the first-serve
baseline is **TTFT specifically**, not the same measurement as the whole-request
ratio. They are not interchangeable.

## The Sweep

64 output tokens, concurrency 1, 3 repeats per cell, median. "Decode, gauge" is the
engine's steady-state counter. "End-to-end" is wall time over the whole request,
prefill included.

| Input tokens | T4G gauge | T4G end-to-end | L4 gauge | L4 end-to-end |
|---:|---:|---:|---:|---:|
| 41 | 12.9 tok/s | 12.43 tok/s | **48.5 tok/s** | 46.23 tok/s |
| 521 | 13.0 tok/s | 11.28 tok/s | **48.4 tok/s** | 42.87 tok/s |
| 2,057 | 12.9 tok/s | 8.22 tok/s | **48.3 tok/s** | 34.57 tok/s |
| 3,593 | — | — | **48.3 tok/s** | 27.55 tok/s |

## Decode Is Flat; End-To-End Is Not

Decode moves 0.8% across a 50x context range on the T4G and 0.4% on the L4. End-to-end
falls hard on both.

That fall is prefill being linear in the padded bucket, not decode degrading. They
are two different claims, and conflating them makes a benchmark a lie. **Quote the
gauge.**

A cost proportional to the **weights** rather than the context produces exactly this
shape, which is why KV is not what sets decode speed on either card — despite
Gemma 4's whole KV story.

On context specifically: `MAX_MODEL_LEN=4096` is the honest number on the T4G.
4,105 prompt tokens serve; 5,120 fails on a prefill transient.

## The Profile: 87% of Decode Was Not Math

Profiling decode with xprof on the Turing card, 20 decode steps with the service
stopped:

| | 🥉 T4G (SM 7.5) | 🥇 L4 (SM 8.9) |
|---|---:|---:|
| dtype conversion | 54.1% | **0.0%** |
| fp32 `gemvx` | 32.8% | **absent** |
| Tensor Core | 0.0% | **0.0%** |
| Total kernel time | 1,466.0 ms | **362.8 ms** |
| Decode, gauge | 12.9 tok/s | **48.4 tok/s** |
| Peak HBM bandwidth | 298.083 GiB/s | 279.441 GiB/s |
| Share of bandwidth roofline | 26% | **~100%** |

1,466 ms of kernels across 108 distinct kernels on a Tensor Core GPU, without one
Tensor Core firing. More than half of decode went to converting numbers between
formats before any math happened.

## The Obvious Explanation Was Wrong

The obvious hypothesis was bf16 weights being converted on a chip with no bf16
datapath. So the checkpoint was converted to float16 host-side and re-run.
Parameter dtypes read `{'float16': 541, 'uint8': 1, 'int8': 1}` — and **conversion
stayed at 54.0%**.

The measurement itself is solid. The same profile on a different instance, a
different AMI and a restored cache landed at 1466.0 ms against 1467.1 ms. **1.1 ms
apart on 1467.**

## It Was the Datapath, Not the Checkpoint

The Ada card resolves it. Converting the stored weights changed nothing because
storage dtype was never the problem: Turing has no native bf16, and the fp32
`gemvx` line is the tell — XLA was round-tripping through fp32 regardless of what
the file on disk said.

Give it a card where storage and compute dtype actually match, and the 54%
conversion and the 32.8% fp32 path vanish **together**. An 87% tax gone, for 3.7x
the throughput, and a rig sitting at its bandwidth roofline instead of 26% of it.

The `/health` endpoint on the L4 reports `weights=bfloat16 activations=bfloat16
kv_cache=bfloat16 pre_ampere=false` — storage dtype and compute dtype matching for
the first time on this engine.

## What Survived, and Is Still Unexplained

Tensor Core utilization is **0.0% on the Ada card too** — 100 distinct kernels,
362.8 ms of them, and not one Tensor Core firing.

Removing the dtype pressure made the machine roughly four times faster without
making it touch the hardware it was sold for. That is the open question now, and it
is a better one than the question this started with.

## Teardown

Both rigs run on spot capacity and are terminated after collection. The XLA cache
is pushed to S3 before teardown, which is what makes the 6-second restore on a
fresh instance possible.

## Summary

The goal of this article was to find out which parts of "it's just JAX" survive a
move between GPU generations. The key to the solution was reading the compute dtype
off the live device rather than a config file. The measured results were:

- **The model code ported untouched.** All four Gemma 4 irregularities, both
  attention geometries, the KV-share map and the ring — identical source on both
  cards.
- **The compilation cache and static-shape discipline ported untouched**, including
  an 805-file restore in 6 seconds onto a fresh instance.
- **The fused Pallas kernel ported to neither card**, because it was written
  against a 16 MB scratchpad and GPU shared memory is 64 KiB per block on Turing.
- **The dtype tax was 87% of decode on Turing and 0.0% on Ada**, worth 3.7x
  throughput, with nothing red in any log.
- **Tensor Core utilization is 0.0% on both cards** and remains unexplained.

Scope: two spot instances, one in `us-east-1d`, each measured once with 3 repeats
per sweep cell and medians reported. The two boxes differ in host architecture
(aarch64 against x86_64) and base image as well as in GPU, so this is not a
single-variable experiment; the payload is byte-identical across them — same build
`51bc52c9e2e9`, same config, same 6,155,450,950 bytes of weights — which is the
basis for attributing the difference to the chip. The Turing profile was reproduced
on a second instance at 1466.0 ms against 1467.1 ms; the Ada profile was measured
once.

The strategy for using MCP for Gemma 4 serving across GPU generations was validated
with a incremental step by step approach.
