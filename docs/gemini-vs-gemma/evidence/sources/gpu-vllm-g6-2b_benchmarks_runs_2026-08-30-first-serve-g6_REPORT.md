# 2026-08-30 — first serve, `gpu-vllm-g6-2b`

**This rig had served nothing before today.** Everything below is measured on
EC2 `g6.2xlarge` (1× NVIDIA L4, Ada SM 8.9), `us-east-1d`, spot, instance
`i-0d09e337977a78287`, terminated at the end of the session.

Typed report: `../../reports/2026-08-30-gemma4-e2b-g6.json` (schema 1.1, validates).

## The three questions this rig existed to answer

**1. Does Triton's 512-wide tile fit Ada's shared memory unpatched? — YES.**

vLLM forced `TRITON_ATTN` exactly as predicted (`Gemma4 model has heterogeneous head
dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available`), and
there was **no `OutOfResources`**. The ~96 KiB tile cleared Ada's ~99 KiB per-block
limit. **The G5g sibling's local patch to `triton_unified_attention.py` is not needed
here and must not be ported in.**

**2. Is SM 8.9 covered by the published image? — YES, but not the way the tool said.**

A real bfloat16 matmul executed. But the torch arch list is
`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` — **`sm_89` is not in it.** Ada
runs the `sm_86` cubins by same-major-version binary compatibility. `verify_gpu_arch`
claimed "SM 8.9 is in the arch list", which is false as literally stated; the matmul is
the real evidence. Message corrected in `server.py`.

**3. vLLM vs JAX on identical silicon — the comparison this rig was built for.**

| | tok/s, single stream | notes |
|---|---:|---|
| `gpu-jax-g6-2b`, 2026-08-28 | 48.3–48.5 | same L4, same checkpoint |
| **this run, vLLM 0.28.0** | **46.09** | **~5% slower** |

**Both sides stock** — no Triton patch, no hand-reduced tiles, no from-source build.
That is what the T4G pair never had, and it is the first clean runtime comparison in
this tree.

vLLM's return is batching, not single-stream latency: **46.09 → 360.17 tok/s** from
concurrency 1 → 8, which is **7.8× on 8× the load (98% scaling efficiency)** with TPOT
almost flat (19.85 → 21.71 ms).

## Sweep

`vllm bench serve`, `--dataset-name random --ignore-eos --seed 1234`, driven **on the
instance against localhost** so no WAN latency is in TTFT/ITL.

| input | output | conc | out tok/s | per-stream | TTFT mean (ms) | TPOT mean (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 128 | 1 | 46.09 | 46.09 | 256.4 | 19.85 |
| 1024 | 128 | 2 | 92.60 | 46.30 | 109.5 | 20.89 |
| 1024 | 128 | 4 | 175.83 | 43.96 | 168.4 | 21.57 |
| 1024 | 128 | 8 | 360.17 | 45.02 | 81.6 | 21.71 |
| 4096 | 128 | 1 | 43.30 | 43.30 | 343.4 | 20.57 |
| 4096 | 128 | 8 | 207.83 | 25.98 | 852.2 | 31.84 |

Context cost is modest alone (1024 → 4096 costs 6% of single-stream decode) but
compounds under load (42% at concurrency 8), where prefill competes with decode.

## Roofline — streamed, not resident, bytes

**Decode is not bandwidth-bound here, and that reframes the runtime comparison.**

E2B streams **4.514 GB/token** — transformer matmuls 3.709 GB plus the tied LM head 0.805 GB.
(Corrected 2026-08-30 from 3.382 GB, which missed `use_double_wide_mlp` doubling
`intermediate_size` on E2B's 20 KV-shared layers. **The heading above was written against the
old figure and should be re-judged**: 69% MBU is not obviously "not bandwidth-bound".)
The 4.698 GB PLE table is an **indexed gather, not a stream** (derivation and cross-check in
the monorepo `MODELS.md`, *"Resident is not streamed"*).

| | value |
|---|---:|
| streamed bytes/token | 4.514 GB |
| demand at 46.09 tok/s | 208.0 GB/s |
| L4 bandwidth | 300 GB/s |
| **memory-bandwidth utilisation** | **69%** |
| bandwidth-implied ceiling | 66.5 tok/s |

This run independently confirms that **resident is the wrong denominator**: 46.09 tok/s ×
9.8 GiB resident would demand **485 GB/s**, which a 300 GB/s bus makes physically impossible.

**Consequence for the vLLM-vs-JAX question:** both runtimes sit at roughly *half* the memory
roofline (46.09 and 48.3–48.5 against a ceiling of 88.7), so the ~5% gap between them is an
**overhead / kernel-launch difference at B=1, not a memory-system one**. That matches the T4G
finding that none of the runtimes tested there was bandwidth-bound at `B=1`. It also explains
why batching pays so well: concurrency 8 reaches 360.17 tok/s, well past the single-stream
ceiling, because the streamed weights are amortised across the batch.

## Memory

`nvidia-smi` reported **23034 MiB**, matching the figure this rig carried. vLLM saw
22.04 GiB free, spent 9.8 GiB on weights, 0.22 GiB peak activation, 0.09 GiB CUDA
graphs, leaving **9.65 GiB of KV = 1,076,849 tokens = 65.73× concurrency at 16 K**.

**KV measured 9622 B/token (9.40 KiB) — roughly HALF the ~18 KiB/token this rig's
CLAUDE.md carried as arithmetic.** KV is nowhere near binding, so fp8 KV stays
unnecessary and unjustified here.

## What went wrong, and it cost a launch

**`vllm/vllm-openai:v0.27.2rc0` is not a published image tag.** The first launch's
cloud-init died at
`failed to resolve reference "docker.io/vllm/vllm-openai:v0.27.2rc0": not found`.

`v0.27.2rc0` is the **G5g sibling's `VLLM_REF`** — a *git* ref that rig compiled from
source, which is why it never had to exist on Docker Hub. The fork copied a **source ref
into an image-tag field**. Published releases are `v0.27.0`, `v0.27.1`, `v0.28.0`; there
is no `v0.27.2` of any kind. The comment directly above the constant *already said* the
sibling "built its real image from `VLLM_REF=v0.27.2rc0`".

`test_image_is_at_or_above_the_measured_vllm_floor` did not catch it: it forbids the
known-bad tags but never asserts the tag **resolves**. A blocklist cannot catch a value
that is in no list.

Fixed by moving to **`v0.28.0`** (2026-08-26, newest release), which is above the
model's `per_layer_config` floor. `get_install_progress` correctly rendered the dead
bootstrap as **failed rather than slow** — the ported behaviour did its job.

## Startup

Image pull 105 s · weights download 37.6 s · model load 40.8 s · engine init 124.03 s
(79.32 s compile) · **~334 s from bootstrap to `/health` 200**.

## Cost

Spot `g6.2xlarge` at **$0.9412/hr** in `us-east-1d`.

| operating point | tok/s | $/M output tokens |
|---|---:|---:|
| conc 1, 1024/128 | 46.09 | $5.67 |
| conc 8, 1024/128 | 360.17 | **$0.73** |
| conc 8, 4096/128 | 207.83 | $1.26 |

No $/token comparison against the JAX sibling is offered: it ran on different capacity at
a different spot price, so a dollar difference would be a **price artifact, not a runtime
result**. Compare tok/s.

## Capacity

`g6.xlarge` spot scored **1 in all five AZs**; `g6.2xlarge` scored **3 in `us-east-1d`**
and launched first try. That reproduces the JAX sibling's 2026-08-28 finding exactly:
**quota is not capacity**, and `get-spot-placement-scores` picks a size and AZ far more
cheaply than launching until one works.
