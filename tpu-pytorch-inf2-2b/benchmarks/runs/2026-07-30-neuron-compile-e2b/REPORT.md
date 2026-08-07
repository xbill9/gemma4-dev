# Can neuronx-cc compile the pure-JAX Gemma 4 engine for inf2?

**Run:** 2026-07-30
**Target:** `inf2` (`inf2.xlarge` — 1x Inferentia2, 2 NeuronCore-v2, 32 GB HBM)
**Compiler:** `neuronx-cc` 2.26.6360.0+6f180f47, cross-checked on 2.23.6484.0+3b612583
`--framework XLA --target inf2 --model-type transformer --optlevel 1`
**JAX:** 0.11.0 / jaxlib 0.11.0
**Hardware used:** none. See "Why no Inf2 instance" below.
**Source:** `jax_neuron/compile_probe.py`

## Purpose

`jax_neuron/probe.py` (2026-07-27) established that the JAX Neuron *runtime*
executes a Gemma-shaped decoder block on a live `inf2.xlarge`. It used a
hand-written block, not this repository's model.

This run asks the question that actually gates the port: does `neuronx-cc`
accept the graphs the shared engine emits — the same `prefill_with_kv_cache`,
`make_cached_decode_step`, and sampler that serve on TPU v6e?

This is a **compilability** result. It is not a correctness or performance
result, and nothing here should be quoted as either.

## Why no Inf2 instance

`neuronx-cc` is an ahead-of-time compiler: it reads an HLO module and emits a
NEFF. A NeuronCore is needed to *execute* a NEFF, not to produce one. The
compiler is an ordinary x86-64 Linux wheel:

```bash
pip install "neuronx-cc==2.26.*" --extra-index-url=https://pip.repos.neuron.amazonaws.com
```

Weights are not needed either. `jax.eval_shape` builds the parameter tree as
`ShapeDtypeStruct`s and `jax.jit(...).lower()` accepts those, so the full-size
E2B tree is described (7.29 GB) without allocating a byte of it.

The consequence is a fast, free, hardware-independent gate: every op-level
finding below was reproduced in seconds to minutes, and the fixes were verified
against the compiler rather than against a guess.

## Result: the engine compiles

### Full gemma-4-E2B geometry

35 layers, vocab 262144, hidden 2048, dual attention geometry (head_dim 256
sliding / 512 global), 20 KV-shared layers, W4A16 weights, BF16 KV, batch 1,
context 512. 7.29 GB parameter tree, described abstractly.

| Stage | Lower | HLO | Compile | NEFF | Result |
|---|---|---|---|---|---|
| `make_cached_decode_step` | 1.4 s | 1.3 MB | **1173.0 s** | 44.3 MB | **PASS** |
| `onchip_sample_tpu_v6e_jax` | 0.1 s | 0.05 MB | 10.1 s | 0.1 MB | **PASS** |

The decode module is the whole per-token graph: 35 decoder layers, the packed
int4 unpack and dequantize-then-matmul for every projection, per-layer
embeddings, the KV cache read/write, the 262144x2048 tied LM head, and the
`tanh` logit softcap over the full vocabulary. Nineteen and a half minutes on 4
cores. `--optlevel` does not change the result — see the DMA section — so use
`-O1`, which is the fastest to compile.

Prefill at full size was not compiled in this run — the decode step is the
strictly larger graph in every dimension that matters here, and it is the one
that runs per token.

### Structural configuration

10 layers, vocab 2048, batch 1, context 128. Preserves every architectural
feature that could trip the compiler (interleaved sliding/full attention, KV
sharing across the tail layers, double-wide MLP on shared layers, per-layer
embeddings, packed int4 weights) and shrinks only the extents. Use this to
iterate; it is ~15x faster.

| Stage | Lower | HLO | Compile | NEFF | Result |
|---|---|---|---|---|---|
| `make_cached_decode_step` | 0.6 s | 0.4 MB | 96.5 s | 1.6 MB | **PASS** |
| `prefill_with_kv_cache` | 0.6 s | 0.5 MB | 79.6 s | 0.8 MB | **PASS** |
| `onchip_sample_tpu_v6e_jax` | 0.1 s | 0.1 MB | 10.0 s | 0.1 MB | **PASS** |

No custom calls appear in any module, at either size, so nothing in them is an
artifact of having lowered on the CPU backend.

Two changes to the shared engine were required to get there. Both are recorded
against the compiler error that forced them.

## Does it fit? Yes — 8.16 GB of 16 GiB

A 44.3 MB NEFF invites the wrong question. The NEFF is the **executable**, not
the model: weights enter this graph as parameters, so they are not in the file.
Recompiling the same HLO with `--verbose info` gets the compiler's own
accounting for the full-E2B decode step:

| DRAM Memory Usage | Size |
|---|---|
| **Total** | **8.16 GB** |
| Model Code | 127.50 MB |
| Model Constants | 182.00 KB |
| Unallocated Tensors | 6.86 GB |
| Allocated Tensors | 1.00 GB |
| DMA Ring IO | 15.94 KB |
| DMA Ring Spill | 181.66 MB |

Against the compiler's stated budget — `DRAM size: 17179869184`, i.e. 16 GiB —
that leaves ~7.8 GB of headroom at context 512.

The budget is 16 GiB and not 32 because the compiler emits
`--logical-nc-config=1`: this is a **single-NeuronCore** graph. An `inf2.xlarge`
carries one Inferentia2 with 2 NeuronCores and 32 GB HBM, but nothing here
splits work across both, so only half the device is addressable by this NEFF.
The second core is idle. Tensor-parallel across both cores is unexplored.

"Model Code" (127.50 MB uncompressed, 44.3 MB packed) is what the NEFF holds:
instructions and DMA descriptors — 11,182,964 of them — for a fully unrolled
35-layer graph. It is large because the graph is unrolled, not because anything
is embedded in it.

Headroom for longer context is bounded by the KV cache, not by the weights. At
context 512 the KV term is small; the 6.86 GB of unallocated tensors is
dominated by the 7.29 GB parameter tree (of which the BF16 per-layer-embedding
table is 4.70 GB). `ple_bits=8` would return ~2.35 GB of that, which is the
first lever to pull if a longer context does not fit.

## Warning sign: 19 GB of DMA per decode step

The same log reports DMA traffic for one execution of the decode graph:

| Queue | Instructions | Transfer |
|---|---|---|
| qPoolDynamic | 12041 (18.6%) | 10.38 GB (54.5%) |
| qSPSpillReload0 | 42944 (66.2%) | 4.39 GB (23.0%) |
| qActSpillReload0 | 9705 (15.0%) | 4.22 GB (22.2%) |
| qDVESpillReload0 | 93 (0.1%) | 48.00 MB (0.2%) |
| qSPIO0 | 51 (0.1%) | 61.00 KB (0.0%) |
| **Total** | **64834** | **19.04 GB** |

Roughly **8.6 GB — 45% — is spill/reload**: values written to DRAM and read
back because they did not stay resident, not weights being streamed for the
first time. Only `qPoolDynamic` resembles useful traffic, and even 10.38 GB
exceeds the 7.29 GB parameter tree.

Applying this repository's own rule — cross-check against an absolute physical
bound, not another configuration — Inferentia2 provides roughly 410 GB/s per
NeuronCore, so 19.04 GB/token implies a floor near **46 ms/token (~21 tok/s)**
at batch 1 before a single MAC is counted. If that estimate reflects runtime, it
makes the port bandwidth-bound by a wide margin.

### `--optlevel` does not change this, and the first attempt to show that was wrong

The obvious objection is that these are `-O1` numbers and `-O2` would fix them.
That was the first thing checked, and the check was botched in an instructive
way. Recompiling the same HLO at `--optlevel 2` produced *byte-identical*
results — 8.16 GB, 19.04 GB, the same queue table. Read naively that says "even
optimization does not help," which is a much stronger claim than the evidence
supported.

It was the wrong conclusion because the A/B had the same baseline on both sides.
Grepping the sub-invocations shows `neuronx-cc --optlevel 1` already launches
the backend that does scheduling and allocation with `--optlevel 2 --policy 3`.
The front-end flag changed exactly one tensorizer option
(`--keep-remat-dma-transpose`) and never reached the pass that would reduce
spill.

Sweeping all three levels on the structural config settles it:

| CLI flag | Backend (`walrus_driver`) | DMA total |
|---|---|---|
| `--optlevel 1` | `--optlevel 2 --policy 3` | 31.63 MB |
| `--optlevel 2` | `--optlevel 2 --policy 3` | 31.63 MB |
| `--optlevel 3` | `--optlevel 2 --policy 3` | 31.63 MB |

So `neuronx-cc --optlevel` is a **no-op for this metric** in 2.26. The useful
consequence is the opposite of the original worry: 19.04 GB is *already* the
backend's optimized schedule at its default policy, not a lazy `-O1` number.
Tuning the CLI level will not improve it, and anyone who tries will measure
nothing and conclude wrongly.

The levers that remain are the graph, not the flag: `dequant_at_load` (drop the
per-forward W4A16 unpack entirely), `ple_bits=8` (−2.35 GB resident), windowed
KV, and batching to amortize weight traffic across tokens.

**Still not a result.** This is a static compiler estimate from a CPU-lowered
HLO for a graph that has never executed. Treat 19.04 GB as a hypothesis to test
on hardware with `neuron-monitor`, not as a throughput claim.

## Finding 1: `lax.top_k` is rejected — the only true model-level blocker

```
[NCC_EVRF001] Operator topk is not supported. Locate the operator in source or
libraries and replace it with an alternate implementation via Neuron Kernel
Interface (NKI).
```

The top-k sampler is the one place the engine used it. Four formulations were
compiled in isolation against `neuronx-cc` (batch 1, vocab 262144):

| Formulation | Result | NEFF |
|---|---|---|
| `argmax` (greedy, `temperature <= 0`) | PASS | 0.02 MB |
| `jax.random.categorical` alone | PASS | 0.1 MB |
| `lax.top_k` threshold | **FAIL** | — |
| iterative masked-max threshold | PASS | 0.1 MB |
| `jnp.sort` threshold | PASS | 2.5 MB |

`ports/gemma4/jax_e_model.py:_kth_largest` now takes the iterative masked-max
when `caps.device_top_k` is False: peel the maximum off k-1 times, masking each
one out, then take the max of what remains. Only `max` and `where`. It was
chosen over the `sort` variant on graph size — 0.1 MB against 2.5 MB — since a
full O(V log V) sort of 262144 elements does far more work than 39 reductions.

Ties: a round masks out *every* element equal to the current max, so with
duplicates this yields the k-th largest **distinct** value and the caller keeps
more than k candidates. That widens the sampled distribution slightly rather
than truncating it. On distinct float logits it is exactly equal to
`lax.top_k`, which `tests/test_backend_caps.py` pins.

Greedy decoding — the default, `temperature=0` — is unaffected: it is `argmax`,
which compiles, and is byte-identical on both paths.

## Finding 2: fp8 KV storage is rejected; int8 is not

```
[NCC_EVRF051] Data type F8E4M3FN is not supported on TRN1/TRN2. Target TRN3 or
later hardware, or use the --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 flag to
cast F8E4M3FN to F8E4M3.
```

`inf2` carries NeuronCore-v2, which is TRN1-class. The same decode step compiles
with an int8 cache:

| KV cache dtype | Compile | NEFF |
|---|---|---|
| `bf16` | PASS 96.5 s | 1.6 MB |
| `int8` | PASS 104.3 s | 1.7 MB |
| `fp8_e4m3` | **FAIL** (verifier) | — |

int8 carries the identical per-token scales for identical one-byte capacity, and
was already the more accurate of the two on TPU, so it is the portable choice.
`jax_engine.resolve_cache_dtype` now rejects the fp8 names on Neuron and names
int8 in the error rather than failing later inside the compiler.

## Correction: the inherited "avoid scatter" rule is false here

`deployments/aws-inf2/README.md` carried a risk table from a sibling
PyTorch/NxD port of the same model family to the same hardware. One row said
`neuronx-cc` prefers plain arithmetic over data-dependent scatter, and named the
sampler's `mask.at[arange(B)[:, None], idx].set(vals)` and the KV cache's
`dynamic_update_slice`.

That was briefly encoded as a `scatter=False` capability here. It is not true of
this stack. Compiled in isolation — indices and values supplied as inputs so
`topk` could not fail first — the vector-indexed scatter **PASSES** (0.02 MB
NEFF), and the scalar `valid.at[:, slot].set(True)` and the KV
`dynamic_update_slice` both compile inside the decode step that passed above.

The capability field was **removed** rather than left asserting something
measured false. The sampler still takes the threshold path, but because
`lax.top_k` is unavailable, not because the scatter is.

The same table warned that the `tanh` logit softcap over the 262144-token vocab
overflowed the 196608 B/partition SBUF (`NCC_INLA001`) in that port. Also not
reproduced: the softcap over the real vocabulary is inside the full-E2B decode
module above, and it compiled.

This is the same failure mode this repository has hit before: a claim inherited
from an adjacent context, plausible, and wrong. Each check cost about ninety
seconds.

## Toolchain note: HLO instruction ids need renumbering

The first compile of every module aborted before reading a single operator:

```
F ./xla/hlo/ir/hlo_instruction.h:1848] Check failed: unique_id_ < (2147483647)
(4294967297 vs. 2147483647)
```

Current XLA (inside jaxlib 0.11) packs an instruction's unique id as
`(computation_id << 32) | index`. `neuronx-cc` 2.26 embeds an older XLA whose
`HloInstruction::unique_id()` is an int32, so everything past the first
computation overflows its CHECK.

Not a model problem and not a Gemma problem — a version skew, and one that
presents as a compiler crash rather than a diagnostic.
`compile_probe.normalize_instruction_ids` renumbers instruction ids densely from
zero and rewrites the three fields that reference them (`operand_ids`,
`control_predecessor_ids`, each computation's `root_id`). Computation ids are a
separate namespace and are already small.

Anyone lowering JAX to `neuronx-cc` on a current jaxlib will hit this.

## Cross-checked against the compiler the deployment actually pins

`deployments/aws-inf2/user_data.sh` pins `jax-neuronx[stable]==0.6.2.1.0.*`, an
SDK-2.28 build, which does not ship `neuronx-cc` 2.26. A finding that only held
on the newer compiler would be useless to that deployment, so the two decisive
cases were re-run against **neuronx-cc 2.23.6484.0**:

| Case | 2.26 | 2.23 |
|---|---|---|
| `lax.top_k` threshold | FAIL `NCC_EVRF001` | FAIL `NCC_EVRF001` (identical) |
| iterative masked-max threshold | PASS | PASS |
| vector-indexed scatter | PASS | PASS |

Same verdicts, same error code. The instruction-id renumbering is required on
both — it is a jaxlib-side change, not a compiler-version one.

## Also measured

- **PRNG.** `threefry2x32`, `rbg`, and `unsafe_rbg` all compile a categorical
  sample over the 262144-token vocabulary. The Inf2 entrypoint's
  `JAX_DEFAULT_PRNG_IMPL=rbg` is AWS guidance and a cost preference, **not** a
  compile requirement, and the comment claiming Threefry ops were unsupported
  was wrong and has been removed.
- **W4A16 reference path.** The packed-int4 unpack (int32 shifts and masks) and
  the dequantize-then-matmul compile as part of every passing decode and prefill
  module above. The fused Pallas kernel was not attempted and cannot be: Mosaic
  has no Neuron backend. `set_w4a16_impl` now refuses `"fused"` there instead of
  routing it to the Pallas interpreter, which would have unrolled the kernel's K
  loop into the graph.

## What this does NOT establish

- **No numerics.** A NEFF exists; nothing here ran it. Greedy-token parity
  against the CPU reference on the real checkpoint is the next gate.
- **No performance.** No claim about tokens/s, latency, or the cost of the
  iterative threshold at runtime versus `lax.top_k` on TPU. The 39 dependent
  reductions are cheap in graph size; their latency on a NeuronCore is unmeasured.
- **No end-to-end serving.** The HTTP path, the safetensors loader, host-side
  PLE quantization, and buffer donation on the real plugin are all untested on
  device. `caps.buffer_donation=True` for Neuron is still AWS documentation, not
  a measurement — it is the one capability in the table that is not backed by a
  compile.
- **CPU-lowered HLO.** The modules were lowered on the CPU backend. No custom
  calls appeared, which is the main risk this creates, but the Neuron PJRT
  plugin could still emit a different module for the same JAX code.

## Reproduce

```bash
pip install "neuronx-cc==2.26.*" --extra-index-url=https://pip.repos.neuron.amazonaws.com

python3 jax_neuron/compile_probe.py --tiny                      # ~3 min, all stages
python3 jax_neuron/compile_probe.py --stage decode              # full E2B, ~20 min
python3 jax_neuron/compile_probe.py --tiny --kv-dtype fp8_e4m3  # reproduces NCC_EVRF051
python3 jax_neuron/compile_probe.py --lower-only                # HLO only, no compiler
```

Exit status is non-zero if any stage fails to lower or compile. Artifacts and
neuronx-cc's scratch land in `.neuron_compile_probe/` (gitignored) — the
compiler otherwise scatters hash-named working directories through its CWD and
cleans up none of them.

Timings above are from 4 cores; the full-size decode compile peaks around
7.5 GB RSS in `walrus_driver`. `--optlevel` is left at 1 deliberately: levels
1, 2, and 3 produced identical memory and DMA figures, so the only thing a
higher level costs is compile time.

To reproduce the memory and DMA tables, compile a dumped HLO directly with
logging on:

```bash
neuronx-cc compile --framework XLA --target inf2 --model-type transformer \
  --verbose info --output decode.neff .neuron_compile_probe/decode.hlo
```
