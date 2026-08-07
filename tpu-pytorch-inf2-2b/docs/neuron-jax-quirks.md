# JAX on Inferentia2: platform quirks, measured on the device

Companion to `gemma4-quirks.md`. That file documents the *model*; this one
documents the *platform* — jax-neuronx on `inf2`, where the runtime announces
itself with `Platform 'neuron' is experimental and not all JAX functionality may
be correctly supported!` and then means it.

Every entry is from a measurement on `inf2.xlarge`, 2026-07-31/08-01, with
jax 0.6.2, jax-neuronx 0.6.2.1.0.6446, libneuronxla 2.2.17544.0, neuronx-cc
2.24.8799.0, Neuron runtime 2.31.24. Full workings in
`benchmarks/runs/2026-07-31-inf2-serving-perf/REPORT.md`.

Status legend: **✅ measured** directly · **⚠️ inferred** from correlated
evidence · **❓ open** — stated because it matters, not because it is settled.

---

## 1. A too-large gather returns **zeros**, not an error ✅

This is the one that costs everything else.

Gathering rows from the 4.70 GB per-layer-embedding table on the NeuronCore does
not fail loudly. Inside the engine it yields an all-zero tensor and execution
continues:

| tensor | correct path | fault path |
|---|---|---|
| `embed_tokens[ids]` absmax | 0.36133 | **0.00000** |
| `gather_ple(...)` absmax | 0.86328 | **0.00000** |
| prefill logits absmax | 26.50908 | **0.00000** |
| decode step 1 logits | 22.39555 | **NaN** |

Zero logits make `argmax` return token 0, which is the pad id, which is in
`_eos_ids()`. So the server returns a clean `200 OK` with `finish_reason: "stop"`
and **zero completion tokens**. Nothing in the response, the logs, or `/metrics`
indicates a fault.

The same gather in isolation, at every geometry that *fits*, is exactly correct —
including the real `embed_tokens` shape:

| vocab | width | dtype | GB | verdict |
|---|---|---|---|---|
| 262144 | 256 | bf16 | 0.13 | OK |
| 262144 | 1536 | bf16 | 0.81 | OK ← `embed_tokens` |
| 262144 | 1536 | f32 | 1.61 | OK |
| 262144 | 8960 | bf16 | 4.70 | **`Failed to allocate 4697620480 bytes on DEVICE`** |

Plain `table[ids]`, `jnp.take`, `take_along_axis` and one-hot matmul are all
correct under both eager and jit at the sizes that fit. **The gather primitive is
fine. The allocation is not.**

In isolation, 4.70 GB raises `RESOURCE_EXHAUSTED` / `NRT_RESOURCE`. Inside the
engine, the same size yields zeros. ⚠️ That the two are the same underlying
failure is inference, not proof — but the size boundary and the affected tensor
match exactly.

**Practical rule:** on this stack, treat any single device tensor approaching
~4–5 GB as unsafe, and never assume a gather that "ran" produced data. Check a
norm, not a shape.

## 2. `NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1` is a correctness workaround costing 2700x ✅

`deployments/aws-inf2/neuron_entrypoint.py` sets it. It reads like a performance
preference. It is not — it is what makes quirk 1 survivable, by pushing the
gathers to the host where they compute correctly.

Same engine, weights, device, checkpoint; only this variable differs:

| | `=1` | unset |
|---|---|---|
| 20 tokens, warm | **163.34 s** | **0.06 s** |
| steady per-token | 6.89 s | 0.002 s |
| host RSS delta per request | 13.71 GB | 0.00 GB |
| end-to-end HTTP output | `'Paris.'`, 5 tokens | `''`, **0 tokens** |

Both halves matter. Set, it is correct and unusable. Unset, it is fast and wrong.
The cost is structural: the gathers it relocates are over the PLE table, which is
72% of the weights, so every decode step ships parameter-scale buffers through
host memory. On a 16 GB host that exhausts RAM and swaps — 116 MB/s out, 55%
iowait — while **device HBM never moves** and the accelerator looks idle and
healthy throughout.

❓ The likely real fix is `ple_bits=8` (2.35 GB) or `ple_bits=4` (1.19 GB), which
puts the table under the boundary in quirk 1. `JaxGemmaEngine` already supports
it; `jax_openai_server.py` never passes it. Confirmed only that `ple_bits=8`
loads 4.23 GB against 6.56 GB — whether it restores correctness with the
workaround off is **untested**, because it is a new graph and the cold compile
exceeded the test window.

## 3. The JAX persistent compilation cache is load-bearing ⚠️

`ports/gemma4/backend.py` records `persistent_compilation_cache=False` for
Neuron, while the deployment sets `JAX_COMPILATION_CACHE_DIR` anyway and has
written 160 entries there. The declared capability and the running system
disagree.

The running system is the one to trust. With the cache: a cold prefill compiles
in ~2.3 s and a first request completes in minutes. Without it, the identical
compile ran **over 38 minutes without finishing** and had to be killed.

Anyone "fixing" the configuration to match the capability table will take first-
request latency from minutes to unbounded. Fix the table, not the config.

## 4. The NEFF cache is configured and never populates ⚠️

`NEURON_COMPILE_CACHE_URL=/opt/gemma4/cache/neuron` is set and pointed at the
persisted volume. That directory has stayed at 4.0K across every request in this
session. Meanwhile `NEURON_CC_FLAGS` carries only `--model-type=transformer` —
no `--cache_dir`.

`NEURON_COMPILE_CACHE_URL` is a *torch-neuronx* variable; the neuronx-cc path
wants `--cache_dir` in `NEURON_CC_FLAGS`. Untested against the plugin docs, but
it matches an empty cache exactly. This matters more than it looks: a
build-on-8xlarge / serve-on-xlarge split depends entirely on that artifact
surviving the move.

## 5. Eager ops invoke the compiler at runtime ✅

A bare `jnp.full` on Neuron reaches `RunNeuronCCImpl` — the first sign of this
was a `FileNotFoundError: 'neuronx-cc'` raised from an array *construction* when
the compiler was not on `PATH`.

Fixed shapes are cached and free. A **new** shape costs ~3.9 s per call:

| case | s/call |
|---|---|
| eager `argmax` `[1,262144]`, fixed shape | 0.000 |
| jit `argmax`, same | 0.000 |
| eager `argmax`, **new shape each call** | **3.929** |

`onchip_sample_tpu_v6e_jax` is not jitted and runs per token, so keep its shapes
constant. Anything shape-varying in a per-token path is a compile per token.

## 6. `max_new_tokens` is a static argname fed from HTTP ✅

`jax_engine.py:353` makes `max_new_tokens` static on `_jit_prefill`, and
`jax_openai_server.py` passes the request's `max_tokens` straight into it. Every
distinct value triggers a fresh trace and compile — measured **4.69 s** for a new
value against 0.06 s for a cached one, and 167 s vs 65 s through HTTP.

Beyond latency this is an availability hazard: a caller varying `max_tokens`
forces unbounded multi-minute compiles. It should ride a bucket ladder, as the
sequence dimension already does.

## 7. `hbm_limit_bytes` is per device, not aggregate ✅

It reports 17,179,869,184 (16 GiB) whether one core is visible or two. The chip's
32 GB is 16 GiB *per core*, so a single-device engine always had 16 GiB.

`NEURON_RT_NUM_CORES=1` therefore withholds a **device**, not memory. Setting it
to 2 cut parameter load from 263.0 s to **68.8 s** (3.8x) and exposed the second
core, but did not change serving latency at all — a visible device a
single-device engine never targets cannot help.

## 8. Per-buffer dispatch has a cliff above ~128 buffers ✅

Total bytes held constant at 2.05 GB, split across N arrays passed as a jit
argument:

| arrays | s/call | compile s |
|---|---|---|
| 8 | 0.008 | 10.4 |
| 128 | 0.008 | 10.6 |
| 512 | **0.418** | 50.0 |

A 52x per-call jump at identical total bytes. A Gemma 4 E2B parameter pytree has
several hundred leaves, so it sits past the cliff. Consolidating per-layer
tensors into stacked arrays would move it back under.

## 9. No compressed-compute path exists on Neuron ✅

`_CAPS.pallas` is `False`, so `set_w4a16_impl` forces `"reference"` — the fused
kernel lowers through Mosaic and neuronx-cc cannot accept it. The reference path
is `qat_w4a16_reference_linear_jax`, which materializes a dense BF16 copy of
every weight before the matmul:

| variant | s/call | compile s |
|---|---|---|
| W4A16 reference (dequantize+matmul) | **0.027** | 39.5 |
| dense BF16 matmul | **0.001** | 3.9 |

27x per call, 10x compile. So W4A16 weights are **stored** compressed and
**read** dense — exactly the milestone 5 caution in `jax_neuron/README.md`,
occurring silently, because the loud fallback warning only fires when `"auto"` or
`"fused"` degrades at runtime and the default is already `"reference"`.

## 10. Buffer donation fails at runtime ✅

`buffer_donation=False` for Neuron in the capability table, against a measured
must-alias donation error. This removes the lever that keeps a large argument in
place on TPU. Note it does **not** cause per-call re-staging: a 2.05 GB pytree
passed as a jit argument re-stages nothing (measured 0.00 GB delta).

Do not "fix" this by closing over the parameters instead. Measured, that is
strictly worse — JAX captures them as lowering constants (`A large amount of
constants were captured during lowering (2.05GB total)`) and parks the whole
tree in host RAM permanently.

## 11. `VmHWM` is the load peak, not residency ✅

`top` shows ~14.5 GB RSS and it is misleading. The load path holds the raw
safetensors and the converted parameter tree simultaneously inside
`convert_safetensors_to_jax_params`, peaking at **12.5 GB**, then frees both:
settled `VmRSS` is **0.29 GB**.

Read `VmRSS` and `VmSwap` from `/proc/<pid>/status`, not `RES` from `top` sampled
during a request. A 16 GB host survives this peak, but only just, and it is why
~1.7 GB lands in swap before the first request arrives.

## 12. The second core is reachable from a second process ✅

`NEURON_RT_VISIBLE_CORES=1` lets a separate process initialise on core 1 while
the server holds core 0 — useful for running probes against a live deployment
without stopping it. With `NEURON_RT_NUM_CORES=2` the server claims both and this
stops working; stop the service first.

---

## Method notes, earned the hard way

**Token count is not correctness.** An isolation run counted 20 tokens returned
and never inspected them. It would have been recorded as a 2700x win. The
configuration that produced those 20 tokens returns an empty string end-to-end.
Check the *content*, and against a known-correct reference, every time.

**Measure outside the server before blaming the engine.** Seven synthetic
isolation tests reproduced nothing. What finally separated the variables was
driving the real `JaxGemmaEngine` directly: 20 tokens in **0.06 s** with zero
host churn, against 60–126 s through HTTP. Same engine, same weights, same
device. That localised the entire cost to *process configuration*, which no
amount of profiling the model would have found.

**A request that looks hung usually is not.** `curl` gave up at its 120 s default
while the server completed the work and recorded it. Check
`tpu_jax_requests_total` in `/metrics` before concluding a failure.

**Cross-check against a physical bound.** 6.56 GB of weights at ~11 s per decode
step is 0.65 GB/s against ~820 GB/s of HBM — roughly 0.1%. Three orders of
magnitude is not a range tuning occupies, and that reading is what redirected the
search from optimisation to a structural fault.

## How to check the next one

1. Dump the tensors, not the tokens. `jax.device_get` the logits and the first
   few intermediates, save to `.npz` under each configuration, and diff. The
   first tensor that disagrees names the op — that is how quirk 1 went from "the
   model is slow and sometimes wrong" to "the embedding gather returns zeros" in
   one run.
2. Reproduce it in isolation at the real geometry. Sizes and dtypes matter here
   in ways they do not on TPU: the same gather is correct at 0.81 GB and silently
   zero at 4.70 GB.
3. Vary one environment variable at a time. Three were set together in the
   entrypoint; only one mattered.
