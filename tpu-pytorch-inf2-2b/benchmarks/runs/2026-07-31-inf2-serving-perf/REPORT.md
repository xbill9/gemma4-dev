# Why is HTTP serving 0.08 tok/s on Inferentia2?

**Run:** 2026-07-31
**Instance:** `inf2.xlarge` spot, `us-east-1d`, `i-048b314be74351710`
**AMI:** `ami-06fdee0f57bad6bc1`, Ubuntu 22.04.5 LTS, Neuron runtime 2.31.24
**Stack:** jax 0.6.2 / jaxlib 0.6.2, jax-neuronx 0.6.2.1.0.6446, libneuronxla 2.2.17544.0, neuronx-cc 2.24.8799.0, transformers 5.14.1
**Checkpoint:** `google/gemma-4-E2B-it-qat-w4a16-ct` (compressed W4A16, `--quant-mode w4a16`, int8 KV)
**Server:** `deployments/aws-inf2/neuron_entrypoint.py`, `--max-model-len 4096`, bound `127.0.0.1:8000`

## Result: one environment variable costs 2700x, and it is load-bearing

Milestone 4's HTTP contract works and is correct — five identical greedy requests
all returned `'Paris.'` for `"The capital of France is"`, `finish_reason: stop`,
6 prompt / 5 completion tokens. It is also **~0.08 tok/s**, about 11–13 s per
decode step.

The cause is a single line, `deployments/aws-inf2/neuron_entrypoint.py:32`:

```python
os.environ.setdefault("NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU", "1")
```

Same engine, same weights, same device, same checkpoint, varying only this:

| | `=1` | unset |
|---|---|---|
| `generate_stream` warm, 20 tokens | **163.34 s** | **0.06 s** |
| steady per-token | 6.89 s | 0.002 s |
| host RSS delta per request | 13.71 GB | 0.00 GB |
| prefill_ms / decode_ms | 21,127.8 / 142,010.5 | 0.9 / 41.8 |

**But it cannot simply be removed.** End-to-end through the HTTP server:

| | output | latency |
|---|---|---|
| `=1` | `'Paris.'`, 5 completion tokens | ~126 s |
| unset | `''`, **0 completion tokens** | ~0.01 s |

Unset, generation emits EOS on the first sampled token, every time. So some
operation in the decode graph **computes the wrong thing on the NeuronCore** and
is only correct when dispatched to the host. The variable is a correctness
workaround, and the 2700x is what the workaround costs: every decode step ships
parameter-sized buffers through host memory, exhausting a 16 GB host and sending
it into swap.

This is the failure mode `CLAUDE.md` warns about, twice over. The slow path runs,
reports success, and is correct. The fast path runs, reports success, and is
wrong — and would have been recorded as a 2700x win if the token *content* had
not been checked. An earlier isolation run in this same session counted 20
tokens returned and never inspected them.

Two further costs were quantified along the way and remain real but secondary:
the W4A16 reference path (dequantize-then-matmul, forced on Neuron) at **27x per
call** against dense, and a **52x per-call cliff** above ~128 parameter buffers.

| | Device HBM | Host RSS |
|---|---|---|
| baseline (idle) | 6,560,986,854 | 910 MB |
| t=3 | 6,562,815,042 | 14.1 GB |
| t=6 | 6,564,642,934 | 3.7 GB |
| t=10 | 6,564,642,934 | 8.8 GB |
| t=15 | 6,564,642,934 | 1.8 GB |
| t=19 | 6,564,642,934 | 8.8 GB |

HBM moves **3.66 MB across the entire request** — exactly the int8 KV cache
appending 4–5 tokens. The parameters never move. Host RSS meanwhile cycles
between ~1 GB and ~15 GB, once per decode step.

On a 15.6 GB host that exhausts RAM and the kernel starts paging:

```
%Cpu(s): 10.0 us, 61.4 sy, 28.6 wa      kswapd0 100% CPU
vmstat during request:  so 72,252 → 115,954 KB/s     si 6,344 → 38,816 KB/s
free: 8,972,260 → 4,193,332 → 131,584 kB, then released back to 14,637,536
```

61% system time and 27–55% iowait mean almost none of the 220% CPU is doing
arithmetic. That is the whole latency budget.

## The mechanism: the compiled graph reads the weights dense

On Neuron `_CAPS.pallas` is `False`, so `set_w4a16_impl`
(`ports/gemma4/jax_e_model.py:243-253`) forces the implementation to
`"reference"` — the fused Pallas kernel lowers through Mosaic and neuronx-cc can
never accept it. The reference path is
`qat_w4a16_reference_linear_jax` (`:281`):

```python
def qat_w4a16_reference_linear_jax(x, packed_int4, scale, group_size=32):
    """Correctness reference: materialize the BF16 weight, then matmul."""
    w_dequant = qat_w4a16_unpack_dequant_jax(packed_int4, scale, group_size=group_size)
    return jnp.matmul(x, w_dequant.T)
```

Every linear layer materializes a dense BF16 copy of its weight before the
matmul. The fused kernel's own docstring (`:307`) puts its advantage at "~4x
lower" weight traffic, so the reference path moves roughly 4x more and
materializes large transients the fused path never creates.

So the weights are **stored** compressed — 6.56 GB resident, `quant_mode` stays
`w4a16`, `/health` reports `w4_int4` honestly — while the **compiled graph reads
them dense**. This is exactly the failure milestone 5 of `jax_neuron/README.md`
cautions against: "Loading a QAT checkpoint is not evidence that compressed
weights remain compressed in the compiled graph."

It happens silently. The loud fallback warning at `:274` ("Decode will read BF16
weights") only fires when `"auto"` or `"fused"` degrades at runtime. Here
`w4a16_impl` defaults to `"reference"` (`jax_engine.py:177`) and nothing
overrides it, so the guard at `:261` is false and control reaches the reference
path at `:278` without logging anything.

This cost is real but it is **not** the host churn, and it is not a memory
headroom problem — see the core-count experiment below, which rules both out.

### What the reference path costs

Measured on the same host, core 1, using the repo's own
`qat_w4a16_reference_linear_jax` over a 24-layer 2048x16384 stack with
device-resident weights. The only variable is packed-int4-plus-dequantize versus
a pre-materialized dense BF16 weight:

| variant | s/call | compile s | host delta |
|---|---|---|---|
| A: W4A16 reference, dequantize+matmul | **0.027** | 39.5 | 0.00 GB |
| B: dense BF16 matmul | **0.001** | 3.9 | 0.00 GB |

The reference path is **27x slower per call** and 10x slower to compile. That is
a large, real cost and it is unavoidable on Neuron today, since the fused
alternative is Pallas-only.

It is, however, **not** the source of the host churn: at this scale (0.40 GB
packed, 1.61 GB dense) neither variant moves host RSS at all. The possibility
that dense transients spill once they exceed available headroom is ruled out
separately by the core-count experiment below.

### The core-count experiment

`/etc/gemma4-inf2.env` was changed from `NEURON_RT_NUM_CORES=1` to `2` and the
service restarted (backup at `/etc/gemma4-inf2.env.bak`).

| | 1 core | 2 cores |
|---|---|---|
| devices visible | `[NeuronCore(id=0)]` | `[NeuronCore(id=0), NeuronCore(id=1)]` |
| parameter load | 263.0 s | **68.8 s** |
| `hbm_limit_bytes` | 17,179,869,184 | 17,179,869,184 |
| warm request, 5 tokens | 67.8 / 60.1 / 68.0 s | 76.0 / 60.3 s |
| host churn per step | yes | yes, unchanged |

Two corrections fall out of this, both to earlier claims in this report's own
drafting:

- **`hbm_limit_bytes` is per device, not aggregate.** It stays at 16 GiB with
  both cores visible. The chip's 32 GB is 16 GiB per core, so device 0 always
  had 16 GiB and `NEURON_RT_NUM_CORES=1` never halved the memory available to a
  single-device engine. It withheld a second device, not headroom.
- **Device memory headroom is not the cause of the host churn.** Free memory
  still collapses from 13.4 GB to 194 MB and recovers, once per decode step,
  with both cores visible.

What the change does buy is a **3.8x faster load** and a second device available
for future sharding. It does not move serving latency, which is what the
single-device engine predicts: a visible device it never targets cannot help.

### Buffer count: a per-call cliff, but not the churn

Same isolation harness, total bytes held at 2.05 GB, split across 8 / 128 / 512
arrays. Run with the service stopped so the test could hold a core.

| arrays | total GB | host delta | s/call | compile s |
|---|---|---|---|---|
| 8 | 2.05 | 0.00 GB | 0.008 | 10.4 |
| 128 | 2.05 | 0.00 GB | 0.008 | 10.6 |
| 512 | 2.05 | **0.00 GB** | **0.418** | 50.0 |

Buffer count does **not** produce host churn — zero delta at every split, so
that hypothesis is refuted too.

It does produce a **52x per-call cliff** between 128 and 512 buffers at
identical total bytes, with compile time up 5x. That is per-buffer dispatch
overhead, and it is a real cost the engine is exposed to: a Gemma 4 E2B
parameter pytree has several hundred leaves, putting it on the wrong side of the
cliff. At ~0.8 ms per buffer it accounts for a few hundred ms per decode step —
material, but a small fraction of the observed 11-13 s, so it is a contributing
factor and not the explanation.

### How it was found: the engine is fast, the process is slow

Seven synthetic isolation tests reproduced nothing. What finally separated the
variables was driving the real `JaxGemmaEngine` directly, outside the server
process, with `JAX_COMPILATION_CACHE_DIR` pointed at the same cache:

| call | delta GB | time |
|---|---|---|
| `_jit_prefill` warm | 0.00 | 0.02 s |
| `_decode_step` warm | 0.00 | 0.02 s |
| `generate_stream` warm, 20 tokens | 0.00 | **0.06 s** |
| `generate` warm, 20 tokens | 0.00 | **0.06 s** |

The engine, including its Python streaming loop, produces 20 tokens in 0.06 s
with no host churn — while the server takes 60–126 s for the same work. That
localised the cost to the *process*, not the engine, the model, or the device.
Calling the engine from a worker thread, a fresh thread per call, and a
`ThreadPoolExecutor` were all 0.042–0.045 s, ruling out the FastAPI threadpool.

The remaining difference was `neuron_entrypoint.py`, which sets three variables
the standalone runs never had. Re-running the identical benchmark under them
reproduced the fault exactly (163.34 s, 13.71 GB), and dropping only
`NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU` while keeping `JAX_PLATFORMS=neuron,cpu`
and `JAX_DEFAULT_PRNG_IMPL=rbg` restored 0.06 s and 0.00 GB. One variable.

### What was tested and refuted

The first hypothesis was that the Neuron plugin fails to hold a large jit
*argument* across executions, re-staging `self.params` per call
(`jax_engine.py:456`). A minimal isolation test on the same host and runtime --
identical shapes, math, device and call count, varying only argument-vs-closure
-- refutes it:

| variant | base GB | peak GB | delta GB |
|---|---|---|---|
| A: 2.05 GB pytree as jit argument | 0.22 | 0.22 | **0.00** |
| B: same pytree captured in closure | 2.22 | 2.22 | 0.00 |

Passing a large pytree as an argument re-stages nothing. The closure variant is
strictly worse: JAX captured the arrays as lowering constants (`A large amount
of constants were captured during lowering (2.05GB total)`) and parked 2.05 GB
in host RAM permanently. Rewriting the decode step to close over the parameters
would have made this worse, not better.

## Cross-check against a physical bound

6.56 GB of weights at ~11 s per decode step is **0.65 GB/s** effective. An
Inferentia2 chip's HBM is ~820 GB/s. That is roughly **0.1% of bandwidth** —
about three orders of magnitude, which is not a range any amount of kernel or
flag tuning occupies. Per the measurement rule in `CLAUDE.md`, the A/B between
configurations was internally consistent and still would have pointed at the
wrong thing; only the absolute bound identified the gap as structural.

## What was ruled out

Each of these was a live hypothesis, and each is disproven by measurement rather
than by argument:

- **Per-request recompilation.** `/opt/gemma4/cache/neuron` stayed at 4.0K and
  the JAX cache at 160 files / 3.9 MB across a full request; no `neuronx-cc`
  process, no compiler temp artifacts. Nothing is built per request.
- **Steady-state memory bloat.** `VmHWM` is 14.5 GB but idle `VmRSS` is 891 MB
  with 14.4 GB free. The 14.5 GB seen in `top` is the per-request peak, not
  residency.
- **The W4A16 checkpoint silently decompressing.** `dequant_at_load` defaults
  `False` and nothing sets it, so `dequantize_params_to_dense`
  (`jax_engine.py:324`) never runs. `ENGINE.quant_mode` stays `w4a16` and
  `/health`'s `"w4_int4"` is read from engine state, not from the CLI flag. The
  milestone 5 caution in `jax_neuron/README.md` still stands unaddressed, but it
  is not what happened here.
- **Device memory pressure.** `hbm_limit_bytes` is 17,179,869,184 (16 GiB) with
  6.56 GB used — 9.4 GB of headroom. The device is not short of memory.
- **Instance undersizing.** `inf2.8xlarge` carries the identical accelerator (1
  chip, 2 NeuronCores, 32 GB) and 8x the host RAM. It would mask this by
  absorbing the transient, at ~4.8x the price ($0.5609/hr vs $0.1177/hr spot),
  and it would leave the per-step host allocation in place. It is the right box
  for *building* and the wrong fix for *serving*.

## Collateral findings

1. **`NEURON_RT_NUM_CORES=1` withheld the second device.**
   `/etc/gemma4-inf2.env` set 1; `deployments/aws-inf2/user_data.sh:179` sets 2,
   but only on the probe invocation, which is why milestone 1 asserted
   `device_count == 2` while the service reported 1 device. It does **not**
   reduce the memory available to a single-device engine —
   `hbm_limit_bytes` is per device and stays 16 GiB either way. What it cost was
   the 3.8x load speedup and any future sharding. Now set to 2.

2. **`max_new_tokens` is a `static_argname`** on `_jit_prefill`
   (`jax_engine.py:353`) and is fed straight from the HTTP `max_tokens` field.
   Every distinct value triggers a fresh trace and compile: `max_tokens=16` cost
   167 s against 65 s for an already-seen 20. Beyond latency this is an
   availability hazard — a caller varying `max_tokens` forces unbounded
   multi-minute compiles. It should ride a bucket ladder, as the sequence
   dimension already does.

3. **The NEFF cache is configured but never populates.**
   `NEURON_COMPILE_CACHE_URL=/opt/gemma4/cache/neuron` is set and the directory
   is empty, while `NEURON_CC_FLAGS=--model-type=transformer` carries no
   `--cache_dir`. `NEURON_COMPILE_CACHE_URL` is a torch-neuronx variable;
   the neuronx-cc path wants `--cache_dir` in `NEURON_CC_FLAGS`. Unverified
   against the plugin docs, but it matches the observed empty cache. This
   matters more than it looks: a build-on-8xlarge / serve-on-1x split depends
   entirely on that artifact surviving the move.

4. **`persistent_compilation_cache=False`** for Neuron
   (`ports/gemma4/backend.py:134`) is contradicted by the deployment, which sets
   `JAX_COMPILATION_CACHE_DIR=/opt/gemma4/cache/jax` anyway — and 3.9 MB across
   160 entries has been written there. Either the capability or the environment
   is wrong.

5. **`onchip_sample_tpu_v6e_jax` is not jitted** (`jax_e_model.py:1631`). It is
   called per token through `_sample`, so its ops dispatch eagerly on an
   ahead-of-time backend. At `temperature=0` it early-returns a bare `argmax`
   over the 262,144-wide vocabulary. Small beside the other costs, but it is
   per-token eager work on an ahead-of-time backend.

Not a defect, though it reads like one: `pad_to_tpu_v6e_bucket` and
`TPUv6eHardwareProfile` on Neuron are justified at `jax_e_model.py:1624` —
Inferentia2's PE array is also 128x128 and neuronx-cc is likewise an
ahead-of-time static-shape compiler. The names mislead; the reuse is sound.

## What to do next

1. **Keep `NEURON_RT_NUM_CORES=2`** — done, worth it for the 3.8x load speedup,
   not a serving fix. Separately, **reduce the parameter leaf count**: the
   512-buffer cliff above is real and the engine sits past it. Consolidating
   per-layer tensors into stacked arrays would move it back under 128. Worth a
   few hundred ms per step, not seconds.
2. **Find the op that miscomputes on Neuron.** This is now the whole game: it is
   worth 2700x and it is the only thing standing between this port and a fast,
   correct server. `jax_neuron/parity.py` is the tool and it already exists —
   run it with `--subject-platform neuron` and
   `NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU` **unset**, which is the configuration
   that produces the fault. Milestone 3's parity run passed, so either it ran
   with a different configuration or the fault is in an op the parity prompts do
   not exercise; establishing which is the first question.
   Bisect from there: the symptom is EOS on the first sampled token, which
   points at the logits path or the sampler rather than the attention stack.
3. **Find a compressed-compute path for Neuron.** The port has no way to keep
   W4A16 weights compressed through the matmul: the fused kernel is
   Pallas/Mosaic and neuronx-cc cannot accept it, and the only alternative
   in-tree materializes dense BF16. Options worth evaluating, none yet tested: a
   neuronx-cc-expressible unpack fused into the matmul, compiler flags in that
   area, or accepting dense BF16 and sizing the deployment for it.
4. **Do not close over the parameters.** Measured worse -- see the refutation
   above.
5. **Re-measure before recording any tok/s.** The current number describes a
   dequantize-per-matmul graph with an unexplained per-step host allocation on
   top of it, not the engine's capability.
6. **Then** decide how to use the second core for compute. Tensor-parallel
   halves per-core weight traffic and footprint; a second replica doubles
   throughput but also doubles host pressure. Neither is worth designing until
   step 3 settles how much weight traffic there actually is.

Nothing here indicates the model is too large for an `inf2.xlarge`. The weights
fit compressed on one core with ~9 GB of device headroom to spare. The host, not
the accelerator, is where this is failing.

## Reproduction

```bash
# device vs host, sampled together during one request
curl -s http://127.0.0.1:8000/metrics | grep hbm_used_bytes | grep -v '^#'
awk '/VmRSS/ {print $2}' /proc/$(pgrep -f neuron_entrypoint)/status

# the paging itself
vmstat 2 12     # si/so climb, wa hits 55%, free collapses to ~131 MB
```

A request that appears to hang is usually not hung: `curl` defaults gave up at
120 s while the server completed the work and recorded it in `/metrics`. Check
`tpu_jax_requests_total` before concluding a failure.
