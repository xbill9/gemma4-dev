# Accelerator characteristics

Properties of the **silicon** — memory, bandwidth, and which numeric formats the matrix units actually
support. Independent of model and runtime, so this file is canonical for the monorepo and rigs should
read it rather than restating the numbers.

The companion files: `MODELS.md` for checkpoint properties, `QUANTIZATION.md` for what the serving stack
can actually do with the formats below, `NAMING.md` for how these values are spelled in directory names,
and each rig's `benchmarks/runs/` for what was measured on it.

## Native numeric format support — the table that decides quantization strategy

| Generation | bf16 | int8 | **fp8** | Notes |
| :--- | :---: | :---: | :---: | :--- |
| v5e | yes | yes (2x bf16) | **no** | Google publishes bf16 and Int8 peaks only |
| v6e (Trillium) | yes | yes (2x bf16) | **no** | same pattern — verified 2026-08-10, see below |
| v7 (Ironwood / TPU7x) | yes | yes | **yes** | first TPU with fp8 in the MXU |

**This is the single most consequential fact for quantization work.** On v5e and v6e:

- **int8 is the only low-precision format with a compute win.** Both generations quote int8 at exactly
  2x their bf16 peak, the signature of a genuinely native MXU path.
- **The v6e fp8 absence is now confirmed at the source, not inferred.** Google's v6e page lists exactly
  three peak-compute rows — `bf16: 918 TFLOPs`, `Int8: 1836 TOPs`, and a per-Pod bf16 figure — and **no
  fp8 row anywhere** (checked 2026-08-10). This was previously read off a second-hand spec set.
- **fp8 is storage-only.** Values widen back to bf16 before the matmul, so fp8 can buy footprint and
  bandwidth but never FLOPS. A benchmark showing no speedup from fp8 on these chips is the expected
  result, not a misconfiguration.
- **4-bit has no MXU path either.** int4 weights buy footprint and bandwidth, then unpack for compute.

On v7 that inverts: fp8 becomes a compute format and the reasoning above stops applying. **Do not carry
a v5e/v6e quantization conclusion forward to Ironwood without rechecking it.**

## Per-chip specifications

| Spec | v5e | v6e (Trillium) | Ratio |
| :--- | ---: | ---: | ---: |
| HBM capacity | 16 GB | 32 GB | 2.0x |
| HBM bandwidth | 800 GiBps | 1,638 GBps | **1.907x** — units differ, see note |
| Peak bf16 | 197 TFLOPs | 918 TFLOPs | 4.66x |
| Peak Int8 | 393 TOPs | 1,836 TOPs | 4.67x |
| TensorCores | 1 (4 MXUs, 128x128) | 1 (2 MXUs, 256x256 — but see below) | |
| ICI bandwidth | 400 GBps bidirectional, 4 ports | 800 GBps bidirectional, 4 ports | 2.0x |
| Single-chip machine type | `ct5lp-hightpu-1t` | `ct6e-standard-1t` | |
| Single-chip VM | — | 44 vCPU, 176 GB RAM | |
| Peak bf16 per Pod | — | 234.9 PFLOPs (256 chips) | |
| On-demand list | ~$1.20 / chip-hr | ~$2.70 / chip-hr | 2.25x |

**Both columns are now verified directly against Google**, v5e against
[the v5e page](https://docs.cloud.google.com/tpu/docs/v5e) on 2026-08-07 and v6e against
[the v6e page](https://docs.cloud.google.com/tpu/docs/v6e) on 2026-08-10. The v6e FLOPS and bandwidth
rows were previously carried second-hand via `tpu-vllm-v5e1-2b/v5e.md` and that caveat is now
withdrawn — every number above matched. MXU dimensions come from
[the system-architecture page](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm):
"An MXU is composed of either 256 x 256 (TPU v6e and TPU7x) or 128 x 128 (TPU versions prior to v6e)
multiply-accumulators in a systolic array." The on-demand rate was read from the Cloud Billing Catalog
on 2026-08-07; v7 fp8 support is from Google's TPU7x documentation and remains less directly verified.

**The v6e MXU *count* does not survive an arithmetic check — do not build on that row.** Google's v6e
page states "Each TensorCore has 2 matrix-multiply units (MXU), a vector unit, and a scalar unit."
Two 256x256 MXUs is 131,072 MACs, exactly 2x v5e's four 128x128 (65,536) — but the published peak is
**4.66x**, which would need a 2.33x clock increase on top. Four 256x256 MXUs closes it almost exactly:
262,144 MACs x 2 flops x 1.75 GHz = **917.5 TFLOPs** against a published 918. Google's own launch blog
says only that they "expanded the size of matrix multiply units (MXUs) and increased the clock speed",
naming no count. The **918 TFLOPs figure itself is sound** — it cross-checks against the same page's
234.9 PFLOPs-per-Pod row (918 x 256 = 235.0 PFLOPs). So treat peak compute as reliable and the MXU
count as unresolved; nothing in this monorepo depends on it.

**Rates are per-region, and the provisioning-model ranking is not fixed.** The $2.70 v6e figure holds
across the US regions; europe-west4 quotes $2.97 and europe-west2 / asia-northeast1 / asia-south1 quote
$3.24. In us-east5, **flex-start ($1.35/chip-hr) is cheaper than spot ($1.4033)** — the reverse of the
v5e ordering, where spot undercut flex-start. Never assume spot is cheapest; read the catalog. The
`estimate_deployment_cost` tool in the vLLM rigs does exactly that and reports nothing rather than
guessing when no SKU matches.

**Units trap — confirmed on both pages 2026-08-10.** Google quotes v5e bandwidth as **`800 GiBps`** and
v6e as **`1638 GBps`**, and the mismatch is real, not a transcription slip: the v5e page uses GiBps for
HBM while using GBps for ICI on the same table. Normalized, 800 GiBps = **858.99 GB/s**, so the true
ratio is **1.907x** — against a naive 1638/800 = 2.047x. Google's launch blog saying Trillium "doubled"
HBM bandwidth is the naive reading. **Don't quote "2x the bandwidth"**; the 7% matters when the
workload is bandwidth-bound, which decode is.

**Shape trap:** v6e is 2.25x the price for 2x memory, ~1.9x bandwidth, but **4.7x the raw FLOPS**. For a
decode-bound workload — which is bandwidth-bound, not FLOPS-bound — v5e is priced close to right. The
4.7x only pays for prefill-heavy or long-context work that actually burns the matrix units.

## Usable memory, measured

Nominal capacity is not what you get, on either generation.

### v5e-1 — 16 GB nominal

Measured on `v5litepod-1` with vLLM:

| | GiB |
| :--- | ---: |
| Total HBM | 15.75 |
| Usable at `gpu_memory_utilization=0.92` | **14.49** |

**14.49 GiB is the number to size against**, not 16. Weights plus KV must fit inside it. See `MODELS.md`
for per-model weight footprints — the short version is that only E2B fits at bf16.

### v6e-1 — 32 GB nominal

Measured on `ct6e-standard-1t` serving E2B under vLLM at 65,536 context. The allocation is recorded in
`tpu-vllm-v6e1-2b/gemma4-e2b-v6e1-demo.html`, cross-checked against the comparison table in
`tpu-vllm-v5e1-2b/v5e.md`:

| | GiB |
| :--- | ---: |
| Total HBM | 31.24 |
| E2B weights, resident | 8.97 |
| KV cache pool | **19.79** |
| Reserved headroom | 2.48 |

**The KV pool is what the generation buys.** 19.79 GiB against roughly 5.5 GiB left over on a v5e-1
after the same weights — about 3.6x — measured as 1,151,744 KV tokens against ~290K derived on v5e.
Read that beside the shape trap above: v6e costs 2.25x for ~1.9x the bandwidth, so it is not the way to
buy decode throughput. It is the way to buy context. What runs out on v5e is memory.

**Sizing caution:** this is one run at one `gpu_memory_utilization`, not a ceiling like the v5e row
above. Take 31.24 GiB total as the firm number and re-measure the split for another model or context
length.

#### 31.24 GiB and 33.55 GB are the same number

This repo quotes both. **XLA prints GiB and `memory_analysis()` returns bytes**, so `31.24G` in an OOM
message is 31.24 **GiB** = 33.55 GB — the figure the JAX-engine reports for the 12B/26B/31B all use.
Mixing the two produced a wrong ratio (3.70x instead of 3.98x) that disagreed with an independent
measurement until the units were fixed. Same trap as the GiBps/GBps row above, one level down.

Two further cautions when reading memory on this chip, both from the 31B work
(`tpu-jax-v6e1-31b-w4a16/docs/gemma4-31b-quirks.md` §C):

- **XLA compares *temporaries alone* against the whole chip** and does not subtract resident weights.
  `available HBM (31.24G)` in an error message is not headroom.
- **`peak_bytes_in_use` does not capture intra-call transients** — it returned exactly `bytes_in_use`
  on every sample. A resident-memory series cannot be used to infer peak; call `memory_analysis()` on
  the compiled executable, which predicted every pass/fail exactly.
- **An HLO instruction's output shape says nothing about allocation.** Ranking instructions by output
  size suggested the 31B's `f32[262144,5376]` lm_head (5.637 GB) was the temp floor; the decode step
  contains the same three instructions at a total temp of 0.146 GB. They are fused, not materialized.

## Spelling: v5e is `v5litepod` to gcloud, v6e is just `v6e`

| Context | v5e single chip | v6e single chip |
| :--- | :--- | :--- |
| Prose, directory names | `v5e-1` / `v5e1` | `v6e-1` / `v6e1` |
| `ACCELERATOR_TYPE`, gcloud | **`v5litepod-1`** | `v6e-1` |
| Flex-start runtime version | `v2-alpha-tpuv5-lite` | `v2-alpha-tpuv6e` |
| `gcloud ... --type` / `--topology` | `v5litepod` / `1x1` | `v6e` / `1x1` |
| TPU API quota id | `TPUV5sLitepodPerProjectPerZoneForTPUAPI` | `TPUV6EPerProjectPerZoneForTPUAPI` |
| Spot (preemptible) quota id | `TPUV5sPreemptibleLitepodPerProjectPerZoneForTPUAPI` | `TPUV6EPreemptiblePerProjectPerZoneForTPUAPI` |

**The v5e rename does not generalize.** v6e keeps its marketing name everywhere, and the quota ids drop
the `Litepod` that the v5e ids carry — so nothing in this table survives a chip retarget by analogy, and
a stale quota id fails quietly by matching no rows rather than erroring. v6e column verified against the
live TPU API and Cloud Quotas on 2026-08-07.

"v5e-1" is fine in prose and required in directory names; it is never valid in a gcloud argument. v6e is
the trap in reverse — the slot value and the gcloud value coincide at `v6e`, but the *directory* spelling
`v6e1` still is not a gcloud value. `NAMING.md` covers the directory form. Rigs spell `ACCELERATOR_TYPE`
inconsistently across siblings (`v5litepod-1` vs `v5e-1`) — read the rig's own env file, never copy a
sibling's.

## Other targets in this monorepo

- **v6e-1** — several benchmark artifacts in this repo were measured on v6e and travelled with forks.
  A report's hardware short-name is the hardware *measured*, not the rig hosting the file.

## inf2 — AWS Inferentia2

Served by `tpu-pytorch-inf2-2b`. Takes platform slot `tpu` per `NAMING.md` (a dedicated inference
accelerator on a VM), with the part in the hardware slot.

Measured on `inf2.xlarge`, 2026-07-31/08-01, with jax 0.6.2 / jax-neuronx 0.6.2.1.0.6446 /
neuronx-cc 2.24.8799.0 / Neuron runtime 2.31.24. Full workings in
`tpu-pytorch-inf2-2b/docs/neuron-jax-quirks.md`.

| Spec | Value | How known |
| :--- | ---: | :--- |
| HBM **per NeuronCore** | **16 GiB** | `hbm_limit_bytes` = 17,179,869,184, reported identically whether one core or two is visible |
| HBM per chip | 32 GB | 16 GiB × 2 cores |
| NeuronCores per `inf2.xlarge` | 2 | |
| HBM bandwidth | ~820 GB/s | published, **not calibrated here** — used only as an order-of-magnitude bound |

**The 16 GiB is per core, not a pool.** `NEURON_RT_NUM_CORES=1` therefore withholds a *device*, not
memory — a single-device engine always had its 16 GiB. Setting it to 2 cut parameter load from 263.0 s
to **68.8 s (3.8x)** and exposed the second core, but changed serving latency not at all: a visible
device that a single-device engine never targets cannot help.

A second process can claim the other core with `NEURON_RT_VISIBLE_CORES=1`, which is useful for
probing a live deployment. With `NEURON_RT_NUM_CORES=2` the server claims both and that stops working.

### Native format support: worse than the TPU rows above, not better

**There is no compressed-compute path at all.** `_CAPS.pallas` is `False`, so a fused W4A16 kernel
cannot lower — neuronx-cc will not accept Mosaic — and the W4A16 implementation is forced to
`"reference"`, which **materializes a dense BF16 copy of every weight before the matmul**:

| variant | s/call | compile s |
| :--- | ---: | ---: |
| W4A16 reference (dequantize + matmul) | **0.027** | 39.5 |
| dense BF16 matmul | **0.001** | 3.9 |

**27x per call, 10x compile.** So on this part 4-bit weights are *stored* compressed and *read* dense,
and it happens silently — the loud fallback warning only fires when `"auto"` or `"fused"` degrades at
runtime, and the default is already `"reference"`.

**Buffer donation also fails at runtime**, removing the lever that keeps a large argument in place on
TPU. Do not "fix" that by closing over the parameters instead: measured, that is strictly worse — JAX
captures them as lowering constants and parks the whole tree in host RAM permanently.

### Three failure modes with no TPU equivalent

1. **A too-large gather returns zeros, not an error.** Above roughly 4–5 GB in a single device tensor,
   a gather that "ran" can yield an all-zero result and execution continues. Zero logits make `argmax`
   return token 0 — the pad id, which is in the EOS set — so the server returns a clean `200 OK` with
   `finish_reason: "stop"` and **zero completion tokens**. Nothing in the response, the logs or
   `/metrics` indicates a fault. The same gather is exactly correct at every geometry that fits
   (0.81 GB `embed_tokens` included). **The gather primitive is fine; the allocation is not.** Check a
   norm, never a shape.
2. **Eager ops invoke the compiler at runtime.** A bare `jnp.full` reaches `RunNeuronCCImpl` — the
   first symptom was `FileNotFoundError: 'neuronx-cc'` raised from an array *construction*. Fixed
   shapes are cached and free; a **new** shape costs ~3.9 s per call. Anything shape-varying in a
   per-token path is a compile per token.
3. **Per-buffer dispatch has a cliff above ~128 buffers.** At a constant 2.05 GB total split across N
   arrays: 8 arrays → 0.008 s/call, 128 → 0.008, **512 → 0.418**. A 52x jump at identical total bytes.
   A Gemma 4 E2B parameter pytree has several hundred leaves and sits past the cliff.

**Do not carry TPU intuition here.** Sizes and dtypes matter in ways they do not on TPU, and the
platform announces itself with `Platform 'neuron' is experimental and not all JAX functionality may be
correctly supported!` — and then means it.
