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
| v6e (Trillium) | yes | yes (2x bf16) | **no** | same pattern — no published fp8 peak |
| v7 (Ironwood / TPU7x) | yes | yes | **yes** | first TPU with fp8 in the MXU |

**This is the single most consequential fact for quantization work.** On v5e and v6e:

- **int8 is the only low-precision format with a compute win.** Both generations quote int8 at exactly
  2x their bf16 peak, the signature of a genuinely native MXU path.
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
| HBM bandwidth | 800 GiBps | 1,638 GBps | ~1.9x (see units note) |
| Peak bf16 | 197 TFLOPs | 918 TFLOPs | 4.66x |
| Peak Int8 | 393 TOPs | 1,836 TOPs | 4.67x |
| TensorCores | 1 (4 MXUs, 128x128) | — | |
| ICI bandwidth | 400 GBps bidirectional, 4 ports | — | |
| Single-chip machine type | `ct5lp-hightpu-1t` | `ct6e-standard-1t` | |
| On-demand list | ~$1.20 / chip-hr | ~$2.70 / chip-hr | 2.25x |

v5e figures verified against [Google's v5e page](https://docs.cloud.google.com/tpu/docs/v5e) on
2026-08-07. v6e FLOPS and bandwidth are carried from the same source set via
`tpu-vllm-v5e1-2b/v5e.md`; v7 fp8 support from Google's TPU7x documentation. Treat the v6e FLOPS and
bandwidth rows, and all of v7, as less directly verified than v5e. Two v6e rows are now firmer than
that: `ct6e-standard-1t` is confirmed by a real deployment (see the measured memory below), and the
on-demand rate was read straight from the Cloud Billing Catalog on 2026-08-07.

**Rates are per-region, and the provisioning-model ranking is not fixed.** The $2.70 v6e figure holds
across the US regions; europe-west4 quotes $2.97 and europe-west2 / asia-northeast1 / asia-south1 quote
$3.24. In us-east5, **flex-start ($1.35/chip-hr) is cheaper than spot ($1.4033)** — the reverse of the
v5e ordering, where spot undercut flex-start. Never assume spot is cheapest; read the catalog. The
`estimate_deployment_cost` tool in the vLLM rigs does exactly that and reports nothing rather than
guessing when no SKU matches.

**Units trap:** Google quotes v5e bandwidth in **GiBps** and v6e in **GBps**. Normalize before
comparing — the real ratio is ~1.9x, not a clean 2x. Don't quote "2x the bandwidth" as exact.

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

- **inf2** — AWS Inferentia2, served by `tpu-pytorch-inf2-2b`. Takes platform slot `tpu` per `NAMING.md`
  (a dedicated inference accelerator on a VM), with the part in the hardware slot. Specs and native
  format support are **not recorded here** — nobody has verified them. Don't infer them from the TPU
  rows above.
- **v6e-1** — several benchmark artifacts in this repo were measured on v6e and travelled with forks.
  A report's hardware short-name is the hardware *measured*, not the rig hosting the file.
