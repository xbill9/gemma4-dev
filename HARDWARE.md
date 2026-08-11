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
| v5p | yes | yes | **treat as no** | The v5p page lists bf16 **and FP8 at the same 459 TFLOPs**. Equal peaks buy no compute win even if the path is real, so no quantization conclusion changes. Unmeasured here — see the v5p note below |
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

v5p sits in its own table further down — it is a different product line (the "p" scale-up part, not the
"e" efficiency part), and folding it into a v5e-vs-v6e ratio column would invite exactly the wrong comparison.

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

## v5p — the scale-up part

Sourced from [the v5p page](https://docs.cloud.google.com/tpu/docs/v5p) and the live TPU API on 2026-08-10.
**Nothing below is measured** — no rig here has booted a v5p yet. `tpu-vllm-v5p1-2b` is the first to target one.

| Spec | v5p | vs v5e |
| :--- | ---: | ---: |
| HBM capacity | **95 GiB** | 5.9x |
| HBM bandwidth | 2,765 GBps | ~3.5x |
| Peak bf16 | 459 TFLOPs | 2.33x |
| TensorCores per chip | **2** | 2x |
| MXUs | 4 per TensorCore | |
| ICI bandwidth | 1,200 GBps bidirectional | 3.0x |
| Chips per host | 4 (`ct5p-hightpu-4t`) | |
| Smallest slice | **2x2x1 — 4 chips, 8 cores** | no 1-chip v5p *in the TPU API* — but see the CE-catalog caveat below |
| Pod | 8,960 chips; 6,144 max schedulable | |

**Two TensorCores per chip is the fact that leaks into everything else.** It is why the slice is named for
8 when it holds 4 chips, why the quota metric counts cores rather than chips, and why a chip-count intuition
carried over from v5e/v6e is wrong by 2x at every step.

**95 GiB per chip changes which constraints bind.** The v5e-1 `--gpu-memory-utilization` ceiling recorded
above is a 1.26 GiB absolute-headroom failure on a 16 GiB chip; on v5p the same fraction leaves tens of GiB,
so that result does not transfer. The mechanism — compiled program images live outside the knob's budget —
still does.

**"v5p is in three zones" and "v5p is in nine zones" are both true and measure different things.** Clarified
2026-08-10 after the two figures collided. The chip is installed far more widely than this project can reach:

| Set | Count | Zones |
| :--- | ---: | :--- |
| **Hardware** — `v5p-8` published by the TPU API | **≥9** | `europe-west1-b`, `-d`, `europe-west4-b`, `us-central1-a`, `us-east1-d`, `us-east5-a`, `-b`, `-c`, `us-south1-a` |
| **Quota** — a stated `TPUV5PPerProjectPerZoneForTPUAPI` value (128 cores) | 10 | `europe-west4-a`, `-b`, `-c`, `us-central1-a`, `-b`, `-c`, `-f`, `us-east1-b`, `-c`, `us-east5-a` |
| **Reachable** — the intersection | **3** | `us-central1-a`, `us-east5-a`, `europe-west4-b` |
| **Flex-start** | **1** | `us-east5-a` |

The two sets barely overlap: the project holds quota in seven zones with no v5p installed, and the chip sits
in six zones where the project has no stated quota. **Three is the operational number and nine is the
hardware number** — say which one you mean, because "v5p exists in three zones" is false about the chip and
true about this project.

```
gcloud compute tpus accelerator-types list --zone=<zone> --filter="type=v5p-8"
```

Filter on the **exact slice name**. `v5p-4` does not exist (see the two-TensorCore note above), so a check
written against the directory spelling matches nothing everywhere and reads as "no v5p anywhere" — the same
class of silent failure as a stale quota id.

**`europe-west1-c` is the one zone where the two catalogs disagree** — it publishes all four `ct5p-*` machine
types to Compute Engine while the TPU API reports no v5p accelerator type at all (verified twice, 2026-08-10;
the zone does carry `v5litepod-*`, so the API itself is answering). Record which catalog a zone finding came
from.

The Google [regions-zones page](https://docs.cloud.google.com/tpu/docs/regions-zones) names three v5p zones,
so it matches the *reachable* set by coincidence and understates the hardware set — **the same understatement
already recorded for v6e**, where the page names 8 against the API's 18. Read the API for both.

## Spelling: v5e is `v5litepod` to gcloud, v6e is just `v6e`, v5p counts cores

**This table is the Cloud TPU API path only** — the one all rigs but `tpu-vllm-v5p1-2b` still use. For the
Compute Engine spellings (machine types, image family, `compute.googleapis.com` quota ids, UPPERCASE
provisioning models) see the mapping table above; nothing in the rows below applies there.

| Context | v5e single chip | v6e single chip | v5p smallest slice (4 chips) |
| :--- | :--- | :--- | :--- |
| Prose, directory names | `v5e-1` / `v5e1` | `v6e-1` / `v6e1` | `v5p-4` / `v5p4` — no such rig exists; CE's 1-chip shape is `v5p1` |
| `ACCELERATOR_TYPE`, gcloud | **`v5litepod-1`** | `v6e-1` | **`v5p-8`** |
| Flex-start runtime version | `v2-alpha-tpuv5-lite` | `v2-alpha-tpuv6e` | **`v2-alpha-tpuv5`** |
| `gcloud ... --type` / `--topology` | `v5litepod` / `1x1` | `v6e` / `1x1` | `v5p` / `2x2x1` |
| TPU API quota id | `TPUV5sLitepodPerProjectPerZoneForTPUAPI` | `TPUV6EPerProjectPerZoneForTPUAPI` | `TPUV5PPerProjectPerZoneForTPUAPI` |
| Spot (preemptible) quota id | `TPUV5sPreemptibleLitepodPerProjectPerZoneForTPUAPI` | `TPUV6EPreemptiblePerProjectPerZoneForTPUAPI` | `TPUV5PPreemptiblePerProjectPerZoneForTPUAPI` |
| Quota unit | chips | chips | **cores** — a `v5p-8` draws 8 |
| Billing catalog family | `TpuV5e` / `DWS Defined Duration V5e` | `TpuV6e` | `TpuV5p` / `DWS Defined Duration V5p` |

**The v5e rename does not generalize.** v6e keeps its marketing name everywhere, and the quota ids drop
the `Litepod` that the v5e ids carry — so nothing in this table survives a chip retarget by analogy, and
a stale quota id fails quietly by matching no rows rather than erroring. v6e column verified against the
live TPU API and Cloud Quotas on 2026-08-07; v5p column on 2026-08-10.

**Three generations, three different spellings of the same idea, and no two agree.** v5e renames the family
(`v5litepod`), v6e renames nothing, v5p keeps the family name but changes what the *number* counts. v5p also
capitalises the generation in its quota ids (`TPUV5P…`) where v5e uses lowercase (`TPUV5s…`), and drops the
`Litepod` infix entirely. Deriving any of these by analogy produces a plausible string that matches nothing.

"v5e-1" is fine in prose and required in directory names; it is never valid in a gcloud argument. v6e is
the trap in reverse — the slot value and the gcloud value coincide at `v6e`, but the *directory* spelling
`v6e1` still is not a gcloud value. `NAMING.md` covers the directory form. Rigs spell `ACCELERATOR_TYPE`
inconsistently across siblings (`v5litepod-1` vs `v5e-1`) — read the rig's own env file, never copy a
sibling's.

## Provisioning: the Cloud TPU API is deprecated, and only some chips have a way out

Recorded 2026-08-10. This is a property of the **chip generation**, not of any rig — which generations have a
second control plane is decided by Google per generation, so it belongs here rather than in a rig's
`CLAUDE.md`.

[Introduction to Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu) and
[Cloud TPU resources in Compute Engine](https://docs.cloud.google.com/tpu/docs/tpus-in-compute-engine):

> The Cloud TPU API is no longer under active development. This includes the Google Cloud CLI for the Cloud
> TPU API and the Cloud Client Libraries for the Cloud TPU API.

Bug fixes and security updates only. **No sunset date is published**, so nothing breaks on a deadline — the
forcing function is forward-looking: *"New hardware generations, starting with TPU7x (Ironwood), are supported
only through Compute Engine or GKE."*

### Which generation has which control plane

| Generation | Cloud TPU API (`gcloud compute tpus …`) | Compute Engine (`gcloud compute instances …`) | GCE machine types **published** |
| :--- | :---: | :---: | :--- |
| v5e | yes | **no** (create refused) | `ct5lp-hightpu-1t`, `-4t`, `-8t` — **26 zones, and unusable from this path** |
| v5p | yes | yes | `ct5p-hightpu-1t-tpu`, `-2t-tpu`, `-4t` |
| v6e | yes | yes | `ct6e-standard-1t`, `-4t`, `-8t` (each also as `-…-tpu`) |
| v7 / TPU7x | **no** | yes | `tpu7x-standard-4t` |

**The last column is what the catalog publishes, not what you can create.** For v5e those are different
things, and an earlier version of this table wrote `—` in that cell — which was wrong twice over: the
machine types do exist, and their existing is not evidence of a path.

**v5e is the one generation with no exit, and this is now measured rather than inferred.** Probed
2026-08-11 with `instances create --machine-type=ct5lp-hightpu-1t` in `us-east1-b`, a zone that publishes
the type, deliberately chosen because `TPU_LITE_PODSLICE_V5` there is **0** so the create could not
succeed even if accepted:

```
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - This user agent is not allowed to use the machine type [ct5lp-hightpu-1t].
```

**That is not a quota error**, which is what a zero-quota zone would otherwise produce, and it is not a
does-not-exist error either. The type is refused to this caller. The control is the same command shape with
`ct6e-standard-1t`, which provisioned a live instance the same day.

`ct5lp-*` is in the catalog because **GKE node pools are created with exactly those strings** — that is
the consumer, and it explains the whole apparent contradiction: a v5e-covering image family
(`ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`), a `compute.googleapis.com` v5e quota metric
(`TPU-LITE-PODSLICE-V5-per-project-zone`), v5e machine types in `machine-types list`, and no
`instances create` path, all at once. The Cloud TPU API and GKE are themselves built on Compute Engine, so
CE-shaped artifacts for v5e are expected whether or not `instances create` will take them. **Do not reason
from catalog presence to creatability for any generation** — check it with a create into a zero-quota zone,
which costs nothing and settles it.

Note also that v5e publishes **no `-tpu` variant at all**, where v5p and v6e both do. Suggestive, not
load-bearing: the bare forms are the documented creatable ones for v6e, so bare-vs-`-tpu` does not predict
creatability and must not be used to argue this point.

So `tpu-vllm-v5e1-2b` (the live-demo rig), `tpu-jax-v5e1-2b`, `tpu-pytorch-v5e1-2b`,
`tpu-pytorch-v5e1-12b` and the two v5e encoding rigs stay on the deprecated API for as long as they exist.
v5p and v6e rigs can move; v7 rigs will have no choice.

### Mapping between the two

| Cloud TPU API | Compute Engine |
| :--- | :--- |
| `queued-resources create --provisioning-model=flex-start` | `instances create --provisioning-model=FLEX_START` |
| `--accelerator-type=v6e-1` / `v5p-8` | `--machine-type=ct6e-standard-1t` / `ct5p-hightpu-4t` |
| `--runtime-version=v2-alpha-tpuv6e` | `--image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --image-project=ubuntu-os-accelerator-images` |
| `--valid-until-duration` (bounds the *request*) | `--request-valid-for-duration` |
| `--max-run-duration` (flex-start only) | `--max-run-duration` + `--instance-termination-action=DELETE` |
| spot via `TPUV6EPreemptible…` quota | `--provisioning-model=SPOT` |
| QR → derived `<resource_id>-node` | the instance **is** the node — no indirection |

`--provisioning-model` on `instances create` takes `FLEX_START | RESERVATION_BOUND | SPOT | STANDARD`, and
`--request-valid-for-duration` is documented as the FLEX_START wait knob specifically. TPU7x adds an
allowlist on spot, flex-start, and calendar-mode reservations; **v5p and v6e are not gated**.

**Flex-start works on the CE path** despite [Request TPU Flex-start VMs](https://docs.cloud.google.com/tpu/docs/request-using-flex-start)
still saying *"You must use the queued resources API to use TPU Flex-start VMs."* That page sits inside the
deprecated API's own doc set and describes flex-start *within* that API; the Compute Engine
[provisioning models](https://docs.cloud.google.com/compute/docs/instances/provisioning-models) page lists
TPU v5p, v6e and TPU7x as flex-start machine series with no such requirement. Treat the TPU-docs sentence as
unreconciled, not as a constraint.

### The two catalogs are different evidence

The rig-level "three gates" framing (see `tpu-vllm-v6e1-2b/CLAUDE.md`) needs one amendment: **gate 1 now has
two sources that do not always agree.**

- For v6e they agree exactly — `ct6e-standard-1t` and `ct6e-standard-1t-tpu` are each listed in the **same 18
  zones** as `v6e-1` in the TPU API, against 37 zones reporting quota.
- For v5p they disagree in `europe-west1-c` (machine types yes, accelerator type no) — see the v5p table above.

So a migration that swaps `accelerator-types list` for `machine-types list` as its availability check is not a
like-for-like swap. Gate 2 (provisioning model accepted for that type in that zone) and gate 3 (capacity right
now) are unchanged, and neither is answered by either catalog.

**There are two machine-type families and what distinguishes them is undocumented.** `ct6e-standard-1t` and
`ct6e-standard-1t-tpu` are identical in vCPU, memory and zone coverage, differing only in
`guestAcceleratorType` (`ct6e` vs `tpu-v6e`). v5p publishes `-tpu` shapes at 1, 2 and 4 chips but a bare
shape only at 4.

**Do not read the bare form as "legacy" — an earlier revision of this file did, and it was wrong.** Google's
own Compute Engine quickstart creates `--machine-type=ct6e-standard-4t`, the *bare* form, and the CE
machine-types page documents `ct6e-standard-{1,4,8}t` and `ct5p-hightpu-4t`, all bare. The bare shapes are
the documented, directly-creatable ones; what the `-tpu` variants are for is not written down anywhere yet
found. Record the exact string in `tpu.env`; they are not known to be interchangeable.

### Can v5e use the Compute Engine path? **No — verified 2026-08-11**

**Settled by running it.** `gcloud compute instances create --machine-type=ct5lp-hightpu-1t` is rejected at
validation, before any resource exists:

```
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - This user agent is not allowed to use the machine type [ct5lp-hightpu-1t].
```

Run from `gce-vllm-v5e1-2b` in `us-west4-a`, `FLEX_START`, image family
`ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`. Nothing was created and nothing billed.

**Read the error text carefully — it is the useful part.** It is not "no capacity in this zone" and not a
quota denial. The API surface refuses the machine type outright, so the rejection says nothing about
`us-west4-a` and would repeat in all 26 zones. "User agent" here means the calling surface: `instances
create` is not an allowed consumer of `ct5lp`, while GKE — which requests the same machine type by name for
a node pool — is.

**The rejection is v5e-specific, not a project-wide GCE problem.** Same project, same command shape, v6e:
`gce-vllm-v6e1-2b  europe-west4-a  ct6e-standard-1t  RUNNING  FLEX_START`. So the Compute Engine path is
live here; v5e alone is excluded from it.

That confirms the documented exclusion, and retires the evidence table this section used to carry. All four
"for" signals had one innocent explanation and it was the right one — the Cloud TPU API and GKE are both
implemented *on* Compute Engine, so a v5e-covering image family, a v5e-shaped CE quota metric, `ct5lp` in
the machine-type catalog, and the OS-images page framing those images as Compute Engine images are all
exactly what you see whether or not `instances create` accepts `ct5lp`. None of them was evidence.

Two corollaries worth keeping:

- **Catalog presence is not creatability, and the insert-path validator does not distinguish them.** A free
  `instance-templates create` probe with `--machine-type=ct5lp-hightpu-1t` **succeeds** (a bogus
  `ct5lp-hightpu-99t` control is correctly rejected with `must provide existing machine type`). So the
  property validator only checks catalog existence. Do not read a template accepting a machine type as
  evidence that an instance will.
- Corrected 2026-08-10 and still true: the missing `ct5lp-*-tpu` shape was never the reason. The bare shapes
  are the creatable ones (above), so that variant's absence proves nothing either way. The real answer came
  from running the command.

**The six v5e rigs therefore have no migration path off the deprecated Cloud TPU API**, and stay `tpu-*`.
When the API is finally sunset, v5e capacity in this project is reachable only through GKE.

**A single-chip v5p exists in the CE catalog** — `ct5p-hightpu-1t-tpu`, 52 vCPU / 112 GB / 1 chip — where the
TPU API's floor is `v5p-8` at 4 chips. **`tpu-vllm-v5p1-2b` was built on it on 2026-08-10** and is the only
rig here off the deprecated control plane: `gcloud compute instances` throughout, at
`TENSOR_PARALLEL_SIZE=1`, which removed that rig's TP=4 KV-replication risk and 4x of its chip bill. Its
`CLAUDE.md` has the full mapping.

**Listed is still not provisionable** — nothing has attempted creation, so capacity is unproven and so are
the two settings the TPU API used to supply implicitly and CE makes you state: the **boot disk** (CE defaults
far too small for the vLLM image plus the model) and **`--scopes`** (TPU VMs defaulted to cloud-platform;
a plain instance does not, and without it the boot-time Secret Manager fetch fails 30 minutes into its retry
loop). Those are the likely first failures, not the machine type.

`NAMING.md` now records `v5p1` as a valid slot value — one the TPU API alone could not express.

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
