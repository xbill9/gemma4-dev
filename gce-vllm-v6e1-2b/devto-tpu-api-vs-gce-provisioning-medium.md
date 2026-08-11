
# The Cloud TPU API is deprecated. Here is what porting off it actually costs.

Google has quietly stopped developing the Cloud TPU API. The notice is one sentence on [Introduction to Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu) and [Cloud TPU resources in Compute Engine](https://docs.cloud.google.com/tpu/docs/tpus-in-compute-engine): *"The Cloud TPU API is no longer under active development. This includes the Google Cloud CLI for the Cloud TPU API and the Cloud Client Libraries for the Cloud TPU API."* Bug fixes and security updates only. **No sunset date is published**, so nothing breaks on a deadline — but new generations from TPU7x (Ironwood) onward are Compute Engine or GKE only, so the API you are on today is the API your next chip will not support.

I run a monorepo of accelerator rigs that each serve Gemma 4 on one hardware shape. To find out what the migration actually costs, I forked one — a v6e-1 (Trillium) chip serving `gemma-4-E2B-it` under vLLM — and rebuilt its provisioning layer on `gcloud compute instances`. Same chip, same checkpoint, same serving flags, different control plane. The two rigs now sit side by side as `tpu-vllm-v6e1-2b` and `gce-vllm-v6e1-2b`.

The headline is not the flag mapping. It is that **your quota does not come with you**, that the thing you provision stops being visible to the API you used to provision it with, and that **the chip does not notice the difference at all** — I have now measured both sides, and where it serves they are indistinguishable. The entire cost of the migration lands before the model loads.

**Scope, stated up front:** the rig has since been provisioned on flex-start, served, and benchmarked against its Cloud TPU API twin — 10 throughput cells and an allocation check, same chip, same checkpoint, same engine build, same harness. So the comparison below is measured, not inferred. Three things remain unverified by a create and are flagged where they appear: whether **v5e** has this path at all, how flex-start behaves **under contention** (my capacity was granted instantly, so the DWS queue never engaged), and whether the fixed startup script **boots clean from scratch** — the one instance I have was repaired by hand.

## Which chips can even move

```
Generation  Cloud TPU API  Compute Engine  GCE machine types
----------  -------------  --------------  ---------------------------------
v5e         yes            no
v5p         yes            yes             ct5p-hightpu-1t-tpu, -2t-tpu, -4t
v6e         yes            yes             ct6e-standard-1t, -4t, -8t
v7 / TPU7x  no             yes
```

**v5e appears to have no exit, and this is the one place I could not settle the question.** Two Google pages exclude it independently. [TPU machines in the accelerator-optimized family](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines) enumerates the supported versions as "TPU7x, TPU v6e, TPU v5p" and documents `tpu7x-standard-4t`, `ct6e-standard-{1,4,8}t` and `ct5p-hightpu-4t`; `ct5lp` — the v5e family — appears nowhere on it. And the [TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e) page itself says: *"TPU v5e is supported using Google Kubernetes Engine and the Cloud TPU API."* Two surfaces named, and Compute Engine is not one of them.

The catalog says something less tidy. `ct5lp-hightpu-1t` exists, in 26 zones, with `guestAcceleratorCount: 1` and 24 vCPU / 48 GB, structurally identical to the v6e shape I do provision. The OS image family is literally named `ubuntu-accel-2204-amd64-tpu-**v5e**-v5p-v6e`. And `TPU-LITE-PODSLICE-V5-per-project-zone` is a `compute.googleapis.com` quota, not a `tpu.googleapis.com` one. That same v5e page even tabulates `ct5lp-hightpu-1t` at 24 vCPU / 48 GB — it documents the machine type while declining to offer a `gcloud compute instances` path to it.

I think all four have the same innocent explanation: **the Cloud TPU API and GKE are themselves implemented on Compute Engine.** Their TPU VMs *are* GCE instances, booting GCE images and drawing on GCE capacity. So a v5e-covering image, a v5e-shaped CE quota metric, and `ct5lp` machine types in the catalog are exactly what you would expect to see whether or not `gcloud compute instances create` will take `ct5lp` directly. The GKE half of that sentence is the sharpest version of the point: **a GKE node pool is created with `--machine-type=ct5lp-hightpu-1t`**, which is precisely why the string is documented on a page that offers no Compute Engine path. The machine type existing has a consumer that isn't me.

Which leaves the documented exclusion as the strongest evidence, and "no" as the working answer — but an unverified one. It is cheap to settle: one `create` against `ct5lp-hightpu-1t`, where a validation rejection is free and conclusive and an acceptance bills until deleted. I have now built the rig that would run it — a v5e twin of this one, `gce-vllm-v5e1-2b`, ported the same way and green on lint and tests — and stopped at the create. Six of my rigs are v5e, including the one I demo from, so I would rather be wrong about this than right.

One thing the v5e port did turn up, which matters if you are planning a chip-by-chip migration: **the quota asymmetry described later in this article is per-generation, not a property of the Compute Engine path.** v6e publishes only a preemptible per-zone id and has to fall back to a regional family-wide quota for everything else. v5 Lite PodSlice publishes *both* halves per-zone. So the zone-sweep logic I had to rewrite for v6e is not the logic v5e would want, and neither would be right for v5p. There is no "migrate to Compute Engine" change you write once.

## Flex-start works on Compute Engine, whatever the docs say

This one cost me an hour, so: [Request TPU Flex-start VMs](https://docs.cloud.google.com/tpu/docs/request-using-flex-start) states flatly that *"You must use the queued resources API to use TPU Flex-start VMs."* If that were true the migration would be dead on arrival, because flex-start is how you get scarce capacity at all.

It is not true. That page lives inside the deprecated API's own doc set and describes flex-start *within* that API. The Compute Engine [provisioning models](https://docs.cloud.google.com/compute/docs/instances/provisioning-models) page lists TPU v5p, v6e and TPU7x as flex-start machine series with no such requirement, `gcloud compute instances create` takes `FLEX_START` as a first-class enum value, and `--request-valid-for-duration` exists specifically as its wait knob. The only allowlist restriction named is on **TPU7x** — v5p and v6e are ungated.

Treat the TPU-docs sentence as unreconciled, not as a constraint.

## The flag mapping

```
Cloud TPU API                                            Compute Engine
-------------------------------------------------------  ------------------------------------------------------
queued-resources create --provisioning-model=flex-start  instances create --provisioning-model=FLEX_START
--accelerator-type=v6e-1                                 --machine-type=ct6e-standard-1t
--runtime-version=v2-alpha-tpuv6e                        --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e
--valid-until-duration (bounds the request)              --request-valid-for-duration
--max-run-duration (flex-start only)                     --max-run-duration (any model)
                                                         --instance-termination-action=DELETE
                                                         --scopes=cloud-platform (required)
                                                         --maintenance-policy=TERMINATE (required)
                                                         --provisioning-model=RESERVATION_BOUND
QR -> derived node <resource_id>-node                    the instance is the node
```

Four of those rows are real behaviour changes rather than renames.

**Values are SCREAMING_CASE.** `flex-start` becomes `FLEX_START`. This is the one mistake on the whole path that costs nothing, because gcloud validates the enum client-side before any round trip.

**`--max-run-duration` stops being flex-start-only.** On the TPU API, gcloud documents that flag as flex-start-only, which is why a spot or on-demand Queued Resource had no automatic stop and billed until somebody destroyed it. On Compute Engine every provisioning model can carry one, and pairing it with `--instance-termination-action=DELETE` makes a demo VM clean up after itself. That is a genuine operational improvement, not a lateral move.

**`--scopes=cloud-platform` is load-bearing and fails late.** My startup script fetches the Hugging Face token from Secret Manager at boot rather than baking it into instance metadata. Without the scope the VM boots fine, then spins through a 30-minute retry loop before giving up — so the symptom is half an hour of apparently-slow startup followed by what looks like a token problem. The Queued Resource path handed you a workable default scope set. `instances create` does not.

**`RESERVATION_BOUND` has no Queued Resource equivalent at all.** It consumes a calendar or dense-deployment reservation for that reservation's whole duration. It also has no Billing Catalog SKU, which matters if you have a cost tool: mine now returns "no rate, read the reservation" rather than falling through to the on-demand SKU as a nearest match. A confident wrong price is worse than no price.

## Your quota does not come with you

This is the finding I would want to know before starting.

The two control planes meter against **entirely separate quota pools**. On the project I tested, verified 2026-08-10:

```
TPU API:         TPUV6EPerProjectPerZoneForTPUAPI, us-east5 → 512 chips
Compute Engine:  TPUS-PER-TPU-FAMILY-per-project-region, (us-east5, CT6E) → no stated value
```

512 chips of headroom on one API and nothing on the other, for the same silicon in the same region. A migration that assumes quota carries over discovers this as a rejected create in a zone the old rig provisions in happily.

**My first draft of this section drew the wrong conclusion from that, and provisioning corrected it.** I had read it as "this project has no Compute Engine v6e quota." It has plenty — **CT6E = 32 in eight regions** (europe-west4, asia-east1, asia-northeast1, asia-south1, asia-southeast1, southamerica-east1, southamerica-west1, us-south1), and no stated value in exactly three: us-east1, us-west1, and us-east5.

The sharper statement, and the more useful one, is that the two pools are **disjoint and regionally misaligned**. The single zone my rig was configured for happened to be the one where the project's large TPU-API holding sits and its Compute Engine holding does not. Moving the rig's default zone to `europe-west4-a` — which publishes `ct6e-standard-1t` *and* has CT6E quota — was the entire fix, and it took longer to diagnose than to apply. **Check the intersection of machine-type availability and Compute Engine quota before you change any code**; neither list alone tells you where you can actually run.

The Compute Engine ids are also **asymmetrical in a way no analogy from the TPU API predicts**:

```
Model                  Quota id                                Scope
---------------------  --------------------------------------  --------------------------------------------------
on-demand, flex-start  TPUS-PER-TPU-FAMILY-per-project-region  regional, dimensioned by (region, tpu_family=CT6E)
spot                   PREEMPTIBLE-TPU-V6E-per-project-zone    per-zone
```

There is **no non-preemptible per-zone v6e id**. `TPU-V6E-per-project-zone` does not exist, although `TPU-V5P-per-project-zone` and `TPU-LITE-PODSLICE-V5-per-project-zone` both do. So on-demand and flex-start are governed by a regional, family-wide ceiling while spot gets a zonal one, and the two are requested from different quota pages.

That has a concrete consequence for tooling. My old rig's `find_tpu` swept zones by reading non-zero TPU API quota, which is zonal. A regional quota **cannot produce a zone list at all**, so the new rig sweeps by where the machine type is published instead — `gcloud compute machine-types list --filter=name=ct6e-standard-1t`. That turns out to be the better signal anyway: it is the Compute Engine analogue of `accelerator-types list`, and for v6e the two agree exactly at **18 zones each**, against **37 zones that merely report quota**.

One more trap: an **unset** family quota reads identically to a **zero** one through `gcloud quotas info`. Absence there is not evidence the hardware is missing. Check the machine-type list for that.

## What you provision becomes invisible to the API you left

A `ct6e-*` instance **does not appear in `gcloud compute tpus tpu-vm list`**. It is an ordinary Compute Engine instance that happens to carry a TPU.

This is the sharpest operational difference, and it is quiet. My old rig's endpoint discovery listed TPU VM nodes — a design I had already fixed once, because an earlier version listed only Queued Resources and reported a healthy hand-provisioned spot VM as "no TPU found". Point that same helper at a Compute Engine instance and it returns an empty list. No error, no warning; the rig simply believes nothing is running.

Three field-shape differences follow, and every one of them fails quietly rather than loudly:

- Status is **`status: RUNNING`**, not `state: READY`. Copy the old check across and it does not throw — it just sorts every healthy instance to the bottom of the candidate ranking, which you only notice when two are up at once.
- The external IP moves from `networkEndpoints[].accessConfig.externalIp` to **`networkInterfaces[].accessConfigs[].natIP`**.
- **`RUNNING` is a weaker claim than `ACTIVE` was.** A Queued Resource reached ACTIVE only once its node was up. An instance is RUNNING the moment the VM boots — long before the startup script has pulled a vLLM image or loaded a model. Any readiness check that treated ACTIVE as "ready" becomes wrong by several minutes.

I keep the two rigs' test fixtures deliberately different shapes for this reason. Sharing them would make discovery look tested while matching nothing.

**And I still got it wrong, in a way worth copying down.** I wrote a test called `test_a_ct6e_instance_is_not_a_tpu_vm_node` and considered the rule enforced. It covered the discovery helper. Meanwhile four other tools — the ones that manage the container, tail its logs, read systemd, and run benchmarks — were still shelling out to `gcloud compute tpus tpu-vm ssh`, and every one of them failed with a not-found against an instance that was plainly `RUNNING`. They are also, precisely, the tools you reach for *when something has already gone wrong*, so the failure surfaces at the worst moment.

The lesson generalises past TPUs: **"this codebase is off the old API" is a claim about every call site, and I had tested it at one.** `grep` for the old command before believing it. In my case three legitimate hits survive — deliberate `queued-resources list` calls, because sibling rigs on the old control plane compete for the same physical chips and I want to see them.

The same trap had a second nest I only found later: my Makefile's `status`, `endpoint`, `benchmark`, `query` and `destroy-tpu` targets were all still running `gcloud compute tpus tpu-vm describe`. Not one of them can see anything the migrated rig deploys.

## The image ships no Docker, and that is a control-plane difference too

Then I actually provisioned one, and the very first boot failed for a reason no flag mapping would have caught.

**`ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` has no `docker` on PATH at first boot.** The Cloud TPU API's runtime versions — `v2-alpha-tpuv6e` and friends — do. My startup script was inherited verbatim from the old rig, so it went straight to `docker pull` and died 100 seconds in:

```
+ sudo docker pull vllm/vllm-tpu:nightly
sudo: docker: command not found
...
ERROR: Failed to pull vLLM Docker image after multiple retries. Exiting.
```

**How that presents is the dangerous part.** The instance reported `status: RUNNING`, and kept reporting it. On this path `RUNNING` means the VM booted — not that the startup script succeeded, and certainly not that anything is serving. This is the "`RUNNING` is a weaker claim than `ACTIVE`" warning from the previous section arriving as a concrete failure: nothing short of reading `/var/log/vllm-startup.log` or curling `:8000` distinguishes a dead boot from a healthy one, and the readiness tool I had could not.

Fixing it took three edits, not one, because the fact bites at three layers: the startup script installs `docker.io` before pulling; a shell prelude now prefixes every Docker-dependent remote command, so the recovery tool you reach for after a failed boot does not fail the same way; and the copy-pasteable one-liner the agent emits carries the install too. If you are porting a startup script across, assume nothing the old image gave you for free is present.

## Then I measured both sides, and the chip cannot tell

With one instance up on each control plane, I ran the same harness against both: same chip, same checkpoint, same engine build (`vllm 0.26.1rc1.dev256+gf5bb701fa`), same serving flags, 10 throughput cells plus an allocation check.

**The allocation is not "within tolerance" — it is the same integers.**

```
                    Compute Engine  Cloud TPU API
------------------  --------------  -------------
KV cache tokens          1,151,744      1,151,744
weights resident          8.97 GiB       8.97 GiB
HBM available            19.77 GiB      19.77 GiB
derived block_size              64             64
KV cache tensors                15             15
```

The control plane does not change what the chip gives you. Nor does it change engine behaviour: the engine logs `Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e` on both paths and allocates bf16 on both, which is a known false signal in my notes and now confirmed to be a property of the engine rather than of how the hardware was requested. That is exactly the sort of claim a migration is tempted to re-attribute to the migration.

Throughput: the three control cells — smallest, run first, carrying almost no KV state — agree to **0.6%** (ratios 1.001, 1.003, 1.006). Larger cells sit 2–8% low on Compute Engine, and **I do not claim that gap is real.** Three back-to-back repeats gave a repeatability spread of 0.14% worst case — which naively makes 2% significant — but the same repeats came in 3–16% *above* the sweep's own numbers for identical cells, purely from a warm cache. So the benchmark has a noise floor near zero and a **history sensitivity up to 16%**, and the cells showing the largest deficits are exactly the ones my harness flags as prefix-cache contaminated. A 2% difference is not attributable when the dominant term is 16%.

**Verdict: no measured serving difference between the control planes.** Which is the reassuring half of this article — the migration is not a performance decision. Everything it costs, it costs before the model loads:

```
                    Cloud TPU API                        Compute Engine
------------------  -----------------------------------  --------------------------------------------------
serving throughput  baseline                             indistinguishable
HBM allocation      baseline                             identical integers
Docker              preinstalled in the runtime version  absent — first boot fails at the pull
quota pool          512 v6e chips in us-east5            CT6E=32 across eight regions, none in us-east5
readiness signal    QR ACTIVE implies the node is up     RUNNING means only that the VM booted
discovery           tpus tpu-vm list                     compute instances list — the other returns nothing
SSH                 tpus tpu-vm ssh                      compute ssh — the other cannot reach it
flex-start          queued-resources only                first-class, and reachable from a Makefile
auto-stop           flex-start only                      any model, via --max-run-duration
```

Every failure in the bottom half of that table was **quiet** on first contact. None of them threw. Flag-mapping — the part that looks like the work — is the part gcloud validates for you.

One number that surprised me, and a warning against reasoning by analogy: **flex-start is cheaper than spot on v6e.** $1.35/chip-hr against $1.4033 in us-east5, and $1.782 for spot in europe-west4. On v5e in us-west4 the ordering inverts and spot genuinely is cheapest at $0.5779 against flex-start's $0.60. Same catalog, same two provisioning models, opposite answers per chip and region. Read the rate; do not assume spot wins.

## What gets simpler

It is not all tax.

**The two-object lifecycle collapses.** A Queued Resource is a request that, if granted, produces a node whose name you did not choose — `<resource_id>-node`. The id you asked for was never the name you got, and my rig carried three separate helpers to reconcile them plus a `--force` flag on teardown, because an ACTIVE resource owns a node the API refuses to delete out from under it. On Compute Engine the instance *is* the node. `_resolve_node_id` went from three lookups to two and can no longer return a name the caller did not effectively ask for. `destroy` needs no `--force`.

**The create call means something stronger.** A QR create returns immediately and you poll a second resource through `ACCEPTED → PROVISIONING → ACTIVE` to learn whether you got hardware. `instances create` returns only once the instance exists, so a zero exit already means capacity was granted. My zone sweep lost an entire state machine.

**The Makefile and the agent stopped disagreeing.** My old rig had a standing gotcha: `make status` described a `tpu-vm` while the MCP tools managed a Queued Resource whose node had a different name, so the two could never see each other's work. Both now call `gcloud compute instances`. A VM made by `make deploy-tpu` is the same object the agent's `list_tpu_instances` returns. `make deploy-tpu-flex` also now exists, which it could not before — gcloud's `tpu-vm` path offers no flex-start at all.

**Compute Engine sells shapes the TPU API cannot.** The clearest case is v5p: the Cloud TPU API's smallest v5p slice is `v5p-8`, which is four chips, because v5p slice names count TensorCores and a v5p chip has two. Compute Engine publishes **`ct5p-hightpu-1t-tpu`** — 52 vCPU, 112 GB, **one chip**. For a model that fits in 95 GiB of HBM that is a 4× cut in chip count for the same work. I have a sibling rig that moved to exactly this shape and dropped from `TENSOR_PARALLEL_SIZE=4` to 1 as a result. Listed is not provisionable, and I have not created one — but it is not a shape the old API could express.

## Two smaller things worth knowing

**There are two machine-type families and nobody says which to use.** `ct6e-standard-1t` and `ct6e-standard-1t-tpu` have identical vCPU, memory and zone coverage, and differ only in `guestAcceleratorType` (`ct6e` versus `tpu-v6e`). v5p publishes `-tpu` shapes at 1, 2 and 4 chips but a bare shape only at 4. It is tempting to read the bare form as legacy and the `-tpu` form as the Compute Engine native one — don't: Google's own CE quickstart creates `--machine-type=ct6e-standard-4t`, and every shape on the CE machine-types page is bare. The bare ones are the documented, directly-creatable ones. What the `-tpu` variants are for, I could not find written down anywhere, so I treat the exact string as configuration rather than something to derive.

**The two catalogs can disagree about a zone.** For v6e they match exactly. For v5p, `europe-west1-c` publishes all four `ct5p-*` machine types to Compute Engine while the TPU API reports no v5p accelerator type there at all — verified twice, and the zone does answer for `v5litepod-*`, so the API is not simply erroring. One zone out of ten is a small disagreement, but it means "is the chip in this zone" now has two answers and you should record which one you asked.

## What I would tell someone starting this

**Read your quota first**, on `compute.googleapis.com`, before touching any code — and read it as an *intersection* with machine-type availability, because neither list alone tells you where you can run. That is the gate most likely to stop you, it is invisible from the TPU API side, and an unset quota looks exactly like a zero one. It cost me a zone move I could have made on day one.

**Then accept that discovery is a rewrite rather than a filter change.** That is where the silent failures live: nothing in it errors when it is wrong, it just reports an empty world. And grep for the old command afterwards — I tested the rule on one function and four other tools were quietly violating it, plus five Makefile targets I did not find until later.

**Budget for the image, not just the API.** The startup script you carry across was written against a runtime version that gave you things the Compute Engine image does not. Mine assumed Docker. Whatever yours assumes, the instance will report `RUNNING` while it fails.

**Do not expect a performance story.** I went looking for one and there is nothing there: identical allocation to the integer, control cells within 0.6%. The whole migration is an operations change. That is good news — it means you can plan it as a refactor rather than a re-qualification — but it also means the work is unglamorous and lives entirely in the failure modes.

**And if you are on v5e, this may not be available to you at all.** Two Google pages exclude it, the catalog is ambiguous for reasons that have an innocent explanation, and nobody — me included — has spent the ten free seconds a rejected `create` would take to settle it. Plan the chip migration and the API migration as one thing, because Google has arranged for them to be one thing.
