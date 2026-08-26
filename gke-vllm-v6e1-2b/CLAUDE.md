# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file MCP server (`server.py`, FastMCP) that serves Gemma 4 (`google/gemma-4-E2B-it`) with vLLM on one
Google Cloud TPU **v6e-1 (Trillium)** chip, provisioned as a **GKE node pool** — not a Cloud TPU API Queued
Resource, and not a Compute Engine instance either. Its tools shell out to `gcloud container` and `kubectl`,
and talk HTTP to the vLLM OpenAI-compatible endpoint behind a Service on port 8000.

**Proven end to end on 2026-08-25**: cluster + one-node `ct6e-standard-1t` pool in `europe-west4-a`, vLLM Ready
about ten minutes after apply, chat-completions answering in 0.40 s; then a second pool created, listed and
destroyed entirely through the MCP tools. `benchmarks/runs/2026-08-25-gke-first-provision-v6e1/` is the record.

**This rig is the third arm of a provisioning A/B.** `tpu-vllm-v6e1-2b` (Cloud TPU API) and
`gce-vllm-v6e1-2b` (Compute Engine) are the same chip, the same checkpoint, the same vLLM flags, reached
through different control planes. Only the provisioning path is supposed to vary — see **Keep the twins in
step** below before changing anything.

## Two ways in, one vocabulary

| | Provisions with | Use it when |
| :--- | :--- | :--- |
| `server.py` MCP tools | `gcloud container clusters/node-pools` + `kubectl` | Anything an agent drives |
| `gke/*.sh` + `make gke-*` | the same commands, in shell | You want to read or paste the exact commands |

They are not two paths. They render **the same manifest template** (`gke/vllm-gemma4.yaml.tmpl`), read the
same `tpu.env`, and use **the same four provisioning-model names** (`on-demand`, `spot`, `flex-start`,
`reservation-bound`). A `deploy_vllm` after a `make gke-deploy` reports `unchanged`, which is the check that
they have not drifted. The flag mapping lives in exactly one place per language:
`_node_pool_provisioning_flags` in `server.py`, the `case` block in `gke/gke-up.sh`.

**The Compute Engine instance path was removed from this rig on 2026-08-25.** `create_tpu_instance`,
`manage_vllm_docker`, `_ssh_command`, the `deploy-tpu*` Makefile targets and `startup_script_template.sh` are
gone, along with everything that shelled to `gcloud compute instances` or `tpus tpu-vm`. Keeping them would
have given one rig two control-plane vocabularies — the exact trap the fork's own Makefile comment
celebrated avoiding. That path still exists, in the rig named for it.

## Keep the twins in step

A change here that is **not** about provisioning — serving flags, `MAX_MODEL_LEN`, the benchmark harness, the
vLLM invocation — has to land in `tpu-vllm-v6e1-2b` and `gce-vllm-v6e1-2b` too, or the comparison stops being
one. `MAX_MODEL_LEN=32768`, `TENSOR_PARALLEL_SIZE=1` and `_vllm_serve_flags()` are identical across all three
on purpose, and `test_manifest_serves_the_same_flags_as_the_sibling_rigs` pins that the manifest cannot drift
away from `_vllm_serve_flags()` here.

What is *allowed* to differ: everything under "The control planes" below.

## Commands

```
make install       # pip install -r requirements.txt
make run           # python server.py (stdio MCP server)
make test          # python test_agent.py — unittest, NOT pytest
make lint          # ruff check . && ruff format --check . && mypy .
make format        # apply ruff formatting and autofixes
make tools         # regenerate GemmaTools.md from the @mcp.tool() decorators

make gke-preflight # kubectl, the GKE auth plugin, envsubst, live gcloud creds
make deploy        # = gke-up + gke-deploy: cluster, TPU node pool, then the model
make status        # = gke-status: node, pod, Service
make gke-logs      # follow the vLLM pod
make endpoint      # LoadBalancer external IP
make query PROMPT="..."
make benchmark
make destroy       # = gke-down-pool: deletes the TPU node pool, releases the chip
make destroy-cluster
```

`make lint` only *checks* formatting — `make format` is what writes it. Both `make lint` and `make test`
pass clean (47 tests); keep them that way. A `PostToolUse` hook in `.claude/settings.json` runs
`ruff format` on every `.py` file Claude edits.

**`make destroy` is not optional housekeeping here.** A Compute Engine instance can carry
`--max-run-duration` + `--instance-termination-action=DELETE` and stop itself; **a node pool has no
self-destruct at all**. Nothing in this rig stops the chip billing except deleting the pool, which is why
`destroy` is aliased to the narrow `gke-down-pool` rather than to deleting the cluster.

## The control planes

This is the rig's whole subject, so it gets the detail. Three planes now reach one v6e chip: the Cloud TPU
API (`tpu-vllm-v6e1-2b`), Compute Engine (`gce-vllm-v6e1-2b`) and GKE (here). Most of the flag-by-flag detail
below was written for the first two and is retained because the GKE path inherits the quota story wholesale —
**a GKE node pool spends the same `compute.googleapis.com` pools a Compute Engine instance does**, with the
same regional misalignment. What GKE changes is everything about how you ask, where the endpoint is, and what
stops the bill.

### The Cloud TPU API is deprecated

> The Cloud TPU API is no longer under active development. This includes the Google Cloud CLI for the Cloud
> TPU API and the Cloud Client Libraries for the Cloud TPU API.

Bug and security fixes only, **no published sunset date**. New generations from TPU7x (Ironwood) are Compute
Engine or GKE only. `@../HARDWARE.md` holds the per-generation table; the short version is that v5p, v6e and
TPU7x have a Compute Engine path and **v5e does not**, so the six v5e rigs can never migrate.

### Flag-by-flag

| Cloud TPU API (`tpu-vllm-v6e1-2b`) | Compute Engine (`gce-vllm-v6e1-2b`) | GKE (this rig) |
| :--- | :--- | :--- |
| `queued-resources create --provisioning-model=flex-start` | `instances create --provisioning-model=FLEX_START` | `node-pools create --flex-start` (**no `--provisioning-model` flag at all**) |
| `--accelerator-type=v6e-1` | `--machine-type=ct6e-standard-1t` | `--machine-type=ct6e-standard-1t` |
| `--runtime-version=v2-alpha-tpuv6e` | `--image-family=…` `--image-project=…` | — (the node image is the cluster's) |
| `--valid-until-duration` | `--request-valid-for-duration` | — (flex-start is an autoscaling shape: `--num-nodes=0 --total-max-nodes=N`) |
| `--max-run-duration` (**flex-start only**) | `--max-run-duration` (**any model**) + `--instance-termination-action=DELETE` | **no equivalent — nothing self-destructs** |
| — | `--scopes=cloud-platform` (**required**) | — (node service account is the cluster's) |
| — | `--maintenance-policy=TERMINATE` (**required**) | — (handled by the node pool) |
| spot via `TPUV6EPreemptible…` TPU-API quota | spot via `PREEMPTIBLE-TPU-V6E-per-project-region` | **same Compute Engine pools as the middle column** |
| QR → derived `<resource_id>-node` | the instance **is** the node | the node pool creates nodes you do not name |
| no equivalent | `--provisioning-model=RESERVATION_BOUND` | `--reservation-affinity=specific --reservation=…` |
| endpoint: node IP | endpoint: `natIP` | endpoint: **a Service**, not a machine |

**Three spellings for one idea, and this rig speaks the third.** `flex-start` vs `FLEX_START` vs a bare
`--flex-start`: same request, three APIs. `_node_pool_provisioning_flags()` is the single place the GKE
mapping lives — `test_every_model_maps_to_node_pool_flags` pins that no `--provisioning-model` ever leaks
into it. The rig's four model names (`on-demand`, `spot`, `flex-start`, `reservation-bound`) are shared with
the shell path on purpose.

The rest of this subsection describes the first two columns and is kept for the comparison:

**Values are SCREAMING_CASE here.** `flex-start` vs `FLEX_START` is the same request to a different API.
This is the one failure on this path that costs nothing — gcloud validates the enum client-side — but
`_provisioning_flags()` is the single place the mapping lives, and `test_every_model_maps_to_a_screaming_case_gcloud_value`
pins it. Don't add a second mapping.

**`--scopes=cloud-platform` is load-bearing and its absence fails late.** Without it the booted VM cannot
reach Secret Manager, and the startup script spins through its full 30-minute retry before giving up — so a
missing scope looks like a slow boot for half an hour and then like a token problem. The Queued Resource
path got a workable default scope set; `instances create` does not.

**`--max-run-duration` is not flex-start-only here.** On the TPU API it is, which is why a spot or on-demand
Queued Resource had no automatic stop and billed until destroyed. On Compute Engine every model can carry
one, and pairing it with `--instance-termination-action=DELETE` is what makes a demo VM clean up after
itself. Defaults: `MAX_RUN_DURATION=4h`, `REQUEST_VALID_FOR=2h`, both in `tpu.env`.

**`RESERVATION_BOUND` has no catalog rate**, and `_lookup_tpu_rate()` returns None with an explanation
rather than falling through to the on-demand SKU. Its cost is whatever its reservation was priced at. Keep
that property — a confident wrong price is worse than no price.

### Quota does not carry over — the trap this rig exists to document

**The two control planes meter against entirely different pools.** Verified 2026-08-10: this project holds
**512 v6e chips in us-east5 under `TPUV6EPerProjectPerZoneForTPUAPI`**, and **no stated value for family
`CT6E` in us-east5** under the Compute Engine quota this path actually consumes. Holding one buys nothing
on the other, and the failure mode is a create rejected in a zone the sibling rig provisions in happily.

**Do not read that as "this project has no Compute Engine v6e quota."** An earlier draft of this file did,
and it is false. The project holds **CT6E = 32 in eight regions** — europe-west4, asia-east1,
asia-northeast1, asia-south1, asia-southeast1, southamerica-east1, southamerica-west1, us-south1 — and no
stated value in us-east1, us-east5, us-west1. The two pools are **disjoint and regionally misaligned**,
which is a sharper and more useful statement than "one is empty": the zone this rig defaulted to was the
one zone where the project's large TPU-API holding sits and its Compute Engine holding does not.

`GOOGLE_CLOUD_ZONE` therefore moved to **`europe-west4-a`** on 2026-08-10. Both zones publish
`ct6e-standard-1t`, so machine-type availability is not the discriminator; regional quota is.

The Compute Engine ids are also asymmetrical, which no amount of analogy from the TPU API predicts:

| Model | Quota id | Scope |
| :--- | :--- | :--- |
| **flex-start** | `PREEMPTIBLE-TPU-V6E-per-project-region`, then the family quota as **fallback** | preemptible pool first |
| spot | `PREEMPTIBLE-TPU-V6E-per-project-region` | preemptible pool |
| on-demand (`STANDARD`) | `TPUS-PER-TPU-FAMILY-per-project-region` | **regional**, dimensioned by `(region, tpu_family=CT6E)` |

**Flex-start spends the PREEMPTIBLE quota, not the family quota** — corrected 2026-08-11, an
earlier version of this file had it the other way round and the error propagated into `tpu.env`
and a draft article. Google's
[provisioning models](https://docs.cloud.google.com/compute/docs/instances/provisioning-models)
page states it in full, and the second sentence is the one people miss:

> When you create a Flex-start VM, preemptible quota is consumed. If your project lacks
> preemptible quota, then standard quota is consumed.

This is counterintuitive because flex-start is not preemptible in behaviour — once granted it
runs uninterrupted for up to seven days — so nothing in the flag names hints at it. **This rig
defaults to flex-start, so `PREEMPTIBLE-TPU-V6E-per-project-region` is what gates it in
practice — but a region is usable if EITHER pool has room, so never write one off on a single
listing.** This file has had the mapping wrong twice: first grouping flex-start with on-demand,
then claiming it spends preemptible and *not* the family quota. Both were wrong.

Do not try to determine this experimentally; a day was spent on that before the docs were found.
A `FLEX_START` create short of quota is **accepted and queues** rather than erroring, and a
capacity stockout produces the identical `PENDING`, so the create is not a quota probe.

**The two ids carry opposite defaults**, which is what makes misreading them costly: a region
absent from the family quota inherits **0**, a region absent from the preemptible quota inherits
**1536**. Reading only the family listing writes off regions that have ample flex-start headroom.
The per-zone spelling `PREEMPTIBLE-TPU-V6E-per-project-zone` exists but holds no entries in this
project — the per-region one has the values.

**Quota held on the Compute Engine path, verified 2026-08-11** (regions publishing `ct6e-standard-1t`):

| region | flex-start / spot | on-demand |
| :--- | ---: | ---: |
| europe-west4 + 7 others | 1536 | 32 |
| us-east1 | 1536 | 32 |
| us-central1, us-west1 | 1536 | 0 |
| us-east5 | **32** | 32 |
| us-east4 | 0 | 0 |

**Do not re-request the denied quotas, and do not try a smaller number.** All five denials were
retried on 2026-08-11 and denied again, identically and within seconds: us-central1 family,
us-west1 family, us-east4 family, us-east4 preemptible, and us-east5 preemptible 32 -> 1536.
Four were then retried at **8 chips instead of 32** and denied again, with no partial grant.
Three sizes have been tested — 8, 32 and 1536 — and the outcome tracked the REGION every time,
never the number. These are stable refusals, not transient ones, and the ask size is not the
variable. Consistent with the denials being capacity-driven: a region with no chips to give has
none to give at 8 either. In particular us-east5's 32 looks like an error to correct — every other live
region has 1536 — and it is not. That region gives 32 whatever you ask for, and 32 is 32x what this single-chip
rig uses.

**QUOTA IS A CEILING, NOT AN ALLOCATION — capacity is the real constraint.** Every zone probed
on 2026-08-11 held 1536 chips of preemptible quota and four of five had no hardware:

| zone | quota | spot create |
| :--- | ---: | :--- |
| europe-west4-a | 1536 | provisioned |
| us-central1-a | 1536 | `reason: stockout` |
| us-central1-b | 1536 | provisioned, then stocked out a minute later |
| us-central1-c | 1536 | `reason: stockout` |
| us-west1-c | 1536 | `reason: stockout` |

`us-central1-b` is the one to remember: a spot instance came up, was deleted, and a flex-start
request a minute later was refused for stockout. **Availability moves faster than a controlled
test can track**, so do not plan around a zone having capacity because it had some earlier.
A SPOT create is the cheap probe — it fails fast with `reason: stockout` where flex-start queues.

**Grants are a property of the region, not of the number requested.** The identical 0 → 32 ask was
approved in us-east5 and us-east1 and denied in us-central1, us-west1 and us-east4. Eleven
requests total: 3 approved, 5 denied, 3 refused at submission for being decreases
(`FAILED_PRECONDITION: ... decreases effective quota unsafely` — you cannot ask for less than you
hold, and because the two ids have different defaults, one blanket number is wrong about half the
time). Every decision returned in seconds. Use `../request-quota.sh`, which reads the current
value per metric before asking.

There is **no non-preemptible per-zone v6e id at all** — `TPU-V6E-per-project-zone` does not exist, though
`TPU-V5P-per-project-zone` and `TPU-LITE-PODSLICE-V5-per-project-zone` do. An unset family quota reads
identically to a zero one through `quotas info`, so absence here is not evidence the hardware is missing;
check `machine-types list` for that.

**This is why `find_tpu` sweeps zones by machine type, not by quota.** A regional quota cannot produce a
zone list, and it is unset in us-east5 besides. `_zones_with_machine_type()` is the Compute Engine analogue
of the TPU API's `accelerator-types list`, and for v6e the two agree exactly — 18 zones each, 2026-08-10.
For v5p they disagree in one zone; see `@../HARDWARE.md`.

### Discovery is a Service, and the wrong answer succeeds

**The endpoint is not on a machine.** The two sibling rigs read an IP off the thing they created — a Queued
Resource's node, or an instance's `networkInterfaces[].accessConfigs[].natIP`. Neither is where the model is
listening here: vLLM runs in a Pod behind a Service, and only the Service knows the address.

**This failure is quieter than the twin's was.** A GKE node *does* appear in `gcloud compute instances list`
— it is an ordinary Compute Engine VM that happens to carry a TPU — so the twin's discovery call does not
error here, it succeeds and returns the wrong object. `test_discovery_never_reads_a_node_ip` pins that
discovery only ever talks to `kubectl`.

Three consequences:

- **`_service_endpoint()` returns None rather than an unreachable address.** A ClusterIP Service, or a
  LoadBalancer whose IP has not been assigned yet, is not reachable from here; handing the caller
  `http://10.x.x.x:8000` would hang instead of failing.
- **Ready is weaker than RUNNING was, which was already weaker than ACTIVE.** A Queued Resource reached
  ACTIVE with a node up. An instance was RUNNING when the VM booted. A GKE **node** is Ready the moment the
  kubelet registers — before the pod is scheduled, before the image is pulled, before the weights load, before
  XLA compiles. That was ten minutes on the first real deployment. Use `verify_model_health` for readiness.
- **A Ready TPU node can advertise zero chips.** Observed 2026-08-25 on a freshly created pool:
  `google.com/tpu: 0` on a node whose status was Ready, because the device plugin had not finished
  registering. A pod scheduled in that window fails with `Insufficient google.com/tpu`, which reads like a
  quota problem and is not one. `get_system_status` prints the allocatable count for exactly this reason.

### Every kubectl call is pinned to this rig's context

`_kubectl()` builds `kubectl --context=gke_<project>_<location>_<cluster>` and nothing here ever relies on
the current context. The current context is machine-global state shared with every cluster this workstation
has ever fetched credentials for, so an unpinned `kubectl delete` is one stale context away from acting on
someone else's cluster. `test_kubectl_is_always_pinned_to_this_rigs_context` pins it.

**`destroy_gke_cluster` only ever deletes the cluster it was named.** It does not enumerate the project and
it has no `--all`. The sibling's `manage_queued_resource` deletes every QR in the zone that is not the named
primary; nothing here sweeps.

### What is unchanged

The vLLM flags, the Secret Manager source for the HF token, and the billing-catalog lookup are identical to
both sibling rigs, deliberately. The quota ids are identical to the Compute Engine twin's, because GKE spends
the same pools.

## Gotchas

**`--tpu-topology` is a multi-host flag, and `1x1` is not a small version of it.** This rig's first node-pool
create passed `--tpu-topology=1x1` and was refused:

```
400: TPU topology can't be specified with single-host TPU slice pool;
     please remove the tpu_topology from the node pool creation request
```

`ct6e-standard-1t` at one node **is** the slice. What makes it a quiet trap is that GKE then labels the node
`cloud.google.com/gke-tpu-topology=1x1` **anyway** — so the value is real as a pod selector and refused as a
create flag. `tpu.env` keeps them apart: `TPU_TOPOLOGY` is the selector, `GKE_TPU_TOPOLOGY` (unset) is the
multi-host create flag. `test_single_host_pool_sends_no_tpu_topology` pins it.

**Deleting the node does not delete anything — the MIG puts it straight back.** A node pool is implemented
as a **managed instance group**, and the node is an ordinary Compute Engine VM inside it. Verified on this
cluster 2026-08-25:

```
gcloud compute instances list          gke-tpu-cfc04f31-8h14              ct6e-standard-1t  RUNNING
gcloud compute instance-groups managed list
                                       gke-gke-vllm-v6e1-2b-tpu-v6e-1-cfc04f31-grp   size 1
```

So `gcloud compute instances delete gke-tpu-…` looks like teardown, succeeds, and the MIG recreates the node
within minutes — still billing, with a new name. **The pool is the unit of teardown**: `destroy_tpu_node_pool`
(or `make destroy`). The same applies to fixing a sick node by hand — a node is cattle here, replaced on
upgrade and repair, so anything you change on its disk is gone at the next replacement. On the Compute
Engine twin the VM *is* the thing you tend and its startup script is where the deployment lives; here the
pod spec is.

This is also the answer to "does the cluster make its own v6e-1?" — yes, and what it makes is a GCE VM with
the same `ct6e-standard-1t` machine type the twin passes to `instances create`. Same silicon, same
attachment; what changes is who calls create and who owns the lifecycle.

**Nothing stops the bill on a timer.** See `make destroy` above. This is the single most expensive difference
from the Compute Engine twin, and `test_a_node_pool_has_no_self_destruct` exists to stop someone copying the
twin's `--max-run-duration` in and believing it.

**`gcloud secrets versions access` needs `--project`.** Without it gcloud uses the machine's *default*
project, which on this workstation is an expired qwiklabs lab, and Secret Manager answers `Permission denied
on resource project qwiklabs-...` — naming a project that appears nowhere in this rig. Inherited from the
fork; the shell path always passed it, so only the MCP tool hit it. Fixed 2026-08-25, pinned by
`test_secret_access_names_the_project`.

**Never write a dollar sign in a comment in `gke/vllm-gemma4.yaml.tmpl`.** The template is rendered two ways
— envsubst from the shell path, `string.Template` from `deploy_vllm` — and `string.Template` raises on
anything that is not a valid placeholder. A `${...}` in a header comment rendered fine under envsubst and
broke the MCP path. `test_the_template_carries_no_dollar_in_prose` pins it.

**The pod needs all three of: both node-selector labels, the `google.com/tpu: 1` limit, and the
`google.com/tpu` toleration.** Drop a selector and it schedules onto the `e2-standard-4` system node and
fails there, which reads as a vLLM problem rather than a placement one. Drop the limit and the device plugin
never attaches the chip. The taint is applied by GKE automatically.

**The LoadBalancer is public and unauthenticated**, matching the twin's `natIP:8000` so that a benchmark is
comparable across the three rigs. Set `GKE_SERVICE_TYPE=ClusterIP` and use `make gke-port-forward` if that is
not wanted.

**The HF token never goes through argv.** `kubectl create secret --from-literal=token=...` puts it in the
process table for every user on the machine; `deploy_vllm` writes a base64 Secret manifest to a 0600 temp
file instead. `test_token_never_reaches_the_process_table` pins it.

**Three things bill, not one:** the v6e chip, the `e2-standard-4` system node, and the cluster management
fee. `make destroy` stops the expensive one and leaves the other two, so a redeploy is one
`create_tpu_node_pool` away.

**`find_tpu` is deliberately reluctant.** A zone sweep is free on both sibling paths; here it means having a
cluster in each candidate zone, which is ten minutes and a standing fee per zone. So
`create_cluster_if_missing` defaults to False and the sweep reports non-cluster zones as candidates rather
than building in them. When it does build one and the node pool then fails, it deletes that cluster again.

**`tpu_zones_status.md` is mutable state, not documentation.** `find_tpu` rewrites it in place. Never seed it
from a sibling rig's file — a zone that refuses a Queued Resource or an instance is not evidence about a node
pool.

**Raw `/v1/completions` returns an empty completion on `-it` models.** `benchmarking_suite.py` uses it, so an
empty result there is expected. `server.py` and `make query` use `/v1/chat/completions` — keep new code on the
chat endpoint.

**`--tensor-parallel-size` is 1.** v6e-1 is a single chip, and `@../MODELS.md` notes E2B has
`num_key_value_heads=1`, which cannot shard — more chips would multiply its KV cost, not divide it.

## Names derive from the rig directory

`RIG_NAME` is `os.path.basename(...)` of this directory. `GKE_CLUSTER_NAME` defaults to it and names the
cluster; `GKE_NODE_POOL` defaults to `tpu-v6e-1`; `MCP_SERVER_NAME` defaults to `RIG_NAME` and names the
FastMCP server. All resolve under `gke-vllm-v6e1-2b`.

`INSTANCE_NAME` survives only as the tpu.env-compatible default the cluster name derives from — **nothing
here provisions an instance**, and the `RESOURCE_ID` alias the forked tool signatures used is gone with them.

The MCP server name has to match the key the client registers it under, because that key prefixes every
tool: `mcp__gke-vllm-v6e1-2b__find_tpu`. That is what distinguishes this rig's tools from the TPU-API twin's
`mcp__tpu-vllm-v6e1-2b__…` and the Compute Engine twin's `mcp__gce-vllm-v6e1-2b__…`; with all three loaded the
prefix is the *only* thing that does.

`load_dotenv` runs *before* `FastMCP(...)` is constructed, because `MCP_SERVER_NAME` set in `tpu.env` would
otherwise arrive too late to name the server. Don't move the FastMCP construction back above the dotenv
block.

**Renaming the rig directory orphans anything already provisioned** — pin `GKE_CLUSTER_NAME` in `tpu.env`
before renaming, or destroy first. A live cluster is the expensive thing to orphan: it keeps billing and no
tool in the renamed rig will look for it.

## Silicon facts live at the monorepo root

`@../HARDWARE.md` is canonical for v6e's memory, bandwidth, native numeric formats, gcloud spelling, and the
control-plane table; `@../MODELS.md` for E2B's layer structure, KV cost per token, and weight footprint.
Read them rather than re-deriving, and correct them there rather than restating a number here.

Two of their facts decide things in this rig:

- **v6e has no native fp8** — int8 is the only low-precision format with a compute win. fp8 buys footprint
  and bandwidth, never FLOPS. v7/Ironwood is the first TPU that changes this.
- **32 GB HBM, ~19.8 GiB free for KV** after E2B's weights. That is what this chip buys: v6e costs 2.25x for
  ~1.9x the bandwidth, so it does not buy decode throughput — it buys context.

## Benchmarks

`benchmarks/runs/2026-08-25-gke-first-provision-v6e1/` records the first provision: what was created, the
phase timings (~28 minutes from nothing to first token), and the four things that were wrong on the way. It
carries **no report JSON on purpose** — nothing was swept, so `benchmarks/rollup.py` should not count it.

`benchmarks/reports/2026-08-25-gemma4-e2b-v6e1.json` is the rig's **first real measurement** (schema 1.1):
a 1/8/32 concurrency sweep at 1024 in / 128 out, 199 → 1180 → 1508 output tok/s, with the cold-start
timeline and the live chip rate. It is one run per point — a first data point, not a characterisation.

Nothing else belongs in `reports/` until it is measured here. The fork's inherited artifacts were dropped
rather than carried, because none was measured on this rig and `rollup.py` globs `*/benchmarks/` — leaving
them would credit this rig with results it never produced.

**The fork's `benchmarks/runs/2026-08-10-gce-flex-v6e1/` was deleted before the rig's first commit.** It is
the `gce-vllm-v6e1-2b` first-provision record — a Compute Engine run, byte-identical to the copy already
tracked in that rig — and `rollup.py` globs `*/benchmarks/`, so committing it here would have credited this
rig with a measurement from another control plane. That is the exact failure `rollup.py` exists to expose.
If a fork ever lands here again, check `benchmarks/` before the first commit.

`benchmarks/serving-report.schema.json`, `benchmarks/README.md`, and `benchmarks/INDEX.md` are generated or
synced from the monorepo root. Don't hand-edit them here.

**A sweep here is only comparable to the siblings' if the serving config matches.** Check `tpu.env` against
`tpu-vllm-v6e1-2b/tpu.env` and `gce-vllm-v6e1-2b/tpu.env` before believing a difference is about the
provisioning path — and note the benchmark now runs *inside the pod* (`kubectl exec`), where the twins run it
in a second container over SSH.

## Auth and env

Requires `gcloud auth login`, `gcloud auth application-default login` (ADC), **and two things the sibling
rigs do not need**: `kubectl` and `gke-gcloud-auth-plugin`, both from the Google Cloud apt repo
(`sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin`). `make gke-preflight` checks all
four plus `envsubst` and fails with the install command rather than halfway through a deploy.

`set_env.sh` must be **sourced**, not executed. `init.sh` blocks on `read` in its error path — don't run it
non-interactively.

`tpu.env` is the single source of truth and is committed. A real environment variable always wins over it
(`load_dotenv` doesn't overwrite, `mcp-run.sh` exports only unset keys, the Makefile uses `?=`), so
`make gke-status GKE_LOCATION=...` works for a one-off.

**Some of `tpu.env` is inert here.** `IMAGE_FAMILY`, `IMAGE_PROJECT`, `BOOT_DISK_SIZE_GB`,
`MAX_RUN_DURATION`, `REQUEST_VALID_FOR` and `PROVISIONING_MODEL` describe a Compute Engine instance and
nothing on this path reads them; they are kept so the three rigs' env files stay diffable. `MACHINE_TYPE`
is *not* inert — a node pool takes the same string.

## Tests

`test_agent.py` mocks the whole `mcp` module and the Google Cloud clients before importing `server`. Keep
unit tests offline. Because `mcp` is a `MagicMock`, anything calling `mcp.list_tools()` needs an explicit
`AsyncMock` patch — see `test_get_help`.

The discovery fixtures are Kubernetes objects — a Service with `status.loadBalancer.ingress[]` and a Pod
with `containerStatuses[].ready`. Copying a sibling's instance-shaped fixture over would make discovery look
tested while matching nothing.

**`test_no_tool_shells_to_instances_create_or_tpu_vm` reads the module source, not one call path.** The
Compute Engine twin learned this the expensive way: it asserted "this rig is off the TPU API" about the code
as a whole and tested it on a single function, while four tools quietly shelled to `tpus tpu-vm ssh` and
failed against a VM that was plainly RUNNING. Grep beats a narrow unit test for a claim about a whole file.

## Write-up and validation order

Two drafts of this rig's own article, written 2026-08-25 from the runs in
`benchmarks/runs/2026-08-25-gke-first-provision-v6e1/` and the sweep in
`benchmarks/reports/2026-08-25-gemma4-e2b-v6e1.json`:

- **`devto-gke-gemma4-v6e1-step-by-step.md`** — the dev.to draft. Markdown tables and fenced code render
  there, so the data is inline.
- **`medium-gke-gemma4-v6e1.md`** — the Medium draft of the same material. **Not a copy**: Medium's importer
  drops markdown tables entirely and flattens multi-line code, so every table and code block is an image
  placeholder with an asset manifest at the end, all headings are `####`, and the prose is written to carry
  the argument without the figures. See `~/.claude/CLAUDE.md` for the import rules each of those follows.

Both are prose-complete and unillustrated; the figures still need rendering. Every number in them is
measured or catalog-read — if one changes here, change it in both.

`devto-tpu-api-vs-gce-provisioning.md` is the older rig-family article, written for two control planes and
now a plane short.

The format reference for a rig write-up — and for the order a validation pass runs in — is the dev.to
series, currently
[12B Gemma 4 with NVIDIA Blackwell 6000, QAT, MTP and Antigravity CLI](https://dev.to/gde/12b-gemma-4-deployment-with-nvidia-blackwell-6000-qat-mtp-and-antigravity-cli-3gn6):
first-person plural, ~20 short practical sections, command-then-output throughout, medal emoji in the
comparison tables. The lifecycle it fixes is the same one the earlier
[GCE + L4 article](https://dev.to/gde/12b-gemma-4-qat-deployment-with-gce-nvidia-l4-mcp-and-antigravity-cli-49d8)
uses:

> environment setup → MCP server over stdio → **deploy** → **validate** (`verify_model_health`,
> `get_system_status`, endpoint) → **benchmark** with a concurrency sweep → **cost comparison** table

That article's rig is an L4 GPU on Compute Engine, and its tool names are the ones used here —
`deploy_vllm`, `verify_model_health`, `get_system_status` — which is why this rig kept them through the GKE
port rather than inventing `deploy_to_gke`. A validation run that stops at "the model answered" is half a
pass: finish with the benchmark and the cost table, so the result is comparable to the sibling write-ups
instead of a fresh invention.

## Git

The git root is the **parent**, `/home/xbill/gemma4-dev` (`xbill9/gemma4-dev`). `git add .` from here stages
only this subdirectory. Commit straight to `main` — no branches, no PRs. Read `git status` before staging:
the tree routinely carries several unrelated bodies of in-progress work, some already staged.

`AGENTS.md` and `GEMINI.md` in this directory are maintained by different tools and overlap with this file.
**They were inherited from the fork and still describe the Queued Resource path** — they are wrong about
this rig until someone rewrites them. `CLAUDE.md` is correct where they disagree.
