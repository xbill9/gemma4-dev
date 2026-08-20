# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file MCP server (`server.py`, FastMCP) that serves Gemma 4 (`google/gemma-4-E2B-it`) with vLLM on an
eight-chip Google Cloud TPU **v6e-8 (Trillium)** slice — one `ct6e-standard-8t` host, provisioned as a
**Compute Engine instance**, not a Cloud TPU API Queued Resource. Its tools shell out to
`gcloud compute instances` and talk HTTP to the vLLM OpenAI-compatible endpoint on port 8000.

**This rig exists to be compared against `tpu-vllm-v6e8-2b`.** That rig is the same slice size, the same
checkpoint, reached through the deprecated control plane. This one is the migration target, and the whole
point is that *only* the provisioning path should vary — see **Keep the twin in step** below before
changing anything.

**Forked from `gce-vllm-v6e1-2b` and retargeted from one chip to eight on 2026-08-19.** Everything about
the *control plane* below was verified on the v6e-1 fork and carries over unchanged — it is a property of
`gcloud compute instances`, not of the slice size. Everything that counts chips does **not** carry over
and has been recomputed: quota arithmetic, cost, and tensor parallelism. **This rig has provisioned
nothing at v6e-8.** The v6e-1 rig's flex-start grant in `europe-west4-a` on 2026-08-10 was for one chip
and is not evidence about an eight-chip request; capacity is sold per slice.

The inherited `benchmarks/runs/2026-08-10-gce-flex-v6e1/` was **deleted** in the same pass — it was a
byte-identical copy of the v6e-1 rig's run, still lives there, and `rollup.py` globs `*/benchmarks/`, so
leaving it would have credited this rig with a measurement from different hardware.

## Keep the twin in step

`tpu-vllm-v6e8-2b` and this rig should differ in slot 1 of their names and in nothing else that matters. A
change here that is **not** about provisioning — serving flags, `MAX_MODEL_LEN`, the benchmark harness, the
startup script's vLLM invocation — has to land in both or the comparison stops being one.
`TENSOR_PARALLEL_SIZE=8` matches today.

**They are not comparable yet, and this is the first thing to fix before any measurement.** The twin is a
different code lineage from the v6e-1 pair: it has no `tpu.env`, and it hardcodes `--max-model-len 16384`
where this rig reads `MAX_MODEL_LEN=32768`. A throughput delta between them today is a serving-config
delta, not a control-plane one. Reconcile the two, or state the difference in the report.

What is *allowed* to differ: everything under "The two control planes" below.

## Commands

```
make install    # pip install -r requirements.txt
make run        # python server.py (stdio MCP server)
make test       # python test_agent.py — unittest, NOT pytest
make lint       # ruff check . && ruff format --check . && mypy .
make format     # apply ruff formatting and autofixes
make tools      # regenerate GemmaTools.md from the @mcp.tool() decorators
make deploy-tpu / deploy-tpu-spot / deploy-tpu-ondemand / deploy-tpu-flex
make benchmark  # discovers the instance IP, then runs benchmarking_suite.py against it
make query PROMPT="..."
```

`make lint` only *checks* formatting — `make format` is what writes it. Both `make lint` and `make test`
pass clean (55 tests as of 2026-08-19); keep them that way. A `PostToolUse` hook in `.claude/settings.json` runs
`ruff format` on every `.py` file Claude edits.

**`make deploy-tpu-flex` has no equivalent in the sibling rig's Makefile**, because gcloud's `tpu-vm` path
offers no flex-start at all — there, flex-start was reachable only through the MCP tools' Queued Resource
path.

## The two control planes

This is the rig's whole subject, so it gets the detail.

### The Cloud TPU API is deprecated

> The Cloud TPU API is no longer under active development. This includes the Google Cloud CLI for the Cloud
> TPU API and the Cloud Client Libraries for the Cloud TPU API.

Bug and security fixes only, **no published sunset date**. New generations from TPU7x (Ironwood) are Compute
Engine or GKE only. `@../HARDWARE.md` holds the per-generation table; the short version is that v5p, v6e and
TPU7x have a Compute Engine path and **v5e does not**, so the six v5e rigs can never migrate.

### Flag-by-flag

| Cloud TPU API (`tpu-vllm-v6e8-2b`) | Compute Engine (this rig) |
| :--- | :--- |
| `queued-resources create --provisioning-model=flex-start` | `instances create --provisioning-model=FLEX_START` |
| `--accelerator-type=v6e-8` | `--machine-type=ct6e-standard-8t` |
| `--runtime-version=v2-alpha-tpuv6e` | `--image-family=…-v5e-v5p-v6e --image-project=ubuntu-os-accelerator-images` |
| `--valid-until-duration` (bounds the request) | `--request-valid-for-duration` |
| `--max-run-duration` (**flex-start only**) | `--max-run-duration` (**any model**) + `--instance-termination-action=DELETE` |
| — | `--scopes=cloud-platform` (**required**, see below) |
| — | `--maintenance-policy=TERMINATE` (**required** — TPU instances cannot live-migrate) |
| spot via `TPUV6EPreemptible…` TPU-API quota | spot via `PREEMPTIBLE-TPU-V6E-per-project-zone` Compute quota |
| QR → derived `<resource_id>-node` | the instance **is** the node |
| no equivalent | `--provisioning-model=RESERVATION_BOUND` |

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
and it is false. Re-read 2026-08-19, the project holds **CT6E = 32 in ten regions** — asia-east1,
asia-northeast1, asia-south1, asia-southeast1, europe-west4, southamerica-east1, southamerica-west1,
us-east1, us-east5, us-south1 — and **no stated value (which reads as 0) in us-central1, us-east4,
us-west1**. The two pools are **disjoint and regionally misaligned**, which is a sharper and more useful
statement than "one is empty": the zone this rig originally defaulted to was the one zone where the
project's large TPU-API holding sits and its Compute Engine holding did not. (An earlier revision listed
us-east1 and us-east5 among the unset regions; that was the 2026-08-10 reading, before the 0 → 32 grants
of 2026-08-11.)

`GOOGLE_CLOUD_ZONE` therefore moved to **`europe-west4-a`** on 2026-08-10. Both zones publish
`ct6e-standard-8t`, so machine-type availability is not the discriminator; regional quota is.

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

**Quota held on the Compute Engine path, verified 2026-08-11** (regions publishing `ct6e-standard-8t`; the CT6E family column re-read 2026-08-19 and unchanged). **Divide every number by 8 to get instances** — quota is metered in chips and this rig spends eight per create, so europe-west4's 32 is four concurrent slices, not thirty-two:

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

### Discovery is a different API, not a different filter

**A `ct6e-*` instance does not appear in `gcloud compute tpus tpu-vm list` at all.** It is an ordinary
Compute Engine instance that happens to carry a TPU. The sibling rig's `_list_tpu_vm_nodes()` lists TPU VM
nodes and would return nothing here, silently — so the two rigs cannot share a discovery helper, and
`test_a_ct6e_instance_is_not_a_tpu_vm_node` pins that this rig never calls `tpu-vm`.

**That test only ever covered discovery, and four tools were violating the rule behind its back.**
`manage_vllm_docker`, `run_vllm_benchmark`, `get_vllm_docker_logs` and `get_tpu_system_logs` all shelled to
`gcloud compute tpus tpu-vm ssh` after the fork — so every one of them failed with a not-found against an
instance that was plainly RUNNING. Fixed 2026-08-10: they go through **`_ssh_command()`**, which builds
`gcloud compute ssh` (no `--tunnel-through-iap`; these instances have an external IP). Two new tests pin
it. **The lesson generalises: "this rig is off the TPU API" was asserted about the code as a whole and
tested on one function.** Grep for `tpu-vm` before believing it again — the three surviving hits are
deliberate `queued-resources list` calls for cross-path collision detection.

Three field-shape differences fall out, and each fails quietly rather than loudly:

- Status is **`status: RUNNING`**, not `state: READY`. Copying the sibling's check demotes every healthy
  instance to the bottom of the ranking instead of erroring — visible only when two are up.
- The IP is at **`networkInterfaces[].accessConfigs[].natIP`**, not
  `networkEndpoints[].accessConfig.externalIp`.
- `RUNNING` is a **weaker claim than the QR's `ACTIVE`**. A Queued Resource reached ACTIVE once its node was
  up; an instance is RUNNING the moment the VM boots, long before the startup script has pulled the vLLM
  image or loaded the model. Use `verify_model_health` for readiness, never `check_tpu_availability`.

`_resolve_node_id()` is two lookups here instead of the sibling's three, because the id you ask for is the
name you get — there is no derived `<id>-node` and no class of mismatch to reconcile.

### Instances are labelled; Queued Resources were not

Every instance this rig creates carries `rig=gce-vllm-v6e8-2b`. Four rigs now provision `ct6e-*` instances
into this project and an instance name does not encode its owner the way a QR id did by convention.

**`manage_tpu_instance` is deliberately narrower than the sibling's `manage_queued_resource`**, which
deletes every Queued Resource in the zone that is not the named primary. This one only deletes instances
carrying this rig's label; anything unlabelled or belonging to a sibling is reported and left alone. **Do
not "fix" this to match the sibling.**

`create_tpu_instance` is non-destructive and touches only the name it was given, so `find_tpu`'s zone sweep
is safe. Keep that split.

### What is unchanged

The startup script, the secret handling, the vLLM flags, and the billing-catalog lookup are all identical to
the sibling rig, and deliberately so.

## Gotchas

**`startup_script_template.sh` is consumed by `str.format()`.** Placeholders are `{project_id}`, `{zone}`,
`{model_name}`, `{hf_secret_id}`, `{tensor_parallel_size}`, `{max_model_len}`, `{max_num_batched_tokens}`,
`{limit_mm_per_prompt}`. Any other literal `{` or `}` — a shell brace expansion, a `${VAR}`, a JSON literal
— raises at format time and breaks the deploy. Escape as `{{` / `}}`.

**The startup script fetches the HF token itself; never add a `{hf_token}` placeholder back.** The rendered
script is uploaded as instance metadata, so a baked-in token would be readable from the instance. It reads
`hf-token` from Secret Manager at boot via the metadata server, retrying for 30 minutes so an IAM grant
applied after creation still lands. The VM's service account needs `roles/secretmanager.secretAccessor` on
the secret, and the instance needs `--scopes=cloud-platform` to use it. Tracing (`set -x`) is off across the
whole token section — keep it that way, and never interpolate `$HF_TOKEN` into a logged string.

**The chip count is derived, never a literal.** `_chips_in_machine_type()` reads the trailing `-<N>t` off
`MACHINE_TYPE`, so `CHIP_COUNT` cannot drift from what gcloud is actually asked for; `CHIP_COUNT` in
`tpu.env` is an override for a future shape that stops encoding the count in its name. It matters because
**quota is metered in chips and every billing-catalog rate is per chip-hour** — a stale `1` here would
understate both by a factor of eight, and understate them *silently*, since a `FLEX_START` create short of
quota queues rather than erroring. `TOPOLOGY` (`2x4`) is the mesh, and is the default `estimate_deployment_cost`
prices; it warns when a passed topology disagrees with `CHIP_COUNT`. Three tests pin the trio.

**`google.com/tpu` in the GKE manifest is a CHIP request, not a TP setting.** It tracks `CHIP_COUNT`. The
two are numerically equal here and are different quantities — TP is a sharding choice, the limit is how
much hardware the pod is handed — and the manifest now also emits the `gke-tpu-accelerator` /
`gke-tpu-topology` node selectors a multi-chip pod needs.

**Serving flags live in one place.** `_vllm_serve_flags()` builds the vLLM arg list from `MAX_MODEL_LEN`,
`MAX_NUM_BATCHED_TOKENS`, `LIMIT_MM_PER_PROMPT`, and `TENSOR_PARALLEL_SIZE`; the startup script takes the
same values as placeholders. Don't reintroduce a second hardcoded flag list. The JSON value needs different
quoting inside a single-quoted argument — that's what the `mm_limit` parameter is for.

**The Compute Engine image ships no Docker, and this is a control-plane difference rather than an
oversight.** `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` has no `docker` on PATH at first boot; the TPU API's
`v2-alpha-tpuv6e` runtime versions do. The startup script was inherited verbatim from the sibling, so the
first instance this rig ever created died 100 seconds in — five pull retries, each `sudo: docker: command
not found` — **while continuing to report `status: RUNNING` indefinitely**. That is the "RUNNING is weaker
than ACTIVE" warning above arriving as a real failure: nothing distinguishes a dead boot from a healthy one
except `/var/log/vllm-startup.log` or curling `:8000`.

Fixed in three places, because the fact bites at three layers: `startup_script_template.sh` installs
`docker.io` before pulling; `_ENSURE_DOCKER` prefixes every Docker-dependent remote command, so
`manage_vllm_docker start` — precisely what you reach for when a boot failed — does not fail the same way;
and `get_vllm_deployment_config` carries the install in the one-liner it emits. Don't "simplify" any of
them away.

**The boot disk default is 200 GB, and the image default of 10 GB cannot hold the vLLM TPU image.**
Undersizing fails late, after boot, during the docker pull.

**Pin the image FAMILY, never a dated build.** Images ship roughly weekly and every superseded build goes
`DEPRECATED`. The current one is `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v20260803`; there is also an older
family spelling, `ubuntu-accelerator-2204-amd64-with-tpu-v5e-v5p-v6e` — use the `ubuntu-accel-…` form.

**There are two machine-type families and they are not known to be interchangeable.** `ct6e-standard-8t`
reports `guestAcceleratorType: ct6e`; `ct6e-standard-8t-tpu` reports `tpu-v6e`. Identical vCPU, memory and
zone coverage. `MACHINE_TYPE` in `tpu.env` is config for exactly this reason. (The v5p rig picked the `-tpu`
form; this one defaults to the bare form. That difference is unexamined, not a considered choice.)

**Do not read the bare form as legacy.** Google's CE quickstart creates `--machine-type=ct6e-standard-4t`
and every shape on the CE machine-types page is bare, so the bare ones are the documented creatable ones.
An earlier draft of `@../HARDWARE.md` had this backwards and used it to argue v5e has no CE path; see the
corrected evidence table there before repeating that reasoning.

**A create can time out client-side with the request still live.** `_create_tpu_instance` caps gcloud at
590s and returns `⏳`, not `❌` — the instance may still appear and bill. `find_tpu` treats that as a
capacity outcome and does **not** record it as a zone failure, because gate 3 is never cached.

**`tpu_zones_status.md` is mutable state, not documentation.** `find_tpu` rewrites it in place. It is empty
because nothing has been attempted. **Never seed it from the sibling rig's file** — a zone that rejects a
Queued Resource is not evidence about an instance create.

**Don't destroy an instance unless asked.** Flex-start capacity can take a long time to come back.

**`estimate_deployment_cost` reads live pricing — never reintroduce a rate table.** It queries the Cloud
Billing Catalog (Compute Engine service `6F81-5844-456A`, where TPU SKUs live). The catalog is shared with
the sibling rig, so both spellings still apply: flex-start is sold as **"DWS Defined Duration V6e"** and
drops the `Tpu` prefix, spot is `usageType: Preemptible` spelled `TpuV6e attached to Spot Preemptible VMs`.
Patterns are anchored with `^` so the `Reserved …`, `Commitment v1: …` and `Capacity Optimized TpuV6e …`
SKUs don't match.

**Don't assume spot is the cheapest.** In us-east5 v6e flex-start lists at $1.35/chip-hr and spot at
$1.4033 — spot is *dearer*. Read it out of the tool.

**Raw `/v1/completions` returns an empty completion on `-it` models.** `make query` and
`benchmarking_suite.py` use it, so an empty result there is expected. `server.py` uses `/v1/chat/completions`
throughout — keep new code on the chat endpoint.

**`--tensor-parallel-size` is 8, and this is the one number on this rig that deserves an argument.**
Eight uses the whole slice and matches the twin. It is *not* a free win for this checkpoint, and the
reason is the subject of the paper sitting in this directory:

`@../MODELS.md` records E2B as full MQA with `num_key_value_heads=1`. Weights shard across chips; the KV
cache does not — it shards across **KV heads**, so a layer's cache shards `min(TP, num_kv_heads)` ways.
For E2B that is **1 at every TP**. At TP=8 the runtime pads the head count up to the parallel degree, and
E2B's per-token KV cost is *multiplied* by eight rather than divided, while the weights that do shard are
only ~4.5 GiB per chip to begin with. What the slice buys E2B is **aggregate HBM and batch headroom, not a
shorter decode step.**

`When_TP_Crosses_the_KV_Head_Count_v6e8.pdf` (Zimbres 2026, in this directory) measures exactly this
crossing on `ct6e-standard-8t` — the same machine type — using Gemma 4 31B, whose hybrid attention puts
both cases in one model. Doubling TP from 4 to 8 sped up the layers whose 16 KV heads still shard by
**1.92x per layer** and the layers at the 4-head limit by only **1.26x**. E2B is the degenerate case of
that result: *every* layer is at the limit. Two consequences for this rig:

- **Measure TP=1/2/4/8 before declaring 8 the operating point.** A 2B model on eight chips is not
  obviously better than on one, and the rig can answer that question directly.
- **Do not copy the paper's throughput figures into this rig's reports.** They are 31B numbers with a
  retuned decode block size and an fp8 cache; the model, the kernel config and the sharding are all
  different. Its *method* transfers — verify at the allocation line, not the flag — its numbers do not.

## Names derive from the rig directory

`RIG_NAME` is `os.path.basename(...)` of this directory. `INSTANCE_NAME` defaults to it and is the default
name of every MCP tool; `MCP_SERVER_NAME` defaults to it and names the FastMCP server; the Makefile's
`SERVICE_NAME` is `$(notdir $(CURDIR))`. All resolve to `gce-vllm-v6e8-2b`.

`RESOURCE_ID` is kept as a back-compat alias of `INSTANCE_NAME` because the forked tool signatures spell it
that way. On this path it names an instance.

The MCP server name has to match the key the client registers it under, because that key prefixes every
tool: `mcp__gce-vllm-v6e8-2b__find_tpu`. That is what distinguishes this rig's tools from the TPU-API twin's
`mcp__tpu-vllm-v6e8-2b__…` and from the one-chip fork's `mcp__gce-vllm-v6e1-2b__…`, and with any two of them
loaded the prefix is the *only* thing that does.

`load_dotenv` runs *before* `FastMCP(...)` is constructed, because `MCP_SERVER_NAME` set in `tpu.env` would
otherwise arrive too late to name the server. Don't move the FastMCP construction back above the dotenv
block.

**Renaming the rig directory orphans anything already provisioned** — pin `INSTANCE_NAME` in `tpu.env`
before renaming, or destroy first.

## Silicon facts live at the monorepo root

`@../HARDWARE.md` is canonical for v6e's memory, bandwidth, native numeric formats, gcloud spelling, and the
control-plane table; `@../MODELS.md` for E2B's layer structure, KV cost per token, and weight footprint.
Read them rather than re-deriving, and correct them there rather than restating a number here.

Two of their facts decide things in this rig:

- **v6e has no native fp8** — int8 is the only low-precision format with a compute win. fp8 buys footprint
  and bandwidth, never FLOPS. v7/Ironwood is the first TPU that changes this.
- **32 GB HBM per chip, ~19.8 GiB free for KV** after E2B's weights on **one** chip. Eight chips is
  ~250 GiB of HBM in aggregate — that is what this slice buys: v6e costs 2.25x a v5e for ~1.9x the
  bandwidth, so it does not buy decode throughput, it buys context and batch. Note the per-chip free-KV
  figure does **not** simply multiply by eight for E2B: TP=8 replicates rather than shards its single-KV-head
  cache, so aggregate KV capacity is closer to one chip's than to eight chips'. Read the allocation line.

## Benchmarks

`benchmarks/runs/` and `benchmarks/reports/` are **empty on purpose**. This fork *did* arrive carrying
`runs/2026-08-10-gce-flex-v6e1/`, a byte-identical copy of the v6e-1 rig's run; it was deleted on
2026-08-19 and the original is untouched in `../gce-vllm-v6e1-2b/`. None of it was measured on this
hardware, and `benchmarks/rollup.py` globs `*/benchmarks/` — leaving it would have credited a v6e-8 rig
with a v6e-1 result, which is exactly the failure `rollup.py` was written to expose.

`benchmarks/serving-report.schema.json`, `benchmarks/README.md`, and `benchmarks/INDEX.md` are generated or
synced from the monorepo root. Don't hand-edit them here.

Writers (`run_sweep.py`, `run_grid_benchmark.py`, `run_fast_sweep.py`, `benchmarking_suite.py --output`)
use bare filenames in the CWD on purpose. File new output into a new dated run dir.

**When the first sweep runs, it is only comparable to the sibling's if the serving config matches.** Check
`tpu.env` against the twin's serving flags before believing a difference is about the provisioning path — and note that `tpu-vllm-v6e8-2b` has no `tpu.env` to compare against, so read its `server.py` defaults.

## Auth and env

Requires both `gcloud auth login` and `gcloud auth application-default login` (ADC, for the
`google-cloud-secret-manager` client). `set_env.sh` must be **sourced**, not executed. `init.sh` blocks on
`read` in its error path — don't run it non-interactively.

`tpu.env` is the single source of truth and is committed. A real environment variable always wins over it
(`load_dotenv` doesn't overwrite, `mcp-run.sh` exports only unset keys, the Makefile uses `?=`), so
`make status ZONE=...` works for a one-off. Defaults are `us-east5-b` / `us-east5`.

## Tests

`test_agent.py` mocks the whole `mcp` module and the Google Cloud clients before importing `server`. Keep
unit tests offline. Because `mcp` is a `MagicMock`, anything calling `mcp.list_tools()` needs an explicit
`AsyncMock` patch — see `test_get_help`.

The `_node()` fixture is deliberately a **different shape** from the sibling rig's (`status`/`RUNNING`,
`networkInterfaces[].accessConfigs[].natIP`). Copying that fixture over would make discovery look tested
while matching nothing.

## Git

The git root is the **parent**, `/home/xbill/gemma4-dev` (`xbill9/gemma4-dev`). `git add .` from here stages
only this subdirectory. Commit straight to `main` — no branches, no PRs. Read `git status` before staging:
the tree routinely carries several unrelated bodies of in-progress work, some already staged.

`AGENTS.md` and `GEMINI.md` in this directory are maintained by different tools and overlap with this file.
**They were inherited from the fork and still describe the Queued Resource path** — they are wrong about
this rig until someone rewrites them. `CLAUDE.md` is correct where they disagree.
