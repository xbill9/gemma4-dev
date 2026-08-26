# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file MCP server (`server.py`, FastMCP) that serves Gemma 4 (`google/gemma-4-31B-it`) with vLLM on
an eight-chip Google Cloud TPU **v6e-8 (Trillium)** slice — one `ct6e-standard-8t` host, provisioned as a
**Compute Engine instance**, not a Cloud TPU API Queued Resource. Its tools shell out to
`gcloud compute instances` and talk HTTP to the vLLM OpenAI-compatible endpoint on port 8000.

**Forked from `gce-vllm-v6e8-2b` on 2026-08-25 and retargeted from E2B to the 31B checkpoint.** Same slice,
same control plane, same code lineage — **the checkpoint is the only thing that changed**, which means the
whole of the control-plane half of this file carries over verbatim while every claim that reasoned from
E2B's attention shape had to be rederived.

> ### The retarget inverts three conclusions this file used to state
>
> | | E2B (what the parent rig serves) | 31B (this rig) |
> | :--- | :--- | :--- |
> | KV per token, bf16 | 18 KiB | **960 KiB at TP=8** — 53x |
> | What TP=8 does to KV | multiplies it by 8 (MQA, 1 KV head) | **9.1% overhead** — 50 layers shard, 10 pad |
> | Is TP=8 optional? | yes, TP=1 fits one chip | **no** — 57.7 GiB of weights need ≥4 chips |
>
> When you find a stale E2B argument anywhere in this rig, **rederive it from `@../MODELS.md`; do not
> port it.** `SERVING-PARAMS.md` tags every row with whether its reason is E2B-only, v5e-only, chip, or
> stack.

**This rig is not an A/B twin of anything.** Its parent `gce-vllm-v6e8-2b` pairs with `tpu-vllm-v6e8-2b`
as a control-plane comparison at fixed checkpoint. Nothing here serves 31B through the Cloud TPU API, so a
measurement from this rig is a 31B-on-v6e-8 measurement and nothing else. **Do not diff it against a 2B
rig on either control plane and call the delta a control-plane result** — at 53x the KV cost, the
checkpoint dominates every number by more than an order of magnitude.

**This rig has provisioned nothing and measured nothing.** Neither has its parent, so there is no
inherited grant, no inherited allocation log, and no inherited throughput. The parent's flex-start grant
in `europe-west4-a` on 2026-08-10 was for **one** chip on the rig two forks back; capacity is sold per
slice and that is not evidence about an eight-chip request.

## There is no twin to keep in step

**The parent rig `gce-vllm-v6e8-2b` carried a "Keep the twin in step" section. It does not apply here and
was removed rather than edited**, because keeping a weakened version of it invites exactly the comparison
it used to enable. That section existed because the parent and `tpu-vllm-v6e8-2b` serve the *same
checkpoint* on the same slice through two control planes, which is a clean A/B. This rig changed the
variable that dominates: nothing here serves 31B through the Cloud TPU API, so there is no counterpart and
no comparison to protect.

Two things follow, and they pull in opposite directions:

- **You are free to tune serving flags for this checkpoint** without breaking anyone's A/B. `MAX_MODEL_LEN`,
  `MAX_NUM_BATCHED_TOKENS` and `TENSOR_PARALLEL_SIZE` are this rig's to set — that is why the KV arithmetic
  in `tpu.env` is written out rather than inherited.
- **A number from here cannot be subtracted from a 2B number.** At 960 KiB/token against 18, any delta you
  compute against `gce-vllm-v6e8-2b` or `tpu-vllm-v6e8-2b` is a checkpoint result wearing a control-plane
  or config disguise. If a report needs a control-plane claim, take it from the parent pair.

What is genuinely shared with the parent, and worth syncing back when you fix it: everything under "The two
control planes" below, the startup script's provisioning half, and any tool-level bug in `server.py`.

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

Every instance this rig creates carries `rig=gce-vllm-v6e8-31b`. Four rigs now provision `ct6e-*` instances
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

`_vllm_serve_flags()` also takes **per-run overrides** — `tensor_parallel_size`, `max_model_len`,
`kv_cache_dtype`, `gpu_memory_utilization`, `extra_flags` — and `manage_vllm_docker` passes them through.
They exist so a sweep can re-serve **one** instance under several configs instead of provisioning one
instance per arm; at $10.80/hr that is the difference between a $27 campaign and a $60 one. Every override
defaults to `None`, so with none passed the rendered flag string is byte-identical to what it was before
they existed and the boot path is unchanged. `kv_cache_dtype` and `gpu_memory_utilization` have no
`tpu.env` default and are **omitted entirely** unless passed — vLLM's own defaults stay in charge.

**An override forces a container REBUILD, and that is load-bearing rather than tidy.** `docker start`
replays the argv the container was *built* with and silently ignores anything new, so a TP=4 sweep step
against a container built at TP=8 would report success, serve at 8, and hand you a completely plausible
wrong number. `manage_vllm_docker` therefore `rm -f`s before recreating whenever an override is present,
and refuses overrides on actions that cannot apply them (`log`, `status`, `stop`, `rm`) rather than
accepting and ignoring them. Two tests pin it. **Confirm the override in the allocation line, never in the
tool's return value** — this rig's whole catalogue of advertised-vs-implemented gaps says so, and Zimbres
2026 §6.3 adds one more on this exact checkpoint: an fp8 banner over a bf16 allocation.

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

**The 62 GB checkpoint pull IS the 90-minute boot budget, and it costs ~$16 of every boot.** Eight chips
bill from the moment the VM boots, so the pull is spent before a single token is served — and spent again
on every fresh instance. `MODEL_GCS_URI` in `tpu.env` points the startup script at a pre-staged tar of the
Hugging Face cache in a bucket in `GOOGLE_CLOUD_REGION`; an in-region GCS read replaces the internet pull
and takes minutes. `./stage_model_to_gcs.sh` creates the bucket, grants the TPU service account
`roles/storage.objectViewer`, and stages the object from a cheap CPU VM that deletes itself via a trap;
`--status` says whether it landed. **The bucket must be in the same region** — one elsewhere restores over
the public backbone and gives back the whole saving.

Three details in the restore are deliberate and each was a wrong turn first:

- **It is a `tar`, streamed, not a directory copy.** The HF cache keeps one real blob under `blobs/` and
  symlinks `snapshots/` at it. `gcloud storage cp -r` *follows* symlinks, so a tree copy uploads 124 GB
  instead of 62 and restores two real copies of every shard.
- **It is not gzipped.** bf16 weights do not compress; gzip would trade GCS bandwidth for a CPU bottleneck.
- **`HF_HUB_OFFLINE=1` is set only after a restore that actually succeeded.** Set unconditionally it turns
  an incomplete cache from a slow boot into a dead one.

**Staging is an optimization and must never become a new failure mode.** The script checks the object is
readable before streaming and falls through to the normal Hugging Face pull on any failure, so an empty,
wrong, or unreadable `MODEL_GCS_URI` costs boot time and never a boot. Five tests pin the fallback, the
offline ordering, and that the template still renders with the URI both set and empty.

**Staged 2026-08-25**, 62,578,739,200 B (58.28 GiB): download 311s, upload 1093s, ~15 minutes wall on one
`e2-standard-8`. The object at 58.28 GiB against 59 GiB of on-disk cache is the check that `tar` stored the
`snapshots/` entries as symlinks — dereferenced they would have roughly doubled it. **The restore leg has
not been exercised yet**; it runs for the first time on the next boot, and the fallback is what makes that
safe to find out.

Incidentally this settles the 1% weight-size discrepancy between `@../MODELS.md` (57.7 GiB) and Zimbres
2026 (58.25 GiB): they measure different things. 57.7 GiB is parameter bytes (31.0B x 2), 58.25 GiB is the
checkpoint on disk, and the staged tar's 58.28 GiB sits with the latter. Neither is wrong; don't
"correct" one to the other.

**Two traps in `stage_model_to_gcs.sh` worth not rediscovering**, both found by running it:

- **No backticks in the stager heredoc, ever.** It is unquoted so `$VAR` interpolates from the outer
  script, which means a backtick is command substitution run by the OUTER shell at generation time. A
  backtick pair in a *comment* ran `hf download` locally and baked its usage text into the generated
  startup script as executable lines — valid bash in the file, corrupt on the VM. The script now
  `bash -n`s the generated artifact and greps it for stray CLI help before booting anything.
- **`huggingface_hub` 1.x has no `cli` or `hf_transfer` extra.** `pip install "huggingface_hub[cli,hf_transfer]"`
  only warns "does not provide the extra" and skips them, so `hf_transfer` silently never installs and the
  download runs single-stream. Install the packages by plain name and `import hf_transfer` to check.

**The stager uploads its log to GCS BEFORE deleting itself** (`--logs` retrieves it, and works after the VM
is gone). The first version deleted the VM in its trap and nothing else, so when it failed six minutes in
it destroyed the only copy of its own log. A cleanup path that erases the evidence of its own failure is
worse than no cleanup path — keep the ordering.

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

**In `europe-west4` — the zone this rig actually defaults to — the gap is far wider.** Read from the
catalog 2026-08-25: flex-start **$1.3500**/chip-hr against spot **$1.7820**, so spot costs **32% more**,
and eight chips are $10.80/hr against $14.26/hr. Spot's only role here is what `tpu.env` already says it
is — the cheap stockout probe that fails fast where flex-start queues. It is not a cost lever anywhere
this rig runs.

**Raw `/v1/completions` returns an empty completion on `-it` models.** `make query` and
`benchmarking_suite.py` use it, so an empty result there is expected. `server.py` uses `/v1/chat/completions`
throughout — keep new code on the chat endpoint.

**`--tensor-parallel-size` is 8, and on this checkpoint it is a floor rather than a preference.** 57.7 GiB
of bf16 weights do not fit one 31.24 GiB chip, so **TP=1 and TP=2 cannot boot this model at all** and TP=4
is the only real alternative. The parent rig's instruction to sweep 1/2/4/8 is not runnable here.

Weights shard across chips; the KV cache does not — it shards across **KV heads**, so a layer's cache
shards `min(TP, num_kv_heads)` ways and the runtime pads the head count up to TP when it falls short.
`@../MODELS.md` gives the 31B two attention geometries that straddle 8:

| | KV heads | at TP=8 |
| :--- | ---: | :--- |
| 50 sliding layers | `num_key_value_heads` = 16 | shards exactly, 2 heads/chip |
| 10 full layers | `num_global_key_value_heads` = 4 | **pads to 8 — 2x aggregate KV** |

So TP=8 costs a 2x KV penalty on a sixth of the layers and nothing on the rest: **960 KiB/token against an
ideal 880, a 9.1% overhead.** That reverses the row this file carried while the rig served E2B, which was
full MQA (`num_key_value_heads=1`) and paid 8x on *every* layer. Do not carry the old argument over.

`When_TP_Crosses_the_KV_Head_Count_v6e8.pdf` (Zimbres 2026, in this directory) measures exactly this
crossing on `ct6e-standard-8t` — the same machine type — **using Gemma 4 31B, the checkpoint this rig now
serves.** Doubling TP from 4 to 8 sped up the layers whose 16 KV heads still shard by **1.92x per layer**
and the layers at the 4-head limit by only **1.26x**. Two consequences:

- **The paper stopped being an analogy.** It is about this model on this machine type, so its *structure*
  now applies directly and TP=4 vs TP=8 is the sweep worth running.
- **Its throughput figures are still not ours to quote.** They were taken with a retuned decode block size
  and an fp8 cache; the kernel config and the sharding differ from what this rig runs. Its method
  transfers — verify at the allocation line, not the flag — its numbers do not.

## Profiling: xprof and TensorBoard

`make profile` captures a trace, `make trace-analyze` summarises it, `make tensorboard` / `make xprof`
render it. Traces land in `benchmarks/runs/<date>-xprof-tp<N>/` in TensorBoard's standard
`plugins/profile/<run>/*.xplane.pb` layout, so both viewers read the directory with no conversion.

**vLLM's documented profiling recipe does not work on this image, and it fails quietly.** Measured on
hardware 2026-08-25 against `vllm/vllm-tpu:nightly`: `VLLM_TORCH_PROFILER_DIR` is **not a known variable**
(`WARNING [envs.py:2208] Unknown vLLM environment variable detected`), `POST /start_profile` returns
**404**, and the OpenAPI document carries no profile route. The only profiler variables the build knows are
`VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`, `VLLM_CUSTOM_SCOPES_FOR_PROFILING`,
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, `VLLM_NVTX_SCOPES_FOR_PROFILING` and `VLLM_TRACE_FUNCTION` —
none of which start a trace. **Don't reinstate the env-var route.** They live in `profiling/` rather than the rig root, because a `sitecustomize.py` at the root would be auto-imported by every Python process started from this directory. A test pins that `capture_profile.sh`
does not use it (checking executable lines only; the comments name it on purpose).

So the trigger lives inside the engine process. `profiling/profiler_sidecar.py` runs a tiny HTTP control on
`VLLM_XPROF_PORT` (9012) whose `/start` and `/stop` call `jax.profiler.start_trace` / `stop_trace`;
`profiling/sitecustomize.py` loads it, and `capture_profile.sh` bind-mounts both into the container and puts them on
`PYTHONPATH`, which is what makes Python auto-import `sitecustomize`.

Three properties of the sidecar are load-bearing rather than defensive:

- **It installs only in the process that owns the TPU.** vLLM runs the API server and the engine as
  separate processes (`APIServer pid=1`, `EngineCore pid=711`). Installing in the API server would capture
  nothing and would bind the port the engine needs.
- **Nothing in it may raise.** `sitecustomize` runs during interpreter startup, so an exception there does
  not fail the module — it fails the container.
- **It is inert unless `VLLM_XPROF_DIR` is set**, so mounting it costs a container nothing.

**`make profile` restarts the engine**, because the sidecar has to be present at process start. On this
checkpoint that is ~7 minutes of recompilation (measured: `init engine ... took 427.88 s (compilation:
399.57 s)`). The checkpoint is already in `/dev/shm`, so nothing is re-downloaded.

**The xprof CLI has no `capture` subcommand** — capture happens in-process; the CLI is all *analysis* over
a captured session. `analyze_trace.py` drives the useful ones: `get_hlo_op_profile`,
`get_device_information`, and the pathology detectors `detect_unnecessary_convert_reduce`,
`detect_layout_mismatch_copies`, `detect_unfused_reshapes`. That first detector is the TPU analogue of the
bug that cost `gpu-jax-g5g-2b` 55% of its decode step to `wrapped_convert`.

**Per-op MEANS, never a sum divided by a step count.** Zimbres 2026 §5 documents the trace export
truncating silently at a fixed event count — ~7 of 127 decode steps at TP=8, ~14 at TP=4 — so any statistic
formed by dividing a raw sum by the requested step count is wrong by the truncation ratio. Each per-layer
op fires once per step per chip, so its mean per firing is its true per-step cost whatever the export did.

**Unvalidated on hardware.** The sidecar was built after the 2026-08-25 slice was torn down and has never
run on a TPU. Its guards are unit-tested; the injection path is not.

## Names derive from the rig directory

`RIG_NAME` is `os.path.basename(...)` of this directory. `INSTANCE_NAME` defaults to it and is the default
name of every MCP tool; `MCP_SERVER_NAME` defaults to it and names the FastMCP server; the Makefile's
`SERVICE_NAME` is `$(notdir $(CURDIR))`. All resolve to `gce-vllm-v6e8-31b`.

`RESOURCE_ID` is kept as a back-compat alias of `INSTANCE_NAME` because the forked tool signatures spell it
that way. On this path it names an instance.

The MCP server name has to match the key the client registers it under, because that key prefixes every
tool: `mcp__gce-vllm-v6e8-31b__find_tpu`. That is what distinguishes this rig's tools from the TPU-API twin's
`mcp__tpu-vllm-v6e8-2b__…` and from the one-chip fork's `mcp__gce-vllm-v6e1-2b__…`, and with any two of them
loaded the prefix is the *only* thing that does.

`load_dotenv` runs *before* `FastMCP(...)` is constructed, because `MCP_SERVER_NAME` set in `tpu.env` would
otherwise arrive too late to name the server. Don't move the FastMCP construction back above the dotenv
block.

**Renaming the rig directory orphans anything already provisioned** — pin `INSTANCE_NAME` in `tpu.env`
before renaming, or destroy first.

## Silicon facts live at the monorepo root

`@../HARDWARE.md` is canonical for v6e's memory, bandwidth, native numeric formats, gcloud spelling, and the
control-plane table; `@../MODELS.md` for the 31B's layer structure, KV cost per token, and weight footprint
— including the `attention_k_eq_v` trap, where **ten missing `v_proj` tensors on load are correct, not
corruption**, and a loader that tolerates `None` yields a silently broken model that still emits fluent text.
Read them rather than re-deriving, and correct them there rather than restating a number here.

Two of their facts decide things in this rig:

- **v6e has no native fp8** — int8 is the only low-precision format with a compute win. fp8 buys footprint
  and bandwidth, never FLOPS. v7/Ironwood is the first TPU that changes this.
- **32 GB HBM per chip, 31.24 GiB usable — ~250 GiB across the slice.** On this checkpoint the weights are
  a first-class term for the first time on this rig: 57.7 GiB, 23% of the slice, before any KV is cut.
  Subtracting weights and ~19.8 GiB of headroom leaves a KV pool near **172 GiB, roughly 184,000 KV
  tokens** at 960 KiB/token — against the 1,151,744 an E2B measured on a *single* v6e-1. Eight times the
  silicon, a sixth as many tokens. **That is arithmetic and no allocation log has been read here**; the
  full derivation and its caveats are in `SERVING-PARAMS.md`. Read the allocation line before quoting it.

## Benchmarks

`benchmarks/runs/` and `benchmarks/reports/` are **empty, and this fork inherited nothing** — the parent
`gce-vllm-v6e8-2b` had already emptied them, so unlike earlier forks in this lineage there was nothing to
strip. Keep it that way until this rig measures something. `benchmarks/rollup.py` globs `*/benchmarks/`,
so any file dropped here is credited to a **31B on v6e-8** whatever hardware and whatever checkpoint it
actually came off — the failure mode that produced this monorepo's misattributed 2B/12B/31B reports.

`benchmarks/serving-report.schema.json`, `benchmarks/README.md`, and `benchmarks/INDEX.md` are generated or
synced from the monorepo root. Don't hand-edit them here.

Writers (`run_sweep.py`, `run_grid_benchmark.py`, `run_fast_sweep.py`, `benchmarking_suite.py --output`)
use bare filenames in the CWD on purpose. File new output into a new dated run dir.

**The first sweep here has no sibling to be comparable to.** Report it as what it is — 31B, bf16, v6e-8,
`ct6e-standard-8t`, flex-start, TP=8 — and record the boot allocation alongside it. **Every capacity claim
in this rig is currently arithmetic**; the first `total_hbm_used_gb` line is what turns any of it into a
measurement, and `@../MODELS.md` documents a 12B rig that published a per-token KV figure wrong twice over
by skipping exactly that check.

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
