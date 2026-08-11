# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file MCP server (`server.py`, FastMCP) that serves Gemma 4 (`google/gemma-4-E2B-it`) with vLLM on one
Google Cloud TPU **v5e-1** chip — provisioned as a **Compute Engine instance**, not a Cloud TPU API Queued
Resource. Its tools shell out to `gcloud compute instances` and talk HTTP to the vLLM OpenAI-compatible
endpoint on port 8000.

Created 2026-08-10 by porting `gce-vllm-v6e1-2b` (itself created that day) onto v5e. **Nothing here has
ever provisioned anything.**

## ⚠️ Read this before spending money: v5e may not have this path at all

Every other rig in this monorepo serves a chip on a control plane that is known to work for it. This one
does not. **Whether `gcloud compute instances create --machine-type=ct5lp-hightpu-1t` is accepted is an open
question, and this rig is the apparatus for settling it** — not a deployment path with a track record.

The evidence, in full, as of 2026-08-10:

| Signal | Reading |
| :--- | :--- |
| Google's [v5e page](https://docs.cloud.google.com/tpu/docs/v5e): *"TPU v5e is supported using Google Kubernetes Engine and the Cloud TPU API"* | **against** — Compute Engine is not named |
| The CE machine-types page enumerates TPU7x, v6e, v5p; `ct5lp` appears nowhere on it | **against** — explicit |
| That same v5e page documents `ct5lp-hightpu-1t` as a VM type, 24 vCPU / 48 GB | for |
| `ct5lp-hightpu-{1,4,8}t` are real machine types in **26 zones**, `us-west4-a` among them | for |
| Image family is literally `ubuntu-accel-2204-amd64-tpu-**v5e**-v5p-v6e` | for |
| `TPU-LITE-PODSLICE-V5-per-project-zone` is a **compute.googleapis.com** quota | for |

**Every "for" signal has one innocent explanation.** The Cloud TPU API and GKE are both implemented *on*
Compute Engine: their TPU VMs are GCE instances, booting GCE images, drawing on GCE quota. GKE in
particular consumes `ct5lp-*` by name when you create a node pool — which is exactly why the machine type
is documented on a page that does not offer a `gcloud compute instances` path. So all four positives are
what you would see whether or not `instances create` takes the type directly.

The working answer is therefore **no**, and it is a documented-but-unattempted no.

**It is cheap to settle.** One `create_tpu_instance` (or `make deploy-tpu`) does it: a rejection at
validation is free and conclusive, an acceptance bills until deleted. When someone runs it, record the
result in `@../HARDWARE.md` (§"Can v5e use the Compute Engine path?") and `@../NAMING.md` (which currently
says a `gce-*-v5e1-*` rig "is not currently buildable") — it decides whether **six** v5e rigs have a
migration path at all. Do not quietly delete either claim; correct it with the evidence.

Everything below is written as if the answer is yes, because that is the only way to run the experiment.

## Keep the twin in step

`tpu-vllm-v5e1-2b` and this rig differ in slot 1 of their names and in nothing else that matters. A change
here that is **not** about provisioning — serving flags, `MAX_MODEL_LEN`, the benchmark harness, the startup
script's vLLM invocation — has to land in both or the comparison stops being one. `MAX_MODEL_LEN=16384`,
`TENSOR_PARALLEL_SIZE=1` and `_vllm_serve_flags()` are identical on purpose.

Note that twin is the **live-demo rig**. Prefer changes there that keep the demo working; this rig carries no
such constraint, which makes it the right place to try provisioning experiments.

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
pass clean (47 tests); keep them that way. A `PostToolUse` hook in `.claude/settings.json` runs
`ruff format` on every `.py` file Claude edits.

**`make deploy-tpu-flex` has no equivalent in the sibling rig's Makefile**, because gcloud's `tpu-vm` path
offers no flex-start at all — there, flex-start was reachable only through the MCP tools' Queued Resource
path.

**Unlike the sibling rig, the make targets and the MCP tools address the same object.** There, the Makefile
spoke `tpu-vm` while the tools spoke `queued-resources`, so `make status` described a different TPU node from
the one the agent had provisioned. Here both call `gcloud compute instances`. The v6e rig this was ported
from still has `tpu-vm describe` in its `status`/`endpoint`/`benchmark`/`query`/`destroy-tpu` targets and
they cannot see anything it deploys; that was fixed here, don't port it back.

## The two control planes

This is the rig's whole subject, so it gets the detail.

### The Cloud TPU API is deprecated

> The Cloud TPU API is no longer under active development. This includes the Google Cloud CLI for the Cloud
> TPU API and the Cloud Client Libraries for the Cloud TPU API.

Bug and security fixes only, **no published sunset date**. New generations from TPU7x (Ironwood) are Compute
Engine or GKE only. `@../HARDWARE.md` holds the per-generation table — and per that table v5e is the one
generation with **no documented exit**, which is what the section above is about.

### Flag-by-flag

| Cloud TPU API (`tpu-vllm-v5e1-2b`) | Compute Engine (this rig) |
| :--- | :--- |
| `queued-resources create --provisioning-model=flex-start` | `instances create --provisioning-model=FLEX_START` |
| `--accelerator-type=v5litepod-1` | `--machine-type=ct5lp-hightpu-1t` |
| `--runtime-version=v2-alpha-tpuv5-lite` | `--image-family=…-v5e-v5p-v6e --image-project=ubuntu-os-accelerator-images` |
| `--valid-until-duration` (bounds the request) | `--request-valid-for-duration` |
| `--max-run-duration` (**flex-start only**) | `--max-run-duration` (**any model**) + `--instance-termination-action=DELETE` |
| — | `--scopes=cloud-platform` (**required**, see below) |
| — | `--maintenance-policy=TERMINATE` (**required** — TPU instances cannot live-migrate) |
| spot via `TPUV5sPreemptibleLitepod…` TPU-API quota | spot via `PREEMPTIBLE-TPU-LITE-PODSLICE-V5-per-project-zone` |
| QR → derived `<resource_id>-node` | the instance **is** the node |
| no equivalent | `--provisioning-model=RESERVATION_BOUND` |

**`v5litepod-1` never appears on this path.** `ACCELERATOR_TYPE` is retained in `tpu.env` as documentation so
reports and the directory name line up with the twin; `gcloud compute instances create` would reject it.
The v5e rename is the sharpest of the three generations — see `@../HARDWARE.md`, and note that `CT5LP`,
`v5litepod`, `TPU-LITE-PODSLICE-V5` and `v5e` are four spellings of one chip across four surfaces.

**Values are SCREAMING_CASE here.** `flex-start` vs `FLEX_START` is the same request to a different API.
This is the one failure on this path that costs nothing — gcloud validates the enum client-side — but
`_provisioning_flags()` is the single place the mapping lives and a test pins it. Don't add a second mapping.

**`--scopes=cloud-platform` is load-bearing and its absence fails late.** Without it the booted VM cannot
reach Secret Manager, and the startup script spins through its full 30-minute retry before giving up — so a
missing scope looks like a slow boot for half an hour and then like a token problem. The Queued Resource
path got a workable default scope set; `instances create` does not.

**`--max-run-duration` is not flex-start-only here.** On the TPU API it is, which is why a spot or on-demand
Queued Resource had no automatic stop and billed until destroyed. On Compute Engine every model can carry
one, and pairing it with `--instance-termination-action=DELETE` is what makes a demo VM clean up after
itself. Defaults: `MAX_RUN_DURATION=4h`, `REQUEST_VALID_FOR=2h`, both in `tpu.env`.

**`RESERVATION_BOUND` has no catalog rate**, and `_lookup_tpu_rate()` returns None with an explanation
rather than falling through to the on-demand SKU. Keep that property — a confident wrong price is worse than
no price.

### Quota does not carry over

**The two control planes meter against entirely different pools**, and holding one buys nothing on the other.
That is the trap the GCE rigs exist to document. On v5e the specifics differ from the v6e twin in a way no
analogy predicts, so read this rather than reasoning across:

| Model | Quota id | Scope |
| :--- | :--- | :--- |
| on-demand, flex-start | `TPU-LITE-PODSLICE-V5-per-project-zone` | **per-zone** |
| spot | `PREEMPTIBLE-TPU-LITE-PODSLICE-V5-per-project-zone` | **per-zone** |

**v5e is symmetrical where v6e is not.** v6e publishes only a preemptible per-zone id — `TPU-V6E-per-project-zone`
does not exist — so `gce-vllm-v6e1-2b` has to fall back to the regional `TPUS-PER-TPU-FAMILY-per-project-region`
for everything non-preemptible. v5 Lite PodSlice publishes both halves per-zone, so this rig never needs that
fallback. **This is the one place the v6e→v5e retarget was not a search-and-replace.**

Verified 2026-08-10: both ids exist on `compute.googleapis.com`, each with a single default bucket reading
**-1 across 130 locations** and no per-zone override on this project. `TPUS-PER-TPU-FAMILY-per-project-region`
holds **no `CT5LP` bucket at all** here (it holds `CT6E`=32 in eight regions, plus `CT3P` and `CT3`).

**A -1 default is not a grant.** It means nobody has ever set this quota on this project, and that reads
identically through `quotas info` whether the real ceiling is unlimited or zero. Do not report it as
"we have v5e quota on Compute Engine".

**This is why `find_tpu` sweeps zones by machine type, not by quota.** Filtering on a quota that is unset
everywhere would admit every zone and prove nothing. `_zones_with_machine_type()` is the Compute Engine
analogue of the TPU API's `accelerator-types list` — and note that for v5e, gate 1 passing is *precisely the
ambiguity* documented at the top of this file, not a green light.

### Zone

`us-west4-a`, inherited from the twin, and the honest reason is that nothing better discriminates:

- It publishes `ct5lp-hightpu-1t` (26 zones do).
- It is the only zone this project has ever provisioned v5e in, and the only one where the TPU API accepted
  **flex-start** for `v5litepod-1` — `europe-west4-a` and `-b` both rejected it at the API (2026-08-04).
- Compute Engine quota cannot discriminate, being unset in all 130 locations.

**Whether the TPU API's zone restriction carries over to Compute Engine is unknown.** It is a second
unverified thing stacked on the first. If a create is rejected here, that alone does not settle the v5e
question — check the error text distinguishes "machine type not supported" from "no capacity in this zone".

### Discovery is a different API, not a different filter

**A `ct5lp-*` instance does not appear in `gcloud compute tpus tpu-vm list` at all.** It is an ordinary
Compute Engine instance that happens to carry a TPU. The sibling rig's `_list_tpu_vm_nodes()` lists TPU VM
nodes and would return nothing here, silently — so the two rigs cannot share a discovery helper, and
`test_a_ct5lp_instance_is_not_a_tpu_vm_node` pins that this rig never calls `tpu-vm`.

**That test only ever covered discovery, and on the v6e rig four tools were violating the rule behind its
back.** `manage_vllm_docker`, `run_vllm_benchmark`, `get_vllm_docker_logs` and `get_tpu_system_logs` all
shelled to `gcloud compute tpus tpu-vm ssh` after the fork, so every one failed with a not-found against an
instance that was plainly RUNNING. They go through **`_ssh_command()`** here, which builds `gcloud compute
ssh` (no `--tunnel-through-iap`; these instances have an external IP). **The lesson generalises: "this rig is
off the TPU API" was asserted about the code as a whole and tested on one function.** Grep for `tpu-vm`
before believing it again — the surviving hits are deliberate `queued-resources list` calls for cross-path
collision detection.

Three field-shape differences fall out, and each fails quietly rather than loudly:

- Status is **`status: RUNNING`**, not `state: READY`. Copying the sibling's check demotes every healthy
  instance to the bottom of the ranking instead of erroring — visible only when two are up.
- The IP is at **`networkInterfaces[].accessConfigs[].natIP`**, not
  `networkEndpoints[].accessConfig.externalIp`. The Makefile targets need the same correction.
- `RUNNING` is a **weaker claim than the QR's `ACTIVE`**. A Queued Resource reached ACTIVE once its node was
  up; an instance is RUNNING the moment the VM boots, long before the startup script has pulled the vLLM
  image or loaded the model. Use `verify_model_health` for readiness, never `check_tpu_availability`.

`_resolve_node_id()` is two lookups here instead of the sibling's three, because the id you ask for is the
name you get — there is no derived `<id>-node` and no class of mismatch to reconcile.

### Cross-path collision matters more here than on the v6e twin

`us-west4-a` is shared by **four** TPU-API v5e rigs — `tpu-vllm-v5e1-2b`, `tpu-jax-v5e1-2b`,
`tpu-pytorch-v5e1-2b`, `tpu-pytorch-v5e1-12b` — plus the two v5e encoding rigs. A chip any of them holds is a
chip this rig cannot get, and they are invisible to `instances list`. That is the entire reason
`list_queued_resources` and `_list_queued_resources_json()` survive on a Compute Engine rig. Nothing here
creates or deletes a Queued Resource.

### Instances are labelled; Queued Resources were not

Every instance this rig creates carries `rig=gce-vllm-v5e1-2b`. An instance name does not encode its owner the
way a QR id did by convention.

**`manage_tpu_instance` is deliberately narrower than the sibling's `manage_queued_resource`**, which deletes
every Queued Resource in the zone that is not the named primary. This one only deletes instances carrying
this rig's label; anything unlabelled or belonging to a sibling is reported and left alone. **Do not "fix"
this to match the sibling.**

`create_tpu_instance` is non-destructive and touches only the name it was given, so `find_tpu`'s zone sweep
is safe. Keep that split.

### What is unchanged

The secret handling, the vLLM flags, and the billing-catalog lookup are identical to the twin, deliberately.
The startup script is not quite — see the Docker gotcha below.

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

**Serving flags live in one place.** `_vllm_serve_flags()` builds the vLLM arg list from `MAX_MODEL_LEN`,
`MAX_NUM_BATCHED_TOKENS`, `LIMIT_MM_PER_PROMPT`, and `TENSOR_PARALLEL_SIZE`; the startup script takes the
same values as placeholders. Don't reintroduce a second hardcoded flag list. The JSON value needs different
quoting inside a single-quoted argument — that's what the `mm_limit` parameter is for.

**The Compute Engine image ships no Docker, and this is a control-plane difference rather than an
oversight.** `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` has no `docker` on PATH at first boot; the TPU API's
`v2-alpha-tpuv5-lite` runtime versions do. On the v6e twin — same image family, so this applies here
unchanged — the inherited startup script died 100 seconds in with five pull retries of `sudo: docker:
command not found`, **while the instance continued to report `status: RUNNING` indefinitely**. That is the
"RUNNING is weaker than ACTIVE" warning arriving as a real failure: nothing distinguishes a dead boot from a
healthy one except `/var/log/vllm-startup.log` or curling `:8000`.

Fixed in three places, because the fact bites at three layers: `startup_script_template.sh` installs
`docker.io` before pulling; `_ENSURE_DOCKER` prefixes every Docker-dependent remote command, so
`manage_vllm_docker start` — precisely what you reach for when a boot failed — does not fail the same way;
and `get_vllm_deployment_config` carries the install in the one-liner it emits. Don't "simplify" any of them
away.

**The boot disk default is 200 GB, and the image default of 10 GB cannot hold the vLLM TPU image.**
Undersizing fails late, after boot, during the docker pull.

**Pin the image FAMILY, never a dated build.** Images ship roughly weekly and every superseded build goes
`DEPRECATED`. There is also an older family spelling, `ubuntu-accelerator-2204-amd64-with-tpu-v5e-v5p-v6e` —
use the `ubuntu-accel-…` form.

**v5e publishes no `<name>-tpu` machine-type twin**, unlike v6e (`ct6e-standard-1t-tpu`) and v5p
(`ct5p-hightpu-1t-tpu`). So there is exactly one string to choose and `MACHINE_TYPE` has no ambiguity to
resolve. **Do not read the missing variant as evidence about the CE path** — an earlier revision of
`@../HARDWARE.md` did exactly that and it was wrong, because the *bare* shapes are the documented creatable
ones.

**A create can time out client-side with the request still live.** `_create_tpu_instance` caps gcloud at
590s and returns `⏳`, not `❌` — the instance may still appear and bill. `find_tpu` treats that as a
capacity outcome and does **not** record it as a zone failure, because gate 3 is never cached.

**`tpu_zones_status.md` is mutable state, not documentation.** `find_tpu` rewrites it in place. It is empty
because nothing has been attempted. **Never seed it from the twin's file** — a zone that rejects a Queued
Resource is not evidence about an instance create, and on this rig that distinction is the whole point.

**Don't destroy an instance unless asked.** Flex-start capacity can take a long time to come back.

**`estimate_deployment_cost` reads live pricing — never reintroduce a rate table.** An earlier hardcoded
table put v5e at $0.12/chip-hr against a $1.20 list rate — wrong by 10x, and undetectable from inside the
code. It queries the Cloud Billing Catalog (Compute Engine service `6F81-5844-456A`, where TPU SKUs live).
**The catalog does not fork with the control plane**: both rigs bill the same SKUs, so flex-start is still
sold as **"DWS Defined Duration V5e"** (dropping the `Tpu` prefix) and spot is `usageType: Preemptible`
spelled `TpuV5e attached to Spot Preemptible VMs`. Patterns are anchored with `^` so the `Reserved …` and
`Commitment v1: …` SKUs don't match. **A published price is not an offer of capacity, and on this rig it is
not even evidence the path exists.**

**On v5e, spot really is the cheapest — but don't generalise it.** us-west4, 2026-08-06: spot $0.5779,
flex-start $0.60, on-demand $1.20. On v6e in us-east5 the order inverts and spot lists *dearer* than
flex-start. Read it out of the tool.

**Raw `/v1/completions` returns an empty completion on `-it` models.** `make query` and
`benchmarking_suite.py` use it, so an empty result there is expected. `server.py` uses `/v1/chat/completions`
throughout — keep new code on the chat endpoint.

**`--tensor-parallel-size` is 1.** v5e-1 is a single chip, and `@../MODELS.md` notes E2B has
`num_key_value_heads=1`, which cannot shard — more chips would multiply its KV cost, not divide it. If you
see `4` anywhere it is copy-paste from a larger topology.

## Names derive from the rig directory

`RIG_NAME` is `os.path.basename(...)` of this directory. `INSTANCE_NAME` defaults to it and is the default
name of every MCP tool; `MCP_SERVER_NAME` defaults to it and names the FastMCP server; the Makefile's
`SERVICE_NAME` is `$(notdir $(CURDIR))`. All resolve to `gce-vllm-v5e1-2b`.

`RESOURCE_ID` is kept as a back-compat alias of `INSTANCE_NAME` because the forked tool signatures spell it
that way. On this path it names an instance.

Nothing reads a *slot* out of the directory — `v5e1` never becomes a gcloud flag. It reads the whole name as
an identifier, which is what keeps sibling rigs off each other's capacity in a shared project and zone.

The MCP server name has to match the key the client registers it under, because that key prefixes every
tool: `mcp__gce-vllm-v5e1-2b__find_tpu`. That is what distinguishes this rig's tools from the TPU-API twin's
`mcp__tpu-vllm-v5e1-2b__…`, and with both loaded the prefix is the *only* thing that does.

`load_dotenv` runs *before* `FastMCP(...)` is constructed, because `MCP_SERVER_NAME` set in `tpu.env` would
otherwise arrive too late to name the server. Don't move the FastMCP construction back above the dotenv
block.

**Renaming the rig directory orphans anything already provisioned** — pin `INSTANCE_NAME` in `tpu.env`
before renaming, or destroy first.

## Silicon facts live at the monorepo root

`@../HARDWARE.md` is canonical for v5e's memory, bandwidth, native numeric formats, gcloud spelling, and the
control-plane table; `@../MODELS.md` for E2B's layer structure, KV cost per token, and weight footprint;
`@../QUANTIZATION.md` for what the JAX path can actually reach. Read them rather than re-deriving, and
correct them there rather than restating a number here.

Three of their facts decide things in this rig:

- **v5e has no native fp8** — int8 is the only low-precision format with a compute win. v7/Ironwood is the
  first TPU that changes this, so quantization conclusions do not carry forward across generations.
- **16 GB HBM per chip**, half the v6e twin's 32. This is why `MAX_MODEL_LEN` stays at 16384 here where the
  v6e GCE rig runs 32768 — do not "align" the two.
- **E2B costs ~18 KiB/token of KV**, measured by difference. The 15 KiB/token figure was retracted.

The twin's measured serving-parameter work — the 0.92 `--gpu-memory-utilization` ceiling (0.95 does not
boot), the 738 s of ~1,000 s to healthy that XLA compilation costs, and the container-local JAX compile
cache — is in `tpu-vllm-v5e1-2b/benchmarks/runs/` and `SERVING-PARAMS.md`. All of it is a property of the
chip, checkpoint and stack rather than the control plane, so it applies here unchanged. **The compile-cache
point is sharper on this path, not softer:** flex-start and spot both delete the instance at
`--max-run-duration`, so every run repays the full compile.

## Benchmarks

`benchmarks/runs/` and `benchmarks/reports/` are **empty on purpose**. This rig was created by copying
`tpu-vllm-v5e1-2b` wholesale, and that copy brought nine run directories and a schema-1.1 report with it;
they were dropped rather than carried, because none was measured on this rig and `benchmarks/rollup.py` globs
`*/benchmarks/` — leaving them would have credited this rig with results it never produced, which is exactly
the failure `rollup.py` was written to expose. The originals are all still in `tpu-vllm-v5e1-2b`.

The same copy brought the twin's dev.to article, demo HTML, cover image and `v5e.md`, all describing Queued
Resource provisioning. Those were dropped too.

`benchmarks/serving-report.schema.json`, `benchmarks/README.md`, and `benchmarks/INDEX.md` are generated or
synced from the monorepo root. Don't hand-edit them here.

Writers (`run_sweep.py`, `run_grid_benchmark.py`, `run_fast_sweep.py`, `benchmarking_suite.py --output`) use
bare filenames in the CWD on purpose. File new output into a new dated run dir.

**When the first sweep runs, it is only comparable to the twin's if the serving config matches.** Check
`tpu.env` against `tpu-vllm-v5e1-2b/tpu.env` before believing a difference is about the provisioning path.

## Auth and env

Requires both `gcloud auth login` and `gcloud auth application-default login` (ADC, for the
`google-cloud-secret-manager` client). `set_env.sh` must be **sourced**, not executed. `init.sh` blocks on
`read` in its error path — don't run it non-interactively.

`tpu.env` is the single source of truth and is committed — never add `*.env` to a `.gitignore`. A real
environment variable always wins over it (`load_dotenv` doesn't overwrite, `mcp-run.sh` exports only unset
keys, the Makefile uses `?=`), so `make status ZONE=...` works for a one-off. Defaults are `us-west4-a` /
`us-west4`.

`.mcp.json` is gitignored at the monorepo root and is **not** present here; regenerate with
`make mcp-config`. `mcp_config.json` is the committed example of the same entry.

## Tests

`test_agent.py` mocks the whole `mcp` module and the Google Cloud clients before importing `server`. Keep
unit tests offline. Because `mcp` is a `MagicMock`, anything calling `mcp.list_tools()` needs an explicit
`AsyncMock` patch — see `test_get_help`.

The `_node()` fixture is deliberately a **different shape** from the twin's (`status`/`RUNNING`,
`networkInterfaces[].accessConfigs[].natIP`). Copying that fixture over would make discovery look tested
while matching nothing.

The billing fixtures in `FAKE_SKUS` are the twin's, verbatim, and that is correct rather than lazy — the
catalog is shared. Their ranking assertion is the reverse of the v6e rig's on purpose.

## Git

The git root is the **parent**, `/home/xbill/gemma4-dev` (`xbill9/gemma4-dev`). `git add .` from here stages
only this subdirectory. Commit straight to `main` — no branches, no PRs. Read `git status` before staging:
the tree routinely carries several unrelated bodies of in-progress work, some already staged.

`AGENTS.md` and `GEMINI.md` in this directory are maintained by different tools and overlap with this file.
`CLAUDE.md` is correct where they disagree.
