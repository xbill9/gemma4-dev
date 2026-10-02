# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file MCP server (`server.py`, MCPServer) that acts as a devops agent for serving Gemma 4
(`xbill9/gemma-4-E2B-it-qat-w8a8-int8`) with vLLM on a Google Cloud TPU v5e-4 (four chips, one host) Flex-start Queued Resource. Its tools shell
out to `gcloud` and talk HTTP to the vLLM OpenAI-compatible endpoint on port 8000.

The checkpoint is int8 W8A8 built from Google's QAT weights (`../jev-tpu-31b/w8a8_from_qat.py`), text only,
and **it serves only on a patched image**: `patches/` (applied in `patches/ORDER`) adds int8 W8A8 to
`tpu_inference`'s JAX compressed-tensors path and a text-only `text_config` fallback. `server.py` embeds
the patches in the rendered startup script as a base64 tar.gz; the VM pulls the pinned
`VLLM_BASE_IMAGE`, patches it, and serves `VLLM_SERVE_IMAGE` (`vllm-tpu-w8a8:patched`). Change the
image digest and the patches together, never one alone. Every path that starts a container
(`manage_vllm_docker`, `make run-container`) must use the patched tag.

This rig is the four-chip fork of `../tpu-vllm-v5e1-2b-w8a8`. The patches are byte-identical to the
sibling's; the int8 W8A8 linears serve at `--tensor-parallel-size 4` (`../QUANTIZATION.md`). The serving settings (`GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`,
`MAX_NUM_BATCHED_TOKENS`, `VLLM_ENV`, the image digest) are kept identical to the sibling on purpose, so
the two rigs differ only in chip count and TP.

**Measured 2026-10-02 (`benchmarks/runs/2026-10-02-rig-boot-v5e4`).** Flex-start `v5litepod-4` in `us-west4-a`, created through this rig's own
`create_tpu_queued_resource`, got capacity within 5 minutes and served about 13 minutes after creation.
Text, greedy chat and a tool call verified. Weights per chip 1.75 GiB (v5e-1: 6.88), KV pool 631,968 tokens
(v5e-1: 333,312, 1.90x). Output tok/s at 1 / 4 / 16 requests: 190 / 644 / 1,465, which is 0.87x / 0.77x /
0.51x the v5e-1 sweep (220 / 841 / 2,872). **For E2B this rig is a worse server than its v5e-1 sibling:**
four chips buy context, and every concurrency measured runs slower.

**Why the pool nearly doubled.** E2B has one KV head (`num_key_value_heads=1`), and a single KV head does not
shard (`../MODELS.md`). At TP=4 every chip still holds a copy of the whole KV cache, so KV per token per chip
stays 18 KiB; the pool grows by the weight room each chip frees, and with weights per chip down from 6.88 to
1.75 GiB that is nearly a second pool. Query heads split 8 -> 2 per chip.

## Commands

```
make install    # pip install -r requirements.txt
make run        # python server.py (stdio MCP server)
make test       # python test_agent.py — unittest, NOT pytest
make lint       # ruff check . && ruff format --check . && mypy .
make format     # apply ruff formatting and autofixes
make tools      # regenerate GemmaTools.md from the @mcp.tool() decorators
make benchmark  # discovers the TPU IP, then runs benchmarking_suite.py against it
make query PROMPT="..."
```

`make lint` only *checks* formatting — `make format` is what writes it. Both `make lint` and `make test`
currently pass clean; keep them that way.
A `PostToolUse` hook in `.claude/settings.json` already runs `ruff format` on every `.py` file Claude edits.

## Style

- ruff is both linter and formatter; no black. `line-length = 120`, but `E501` is in the ignore list, so the
  formatter enforces width and the linter does not.
- Lint rules are `E, F, B, I` — import sorting comes from ruff's `I`, not a separate isort.
- mypy is deliberately non-strict: `check_untyped_defs = true` but `attr-defined` is globally disabled.
- Python 3.13 is the minimum; ruff targets `py313` and mypy runs at `python_version = "3.13"`.
- Existing code uses `Optional[str]` from `typing` rather than `X | None`. The target no longer requires this
  — it's now just consistency with the surrounding code, so match what's already in the file you're editing.
- Every subprocess call goes through `run_command(cmd: list[str])` — list args via
  `asyncio.create_subprocess_exec`, never `shell=True`. Keep it that way.
- MCP tools are `async def` and return markdown strings with emoji status prefixes (`✅`, `❌`, `📡`).

## Tool catalog is generated — don't hand-edit it

`GemmaTools.md` and the `get_help` tool both build their tool list from `mcp.list_tools()`, so they cannot
drift from the `@mcp.tool()` decorators. After adding or removing a tool, run `make tools` to refresh the
doc. `README.md` intentionally lists only a handful of highlights and points at `GemmaTools.md` for the rest.

Source of truth either way: `grep -n "^@mcp.tool" server.py`.

## Gotchas

**`startup_script_template.sh` is consumed by `str.format()`.** Placeholders are `{project_id}`, `{zone}`,
`{model_name}`, `{hf_secret_id}`, `{tensor_parallel_size}`, `{max_model_len}`, `{max_num_batched_tokens}`,
`{limit_mm_per_prompt}`. Any other literal `{` or `}` added to that bash file — a shell brace expansion, a
`${VAR}`, a JSON literal — raises at format time and breaks the deploy. Escape as `{{` / `}}`.

**The startup script fetches the HF token itself; never add a `{hf_token}` placeholder back.** The rendered
script is uploaded as instance metadata, so a baked-in token would be readable from the instance. It reads
`hf-token` from Secret Manager at boot via the metadata server, retrying for 30 minutes so an IAM grant
applied after creation still lands. The VM's service account needs
`roles/secretmanager.secretAccessor` on the secret. Tracing (`set -x`) is off across the whole token section —
keep it that way, and never interpolate `$HF_TOKEN` into a logged string.

**Serving flags live in one place.** `_vllm_serve_flags()` builds the vLLM arg list from `MAX_MODEL_LEN`,
`MAX_NUM_BATCHED_TOKENS`, `LIMIT_MM_PER_PROMPT`, and `TENSOR_PARALLEL_SIZE`; the startup script takes the same
values as placeholders. Both deploy paths and the generated one-liner therefore agree. Don't reintroduce a
second hardcoded flag list. Note the JSON value needs different quoting inside a single-quoted argument —
that's what the `mm_limit` parameter is for.

**`create_tpu_queued_resource` is non-destructive; `manage_queued_resource` is not.** The latter deletes every
Queued Resource in the zone that isn't the named primary. `create_tpu_queued_resource` touches only the id it
was given, so `find_tpu`'s zone sweep is safe. Keep that split.

**`tpu_zones_status.md` is mutable state, not documentation.** `find_tpu` rewrites it in place to record which
zones have failed, and reads it back to skip known-bad zones. Do not hand-edit it as if it were docs.

**Endpoint discovery is dynamic, and provisioning-path agnostic.** `_discover_vllm_node()` lists **TPU VM
nodes** in `ZONE` — not queued resources — because both provisioning paths end at a node in that same
namespace: a Queued Resource creates one indirectly, and `make deploy-tpu-spot` / any hand-provisioned VM
creates one with no QR behind it at all. It ranks candidates (this rig's names first, then `READY`, then
QR-backed), probes each on `/v1/models`, and returns the first that answers, as
`VllmNode(name, url, serving)`; `discover_vllm_url()` is the thin wrapper. Never hardcode an endpoint — the
IP changes every time the node is recreated. Use the `get_vllm_endpoint` tool.

Two rules fall out of sibling rigs sharing this zone: a node of **ours** that is up but not yet answering is
returned with `serving=False` so callers can poll it while vLLM boots, and a node that is **not** ours is
never returned unless a probe confirmed it is serving. That second rule is stricter than the old code, which
would hand back the first `ACTIVE` QR in the zone even if it belonged to the jax or pytorch rig.

Until 2026-08-06 discovery listed queued resources only, so a healthy hand-provisioned spot VM reported as
"No ACTIVE Queued Resource found" and every query tool refused to run against a TPU that was serving fine.

**`_resolve_node_id()` is the matching fix for the SSH-based tools** (`manage_vllm_docker`,
`run_vllm_benchmark`, `get_vllm_docker_logs`, `get_tpu_system_logs`). It tries the queued resource's node,
then a TPU VM named exactly `resource_id` or `<resource_id>-node`, then — last resort — the node that
discovery confirmed is serving vLLM, so the default `resource_id` still reaches a deployment whose node was
named by another convention. `_get_node_id()` remains the QR-only primitive; don't call it directly from a
tool.

**Resource names are derived from the rig directory.** `RIG_NAME` in `server.py` is
`os.path.basename(...)` of the rig directory; `RESOURCE_ID` defaults to it and is the default `resource_id`
of every MCP tool, and the Makefile's `SERVICE_NAME` is `$(notdir $(CURDIR))`. So in this directory both
resolve to `tpu-vllm-v5e4-2b-w8a8`. `RIG_NAME` also supplies the default `MCP_SERVER_NAME`, which names the
MCP server (`MCPServer`). Nothing here reads a *slot* out of
the directory — that is still forbidden (`v5e4` never becomes a gcloud flag); it reads the whole name as an
identifier, which is what keeps sibling rigs off each other's capacity in a shared project and zone.

The derivation is a default, not a lock: `RESOURCE_ID` in `tpu.env` (or the environment), `MCP_SERVER_NAME`
in either, and `SERVICE_NAME`
on the make command line all win. **Renaming the rig directory orphans anything already provisioned** —
the tools will look for the new name and the old resource keeps billing. Pin the old name in `tpu.env`
before renaming, or destroy first.

**The MCP server name has to match the key the client registers it under**, because that key is what
prefixes every tool: `mcp__tpu-vllm-v5e4-2b-w8a8__find_tpu`. All six rigs used to register as `tpu-devops`, so a
tool call was ambiguous whenever more than one was loaded, and a user-scope `tpu-devops` shadowed this rig
entirely (it has no committed `.mcp.json`). `mcp_config.json` is the committed example of the entry;
`make mcp-config` writes a real `.mcp.json` using `MCP_SERVER_NAME` (default `$(notdir $(CURDIR))`), merging
into any existing file rather than replacing it. `.mcp.json` is gitignored at the monorepo root.

Note the ordering constraint in `server.py`: `load_dotenv` now runs *before* `MCPServer(...)` is constructed,
because `MCP_SERVER_NAME` set in `tpu.env` would otherwise arrive too late to name the server. Don't move
the MCPServer construction back above the dotenv block.

**The Makefile's TPU targets are still a separate, hand-provisioned path.** `make endpoint` / `status` /
`benchmark` / `query` all `describe` a tpu-vm named `$(SERVICE_NAME)` = `tpu-vllm-v5e4-2b-w8a8`, while the MCP
tools manage a Queued Resource of the same name whose *node* is `tpu-vllm-v5e4-2b-w8a8-node`. Those are two
different TPU nodes, so the make targets still will not find an agent-provisioned Queued Resource — the names
are now merely related instead of unrelated. Go through the MCP tools for anything the agent deployed.

Older node names: none. This rig is new, so nothing was provisioned under an earlier convention.

**Raw `/v1/completions` returns an empty completion on `-it` models.** `make query` and
`benchmarking_suite.py` use it, so an empty result there is expected, not a broken deploy. `server.py`
correctly uses `/v1/chat/completions` throughout — keep new code on the chat endpoint; raw completions are
only useful for prefill-only benchmarks.

**The comparison/plot scripts still carry v6e and v5e-1 labels.** They came through the fork chain from the
v6e rigs and the v5e-1 sibling: `compare_chips.py`, `compare_benchmarks.py`, and `plot_grid.py` were copied over unchanged — they hardcode
"v6e-4"/"v6e-1" titles and read CSVs out of sibling `../tpu-*-v6e*-devops-agent/` directories.
`plot_sweep_v5e1.py` is titled for the v5e-1 sibling. Don't read those labels as describing this rig.

**Flex-start `v5litepod-4` is accepted in `us-west4-a`** (2026-10-02, capacity in under 5 minutes,
`benchmarks/runs/2026-10-02-rig-boot-v5e4`); the other zones are untried. `v5litepod-4` is listed in `us-west4-a` (and
`us-west1-c`, `us-east5-b`, `us-central1-a`, `us-south1-a`, `europe-west4-b`, `us-east1-c`), and the quota
`TPUV5sLitepodPerProjectPerZoneForTPUAPI` is 512 chips per zone in 44 zones (spot 1536), so quota does not
bind at four chips (checked 2026-10-02). The zone finding below belongs to the v5e-1 sibling's
`v5litepod-1`.

Sibling finding — flex-start `v5litepod-1` is only accepted in `us-west4-a`. Verified 2026-08-04 by attempting creation:
`europe-west4-a` and `europe-west4-b` both reject it at the API with `FLEX_START provisioning model is not
supported for accelerator type "v5litepod-1" in location "..."`; `us-west4-a` accepts. Non-zero quota in a
zone (all 44 have it) says nothing about this — the provisioning model is the blocker, not capacity.
The skill's reference guide lists `europe-west4-b` as flex-start-capable for v5e, and its example uses
`v5litepod-4`, this rig's shape; that is the only evidence for `europe-west4-b` here, and it is unverified.

**The Queued Resource path takes three provisioning models.** `_provisioning_flags()` in `server.py` is the
one place that maps `flex-start` / `spot` / `on-demand` to gcloud flags; every creation tool
(`create_tpu_queued_resource`, `manage_queued_resource`, `find_tpu`) takes a `provisioning_model` argument
defaulting to `PROVISIONING_MODEL` in `tpu.env`. Two things do not generalize across them:

- **Only flex-start passes `--max-run-duration`** — gcloud documents that flag as flex-start-only. A spot or
  on-demand node has no automatic stop and bills until it is preempted or destroyed. `--valid-until-duration`
  bounds the *request*, not the run, so it is shared by all three.
- **Spot is metered by a different quota**, `TPUV5sPreemptibleLitepodPerProjectPerZoneForTPUAPI`
  (`TPU_SPOT_QUOTA_ID`), not `TPU_QUOTA_ID`. `find_tpu` and `get_zones_with_available_quota` pick the id from
  the provisioning model, so don't pass `quota_id` explicitly unless you mean to override that.

`tpu_zones_status.md` rows now carry a `[model]` prefix in the detail column and `find_tpu` only skips a zone
whose recorded failure was under the *same* model — a zone that rejects flex-start is not evidence about spot,
which is the whole reason spot exists here. Untagged rows predate this and read as flex-start.

The Makefile's separate hand-provisioned `tpu-vm` path has its own knob, `TPU_PROVISIONING_MODEL`
(`standard` | `spot` | `reservation-bound` — gcloud's vocabulary there, with no flex-start), surfaced as
`make deploy-tpu-spot` / `make deploy-tpu-ondemand`. Don't confuse the two spellings.

**`estimate_deployment_cost` reads live pricing — never reintroduce a rate table.** It queries the Cloud
Billing Catalog API (Compute Engine service `6F81-5844-456A`, where TPU SKUs live), matching on region,
`usageType`, and a description pattern per provisioning model. The previous hardcoded table said v5e was
$0.12/chip-hr against a $1.20 list rate — wrong by 10x, and undetectable from inside the code. If no SKU
matches, the tool says so rather than falling back to a guess; keep that property. Requires a working
`gcloud auth print-access-token` and the Cloud Billing API enabled.

Two naming traps in the catalog: flex-start is sold as **"DWS Defined Duration"** (Dynamic Workload
Scheduler) and drops the `Tpu` prefix (`DWS Defined Duration V5e`), and spot is `usageType: Preemptible`
spelled `TpuV5e attached to Spot Preemptible VMs`. The `Reserved …` and `Commitment v1: …` SKUs describe the
same chip in the same region and two of them are also `OnDemand`, so the patterns are anchored with `^`.
A price existing does not mean capacity is obtainable — `europe-west4` quotes a flex-start v5e rate while
still rejecting `v5litepod-1` (the sibling's shape) at the API.

**`tpu.env` is the single source of truth for deployment parameters.** Project, region, zone, model,
accelerator type, and tensor-parallel size are defined once there and consumed by `server.py` (via
`load_dotenv`), `mcp-run.sh` (which is what the `mcp_config.json` files launch), the `Makefile` (via
`-include`), and `set_env.sh`. Change the zone there, not in five places. A real environment variable always
beats the file in all four consumers — `load_dotenv` doesn't overwrite, the wrapper only exports what's unset,
and the Makefile uses `?=` — so `make status ZONE=...` still works for a one-off. Defaults are `us-west4-a` /
`us-west4`. Still check what is actually running before assuming: `list_queued_resources` and
`discover_vllm_url` only look in the configured zone.

**`--tensor-parallel-size` is 4.** v5e-4 is four chips on one host (`ct5lp-hightpu-4t`: 112 vCPU, 192 GiB
RAM). If you see `1` as a TP size or `1x1` as a topology anywhere, it is copy-paste from the v5e-1 sibling.

**v5e is spelled `v5litepod` to gcloud.** The accelerator type is `v5litepod-4` (not `v5e-4`), the Flex-start
runtime is `v2-alpha-tpuv5-lite` (the same as v5e-1), and `make deploy-tpu` passes `--type=v5litepod --topology=2x2`. All three
live in one place each — `ACCELERATOR_TYPE` / `TPU_RUNTIME_VERSION` in `server.py`, the Makefile flags — and
all are env-overridable. "v5e-4" is fine in prose; never put it in a gcloud argument.

**Don't destroy a queued resource unless asked.** Teardown is not part of routine debugging, and Flex-start
capacity can take up to 2 hours to come back.

## Auth and env

Requires both `gcloud auth login` (for the `gcloud` subprocess calls) and `gcloud auth application-default
login` (ADC, for the `google-cloud-secret-manager` client). `set_env.sh` must be **sourced**, not executed.
`init.sh` is a one-time bootstrap that blocks on `read` in its error path — don't run it non-interactively.

Env vars `server.py` reads: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_ZONE`, `GOOGLE_CLOUD_REGION`, `MODEL_NAME`,
`ACCELERATOR_TYPE`, `TPU_RUNTIME_VERSION`, `TPU_QUOTA_ID`, `TENSOR_PARALLEL_SIZE`, `LOCAL_DOCKER_IMAGE`,
`MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `LIMIT_MM_PER_PROMPT`, `TPU_NETWORK`, `TPU_SUBNETWORK`,
`RESOURCE_ID` and `MCP_SERVER_NAME` (both default to the rig directory name). The HF
token lives in GCP Secret Manager under the secret id `hf-token` — never log, return, or commit it.

`TPU_NETWORK` / `TPU_SUBNETWORK` default to empty, which means gcloud uses the project's default network.
`aisprint-491218` has only the auto-mode `default` network — it has no custom VPC. Setting these to a network
that doesn't exist fails creation in every zone, which is what a screenful of failed zones in
`tpu_zones_status.md` usually means.

## Tests

`test_agent.py` mocks the whole `mcp` module and the Google Cloud clients before importing `server`. Keep unit
tests offline: mock the cloud, subprocess, and network boundaries rather than reaching out. Because `mcp` is a
`MagicMock`, anything calling `mcp.list_tools()` needs an explicit `AsyncMock` patch — see `test_get_help`.

## Git

The git root is the **parent** directory, `/home/xbill/gemma4-dev` (`xbill9/gemma4-dev`) — this rig is one
subdirectory of that monorepo, alongside the `tpu-jax-*` and `tpu-pytorch-*` rigs. `git add .` from here
stages only this subdirectory; run git commands from the repo root when you mean the whole tree.

This rig was forked out of `/home/xbill/gemma4-queens`, which is still a separate repo with the older
`-devops-agent` naming. Nothing here is shared with it any more — don't look for this project's history
there.

No benchmark artifacts were carried over from the fork. This rig's own measurement is `benchmarks/runs/2026-10-02-rig-boot-v5e4`. The v5e-1
sibling's measurements (bf16, W4A16 and two W8A8 builds of E2B on one v5e chip, 2026-09-29) were made with
the `../jev-tpu-v5e1` runner and live in `../jev-tpu-v5e1/results/`; README.md quotes them as the baseline. New runs from this rig go
under `benchmarks/runs/<date>-<what>-<hw>/` per the root `CLAUDE.md`.

`AGENTS.md` in this directory is maintained by a different tool and overlaps with this file — if you change a
convention here, check whether it needs the same change there. It has already drifted on two points: it claims
`ZONE`/`REGION` are hardcoded in `server.py` (they read the environment, `server.py:26-27`) and that
`get_help()` is hand-maintained (it is generated from `mcp.list_tools()`). This file is correct on both.
