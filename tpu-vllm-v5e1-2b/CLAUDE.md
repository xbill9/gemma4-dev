# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file MCP server (`server.py`, FastMCP) that acts as a devops agent for serving Gemma 4
(`google/gemma-4-E2B-it`) with vLLM on a Google Cloud TPU v5e-1 Flex-start Queued Resource. Its tools shell
out to `gcloud` and talk HTTP to the vLLM OpenAI-compatible endpoint on port 8000. This rig is used for
**live demos** — prefer changes that keep the demo working over broad refactors.

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

**Endpoint discovery is dynamic.** `discover_vllm_url()` lists queued resources in `ZONE`, takes the first
`ACTIVE` one, resolves its node and external IP, and builds `http://{ip}:8000`. Never hardcode an endpoint —
the IP changes every time the Queued Resource is recreated. Use the `get_vllm_endpoint` tool.

**The Makefile's TPU targets are a separate, hand-provisioned path.** `make endpoint` / `status` / `benchmark`
/ `query` all `describe` a tpu-vm named `$(SERVICE_NAME)` = `tpu-2B-v5e1-devops-agent`. Every MCP tool
defaults to `resource_id="vllm-gemma4-qr"`, whose node is `vllm-gemma4-qr-node`. The names don't match, so
those targets will not find an agent-provisioned Queued Resource. Go through the MCP tools for anything the
agent deployed.

**Raw `/v1/completions` returns an empty completion on `-it` models.** `make query` and
`benchmarking_suite.py` use it, so an empty result there is expected, not a broken deploy. `server.py`
correctly uses `/v1/chat/completions` throughout — keep new code on the chat endpoint; raw completions are
only useful for prefill-only benchmarks.

**The comparison/plot scripts still carry v6e labels.** This rig moved from v6e-1 to v5e-1, but
`compare_chips.py`, `compare_benchmarks.py`, and `plot_grid.py` were copied over unchanged — they hardcode
"v6e-4"/"v6e-1" titles and read CSVs out of sibling `../tpu-*-v6e*-devops-agent/` directories.
`benchmark_tables.md` is likewise a v6e-era report. Don't read those labels as describing this rig.

**Flex-start v5litepod-1 is only accepted in `us-west4-a`.** Verified 2026-08-04 by attempting creation:
`europe-west4-a` and `europe-west4-b` both reject it at the API with `FLEX_START provisioning model is not
supported for accelerator type "v5litepod-1" in location "..."`; `us-west4-a` accepts. Non-zero quota in a
zone (all 44 have it) says nothing about this — the provisioning model is the blocker, not capacity. So the
default `ZONE` of `europe-west4-a` cannot ever provision this rig; export `GOOGLE_CLOUD_ZONE=us-west4-a`.
The skill's reference guide lists `europe-west4-b` as flex-start-capable for v5e, but its example uses
`v5litepod-4` — the single-chip shape is narrower than the table suggests.

**`tpu.env` is the single source of truth for deployment parameters.** Project, region, zone, model,
accelerator type, and tensor-parallel size are defined once there and consumed by `server.py` (via
`load_dotenv`), `mcp-run.sh` (which is what the `mcp_config.json` files launch), the `Makefile` (via
`-include`), and `set_env.sh`. Change the zone there, not in five places. A real environment variable always
beats the file in all four consumers — `load_dotenv` doesn't overwrite, the wrapper only exports what's unset,
and the Makefile uses `?=` — so `make status ZONE=...` still works for a one-off. Defaults are `us-west4-a` /
`us-west4`. Still check what is actually running before assuming: `list_queued_resources` and
`discover_vllm_url` only look in the configured zone.

**`--tensor-parallel-size` is 1.** v5e-1 is a single chip. If you see `4` anywhere, it's copy-paste from a
larger topology.

**v5e is spelled `v5litepod` to gcloud.** The accelerator type is `v5litepod-1` (not `v5e-1`), the Flex-start
runtime is `v2-alpha-tpuv5-lite`, and `make deploy-tpu` passes `--type=v5litepod --topology=1x1`. All three
live in one place each — `ACCELERATOR_TYPE` / `TPU_RUNTIME_VERSION` in `server.py`, the Makefile flags — and
all are env-overridable. "v5e-1" is fine in prose; never put it in a gcloud argument.

**Don't destroy a queued resource unless asked.** Teardown is not part of routine debugging, and Flex-start
capacity can take up to 2 hours to come back.

## Auth and env

Requires both `gcloud auth login` (for the `gcloud` subprocess calls) and `gcloud auth application-default
login` (ADC, for the `google-cloud-secret-manager` client). `set_env.sh` must be **sourced**, not executed.
`init.sh` is a one-time bootstrap that blocks on `read` in its error path — don't run it non-interactively.

Env vars `server.py` reads: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_ZONE`, `GOOGLE_CLOUD_REGION`, `MODEL_NAME`,
`ACCELERATOR_TYPE`, `TPU_RUNTIME_VERSION`, `TPU_QUOTA_ID`, `TENSOR_PARALLEL_SIZE`, `LOCAL_DOCKER_IMAGE`,
`MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `LIMIT_MM_PER_PROMPT`, `TPU_NETWORK`, `TPU_SUBNETWORK`. The HF
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

Committed benchmark artifacts (`*.png` plots, `benchmark_results.*`, `grid_benchmark_results.csv`) are
intentionally tracked. Don't regenerate or delete them unless asked.

`AGENTS.md` in this directory is maintained by a different tool and overlaps with this file — if you change a
convention here, check whether it needs the same change there. It has already drifted on two points: it claims
`ZONE`/`REGION` are hardcoded in `server.py` (they read the environment, `server.py:26-27`) and that
`get_help()` is hand-maintained (it is generated from `mcp.list_tools()`). This file is correct on both.
