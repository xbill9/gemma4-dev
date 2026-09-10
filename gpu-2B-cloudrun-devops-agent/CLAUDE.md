# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An MCP server (`server.py`, `MCPServer` from `mcp>=2` — v1 called it FastMCP) exposing SRE/DevOps tools that talk to a self-hosted
Gemma 4 vLLM service on Cloud Run GPU (NVIDIA L4). Nearly every tool either shells out to
`gcloud` or calls the vLLM endpoint over the OpenAI-compatible API.

## Commands

- `make lint` — `ruff check` + `ruff format --check` + `mypy`. Not wired into `make test`; run it separately.
- CI: no workflow covers this tree yet. The sibling's `../.github/workflows/gpu-4B-cloudrun-devops-agent.yml`
  runs `make install lint test` on **Python 3.10** — target that when adding one here.
  mypy is pinned to `python_version = "3.13"`, so it won't flag 3.11+ syntax or stdlib use — keep code
  3.10-compatible.
- `make test` — `python test_agent.py`. This is **`unittest`**, not pytest (`unittest.IsolatedAsyncioTestCase`).
  Single test: `python -m unittest test_agent.TestDevOpsAgent.test_query_gemma4`.
  Tests mock `subprocess.run` and the OpenAI/httpx clients — they do not need GCP credentials.
- `make run` — starts the MCP server on stdio.
- A `PostToolUse` hook in `.claude/settings.json` runs `ruff format` + `ruff check --fix` on every `.py`
  file edited via Write/Edit. Bulk edits made another way (e.g. `sed`) skip it — run `ruff format .` first.
- `make deploy` / `make destroy` / `make status` / `make endpoint` / `make query PROMPT="..."` — act on the
  live Cloud Run service. `deploy` provisions a real L4 GPU.

## Working in this directory

This directory is one of several sibling agent projects under `gemma4-dev/` (other TPU, GPU, and EC2
variants). They are independent — do not edit siblings unless asked. The git root is the parent, so
`git status` shows their changes too; never stage them.

`GEMINI.md` is a parallel context file for the Gemini CLI. When conventions, config, or the tool list
change here, update `GEMINI.md` too.

Three MCP client configs register this server and each hardcodes project, region, `VLLM_BASE_URL`, and
`MODEL_NAME`: `.mcp.json` (Claude Code), `.agents/mcp_config.json`, and `.codex/config.toml`. Change one,
change all three.

## vLLM configuration

The `deploy-vllm` target in the `Makefile` is the authoritative vLLM config. `GEMINI.md` transcribes
parts of it — when you change the target, update it too, and never edit it as the primary source.
`DEPLOY.md` deliberately does *not* transcribe the command (it points at `make -n deploy`); keep it
that way rather than re-adding a copy that drifts. The `cloudrun_deploy` tool and the
`cloudrun_get_deployment_config` tool in `server.py` build the same flag list and must match the target.
Two more copies are *not* kept in sync: the `config://vllm-deployment-template` resource (it adds
`startup-cpu-boost`) and `cloudrun_get_gpu_deployment_config` (the GKE manifest; it lacks both gemma4
parsers). The image tag `vllm/vllm-openai:v0.26.0-cu129` is hardcoded in ~7 places — grep for it when bumping.

Gemma 4 tool calling requires `--tool-call-parser gemma4` and `--reasoning-parser gemma4` on the vLLM
container; dropping either silently breaks tool use.

## Gotchas

- Most failures are auth, not code. If a tool errors, check gcloud ADC first: `source ./set_adc.sh`.
  Cloud Run here is `--no-allow-unauthenticated`, so calls need `gcloud auth print-identity-token`.
- **Never log to stdout.** Logging is configured to stderr only because stdout is the MCP stdio channel.
  A stray `print()` corrupts the protocol. (`test_logging.py` is a standalone GCP connectivity script,
  not part of `make test` — its `print()`s are fine.)
- `VLLM_BASE_URL` is optional — when unset, `discover_vllm_url()` finds the service via `gcloud`.
  Config is env-driven: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `VLLM_BASE_URL`, `MODEL_NAME`,
  `SERVICE_NAME` (the last sets `DEFAULT_SERVICE_NAME`, so it must name the same service `VLLM_BASE_URL`
  points at — otherwise the pinned URL is reported under a different service's name).
  `server.py` never calls `load_dotenv`, so config must be exported into the shell that launches the
  server — `source ./set_env.sh` does that (it exports only; it no longer writes a `.env`). Its
  defaults mirror the `Makefile`, so update both together. But the `Makefile` reads `PROJECT_ID`/`REGION`,
  not `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` (only `SERVICE_NAME` is shared), so `set_env.sh` does
  not retarget `make deploy`/`status` — pass `make deploy PROJECT_ID=... REGION=...`.
- `BUCKET_NAME` is always `{PROJECT_ID}-bucket` in `server.py`; it is not read from the environment.
- `make query PROMPT=...` pastes the prompt into JSON unescaped — a double quote in it breaks the request.

## Code conventions in `server.py`

- Tools are `@mcp.tool()` functions; the docstring is what the model sees — write it for the caller.
- Every tool name is prefixed `cloudrun_` (e.g. `cloudrun_deploy`, `cloudrun_get_system_status`) so it
  does not collide with the sibling TPU agent's tools. Don't repeat "cloudrun" later in the name.
- New code must run blocking Google Cloud SDK / `subprocess` calls through `await asyncio.to_thread(...)`.
  Existing tools don't all do this (`cloudrun_analyze_cloud_logging`, `cloudrun_get_system_status`) — don't
  copy them as examples.
- Use `get_vllm_client()` (handles URL discovery + identity token) rather than constructing `AsyncOpenAI`.
  Use `get_active_model_name(client)` rather than trusting `MODEL_NAME`.
- Service URLs are memoized in `_VLLM_URL_CACHE`. Any tool that creates, replaces, or deletes a service
  must call `invalidate_vllm_url_cache(service_name)` — `cloudrun_deploy` and `cloudrun_destroy` do, and
  `test_agent.py` asserts it.
- New tools return human-readable strings prefixed with a status emoji (✅ / 🔴 / ⚠️), and catch their own
  exceptions into an error string instead of raising. Many older tools predate this.
- ruff: line-length 120, rules `E,F,B,I` with `E501` ignored — width is enforced by `ruff format`, not
  `ruff check`. mypy runs with `check_untyped_defs`.

## Tests

Every new tool must be added to the `EXPECTED_TOOLS` set in `test_agent.py`, which
`test_tools_registered` compares against the live registry with `assertEqual` — adding, removing, or
renaming a tool without updating the set fails the test. `server.py` also exposes an `@mcp.resource`
(`config://vllm-deployment-template`); add new resources to `test_resources_registered` too, though it
checks with `assertIn`, so a missing entry isn't caught.

`EXPECTED_TOOLS` is the only one of these the test suite enforces. A tool name is also hand-listed in
three unchecked places that silently drift: the `cloudrun_get_help` tool body in `server.py`, the
"Available Tools" section of `README.md`, and the capabilities list in `GEMINI.md`. Update all four.
