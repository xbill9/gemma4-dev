# AGENTS.md

Guidance for Codex and other coding agents working in this directory.

## Project overview

This is a live-demo DevOps/SRE agent for serving `google/gemma-4-E2B-it` with vLLM on a Google Cloud TPU
v5e-1 Flex-start Queued Resource. The main application is a single-file FastMCP server in `server.py`. Its
tools invoke `gcloud`, inspect Google Cloud resources and logs, manage the remote vLLM container, and call the
OpenAI-compatible inference API on port 8000.

Prefer small, reliable changes that keep the demo working over broad refactors. Treat cloud state as live and
potentially costly.

## Common commands

```bash
make install                         # pip install -r requirements.txt
make run                             # run the stdio MCP server
make test                            # python test_agent.py (unittest, not pytest)
make lint                            # ruff check, format check, and mypy
ruff format .                        # apply formatting
make benchmark                       # discover a TPU endpoint and run the benchmark suite
make query PROMPT="Your question"    # query the deployed model
```

Run the narrowest useful check while developing, then run `make test` and `make lint` when the change warrants
the full suite. Tests mock FastMCP and Google Cloud dependencies before importing `server.py`; keep unit tests
offline and mock cloud, subprocess, and network boundaries.

## Source of truth

- Use `server.py` as the source of truth for MCP tools and runtime configuration. The tool inventories in
  `README.md`, `GEMINI.md`, `GemmaTools.md`, the `Makefile`, and the hardcoded `get_help()` text can be stale.
- To inspect the registered tools, use `rg -n '^@mcp\.tool' server.py` and read the decorated functions.
- If the tool set changes, review and update `get_help()` and relevant documentation explicitly; it is not
  generated.
- Do not trust IP addresses or ONLINE status recorded in markdown or scripts. Discover the current endpoint and
  verify live state before making claims or running operations.
- Deployment parameters (project, region, zone, model, accelerator type, tensor-parallel size) live in
  `tpu.env` and are read by `server.py`, `mcp-run.sh`, the `Makefile`, and `set_env.sh`. Edit that file rather
  than any individual consumer. Environment variables override it everywhere.

## Code conventions

- Minimum and target Python is 3.13. Ruff is the formatter and linter; do not introduce Black or a separate
  isort setup.
- Ruff uses a 120-character formatter width and lint rules `E`, `F`, `B`, and `I`; `E501` is intentionally
  ignored. Mypy is deliberately non-strict with `check_untyped_defs = true` and `attr-defined` disabled.
- Follow the existing type style, including `Optional[str]` rather than `str | None`.
- Route subprocesses through `run_command(cmd: list[str])`, which uses `asyncio.create_subprocess_exec`. Pass
  argument lists and never add `shell=True` or interpolate untrusted values into shell commands.
- MCP tools are `async def` functions and generally return user-facing Markdown strings with status prefixes
  such as `✅`, `❌`, and `📡`. Preserve that interface when editing existing tools.
- Keep cloud and HTTP work async. Reuse the endpoint-discovery and client helpers instead of copying discovery
  logic or hardcoding an address.
- Never log, commit, or return Hugging Face tokens or other credentials. The HF token is stored in Secret
  Manager under `hf-token`.

## Deployment invariants and hazards

- A v5e-1 is a single chip, so the correct default tensor parallel size is `1`. Older documentation and some
  Makefile examples incorrectly show `4`.
- gcloud calls v5e `v5litepod`. The accelerator type is `v5litepod-1`, the Flex-start runtime version is
  `v2-alpha-tpuv5-lite`, and `--type=v5litepod --topology=1x1` is the tpu-vm form. Use "v5e-1" in prose only,
  never as a gcloud argument.
- `discover_vllm_url()` dynamically finds the first ACTIVE queued resource in the configured zone, resolves
  its node and external IP, and constructs `http://{ip}:8000`. Use it instead of stale endpoints.
- `startup_script_template.sh` is rendered with Python `str.format()`. Its supported placeholders are
  `{project_id}`, `{zone}`, `{model_name}`, `{hf_secret_id}`, `{tensor_parallel_size}`, `{max_model_len}`,
  `{max_num_batched_tokens}`, and `{limit_mm_per_prompt}`. Escape every other literal brace as `{{` or `}}`,
  including shell `${VAR}` and JSON braces, or deployment rendering will fail.
- There is deliberately no `{hf_token}` placeholder. The rendered script is uploaded as instance metadata, so
  it fetches `hf-token` from Secret Manager at boot using the VM's own credentials instead. The VM service
  account needs `roles/secretmanager.secretAccessor` on that secret.
- `create_tpu_queued_resource` is non-destructive and touches only the resource id it is given.
  `manage_queued_resource` deletes every other Queued Resource in the zone — treat it as destructive.
- `tpu_zones_status.md` is mutable program state. `find_tpu` rewrites and reads it to track failed zones; do not
  treat it as ordinary documentation or hand-edit it casually.
- `set_env.sh` must be sourced. `init.sh` may block on interactive input in its error path, so do not run it in
  unattended workflows.
- Authentication may require both `gcloud auth login` for CLI subprocesses and
  `gcloud auth application-default login` for Google client libraries.
- Environment variables actually consumed at startup include `GOOGLE_CLOUD_PROJECT`, `MODEL_NAME`,
  `ACCELERATOR_TYPE`, `TPU_RUNTIME_VERSION`, `TPU_QUOTA_ID`, `TENSOR_PARALLEL_SIZE`, and `LOCAL_DOCKER_IMAGE`.
  Confirm the source before documenting additional variables.

## Cloud safety

- Never destroy a Queued Resource, TPU node, VM, container, reservation, secret, or other cloud resource unless
  the user explicitly asks for that destructive action.
- Do not run `make destroy`, `make destroy-tpu`, or destructive MCP tools as routine cleanup or debugging.
- Before any requested destructive or costly operation, inspect the active project, zone, resource name, and
  current state. State the exact target and avoid broad cleanup.
- Prefer read-only status, describe, logs, metrics, and health checks during diagnosis. Creating resources,
  restarting production services, benchmarking live capacity, and changing IAM or secrets can have cost or
  availability impact and should stay within the user's explicit scope.
- Do not assume a queued request is abandoned because it is waiting; Flex-start allocation can remain queued
  for a substantial period.

## Repository hygiene

- The Git root is the parent directory, `/home/xbill/gemma4-dev` (`xbill9/gemma4-dev`); this rig is a
  subdirectory of that monorepo alongside the other Gemma 4 serving rigs. Run Git commands from the root when
  repository-wide scope is intended.
- Preserve unrelated user changes and untracked files. Do not reset, overwrite, or clean them.
- Generated benchmark JSON, CSV, Markdown, and PNG plots are committed artifacts in this project. Do not
  regenerate or delete them unless the task calls for it.
- Avoid drive-by edits to stale documentation. If a change exposes a concrete discrepancy, update only the
  affected documentation and clearly distinguish live-discovered state from examples.
