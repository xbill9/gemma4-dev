# AGENTS.md

Guidance for Codex and other coding agents working in this directory.

## Project overview

This is a DevOps/SRE agent for serving `google/gemma-4-E2B-it` with vLLM on one Google Cloud TPU v5e-1 chip,
provisioned as a **Compute Engine instance** (`gcloud compute instances create --machine-type=ct5lp-hightpu-1t`)
rather than a Cloud TPU API Queued Resource. It is the A/B twin of `tpu-vllm-v5e1-2b`, which is the live-demo
rig and stays on the deprecated control plane. The main application is a single-file FastMCP server in
`server.py`. Its
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
- **Whether v5e can be provisioned through Compute Engine at all is an open question.** Google documents v5e
  as GKE and Cloud TPU API only; `ct5lp` is absent from the Compute Engine machine-types page. The machine
  type, image family and CE quota this rig uses are all real but are equally explained by the TPU API and GKE
  being implemented on Compute Engine underneath. Nothing here has attempted a create. Read `CLAUDE.md` before
  provisioning, and record the outcome in `../HARDWARE.md` if you do.
- What gcloud consumes here is `--machine-type=ct5lp-hightpu-1t` (24 vCPU / 48 GB), plus
  `--image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --image-project=ubuntu-os-accelerator-images`,
  `--boot-disk-size=200GB`, `--maintenance-policy=TERMINATE` and `--scopes=cloud-platform`. The last two are
  required, and the missing scope fails late — half an hour into a Secret Manager retry loop.
- `ACCELERATOR_TYPE=v5litepod-1` is retained as **documentation only**, so reports line up with the twin.
  `gcloud compute instances create` would reject it. Use "v5e-1" in prose only, never as a gcloud argument.
- Provisioning models are gcloud's SCREAMING_CASE values — `FLEX_START`, `SPOT`, `STANDARD`,
  `RESERVATION_BOUND`. `_provisioning_flags()` is the single place the rig's lowercase labels map to them.
  `--max-run-duration` works with every one of them here, unlike on the Cloud TPU API.
- `_discover_vllm_node()` / `discover_vllm_url()` dynamically find the instance serving vLLM in the configured
  zone and construct `http://{ip}:8000`. They list **TPU-bearing Compute Engine instances**; a `ct5lp-*`
  instance does not appear in `gcloud compute tpus tpu-vm list` at all, so the twin's discovery helper would
  silently return nothing here. Instances are ranked with this rig's own names first and probed on
  `/v1/models`; one belonging to another rig is only used if a probe confirmed it is serving. Use these
  instead of stale endpoints.
- Status is `status: RUNNING`, not `state: READY`, and the IP is at
  `networkInterfaces[].accessConfigs[].natIP`, not `networkEndpoints[].accessConfig.externalIp`. `RUNNING` is
  a weaker claim than the Queued Resource path's `ACTIVE` — the VM reports it long before vLLM is up, so use
  `verify_model_health` for readiness.
- Every SSH-based tool goes through `_ssh_command()`, which builds `gcloud compute ssh`. **Never
  `gcloud compute tpus tpu-vm ssh`** — it cannot reach these instances and fails with a not-found against a VM
  that is plainly RUNNING. On the v6e rig four tools got this wrong after the fork.
- The SSH-based tools resolve their `resource_id` through `_resolve_node_id()`, which is two lookups here
  rather than the twin's three: the id you ask for is the name you get, with no derived `<id>-node`.
- `startup_script_template.sh` is rendered with Python `str.format()`. Its supported placeholders are
  `{project_id}`, `{zone}`, `{model_name}`, `{hf_secret_id}`, `{tensor_parallel_size}`, `{max_model_len}`,
  `{max_num_batched_tokens}`, and `{limit_mm_per_prompt}`. Escape every other literal brace as `{{` or `}}`,
  including shell `${VAR}` and JSON braces, or deployment rendering will fail.
- There is deliberately no `{hf_token}` placeholder. The rendered script is uploaded as instance metadata, so
  it fetches `hf-token` from Secret Manager at boot using the VM's own credentials instead. The VM service
  account needs `roles/secretmanager.secretAccessor` on that secret.
- `create_tpu_instance` is non-destructive and touches only the instance name it is given.
  `manage_tpu_instance` is destructive, but deliberately narrower than the twin's `manage_queued_resource`:
  it only deletes instances carrying this rig's `rig=` label, and reports anything unlabelled or owned by a
  sibling rather than deleting it. Do not "fix" it to match the twin.
- `list_queued_resources` and `_list_queued_resources_json()` survive here **only to detect cross-path
  collisions** — four TPU-API v5e rigs share `us-west4-a` and compete for the same physical chips. Nothing
  here creates or deletes a Queued Resource.
- `tpu_zones_status.md` is mutable program state. `find_tpu` rewrites and reads it to track failed zones; do not
  treat it as ordinary documentation or hand-edit it casually.
- `set_env.sh` must be sourced. `init.sh` may block on interactive input in its error path, so do not run it in
  unattended workflows.
- Authentication may require both `gcloud auth login` for CLI subprocesses and
  `gcloud auth application-default login` for Google client libraries.
- Environment variables actually consumed at startup include `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_ZONE`,
  `MODEL_NAME`, `MACHINE_TYPE`, `IMAGE_FAMILY`, `IMAGE_PROJECT`, `BOOT_DISK_SIZE_GB`, `PROVISIONING_MODEL`,
  `MAX_RUN_DURATION`, `REQUEST_VALID_FOR`, `GCE_QUOTA_ID`, `GCE_SPOT_QUOTA_ID`, `GCE_TPU_FAMILY`,
  `TENSOR_PARALLEL_SIZE`, `INSTANCE_NAME`, `MCP_SERVER_NAME`, and `LOCAL_DOCKER_IMAGE`. `ACCELERATOR_TYPE`,
  `TPU_QUOTA_ID` and `TPU_SPOT_QUOTA_ID` are read but describe the twin's control plane. Confirm the source
  before documenting additional variables.
- The startup script installs `docker.io` before pulling. The Compute Engine image ships **no Docker**, unlike
  the TPU API's runtime versions — do not remove the install, and note `_ENSURE_DOCKER` prefixes every
  Docker-dependent remote command for the same reason.

## Cloud safety

- Never destroy an instance, container, reservation, secret, or other cloud resource unless the user
  explicitly asks for that destructive action. That includes any Queued Resource you find in the zone — those
  belong to sibling rigs, and this one has no business deleting them.
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
