---
name: tpu-management
description: Manage Google Cloud TPU capacity, Gemma 4 vLLM serving, and bare JAX dev VMs on TPU. Use when the user asks about provisioning, finding, listing, or destroying TPUs / queued resources / flex-start VMs, starting or debugging vLLM on TPU (v6e, v5p, v5e), TPU quotas and zones, TPU cost estimates, benchmarking TPU serving, or the TPU devops MCP agent. Triggers include "TPU", "queued resource", "flex-start", "v6e", "vLLM on TPU", "TPU quota", "JAX on TPU".
---

# TPU Management

Operate Google Cloud TPU infrastructure: acquire capacity, run Gemma 4 vLLM
serving (verify health, benchmark), and tear down. Two ways to act:

1. **Preferred — MCP agent tools.** If the `tpu-devops-agent` MCP server is
   connected in this session, use its tools (catalog below). They wrap the correct
   `gcloud` invocations, discovery, and retry/cleanup logic.
2. **Fallback — direct `gcloud`.** If the MCP server is not connected, either offer to
   register the bundled server (see "Registering the MCP server") or run the equivalent
   `gcloud` commands from `references/tpu-guide.md`.

## Bundled files

- `mcp/server.py` — the FastMCP DevOps agent (snapshot of the repo-root `server.py`;
  the live copy at the repo root is authoritative if the two differ).
- `mcp/project-setup.sh` — one-command installer: copies this skill into a target project and
  registers the MCP server (see "Registering the MCP server").
- `mcp/startup_script_template.sh` — the TPU VM startup script the agent injects when
  creating a queued resource (pulls `vllm/vllm-tpu:nightly` and serves the model).
- `mcp/startup_script_jax_template.sh` — the startup script for `workload="jax"` VMs
  (installs a current CPython + `jax[tpu]` on the bare VM, no docker and no HF token,
  and asserts a TPU device is visible before reporting ready).
- `references/tpu-guide.md` — the TPU getting started guide: prerequisites,
  flex-start capacity zones per TPU family, `gcloud` creation templates for v6e/v5p/v5e,
  persistent-disk + startup-script patterns, quota metrics and request procedure,
  troubleshooting/FAQ. Read it when working without the MCP tools, diagnosing
  provisioning failures, or answering quota/capacity/billing questions.

## Registering the MCP server

Easiest path — run the bundled installer (idempotent; installs this skill into the
target project and writes the `tpu-devops` entry into the project's `.mcp.json`,
using the system `python3` — it warns if the pip deps below are missing but never
creates a venv):

```bash
mcp/project-setup.sh /path/to/project --project <gcp-project-id>   # one project
mcp/project-setup.sh --global                                      # all projects (user scope)
# from the skill repo root: make init TARGET=/path/to/project ARGS='--project <id>'
```

Run `mcp/project-setup.sh --help` for all options (`--model`, `--accelerator`, `--tp`,
`--server-name`, `--skip-deps`). Then restart Codex in the target project and
approve the server when prompted; `/mcp` should list `tpu-devops`.

Manual alternative:

```bash
Codex mcp add tpu-devops \
  --env GOOGLE_CLOUD_PROJECT=<project-id> \
  --env MODEL_NAME=google/gemma-4-31B-it \
  --env ACCELERATOR_TYPE=v6e-8 \
  --env TENSOR_PARALLEL_SIZE=8 \
  -- python .Codex/skills/tpu-management/mcp/server.py
```

Requires: `pip install -r mcp/requirements.txt`, an authenticated
`gcloud` CLI with alpha components (`gcloud components install alpha`), and the TPU API
enabled. The server reads config from env vars: `GOOGLE_CLOUD_PROJECT` (falls back to
the active gcloud config), `GOOGLE_CLOUD_ZONE` (default `europe-west4-a`),
`GOOGLE_CLOUD_REGION`, `MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`.
A Hugging Face token must exist as Secret Manager secret `hf-token` (save one with the
`save_hf_token` tool) before any resource creation.

## Standard lifecycle

1. **Status first.** `get_system_status` (dashboard) or `list_queued_resources` /
   `find_gpu`. Never create before checking what already exists.
2. **Acquire capacity.**
   - Preferred (v6e/v5p): `create_tpu_vm_instance` — GCE flex-start VM with vLLM
     auto-start — or `find_tpu_vm` to sweep zones until one grants capacity (family
     quota is only discoverable by attempting creation). Then `wait_for_vllm_ready`
     polls until serving is up; `get_tpu_vm_serial_log` for manual watching.
   - Known zone, legacy API: `create_tpu_queued_resource` (non-destructive; skips if
     the resource already exists) or `manage_queued_resource` (destructive — deletes
     every other queued resource in the zone). Flex-start by default: 4h max-run;
     `reserved=True` for reservations.
   - Unknown zone: `get_zones_with_available_quota`, or `find_tpu` which sweeps every
     zone with quota, polls until ACTIVE (3 min, extended to 10 min once PROVISIONING),
     and cleans up failures. It skips zones previously marked failed in
     `~/.cache/tpu-devops/tpu_zones_status.md`.
3. **Wait for ACTIVE.** `describe_queued_resource`.
   Queued resources move QUEUED → PROVISIONING → ACTIVE; FAILED/SUSPENDED means
   delete and retry (the manage tool does this automatically).
4. **Serve.** The creation startup script auto-starts vLLM. Otherwise
   `manage_vllm_docker` with action `start|stop|restart|status|log|rm` (targets the
   queued resource's node by default, or a GCE VM via `instance_name` — same for
   `get_vllm_docker_logs`, `get_tpu_system_logs`, `run_vllm_benchmark`).
   **Switching the model:** call `start` with `model_name` (or any serving param) —
   that replaces the container with the new config; a plain `start` just restarts
   the existing container unchanged. It auto-picks
   load format, max-model-len, and memory utilization from the model size
   (26B/31B → `tpu_streaming_loader`, 16384 ctx, 0.80 util; smaller → `runai_streamer`,
   65536 ctx, 0.90 util). Model load can take many minutes — check
   `get_vllm_docker_logs` for "Application startup complete."
5. **Verify.** `verify_model_health`, `get_vllm_endpoint`, `get_model_details`,
   `query_queued_gemma4` (`include_stats=True` for TTFT/throughput). Health checks,
   queries, and benchmarks auto-target whatever model the server actually loaded
   (via `/v1/models`), so they keep working after a deploy-time `model_name` override.
6. **Benchmark (optional).** `run_vllm_benchmark` (runs `vllm bench serve` in a
   separate container on the VM). Pass `save_result=True` to also get the run's
   metrics as a `throughput.sweep[]` entry for the repo's benchmark report format
   (`benchmarks/serving-report.schema.json`) — use it once per concurrency level
   when building a report.
7. **Tear down.** `destroy_queued_resource`. Flex-start bills until deletion and
   cannot be paused — always confirm teardown of idle resources with the user, and
   remind them a flex-start resource left running expires at max-run-duration.

## MCP tool catalog (by task)

**Capacity & lifecycle (GCE flex-start — recommended for v6e/v5p):**
`create_tpu_vm_instance` (creates the VM with the proven flags: 200GB boot disk,
docker-installing startup script, cloud-platform scopes; `workload="jax"` for a bare
JAX VM instead), `find_tpu_vm` (zone sweep, same workload choice),
`wait_for_vllm_ready` (poll until serving), `wait_for_jax_ready`,
`verify_jax_tpu` (assert jax.devices() sees a TPU), `list_tpu_vm_instances`,
`destroy_tpu_vm_instance`, `get_tpu_vm_serial_log`, `get_tpu_vm_endpoint`

**Capacity & lifecycle (queued resources — legacy, v5e):** `find_tpu`,
`create_tpu_queued_resource` (non-destructive),
`manage_queued_resource` (destructive cleanup), `destroy_queued_resource`,
`list_queued_resources`, `describe_queued_resource`,
`get_zones_with_available_quota`, `find_gpu`, `estimate_deployment_cost`

**Serving:** `manage_vllm_docker`, `get_vllm_endpoint`,
`get_vllm_deployment_config` (gcloud one-liner), `save_hf_token`

**Health, logs & diagnostics:** `get_system_status`, `verify_model_health`,
`get_model_details`, `get_metrics`, `get_vllm_docker_logs`, `get_tpu_system_logs`,
`get_cloud_logging_logs`, `analyze_cloud_logging` (Gemma-4-powered log triage)

**Inference & benchmarking:** `query_queued_gemma4` (`include_stats=True` for
latency/throughput), `run_vllm_benchmark`

Every agent in this repo also exposes `get_help` for its live configuration.

## vLLM on TPU — required flags (Gemma 4)

When composing or reviewing a vLLM serve command for TPU, use:
`--tensor-parallel-size 8` (v6e-8), `--max-model-len 16384`,
`--disable_chunked_mm_input`, `--max_num_batched_tokens 4096`,
`--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4`,
and `--limit-mm-per-prompt '{"image":4,"audio":1}'` for multimodal
(the agent uses `{"image":0,"audio":0}` for text-only serving).
Image: `vllm/vllm-tpu:nightly`, run with `--privileged --net=host --shm-size 10gb`
and `HF_HOME=/dev/shm`.

Upstream references: [vLLM TPU docs](https://docs.vllm.ai/projects/tpu/en/latest/),
[Recommended Models & Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/)
(the support matrix — check it before serving quantized checkpoints),
[vLLM Recipes](https://recipes.vllm.ai) (per-model deployment guides), and the
[tpu-inference GitHub repo](https://github.com/vllm-project/tpu-inference)
([releases](https://github.com/vllm-project/tpu-inference/releases) track newly
landed quantization/model support).

Known-broken (verified on `vllm-tpu:nightly`, Jul 2026): the Gemma 4 **E2B QAT**
checkpoints do not load on TPU in any form — `-qat-w4a16-ct` fails with
"compressed-tensors scheme for layer 'per_layer_model_projection' is not yet
supported in the JAX path", and `-qat-q4_0-unquantized` fails on both the JAX and
`MODEL_IMPL_TYPE=vllm` (torchax) paths with "weights not initialized from
checkpoint: layers.15-34 self_attn.k_norm.weight" (the export omits k_norm for
the upper KV-sharing layers). Serve the plain `google/gemma-4-E2B-it` instead.

## JAX on TPU (bare dev VMs)

For direct JAX work — kernels, benchmarks, custom inference — instead of vLLM
serving, provision with `workload="jax"`:

1. `create_tpu_vm_instance(workload="jax")` (default name `jax-tpu-vm`) or
   `find_tpu_vm(workload="jax")`. The startup script installs `JAX_PYTHON_VERSION`
   (default 3.13) from deadsnakes and pip-installs `JAX_PIP_SPEC` (default
   `jax[tpu]`) into it directly — no venv, per this repo's standard; the dedicated
   interpreter provides the isolation. No docker, no Hugging Face token, no
   200 GB image pull, so it boots in a couple of minutes rather than ~8.5.
2. `wait_for_jax_ready` polls the serial console for
   `JAX-BOOTLOADER: TPU environment ready.` and fails fast on
   `JAX-BOOTLOADER: FAILED`. `verify_jax_tpu` re-runs the device check over SSH.
3. Run workloads with `python3.13` (or whatever `JAX_PYTHON_VERSION` is set to),
   **not** `python3` — the system interpreter is 3.10 and has no JAX.

**Why the script asserts on `jax.devices()`:** importing `jax` succeeds on a host
with no TPU backend, so an import-only check reports success on a broken VM. The
script exits non-zero unless a device with `platform == "tpu"` is present, and only
then prints the ready marker.

**Field notes:**

- Ubuntu 22.04's system Python is 3.10, which resolves `jax[tpu]` to an old
  release (0.6.x). If you need a specific JAX, pin `JAX_PIP_SPEC`
  (e.g. `jax[tpu]==0.11.0`) — and remember `libtpu` comes from the JAX releases
  index, not PyPI.
- The system `pip` (22.x) predates `--break-system-packages`; the script
  bootstraps pip into the new interpreter with `get-pip.py` instead.
- `add-apt-repository` is not on the accelerator image by default — the script
  installs `software-properties-common` first.
- Direct SSH often times out; use `--tunnel-through-iap`.
- Only one process may hold the TPU. A crashed or backgrounded run keeps the chip
  ("The TPU is already in use by process with pid N"); kill it before retrying.

## Field notes — GCE flex-start path (`gcloud compute instances create`)

Verified on a live v6e-1 deployment (Jul 2026). When creating TPU VMs as GCE
instances (the reference guide's template) rather than queued resources, the
guide's command as written will fail; apply all of these:

- **Boot disk:** the default is only 10 GB (hyperdisk-balanced) — `vllm/vllm-tpu:nightly`
  overflows it during layer extraction ("no space left on device"). Add
  `--boot-disk-size=200GB`. If already created, recover without losing flex-start
  capacity: `gcloud compute disks resize <name> --size=200GB` then
  `gcloud compute instances reset <name>` (never delete/recreate — that forfeits
  the capacity grant and restarts the max-run clock).
- **Docker:** not preinstalled on the `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`
  image (unlike TPU runtime images). The bundled startup script template now
  installs `docker.io` when missing; custom scripts must do the same.
- **Secrets at boot:** add `--scopes=cloud-platform` at creation and grant the
  default compute SA `roles/secretmanager.secretAccessor` on `hf-token`
  (`gcloud secrets add-iam-policy-binding hf-token --member=serviceAccount:<project-number>-compute@developer.gserviceaccount.com --role=roles/secretmanager.secretAccessor`).
  The bundled startup template fetches the token at boot via the metadata server +
  Secret Manager REST API with a retry loop (~30 min) so an IAM grant applied after
  creation still lands — the token is never written into instance metadata. Custom
  scripts should do the same. Symptom of a missing grant/scope: the fetch 403s forever.
- **Watch boot via serial console, not SSH:** SSH is often blocked by firewall
  policy. The startup template mirrors its log to `/dev/console`; follow it with
  `gcloud compute instances get-serial-port-output <name>`. Grep for the final
  "vLLM application startup complete." line — the earlier "Waiting for
  'Application startup complete.'" echo is a false-positive match.
- **When direct SSH times out, tunnel through IAP:** even with a VPC rule
  allowing tcp:22, an org policy or the client network may drop direct port-22
  traffic (symptom: `gcloud compute ssh` hangs then "Connection timed out").
  `gcloud compute ssh <name> --tunnel-through-iap` rides over HTTPS instead and
  needs only the standard IAP firewall rule (source `35.235.240.0/20`) plus
  `roles/iap.tunnelResourceAccessor`. Note the MCP agent's SSH-based tools
  (`manage_vllm_docker`, `get_vllm_docker_logs`, `run_vllm_benchmark`,
  `get_tpu_system_logs`) do not use IAP; on such networks run their documented
  equivalents manually with `--tunnel-through-iap`.
- **Quota is per region AND per TPU family:** creation fails immediately with
  `Quota 'TPUS_PER_TPU_FAMILY' exceeded. Limit: 0.0` in regions without CT6E
  quota (observed: us-east5 = 0, europe-west4 OK). This dimensioned quota is not
  visible via `gcloud compute regions describe` — attempt creation (fails fast)
  or check the console. Failure sequence for a v6e-1: boot ~1 min → docker
  install ~1 min → image pull ~5 min → model download/compile ~5-10 min.

## Cautions

- `destroy_queued_resource` and `manage_queued_resource` delete infrastructure —
  `manage_queued_resource` deletes ALL queued resources in the zone other than the
  named primary. Confirm with the user before invoking against a zone that may hold
  resources they want kept.
- Flex-start requests expire (`--valid-until-duration`) and instances self-delete at
  `--max-run-duration`; data on the VM is lost. Persist data on a separate disk or GCS
  (see the reference guide).
- Stuck in `WAITING_FOR_RESOURCES`/`PROVISIONING` or `STOCKOUT`: usually the
  `GPUS_ALL_REGIONS` global quota is 0 — see the Troubleshooting section of
  `references/tpu-guide.md` before retrying other zones.
- v5e uses the legacy queued-resources API and separate quota metrics; v6e/v5p use GCE
  machine types (`ct6e-standard-4t`, `ct5p-hightpu-4t`). Zone/family table is in the
  reference guide.
