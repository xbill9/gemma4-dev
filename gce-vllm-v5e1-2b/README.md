# TPU vLLM DevOps Agent — v5e on the Compute Engine path (MCP Server)

## Role
This project functions as an expert TPU SRE and DevOps Engineer, specialized in the **Gemma 4** ecosystem. Its primary goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted via vLLM on Cloud TPUs**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## What makes this rig different

It provisions its TPU as a **Compute Engine instance** (`gcloud compute instances create --machine-type=ct5lp-hightpu-1t`) rather than a Cloud TPU API Queued Resource. The Cloud TPU API is deprecated — no longer under active development, and TPU7x and later are Compute Engine or GKE only.

It is the **A/B twin of `tpu-vllm-v5e1-2b`**: same v5e-1 chip, same `gemma-4-E2B-it`, same vLLM flags, different control plane. Only the provisioning path varies, on purpose. See `CLAUDE.md` for the flag-by-flag mapping and the traps — quota does not carry between the two APIs, and a `ct5lp-*` instance is invisible to `gcloud compute tpus tpu-vm list`.

## ⚠️ This path is not known to exist for v5e

Unlike the v5p and v6e Compute Engine rigs, **this one rests on an open question.** Google's
[v5e page](https://docs.cloud.google.com/tpu/docs/v5e) says v5e "is supported using Google Kubernetes Engine
and the Cloud TPU API" — Compute Engine is not named — and `ct5lp` is absent from the Compute Engine
machine-types page.

Against that: `ct5lp-hightpu-1t` is a real machine type in 26 zones, the shared OS image family is named for
v5e, and there is a v5-lite quota on `compute.googleapis.com`. All three are equally explained by the Cloud
TPU API and GKE being implemented *on* Compute Engine underneath, so none of them is evidence the path is
directly reachable.

**The working answer is no, and nothing here has attempted a create.** One `create_tpu_instance` settles it:
a rejection at validation is free and conclusive; an acceptance bills until deleted. Record the outcome in
the monorepo's `HARDWARE.md` and `NAMING.md` — it decides whether six v5e rigs have a migration path.

**Nothing here has been provisioned or measured.**

## Current Deployment
*   **Model:** `google/gemma-4-E2B-it` on TPU v5e-1, one chip, `ct5lp-hightpu-1t` (24 vCPU / 48 GB host).
*   **Endpoint:** discovered at runtime — the agent lists TPU-bearing Compute Engine
    instances, probes them on `/v1/models`, and serves on port 8000. Ask the agent for
    `get_vllm_endpoint`, or run `make endpoint`. Endpoints are ephemeral; don't hardcode them.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on TPU v5e) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your TPU deployment for the model needs:
*   **Hardware:** one Cloud TPU v5e chip, requested as machine type `ct5lp-hightpu-1t`. 16 GB HBM.
*   **Image:** `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` from `ubuntu-os-accelerator-images`. Pin the
    *family*, never a dated build. **This image ships no Docker** — the startup script installs it.
*   **Boot disk:** 200 GB. The 10 GB image default cannot hold the vLLM TPU image, and undersizing fails
    late, during the docker pull.
*   **Scopes:** `--scopes=cloud-platform`, required so the VM can read the HF token from Secret Manager.
*   **Maintenance policy:** `TERMINATE`. A TPU instance cannot live-migrate.
*   **Software:** `vllm/vllm-tpu:nightly` specialized container (v0.19.2+ recommended for Gemma 4 fixes).
*   **Model:** `google/gemma-4-E2B-it` (Hugging Face ID).
*   **Networking:** Private Google Access must be enabled for internal connectivity, or direct internet access for Hugging Face downloads.

### 2. Software & API Dependencies
The agent relies on several Google Cloud services and Python libraries:
*   **Libraries:** `mcp` (FastMCP ships inside it), `google-cloud-logging`, `google-cloud-secret-manager`, `openai`, and `httpx`.
*   **Permissions:** The service account running the agent needs:
    *   `logging.logEntries.list` (to read logs).
    *   `compute.instances.get`, `compute.instances.list`, `compute.instances.create`,
        `compute.instances.delete` (this path uses Compute Engine, not `tpu.nodes.*`).
    *   `secretmanager.versions.access` (for Hugging Face tokens).

### 3. Environment Variables
Configuration lives in `tpu.env`, which is the single source of truth; a real environment variable always
wins over it. The ones that matter:
*   `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID (defaults to `aisprint-491218`).
*   `GOOGLE_CLOUD_ZONE`: Zone to provision and discover in (defaults to `us-west4-a`).
*   `GOOGLE_CLOUD_REGION`: Region for network resources (defaults to `us-west4`).
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-E2B-it`).
*   `MACHINE_TYPE`: **What Compute Engine actually consumes** (defaults to `ct5lp-hightpu-1t`).
*   `IMAGE_FAMILY` / `IMAGE_PROJECT` / `BOOT_DISK_SIZE_GB`: replace the TPU API's `--runtime-version`.
*   `ACCELERATOR_TYPE`: **documentation only** on this path (`v5litepod-1`, the Cloud TPU API's spelling,
    kept so reports line up with the twin). Never passed to `gcloud compute instances create`.
*   `PROVISIONING_MODEL`: `flex-start` (default), `spot`, `on-demand`, or `reservation-bound`. This rig's
    lowercase labels; `server.py` maps them to gcloud's `FLEX_START` / `SPOT` / `STANDARD` /
    `RESERVATION_BOUND`.
*   `GCE_QUOTA_ID` / `GCE_SPOT_QUOTA_ID`: the Compute Engine quota ids this path consumes
    (`TPU-LITE-PODSLICE-V5-per-project-zone` and its `PREEMPTIBLE-` twin — both per-zone on v5e).
    `TPU_QUOTA_ID` / `TPU_SPOT_QUOTA_ID` are the twin's TPU-API ids, kept for comparison only.
*   `TENSOR_PARALLEL_SIZE`: Tensor parallel size (defaults to `1`).
*   `MCP_SERVER_NAME`: Name this server advertises, and the key it must be registered under — it prefixes
    every tool as `mcp__<name>__find_tpu` (defaults to the rig directory name, `gce-vllm-v5e1-2b`). Set it
    only to match a client that already registered this server under a different key. `make mcp-config`
    writes a `.mcp.json` using the same value.

## Technical Standards
-   **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
-   **Optimization Flags:**
    -   `--tensor-parallel-size 1` (v5e-1 is a single chip)
    -   `--max-model-len 16384` (held equal to the twin; v5e has half the v6e rig's HBM)
    -   `--disable_chunked_mm_input`
    -   `--max_num_batched_tokens 4096` (required for multimodal compatibility)
    -   `--limit-mm-per-prompt '{"image":4,"audio":1}'` (JSON format required in nightly)
-   **Tooling:** Enable `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, and `--reasoning-parser gemma4`.

## Provisioning models

All four are `--provisioning-model` values on `gcloud compute instances create`. Unlike the Cloud TPU API,
`--max-run-duration` is available on every one of them, and pairing it with
`--instance-termination-action=DELETE` is what makes a VM clean up after itself.

*   **`FLEX_START`** — queues through Dynamic Workload Scheduler, waiting up to `REQUEST_VALID_FOR` (2h) for
    capacity, then self-deletes at `MAX_RUN_DURATION` (4h). Places nodes densely. Cannot consume an existing
    reservation, and does not support live migration.
*   **`SPOT`** — reclaimable with ~30s notice. **On v5e in us-west4 this is genuinely the cheapest**
    ($0.5779/chip-hr against flex-start's $0.60 and on-demand's $1.20) — but that is a fact about this chip
    in this region, not a rule: on v6e in us-east5 the order inverts. Read `estimate_deployment_cost`.
*   **`STANDARD`** — on-demand, full price, no preemption.
*   **`RESERVATION_BOUND`** — consumes a calendar or dense-deployment reservation. **Has no Queued Resource
    equivalent**, so it exists only on this path, and it has no catalog rate.

## 🛠 Usage & Setup

### Step 1: Turnkey Deployment to TPU
Use the `create_tpu_instance` or `manage_tpu_instance` tool within the MCP server for a seamless setup, use
`make deploy-tpu-flex` / `deploy-tpu-spot` / `deploy-tpu-ondemand`, or run the `gcloud` command generated by
`get_vllm_deployment_config`. All three go through `gcloud compute instances`, so they address the same
object — which was not true of the TPU-API rigs.

### Step 2: Run the MCP Server
Install dependencies and run the server locally:
```bash
make install
make run
```

## 🛠 Available Tools

The MCP server exposes 32 tools. The full catalog lives in
[GemmaTools.md](GemmaTools.md), generated straight from the `@mcp.tool()`
decorators in `server.py` — regenerate it with `make tools`. You can also call
the `get_help` tool, which builds the same list at runtime.

Highlights:

*   **`find_tpu`**: Sweeps the zones publishing `ct5lp-hightpu-1t` and creates an instance in the first one that takes it.
*   **`create_tpu_instance`**: Creates the instance. Non-destructive — touches only the name it was given.
*   **`manage_tpu_instance`**: Ensures the primary instance exists and deletes redundant ones **carrying this rig's label**, leaving siblings alone.
*   **`manage_vllm_docker`**: Starts, stops, restarts, or inspects the vLLM container on the instance.
*   **`get_system_status`**: High-level dashboard of instance state, quota, and vLLM health.
*   **`query_queued_gemma4_with_stats`**: Queries the self-hosted model and reports latency and throughput.
*   **`analyze_cloud_logging`**: Summarizes TPU errors from Cloud Logging **using the self-hosted Gemma 4 model** — the agent debugging its own infrastructure.

`list_queued_resources` is retained deliberately: it reports the *other* control plane's resources, because
four TPU-API v5e rigs share this project and zone and compete for the same physical chips.

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities:
```bash
python demo_launcher.py
```
