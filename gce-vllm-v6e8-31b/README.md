# TPU vLLM DevOps Agent — Compute Engine path (MCP Server)

## Role
This project functions as an expert TPU SRE and DevOps Engineer, specialized in the **Gemma 4** ecosystem. Its primary goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted via vLLM on Cloud TPUs**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## What makes this rig different

It provisions its TPU as a **Compute Engine instance** (`gcloud compute instances create --machine-type=ct6e-standard-8t`) rather than a Cloud TPU API Queued Resource. The Cloud TPU API is deprecated — no longer under active development, and TPU7x and later are Compute Engine or GKE only.

It serves the **31B** checkpoint. Forked 2026-08-25 from `gce-vllm-v6e8-2b`, which is the E2B rig on the same slice and the same control plane; the checkpoint is the only thing that changed. **This rig is not an A/B twin of anything** — no rig here serves 31B through the Cloud TPU API, so a number measured here is a 31B-on-v6e-8 number and nothing else. Do not diff it against a 2B rig and read the delta as a control-plane result: at 960 KiB/token against E2B's 18, the checkpoint dominates.

See `CLAUDE.md` for the flag-by-flag control-plane mapping and the traps — quota does not carry between the two APIs, and a `ct6e-*` instance is invisible to `gcloud compute tpus tpu-vm list`.

**Nothing here has been provisioned or measured yet.**

## Current Deployment
*   **Model:** `google/gemma-4-31B-it` on TPU v6e-8 (Trillium), eight chips on one host, `ct6e-standard-8t`.
    Reference bf16 release — 62 GB on disk, 57.7 GiB resident, ~7.2 GiB per chip at TP=8.
*   **Endpoint:** discovered at runtime — the agent lists TPU-bearing Compute Engine
    instances, probes them on `/v1/models`, and serves on port 8000. Ask the agent for
    `get_vllm_endpoint`, or run `make endpoint`. Endpoints are ephemeral; don't hardcode them.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on TPU v6e) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your TPU deployment for the model needs:
*   **Hardware:** Cloud TPU v6e (Trillium), eight chips on one host — `ct6e-standard-8t`, topology `2x4`.
    **Eight is a floor, not a preference:** 57.7 GiB of bf16 weights do not fit one 31.24 GiB chip.
*   **Software:** `vllm/vllm-tpu:nightly` specialized container (v0.19.2+ recommended for Gemma 4 fixes).
*   **Model:** `google/gemma-4-31B-it` (Hugging Face ID). The 62 GB pull is why the startup script
    waits 90 minutes for readiness rather than 20.
*   **Runtime:** none — this path takes an `--image-family`, not a `--runtime-version`. The
    `v2-alpha-tpuv6e` runtime belongs to the Cloud TPU API and is not used here.
*   **Networking:** Private Google Access must be enabled for internal connectivity, or direct internet access for Hugging Face downloads.

### 2. Software & API Dependencies
The agent relies on several Google Cloud services and Python libraries:
*   **Libraries:** `mcp` (FastMCP ships inside it), `google-cloud-logging`, `google-cloud-secret-manager`, `openai`, and `httpx`.
*   **Permissions:** The service account running the agent needs:
    *   `logging.logEntries.list` (to read logs).
    *   `tpu.nodes.get` and `tpu.nodes.list` (for discovery).
    *   `secretmanager.versions.access` (for Hugging Face tokens).

### 3. Environment Variables
You can configure the following variables for the MCP server:
*   `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID (defaults to `aisprint-491218`).
*   `GOOGLE_CLOUD_ZONE`: Zone to provision and discover in (defaults to `us-east5-b`; us-west4 has no v6e).
*   `GOOGLE_CLOUD_REGION`: Region for network resources (defaults to `us-east5`).
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-31B-it`).
*   `ACCELERATOR_TYPE`: TPU accelerator type (defaults to `v6e-8`; v6e keeps its marketing name in the
    TPU API, unlike v5e, which gcloud calls `v5litepod-1`).
*   `TPU_RUNTIME_VERSION`: TPU VM runtime (defaults to `v2-alpha-tpuv6e`).
*   `TPU_QUOTA_ID`: Cloud Quotas id scanned by `find_tpu` (defaults to `TPUV6EPerProjectPerZoneForTPUAPI`;
    spot draws on `TPUV6EPreemptiblePerProjectPerZoneForTPUAPI` instead).
*   `TENSOR_PARALLEL_SIZE`: Tensor parallel size (defaults to `8`, and 1 and 2 cannot boot this
    checkpoint). The 31B's 50 sliding layers carry 16 KV heads and shard exactly at 8; its 10 full layers
    carry 4 and pad to 8, a 2x KV penalty on a sixth of the layers. Do not carry over E2B's version of this
    note — that model was full MQA and TP=8 multiplied its KV by eight. See `CLAUDE.md`.
*   `CHIP_COUNT` / `TOPOLOGY`: chips per instance (`8`) and their mesh (`2x4`). `CHIP_COUNT` is derived from
    `MACHINE_TYPE` and only read as an override; both drive the cost estimate and the quota arithmetic.
*   `MCP_SERVER_NAME`: Name this server advertises, and the key it must be registered under — it prefixes
    every tool as `mcp__<name>__find_tpu` (defaults to the rig directory name, `gce-vllm-v6e8-31b`). Set it
    only to match a client that already registered this server under a different key. `make mcp-config`
    writes a `.mcp.json` using the same value.

## Technical Standards
-   **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
-   **Optimization Flags:**
    -   `--tensor-parallel-size 8` (one worker per chip on the single host)
    -   `--max-model-len 32768`
    -   `--disable_chunked_mm_input`
    -   `--max_num_batched_tokens 4096` (required for multimodal compatibility)
    -   `--limit-mm-per-prompt '{"image":4,"audio":1}'` (JSON format required in nightly)
-   **Tooling:** Enable `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, and `--reasoning-parser gemma4`.

## Flex-start VMs
Our stack leverages **Flex-start VMs** — `--provisioning-model=FLEX_START` on `gcloud compute instances create`. (The `v2-alpha-tpuv6e` runtime this line used to name belongs to the Cloud TPU API and is not part of this path.)

### Key Characteristics
*   **Dynamic Workload Scheduler (DWS):** Provisions resources from a secure pool, increasing the probability of securing high-demand TPU v6e chips.
*   **Wait-Time Mechanism:** Requests can wait up to 2 hours for resources if capacity is full.
*   **Execution Limit:** VMs have a maximum run duration of **7 days**, requiring `maxRunDuration` and a termination action.
*   **Dense Placement:** TPU nodes are placed in close physical proximity to minimize network latency.
*   **Cost Efficiency:** Offers discounted pricing for vCPUs, memory, and TPU accelerators.

### Constraints
*   **No Live Migration:** Flex-start VMs do not support live migration.
*   **Quota Requirements:** Requires sufficient **preemptible quota**.
*   **No Reservations:** These instances **cannot** consume existing TPU reservations.

## 🛠 Usage & Setup

### Step 1: Turnkey Deployment to TPU
Use the `manage_queued_resource` tool within the MCP server for a seamless setup, or use the `gcloud` command generated by `get_vllm_deployment_config`.

### Step 2: Run the MCP Server
Install dependencies and run the server locally:
```bash
make install
make run
```

## 🛠 Available Tools

The MCP server exposes 31 tools. The full catalog lives in
[GemmaTools.md](GemmaTools.md), generated straight from the `@mcp.tool()`
decorators in `server.py` — regenerate it with `make tools`. You can also call
the `get_help` tool, which builds the same list at runtime.

Highlights:

*   **`find_tpu`**: Scans zones for available v6e quota and provisions the Queued Resource in the first one that takes it.
*   **`manage_queued_resource`**: Ensures the primary Queued Resource exists and cleans up redundant ones.
*   **`manage_vllm_docker`**: Starts, stops, restarts, or inspects the vLLM container on the TPU VM.
*   **`get_system_status`**: High-level dashboard of Queued Resource state, quota, and vLLM health.
*   **`query_queued_gemma4_with_stats`**: Queries the self-hosted model and reports latency and throughput.
*   **`analyze_cloud_logging`**: Summarizes TPU errors from Cloud Logging **using the self-hosted Gemma 4 model** — the agent debugging its own infrastructure.

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities:
```bash
python demo_launcher.py
```