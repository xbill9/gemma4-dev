# TPU vLLM DevOps Agent (MCP Server)

## Role
This project functions as an expert TPU SRE and DevOps Engineer, specialized in the **Gemma 4** ecosystem. Its primary goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted via vLLM on Cloud TPUs**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## What this rig serves

This rig serves **`google/gemma-4-E2B-it-qat-w4a16-ct`** — genuine 4-bit weights with 16-bit
activations, packaged in a compressed-tensors container. Unlike its sibling
[`../tpu-vllm-v5e1-2b-q4_0`](../tpu-vllm-v5e1-2b-q4_0/), whose `-unquantized` checkpoint is bf16 on disk, **these tensors really are 4-bit**.

Slot 5 is `w4a16`, not `w4a16-ct`: per [NAMING.md](../NAMING.md#slot-5--encoding-optional),
compressed-tensors is the *container*, not the encoding, and a hyphen inside a slot would break
parsing. The container is recorded in `MODEL_NAME`, which spells it in full.

### The image

The rig serves on the pinned `vllm/vllm-tpu@sha256:19a1a052…` with `patches/` applied at boot, in `patches/ORDER`,
as `vllm-tpu-w4a16:patched` (`startup_script_template.sh`). Two of the patches carry this checkpoint:

- `wna16.diff` adds the W4A16 linear method on the JAX path. Upstream `tpu-inference` @ `0425df5` (read 2026-08-07)
  had no integer wNa16 scheme on either backend (`compressed_tensors.py:149` raises `NotImplementedError` on the vLLM
  backend; the JAX backend carries `TODO: w4a8 / wNa16 schemes need their own JAX methods`).
- `kvshare.diff` builds no K/V weights for the 20 KV-shared layers (15-34). Google's QAT export omits their
  `k_proj`, `v_proj` and `k_norm`; `vllm/vllm-tpu:nightly` as pulled 2026-10-02 (`sha256:106a30b6…`) requires them
  on every layer and exits at model load (upstream tpu-inference #3225).

The patches are written against that digest, so it is pinned. Moving to a newer image means re-checking that
`patches/` still applies and that the checkpoint loads.

### Booted 2026-10-02 on the patched image: does not yet reach serving

Two boots through `create_tpu_queued_resource` (`benchmarks/runs/2026-10-02-rig-boot-patched-v5e1`). The checkpoint loads on the patched image (weights 7.17 GiB). At vLLM's default `--gpu-memory-utilization` 0.90 the KV pool left 1.13 GiB of HBM free and loading the compiled program asked for 1.15 GiB (`RESOURCE_EXHAUSTED`), so the rig sets `GPU_MEMORY_UTILIZATION=0.80`. At 0.80 the host's OOM killer stopped the engine during compilation (35.1 GiB resident on the 48 GiB host, model files also in `/dev/shm`). The v5e-4 sibling serves this checkpoint on the same image (`../tpu-vllm-v5e4-2b-w4a16/benchmarks/runs/2026-10-02-rig-boot-patched-v5e4`). The patched v5e-1 rigs that serve cap `--max_num_batched_tokens` at 512 with `MIN_TOKEN_BUCKET=64` to keep compile host memory down; this rig keeps 4096.

### Measured 2026-09-30 (throughput)

Served beside the E2B QAT repack on the same VM (`benchmarks/runs/2026-09-30-gspeedb-e2b-google-v5e1`): output tok/s at 1 / 4 / 16 requests 136.6 / 532.3 / 1,911 against the repack's 136.5 / 532.1 / 1,910. The repack's 2.4-point suite lead costs no speed.

## Current Deployment
*   **Model:** `google/gemma-4-E2B-it-qat-w4a16-ct` on TPU v5e-1 (v5litepod).
*   **Endpoint:** discovered at runtime — the agent finds the `ACTIVE` Queued Resource,
    resolves its node IP, and serves on port 8000. Ask the agent for `get_vllm_endpoint`,
    or run `make endpoint`. Endpoints are ephemeral; don't hardcode them.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on TPU v5e) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your TPU deployment for the model needs:
*   **Hardware:** Cloud TPU v5e (v5litepod) with topology `1x1` (1 chip).
*   **Software:** `vllm/vllm-tpu@sha256:19a1a052…` with this rig's `patches/` applied, built on the VM at boot as `vllm-tpu-w4a16:patched` (`startup_script_template.sh`). `vllm/vllm-tpu:nightly` cannot load this checkpoint (upstream tpu-inference #3225).
*   **Model:** `google/gemma-4-E2B-it-qat-w4a16-ct` (Hugging Face ID).
*   **Runtime:** `v2-alpha-tpuv5-lite` for Flex-start / Queued Resources.
*   **Networking:** Private Google Access must be enabled for internal connectivity, or direct internet access for Hugging Face downloads.

### 2. Software & API Dependencies
The agent relies on several Google Cloud services and Python libraries:
*   **Libraries:** `mcp` (`MCPServer` ships inside it; v1 called it FastMCP), `google-cloud-logging`, `google-cloud-secret-manager`, `openai`, and `httpx`.
*   **Permissions:** The service account running the agent needs:
    *   `logging.logEntries.list` (to read logs).
    *   `tpu.nodes.get` and `tpu.nodes.list` (for discovery).
    *   `secretmanager.versions.access` (for Hugging Face tokens).

### 3. Environment Variables
You can configure the following variables for the MCP server:
*   `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID (defaults to `aisprint-491218`).
*   `GOOGLE_CLOUD_ZONE`: Zone to provision and discover in (defaults to `europe-west4-a`).
*   `GOOGLE_CLOUD_REGION`: Region for network resources (defaults to `europe-west4`).
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-E2B-it-qat-w4a16-ct`).
*   `ACCELERATOR_TYPE`: TPU accelerator type (defaults to `v5litepod-1` — gcloud's name for v5e-1).
*   `TPU_RUNTIME_VERSION`: TPU VM runtime (defaults to `v2-alpha-tpuv5-lite`).
*   `TPU_QUOTA_ID`: Cloud Quotas id scanned by `find_tpu` (defaults to `TPUV5sLitepodPerProjectPerZoneForTPUAPI`).
*   `TENSOR_PARALLEL_SIZE`: Tensor parallel size (defaults to `1`).
*   `MCP_SERVER_NAME`: Name this server advertises, and the key it must be registered under — it prefixes
    every tool as `mcp__<name>__find_tpu` (defaults to the rig directory name, `tpu-vllm-v5e1-2b-w4a16`). Set it
    only to match a client that already registered this server under a different key. `make mcp-config`
    writes a `.mcp.json` using the same value.

## Technical Standards
-   **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
-   **Optimization Flags:**
    -   `--tensor-parallel-size 1` (v5e-1 is a single chip)
    -   `--max-model-len 16384`
    -   `--disable_chunked_mm_input`
    -   `--max_num_batched_tokens 4096` (required for multimodal compatibility)
    -   `--limit-mm-per-prompt '{"image":4,"audio":1}'` (JSON format)
-   **Tooling:** Enable `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, and `--reasoning-parser gemma4`.

## Flex-start VMs
Our stack leverages **Flex-start VMs** (via the `v2-alpha-tpuv5-lite` runtime) to maximize TPU availability and minimize costs.

### Key Characteristics
*   **Dynamic Workload Scheduler (DWS):** Provisions resources from a secure pool, increasing the probability of securing high-demand TPU v5e chips.
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

*   **`find_tpu`**: Scans zones for available v5e quota and provisions the Queued Resource in the first one that takes it.
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