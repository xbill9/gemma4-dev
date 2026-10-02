# TPU vLLM DevOps Agent (MCP Server)

## Role
This project functions as an expert TPU SRE and DevOps Engineer, specialized in the **Gemma 4** ecosystem. Its primary goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted via vLLM on Cloud TPUs**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## What this rig is for: an expected failure

This rig serves **`google/gemma-4-E2B-it-qat-w4a16-ct`** — genuine 4-bit weights with 16-bit
activations, packaged in a compressed-tensors container. Unlike its sibling
[`tpu-vllm-v5e1-2b-q4_0`](../tpu-vllm-v5e1-2b-q4_0/), whose `-unquantized` checkpoint is bf16 on
disk, **these tensors really are 4-bit, and the current stack cannot load them.**

Slot 5 is `w4a16`, not `w4a16-ct`: per [NAMING.md](../NAMING.md#slot-5--encoding-optional),
compressed-tensors is the *container*, not the encoding, and a hyphen inside a slot would break
parsing. The container is recorded in `MODEL_NAME`, which spells it in full.

### Where it refuses

Verified by reading `tpu-inference` @ `0425df5` on 2026-08-07. The dispatch reaches a real
compressed-tensors code path — further than a GGUF file gets — then raises:

```
tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py:149
    raise NotImplementedError(
        "No compressed-tensors compatible scheme was found for layer {layer_name}.")
```

That line is the fall-through after the four schemes that *are* implemented: nvfp4 (W4A4 and
NVFP4A16), fp8-W4A8, fp8-W8A8, and int8-W8A8. There is no integer wNa16 scheme in
`.../compressed_tensors/schemes/`. The JAX backend stops at the same wall one step earlier, and
says so in a comment:

```
tpu_inference/layers/jax/quantization/compressed_tensors.py:145
    # TODO: w4a8 / wNa16 schemes need their own JAX methods (not yet ported).
```

So the useful output of this rig today is **the traceback and how far the load got**, not
throughput. Run it with `DISABLE_VLLM_SERVER=true` and load the model directly so the raise is
visible, rather than buried in server startup.

### When this starts working

Nothing here needs to change except the expectation. `supported_quantization` in
`tpu_inference/platforms/tpu_platform.py:112` already lists `compressed-tensors`, so the checkpoint
passes platform validation; only the per-layer scheme is missing. If a `wNa16` scheme lands in
either backend, this rig serves it with no config change.

### Measured on v5e-4 (2026-10-02)

The rig provisioned through its own `create_tpu_queued_resource`: flex-start `v5litepod-4` in `us-west4-a`, runtime
`v2-alpha-tpuv5-lite`, `--tensor-parallel-size 4`, capacity granted within 5 minutes of the request. On
`vllm/vllm-tpu:nightly` as pulled that day (digest `sha256:106a30b6…`) the container exits during model load, inside
`get_flax_model` → `create_jit_model`:

```
TypeError: Argument 'model.states[0][378]' of shape bfloat16[256] of type <class 'jax.ShapeDtypeStruct'> is not a valid JAX type.
```

The same container at `--tensor-parallel-size 1` on the same VM, every other argument identical, fails with the same
error, so the failure belongs to the image and checkpoint and is independent of chip count. The raise site on this
nightly is a JAX type error on a model state; the 2026-08-07 source reading above predicted the compressed-tensors
`NotImplementedError`. The rig does not serve as committed. Record and logs:
[`benchmarks/runs/2026-10-02-rig-boot-v5e4`](benchmarks/runs/2026-10-02-rig-boot-v5e4/).
Cause (diagnosed 2026-10-02, run record's Diagnosis section): Google's QAT export omits `k_proj`/`v_proj`/`k_norm`
for the 20 KV-shared layers and that day's nightly allocates and requires them on every layer. The same checkpoint
serves at TP=4 on the pinned digest with `patches/` (`kvshare.diff`); moving this rig onto that image is the fix.

### Four chips and one KV head

E2B has `num_key_value_heads=1`, and a single KV head does not shard ([MODELS.md](../MODELS.md#single-kv-head-does-not-shard)). At TP=4 every chip holds the whole KV cache, so KV per token per chip stays 18 KiB, as on v5e-1. Query heads split 8 → 2 per chip and the weights split four ways, so each chip's KV pool grows only by the weights it no longer holds. Four chips give E2B compute and per-chip weight room; context per chip grows only by that freed room. No KV pool has been measured for this checkpoint at TP=4: on the 2026-10-02 nightly it fails at model load at both TP=4 and TP=1 (`benchmarks/runs/2026-10-02-rig-boot-v5e4`).

### Baseline: the v5e-1 sibling

This rig provisioned on 2026-10-02 and failed at model load (above), so it has no throughput of its own. The numbers below were measured on **v5e-1** by the sibling, [`tpu-vllm-v5e1-2b-w4a16`](../tpu-vllm-v5e1-2b-w4a16/), served beside the E2B QAT repack on the same VM through [`jev-tpu-v5e1`](../jev-tpu-v5e1/) (`benchmarks/runs/2026-09-30-gspeedb-e2b-google-v5e1` in the sibling): output tok/s at 1 / 4 / 16 requests 136.6 / 532.3 / 1,911 against the repack's 136.5 / 532.1 / 1,910, with the repack 2.4 suite points ahead.

## Current Deployment
*   **Model:** `google/gemma-4-E2B-it-qat-w4a16-ct` on TPU v5e-4 (v5litepod, one host, four chips).
*   **Endpoint:** discovered at runtime — the agent finds the `ACTIVE` Queued Resource,
    resolves its node IP, and serves on port 8000. Ask the agent for `get_vllm_endpoint`,
    or run `make endpoint`. Endpoints are ephemeral; don't hardcode them.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on TPU v5e) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your TPU deployment for the model needs:
*   **Hardware:** Cloud TPU v5e (v5litepod) with topology `2x2` (4 chips, one `ct5lp-hightpu-4t` host: 112 vCPU, 192 GiB RAM).
*   **Software:** `vllm/vllm-tpu:nightly` specialized container (v0.19.2+ recommended for Gemma 4 fixes).
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
*   `GOOGLE_CLOUD_ZONE`: Zone to provision and discover in (defaults to `us-west4-a`).
*   `GOOGLE_CLOUD_REGION`: Region for network resources (defaults to `us-west4`).
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-E2B-it-qat-w4a16-ct`).
*   `ACCELERATOR_TYPE`: TPU accelerator type (defaults to `v5litepod-4` — gcloud's name for v5e-4).
*   `TPU_RUNTIME_VERSION`: TPU VM runtime (defaults to `v2-alpha-tpuv5-lite`).
*   `TPU_QUOTA_ID`: Cloud Quotas id scanned by `find_tpu` (defaults to `TPUV5sLitepodPerProjectPerZoneForTPUAPI`).
*   `TENSOR_PARALLEL_SIZE`: Tensor parallel size (defaults to `4`).
*   `MCP_SERVER_NAME`: Name this server advertises, and the key it must be registered under — it prefixes
    every tool as `mcp__<name>__find_tpu` (defaults to the rig directory name, `tpu-vllm-v5e4-2b-w4a16`). Set it
    only to match a client that already registered this server under a different key. `make mcp-config`
    writes a `.mcp.json` using the same value.

## Technical Standards
-   **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
-   **Optimization Flags:**
    -   `--tensor-parallel-size 4` (v5e-4 is four chips on one host)
    -   `--max-model-len 16384`
    -   `--disable_chunked_mm_input`
    -   `--max_num_batched_tokens 4096` (required for multimodal compatibility)
    -   `--limit-mm-per-prompt '{"image":4,"audio":1}'` (JSON format required in nightly)
-   Every serving setting except the tensor-parallel size is kept identical to the v5e-1 sibling,
    [`tpu-vllm-v5e1-2b-w4a16`](../tpu-vllm-v5e1-2b-w4a16/), on purpose, so the two rigs differ only in chip count and TP.
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