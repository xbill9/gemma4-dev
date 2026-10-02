# TPU vLLM DevOps Agent (MCP Server)

## Role
This project functions as an expert TPU SRE and DevOps Engineer, specialized in the **Gemma 4** ecosystem. Its primary goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted via vLLM on Cloud TPUs**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## What this rig serves: 12B W8A8 with int4 embeddings

This rig serves **[`xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4`](https://huggingface.co/xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4)**:
Gemma 4 12B-it built from Google's QAT export (`google/gemma-4-12B-it-qat-q4_0-unquantized`), text
only, 11.21 GiB. The linears are int8 W8A8 (one scale per output channel, activations quantized to
int8 per token), from `../jev-tpu-31b/w8a8_from_qat.py`; `embed_tokens` and an untied `lm_head` are
int4 (group 32, fp16 scales), taken byte for byte from the `-emb4` build by
`../jev-tpu-31b/w8a8_emb4.py`. On one v5e chip it is the largest Gemma 4 model measured to serve
(the v5e-1 sibling); this rig serves it on a v5e-4, one host with four chips at `TENSOR_PARALLEL_SIZE=4`.

Slot 5 is `w8a8emb4`: `w8a8` names the linears as `tpu-vllm-v5e1-2b-w8a8` does, and `emb4` records
that the vocabulary tables and `lm_head` are int4, as in `tpu-vllm-v5e1-2b-w4a16emb4`.

### Measured on v5e-4

On 2026-10-02 this rig provisioned (flex-start `v5litepod-4` in `us-west4-a`, capacity within 5 minutes)
and booted through its own `create_tpu_queued_resource` at `TENSOR_PARALLEL_SIZE=4`
([`benchmarks/runs/2026-10-02-rig-boot-v5e4`](benchmarks/runs/2026-10-02-rig-boot-v5e4)). The patched int8 W8A8 linears, int4 embedding tables and int4 `lm_head`
all work at TP=4: chat and a tool call verified, ready about 17.5 minutes after the queued resource was
created (engine init 541 s, 501 s of it compilation).

| | v5e-1 sibling | v5e-4, TP=4 | v5e-4 / v5e-1 |
|---|---:|---:|---:|
| Weights per chip | 11.31 GiB | 3.25 GiB | |
| KV cache | 9,728 tokens | 122,624 tokens | 12.61x |
| Output tok/s, 1 request | 57.0 | 128.4 | 2.25x |
| Output tok/s, 4 requests | 215.7 | 450.9 | 2.09x |
| Output tok/s, 16 requests | 713.4 | 1,212.9 | 1.70x |

Both columns are each rig's own boot run with the same load client (256 output tokens, greedy, median of
three passes). Predicted ~122,400 KV tokens and 3.22 GiB per chip; measured 122,624 and 3.25. At
`MAX_MODEL_LEN` 8192 the pool holds 14.97 requests of full length, so the context limit sits far below
what memory allows; raising it is the next experiment and has no measurement yet.

### Baseline: the v5e-1 sibling

Every number in this section was
measured on one `v5litepod-1` and belongs to [`../tpu-vllm-v5e1-12b-w8a8emb4`](../tpu-vllm-v5e1-12b-w8a8emb4)
and the `../jev-tpu-v5e1` runner (`2026-09-29-12b-v5e1`, `2026-09-30-12bctx5-v5e1`):

*   **v5e-1:** suite 0.761 against bf16's 0.760 (on v6e), 57 / 217 / 723 output tok/s at 1 / 4 / 16 requests,
    11.31 GiB of weights on the chip, 9,728 KV tokens at utilization 0.92, GSM8K 0.964 at a 2,048-token
    limit, BFCL 0.955.
*   **v5e-1, fp8 KV cache:** 18,944 KV tokens, 65% more output at 16 long prompts, −0.7 suite points,
    GSM8K and BFCL level (`../jev-tpu-v5e1/results/2026-09-30-fillF-v5e1-KVFP8-VS-KVBF16.md`).
*   **v5e-1, the sibling's own boot** (`benchmarks/runs/2026-09-30-rig-boot-v5e1` there): ready 17 minutes
    after the queued resource was created, 57.0 / 215.7 / 713.4 output tok/s, within 1.3% of the sweep.

On v5e-1 memory set every limit: raising utilization or adding a block override failed at boot.

### What four chips change (the arithmetic behind the prediction)

12B KV costs 336 KiB per token: 40 sliding layers x 8 KV heads x 256 plus 8 full layers x 1 global KV
head x 512, K and V, bf16. At TP=4 the sliding heads split 8 -> 2 per chip and the single global head
is replicated, so each chip holds 96 KiB per token. Holding the sibling's per-chip budget at 0.92
(14.43 GiB for weights + KV) gives about 122,400 KV tokens: `patches/` keeps the int4 `embed_tokens`
table (0.53 GiB) whole on every chip (`WNA16EmbedMethod`, "Tables are replicated") and splits the rest,
3.22 GiB per chip. Measured: 122,624 tokens and 3.25 GiB per chip (`benchmarks/runs/2026-10-02-rig-boot-v5e4`). `MAX_MODEL_LEN` stays
8192 for now; on this rig memory stops being the limit, so raising it is the next experiment.

### Why it needs a patched image

Upstream vLLM TPU (`tpu_inference`) has no int8 W8A8 method on the JAX path (only fp8), no
quantized embedding table and no quantized `lm_head`, and its `Gemma4ForCausalLM` reads
`hf_config.text_config`, which a text-only config lacks. `patches/` holds the fix, applied in
`patches/ORDER` on top of the pinned image `vllm/vllm-tpu@sha256:19a1a052…`. The patches are
byte-identical to the v5e-1 sibling's:

| Patch | What |
|---|---|
| `kvshare.diff` | vllm-project/tpu-inference#3299: KV-shared layers own no K/V parameters |
| `wna16.diff`, `moe.diff` | #3653 and #3660: the W4A16 methods `lowmem.diff` is written on top of (the int4 `lm_head` uses the W4A16 linear) |
| `lowmem.diff` | int8 W8A8 linears; int4 embedding tables stored packed and padded to 128 columns, with only gathered rows unpacked; an int4 `lm_head` built when a config group targets it; the `text_config` fallback; opt-in memory and tiling switches |
| `textonly.diff` | the `text_config` fallback in the `load_weights` line `kvshare.diff` adds |

`lowmem.diff` is `git diff gemma4-w4a16-moe gemma4-w4a16-moe-lowmem -- tpu_inference` in
`github.com/xbill9/tpu-inference`.

The boot script (`startup_script_template.sh`, rendered by `server.py` with the patches embedded)
pulls the pinned image, applies them, commits `vllm-tpu-w8a8emb4:patched`, and serves that with
`MIN_TOKEN_BUCKET=64`, `GMM_V2_TILE_VMEM_FRACTION=0.85`, `--tensor-parallel-size 4`,
`--gpu-memory-utilization 0.92`, `--max-model-len 8192` and `--max_num_batched_tokens 512`, and waits up
to 60 minutes for readiness. Every setting but the TP size is identical to the v5e-1 sibling's on purpose,
so the two rigs differ only in chip count and TP.

## Current Deployment
*   **Model:** `xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4` on TPU v5e-4 (`v5litepod-4`).
*   **Endpoint:** discovered at runtime — the agent finds the `ACTIVE` Queued Resource,
    resolves its node IP, and serves on port 8000. Ask the agent for `get_vllm_endpoint`,
    or run `make endpoint`. Endpoints are ephemeral; don't hardcode them.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on TPU v5e) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your TPU deployment for the model needs:
*   **Hardware:** Cloud TPU v5e (v5litepod) with topology `2x2` (4 chips, one `ct5lp-hightpu-4t` host: 112 vCPU, 192 GiB RAM).
*   **Software:** `vllm/vllm-tpu@sha256:19a1a052…` with `patches/` applied at boot (see above).
*   **Model:** `xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4` (Hugging Face ID).
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
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4`).
*   `ACCELERATOR_TYPE`: TPU accelerator type (defaults to `v5litepod-4` — gcloud's name for v5e-4).
*   `TPU_RUNTIME_VERSION`: TPU VM runtime (defaults to `v2-alpha-tpuv5-lite`).
*   `TPU_QUOTA_ID`: Cloud Quotas id scanned by `find_tpu` (defaults to `TPUV5sLitepodPerProjectPerZoneForTPUAPI`).
*   `TENSOR_PARALLEL_SIZE`: Tensor parallel size (defaults to `4`).
*   `MCP_SERVER_NAME`: Name this server advertises, and the key it must be registered under — it prefixes
    every tool as `mcp__<name>__find_tpu` (defaults to the rig directory name, `tpu-vllm-v5e4-12b-w8a8emb4`). Set it
    only to match a client that already registered this server under a different key. `make mcp-config`
    writes a `.mcp.json` using the same value.

## Technical Standards
-   **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
-   **Optimization Flags:**
    -   `--tensor-parallel-size 4` (v5e-4 is four chips on one host)
    -   `--max-model-len 8192`
    -   `--disable_chunked_mm_input`
    -   `--max_num_batched_tokens 512`
    -   `--gpu-memory-utilization 0.92`
    -   `--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'` (the checkpoint is text only)
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