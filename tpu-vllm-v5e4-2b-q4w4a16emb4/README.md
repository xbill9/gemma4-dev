# TPU vLLM DevOps Agent (MCP Server)

## Role
This project functions as an expert TPU SRE and DevOps Engineer, specialized in the **Gemma 4** ecosystem. Its primary goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted via vLLM on Cloud TPUs**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## What this rig serves: W4A16 with int4 embeddings

This rig serves **[`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4)**:
Gemma 4 E2B-it repacked from Google's QAT export (`google/gemma-4-E2B-it-qat-q4_0-unquantized`)
with every large tensor int4, text only. The linears are the W4A16 repack
(`../jev-tpu-31b/repack_q4_0.py`); `embed_int4.py` then packs the three vocabulary tables the
same way: symmetric int4, group 32, fp16 scales.

| Tensor | `-text` build | This checkpoint |
|---|---:|---:|
| `embed_tokens_per_layer` (262144 × 8960) | 4.375 GiB bf16 | 1.230 GiB int4 |
| `embed_tokens` (262144 × 1536) | 0.750 GiB bf16 | 0.211 GiB int4 |
| `lm_head` | tied to `embed_tokens` | 0.211 GiB int4, untied copy |
| Checkpoint | 6.11 GiB | **2.64 GiB** |

QAT put the embedding tables on the same 4-bit grid as the linears, so the repack recovers each
group's step rather than choosing new ones. Read back through this rig's patched `tpu_inference`
on real rows, 90–92% of values are bit-identical to the bf16 source and the relative error is
0.17–0.23% per row, the same as an independent numpy dequantization of the same rows.

Slot 5 is `q4w4a16emb4`. `q4w4a16` names the linears: W4A16 holding the QAT Q4_0 grid exactly, which
separates this repack from Google's `-qat-w4a16-ct` (`w4a16`), whose min-max rounding moves every
value. `emb4` records that the embedding tables and `lm_head` are int4 as well.

### Baseline: the v5e-1 sibling

Every number in this section was measured
on **one `v5litepod-1`** by [`../tpu-vllm-v5e1-2b-q4w4a16emb4`](../tpu-vllm-v5e1-2b-q4w4a16emb4) (runner
`../jev-tpu-v5e1` at `--max-model-len 2048`, 2026-09-29): weights 2.66 GiB on the chip, 578,944 KV tokens
(1.61x the W4A16 repack's), suite 0.681 against bf16's 0.683 (−0.2, −1.2 to +0.7), **150 / 583 / 2,067
output tok/s at 1 / 4 / 16 requests** against bf16's 144 / 560 / 2,008. That needs the tables padded to
128 columns (in `patches/lowmem.diff`); unpadded, the lookup read the whole table every step and the model
decoded at 85 tok/s. The W8A8 + int4-table build of the same model runs at 243 / 923 / 3,086 on v5e-1
(`../jev-tpu-v5e1/README.md`).

### What four chips change

Measured 2026-10-02 on `v5litepod-4` (flex-start, `us-west4-a`), booted by this rig's own
`create_tpu_queued_resource` ([`benchmarks/runs/2026-10-02-rig-boot-v5e4`](benchmarks/runs/2026-10-02-rig-boot-v5e4/README.md)). The patched W4A16 linears, int4 tables and
int4 `lm_head` serve at tensor parallelism 4: `/v1/models`, a greedy chat answer and a tool call verified.

| | v5e-1 | v5e-4, TP=4 | v5e-4 / v5e-1 |
|---|---:|---:|---:|
| Weights per chip | 2.84 GiB | 1.95 GiB | |
| KV cache | 578,944 tokens | 620,512 tokens | 1.07x |
| Output tok/s, 1 request | 150 | 187.2 | 1.25x |
| Output tok/s, 4 requests | 583 | 613.2 | 1.05x |
| Output tok/s, 16 requests | 2,067 | 1,424.0 | 0.69x |

Four chips run one request faster and 16 requests slower than one chip. The KV pool grows by 7%: E2B has
one KV head, and a single KV head does not shard (`../MODELS.md`), so every chip holds the whole KV cache at
18 KiB per token, as on v5e-1, while query heads split 8 -> 2 per chip. The pool grows only by the weights
each chip no longer holds, and `patches/lowmem.diff` replicates the int4 embedding tables (1.44 GiB of the
2.64) on every chip. Predicted about +52,000 KV tokens, measured +41,568.

Ready came about 23 minutes after the Queued Resource was created: 17 minutes after `docker run`, 12.5 of
them precompiling subgraphs. The boot script's readiness wait is 20 minutes from `docker run`, and this boot
cleared it at check 103 of 120. A slower boot logs `ERROR: vLLM did not report 'Application startup
complete.' within the timeout.` while the container keeps compiling; check `docker logs vllm-gemma4` before
treating that as a failure.

### Why it needs a patched image

Upstream vLLM TPU (`tpu_inference`) has no W4A16 method on the JAX path, no quantized embedding
table and no quantized `lm_head`, and its `Gemma4ForCausalLM` reads `hf_config.text_config`,
which a text-only config lacks. `patches/` holds the fix, applied in `patches/ORDER` on top of the
pinned image `vllm/vllm-tpu@sha256:19a1a052…` (the digest they are written against):

| Patch | What |
|---|---|
| `kvshare.diff` | vllm-project/tpu-inference#3299: the E2B/E4B KV-shared layers own no K/V parameters |
| `wna16.diff`, `moe.diff` | #3653 and #3660: the W4A16 linear and MoE methods |
| `lowmem.diff` | int4 embedding tables (`WNA16EmbedMethod`: packed on the device, only gathered rows unpacked) and an int4 `lm_head` built when a config group targets it; the `text_config` fallback; int8 W8A8; opt-in memory and tiling switches |
| `textonly.diff` | the `text_config` fallback in the `load_weights` line `kvshare.diff` adds |

`lowmem.diff` is `git diff gemma4-w4a16-moe gemma4-w4a16-moe-lowmem -- tpu_inference` in
`github.com/xbill9/tpu-inference`. The int4 tables keep the checkpoint's fp16 scales; the
`lm_head`'s fp16 scales widen to float32 for the kernel.

The boot script (`startup_script_template.sh`, rendered by `server.py` with the patches embedded)
pulls the pinned image, applies them, commits `vllm-tpu-q4w4a16emb4:patched`, and serves that with
`MIN_TOKEN_BUCKET=64`, `GMM_V2_TILE_VMEM_FRACTION=0.85`, `--gpu-memory-utilization 0.80` and
`--max_num_batched_tokens 512`, the settings the E2B W4A16 repack was measured with on v5e-1. They are
kept identical to the v5e-1 sibling's on purpose, so the two rigs differ only in chip count and TP.

## Current Deployment
*   **Model:** `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` on TPU v5e-4 (v5litepod, four chips on one host).
*   **Endpoint:** discovered at runtime — the agent finds the `ACTIVE` Queued Resource,
    resolves its node IP, and serves on port 8000. Ask the agent for `get_vllm_endpoint`,
    or run `make endpoint`. Endpoints are ephemeral; don't hardcode them.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on TPU v5e) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your TPU deployment for the model needs:
*   **Hardware:** Cloud TPU v5e (v5litepod) with topology `2x2` (4 chips, one `ct5lp-hightpu-4t` host: 112 vCPU, 192 GiB RAM).
*   **Software:** `vllm/vllm-tpu@sha256:19a1a052…` with `patches/` applied at boot (see above).
*   **Model:** `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` (Hugging Face ID).
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
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`).
*   `ACCELERATOR_TYPE`: TPU accelerator type (defaults to `v5litepod-4` — gcloud's name for v5e-4).
*   `TPU_RUNTIME_VERSION`: TPU VM runtime (defaults to `v2-alpha-tpuv5-lite`).
*   `TPU_QUOTA_ID`: Cloud Quotas id scanned by `find_tpu` (defaults to `TPUV5sLitepodPerProjectPerZoneForTPUAPI`).
*   `TENSOR_PARALLEL_SIZE`: Tensor parallel size (defaults to `4`).
*   `MCP_SERVER_NAME`: Name this server advertises, and the key it must be registered under — it prefixes
    every tool as `mcp__<name>__find_tpu` (defaults to the rig directory name, `tpu-vllm-v5e4-2b-q4w4a16emb4`). Set it
    only to match a client that already registered this server under a different key. `make mcp-config`
    writes a `.mcp.json` using the same value.

## Technical Standards
-   **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
-   **Optimization Flags:**
    -   `--tensor-parallel-size 4` (v5e-4 is four chips)
    -   `--max-model-len 16384`
    -   `--disable_chunked_mm_input`
    -   `--max_num_batched_tokens 4096` (required for multimodal compatibility)
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