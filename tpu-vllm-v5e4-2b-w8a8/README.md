# TPU vLLM DevOps Agent (MCP Server)

## Role
This project functions as an expert TPU SRE and DevOps Engineer, specialized in the **Gemma 4** ecosystem. Its primary goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted via vLLM on Cloud TPUs**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## What this rig serves: int8 W8A8 from Google's QAT weights

This rig serves **[`xbill9/gemma-4-E2B-it-qat-w8a8-int8`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-w8a8-int8)**:
Gemma 4 E2B-it with int8 weights (one scale per output channel) and activations quantized to
int8 per token, built by `../jev-tpu-31b/w8a8_from_qat.py` from Google's QAT export
(`google/gemma-4-E2B-it-qat-q4_0-unquantized`). Text only. On v5e int8 x int8 runs natively in
the matrix units at twice the bf16 rate.

Slot 5 is `w8a8`: the encoding at the granularity the artifact names it (compressed-tensors W8A8),
as `w4a16` is for the sibling. `int8` alone would also describe a weight-only build.

### Baseline: the v5e-1 sibling

This rig is the four-chip fork of [`../tpu-vllm-v5e1-2b-w8a8`](../tpu-vllm-v5e1-2b-w8a8). The numbers in
this section were measured on **v5e-1** (one chip, TP=1) and are the baseline for this rig's own
measurement in the next section. The serving settings
(`--gpu-memory-utilization`, `--max-model-len`, `--max_num_batched_tokens`, `VLLM_ENV`, image digest) are
identical to the sibling's, so the two rigs differ only in chip count and TP.


v5e-1, one `v5litepod-1`, same flags for every build, suite of 3,880 records read by label probability
(runs in `../jev-tpu-v5e1/results/2026-09-29-*`):

| E2B | Suite | Output tok/s at 1 / 4 / 16 | First-token latency |
|---|---:|---|---:|
| `google/gemma-4-E2B-it` bf16 | 0.683 | 144 / 560 / 2,008 | 12.6 ms |
| W4A16 QAT repack | 0.678 | 136 / 532 / 1,906 | 16.2 ms |
| `glenic/gemma-4-E2B-it-W8A8-INT8` (W8A8 rounded from bf16) | 0.670 | 220 / 842 / 2,876 | 10.7 ms |
| **This checkpoint (v5e-1)** | **0.686** | **220 / 841 / 2,872** | **10.4 ms** |

Paired against bf16: +0.3 points (95% range −0.7 to +1.3). Measured on v5e-1 at `--max-model-len 2048`;
both rigs serve 16,384.

### This rig on v5e-4, 2026-10-02

Flex-start `v5litepod-4` in `us-west4-a`, created through this rig's own `create_tpu_queued_resource`:
capacity within 5 minutes, serving about 13 minutes after creation, text, greedy chat and a tool call
verified ([`benchmarks/runs/2026-10-02-rig-boot-v5e4`](benchmarks/runs/2026-10-02-rig-boot-v5e4/README.md)). The int8 W8A8 patches serve at TP=4.

| | v5e-1 | v5e-4, TP=4 | v5e-4 / v5e-1 |
|---|---:|---:|---:|
| Weights per chip | 6.88 GiB | 1.75 GiB | |
| KV pool | 333,344 tokens | 631,968 tokens | 1.90x |
| Output tok/s, 1 request | 175 | 190 | 1.09x |
| Output tok/s, 4 requests | 581 | 644 | 1.11x |
| Output tok/s, 16 requests | 1,386 | 1,465 | 1.06x |

The v5e-1 column is the v5e-1 rig booted the same day with identical serving arguments
([`../tpu-vllm-v5e1-2b-w8a8/benchmarks/runs/2026-10-02-rig-boot-v5e1`](../tpu-vllm-v5e1-2b-w8a8/benchmarks/runs/2026-10-02-rig-boot-v5e1/README.md)).
**Four chips buy E2B a 1.9x KV pool and up to 11% more output.** The sweep figures above (220 / 841 / 2,872)
were served with `--max-num-seqs 16`, `--max-model-len 2048` and no parsers, worth about 2x at 16 requests on
one chip, so they pair with neither rig. On the pool: E2B has one KV head, which does not shard
([`../MODELS.md`](../MODELS.md)), so every chip holds a copy of the whole KV cache at 18 KiB per token, and
the weight room each chip frees (6.88 -> 1.75 GiB) nearly doubles the pool. Query heads split 8 -> 2 per chip.

### Why it needs a patched image

Upstream vLLM TPU (`tpu_inference`) has no int8 W8A8 method on the JAX path (only fp8 W8A8), and
its `Gemma4ForCausalLM` reads `hf_config.text_config`, which a text-only config lacks. `patches/`
holds the fix, applied in `patches/ORDER` on top of the pinned image
`vllm/vllm-tpu@sha256:19a1a052…` (the digest they are written against):

| Patch | What |
|---|---|
| `kvshare.diff` | vllm-project/tpu-inference#3299: the E2B/E4B KV-shared layers own no K/V parameters |
| `wna16.diff`, `moe.diff` | #3653 and #3660: the W4A16 methods `lowmem.diff` is written on top of |
| `lowmem.diff` | int8 W8A8 (per-channel weights, per-token dynamic activations) on the JAX compressed-tensors path; the `text_config` fallback; opt-in memory and tiling switches |
| `textonly.diff` | the `text_config` fallback in the `load_weights` line `kvshare.diff` adds |

The boot script (`startup_script_template.sh`, rendered by `server.py` with the patches embedded)
pulls the pinned image, applies them, commits `vllm-tpu-w8a8:patched`, and serves that with
`MIN_TOKEN_BUCKET=64`, `GMM_V2_TILE_VMEM_FRACTION=0.85`, `--gpu-memory-utilization 0.80` and
`--max_num_batched_tokens 512`. Boot to ready on the v5e-1 sibling is about 5 minutes; this rig served about 13 minutes after Queued
Resource creation on 2026-10-02 (`benchmarks/runs/2026-10-02-rig-boot-v5e4`).

### v5e-1 sibling, 2026-09-30 (generation, tool calling, long prompts)

Paired against E2B bf16 and E2B QAT bf16 on one v5e chip (`../jev-tpu-v5e1/results/2026-09-30-gen2048a-v5e1-GEN-VS-BF16.md`):

| | this checkpoint | vs E2B bf16 (95% range) | vs QAT bf16 (95% range) |
|---|---:|---|---|
| GSM8K, 2,048-token limit | 0.889 | **−2.0** (−3.4 to −0.8) | −0.8 (−2.0 to +0.5) |
| BFCL | 0.915 | −1.3 | −0.7 |

The suite read level with bf16 (+0.3); GSM8K does not. The loss sits in the QAT weights (QAT bf16 −1.3) and in the int8 rounding on top (−0.8); neither step is significant alone, the two together are. Only 3 answers hit the 2,048-token limit.

Long prompts on v5e-1, output tok/s at 1 / 16 requests: about 1,000 tokens 210 / 2,097; about 3,600 tokens 194 / 1,183, 0.85 s to first token at 16 (bf16: 139 / 1,521 and 130 / 906).

## Current Deployment
*   **Model:** `xbill9/gemma-4-E2B-it-qat-w8a8-int8` on TPU v5e-4 (`v5litepod-4`). Booted and measured 2026-10-02 (`benchmarks/runs/2026-10-02-rig-boot-v5e4`).
*   **Endpoint:** discovered at runtime — the agent finds the `ACTIVE` Queued Resource,
    resolves its node IP, and serves on port 8000. Ask the agent for `get_vllm_endpoint`,
    or run `make endpoint`. Endpoints are ephemeral; don't hardcode them.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on TPU v5e) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your TPU deployment for the model needs:
*   **Hardware:** Cloud TPU v5e (v5litepod) with topology `2x2` (4 chips, one host, `v5litepod-4`).
*   **Software:** `vllm/vllm-tpu@sha256:19a1a052…` with `patches/` applied at boot (see above).
*   **Model:** `xbill9/gemma-4-E2B-it-qat-w8a8-int8` (Hugging Face ID).
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
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `xbill9/gemma-4-E2B-it-qat-w8a8-int8`).
*   `ACCELERATOR_TYPE`: TPU accelerator type (defaults to `v5litepod-4` — gcloud's name for v5e-4).
*   `TPU_RUNTIME_VERSION`: TPU VM runtime (defaults to `v2-alpha-tpuv5-lite`).
*   `TPU_QUOTA_ID`: Cloud Quotas id scanned by `find_tpu` (defaults to `TPUV5sLitepodPerProjectPerZoneForTPUAPI`).
*   `TENSOR_PARALLEL_SIZE`: Tensor parallel size (defaults to `4`).
*   `MCP_SERVER_NAME`: Name this server advertises, and the key it must be registered under — it prefixes
    every tool as `mcp__<name>__find_tpu` (defaults to the rig directory name, `tpu-vllm-v5e4-2b-w8a8`). Set it
    only to match a client that already registered this server under a different key. `make mcp-config`
    writes a `.mcp.json` using the same value.

## Technical Standards
-   **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
-   **Optimization Flags:**
    -   `--tensor-parallel-size 4` (v5e-4 is four chips on one host)
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