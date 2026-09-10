# 🤖 Gemini Workspace Context: GPU vLLM DevOps Agent

This context guide summarizes the configuration, optimal serving parameters, and capabilities of the self-hosted **Gemma 4 DevOps/SRE Agent** running on **Cloud Run GPU** (NVIDIA L4).

---

## ⚙️ Active Environment Configuration

This agent targets Google Cloud Platform (GCP) deployments utilizing:
- **Project ID**: `aisprint-491218` (configurable via `GOOGLE_CLOUD_PROJECT`)
- **Region**: `us-east4` (configurable via `GOOGLE_CLOUD_LOCATION`)
- **Default Model**: `/mnt/models/gemma-4-E2B-it` (configurable via `MODEL_NAME`) — the container is
  started with `--model=/mnt/models/<path>`, so that path, not the Hugging Face repo id, is the model
  name the OpenAI API expects
- **GPU Accelerator**: NVIDIA L4 GPU (1 unit) on Cloud Run Gen2
- **Default Cloud Run Service Name**: `gpu-2b-l4-devops-agent` (configurable via `SERVICE_NAME`) — when
  `VLLM_BASE_URL` is also pinned it must point at this same service, since the pin is only honoured for
  the default service name

`server.py` reads these from the process environment only (no `.env` loading). Export them with
`source ./set_env.sh` — add `--resolve-endpoint` to pin `VLLM_BASE_URL` to the deployed service URL,
otherwise it is left unset and discovered via `gcloud` at call time. Auth: `source ./set_adc.sh`.

---

## 🚀 Recommended vLLM Configuration for Gemma 4

### Container Image (pinned — do not use `:latest`)

```
vllm/vllm-openai:v0.26.0-cu129
```

Pin the image tag. `:latest` is a moving tag and has already changed CUDA major version underneath
this deploy once (on 2026-07-25 it moved from the CUDA 12.x line to a CUDA 13.0 base). Pinning makes
deploys reproducible; `v0.26.0-cu129` is the same vLLM version as `v0.26.0`, built against CUDA 12.9,
and retains `TORCH_CUDA_ARCH_LIST` coverage for L4 (compute capability 8.9).

### Host GPU driver constraint

The Cloud Run L4 fleet runs NVIDIA driver **535.129.03 (CUDA 12.2)**. It is managed by Google and
cannot be changed from the container. Every image choice must work against that driver.

A CUDA **12.9** runtime runs on this driver via CUDA minor-version compatibility, with no compat
libraries required. This is the configuration currently deployed and verified.

### Known failure: CUDA error 803 on startup

If the revision fails its startup probe with:

```
RuntimeError: Unexpected error from cudaGetDeviceCount().
Error 803: system has unsupported display driver / cuda driver combination
```

this is a CUDA driver mismatch raised in `init_device()`, before any VRAM is allocated — it looks
like a GPU or capacity problem but is not one.

The cause is `VLLM_ENABLE_CUDA_COMPATIBILITY=1`. Setting it to `1` makes the entrypoint prepend
bundled CUDA forward-compatibility libraries to `LD_LIBRARY_PATH`, replacing the host `libcuda.so`;
when those compat libs do not match the host kernel driver, the result is error 803. Upstream images
ship this set to `0`, and `deploy-vllm` now sets `0` explicitly. **Do not set it to `1`.**

Confirmed on 2026-08-03: the error reproduced with both the CUDA 13.0 (`v0.26.0`) and CUDA 12.9
(`v0.26.0-cu129`) images while compat was `1`, and cleared on `v0.26.0-cu129` with compat `0`. The
container's CUDA version was never the trigger — note that the long-serving May 2026 revision ran
`v0.22.0`, which is *also* a CUDA 13.0.2 image.

### Known failure: HTTP 503 "Service is disabled" with a healthy container

If the revision reports `ContainerHealthy: True` but every request returns 503 in under 200ms and no
request appears in the revision logs, check the service scaling annotations:

```
run.googleapis.com/scalingMode: manual
run.googleapis.com/manualInstanceCount: '0'
```

Manual scaling pinned at zero instances means there is nothing to route to, and Cloud Run reports
that as 503 at the frontend. This is independent of `--min-instances` / `--max-instances`, which do
not set the scaling *mode*. `deploy-vllm`, `cloudrun_deploy`, and `cloudrun_get_deployment_config` all
pass `--scaling=auto` to prevent a stale manual-scaling setting from being inherited across deploys.
To deploy with manual scaling on purpose (e.g. a demo that must never cold-start), use
`make deploy SCALING=1`; gcloud only accepts a positive count, so this can't pin the service at zero.

Fix on a live service with the `cloudrun_update_scaling` tool (which also passes `--scaling=auto`), or
by hand:

```bash
gcloud beta run services update gpu-2b-l4-devops-agent \
  --region=us-east4 --scaling=auto --min-instances=0 --max-instances=1
```

### Startup Arguments

The `deploy-vllm` target in the `Makefile` is the source of truth for this configuration — these values
are transcribed from it. If the two ever disagree, the `Makefile` wins and this section is stale.

```yaml
args:
  - --model
  - /mnt/models/gemma-4-E2B-it
  - --dtype
  - bfloat16
  - --max-model-len
  - "16384"
  - --disable-chunked-mm-input
  - --gpu-memory-utilization
  - "0.95"
  - --kv-cache-dtype
  - fp8
  - --tensor-parallel-size
  - "1"
  - --max-num-seqs
  - "8"
  - --enable-chunked-prefill
  - --max-num-batched-tokens
  - "4096"
  - --enable-auto-tool-choice
  - --tool-call-parser
  - gemma4
  - --reasoning-parser
  - gemma4
  - --async-scheduling
  - --limit-mm-per-prompt
  - "{}"
  - --host
  - 0.0.0.0
  - --port
  - "8000"
```

Cloud Run side: `--memory=32Gi`, `--cpu=8`, `--concurrency=4`, `--port=8000`, `--timeout=3600`,
`--scaling=auto` with `--min-instances=0 --max-instances=1` (or `--scaling=<n>` alone with `make deploy
SCALING=<n>`), `--no-cpu-throttling`, and a startup probe
on `/health` with `initialDelaySeconds=180`.

### Key Parameters Explained
*   **`--gpu-memory-utilization 0.95`**: Allocates 95% of VRAM to vLLM's weights and KV cache.
*   **`--kv-cache-dtype fp8`**: Uses FP8 for KV cache, reducing memory consumption and increasing capacity.
*   **`--tool-call-parser gemma4` & `--reasoning-parser gemma4`**: Essential settings for correct parsing of tool calls and structured reasoning steps generated by Gemma 4.
*   **`--enable-auto-tool-choice`**: Prompts the model to automatically select registered tools.
*   **`--max-model-len 16384`**: Context window that fits alongside the model weights in 24GB of L4 VRAM.
    The model supports far more; this is a memory-driven ceiling, not a model limit.
*   **`--limit-mm-per-prompt {}`** with **`--disable-chunked-mm-input`**: Disables multimodal inputs — this
    deploy serves text only.

---

## 🧰 Key SRE & DevOps Capabilities

This agent exposes several tool categories via the Model Context Protocol (MCP):
- **Deployment & Scaling:** `cloudrun_deploy`, `cloudrun_destroy`, `cloudrun_status`, `cloudrun_update_scaling`, `cloudrun_get_deployment_config`, `cloudrun_get_gpu_deployment_config`, `cloudrun_check_gpu_quotas`, `cloudrun_get_endpoint_url`.
- **Model Transfer & Secret Management:** `cloudrun_list_vertex_models`, `cloudrun_list_bucket_models`, `cloudrun_save_hf_token`, `cloudrun_get_vertex_ai_model_copy_instructions`, `cloudrun_get_huggingface_model_copy_instructions`, `cloudrun_get_huggingfacehub_download_path`.
- **System Monitoring & Health:** `cloudrun_get_system_status`, `cloudrun_get_endpoint`, `cloudrun_get_model_details`, `cloudrun_verify_model_health`, `cloudrun_get_metrics`.
- **Performance Benchmarking:** `cloudrun_run_benchmark` (supports custom concurrency and request count sweeps).
- **Diagnostics & SRE Remediation:** `cloudrun_query_gemma4`, `cloudrun_query_gemma4_with_stats`, `cloudrun_query`, `cloudrun_analyze_cloud_logging`, `cloudrun_analyze_gpu_logs`, `cloudrun_suggest_sre_remediation`, `cloudrun_get_help`.

---

## 🛠 Command Line Setup

### Deploy/Run Quickstart
```bash
# 1. Install dependencies
make install

# 2. Deploy vLLM to Cloud Run (with NVIDIA L4)
make deploy

# 3. Check deployment status
make status

# 4. Start the MCP server locally
make run
```

---

## 📚 Key Source Code File Locations
- **MCP Server entrypoint**: [server.py](file:///home/xbill/gemma4-dev/gpu-2B-cloudrun-devops-agent/server.py)
- **Deployment Manifests & Logic**: Generated by `cloudrun_get_deployment_config` and `cloudrun_get_gpu_deployment_config` in [server.py](file:///home/xbill/gemma4-dev/gpu-2B-cloudrun-devops-agent/server.py).
- **Test Suite**: [test_agent.py](file:///home/xbill/gemma4-dev/gpu-2B-cloudrun-devops-agent/test_agent.py)
- **Standalone Grand Demo**: [demo_launcher.py](file:///home/xbill/gemma4-dev/gpu-2B-cloudrun-devops-agent/demo_launcher.py)
