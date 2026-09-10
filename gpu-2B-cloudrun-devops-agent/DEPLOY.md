# Deployment Guide: Self-Hosted vLLM on Cloud Run (Gemma 4 E2B-it)

This document summarizes the deployment state and configuration for the vLLM inference server used by the DevOps Agent.

## 📦 Model Artifacts
Uploaded 2026-09-10 from Hugging Face `google/gemma-4-E2B-it` at revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7` (`model.safetensors` SHA-256 `2db5482b…c550`).


*   **Bucket:** `gs://aisprint-491218-bucket/`
*   **Path:** `gemma-4-E2B-it/`
*   **Format:** Hugging Face Transformers (Safetensors)

## 🚀 Inference Stack (vLLM)
The inference server is deployed on Cloud Run with GPU acceleration and GCS FUSE.

*   **Service Name:** `gpu-2b-l4-devops-agent`
*   **Service URL:** `https://gpu-2b-l4-devops-agent-289270257791.us-east4.run.app`
*   **Region:** `us-east4`
*   **Hardware:**
    *   **GPU:** 1x NVIDIA L4
    *   **vCPU:** 8
    *   **Memory:** 32GiB
*   **Configuration:**
    *   **Container Port:** `8000`
    *   **Max Model Length:** `16384`
    *   **Storage:** GCS FUSE mounted at `/mnt/models`
    *   **Zonal Redundancy:** Disabled (`--no-gpu-zonal-redundancy`)

## 🛠 Usage
To connect the MCP Agent to this service, export the following environment variables:

```bash
export VLLM_BASE_URL="https://gpu-2b-l4-devops-agent-289270257791.us-east4.run.app"
export MODEL_NAME="/mnt/models/gemma-4-E2B-it"
export GOOGLE_CLOUD_PROJECT="aisprint-491218"
```

Then run the agent:
```bash
make run
```

## 📜 Deployment Command

The `deploy-vllm` target in the `Makefile` is the single source of truth for the deploy configuration.
Deploy with:

```bash
make deploy
```

To see the exact `gcloud` invocation without running it, use `make -n deploy`, or ask the agent for
`cloudrun_get_deployment_config`. Do not hand-write the `gcloud beta run deploy` command — the argument
list is long and order-sensitive, and a copy that drifts from the `Makefile` is how this file came to
record a container port and context length the service never used.
