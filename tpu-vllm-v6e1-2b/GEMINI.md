# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the project and assist in development. It contains information relevant to the Gemini CLI's operation and interaction with this project, including specific setup instructions for integrating with the CLI.

## Role
This project functions as an expert TPU SRE and DevOps Engineer, specialized in the **Gemma 4** ecosystem. Its primary goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted via vLLM on Cloud TPUs**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## Current Deployment
*   **Model:** `google/gemma-4-E2B-it` on TPU v6e-1 (Trillium).
*   **Endpoint:** discovered at runtime from the `ACTIVE` Queued Resource — use the
    `get_vllm_endpoint` tool or `make endpoint`. Do not hardcode an IP.

vLLM recipes
* https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/inference/trillium/vLLM/Gemma4/README.md

vLLM github
* https://github.com/vllm-project/vllm/releases

chip comparison (third-party, directional only — different model, different harness, and since this rig
moved to v6e-1, a different chip from the one it benchmarks)
* https://deploybase.ai/articles/v5e-1-tpu-vs-t4-gpu

never destroy the queued resource without explicit asking for it

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on TPU v6e) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your TPU deployment for the model needs:
*   **Hardware:** Cloud TPU v6e (Trillium) with topology `1x1` (1 chip).
*   **Software:** `vllm/vllm-tpu:nightly` specialized container (v0.19.2+ recommended for Gemma 4 fixes).
*   **Model:** `google/gemma-4-E2B-it` (Hugging Face ID).
*   **Runtime:** `v2-alpha-tpuv6e` for Flex-start / Queued Resources.
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
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-E2B-it`).
*   `MCP_SERVER_NAME`: Name this server advertises, and the key it must be registered under — it prefixes
    every tool as `mcp__<name>__find_tpu` (defaults to the rig directory name, `tpu-vllm-v6e1-2b`).

## Technical Standards
-   **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
-   **Optimization Flags:**
    -   `--tensor-parallel-size 1` (v6e-1 is a single chip)
    -   `--max-model-len 16384`
    -   `--disable_chunked_mm_input`
    -   `--max_num_batched_tokens 4096` (required for multimodal compatibility)
    -   `--limit-mm-per-prompt '{"image":4,"audio":1}'` (JSON format required in nightly)
-   **Tooling:** Enable `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, and `--reasoning-parser gemma4`.
-   **Image:** `vllm/vllm-tpu:nightly` (v0.19.2+ recommended) is preferred for stable Gemma 4 support.

## Flex-start VMs
Our stack leverages **Flex-start VMs** (via the `v2-alpha-tpuv6e` runtime) to maximize TPU availability and minimize costs.

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

### Step 3: LiteLLM Proxy Setup
To enable seamless integration with the Gemini CLI, you can set up a LiteLLM proxy to route requests to your self-hosted vLLM endpoint on TPU.

#### 1. Install LiteLLM Proxy
You need the [proxy] version of LiteLLM to handle the translation:
```bash
pip install 'litellm[proxy]'
```

#### 2. Create a configuration file (`litellm_config.yaml`)
Create this file to map the Gemini model names used by the CLI to your TPU endpoint:
```yaml
model_list:
  - model_name: "gemma4-tpu"
    litellm_params:
      model: "openai/google/gemma-4-E2B-it" # Tell LiteLLM it's an OpenAI-style endpoint
      api_base: "http://<TPU_VM_IP>:8000/v1" # Get this from `make endpoint`
      api_key: "none" # vLLM doesn't require a key by default
    router_settings:
      model_group_alias:
        # Map common Gemini model names to your TPU-hosted Gemma 4
        "gemini-2.0-flash": "gemma4-tpu"
        "gemini-2.0-flash-lite": "gemma4-tpu"
        "gemini-1.5-flash": "gemma4-tpu"
        "gemini-1.5-pro": "gemma4-tpu"
```
*Note: the TPU IP changes every time the Queued Resource is recreated — re-run
`make endpoint` and update `api_base` before starting the proxy.*

#### 3. Start the LiteLLM Proxy
Run this in a separate terminal (or in the background):
```bash
litellm --config litellm_config.yaml --port 4000
```

#### 4. Configure Gemini CLI to use the Proxy
Set these environment variables in your shell (e.g., in `~/.bashrc` or `~/.zshrc`) to make it permanent:
```bash
# Point the CLI to your local LiteLLM proxy
export GOOGLE_GEMINI_BASE_URL="http://localhost:4000"

# Set the default model globally
export GEMINI_MODEL="google/gemma-4-E2B-it"

# The CLI requires a key even if the proxy ignores it
export GEMINI_API_KEY="local-proxy-token"
```

**Why this works:**
*   **API Translation:** When you run `gemini "Hello"`, the CLI sends a request to `localhost:4000` in Google format. LiteLLM translates this to the OpenAI format and forwards it to your TPU.
*   **Tool Calling Compatibility:** Because we deployed your Gemma 4 stack with `--tool-call-parser gemma4`, the model's reasoning and tool outputs will be perfectly understood by the Gemini CLI when it tries to run shell commands or edit files.

Now, every time you run `gemini`, it will be powered by your private TPU v6e cluster.

## 🛠 Available Tools

The following tools are available via the MCP server:

### Infrastructure & Deployment
*   **`create_tpu_queued_resource`**: Creates a new TPU Queued Resource (Flex-start VM) in the specified zone.
*   **`destroy_queued_resource`**: Safely deletes a Queued Resource and its corresponding TPU node.
*   **`manage_queued_resource`**: Ensures the primary Queued Resource exists and cleans up redundant ones.
*   **`get_zones_with_available_quota`**: Gets GCP zones that have available TPU quota for TPU v6e.
*   **`find_tpu`**: Scans multiple zones sequentially to find and provision a TPU v6e resource.
*   **`find_gpu`**: Scans the GCP project for available GPU resources (GCE VMs, Cloud Run services, and zones with quota).
*   **`get_vllm_deployment_config`**: Generates the `gcloud` command for a single-host TPU v6e vLLM deployment.
*   **`get_vllm_tpu_deployment_config`**: Generates GKE manifests for TPU-based deployments.
*   **`list_queued_resources`**: Lists all Queued Resources in a specific zone.
*   **`describe_queued_resource`**: Provides detailed information about a specific Queued Resource.
*   **`get_reservation_status`**: Checks the lifecycle state and expiry time of a Queued Resource.
*   **`check_tpu_availability`**: Checks if a Queued Resource has reached the `ACTIVE` state.
*   **`manage_vllm_docker`**: Manages the vLLM Docker container on the TPU VM (`start`, `stop`, `restart`, `status`, `log`, and `rm` actions).
*   **`estimate_deployment_cost`**: Estimates the cost of a TPU deployment.

### Observability & Performance
*   **`get_system_status`**: Provides a high-level dashboard of system, TPU, and vLLM status.
*   **`get_vllm_endpoint`**: Returns the active vLLM service URL if available.
*   **`get_deployed_endpoint`**: Returns the raw URL of the active vLLM service.
*   **`get_metrics`**: Fetches raw Prometheus metrics from the running vLLM service's `/metrics` endpoint.
*   **`get_vllm_docker_logs`**: Retrieves logs from the vLLM Docker container on the TPU VM.
*   **`get_tpu_system_logs`**: Retrieves systemd logs for a specific service from the TPU VM.
*   **`get_cloud_logging_logs`**: Fetches logs from Google Cloud Logging.
*   **`get_model_details`**: Retrieves detailed information about the running model, vLLM engine, and versions.
*   **`verify_model_health`**: Runs a deep logic check with latency reporting.
*   **`run_vllm_benchmark`**: Runs vLLM's internal benchmark tool inside the container on the TPU VM.
*   **`get_help`**: Provides help text and summarizes the configuration options and all available SRE/DevOps tools.
*   **`get_active_models`**: Gets the active resource usage via `ollama ps` (if Ollama backend is used).
*   **`get_model_show_details`**: Gets deep model parameters via `ollama show` (if Ollama backend is used).

### AI & Interaction
*   **`query_queued_gemma4`**: Queries the self-hosted Gemma 4 model on the active Queued Resource.
*   **`query_queued_gemma4_with_stats`**: Queries the self-hosted Gemma 4 model and returns detailed performance statistics.
*   **`analyze_cloud_logging`**: Summarizes TPU-related errors using the self-hosted Gemma 4 model.
*   **`save_hf_token`**: Securely saves a Hugging Face API token to GCP Secret Manager.

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities:
```bash
python demo_launcher.py
```
