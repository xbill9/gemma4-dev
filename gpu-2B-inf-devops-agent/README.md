# Self-Hosted vLLM DevOps Agent (MCP Server)

This project provides an automated DevOps/SRE assistant that leverages **Gemma models self-hosted via vLLM on AWS Inferentia (Inf2)**. It bridges CloudWatch Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on EC2 Inferentia) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your EC2 deployment for the model needs:
*   **Hardware Platform:** AWS Inferentia (Inf2 instances).
*   **Neuron Accelerator:** 1x AWS Inferentia2 Device (2 Neuron Cores) e.g., `inf2.xlarge`.
*   **Storage:** An S3 Bucket containing the Gemma model weights (e.g., `s3://vllm-models-bucket/google/gemma-4-E2B-it/`).
*   **Image:** `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04`

### 2. Software & API Dependencies
The agent relies on AWS services and Python libraries:
*   **Libraries:** `mcp`, `fastmcp`, `boto3`, `httpx`, `openai`, and `huggingface_hub`.
*   **Permissions:** The IAM credentials running the agent need:
    *   `cloudwatch:GetLogEvents` / `logs:FilterLogEvents` (to read logs).
    *   Access to read/write from the target S3 bucket.
    *   Access to the vLLM endpoint (port 8080 on the EC2 instance).

### 3. Environment Variables
You can configure the following variables for the MCP server:
*   `AWS_DEFAULT_REGION`: Your AWS region (defaults to `us-east-1`).
*   `AWS_BUCKET_NAME`: S3 bucket name (defaults to `vllm-models-bucket`).
*   `VLLM_BASE_URL`: The URL of your EC2 vLLM service. **If omitted, the agent will attempt to auto-discover it using EC2 tags.**
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-E2B-it`).

## 🛠 Usage & Setup

### Step 1: Prepare Model Weights
Use the built-in tool `get_huggingface_model_copy_instructions` to download Gemma weights and upload to your S3 bucket.

### Step 2: Deploy vLLM to AWS EC2 (Inferentia)
Run the `get_vllm_deployment_config` tool within the MCP server to generate the exact deployment commands, or deploy via the provided [Makefile](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/Makefile):
```bash
make deploy
```

### Step 3: Run the MCP Server
Install dependencies and run the server:
```bash
make install
# Optional: export VLLM_BASE_URL="your-vllm-url"
make run
```

## 🛠 Available Tools

The following tools are available via the MCP server:

### 🐳 Infrastructure & Deployment
*   **[start_ec2](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L948)**: Starts an existing stopped EC2 instance, or provisions a new one (with AWS Inferentia) if none exists.
*   **[status_ec2](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1010)**: Checks the state, type, public IP, DNS, and launch details of EC2 instances.
*   **[stop_ec2](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1304)**: Safely stops active EC2 instances without deleting the root EBS volumes.
*   **[check_vllm](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1126)**: Checks the status of the vLLM container and engine running on the EC2 instance(s).
*   **[deploy_vllm](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L712)**: Deploys vLLM to AWS EC2.
*   **[destroy_vllm](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1261)**: Cleans up the vLLM Docker container on the AWS EC2 instance without terminating it.
*   **[status_vllm](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1348)**: Checks the status of the AWS EC2 instance vLLM service.
*   **[update_vllm_scaling](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1381)**: Scales EC2 instance type vertically.
*   **[get_vllm_deployment_config](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L606)**: Generates the AWS EC2 deployment command and user data.
*   **[get_vllm_gpu_deployment_config](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1625)**: Generates an AWS EKS nodegroup config and Kubernetes manifest for Inferentia.
*   **[check_gpu_quotas](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1739)**: Checks Inferentia/Neuron quotas for an AWS region.

### 📦 Model Management
*   **[list_bucket_models](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L235)**: Lists model weights in S3 bucket.
*   **[save_hf_token](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L49)**: Securely saves a Hugging Face API token to AWS Secrets Manager.
*   **[get_huggingface_model_copy_instructions](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1756)**: Instructions to download model from Hugging Face and upload to S3.
*   **[get_huggingfacehub_download_path](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1702)**: Resolves local cache path using huggingface_hub.

### 📊 Monitoring & Status
*   **[get_metrics](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L2015)**: Fetches raw Prometheus metrics from the running vLLM service's /metrics endpoint.
*   **[get_system_status](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1925)**: Provides a high-level status dashboard of the service and health.
*   **[get_endpoint](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L2042)**: Verifies connectivity and returns the active service URL.
*   **[get_model_details](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1886)**: Retrieves detailed model metadata and engine state from `/v1/models`.
*   **[verify_model_health](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1775)**: Deep health check by querying the model with a prompt and measuring latency.

### 📈 Performance & Benchmarking
*   **[run_benchmark](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L2066)**: Runs performance/concurrency benchmark sweeps against the vLLM Inferentia endpoint.

### 💬 Interaction & Diagnostics
*   **[query_gemma4](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1804)**: Primary tool to query the self-hosted model with standard chat message format.
*   **[query_gemma4_with_stats](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L1823)**: Queries the model and returns streaming performance statistics (TTFT, throughput).
*   **[query_vllm](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L368)**: Direct text completions querying tool.
*   **[analyze_cloud_logging](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L297)**: Fetches logs from AWS CloudWatch and analyzes them using the model.
*   **[analyze_gpu_logs](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L2215)**: Fetches service logs and uses Gemma 4 to analyze them for SRE/DevOps errors.
*   **[suggest_sre_remediation](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L343)**: Suggests remediation plans for SRE errors using the model.
*   **[get_help](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py#L2228)**: Provides help text and summarizes the configuration options and all available SRE/DevOps tools.

## 📦 Resources
The server exposes the following MCP resources:
*   **`config://vllm-deployment-template`**: A YAML template for AWS EC2 Inferentia deployment.

## 📊 Performance (Gemma-4-E2B on Inferentia2)

Stock vLLM/NxD emits gibberish for Gemma-4 because of its cross-layer KV-sharing, so this
project ships the **Option B** port instead: a `torch_neuronx`-traced two-graph
(prefill + KV-cache decode) server. Measured decode throughput:
* **~44 tok/s** on `inf2.8xlarge` (full `xbill9/gemma4-optb:latest` image, 128 GB host RAM).
* **~24 tok/s** on `inf2.xlarge` (host-only bf16 embeddings `xbill9/gemma4-optb:slim`, 16 GB host RAM — **swap required** to survive the ~14.5 GB neff-load peak).
* Context up to 512 total / 128 prompt tokens with the published `512-128` neff set.
* Details and the recipe are in [nxd-gemma4-inf2-work/optb/README.md](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/nxd-gemma4-inf2-work/optb/README.md); benchmark tooling in [benchmark_report_summary.md](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/benchmark_report_summary.md).

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities:
```bash
python demo_launcher.py
```
This script simulates log analysis, remediation suggestions, and infrastructure configuration generation.

## 🛠 Makefile Helpers
The included [Makefile](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/Makefile) provides several shortcuts:
*   `make install`: Installs Python dependencies listed in [requirements.txt](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/requirements.txt).
*   `make run`: Starts the MCP server via [server.py](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/server.py).
*   `make deploy`: Deploys the inference server to AWS EC2 Inferentia (Inf2).
*   `make destroy`: Tears down the EC2 inference deployment.
*   `make status`: Checks the status of the inference service.
*   `make query PROMPT="your prompt"`: Queries the vLLM model directly via `curl`.
*   `make test`: Runs the test suite in [test_agent.py](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/test_agent.py).

## 🧪 Testing
Run the included test suite in [test_agent.py](file:///home/xbill/gemma4-tips-aws/gpu-2B-inf-devops-agent/test_agent.py) to verify the tool registration and basic functionality:
```bash
make test
```
