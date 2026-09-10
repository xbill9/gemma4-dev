import asyncio
import csv
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import time
from typing import Optional

import httpx
from google.cloud import aiplatform, secretmanager, storage
from google.cloud import logging as cloud_logging
from mcp.server.mcpserver import MCPServer
from openai import AsyncOpenAI

# Setup logging to stderr ONLY to avoid interfering with MCP stdio communication
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vllm-devops-agent")
logger.info("Initializing DevOps Agent MCP Server...")

# Initialize MCP server (mcp 2.x; FastMCP was renamed MCPServer)
mcp = MCPServer("Self-Hosted vLLM DevOps Agent")

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east4")
BUCKET_NAME = f"{PROJECT_ID}-bucket"
# The URL of the self-hosted vLLM service on Cloud Run
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
# The container is started with --model=/mnt/models/<path>, so that path — not the HF repo id — is
# the model name the OpenAI API expects. Prefer get_active_model_name() over trusting this default.
MODEL_NAME = os.getenv("MODEL_NAME", "/mnt/models/gemma-4-E2B-it")
HF_SECRET_ID = "hf-token"


async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    try:
        response = await asyncio.to_thread(client.access_secret_version, request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception:
        return None


@mcp.tool()
async def cloudrun_save_hf_token(token: str) -> str:
    """Securely saves a Hugging Face API token to GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    secret_parent = f"projects/{PROJECT_ID}/secrets/{HF_SECRET_ID}"

    try:
        # Check if the secret already exists
        await asyncio.to_thread(client.get_secret, request={"name": secret_parent})
    except Exception:
        # If not, create it
        await asyncio.to_thread(
            client.create_secret,
            request={
                "parent": f"projects/{PROJECT_ID}",
                "secret_id": HF_SECRET_ID,
                "secret": {"replication": {"automatic": {}}},
            },
        )

    # Add the new version
    response = await asyncio.to_thread(
        client.add_secret_version,
        request={"parent": secret_parent, "payload": {"data": token.encode("UTF-8")}},
    )
    return f"✅ Token saved. Version: {response.name}"


# The service every tool acts on unless the caller passes an explicit service_name. Env-driven so
# that `SERVICE_NAME=... source ./set_env.sh --resolve-endpoint` pins VLLM_BASE_URL and names the
# same service — otherwise the pinned URL would be reported under the hardcoded default's name.
DEFAULT_SERVICE_NAME = os.getenv("SERVICE_NAME", "gpu-2b-l4-devops-agent")


def discover_vllm_url(service_name: str = DEFAULT_SERVICE_NAME) -> Optional[str]:
    """Attempts to automatically discover the Cloud Run service URL.

    VLLM_BASE_URL overrides discovery, but only for the default service. It names one specific
    endpoint, so honouring it for every service_name would make every tool report the default
    service's URL — and its health — under whatever name the caller passed.
    """
    if VLLM_BASE_URL and service_name == DEFAULT_SERVICE_NAME:
        logger.info(f"Using provided VLLM_BASE_URL: {VLLM_BASE_URL}")
        return VLLM_BASE_URL

    logger.info(f"Attempting to discover vLLM URL for service: {service_name}")
    try:
        cmd = [
            "gcloud",
            "run",
            "services",
            "describe",
            service_name,
            f"--project={PROJECT_ID}",
            "--region",
            LOCATION,
            "--format",
            "value(status.url)",
        ]
        # Added timeout and improved error handling
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if process.returncode == 0:
            url = process.stdout.strip()
            if url:
                logger.info(f"📡 Automatically discovered vLLM at: {url}")
                return url
            else:
                logger.warning("⚠️ gcloud returned empty URL for service.")
        else:
            logger.warning(
                f"⚠️ gcloud failed to discover Cloud Run URL (code {process.returncode}): {process.stderr.strip()}"
            )
    except subprocess.TimeoutExpired:
        logger.warning("⚠️ Discovery timed out after 15 seconds.")
    except Exception as e:
        logger.warning(f"⚠️ Error during vLLM discovery: {str(e)}")

    logger.error("❌ Failed to discover Cloud Run URL and localhost fallback is disabled.")
    return None


# Resolved base URLs, keyed by Cloud Run service name
_VLLM_URL_CACHE: dict[str, str] = {}


def invalidate_vllm_url_cache(service_name: Optional[str] = None) -> None:
    """Drops cached service URLs so the next lookup re-discovers via gcloud.

    Call this after any operation that creates, replaces, or deletes a service. An explicit
    VLLM_BASE_URL still wins for the default service — it is a deliberate pin, not a cache entry.
    """
    if service_name is None:
        _VLLM_URL_CACHE.clear()
    else:
        _VLLM_URL_CACHE.pop(service_name, None)


def get_vllm_url(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """Returns the cached vLLM URL for a service or discovers it if needed."""
    cached = _VLLM_URL_CACHE.get(service_name)
    if cached:
        return cached

    url = discover_vllm_url(service_name)
    if not url:
        raise Exception(
            f"Could not determine the vLLM Cloud Run URL for '{service_name}'. "
            "Ensure you are authenticated with gcloud and the service exists."
        )

    _VLLM_URL_CACHE[service_name] = url
    return url


def get_auth_token() -> str:
    """Gets a Google Cloud Identity Token for authenticating to Cloud Run."""
    try:
        # Use a timeout for the token generation too
        return (
            subprocess.check_output(
                ["gcloud", "auth", "print-identity-token"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        logger.warning(f"⚠️ Could not obtain identity token: {str(e)}")
        return ""


async def get_vllm_client(service_name: str = DEFAULT_SERVICE_NAME) -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for a Cloud Run vLLM service."""
    vllm_url = get_vllm_url(service_name)
    token = get_auth_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return AsyncOpenAI(
        base_url=f"{vllm_url}/v1",
        api_key=token or "not-needed",
        default_headers=headers,
    )


async def get_active_model_name(client: AsyncOpenAI) -> str:
    """Queries the vLLM endpoint to find the active model name, or falls back to configuration."""
    try:
        models_response = await client.models.list()
        if models_response.data:
            return models_response.data[0].id
    except Exception as e:
        logger.warning(f"⚠️ Failed to dynamically query active model from vLLM: {e}")

    # Fallback
    if "/" not in MODEL_NAME:
        return f"/mnt/models/{MODEL_NAME}"
    return MODEL_NAME


# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)


@mcp.resource("config://vllm-deployment-template")
def get_deployment_template() -> str:
    """Returns a base template for Cloud Run GPU deployment."""
    return f"""
# Cloud Run vLLM Deployment Template
# Required: Second Generation execution environment
# Required: NVIDIA L4 GPU
# Required: GCS FUSE mount

service: {DEFAULT_SERVICE_NAME}
image: vllm/vllm-openai:v0.26.0-cu129  # Pin the tag; :latest has silently changed CUDA major version
resources:
  limits:
    nvidia.com/gpu: 1
    cpu: 8
    memory: 32Gi
annotations:
  run.googleapis.com/execution-environment: gen2
  run.googleapis.com/gpu-zonal-redundancy-disabled: "true"
  run.googleapis.com/cpu-throttling: "false"  # Mandatory for GPU
  run.googleapis.com/startup-cpu-boost: "true"
  run.googleapis.com/scalingMode: automatic  # Manual scaling at 0 instances 503s on every request
  run.googleapis.com/maxScale: "1"
env:
  VLLM_ENABLE_CUDA_COMPATIBILITY: "0"  # "1" prepends compat libs and fails with CUDA error 803
container:
  concurrency: 4
  timeout: 3600s
startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 180
  periodSeconds: 60
  failureThreshold: 10
  timeoutSeconds: 60
# For gcloud deployment, use `make deploy` — the deploy-vllm target in the Makefile is the source of
# truth. The equivalent invocation is:
# gcloud beta run deploy {DEFAULT_SERVICE_NAME} --no-cpu-throttling --no-allow-unauthenticated --concurrency=4 \\
#   --timeout=3600 --startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=10,initialDelaySeconds=180,httpGet.port=8000,httpGet.path=/health \\
#   --scaling=auto --max-instances=1 --set-env-vars=VLLM_ENABLE_CUDA_COMPATIBILITY=0 --args=--model=/mnt/models/gemma-4-E2B-it,--dtype=bfloat16,--max-model-len=16384,--disable-chunked-mm-input,--gpu-memory-utilization=0.95,--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={{}},--host=0.0.0.0,--port=8000
volumes:
  - name: model-volume
    cloudStorage:
      bucket: {BUCKET_NAME}
      readonly: true
"""


@mcp.tool()
def cloudrun_get_endpoint_url(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Returns the current active vLLM endpoint URL without checking whether it responds.

    Args:
        service_name: The Cloud Run service name to describe (defaults to 'gpu-2b-l4-devops-agent').
    """
    try:
        return f"✅ Endpoint for `{service_name}`: {get_vllm_url(service_name)}"
    except Exception as e:
        return f"🔴 Could not resolve the endpoint for `{service_name}`: {e}"


@mcp.tool()
def cloudrun_list_vertex_models() -> str:
    """
    Uses the Vertex AI SDK (part of ADK ecosystem) to list models in the project registry.
    """
    try:
        models = aiplatform.Model.list()
        if not models:
            return "No models found in Vertex AI Model Registry."

        model_list = [f"- {m.display_name} (ID: {m.name})" for m in models]
        return "### Vertex AI Model Registry\n" + "\n".join(model_list)
    except Exception as e:
        return f"Error listing models from Vertex AI: {str(e)}"


@mcp.tool()
def cloudrun_list_bucket_models(bucket_name: str = BUCKET_NAME) -> str:
    """
    Lists the contents of the GCS bucket to check for uploaded model files.

    Args:
        bucket_name: The GCS bucket name to check. Defaults to BUCKET_NAME.
    """
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        # List up to 100 blobs
        blobs = list(bucket.list_blobs(max_results=100))

        if not blobs:
            return f"The bucket '{bucket_name}' is empty."

        # Display up to 50 for brevity
        file_list = [f"- {b.name} ({b.size / 1024 / 1024:.2f} MB)" for b in blobs[:50]]
        summary = f"### Contents of GCS Bucket: {bucket_name}\n"
        summary += "\n".join(file_list)

        if len(blobs) > 50:
            summary += f"\n\n(Showing 50 of {len(blobs)} items)"

        return summary
    except Exception as e:
        return f"Error listing objects in bucket '{bucket_name}': {str(e)}"


@mcp.tool()
async def cloudrun_analyze_cloud_logging(filter_query: str, limit: int = 5) -> str:
    """
    Fetches and summarizes error logs from Google Cloud Logging using a self-hosted vLLM endpoint on Cloud Run.

    Args:
        filter_query: Filter for Cloud Logging (e.g., 'severity=ERROR').
        limit: Number of recent logs to fetch.
    """
    try:
        logging_client = cloud_logging.Client(project=PROJECT_ID)
        entries = list(
            logging_client.list_entries(filter_=filter_query, order_by=cloud_logging.DESCENDING, page_size=limit)
        )

        if not entries:
            return "No matching logs found."

        log_texts = [
            f"Timestamp: {e.timestamp} | Severity: {e.severity} | Message: {str(e.payload)[:1000] if isinstance(e.payload, str) else json.dumps(e.payload)[:1000]}"
            for e in entries
        ]
        combined_logs = "\n---\n".join(log_texts)

        # Truncate combined logs to ~3000 tokens (approx 12000 chars) to stay within 4096 context limit
        if len(combined_logs) > 12000:
            combined_logs = combined_logs[:12000] + "... (truncated)"

        # Prepare prompt for Gemma
        prompt = f"Analyze the following DevOps logs and provide a high-level summary of the critical issues and potential root causes:\n\n{combined_logs}\n\nSummary:"

        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=512,
            temperature=0.2,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### Log Analysis (Self-Hosted vLLM)\n\n{response_text}"

    except Exception as e:
        return f"Error analyzing logs via self-hosted vLLM: {str(e)}"


@mcp.tool()
async def cloudrun_suggest_sre_remediation(error_message: str) -> str:
    """
    Proposes remediation steps for a specific SRE incident using self-hosted vLLM.

    Args:
        error_message: The error or incident description to remediate.
    """
    prompt = f"As an expert SRE, suggest a 3-step remediation plan for the following error:\n\nError: {error_message}\n\nRemediation Plan:"

    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=512,
            temperature=0.2,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### Remediation Plan\n\n{response_text}"
    except Exception as e:
        return f"Error fetching remediation plan: {str(e)}"


@mcp.tool()
async def cloudrun_query(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Directly queries the self-hosted vLLM model with a custom prompt.

    Args:
        prompt: The text prompt to send to the model.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for deterministic).
    """
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### vLLM Response\n\n{response_text}"
    except Exception as e:
        return f"Error querying vLLM: {str(e)}"


@mcp.tool()
def cloudrun_get_deployment_config(
    service_name: str = DEFAULT_SERVICE_NAME,
    bucket_name: str = BUCKET_NAME,
    model_path: str = "gemma-4-E2B-it",
    allow_unauthenticated: bool = False,
    min_instances: int = 0,
    gpu_memory_utilization: float = 0.95,
) -> str:
    """
    Generates the gcloud command to deploy vLLM to Cloud Run with GCS FUSE and NVIDIA L4 GPU.

    Args:
        service_name: The name for the Cloud Run service.
        bucket_name: The GCS bucket containing the model weights.
        model_path: The sub-path inside the bucket (e.g., 'gemma-4-E2B-it') or Hugging Face repo ID.
        allow_unauthenticated: Whether to allow unauthenticated access to the service.
        min_instances: The minimum number of instances to keep warm (default: 0).
        gpu_memory_utilization: The fraction of GPU memory to use for KV cache (default: 0.95).
    """
    # Check if we are pulling directly from Hugging Face
    is_hf = "/" in model_path and not model_path.startswith("/")

    command = [
        "gcloud beta run deploy",
        service_name,
        f"--project={PROJECT_ID}",
        "--image=vllm/vllm-openai:v0.26.0-cu129",  # Pinned: :latest has changed CUDA major version underneath this deploy
        "--gpu=1",
        "--gpu-type=nvidia-l4",
        "--no-gpu-zonal-redundancy",  # Fix for quota issues in us-east4
        "--no-cpu-throttling",  # Required for GPU deployment
        "--concurrency=4",  # Optimal for LLM throughput vs latency
        "--timeout=3600",  # 1 hour timeout for long generations
        "--startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=10,initialDelaySeconds=180,httpGet.port=8000,httpGet.path=/health",
        "--scaling=auto",  # Without this a stale manual-scaling setting is inherited and every request 503s
        "--max-instances=1",  # Prevent scaling beyond quota
        f"--min-instances={min_instances}",
        "--port=8000",  # vLLM default port
        "--memory=32Gi",
        "--cpu=8",
        "--execution-environment=gen2",
        "--set-env-vars=VLLM_ENABLE_CUDA_COMPATIBILITY=0",
    ]

    if is_hf:
        command.append("--set-secrets=HF_TOKEN=hf-token:latest")
        command.append(
            f"--args=--model={model_path},--dtype=bfloat16,--max-model-len=16384,--disable-chunked-mm-input,--gpu-memory-utilization={gpu_memory_utilization},--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={{}},--host=0.0.0.0,--port=8000"
        )
    else:
        command.append(f"--add-volume=name=model-volume,type=cloud-storage,bucket={bucket_name},readonly=true")
        command.append("--add-volume-mount=volume=model-volume,mount-path=/mnt/models")
        command.append(
            f"--args=--model=/mnt/models/{model_path},--dtype=bfloat16,--max-model-len=16384,--disable-chunked-mm-input,--gpu-memory-utilization={gpu_memory_utilization},--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={{}},--host=0.0.0.0,--port=8000"
        )

    command.append("--allow-unauthenticated" if allow_unauthenticated else "--no-allow-unauthenticated")
    command.append(f"--region={LOCATION}")

    return " ".join(command)


@mcp.tool()
async def cloudrun_deploy(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "gemma-4-E2B-it",
    bucket_name: str = BUCKET_NAME,
) -> str:
    """
    Deploys vLLM to Cloud Run with GPU.

    Args:
        service_name: Name of the service to deploy.
        model_path: Path to the model (GCS folder name or Hugging Face repo ID).
        bucket_name: GCS bucket name (only used if using GCS FUSE).
    """
    is_hf = "/" in model_path and not model_path.startswith("/")

    cmd = [
        "gcloud",
        "beta",
        "run",
        "deploy",
        service_name,
        f"--project={PROJECT_ID}",
        "--image=vllm/vllm-openai:v0.26.0-cu129",  # Pinned: :latest has changed CUDA major version underneath this deploy
        "--gpu=1",
        "--gpu-type=nvidia-l4",
        "--no-gpu-zonal-redundancy",  # Fix for quota issues in us-east4
        "--no-cpu-throttling",  # Required for GPU deployment
        "--concurrency=4",
        "--timeout=3600",
        "--startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=10,initialDelaySeconds=180,httpGet.port=8000,httpGet.path=/health",
        "--scaling=auto",  # Without this a stale manual-scaling setting is inherited and every request 503s
        "--max-instances=1",  # Prevent scaling beyond quota
        "--min-instances=0",
        "--port=8000",
        "--memory=32Gi",
        "--cpu=8",
        "--execution-environment=gen2",
        "--no-allow-unauthenticated",
        f"--region={LOCATION}",
        "--set-env-vars=VLLM_ENABLE_CUDA_COMPATIBILITY=0",  # "1" prepends compat libs and fails with CUDA error 803
    ]

    if is_hf:
        cmd.append("--set-secrets=HF_TOKEN=hf-token:latest")
        cmd.append(
            f"--args=--model={model_path},--dtype=bfloat16,--max-model-len=16384,--disable-chunked-mm-input,--gpu-memory-utilization=0.95,--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={{}},--host=0.0.0.0,--port=8000"
        )
    else:
        cmd.append(f"--add-volume=name=model-volume,type=cloud-storage,bucket={bucket_name},readonly=true")
        cmd.append("--add-volume-mount=volume=model-volume,mount-path=/mnt/models")
        cmd.append(
            f"--args=--model=/mnt/models/{model_path},--dtype=bfloat16,--max-model-len=16384,--disable-chunked-mm-input,--gpu-memory-utilization=0.95,--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={{}},--host=0.0.0.0,--port=8000"
        )

    try:
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, check=True)
        # A deploy can mint a new URL; drop the cached one so the next lookup re-discovers it.
        invalidate_vllm_url_cache(service_name)
        return f"Successfully deployed {service_name}:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        invalidate_vllm_url_cache(service_name)
        return f"Failed to deploy {service_name}:\nError: {e.stderr}\nOutput: {e.stdout}"


@mcp.tool()
def cloudrun_destroy(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Destroys the Cloud Run vLLM service.

    Args:
        service_name: Name of the service to destroy.
    """
    cmd = [
        "gcloud",
        "run",
        "services",
        "delete",
        service_name,
        f"--project={PROJECT_ID}",
        f"--region={LOCATION}",
        "--quiet",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # The service is gone; a stale cached URL would silently serve the next query tool.
        invalidate_vllm_url_cache(service_name)
        return f"Successfully destroyed {service_name}:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        invalidate_vllm_url_cache(service_name)
        return f"Failed to destroy {service_name}:\nError: {e.stderr}\nOutput: {e.stdout}"


@mcp.tool()
def cloudrun_status(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Checks the status of the Cloud Run vLLM service.

    Args:
        service_name: Name of the service to check.
    """
    cmd = [
        "gcloud",
        "run",
        "services",
        "describe",
        service_name,
        f"--project={PROJECT_ID}",
        f"--region={LOCATION}",
        "--format=yaml(status.conditions,status.latestCreatedRevisionName,status.url)",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return f"### Status for {service_name}:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Failed to get status for {service_name}:\nError: {e.stderr}\nOutput: {e.stdout}"


@mcp.tool()
def cloudrun_update_scaling(min_instances: int, max_instances: int, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Updates the scaling configuration (min and max instances) for the Cloud Run vLLM service.

    Also forces the service back to automatic scaling. Min/max instances alone do not set the
    scaling *mode*, so a service left in manual scaling at 0 instances keeps returning 503 no
    matter what bounds are set.

    Args:
        min_instances: The minimum number of instances to keep warm.
        max_instances: The maximum number of instances to scale out to.
        service_name: The name of the Cloud Run service to update.
    """
    cmd = [
        "gcloud",
        "beta",
        "run",
        "services",
        "update",
        service_name,
        "--scaling=auto",
        f"--min-instances={min_instances}",
        f"--max-instances={max_instances}",
        f"--project={PROJECT_ID}",
        f"--region={LOCATION}",
    ]

    try:
        # We use 'update' which doesn't require a full image/env specification if the service exists
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return f"Successfully updated scaling for {service_name} to min={min_instances}, max={max_instances}.\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Failed to update scaling for {service_name}:\nError: {e.stderr}\nOutput: {e.stdout}"


@mcp.tool()
def cloudrun_get_gpu_deployment_config(
    cluster_name: str = "gpu-cluster", model_name: str = "google/gemma-4-E2B-it"
) -> str:
    """
    Generates a GKE (not Cloud Run) manifest and setup instructions for deploying vLLM on GPU (NVIDIA L4).

    Args:
        cluster_name: The name of the GKE cluster.
        model_name: The model identifier (e.g., 'google/gemma-4-E2B-it').
    """
    manifest = f"""
### 🌀 vLLM on GPU (GKE Deployment)

To deploy vLLM on GPUs, use the following GKE manifest. This configuration targets a single **NVIDIA L4 GPU** which is ideal for Gemma 4 E2B.

#### 1. Create a GPU Node Pool (if not exists)
```bash
gcloud container node-pools create gpu-l4 \\
    --cluster={cluster_name} \\
    --location={LOCATION} \\
    --machine-type=g2-standard-4 \\
    --accelerator=type=nvidia-l4,count=1 \\
    --num-nodes=1
```

#### 2. Kubernetes Manifest (vllm-gpu.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-gpu
  template:
    metadata:
      labels:
        app: vllm-gpu
    spec:
      nodeSelector:
        cloud.google.com/gke-gpu: "true"
      containers:
      - name: vllm-gpu
        image: vllm/vllm-openai:v0.26.0-cu129
        resources:
          limits:
            nvidia.com/gpu: "1"
          requests:
            nvidia.com/gpu: "1"
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - "--model={model_name}"
        - "--gpu-memory-utilization=0.9"
        - "--max-model-len=4096"
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-gpu-service
spec:
  selector:
    app: vllm-gpu
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### 3. Deployment Steps
1. Save the YAML above to `vllm-gpu.yaml`.
2. Apply it: `kubectl apply -f vllm-gpu.yaml`.
3. (Optional) If using a private model, ensure a Hugging Face token is provided via secret.
"""
    return manifest


@mcp.tool()
def cloudrun_get_vertex_ai_model_copy_instructions(model_name: str = "gemma-4-E2B-it") -> str:
    """
    Provides instructions and commands to transfer Gemma model artifacts from Vertex AI Model Garden to your GCS bucket.
    """
    instructions = f"""
### 🚀 Transferring {model_name} from Vertex AI Model Garden

To use vLLM with Cloud Storage FUSE without Hugging Face, follow these steps:

1. **Accept Terms:** Go to the Vertex AI Model Garden page for Gemma (https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335) and click 'Accept' on the license agreement.
2. **Download via Signed URL:** After accepting, the console provides a 'Download' button or a signed URL.
3. **Transfer to GCS:**
   If you have the artifacts locally after downloading from the console, use:
   `gcloud storage cp -r ./model_artifacts/* gs://{BUCKET_NAME}/{model_name}/`

4. **Alternative (Direct GCS Copy):**
   Google occasionally provides a managed GCS path for verified projects. If accessible, you can use:
   `gcloud storage cp -r gs://vertex-ai-models/gemma/{model_name}/* gs://{BUCKET_NAME}/{model_name}/`

Once the artifacts are in your bucket, use `cloudrun_get_deployment_config` to generate your Cloud Run deployment command.
"""
    return instructions


@mcp.tool()
async def cloudrun_get_huggingfacehub_download_path(
    repo_id: str = "google/gemma-4-E2B-it",
) -> str:
    """
    Returns the local cache path for a Hugging Face model using huggingface_hub.
    Note: This may trigger a download if the model is not already in the cache.
    """
    try:
        from huggingface_hub import snapshot_download

        token = await get_secret() or os.getenv("HF_TOKEN")
        # Run synchronous snapshot_download in a separate thread to avoid blocking the async event loop
        path = await asyncio.to_thread(snapshot_download, repo_id=repo_id, token=token)
        return f"Model '{repo_id}' is available at: {path}"
    except Exception as e:
        return f"Error resolving huggingface_hub path: {str(e)}"


@mcp.tool()
def cloudrun_get_huggingface_model_copy_instructions(
    repo_id: str = "google/gemma-4-E2B-it",
    bucket_name: str = BUCKET_NAME,
) -> str:
    """
    Provides instructions and commands to transfer Gemma model weights from Hugging Face to your GCS bucket.

    Args:
        repo_id: The Hugging Face repo ID (e.g., 'google/gemma-4-E2B-it').
        bucket_name: The target GCS bucket name.
    """
    model_name = repo_id.split("/")[-1]

    instructions = f"""
### 📦 Transferring {model_name} from Hugging Face to GCS

To use Hugging Face weights with vLLM on Cloud Run via GCS FUSE, follow these steps:

#### Option A: Using `huggingface_hub` Python Library (Recommended)
`huggingface_hub` simplifies the download process and can be run directly from python:

1. **Download Model:**
   `python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('{repo_id}'))"`

2. **Upload to GCS:**
   The command above outputs the local path. Use it to copy the artifacts:
   `gcloud storage cp -r /path/to/downloaded/model/* gs://{bucket_name}/{model_name}/`

#### Option B: Using `huggingface-cli`
1. **Setup Hugging Face CLI:**
   `pip install huggingface_hub`
   `huggingface-cli login`

2. **Download Model Artifacts:**
   `huggingface-cli download {repo_id} --local-dir ./{model_name}`

3. **Upload to GCS Bucket:**
   `gcloud storage cp -r ./{model_name}/* gs://{bucket_name}/{model_name}/`

Once uploaded, you can deploy using:
`cloudrun_get_deployment_config(model_path="{model_name}")`
"""
    return instructions


@mcp.tool()
def cloudrun_check_gpu_quotas(region: str = LOCATION) -> str:
    """
    Checks GPU quotas for a specific region using gcloud compute regions describe.

    Args:
        region: The Google Cloud region to check quotas for (defaults to LOCATION).
    """
    cmd = [
        "gcloud",
        "compute",
        "regions",
        "describe",
        region,
        f"--project={PROJECT_ID}",
        "--format=json(quotas)",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        quotas = data.get("quotas", [])

        # Filter for GPU quotas
        gpu_quotas = []
        for q in quotas:
            metric = q.get("metric", "")
            if "GPU" in metric or "ACCELERATOR" in metric:
                gpu_quotas.append(f"- **{metric}**:\n  - Limit: `{q.get('limit')}`\n  - Usage: `{q.get('usage')}`")

        if not gpu_quotas:
            return f"No GPU/Accelerator quotas found in region `{region}`. This usually means the quota is 0 or not assigned."

        return f"### 📊 GPU Quotas for region `{region}`\n\n" + "\n".join(gpu_quotas)

    except subprocess.CalledProcessError as e:
        return f"Failed to retrieve GPU quotas for region `{region}`:\nError: {e.stderr}\nOutput: {e.stdout}"
    except Exception as e:
        return f"Error checking GPU quotas: {str(e)}"


@mcp.tool()
async def cloudrun_verify_model_health() -> str:
    """Runs a deep health check with latency reporting on the Cloud Run GPU-hosted model."""
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        start_time = time.monotonic()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, is the model working?"}],
            model=model_name,
            max_tokens=200,
        )
        end_time = time.monotonic()
        latency = end_time - start_time
        response_content = chat_completion.choices[0].message.content

        if response_content:
            return (
                f"✅ Model health check PASSED.\n"
                f"Model: {model_name}\n"
                f"Response: '{response_content[:50]}...'\n"
                f"Latency: {latency:.2f} seconds."
            )
        else:
            return "❌ Model health check FAILED: Empty response."
    except Exception as e:
        return f"❌ Model health check FAILED: {e}"


@mcp.tool()
async def cloudrun_query_gemma4(prompt: str) -> str:
    """Queries the self-hosted Gemma 4 model on Cloud Run."""
    logger.info(f"Querying Cloud Run model with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
        )
        response = chat_completion.choices[0].message.content or "No response from model."
        logger.info(f"Model response: '{response[:100]}...'")
        return response
    except Exception as e:
        logger.error(f"Error querying model: {e}")
        return f"❌ An error occurred while querying the model: {e}"


@mcp.tool()
async def cloudrun_query_gemma4_with_stats(prompt: str) -> str:
    """
    Queries the self-hosted Gemma 4 model on Cloud Run and returns detailed performance statistics.

    This tool provides:
    - The full model response.
    - Time to First Token (TTFT).
    - Total generation time.
    - Tokens per second.
    """
    logger.info(f"Querying model with stats with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)

        start_time = time.monotonic()
        ttft = None
        response_content = ""
        total_tokens = 0

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            stream=True,
        )

        async for chunk in stream:
            if ttft is None:
                ttft = time.monotonic() - start_time

            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_content += content
                total_tokens += 1  # Rough token count

        end_time = time.monotonic()
        total_time = end_time - start_time

        if not response_content:
            return "❌ Model returned an empty response."

        tokens_per_second = total_tokens / (total_time - ttft) if ttft and total_time > ttft else 0

        stats_report = (
            f"### 📊 Performance Stats\n"
            f"- **Model:** `{model_name}`\n"
            f"- **Time to First Token (TTFT):** `{ttft:.3f}s`\n"
            f"- **Total Generation Time:** `{total_time:.3f}s`\n"
            f"- **Tokens per Second:** `{tokens_per_second:.2f} tokens/s`\n"
            f"- **Total Tokens (approx.):** `{total_tokens}`\n"
            f"\n### 💬 Model Response\n"
            f"{response_content}"
        )

        logger.info(f"Model response with stats: TTFT={ttft:.3f}s, TotalTime={total_time:.3f}s")
        return stats_report

    except Exception as e:
        logger.error(f"Error querying model with stats: {e}")
        return f"❌ An error occurred while querying the model with stats: {e}"


@mcp.tool()
async def cloudrun_get_model_details() -> str:
    """Retrieves detailed information about the running Cloud Run model, engine, and versions."""
    report = ""
    try:
        vllm_url = get_vllm_url()
        report += f"### 🧩 Model Details ({vllm_url})\n\n"
        client = await get_vllm_client()

        # 1. Get Model Details from /v1/models
        try:
            models_res = await client.models.list()
            report += "**Model Information (`/v1/models`):**\n"
            models_list = [{"id": m.id, "object": m.object, "owned_by": m.owned_by} for m in models_res.data]
            report += f"```json\n{json.dumps(models_list, indent=2)}\n```\n"
        except Exception as e:
            report += f"❌ Error fetching model details via client: {e}\n\n"

        # 2. Get Health Status
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=10) as http_client:
            try:
                res = await http_client.get(f"{vllm_url}/health", headers=headers)
                if res.status_code == 200:
                    report += "**Health Status (`/health`):**\n- Status: `Healthy` ✅\n\n"
                else:
                    report += f"**Health Status (`/health`):**\n- Status: `Unhealthy` (Code: {res.status_code}) ❌\n\n"
            except Exception as e:
                report += f"**Health Status (`/health`):**\n- Status: `Unreachable` (Error: {e}) ❌\n\n"
    except Exception as e:
        report += f"❌ Error retrieving system URL or auth token: {e}"

    return report


@mcp.tool()
async def cloudrun_get_system_status(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Provides a high-level dashboard of Cloud Run system status.

    Args:
        service_name: The name of the Cloud Run service to status.
    """
    health = "🔴 Offline"
    url = None
    try:
        url = get_vllm_url(service_name)
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health", headers=headers)
            if res.status_code == 200:
                health = f"🟢 Online ({url})"
            else:
                health = f"🔴 Offline (Status {res.status_code}) ({url})"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

    cloud_run_status = "🔴 Unknown"
    try:
        cmd = [
            "gcloud",
            "run",
            "services",
            "describe",
            service_name,
            f"--project={PROJECT_ID}",
            f"--region={LOCATION}",
            "--format=json",
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if process.returncode == 0:
            data = json.loads(process.stdout)
            conditions = data.get("status", {}).get("conditions", [])
            ready_cond = next((c for c in conditions if c.get("type") == "Ready"), None)
            if ready_cond and ready_cond.get("status") == "True":
                cloud_run_status = "🟢 Ready"
            elif ready_cond:
                cloud_run_status = f"🔴 Not Ready ({ready_cond.get('status')})"
            else:
                cloud_run_status = "🔴 Not Ready (No Ready condition)"
        else:
            cloud_run_status = f"🔴 Error checking service ({process.stderr.strip()})"
    except Exception as e:
        cloud_run_status = f"🔴 Error: {str(e)}"

    if "🟢" in health:
        next_step = "Use `cloudrun_query_gemma4` to interact with the model."
    else:
        next_step = f"Call `cloudrun_deploy` to provision or start the Cloud Run service `{service_name}`."

    return (
        f"### 🌀 GPU Cloud Run System Status (`{service_name}`)\n"
        f"- **vLLM Health:** {health}\n"
        f"- **Cloud Run Service Status:** {cloud_run_status}\n"
        f"**👉 Next Step:** {next_step}"
    )


@mcp.tool()
async def cloudrun_get_endpoint(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Returns the active Cloud Run vLLM service URL if available.

    Args:
        service_name: The name of the Cloud Run service to query.
    """
    try:
        url = get_vllm_url(service_name)
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health", headers=headers)
            if res.status_code == 200:
                return f"🟢 Cloud Run vLLM `{service_name}` is Online at: {url}"
            else:
                return (
                    f"🔴 Cloud Run vLLM `{service_name}` is configured at {url} but returned status {res.status_code}."
                )
    except Exception as e:
        return (
            f"🔴 Cloud Run vLLM endpoint check for `{service_name}` failed: {e}. "
            "Try deploying/starting it with `cloudrun_deploy`."
        )


@mcp.tool()
async def cloudrun_run_benchmark(
    model: Optional[str] = None,
    num_prompts: int = 20,
    random_output_len: int = 128,
    max_concurrency: int = 8,
) -> str:
    """
    Runs a performance/concurrency benchmark sweep against the Cloud Run vLLM GPU endpoint.

    Args:
        model: Model name to request (defaults to the active model from /v1/models).
        num_prompts: Number of requests to send per concurrency level.
        random_output_len: Max tokens to generate per request.
        max_concurrency: Maximum concurrency level to sweep up to (powers of 2, e.g. 1, 2, 4, 8).
    """
    from datetime import datetime

    try:
        url = get_vllm_url()
        token = get_auth_token()
    except Exception as e:
        return f"❌ Cannot run benchmark: {e}"

    # Get active model name if not provided
    client = await get_vllm_client()
    if not model:
        model = await get_active_model_name(client)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Chat completions, not /v1/completions: the raw completions route returns a single empty token
    # on instruction-tuned Gemma checkpoints, which would time an empty generation and report
    # meaningless throughput.
    base_url = f"{url.rstrip('/')}/v1/chat/completions"
    prompt = "Explain the importance of Site Reliability Engineering for large scale AI deployments."

    # Clamp before building the sweep: max_concurrency=0 would yield a lone Semaphore(0) level that
    # never admits a request (the gather below has no timeout, so the tool would hang forever), and
    # a negative value raises straight out of Semaphore().
    max_concurrency = max(1, max_concurrency)

    concurrencies = []
    c = 1
    while c <= max_concurrency:
        concurrencies.append(c)
        c *= 2
    if max_concurrency not in concurrencies:
        concurrencies.append(max_concurrency)

    results = []

    async def send_request(http_client: httpx.AsyncClient, sem: asyncio.Semaphore) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": random_output_len,
            "temperature": 0.0,
            "stream": False,
        }
        async with sem:
            start_time = time.perf_counter()
            try:
                response = await http_client.post(base_url, json=payload, headers=headers, timeout=120)
                end_time = time.perf_counter()
                if response.status_code == 200:
                    latency = end_time - start_time
                    data = response.json()
                    # Count only tokens the server actually reported. Falling back to the requested
                    # length would credit a request that generated nothing with a full response.
                    tokens = data.get("usage", {}).get("completion_tokens", 0)
                    return {"success": True, "latency": latency, "tokens": tokens}
                else:
                    return {"success": False, "error": f"Status {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

    # Warmup. This also doubles as a reachability check: the service runs at --min-instances=0 and a
    # cold start can exceed the per-request timeout, which would otherwise yield an all-zeros table
    # with no hint that the service was simply still booting.
    logger.info("Warming up model for benchmark...")
    async with httpx.AsyncClient() as http_client:
        warmup = await send_request(http_client, asyncio.Semaphore(1))

    if not warmup["success"]:
        return (
            f"❌ Benchmark aborted: the warmup request to {base_url} failed ({warmup['error']}).\n\n"
            "The service may be cold — a scale-from-zero GPU start can take several minutes. "
            "Check `cloudrun_get_endpoint` and retry once it reports 🟢 Online."
        )

    logger.info(f"Starting GPU benchmark sweep against {url} with model {model}...")
    for concurrency in concurrencies:
        logger.info(f"Running sweep with concurrency={concurrency}...")
        sem = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient() as http_client:
            start_batch = time.perf_counter()
            tasks = [send_request(http_client, sem) for _ in range(num_prompts)]
            batch_results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_batch

        successes = [r for r in batch_results if r["success"]]
        latencies = [r["latency"] for r in successes]

        if not latencies:
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "concurrency": concurrency,
                    "total_requests": num_prompts,
                    "success_rate": 0.0,
                    "avg_latency": 0.0,
                    "p95_latency": 0.0,
                    "req_per_sec": 0.0,
                    "tokens_per_sec": 0.0,
                }
            )
            continue

        avg_lat = statistics.mean(latencies)
        # Nearest-rank P95: the ceil(0.95 * n)-th sample, 1-indexed. Truncating instead (int(n * 0.95))
        # lands on the last index whenever n * 0.95 is a whole number — at the default num_prompts=20
        # that reports the slowest request as the P95.
        p95_lat = sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1]
        throughput = len(successes) / total_time
        tokens_per_sec = sum(r["tokens"] for r in successes) / total_time

        results.append(
            {
                "timestamp": datetime.now().isoformat(),
                "concurrency": concurrency,
                "total_requests": num_prompts,
                "success_rate": len(successes) / num_prompts,
                "avg_latency": avg_lat,
                "p95_latency": p95_lat,
                "req_per_sec": throughput,
                "tokens_per_sec": tokens_per_sec,
            }
        )

    # Save to CSV. The sweep has already run at this point, so a write failure (read-only checkout,
    # permissions) must not discard the results or escape as a raw exception — report it in the
    # summary instead.
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
    saved_note = f"\n\nResults saved to `{output_file}`"
    try:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "concurrency",
                    "total_requests",
                    "success_rate",
                    "avg_latency",
                    "p95_latency",
                    "req_per_sec",
                    "tokens_per_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
    except OSError as e:
        logger.warning(f"⚠️ Could not write benchmark CSV to {output_file}: {e}")
        saved_note = f"\n\n⚠️ Results could not be saved to `{output_file}`: {e}"

    summary_str = f"### 📊 GPU Benchmark Results (Model: `{model}`)\n\n"
    summary_str += "| Concurrency | Success Rate | Req/s | Tokens/s | Avg Latency | P95 Latency |\n"
    summary_str += "|---:|---:|---:|---:|---:|---:|\n"
    for r in results:
        summary_str += f"| {r['concurrency']} | {r['success_rate']:.1%} | {r['req_per_sec']:.2f} | {r['tokens_per_sec']:.2f} | {r['avg_latency']:.2f}s | {r['p95_latency']:.2f}s |\n"
    summary_str += saved_note
    return summary_str


@mcp.tool()
async def cloudrun_analyze_gpu_logs(limit: int = 15, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Fetches Cloud Run logs for the specified service and uses Gemma 4 to analyze them for errors.

    Args:
        limit: Number of log entries to fetch.
        service_name: Name of the Cloud Run service.
    """
    filter_query = f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service_name}"'
    return await cloudrun_analyze_cloud_logging(filter_query, limit)


@mcp.tool()
async def cloudrun_get_help() -> str:
    """Provides help text and summarizes the configuration options and all available SRE/DevOps tools for this Cloud Run MCP server."""
    return (
        "### 🛠️ Cloud Run Gemma 4 SRE Agent Help & Configuration\n\n"
        "You can configure this MCP server using the following environment variables:\n\n"
        f"- **`GOOGLE_CLOUD_PROJECT`**: Your GCP Project ID.\n"
        f"  - *Current Value:* `{PROJECT_ID}`\n"
        f"- **`GOOGLE_CLOUD_LOCATION`**: The GCP Region for deployment.\n"
        f"  - *Current Value:* `{LOCATION}`\n"
        f"- **`BUCKET_NAME`**: GCS Bucket used to store model weights.\n"
        f"  - *Current Value:* `{BUCKET_NAME}`\n"
        f"- **`MODEL_NAME`**: Default Hugging Face repository or path.\n"
        f"  - *Current Value:* `{MODEL_NAME}`\n"
        f"- **`SERVICE_NAME`**: The Cloud Run service tools act on unless given an explicit `service_name`.\n"
        f"  - *Current Value:* `{DEFAULT_SERVICE_NAME}`\n"
        f"- **`VLLM_BASE_URL`**: The explicit URL of your Cloud Run service. (If not set, it is auto-discovered via `gcloud`)\n"
        f"  - *Current Value:* `{VLLM_BASE_URL or 'Not set (auto-discovering)'}`\n"
        f"  - Only honoured for `{DEFAULT_SERVICE_NAME}`; other services are always discovered via `gcloud`.\n\n"
        "### ℹ️ Active Mode Summary\n"
        f"The server is running in **CLOUD RUN** mode targeting NVIDIA L4 GPU in region `{LOCATION}`.\n\n"
        "---\n\n"
        "### 🧰 Available MCP Tools\n\n"
        "Below is a summary of the tools exposed by this SRE/DevOps agent:\n\n"
        "#### 🐳 Infrastructure & Deployment\n"
        "- **`cloudrun_deploy`**: Deploys vLLM to Cloud Run GPU (NVIDIA L4 in us-east4).\n"
        "- **`cloudrun_destroy`**: Deletes the Cloud Run vLLM service.\n"
        "- **`cloudrun_status`**: Checks the status of the Cloud Run vLLM service.\n"
        "- **`cloudrun_update_scaling`**: Updates min/max instances for scaling.\n"
        "- **`cloudrun_get_deployment_config`**: Generates the gcloud deployment command.\n"
        "- **`cloudrun_get_gpu_deployment_config`**: Generates a GKE (not Cloud Run) manifest for GPU (NVIDIA L4).\n"
        "- **`cloudrun_check_gpu_quotas`**: Checks L4 and other GPU quotas for a region.\n\n"
        "#### 📊 Model Management\n"
        "- **`cloudrun_list_vertex_models`**: Lists models in the Vertex AI Registry.\n"
        "- **`cloudrun_list_bucket_models`**: Lists model weights in GCS bucket.\n"
        "- **`cloudrun_save_hf_token`**: Securely saves a Hugging Face API token to Secret Manager.\n"
        "- **`cloudrun_get_vertex_ai_model_copy_instructions`**: Instructions to copy model from Vertex AI Model Garden to GCS.\n"
        "- **`cloudrun_get_huggingface_model_copy_instructions`**: Instructions to download model from Hugging Face and upload to GCS.\n"
        "- **`cloudrun_get_huggingfacehub_download_path`**: Resolves local cache path using huggingface_hub.\n\n"
        "#### 📊 Monitoring & Status\n"
        "- **`cloudrun_get_metrics`**: Fetches raw Prometheus metrics from the running vLLM service's /metrics endpoint.\n"
        "- **`cloudrun_get_system_status`**: Provides a high-level status dashboard of the Cloud Run service and health.\n"
        "- **`cloudrun_get_endpoint`**: Verifies connectivity and returns the active service URL.\n"
        "- **`cloudrun_get_endpoint_url`**: Resolves the service URL without probing it for health.\n"
        "- **`cloudrun_get_model_details`**: Retrieves detailed model metadata and engine state from `/v1/models`.\n"
        "- **`cloudrun_verify_model_health`**: Deep health check by querying the model with a simple prompt and measuring latency.\n\n"
        "#### 📈 Performance & Benchmarking\n"
        "- **`cloudrun_run_benchmark`**: Runs performance/concurrency benchmark sweeps against the Cloud Run vLLM GPU endpoint.\n\n"
        "#### 💬 Interaction & Diagnostics\n"
        "- **`cloudrun_query_gemma4`**: Primary tool to query the self-hosted model with standard chat message format.\n"
        "- **`cloudrun_query_gemma4_with_stats`**: Queries the model and returns streaming performance statistics (TTFT, throughput).\n"
        "- **`cloudrun_query`**: Chat completion with explicit `max_tokens` / `temperature` control.\n"
        "- **`cloudrun_analyze_cloud_logging`**: Fetches logs from GCP Logging and analyzes them using the model.\n"
        "- **`cloudrun_analyze_gpu_logs`**: Fetches Cloud Run logs and uses Gemma 4 to analyze them for SRE/DevOps errors.\n"
        "- **`cloudrun_suggest_sre_remediation`**: Suggests remediation plans for SRE errors using the model.\n"
    )


@mcp.tool()
async def cloudrun_get_metrics() -> str:
    """
    Fetches the Prometheus metrics from the active Cloud Run vLLM service.
    """
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{url}/metrics", headers=headers)
            if res.status_code == 200:
                return res.text
            else:
                return f"🔴 Failed to retrieve metrics. Status code: {res.status_code}\n\nResponse:\n{res.text[:1000]}"
    except Exception as e:
        return f"🔴 Error fetching metrics from Cloud Run: {e}"


if __name__ == "__main__":
    mcp.run()
