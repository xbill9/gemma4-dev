import asyncio
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from typing import Optional

import boto3
import httpx
from mcp.server.mcpserver import MCPServer
from openai import AsyncOpenAI

# Setup logging to stderr ONLY to avoid interfering with MCP stdio communication
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vllm-devops-agent")
logger.info("Initializing DevOps Agent MCP Server...")

# Initialize MCPServer server
mcp = MCPServer("Self-Hosted vLLM DevOps Agent")

# Load AWS credentials if .aws_creds exists
def load_aws_credentials():
    # An explicit AWS_PROFILE (set by mcp-run.sh) refreshes itself; the static keys cached in
    # .aws_creds expire and, as env vars, would outrank that profile in boto3's chain.
    if os.getenv("AWS_PROFILE"):
        return
    aws_creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".aws_creds")
    if os.path.exists(aws_creds_path):
        with open(aws_creds_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    os.environ[key] = val

load_aws_credentials()

# Patch boto3 to reload credentials dynamically on every client instantiation
import boto3
_original_boto3_client = boto3.client
def patched_boto3_client(*args, **kwargs):
    load_aws_credentials()
    import boto3
    boto3.DEFAULT_SESSION = None
    return _original_boto3_client(*args, **kwargs)
boto3.client = patched_boto3_client


# Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "vllm-models-bucket")

# The URL of the self-hosted vLLM service on AWS EC2
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-E2B-it")
HF_SECRET_ID = "hf-token"


async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from AWS Secrets Manager or environment variables."""
    # 1. Check environment variable
    val = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    if val:
        return val

    # 2. Check AWS Secrets Manager
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        response = client.get_secret_value(SecretId=secret_id)
        if "SecretString" in response:
            return response["SecretString"]
    except Exception as e:
        logger.debug(f"AWS Secrets Manager failed: {e}")

    return None


@mcp.tool()
async def save_hf_token(token: str) -> str:
    """Securely saves a Hugging Face API token to AWS Secrets Manager."""
    saved_aws = False

    try:
        import boto3
        from botocore.exceptions import ClientError

        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        try:
            client.create_secret(Name=HF_SECRET_ID, SecretString=token)
            saved_aws = True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceExistsException":
                client.put_secret_value(SecretId=HF_SECRET_ID, SecretString=token)
                saved_aws = True
    except Exception as e:
        logger.warning(f"AWS Secrets Manager failed: {e}")

    if saved_aws:
        return "✅ Token saved to AWS Secrets Manager."
    else:
        return "❌ Failed to save token to AWS Secrets Manager."


DEFAULT_SERVICE_NAME = "inferentia-2b-devops-agent"


def discover_vllm_url(service_name: str = DEFAULT_SERVICE_NAME) -> Optional[str]:
    """Attempts to automatically discover the AWS EC2 instance public IP/DNS."""
    if VLLM_BASE_URL:
        logger.info(f"Using provided VLLM_BASE_URL: {VLLM_BASE_URL}")
        return VLLM_BASE_URL

    # 1. AWS EC2 Discovery
    if True:
        logger.info(f"Attempting to discover AWS EC2 vLLM URL for: {service_name}")
        try:
            import boto3

            ec2 = boto3.client("ec2", region_name=AWS_REGION)
            response = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Name", "Values": [service_name]},
                    {"Name": "instance-state-name", "Values": ["running"]},
                ]
            )
            for reservation in response.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    ip = instance.get("PublicIpAddress")
                    if ip:
                        url = f"http://{ip}:8080"
                        logger.info(f"📡 Automatically discovered AWS vLLM at: {url}")
                        return url
        except Exception as e:
            logger.warning(f"⚠️ Error during AWS vLLM discovery: {str(e)}")

    logger.error("❌ Failed to discover service URL.")
    return None


# Resolve base URL at runtime
_ACTIVE_VLLM_URL = None


def get_vllm_url() -> str:
    """Returns the cached vLLM URL or discovers it if needed."""
    global _ACTIVE_VLLM_URL
    if not _ACTIVE_VLLM_URL:
        _ACTIVE_VLLM_URL = discover_vllm_url()

    if not _ACTIVE_VLLM_URL:
        raise Exception(
            "Could not determine vLLM service URL. Ensure you are authenticated and the service/instance exists."
        )

    return _ACTIVE_VLLM_URL


def get_auth_token() -> str:
    """Returns empty string as authentication is not required for private EC2 vLLM."""
    return ""


async def get_vllm_client() -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for the vLLM service."""
    vllm_url = get_vllm_url()
    return AsyncOpenAI(
        base_url=f"{vllm_url}/v1",
        api_key="not-needed",
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


@mcp.resource("config://vllm-deployment-template")
def get_deployment_template() -> str:
    """Returns a base template for AWS EC2 Inferentia deployment of the Option B (optb) server.

    Option B replaces vLLM/NxD (gibberish on Gemma-4's cross-layer KV-sharing) with a
    torch_neuronx two-graph KV-cache server. The public Docker Hub image is self-contained
    (weights + neffs + server baked in) and serves an OpenAI-compatible API on :8080.
    """
    return """
# AWS EC2 Inferentia Deployment Template (Option B / optb server, NOT vLLM)
# Instance: inf2.xlarge (16GB host RAM, needs swap) with the :slim image,
#           OR inf2.8xlarge (128GB host RAM) with the :latest image.
# AMI: any Deep Learning Base Neuron AMI (Ubuntu 22.04) — only the Neuron driver + docker are needed.

InstanceType: inf2.8xlarge        # or inf2.xlarge (use the :slim image)
Ports:
  - Container Port: 8080
  - Host Port: 8080

# REQUIRED on a 16GB host (inf2.xlarge): add swap so the ~14.5GB neff-load peak doesn't OOM:
#   fallocate -l 32G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile

Docker Run Command (public image, no auth, weights+neffs baked in):
docker run -d --name gemma-optb \\
  --device /dev/neuron0 \\
  --restart unless-stopped \\
  -p 8080:8080 \\
  xbill9/gemma4-optb:latest        # inf2.8xlarge; use xbill9/gemma4-optb:slim on inf2.xlarge

# Serves: /v1/chat/completions, /v1/completions, /v1/models, /generate, /health
"""


@mcp.tool()
def get_vllm_endpoint(service_name: str = DEFAULT_SERVICE_NAME) -> Optional[str]:
    """
    Returns the current active vLLM endpoint URL.

    Args:
        service_name: The service name or instance Name tag to describe (defaults to 'gpu-2b-qat-inf-devops-agent').
    """
    if service_name == DEFAULT_SERVICE_NAME:
        return get_vllm_url()
    return discover_vllm_url(service_name)





@mcp.tool()
def list_bucket_models(bucket_name: Optional[str] = None) -> str:
    """
    Lists the contents of an S3 bucket to check for uploaded model files.

    Args:
        bucket_name: The S3 bucket name. Defaults to AWS_BUCKET_NAME.
    """
    if not bucket_name:
        bucket_name = os.getenv("AWS_BUCKET_NAME", "vllm-models-bucket")

    clean_bucket = bucket_name.replace("s3://", "").replace("gs://", "")
    try:
        import boto3

        s3 = boto3.client("s3", region_name=AWS_REGION)
        response = s3.list_objects_v2(Bucket=clean_bucket, MaxKeys=100)
        contents = response.get("Contents", [])
        if not contents:
            return f"The S3 bucket '{clean_bucket}' is empty or does not exist."

        file_list = [
            f"- s3://{clean_bucket}/{obj['Key']} ({obj['Size'] / 1024 / 1024:.2f} MB)" for obj in contents[:50]
        ]
        summary = f"### Contents of S3 Bucket: {clean_bucket}\n"
        summary += "\n".join(file_list)

        if len(contents) > 50:
            summary += f"\n\n(Showing 50 of {len(contents)} items)"

        return summary
    except Exception as e:
        return f"Error listing S3 bucket '{clean_bucket}': {str(e)}"


@mcp.tool()
async def analyze_cloud_logging(filter_query: str, limit: int = 5) -> str:
    """
    Fetches and summarizes error logs from AWS CloudWatch.

    Args:
        filter_query: Query filter (log group name).
        limit: Number of recent logs to fetch.
    """
    combined_logs = ""

    try:
        import boto3

        logs_client = boto3.client("logs", region_name=AWS_REGION)
        log_group_name = filter_query

        try:
            groups = logs_client.describe_log_groups(logGroupNamePrefix=log_group_name)
            group_list = groups.get("logGroups", [])
            if group_list:
                log_group_name = group_list[0]["logGroupName"]
        except Exception:
            pass

        streams = logs_client.describe_log_streams(
            logGroupName=log_group_name, orderBy="LastEventTime", descending=True, limit=1
        )
        stream_list = streams.get("logStreams", [])
        if stream_list:
            stream_name = stream_list[0]["logStreamName"]
            events = logs_client.get_log_events(logGroupName=log_group_name, logStreamName=stream_name, limit=limit)
            log_texts = [
                f"Timestamp: {ev.get('timestamp')} | Message: {ev.get('message')}"
                for ev in events.get("events", [])
            ]
            combined_logs = "\n---\n".join(log_texts)
    except Exception as e:
        return f"Failed to fetch AWS CloudWatch logs: {str(e)}"

    if not combined_logs:
        return "No matching logs found."

    try:
        if len(combined_logs) > 12000:
            combined_logs = combined_logs[:12000] + "... (truncated)"

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
async def suggest_sre_remediation(error_message: str) -> str:
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
async def query_vllm(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
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



def _get_inferentia_user_data(model_path: str, hf_token_expr: str = "", instance_type: str = "inf2.8xlarge") -> str:
    """Cloud-init user-data that deploys the Option B (optb) OpenAI-compatible server.

    Option B replaces vLLM/NxD (which emits gibberish for Gemma-4's cross-layer KV-sharing) with a
    torch_neuronx-traced two-graph KV-cache server. The public Docker Hub image bakes in the compiled
    neffs + weights + server and exposes /v1/chat/completions, /v1/completions, /v1/models, /generate,
    /health on :8080. `model_path`/`hf_token_expr` are unused (the image is self-contained).
    """
    # TP=2 + KV-aliasing build (~59-72 tok/s across both NeuronCores) — the fast default.
    # tp2-slim = slim host embeddings (bf16, ~6 GB): fits the 16 GB inf2.xlarge AND runs fine on
    # the big-RAM 8xlarge (just doesn't use the extra RAM), so ONE image works everywhere.
    # ~73 GB image (fits the 300 GB root below), uses the 48 GB swap for the neff-load peak on
    # small hosts, and MUST run with --ipc=host (parallel_model_load). Supersedes the single-core
    # :slim/:latest (~25 tok/s). For a marginally faster host load on 8xlarge, :tp2-2048 also works.
    optb_image = "xbill9/gemma4-optb:tp2-slim"

    user_data = f"""#!/bin/bash
# Install and start SSM agent (if not present) and add SSH key
snap install amazon-ssm-agent || true
systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service || true
echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC5D8tcXzOYvDeIT2QHDUMdwOYqZQ6J6c8TXxfKCFo1q76Sf1tRW9ajeRhmVJW/KrimNWJegwuXnoFwoDLD6wKo7dY5/QrO66od2/QkcmSdn3pa1ALwPbzZpNKcwVT736wLdm/iFnVXBTqmNkZREsbq16d3fQiLvnyJ74kYzt23YMLkv27Ik16hQEbBcO/4Lp6B1tjqKwO0gtvSP5c3b3MQ3qjnNVjqnpYLvQZgkXsczWWgb3U8VADofpo902IMOYPLwz6wvKnt300DNEO7B5gKShbYkGThTkjeSDAaBmVte3GOd6L6fV7E8tKQNgXG3jyK0B7Vs2wJFVEANvXQQvsb' >> /home/ubuntu/.ssh/authorized_keys

# 1. Resize root partition and filesystem to utilize full volume size
echo "Starting root partition resize..."
sudo growpart /dev/nvme0n1 1 || true
sudo growpart /dev/xvda 1 || true
sudo growpart /dev/sda 1 || true
sudo resize2fs /dev/nvme0n1p1 || true
sudo resize2fs /dev/xvda1 || true
sudo resize2fs /dev/sda1 || true
df -h

if ! command -v docker &> /dev/null; then
    apt-get update -y
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
fi

# 2. Optimized Swap for 2B model on 16GB RAM
if [ ! -f /swapfile_large ]; then
    echo "Creating 48GB optimized swap..."
    fallocate -l 48G /swapfile_large
    chmod 600 /swapfile_large
    mkswap /swapfile_large
    swapon /swapfile_large
    echo "/swapfile_large swap swap defaults 0 0" >> /etc/fstab
fi

# Tune virtual memory for heavy swapping
sysctl -w vm.swappiness=100
sysctl -w vm.vfs_cache_pressure=50
echo 1 > /proc/sys/vm/oom_kill_allocating_task

# Detect available Neuron devices
devices=""
device_count=0
for dev in /dev/neuron*; do
    if [ -e "$dev" ]; then
        devices="$devices --device $dev"
        device_count=$((device_count + 1))
    fi
done

if [ $device_count -eq 0 ]; then
    devices="--device /dev/neuron0"
    device_count=1
fi
tensor_parallel_size=$((device_count * 2))

# Detect and mount secondary cache volume if present
echo "Waiting for secondary cache volume..."
for i in {{1..30}}; do
    CACHE_DEV=""
    CACHE_PART=""
    for dev in $(lsblk -d -n -o NAME | grep -E "^(nvme|xvdf|sdf)" | grep -v "nvme0n1"); do
        if ! lsblk -n -r "/dev/$dev" | grep -q /; then
            if lsblk -n -r "/dev/$dev" | grep -q part; then
                CACHE_PART="/dev/$(lsblk -n -r -o NAME,TYPE "/dev/$dev" | grep part | sort -k 4 -n | tail -n 1 | cut -d' ' -f1)"
                CACHE_DEV="/dev/$dev"
                break
            else
                CACHE_DEV="/dev/$dev"
                break
            fi
        fi
    done
    
    if [ -n "$CACHE_DEV" ]; then
        echo "Found secondary cache volume: $CACHE_DEV"
        mkdir -p /home/ubuntu/.cache
        if [ -n "$CACHE_PART" ]; then
            echo "Mounting existing partition: $CACHE_PART"
            mount "$CACHE_PART" /home/ubuntu/.cache
            echo "$CACHE_PART /home/ubuntu/.cache ext4 defaults,nofail 0 2" >> /etc/fstab
        else
            echo "Formatting and mounting clean disk: $CACHE_DEV"
            mkfs.ext4 -F "$CACHE_DEV"
            mount "$CACHE_DEV" /home/ubuntu/.cache
            echo "$CACHE_DEV /home/ubuntu/.cache ext4 defaults,nofail 0 2" >> /etc/fstab
        fi
        break
    fi
    sleep 1
done

# Ensure model cache directories exist and have permissions
mkdir -p /home/ubuntu/.cache/huggingface /home/ubuntu/.cache/neuron
chmod -R 777 /home/ubuntu/.cache

# (Option B image is self-contained: weights + neffs + server baked in; no patches / model download.)



# --- Option B (optb) server: pull the public Docker Hub image and run it. No vLLM, no patching. ---
echo "Pulling Option B image {optb_image} ..."
for i in $(seq 1 12); do docker pull {optb_image} && break; sleep 20; done
docker rm -f gemma-optb 2>/dev/null || true
# --ipc=host is required for the TP build (neuronx-distributed parallel_model_load).
docker run -d --name gemma-optb --restart unless-stopped --ipc=host $devices -p 8080:8080 {optb_image}
echo "Option B TP server starting on :8080 (OpenAI-compatible; ~80-210s warmup incl. on-device graph load)."
"""
    return user_data


@mcp.tool()
def get_vllm_deployment_config(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-E2B-it",
    key_name: str = "alinux",
    gpu_memory_utilization: float = 0.95,
    instance_type: str = "inf2.8xlarge",
) -> str:
    """
    Generates the AWS CLI command and UserData script to deploy vLLM to an AWS EC2 Inferentia instance.

    Args:
        service_name: The Name tag for the EC2 instance.
        model_path: Hugging Face repo ID or S3 URI of the model.
        key_name: Key Pair name for SSH access (default: 'alinux').
        gpu_memory_utilization: The fraction of GPU VRAM to use for KV cache (GPU only, ignored for Inferentia).
        instance_type: The EC2 instance type (default: 'inf2.xlarge').
    """
    if any(q in model_path.lower() for q in ["qat", "w4a16", "ct"]):
        model_path = "google/gemma-4-E2B-it"

    image_id = "ami-04604f21b81ffbd87" # Fallback for us-east-1

    hf_token_expr = '$(aws ssm get-parameter --name /vllm/HF_TOKEN --with-decryption --query Parameter.Value --output text 2>/dev/null || echo \'\')'
    user_data = _get_inferentia_user_data(model_path, hf_token_expr, instance_type)
    aws_cmd = (
        f"aws ec2 run-instances \\\n"
        f"  --image-id {image_id} \\\n"
        f"  --instance-type {instance_type} \\\n"
        f"  --key-name {key_name} \\\n"
        f"  --tag-specifications 'ResourceType=instance,Tags=[{{Key=Name,Value={service_name}}}]' \\\n"
        f'  --instance-market-options \'{{"MarketType":"spot","SpotOptions":{{"SpotInstanceType":"one-time"}}}}\' \\\n'
        f"  --user-data file://user_data.sh"
    )

    return (
        f"### 🚀 AWS EC2 {instance_type} (AWS Inferentia) Option B (optb) Deployment Config\n\n"
        f"#### 1. UserData Script (`user_data.sh`):\n"
        f"```bash\n{user_data}\n```\n\n"
        f"#### 2. Run Instance CLI Command:\n"
        f"```bash\n{aws_cmd}\n```\n\n"
        f"#### 3. Prerequisites:\n"
        f"- No HF token needed — the `xbill9/gemma4-optb` image bakes in the weights + compiled neffs.\n"
        f"- Ensure the security group allows inbound TCP traffic on port `8080`.\n"
        f"- On `inf2.xlarge` (16GB RAM) the user-data adds swap (required for the neff-load peak) and uses the `:slim` image.\n"
        f"- *Note:* The resolved fallback AMI for AWS Neuron on Ubuntu 22.04 in region `{AWS_REGION}` is `{image_id}`."
    )


@mcp.tool()
async def deploy_vllm(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-E2B-it",
    key_name: str = "alinux",
    subnet_id: Optional[str] = None,
    instance_type: str = "inf2.8xlarge",
    market_type: str = "spot",
) -> str:
    """
    Deploys vLLM to AWS EC2 instance using AWS Inferentia.

    Args:
        service_name: Tag Name for the EC2 instance.
        model_path: Hugging Face repo ID or S3 URI.
        key_name: AWS EC2 Key Pair name (default: 'alinux').
        subnet_id: Subnet ID to launch in (optional).
        instance_type: EC2 instance type (default: 'inf2.xlarge').
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    # 1. Resolve Subnet and its VPC ID
    candidate_subnets = []
    vpc_id = None
    if subnet_id:
        try:
            subnets = ec2.describe_subnets(SubnetIds=[subnet_id])
            if subnets["Subnets"]:
                vpc_id = subnets["Subnets"][0]["VpcId"]
                candidate_subnets.append((subnet_id, vpc_id, subnets["Subnets"][0].get("AvailabilityZone")))
        except Exception as e:
            logger.warning(f"Failed to describe subnet {subnet_id}: {e}")

    if not subnet_id:
        try:
            subnets = ec2.describe_subnets(
                Filters=[
                    {"Name": "map-public-ip-on-launch", "Values": ["true"]},
                ]
            )
            for s in subnets.get("Subnets", []):
                # Skip us-east-1a since it doesn't support inf2.xlarge
                if not s["AvailabilityZone"].endswith("a"):
                    candidate_subnets.append((s["SubnetId"], s["VpcId"], s["AvailabilityZone"]))
        except Exception as e:
            logger.warning(f"Failed to discover subnets: {e}")

        if not candidate_subnets:
            try:
                subnets = ec2.describe_subnets()
                for s in subnets.get("Subnets", []):
                    candidate_subnets.append((s["SubnetId"], s["VpcId"], s["AvailabilityZone"]))
            except Exception as e:
                logger.warning(f"Failed to fetch fallback subnets: {e}")

    if candidate_subnets:
        vpc_id = candidate_subnets[0][1]

    if not vpc_id:
        try:
            vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
            vpc_id = vpcs["Vpcs"][0]["VpcId"] if vpcs["Vpcs"] else None
            if not vpc_id:
                vpcs = ec2.describe_vpcs()
                vpc_id = vpcs["Vpcs"][0]["VpcId"] if vpcs["Vpcs"] else None
        except Exception as e:
            logger.warning(f"Failed to resolve fallback VPC: {e}")

    # 3. Retrieve HF Token from Secret Manager/environment to inject
    hf_token = await get_secret() or ""

    # 4. Resolve Image ID and build UserData
    image_id = "ami-0fd664467b3cf8dfd" if AWS_REGION == "us-east-1" else "ami-01807ad0e6484b5a8"
    try:
        images_resp = ec2.describe_images(
            Owners=["amazon"],
            Filters=[{"Name": "name", "Values": ["*Deep Learning AMI Neuron*Ubuntu 22.04*"]}]
        )
        if images_resp.get("Images"):
            sorted_images = sorted(images_resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
            image_id = sorted_images[0]["ImageId"]
            logger.info(f"Dynamically resolved latest DLAMI Neuron AMI ID: {image_id}")
    except Exception as e:
        logger.info(f"Failed to describe DLAMI images: {e}. Using fallback `{image_id}`.")

    if any(q in model_path.lower() for q in ["qat", "w4a16", "ct"]):
        logger.warning("QAT compressed-tensors are GPU-only. Falling back to 'google/gemma-4-E2B-it' for Inferentia.")
        model_path = "google/gemma-4-E2B-it"

    user_data = _get_inferentia_user_data(model_path, hf_token, instance_type)

    # 5. Launch EC2 Instance
    last_error = None
    for sub_id, v_id, az in candidate_subnets:
        logger.info(f"Attempting deployment in subnet {sub_id} ({az}) VPC {v_id}...")

        # 2b. Resolve or create security group vllm-devops-sg in THIS VPC
        sg_name = "vllm-devops-sg"
        sg_id = None
        try:
            sgs = ec2.describe_security_groups(
                Filters=[
                    {"Name": "group-name", "Values": [sg_name]},
                    {"Name": "vpc-id", "Values": [v_id]}
                ]
            )
            if sgs["SecurityGroups"]:
                sg_id = sgs["SecurityGroups"][0]["GroupId"]
        except Exception as e:
            logger.info(f"Checking existing security group in {v_id} failed: {e}")

        if not sg_id:
            try:
                sg_res = ec2.create_security_group(
                    GroupName=sg_name, Description="Security Group for vLLM DevOps Agent", VpcId=v_id
                )
                sg_id = sg_res["GroupId"]
                ec2.authorize_security_group_ingress(
                    GroupId=sg_id,
                    IpPermissions=[
                        {
                            "IpProtocol": "tcp",
                            "FromPort": 22,
                            "ToPort": 22,
                            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Allow SSH"}],
                        },
                        {
                            "IpProtocol": "tcp",
                            "FromPort": 8080,
                            "ToPort": 8080,
                            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Allow vLLM API"}],
                        },
                    ],
                )
            except Exception as e:
                logger.warning(f"Failed to create/configure security group in VPC {v_id}: {str(e)}")
                last_error = e
                continue

        
        # Check for existing available cache volume in this AZ
        cache_volume_id = None
        try:
            v_resp = ec2.describe_volumes(
                Filters=[
                    {"Name": "tag:Name", "Values": [f"{service_name}-cache"]},
                    {"Name": "availability-zone", "Values": [az]},
                    {"Name": "status", "Values": ["available"]}
                ]
            )
            if v_resp.get("Volumes"):
                cache_volume_id = v_resp["Volumes"][0]["VolumeId"]
                logger.info(f"Found existing cache volume {cache_volume_id} in {az}. Reusing it.")
        except Exception as e:
            logger.warning(f"Failed to check for existing volumes in {az}: {e}")

        # Check if specified KeyName exists, if not, find first available key pair, or omit
        actual_key_name = None
        if key_name:
            try:
                kp_resp = ec2.describe_key_pairs(KeyNames=[key_name])
                if kp_resp.get("KeyPairs"):
                    actual_key_name = key_name
            except Exception:
                logger.info(f"Key pair '{key_name}' not found. Finding fallback key pair...")
                try:
                    kp_resp = ec2.describe_key_pairs()
                    if kp_resp.get("KeyPairs"):
                        actual_key_name = kp_resp["KeyPairs"][0]["KeyName"]
                        logger.info(f"Using fallback key pair: '{actual_key_name}'")
                except Exception as kpe:
                    logger.warning(f"Failed to describe key pairs: {kpe}")

        run_args = {
            "ImageId": image_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "SecurityGroupIds": [sg_id],
            "UserData": user_data,
            "TagSpecifications": [
                {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": service_name}]},
                {"ResourceType": "volume", "Tags": [{"Key": "Name", "Value": f"{service_name}-cache"}]}
            ],
            "IamInstanceProfile": {"Name": "aws-elasticbeanstalk-ec2-role"},
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 300,
                        "VolumeType": "gp3",
                        "Iops": 16000,
                        "Throughput": 1000,
                        "DeleteOnTermination": True,
                    }
                },
                {
                    "DeviceName": "/dev/sdf",
                    "Ebs": {
                        "VolumeSize": 50,
                        "VolumeType": "gp3",
                        "Iops": 3000,
                        "Throughput": 125,
                        "DeleteOnTermination": False, # Preserve cache volume
                    }
                }
            ],
        }
        if actual_key_name:
            run_args["KeyName"] = actual_key_name

        if market_type == "spot":
            run_args["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}}

        current_run_args = run_args.copy()

        if cache_volume_id:
            # Only use root in run_instances; we'll attach the existing cache volume manually
            current_run_args["BlockDeviceMappings"] = [run_args["BlockDeviceMappings"][0]]

        current_run_args["SubnetId"] = sub_id
        try:
            instance = ec2.run_instances(**current_run_args)
            inst_id = instance["Instances"][0]["InstanceId"]
            
            if cache_volume_id:
                try:
                    logger.info(f"Attaching existing volume {cache_volume_id} to {inst_id}...")
                    ec2.get_waiter('instance_exists').wait(InstanceIds=[inst_id])
                    ec2.attach_volume(VolumeId=cache_volume_id, InstanceId=inst_id, Device='/dev/sdf')
                except Exception as e:
                    logger.error(f"Failed to attach existing volume {cache_volume_id} to {inst_id}: {e}")

            market_desc = "Spot" if market_type == "spot" else "On-Demand"
            return (
                f"🚀 Successfully requested AWS EC2 {instance_type} {market_desc} Instance deployment for service '{service_name}'.\n"
                f"Instance ID: `{inst_id}`\n"
                f"Key Pair: `{key_name}`\n"
                f"Subnet ID: `{sub_id}` (AZ: {az})\n"
                f"AMI ID: `{image_id}`\n"
                + (f"Reusing Cache Volume: `{cache_volume_id}`\n" if cache_volume_id else "Created New Cache Volume (Persistent)\n") +
                f"Please wait a few minutes for the instance to initialize and pull the vLLM docker image."
            )
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Failed to launch in subnet {sub_id} ({az}): {err_msg}")
            if "InsufficientInstanceCapacity" in err_msg or "Unsupported" in err_msg:
                last_error = e
                continue
            else:
                return f"Failed to deploy AWS EC2 instance:\nError: {err_msg}"

    return f"Failed to deploy AWS EC2 instance (tried all Availability Zones):\nError: {str(last_error)}"


@mcp.tool()
async def start_ec2(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-E2B-it",
    key_name: str = "alinux",
    subnet_id: Optional[str] = None,
    instance_type: str = "inf2.xlarge",
    market_type: str = "on-demand",
    instance_id: Optional[str] = None,
) -> str:
    """
    Starts an existing stopped EC2 instance, or provisions a new one with AWS Inferentia if none exists.

    Args:
        service_name: Tag Name for the EC2 instance.
        model_path: Hugging Face repo ID or S3 URI (used if launching a new instance).
        key_name: AWS EC2 Key Pair name (default: 'alinux').
        subnet_id: Subnet ID to launch in (optional).
        instance_type: EC2 instance type (default: 'inf2.xlarge').
        market_type: Market type for the instance ('spot' or 'on-demand', default: 'on-demand').
        instance_id: Direct Instance ID to start if it already exists (optional).
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    # Check if instance already exists (either by instance_id or service_name)
    existing_instance_ids = []
    try:
        if instance_id:
            res = ec2.describe_instances(InstanceIds=[instance_id])
            for reservation in res.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    if inst["State"]["Name"] in ["stopped", "stopping"]:
                        existing_instance_ids.append(inst["InstanceId"])
        else:
            res = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Name", "Values": [service_name]},
                    {"Name": "instance-state-name", "Values": ["stopped", "stopping"]},
                ]
            )
            for reservation in res.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    existing_instance_ids.append(inst["InstanceId"])
    except Exception as e:
        logger.info(f"Checking existing instances returned: {e}")

    # If stopped instance exists, start it!
    if existing_instance_ids:
        try:
            ec2.start_instances(InstanceIds=existing_instance_ids)
            return f"🚀 Successfully requested start for existing stopped EC2 Instance(s): {', '.join(existing_instance_ids)}"
        except Exception as e:
            return f"Failed to start existing EC2 instance(s) {existing_instance_ids}:\nError: {str(e)}"

    # Otherwise, provision a new one!
    # 1. Resolve Subnet and its VPC ID
    vpc_id = None
    if subnet_id:
        try:
            subnets = ec2.describe_subnets(SubnetIds=[subnet_id])
            if subnets["Subnets"]:
                vpc_id = subnets["Subnets"][0]["VpcId"]
        except Exception as e:
            logger.warning(f"Failed to describe subnet {subnet_id}: {e}")

    if not subnet_id:
        try:
            subnets = ec2.describe_subnets(
                Filters=[
                    {"Name": "map-public-ip-on-launch", "Values": ["true"]},
                    {"Name": "availability-zone", "Values": [f"{AWS_REGION}f", f"{AWS_REGION}d", f"{AWS_REGION}a"]},
                ]
            )
            if subnets["Subnets"]:
                subnet_id = subnets["Subnets"][0]["SubnetId"]
                vpc_id = subnets["Subnets"][0]["VpcId"]
            else:
                subnets = ec2.describe_subnets()
                if subnets["Subnets"]:
                    subnet_id = subnets["Subnets"][0]["SubnetId"]
                    vpc_id = subnets["Subnets"][0]["VpcId"]
        except Exception as e:
            logger.warning(f"Failed to discover subnet: {e}")

    if not vpc_id:
        try:
            vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
            vpc_id = vpcs["Vpcs"][0]["VpcId"] if vpcs["Vpcs"] else None
            if not vpc_id:
                vpcs = ec2.describe_vpcs()
                vpc_id = vpcs["Vpcs"][0]["VpcId"] if vpcs["Vpcs"] else None
        except Exception as e:
            logger.warning(f"Failed to resolve fallback VPC: {e}")

    # 2. Resolve or create security group vllm-devops-sg in that VPC
    sg_name = "vllm-devops-sg"
    sg_id = None
    try:
        sgs = ec2.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [sg_name]},
                {"Name": "vpc-id", "Values": [vpc_id]}
            ]
        )
        if sgs["SecurityGroups"]:
            sg_id = sgs["SecurityGroups"][0]["GroupId"]
    except Exception as e:
        logger.info(f"Checking existing security group failed: {e}")

    if not sg_id:
        try:
            sg_res = ec2.create_security_group(
                GroupName=sg_name, Description="Security Group for vLLM DevOps Agent", VpcId=vpc_id
            )
            sg_id = sg_res["GroupId"]
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Allow SSH"}],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8080,
                        "ToPort": 8080,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Allow vLLM API"}],
                    },
                ],
            )
        except Exception as e:
            return f"Failed to create/configure security group in VPC {vpc_id}: {str(e)}"

    # 3. Retrieve HF Token
    try:
        hf_token = await get_secret() or ""
    except Exception as e:
        logger.warning(f"Failed to retrieve HF token: {e}. Defaulting to empty.")
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or ""
    ami_type = "neuron-ubuntu-22.04"
    image_id = "ami-04604f21b81ffbd87"

    try:
        ssm = boto3.client("ssm", region_name=AWS_REGION)
        path = f"/aws/service/deeplearning/ami/x86_64/{ami_type}/latest/ami-id"
        response = ssm.get_parameter(Name=path)
        image_id = response["Parameter"]["Value"]
    except Exception as e:
        logger.info(f"Failed to fetch {ami_type} DLAMI dynamically via SSM: {e}. Using fallback `{image_id}`.")

    if any(q in model_path.lower() for q in ["qat", "w4a16", "ct"]):
        logger.warning("QAT compressed-tensors are GPU-only. Falling back to 'google/gemma-4-E2B-it' for Inferentia.")
        model_path = "google/gemma-4-E2B-it"

    user_data = _get_inferentia_user_data(model_path, hf_token, instance_type)

    # 5. Launch EC2 Instance
    try:
        run_args = {
            "ImageId": image_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "KeyName": key_name,
            "SecurityGroupIds": [sg_id],
            "UserData": user_data,
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": service_name}]}],
            "IamInstanceProfile": {"Name": "aws-elasticbeanstalk-ec2-role"},
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 300,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    }
                }
            ],
        }
        if subnet_id:
            run_args["SubnetId"] = subnet_id

        if market_type.lower() == "spot":
            run_args["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}}

        instance = ec2.run_instances(**run_args)
        inst_id = instance["Instances"][0]["InstanceId"]

        return (
            f"🚀 Successfully requested AWS EC2 {instance_type} {market_type} Instance deployment for service '{service_name}'.\n"
            f"Instance ID: `{inst_id}`\n"
            f"Key Pair: `{key_name}`\n"
            f"Subnet ID: `{subnet_id}`\n"
            f"AMI ID: `{image_id}`\n"
            f"Please wait a few minutes for the instance to initialize and pull the vLLM docker image."
        )
    except Exception as e:
        return f"Failed to deploy AWS EC2 instance:\nError: {str(e)}"


@mcp.tool()
async def destroy_vllm(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Cleans up the vLLM Docker container on the AWS EC2 instance(s) matching the specified service Name tag,
    without terminating the EC2 instance(s).

    Args:
        service_name: Name tag of the instance(s) to clean up.
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    ssm = boto3.client("ssm", region_name=AWS_REGION)

    try:
        response = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [service_name]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]},
            ]
        )
        instance_ids = []
        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                instance_ids.append(instance["InstanceId"])

        if not instance_ids:
            return f"No active EC2 instances found matching service name tag '{service_name}' to clean up."

        # Send SSM command to stop and remove container
        cmd_response = ssm.send_command(
            InstanceIds=instance_ids,
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": ["docker ps -aq --filter name=gemma | xargs -r docker rm -f || true"]},
        )
        command_id = cmd_response["Command"]["CommandId"]

        return (
            f"🧹 Successfully requested cleanup of the Option B (`gemma-*`) Docker container(s) on EC2 Instance(s): {', '.join(instance_ids)}.\n"
            f"SSM Command ID: `{command_id}` (EC2 instance(s) remain running)."
        )
    except Exception as e:
        return f"Failed to clean up container for service '{service_name}':\nError: {str(e)}"


@mcp.tool()
def stop_ec2(
    service_name: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> str:
    """
    Stops AWS EC2 instance(s) by service name tag or instance ID.

    Args:
        service_name: Name tag of the instance(s) to stop (optional).
        instance_id: Direct Instance ID to stop (optional).
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    instance_ids = []

    if instance_id:
        instance_ids.append(instance_id)
    else:
        target_name = service_name or DEFAULT_SERVICE_NAME
        try:
            response = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Name", "Values": [target_name]},
                    {"Name": "instance-state-name", "Values": ["pending", "running"]},
                ]
            )
            for reservation in response.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    instance_ids.append(inst["InstanceId"])
        except Exception as e:
            return f"Failed to search for EC2 instances to stop:\nError: {str(e)}"

    if not instance_ids:
        target = f"Instance ID '{instance_id}'" if instance_id else f"service tag '{service_name or DEFAULT_SERVICE_NAME}'"
        return f"No active/pending EC2 instances found to stop matching {target}."

    try:
        ec2.stop_instances(InstanceIds=instance_ids)
        return f"🛑 Successfully requested stopping for EC2 Instance(s): {', '.join(instance_ids)}"
    except Exception as e:
        return f"Failed to stop EC2 instances {instance_ids}:\nError: {str(e)}"


@mcp.tool()
def status_vllm(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Checks the status of the AWS EC2 instance(s) matching the specified service Name tag.

    Args:
        service_name: Name tag of the instance(s) to check.
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    try:
        response = ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [service_name]}])
        instances_info = []
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                info = (
                    f"- **Instance ID**: `{inst['InstanceId']}`\n"
                    f"  - **Type**: `{inst['InstanceType']}`\n"
                    f"  - **State**: `{inst['State']['Name']}`\n"
                    f"  - **Public IP**: `{inst.get('PublicIpAddress', 'None')}`\n"
                    f"  - **Public DNS**: `{inst.get('PublicDnsName', 'None')}`\n"
                    f"  - **Launch Time**: `{inst['LaunchTime']}`\n"
                )
                instances_info.append(info)

        if not instances_info:
            return f"No EC2 instances found matching service name tag '{service_name}'."

        return f"### AWS EC2 Status for service tag '{service_name}':\n\n" + "\n".join(instances_info)
    except Exception as e:
        return f"Failed to get status for service '{service_name}':\nError: {str(e)}"


@mcp.tool()
def status_ec2(
    service_name: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> str:
    """
    Checks the status of AWS EC2 instance(s) by service name tag or instance ID.

    Args:
        service_name: Name tag of the instance(s) to check (optional).
        instance_id: Direct Instance ID to check (optional).
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    filters = []
    instance_ids = []

    if instance_id:
        instance_ids.append(instance_id)
    elif service_name:
        filters.append({"Name": "tag:Name", "Values": [service_name]})
    else:
        filters.append({"Name": "tag:Name", "Values": [DEFAULT_SERVICE_NAME]})

    try:
        run_args = {}
        if filters:
            run_args["Filters"] = filters
        if instance_ids:
            run_args["InstanceIds"] = instance_ids

        response = ec2.describe_instances(**run_args)
        instances_info = []
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                info = (
                    f"- **Instance ID**: `{inst['InstanceId']}`\n"
                    f"  - **Type**: `{inst['InstanceType']}`\n"
                    f"  - **State**: `{inst['State']['Name']}`\n"
                    f"  - **Public IP**: `{inst.get('PublicIpAddress', 'None')}`\n"
                    f"  - **Public DNS**: `{inst.get('PublicDnsName', 'None')}`\n"
                    f"  - **Launch Time**: `{inst['LaunchTime']}`\n"
                )
                instances_info.append(info)

        if not instances_info:
            target = f"Instance ID '{instance_id}'" if instance_id else f"service tag '{service_name or DEFAULT_SERVICE_NAME}'"
            return f"No EC2 instances found matching {target}."

        target_desc = f"Instance ID '{instance_id}'" if instance_id else f"service tag '{service_name or DEFAULT_SERVICE_NAME}'"
        return f"### AWS EC2 Status for {target_desc}:\n\n" + "\n".join(instances_info)
    except Exception as e:
        return f"Failed to get status for EC2 target:\nError: {str(e)}"


@mcp.tool()
async def check_vllm(
    service_name: str = DEFAULT_SERVICE_NAME,
    instance_id: Optional[str] = None,
) -> str:
    """
    Checks the status of the vLLM container and engine running on the EC2 instance(s).

    Args:
        service_name: Name tag of the instance(s) to check.
        instance_id: Direct Instance ID to check (optional).
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    ssm = boto3.client("ssm", region_name=AWS_REGION)

    # 1. Resolve instance
    filters = []
    instance_ids = []
    if instance_id:
        instance_ids.append(instance_id)
    else:
        filters.append({"Name": "tag:Name", "Values": [service_name]})
        filters.append({"Name": "instance-state-name", "Values": ["pending", "running"]})

    try:
        run_args = {}
        if filters:
            run_args["Filters"] = filters
        if instance_ids:
            run_args["InstanceIds"] = instance_ids

        response = ec2.describe_instances(**run_args)
        instances = []
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                instances.append(inst)
    except Exception as e:
        return f"Failed to describe EC2 instances:\nError: {str(e)}"

    if not instances:
        target = f"Instance ID '{instance_id}'" if instance_id else f"active service tag '{service_name}'"
        return f"No active EC2 instances found matching {target}."

    reports = []
    for inst in instances:
        inst_id = inst["InstanceId"]
        state = inst["State"]["Name"]
        ip = inst.get("PublicIpAddress")

        report = f"### 🖥️ Instance: `{inst_id}` ({state})\n"
        if state != "running":
            report += f"❌ Instance is not running (Current State: `{state}`). Skipping container checks.\n"
            reports.append(report)
            continue

        if not ip:
            report += "⚠️ No Public IP associated with this running instance.\n"
            reports.append(report)
            continue

        # 2. Check Docker Container status via SSM
        docker_status = "Unknown (SSM Unreachable)"
        try:
            cmd_res = ssm.send_command(
                InstanceIds=[inst_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": ["docker ps -a --filter name=gemma --format '{{.Names}}: {{.Status}}' 2>/dev/null | head -1 | grep . || echo 'no gemma-* container'"]},
            )
            cmd_id = cmd_res["Command"]["CommandId"]

            # Poll for completion (up to 5 seconds)
            for _ in range(5):
                await asyncio.sleep(1)
                try:
                    result = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=inst_id)
                    if result["Status"] == "Success":
                        docker_status = result["StandardOutputContent"].strip()
                        break
                    elif result["Status"] in ["Failed", "TimedOut", "Cancelled"]:
                        docker_status = f"Failed (SSM Status: {result['Status']})"
                        break
                except Exception:
                    pass
        except Exception as e:
            docker_status = f"Error querying SSM: {str(e)}"

        report += f"- **Option B Container (`gemma-*`)**: `{docker_status}`\n"

        # 3. Check vLLM HTTP health endpoint
        http_status = "Unreachable"
        try:
            async with httpx.AsyncClient(timeout=3) as http_client:
                res = await http_client.get(f"http://{ip}:8080/health")
                if res.status_code == 200:
                    http_status = "Healthy ✅"
                else:
                    http_status = f"Unhealthy (HTTP Code: {res.status_code}) ❌"
        except Exception as e:
            http_status = f"Unreachable (Error: {e}) ❌"

        report += f"- **vLLM API Endpoint (`http://{ip}:8080/health`)**: `{http_status}`\n"
        reports.append(report)

    return "\n".join(reports)


@mcp.tool()
def update_vllm_scaling(instance_type: str, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Updates the EC2 instance type (scaling vertically) for the vLLM service instance.
    Note: The instance must be stopped to change its type.

    Args:
        instance_type: The new AWS EC2 instance type (e.g. 'g6.4xlarge', 'g6.2xlarge').
        service_name: The Name tag of the instance to scale.
    """
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    try:
        response = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [service_name]},
                {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
            ]
        )
        instances = []
        for reservation in response.get("Reservations", []):
            instances.extend(reservation.get("Instances", []))

        if not instances:
            return f"No active EC2 instances found matching service name tag '{service_name}'."

        inst = instances[0]
        inst_id = inst["InstanceId"]
        current_type = inst["InstanceType"]
        state = inst["State"]["Name"]

        if state == "running":
            ec2.stop_instances(InstanceIds=[inst_id])
            return (
                f"⏸️ Instance `{inst_id}` is currently running. We have requested it to stop to perform vertical scaling.\n"
                f"Current Type: `{current_type}` -> Target Type: `{instance_type}`.\n"
                f"Please wait for the instance to stop, then run this tool again to apply the new instance type."
            )
        elif state == "stopped":
            ec2.modify_instance_attribute(InstanceId=inst_id, InstanceType={"Value": instance_type})
            ec2.start_instances(InstanceIds=[inst_id])
            return (
                f"⚡ Successfully scaled EC2 instance `{inst_id}` from `{current_type}` to `{instance_type}`.\n"
                f"Requested the instance to start back up."
            )
        else:
            return f"Instance `{inst_id}` is in state '{state}'. Vertical scaling is only supported when stopped."
    except Exception as e:
        return f"Failed to scale service '{service_name}':\nError: {str(e)}"


@mcp.tool()
def get_vllm_gpu_deployment_config(
    cluster_name: str = "eks-gpu-cluster",
    model_name: str = "google/gemma-4-E2B-it",
    instance_type: str = "inf2.xlarge",
) -> str:
    """
    Generates an AWS EKS manifest and node group instructions for deploying vLLM on AWS Inferentia.

    Args:
        cluster_name: The name of the EKS cluster.
        model_name: The model identifier.
        instance_type: The EC2 instance type (default: 'inf2.xlarge').
    """
    if any(q in model_name.lower() for q in ["qat", "w4a16", "ct"]):
        model_name = "google/gemma-4-E2B-it"

    manifest = f"""
### 🌀 vLLM on EKS AWS Inferentia (AWS Neuron Deployment)

To deploy vLLM on EKS with AWS Inferentia, use the following EKS Nodegroup config and Kubernetes manifest. This targets **AWS Inferentia2** devices via **{instance_type}** instances.

#### 1. Create a Neuron Node Group via eksctl
```bash
eksctl create nodegroup \\
    --cluster={cluster_name} \\
    --region={AWS_REGION} \\
    --name=neuron-nodes \\
    --node-type={instance_type} \\
    --nodes=1 \\
    --nodes-min=1 \\
    --nodes-max=2 \\
    --node-volume-size=150
```

#### 2. Kubernetes Manifest (vllm-neuron.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-neuron
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-neuron
  template:
    metadata:
      labels:
        app: vllm-neuron
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: {instance_type}
      containers:
      - name: vllm-neuron
        image: public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04
        resources:
          limits:
            aws.amazon.com/neuron: "1"
          requests:
            aws.amazon.com/neuron: "1"
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - "--model={model_name}"
        - "--quantization=neuron_quant"
        - "--max-model-len=16384"
        - "--tensor-parallel-size=2"
        ports:
        - containerPort: 8080
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
  name: vllm-neuron-service
spec:
  selector:
    app: vllm-neuron
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

#### 3. Deployment Steps
1. Install AWS Neuron Device Plugin for Kubernetes on your EKS cluster:
   `kubectl apply -f https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/src/k8s/neuron-device-plugin.yml`
2. Save the YAML above to `vllm-neuron.yaml`.
3. Apply it: `kubectl apply -f vllm-neuron.yaml`.
"""
    return manifest






@mcp.tool()
async def get_huggingfacehub_download_path(
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
def get_huggingface_model_copy_instructions(
    repo_id: str = "google/gemma-4-E2B-it",
    bucket_name: Optional[str] = None,
) -> str:
    """
    Provides instructions and commands to transfer Gemma model weights from Hugging Face to your S3 bucket.

    Args:
        repo_id: The Hugging Face repo ID (e.g., 'google/gemma-4-E2B-it').
        bucket_name: The target S3 bucket name (defaults to AWS_BUCKET_NAME).
    """
    if not bucket_name:
        bucket_name = AWS_BUCKET_NAME

    model_name = repo_id.split("/")[-1]

    instructions = f"""
### 📦 Transferring {model_name} from Hugging Face to AWS S3

To use Hugging Face weights with vLLM on EC2 via S3, follow these steps:

#### Option A: Download directly inside EC2 instance (Recommended)
You don't need to copy weights to S3 if you specify the Hugging Face Repo ID directly in `deploy_vllm`.
vLLM will download it automatically from Hugging Face when starting, using your `HF_TOKEN`.

#### Option B: Upload to S3 Bucket
If you prefer to host weights privately in S3:

1. **Download Model locally:**
   `python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('{repo_id}'))"`

2. **Upload to S3:**
   The command above outputs the local path. Use it to copy the artifacts:
   `aws s3 cp --recursive /path/to/downloaded/model/ s3://{bucket_name}/{model_name}/`

Once uploaded, you can deploy using:
`deploy_vllm(model_path="s3://{bucket_name}/{model_name}/")`
"""
    return instructions


@mcp.tool()
def check_gpu_quotas(region: Optional[str] = None) -> str:
    """
    Checks AWS EC2 Inferentia/Neuron quotas for a specific AWS region (via Service Quotas).

    Args:
        region: The AWS region (defaults to AWS_REGION).
    """
    if not region:
        region = AWS_REGION

    try:
        import boto3

        sq = boto3.client("service-quotas", region_name=region)
        quotas_to_check = [
            {"QuotaCode": "L-1945791B", "Name": "Running On-Demand Inf instances"},
            {"QuotaCode": "L-B5D1601B", "Name": "All Inf Spot Instance Requests"},
        ]
        report = f"### 📊 AWS EC2 Inferentia Quotas for region `{region}`\n\n"
        for q in quotas_to_check:
            try:
                res = sq.get_service_quota(ServiceCode="ec2", QuotaCode=q["QuotaCode"])
                quota = res["Quota"]
                report += f"- **{q['Name']}** ({quota['QuotaName']}):\n"
                report += f"  - Limit: `{quota['Value']}` (vCPUs)\n"
                report += f"  - Adjustable: `{quota['Adjustable']}`\n"
            except Exception as e:
                report += f"- **{q['Name']}** ({q['QuotaCode']}): Could not fetch quota ({str(e)})\n"
        return report
    except Exception as e:
        return f"Failed to check AWS EC2 quotas in `{region}`: {str(e)}"


@mcp.tool()
async def verify_model_health() -> str:
    """Runs a deep health check with latency reporting on the AWS Inferentia vLLM model."""
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
async def query_gemma4(prompt: str) -> str:
    """Queries the self-hosted Gemma 4 model on AWS Inferentia."""
    logger.info(f"Querying AWS Inferentia model with prompt: '{prompt[:50]}...'")
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
async def query_gemma4_with_stats(prompt: str) -> str:
    """
    Queries the self-hosted Gemma 4 model on AWS Inferentia and returns detailed performance statistics.

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
async def get_model_details() -> str:
    """Retrieves detailed information about the running AWS Inferentia model, engine, and versions."""
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
        async with httpx.AsyncClient(timeout=10) as http_client:
            try:
                res = await http_client.get(f"{vllm_url}/health")
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
async def get_system_status(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Provides a high-level dashboard of AWS EC2 system status and vLLM health.

    Args:
        service_name: The name tag of the EC2 service instance.
    """
    health = "🔴 Offline"
    url = None
    try:
        url = get_vllm_url()
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health")
            if res.status_code == 200:
                health = f"🟢 Online ({url})"
            else:
                health = f"🔴 Offline (Status {res.status_code}) ({url})"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

    ec2_status = "🔴 Unknown"
    try:
        import boto3

        ec2 = boto3.client("ec2", region_name=AWS_REGION)
        response = ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [service_name]}])
        instances = []
        for reservation in response.get("Reservations", []):
            instances.extend(reservation.get("Instances", []))

        if instances:
            # Prioritize active (running/pending) instances
            active_instances = [i for i in instances if i["State"]["Name"] in ["running", "pending"]]
            target_instance = active_instances[0] if active_instances else instances[0]
            state = target_instance["State"]["Name"]
            inst_id = target_instance["InstanceId"]
            if state == "running":
                ec2_status = f"🟢 Running ({inst_id})"
            else:
                ec2_status = f"🔴 {state.capitalize()} ({inst_id})"
        else:
            ec2_status = "🔴 Instance Not Found"
    except Exception as e:
        ec2_status = f"🔴 AWS Error: {str(e)}"

    if "🟢" in health:
        next_step = "Use `query_gemma4` to interact with the model."
    else:
        next_step = f"Call `deploy_vllm` to provision/start the AWS EC2 instance `{service_name}`."

    return (
        f"### 🌀 Inferentia vLLM System Status\n"
        f"- **vLLM Health:** {health}\n"
        f"- **Hosting Status:** {ec2_status}\n"
        f"**👉 Next Step:** {next_step}"
    )


@mcp.tool()
async def get_endpoint(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Returns the active vLLM service URL if available.

    Args:
        service_name: The name of the service or instance Name tag to query.
    """
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health", headers=headers)
            if res.status_code == 200:
                return f"🟢 vLLM is Online at: {url}"
            else:
                return f"🔴 vLLM is configured at {url} but returned status {res.status_code}."
    except Exception as e:
        return f"🔴 vLLM endpoint check failed: {e}. Try deploying/starting it with `deploy_vllm`."


@mcp.tool()
async def run_benchmark(
    model: Optional[str] = None,
    num_prompts: int = 20,
    random_output_len: int = 128,
    max_concurrency: int = 8,
) -> str:
    """
    Runs a performance/concurrency benchmark sweep against the AWS Inferentia vLLM endpoint.

    Args:
        model: Model name to request (defaults to the active model from /v1/models).
        num_prompts: Number of requests to send per concurrency level.
        random_output_len: Max tokens to generate per request.
        max_concurrency: Maximum concurrency level to sweep up to (powers of 2, e.g. 1, 2, 4, 8).
    """
    from datetime import datetime

    try:
        url = get_vllm_url()
    except Exception as e:
        return f"❌ Cannot run benchmark: {e}"

    # Get active model name if not provided
    client = await get_vllm_client()
    if not model:
        model = await get_active_model_name(client)

    base_url = f"{url.rstrip('/')}/v1/completions"
    prompt = "Explain the importance of Site Reliability Engineering for large scale AI deployments."

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
            "prompt": prompt,
            "max_tokens": random_output_len,
            "temperature": 0.0,
            "stream": False,
        }
        async with sem:
            start_time = time.perf_counter()
            try:
                response = await http_client.post(base_url, json=payload, timeout=120)
                end_time = time.perf_counter()
                if response.status_code == 200:
                    latency = end_time - start_time
                    data = response.json()
                    tokens = data.get("usage", {}).get("completion_tokens", random_output_len)
                    return {"success": True, "latency": latency, "tokens": tokens}
                else:
                    return {"success": False, "error": f"Status {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

    # Warmup
    logger.info("Warming up model for benchmark...")
    async with httpx.AsyncClient() as http_client:
        await send_request(http_client, asyncio.Semaphore(1))

    logger.info(f"Starting Inferentia benchmark sweep against {url} with model {model}...")
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
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
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

    # Save to CSV
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
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

    summary_str = f"### 📊 Inferentia Benchmark Results (Model: `{model}`)\n\n"
    summary_str += "| Concurrency | Success Rate | Req/s | Tokens/s | Avg Latency | P95 Latency |\n"
    summary_str += "|---:|---:|---:|---:|---:|---:|\n"
    for r in results:
        summary_str += f"| {r['concurrency']} | {r['success_rate']:.1%} | {r['req_per_sec']:.2f} | {r['tokens_per_sec']:.2f} | {r['avg_latency']:.2f}s | {r['p95_latency']:.2f}s |\n"
    summary_str += f"\n\nResults saved to `{output_file}`"
    return summary_str


async def fetch_ec2_logs(instance_id: str, limit: int = 50) -> str:
    """Fetches docker logs from the running EC2 instance via SSM Run Command."""
    try:
        import boto3

        ssm = boto3.client("ssm", region_name=AWS_REGION)
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [f"CN=$(docker ps -a --filter name=gemma --format '{{{{.Names}}}}' | head -1); docker logs --tail {limit} \"${{CN:-gemma-optb}}\" 2>&1"]},
        )
        command_id = response["Command"]["CommandId"]

        # Poll for completion (up to 10 seconds)
        for _ in range(10):
            await asyncio.sleep(1)
            try:
                result = ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id,
                )
                if result["Status"] in ["Success", "Failed", "TimedOut", "Cancelled"]:
                    if result["Status"] == "Success":
                        return result["StandardOutputContent"]
                    else:
                        return f"SSM Command failed: {result.get('StandardErrorContent')}"
            except Exception:
                pass
        return "SSM Command timed out."
    except Exception as e:
        return f"Failed to fetch logs via SSM: {str(e)}"


@mcp.tool()
async def analyze_gpu_logs(limit: int = 15, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Fetches vLLM logs for the specified service and uses Gemma 4 to analyze them for errors.

    Args:
        limit: Number of log entries to fetch.
        service_name: AWS EC2 instance tag Name.
    """
    # Try AWS SSM logs first if AWS credentials exist
    try:
        import boto3

        ec2 = boto3.client("ec2", region_name=AWS_REGION)
        response = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [service_name]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )
        instances = []
        for reservation in response.get("Reservations", []):
            instances.extend(reservation.get("Instances", []))

        if instances:
            inst_id = instances[0]["InstanceId"]
            logger.info(f"Fetching EC2 logs for instance {inst_id} via SSM...")
            raw_logs = await fetch_ec2_logs(inst_id, limit)
            # Prepare prompt for Gemma
            prompt = f"Analyze the following vLLM docker container logs and provide a high-level summary of critical issues:\n\n{raw_logs}\n\nSummary:"
            client = await get_vllm_client()
            model_name = await get_active_model_name(client)
            chat_completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                max_tokens=512,
                temperature=0.2,
            )
            response_text = chat_completion.choices[0].message.content or ""
            return f"### AWS EC2 Log Analysis (Self-Hosted vLLM)\n\n{response_text}"
    except Exception as e:
        return f"Failed to fetch/analyze AWS EC2 logs: {str(e)}"

    return "No active EC2 instance logs found."


@mcp.tool()
async def get_raw_ec2_logs(limit: int = 100, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Fetches raw docker container logs directly from the active EC2 instance via SSM.
    Use this when the model is offline, failing to start, or to view raw stack traces.

    Args:
        limit: Number of log lines to retrieve (default 100).
        service_name: Name tag of the EC2 instance.
    """
    try:
        import boto3

        ec2 = boto3.client("ec2", region_name=AWS_REGION)
        response = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [service_name]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )
        instances = []
        for reservation in response.get("Reservations", []):
            instances.extend(reservation.get("Instances", []))

        if not instances:
            return "No running EC2 instance found for this service name."

        inst_id = instances[0]["InstanceId"]
        logger.info(f"Retrieving raw EC2 logs for instance {inst_id} via SSM...")
        raw_logs = await fetch_ec2_logs(inst_id, limit)
        return f"### Raw vLLM Container Logs ({inst_id})\n\n```\n{raw_logs}\n```"
    except Exception as e:
        return f"Failed to retrieve raw EC2 logs: {str(e)}"


@mcp.tool()
async def get_help() -> str:
    """Provides help text and summarizes the configuration options and all available SRE/DevOps tools for this AWS MCP server."""
    return (
        "### 🛠️ AWS Gemma 4 SRE Agent Help & Configuration\n\n"
        "You can configure this MCP server using the following environment variables:\n\n"
        "**AWS Configuration:**\n"
        f"- **`AWS_REGION`**: The AWS Region for EC2/EKS deployment.\n"
        f"  - *Current Value:* `{AWS_REGION}`\n"
        f"- **`AWS_BUCKET_NAME`**: S3 Bucket used to store model weights.\n"
        f"  - *Current Value:* `{AWS_BUCKET_NAME}`\n\n"
        "**General serving:**\n"
        f"- **`MODEL_NAME`**: Default Hugging Face repository or path.\n"
        f"  - *Current Value:* `{MODEL_NAME}`\n"
        f"- **`VLLM_BASE_URL`**: The explicit URL of your vLLM service. (If not set, it is auto-discovered via EC2 tags)\n"
        f"  - *Current Value:* `{VLLM_BASE_URL or 'Not set (auto-discovering)'}`\n\n"
        "### ℹ️ Active Mode Summary\n"
        "The server is running in AWS Inferentia mode.\n\n"
        "### 🧰 Available MCP Tools\n\n"
        "Below is a summary of the tools exposed by this SRE/DevOps agent:\n\n"
        "#### 🐳 Infrastructure & Deployment\n"
        "- **`start_ec2`**: Starts an existing stopped EC2 instance, or provisions a new one (with AWS Inferentia) if none exists.\n"
        "- **`status_ec2`**: Checks the state, type, public IP, DNS, and launch details of EC2 instances.\n"
        "- **`stop_ec2`**: Safely stops active EC2 instances without deleting the root EBS volumes.\n"
        "- **`check_vllm`**: Checks the status of the vLLM container and engine running on the EC2 instance(s).\n"
        "- **`deploy_vllm`**: Deploys vLLM to AWS EC2 inf2.xlarge.\n"
        "- **`destroy_vllm`**: Cleans up the vLLM Docker container on the AWS EC2 instance without terminating it.\n"
        "- **`status_vllm`**: Checks the status of the AWS EC2 instance vLLM service.\n"
        "- **`update_vllm_scaling`**: Scales EC2 instance type vertically.\n"
        "- **`get_vllm_deployment_config`**: Generates the AWS EC2 deployment command and user data.\n"
        "- **`get_vllm_gpu_deployment_config`**: Generates an AWS EKS nodegroup config and Kubernetes manifest for Inferentia.\n"
        "- **`check_gpu_quotas`**: Checks Inferentia/Neuron quotas for an AWS region.\n\n"
        "#### 📊 Model Management\n"
        "- **`list_bucket_models`**: Lists model weights in S3 bucket.\n"
        "- **`save_hf_token`**: Securely saves a Hugging Face API token to AWS Secrets Manager.\n"
        "- **`get_huggingface_model_copy_instructions`**: Instructions to download model from Hugging Face and upload to S3.\n"
        "- **`get_huggingfacehub_download_path`**: Resolves local cache path using huggingface_hub.\n\n"
        "#### 📊 Monitoring & Status\n"
        "- **`get_metrics`**: Fetches raw Prometheus metrics from the running vLLM service's /metrics endpoint.\n"
        "- **`get_system_status`**: Provides a high-level status dashboard of the service and health.\n"
        "- **`get_endpoint`**: Verifies connectivity and returns the active service URL.\n"
        "- **`get_model_details`**: Retrieves detailed model metadata and engine state from `/v1/models`.\n"
        "- **`verify_model_health`**: Deep health check by querying the model with a simple prompt and measuring latency.\n\n"
        "#### 📈 Performance & Benchmarking\n"
        "- **`run_benchmark`**: Runs performance/concurrency benchmark sweeps against the vLLM Inferentia endpoint.\n\n"
        "#### 💬 Interaction & Diagnostics\n"
        "- **`query_gemma4`**: Primary tool to query the self-hosted model with standard chat message format.\n"
        "- **`query_gemma4_with_stats`**: Queries the model and returns streaming performance statistics (TTFT, throughput).\n"
        "- **`query_vllm`**: Direct text completions querying tool.\n"
        "- **`analyze_cloud_logging`**: Fetches logs from AWS CloudWatch and analyzes them using the model.\n"
        "- **`analyze_gpu_logs`**: Fetches service logs and uses Gemma 4 to analyze them for SRE/DevOps errors.\n"
        "- **`suggest_sre_remediation`**: Suggests remediation plans for SRE errors using the model.\n"
    )


@mcp.tool()
async def get_metrics() -> str:
    """
    Fetches the Prometheus metrics from the active vLLM service.
    """
    try:
        url = get_vllm_url()
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{url}/metrics")
            if res.status_code == 200:
                return res.text
            else:
                return f"🔴 Failed to retrieve metrics. Status code: {res.status_code}\n\nResponse:\n{res.text[:1000]}"
    except Exception as e:
        return f"🔴 Error fetching metrics: {e}"


if __name__ == "__main__":
    mcp.run()
