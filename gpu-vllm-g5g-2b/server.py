"""EC2 G5g (Graviton2 + NVIDIA T4G) lifecycle and inference MCP server.

Like the Inf2 sibling, this uses boto3 rather than shelling out to the AWS CLI so
it works with profiles, environment credentials, IAM roles, and SSO-backed
credential processes, and uses Systems Manager Run Command for remote
administration — no inbound SSH rule or private key.

What is different about this rig, and why most of the code below exists:

G5g pairs a Graviton2 (aarch64) host with a T4G GPU (Turing, SM 7.5). Every
prebuilt CUDA artifact in the ecosystem covers one of those two axes but not
both. `vllm/vllm-openai:v0.27.1` publishes an arm64 manifest, and it is compiled
for `8.0 8.7 8.9 9.0 10.0 11.0 12.0` — the amd64 manifest of the same tag is the
one that carries 7.5. So the single arch this rig needs falls in the gap between
the two published images, and the Dockerfile sets no `+PTX`, so there is no JIT
fallback to rescue it. See docs/turing-aarch64-gap.md.
"""

import asyncio
import base64
import logging
import os
import time
from typing import Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # Keep offline schema/tests usable before `pip install -r`.
    boto3 = None

    class BotoCoreError(Exception):
        """Fallback used only when the optional AWS dependency is absent."""

    class ClientError(Exception):
        """Fallback used only when the optional AWS dependency is absent."""


# This rig's identity. The directory name is the single identifier everything else
# derives from — MCP server name, log channel, and the ManagedBy tag that scopes
# instance discovery. A literal rather than basename(__file__) because the
# installed skill copy lives at .claude/skills/<skill>/mcp/server.py, where
# deriving from the path would yield "mcp".
RIG_NAME = "gpu-vllm-g5g-2b"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(RIG_NAME)

MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", RIG_NAME)
mcp = FastMCP(MCP_SERVER_NAME)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-E2B-it")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "g5g.2xlarge")
SERVICE_NAME = os.getenv("SERVICE_NAME", "vllm-g5g")
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))

# Turing has no bf16 and no fp8 datapath. These are not tuning knobs on this
# part; bfloat16 fails outright and fp8 KV is unavailable. tpu.env explains.
DTYPE = os.getenv("DTYPE", "float16")
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto")
ATTENTION_BACKEND = os.getenv("VLLM_ATTENTION_BACKEND", "XFORMERS")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.90")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "8"))

VLLM_STOCK_IMAGE = os.getenv("VLLM_STOCK_IMAGE", "vllm/vllm-openai:v0.27.1")
VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm-openai:v0.27.1-sm75-arm64")
TORCH_CUDA_ARCH_LIST = os.getenv("TORCH_CUDA_ARCH_LIST", "7.5")
VLLM_REF = os.getenv("VLLM_REF", "v0.27.1")
# AWS publishes the ARM64 GPU DLAMI as a public SSM parameter. Prefer it: it is
# single-valued and authoritative, where a describe-images name filter is a fuzzy
# match over a set that also contains ARM64 DLAMIs with NO NVIDIA driver (built
# for Graviton CPU inference). Those match a loose "Deep Learning*ARM64*Ubuntu*"
# pattern, can be the newest by CreationDate, and boot perfectly well on a G5g
# with no GPU — a failure that looks like a broken container, not a wrong AMI.
DLAMI_SSM_PARAMETER = os.getenv(
    "DLAMI_SSM_PARAMETER",
    "/aws/service/deeplearning/ami/arm64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id",
)
# Fallback only, and deliberately narrower than the pattern it replaced: it
# requires the driver in the name so the driverless images cannot match.
DLAMI_NAME = os.getenv("DLAMI_NAME", "Deep Learning ARM64 AMI*Nvidia Driver*Ubuntu*")

MANAGED_BY = RIG_NAME

# G5g topology: (GPUs, host RAM GiB). Verified against the AWS G5g product page.
_G5G_SIZES = {
    "g5g.xlarge": (1, 8),
    "g5g.2xlarge": (1, 16),
    "g5g.4xlarge": (1, 32),
    "g5g.8xlarge": (1, 64),
    "g5g.16xlarge": (2, 128),
    "g5g.metal": (2, 128),
}

# MODELS.md: E2B is 9.5 GiB of bf16 weights (8.97 measured). Staging that through
# host RAM does not fit g5g.xlarge's 8 GiB, so the smallest size is rejected for
# this model rather than left to OOM-kill the container at load time.
_MIN_HOST_RAM_GB = 16


def _session():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed; run `python3 -m pip install -r requirements.txt`")
    kwargs = {"region_name": AWS_REGION}
    if AWS_PROFILE:
        kwargs["profile_name"] = AWS_PROFILE
    return boto3.Session(**kwargs)


def _client(service: str):
    return _session().client(service)


def _is_g5g(instance_type: str) -> bool:
    return instance_type in _G5G_SIZES


def _gpu_count(instance_type: str) -> int:
    return _G5G_SIZES.get(instance_type, (0, 0))[0]


def _host_memory_gb(instance_type: str) -> int:
    return _G5G_SIZES.get(instance_type, (0, 0))[1]


def _validate_instance_type(instance_type: str) -> None:
    if not _is_g5g(instance_type):
        raise ValueError(f"instance_type must be one of {', '.join(sorted(_G5G_SIZES))}")
    if _host_memory_gb(instance_type) < _MIN_HOST_RAM_GB:
        raise ValueError(
            f"{instance_type} has {_host_memory_gb(instance_type)} GiB of host RAM; "
            f"E2B stages 9.5 GiB of weights and needs at least {_MIN_HOST_RAM_GB} GiB. "
            "Use g5g.2xlarge or larger."
        )


def _tensor_parallel_size(instance_type: str) -> int:
    return _gpu_count(instance_type)


async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)


def _serve_flags(model: str, instance_type: str) -> str:
    """vLLM flags for Turing. Deliberately unlike the L4 rigs' flag set."""
    return (
        f"--model {model} --host 0.0.0.0 --port 8000 "
        f"--dtype {DTYPE} --kv-cache-dtype {KV_CACHE_DTYPE} "
        f"--tensor-parallel-size {_tensor_parallel_size(instance_type)} "
        f"--gpu-memory-utilization {GPU_MEMORY_UTILIZATION} "
        f"--max-model-len {MAX_MODEL_LEN} --max-num-seqs {MAX_NUM_SEQS}"
    )


def _user_data(model: str, instance_type: str, serving: str = "build") -> str:
    """Render idempotent cloud-init for the chosen serving stack.

    ``build`` compiles vLLM for SM 7.5 on the instance and then serves. This is
    the only path that can work on T4G, and it is slow — a from-source vLLM build
    on a Graviton2 takes hours. It writes progress to /var/log/vllm-build.log and
    drops /opt/vllm-g5g/BUILD_DONE when the image exists.

    ``stock`` runs the published arm64 image unchanged. It is expected to fail
    with "no kernel image is available for execution on the device" and exists so
    that failure can be reproduced on real hardware rather than asserted. Use
    verify_gpu_arch first — it answers the same question in minutes, not hours.
    """
    _validate_instance_type(instance_type)
    if serving not in {"build", "stock"}:
        raise ValueError("serving must be 'build' or 'stock'")

    flags = _serve_flags(model, instance_type)
    gpus = _gpu_count(instance_type)

    common = """#!/usr/bin/env bash
set -euxo pipefail
systemctl enable --now docker
mkdir -p /opt/vllm-g5g
"""

    if serving == "stock":
        return common + f"""cat >/opt/vllm-g5g/start.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
HF_TOKEN=$(aws secretsmanager get-secret-value --region {AWS_REGION} --secret-id {HF_SECRET_ID} --query SecretString --output text 2>/dev/null || true)
docker rm -f vllm-g5g 2>/dev/null || true
docker run -d --name vllm-g5g --restart unless-stopped --ipc=host \\
  --gpus all -e HF_TOKEN="$HF_TOKEN" \\
  -e VLLM_ATTENTION_BACKEND={ATTENTION_BACKEND} \\
  -p {VLLM_PORT}:8000 \\
  {VLLM_STOCK_IMAGE} \\
  {flags}
SCRIPT
chmod 700 /opt/vllm-g5g/start.sh
/opt/vllm-g5g/start.sh
"""

    return common + f"""cat >/opt/vllm-g5g/build.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
# The published arm64 image is compiled for SM 8.0+; T4G is SM 7.5 and the
# Dockerfile adds no +PTX, so nothing JITs. Rebuild the same tag with the arch
# list the release pipeline overrides away on arm64.
if docker image inspect {VLLM_IMAGE} >/dev/null 2>&1; then
  echo "image {VLLM_IMAGE} already present"
  exit 0
fi
rm -rf /opt/vllm-g5g/src
git clone --depth 1 --branch {VLLM_REF} https://github.com/vllm-project/vllm.git /opt/vllm-g5g/src
cd /opt/vllm-g5g/src
docker build --platform linux/arm64 \\
  --build-arg torch_cuda_arch_list='{TORCH_CUDA_ARCH_LIST}' \\
  --target vllm-openai \\
  -f docker/Dockerfile \\
  -t {VLLM_IMAGE} .
SCRIPT
chmod 700 /opt/vllm-g5g/build.sh

cat >/opt/vllm-g5g/start.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
HF_TOKEN=$(aws secretsmanager get-secret-value --region {AWS_REGION} --secret-id {HF_SECRET_ID} --query SecretString --output text 2>/dev/null || true)
docker rm -f vllm-g5g 2>/dev/null || true
docker run -d --name vllm-g5g --restart unless-stopped --ipc=host \\
  --gpus all -e HF_TOKEN="$HF_TOKEN" \\
  -e VLLM_ATTENTION_BACKEND={ATTENTION_BACKEND} \\
  -p {VLLM_PORT}:8000 \\
  {VLLM_IMAGE} \\
  {flags}
SCRIPT
chmod 700 /opt/vllm-g5g/start.sh

# Build then serve, detached: cloud-init must not block for hours.
nohup bash -c '/opt/vllm-g5g/build.sh && touch /opt/vllm-g5g/BUILD_DONE && /opt/vllm-g5g/start.sh' \\
  >/var/log/vllm-build.log 2>&1 &
echo "vLLM SM 7.5 build started for {gpus} GPU(s); follow /var/log/vllm-build.log"
"""


async def _resolve_ami(ec2=None) -> str:
    """Resolve the ARM64 **GPU** DLAMI for this region.

    Two things have to hold and they are separate: the image must be arm64 (the
    x86_64 DLAMI ids hardcoded by the legacy tips-tree rigs cannot boot on
    Graviton2 at all), and it must carry the NVIDIA driver (an ARM64 DLAMI built
    for Graviton CPU inference boots fine on a G5g and simply has no GPU).

    The SSM public parameter pins both. describe-images is the fallback.
    """
    try:
        result = await _call(_client("ssm").get_parameter, Name=DLAMI_SSM_PARAMETER)
        return result["Parameter"]["Value"]
    except Exception as exc:
        logger.info("SSM DLAMI lookup failed (%s); falling back to describe-images", exc)

    ec2 = ec2 or _client("ec2")
    result = await _call(
        ec2.describe_images,
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": [DLAMI_NAME]},
            {"Name": "architecture", "Values": ["arm64"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(result.get("Images", []), key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError(
            f"No arm64 GPU DLAMI in {AWS_REGION} via SSM ({DLAMI_SSM_PARAMETER}) "
            f"or name filter {DLAMI_NAME!r}. Override with DLAMI_SSM_PARAMETER or DLAMI_NAME."
        )
    return images[0]["ImageId"]


async def _instances(name: str | None = None, states: list[str] | None = None):
    filters = [
        {"Name": "tag:ManagedBy", "Values": [MANAGED_BY]},
        {"Name": "instance-state-name", "Values": states or ["pending", "running", "stopping", "stopped"]},
    ]
    if name:
        filters.append({"Name": "tag:Name", "Values": [name]})
    response = await _call(_client("ec2").describe_instances, Filters=filters)
    return [i for r in response.get("Reservations", []) for i in r.get("Instances", [])]


async def _ssm(instance_id: str, command: str, timeout: int = 300) -> str:
    ssm = _client("ssm")
    response = await _call(
        ssm.send_command,
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command]},
        TimeoutSeconds=timeout,
    )
    command_id = response["Command"]["CommandId"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = await _call(
                ssm.get_command_invocation,
                CommandId=command_id,
                InstanceId=instance_id,
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "InvocationDoesNotExist":
                await asyncio.sleep(2)
                continue
            raise
        if result["Status"] in {"Success", "Failed", "TimedOut", "Cancelled"}:
            output = (
                result.get("StandardOutputContent", "") + result.get("StandardErrorContent", "")
            ).strip()
            if result["Status"] != "Success":
                raise RuntimeError(f"SSM {result['Status']}: {output}")
            return output
        await asyncio.sleep(2)
    raise TimeoutError(f"SSM command did not finish in {timeout}s")


def _error(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        detail = exc.response.get("Error", {})
        return f"❌ AWS {detail.get('Code', 'error')}: {detail.get('Message', exc)}"
    if isinstance(exc, BotoCoreError):
        return f"❌ AWS client error: {exc}"
    return f"❌ {exc}"


@mcp.tool(title="Save Hugging Face token", annotations=WRITE)
async def save_hf_token(token: str) -> str:
    """Create or update the configured AWS Secrets Manager secret."""
    secrets = _client("secretsmanager")
    try:
        try:
            await _call(secrets.put_secret_value, SecretId=HF_SECRET_ID, SecretString=token)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            await _call(secrets.create_secret, Name=HF_SECRET_ID, SecretString=token)
        return f"✅ Stored token in Secrets Manager secret `{HF_SECRET_ID}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Generate G5g deployment configuration", annotations=READ_ONLY)
async def get_deployment_config(
    model_name: str = MODEL_NAME,
    instance_type: str = INSTANCE_TYPE,
    subnet_id: str = "<subnet-id>",
    security_group_id: str = "<security-group-id>",
    iam_instance_profile: str = "g5g-vllm-instance-profile",
    serving: Literal["build", "stock"] = "build",
    spot: bool = True,
) -> str:
    """Return cloud-init and an AWS CLI launch command without changing AWS."""
    try:
        _validate_instance_type(instance_type)
        script = _user_data(model_name, instance_type, serving)
        encoded = base64.b64encode(script.encode()).decode()
        market = (
            "--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' "
            if spot
            else ""
        )
        note = (
            "serving='build' compiles vLLM for SM 7.5 on the instance (hours on Graviton2)."
            if serving == "build"
            else "serving='stock' runs the published arm64 image, which has no SM 7.5 kernels "
            "and is expected to fail. Run verify_gpu_arch instead."
        )
        return (
            f"### EC2 G5g deployment ({instance_type}, {_gpu_count(instance_type)}x T4G)\n\n```bash\n"
            f"# ARM64 *GPU* DLAMI. The SSM parameter pins both the architecture and the\n"
            f"# NVIDIA driver; a name filter can also match driverless ARM64 DLAMIs.\n"
            f"AMI_ID=$(aws ssm get-parameter --region {AWS_REGION} "
            f"--name {DLAMI_SSM_PARAMETER} "
            "--query 'Parameter.Value' --output text)\n"
            f'aws ec2 run-instances --region {AWS_REGION} --image-id "$AMI_ID" '
            f"--instance-type {instance_type} --subnet-id {subnet_id} "
            f"--security-group-ids {security_group_id} "
            f"--iam-instance-profile Name={iam_instance_profile} "
            f"{market}"
            f"--block-device-mappings 'DeviceName=/dev/sda1,Ebs={{VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}}' "
            f"--user-data '{encoded}' --tag-specifications "
            f"'ResourceType=instance,Tags=[{{Key=Name,Value={SERVICE_NAME}}},"
            f"{{Key=ManagedBy,Value={MANAGED_BY}}}]'\n```\n\n"
            f"{note}\n\n"
            f"Serving flags: `{_serve_flags(model_name, instance_type)}`\n"
            f"dtype is `{DTYPE}`, not bfloat16 — Turing has no bf16 datapath."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Create G5g instance", annotations=WRITE)
async def create_g5g_instance(
    subnet_id: str,
    security_group_id: str,
    iam_instance_profile: str,
    name: str = SERVICE_NAME,
    model_name: str = MODEL_NAME,
    instance_type: str = INSTANCE_TYPE,
    serving: Literal["build", "stock"] = "build",
    spot: bool = True,
) -> str:
    """Launch one tagged G5g instance using the latest regional ARM64 DLAMI.

    serving='build' compiles a Turing-capable vLLM on the instance, then serves.
    serving='stock' runs the published arm64 image, which lacks SM 7.5 kernels.
    Spot is the default; pass spot=False for on-demand.
    """
    try:
        _validate_instance_type(instance_type)
        if await _instances(name):
            return f"❌ A managed instance named `{name}` already exists."
        ec2 = _client("ec2")
        args = {
            "ImageId": await _resolve_ami(ec2),
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "SubnetId": subnet_id,
            "SecurityGroupIds": [security_group_id],
            "IamInstanceProfile": {"Name": iam_instance_profile},
            "UserData": _user_data(model_name, instance_type, serving),
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True},
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": name},
                        {"Key": "ManagedBy", "Value": MANAGED_BY},
                    ],
                }
            ],
        }
        if spot:
            args["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {"SpotInstanceType": "one-time"},
            }
        response = await _call(ec2.run_instances, **args)
        instance_id = response["Instances"][0]["InstanceId"]
        market = "spot" if spot else "on-demand"
        tail = (
            f"Building vLLM for SM {TORCH_CUDA_ARCH_LIST} on the instance — this takes hours. "
            "Follow with get_build_progress."
            if serving == "build"
            else f"Running `{VLLM_STOCK_IMAGE}` unchanged; expect a kernel-image failure."
        )
        return (
            f"✅ Launching `{instance_id}` ({instance_type}, {market}, "
            f"{_gpu_count(instance_type)}x T4G) in `{AWS_REGION}`.\n{tail}"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List managed G5g instances", annotations=READ_ONLY)
async def list_g5g_instances() -> str:
    """List instances tagged ManagedBy=gpu-vllm-g5g-2b."""
    try:
        found = await _instances()
        if not found:
            return f"No instances tagged `ManagedBy={MANAGED_BY}` in `{AWS_REGION}`."
        lines = ["| Instance | Type | State | Private IP | Name |", "| --- | --- | --- | --- | --- |"]
        for inst in found:
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            lines.append(
                f"| `{inst['InstanceId']}` | {inst['InstanceType']} | {inst['State']['Name']} "
                f"| {inst.get('PrivateIpAddress', '-')} | {tags.get('Name', '-')} |"
            )
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Start G5g instance", annotations=WRITE)
async def start_g5g_instance(instance_id: str) -> str:
    """Start a stopped managed instance."""
    try:
        await _call(_client("ec2").start_instances, InstanceIds=[instance_id])
        return f"✅ Starting `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Stop G5g instance", annotations=DESTRUCTIVE)
async def stop_g5g_instance(instance_id: str) -> str:
    """Stop a running managed instance. One-time spot instances cannot be
    stopped, only terminated."""
    try:
        await _call(_client("ec2").stop_instances, InstanceIds=[instance_id])
        return f"🛑 Stopping `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Terminate G5g instance", annotations=DESTRUCTIVE)
async def terminate_g5g_instance(instance_id: str) -> str:
    """Terminate a managed instance. Permanent — the locally built SM 7.5 image
    dies with the root volume and the next launch rebuilds it from source."""
    try:
        await _call(_client("ec2").terminate_instances, InstanceIds=[instance_id])
        return f"🗑️ Terminating `{instance_id}`. The built vLLM image is lost with the volume."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify GPU compute capability and kernel coverage", annotations=READ_ONLY)
async def verify_gpu_arch(instance_id: str, image: str = "") -> str:
    """Measure whether an image's CUDA kernels actually cover this GPU.

    This is the rig's central check and the cheapest way to settle the question
    the whole rig turns on. It reports the device's compute capability, the arch
    list the installed torch was compiled for, and the result of one real matmul
    on the device. A config flag being accepted proves nothing; a kernel either
    launches or it does not.

    Defaults to the published arm64 image, where SM 7.5 is expected to be absent.
    Pass image=VLLM_IMAGE to check a locally built one.
    """
    target = image or VLLM_STOCK_IMAGE
    probe = (
        "import torch;"
        "print('device:', torch.cuda.get_device_name(0));"
        "print('capability:', torch.cuda.get_device_capability(0));"
        "print('torch arch list:', torch.cuda.get_arch_list());"
        "x = torch.randn(256, 256, device='cuda', dtype=torch.float16);"
        "print('matmul ok:', float((x @ x).sum()) == float((x @ x).sum()))"
    )
    command = (
        f"nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true; "
        f'docker run --rm --gpus all --entrypoint python3 {target} -c "{probe}" 2>&1 || true'
    )
    try:
        output = await _ssm(instance_id, command, timeout=600)
        verdict = (
            "\n\n❌ No SM 7.5 in the arch list — this image cannot run on T4G."
            if "7.5" not in output and "arch list" in output
            else ""
        )
        if "no kernel image is available" in output:
            verdict = (
                "\n\n❌ Confirmed: `no kernel image is available for execution on the device`. "
                "The image has no SM 7.5 kernels and no PTX to JIT from."
            )
        return f"### GPU arch probe on `{instance_id}` ({target})\n\n```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get vLLM build progress", annotations=READ_ONLY)
async def get_build_progress(instance_id: str, tail: int = 40) -> str:
    """Tail the from-source vLLM build started by serving='build'."""
    try:
        tail = max(1, min(tail, 5000))
        command = (
            "test -f /opt/vllm-g5g/BUILD_DONE && echo 'BUILD COMPLETE' || echo 'BUILD IN PROGRESS'; "
            f"tail -n {tail} /var/log/vllm-build.log 2>/dev/null || echo 'no build log yet'"
        )
        return f"```\n{await _ssm(instance_id, command)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get vLLM logs", annotations=READ_ONLY)
async def get_vllm_logs(instance_id: str, tail: int = 100) -> str:
    """Tail the vLLM container log."""
    try:
        tail = max(1, min(tail, 5000))
        return f"```\n{await _ssm(instance_id, f'docker logs --tail {tail} vllm-g5g 2>&1')}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get G5g endpoint", annotations=READ_ONLY)
async def get_endpoint(instance_id: str) -> str:
    """Resolve the instance's OpenAI-compatible base URL. Never hardcoded."""
    try:
        response = await _call(_client("ec2").describe_instances, InstanceIds=[instance_id])
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                host = inst.get("PublicIpAddress") or inst.get("PrivateIpAddress")
                if not host:
                    return f"❌ `{instance_id}` has no reachable address yet."
                return f"📡 `http://{host}:{VLLM_PORT}/v1`"
        return f"❌ `{instance_id}` not found in `{AWS_REGION}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify model health", annotations=READ_ONLY)
async def verify_model_health(instance_id: str) -> str:
    """Check /health and confirm the served model reports a non-empty completion.

    Uses /v1/chat/completions: raw /v1/completions returns an empty completion on
    `-it` models, so an empty body there is not evidence of a broken deploy.
    """
    try:
        endpoint = await get_endpoint(instance_id)
        if not endpoint.startswith("📡"):
            return endpoint
        base = endpoint.strip("📡 `")
        async with httpx.AsyncClient(timeout=60) as client:
            health = await client.get(base.replace("/v1", "/health"))
            chat = await client.post(
                f"{base}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                    "max_tokens": 16,
                },
            )
        body = chat.json()
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        status = "✅" if health.status_code == 200 and text.strip() else "❌"
        return (
            f"{status} health={health.status_code} tokens="
            f"{body.get('usage', {}).get('completion_tokens', 0)} reply={text!r}"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Query model", annotations=READ_ONLY)
async def query_model(instance_id: str, prompt: str, max_tokens: int = 256) -> str:
    """Send a chat completion to the served model."""
    try:
        endpoint = await get_endpoint(instance_id)
        if not endpoint.startswith("📡"):
            return endpoint
        base = endpoint.strip("📡 `")
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{base}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
            )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check G5g quotas", annotations=READ_ONLY)
async def check_g5g_quotas() -> str:
    """Report the On-Demand and Spot G instance vCPU quotas for the region.

    G5g draws on the same 'Running On-Demand G and VT instances' quota as other
    G-family types, counted in vCPUs — a g5g.2xlarge needs 8.
    """
    try:
        quotas = _client("service-quotas")
        wanted = {
            "L-DB2E81BA": "Running On-Demand G and VT instances (vCPU)",
            "L-3819A6DF": "All G and VT Spot Instance Requests (vCPU)",
        }
        lines = [f"### G-family quotas in `{AWS_REGION}`", "", "| Quota | vCPUs |", "| --- | --- |"]
        for code, label in wanted.items():
            try:
                result = await _call(
                    quotas.get_service_quota, ServiceCode="ec2", QuotaCode=code
                )
                lines.append(f"| {label} | {int(result['Quota']['Value'])} |")
            except ClientError as exc:
                lines.append(f"| {label} | unavailable ({exc.response['Error']['Code']}) |")
        lines.append("")
        lines.append(f"`{INSTANCE_TYPE}` needs {_host_memory_gb(INSTANCE_TYPE) // 2} vCPUs.")
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Help and configuration", annotations=READ_ONLY)
async def get_help() -> str:
    """Show this rig's resolved configuration and the constraints that shape it."""
    return f"""### {RIG_NAME}

Serving `{MODEL_NAME}` with vLLM on **EC2 G5g** — AWS Graviton2 (aarch64) host,
NVIDIA **T4G** GPU (Turing, SM 7.5, 16 GB).

| Setting | Value |
| --- | --- |
| Region | `{AWS_REGION}` |
| Instance type | `{INSTANCE_TYPE}` ({_gpu_count(INSTANCE_TYPE)}x T4G, {_host_memory_gb(INSTANCE_TYPE)} GiB RAM) |
| Tensor parallel | `{_tensor_parallel_size(INSTANCE_TYPE)}` |
| dtype | `{DTYPE}` |
| KV cache dtype | `{KV_CACHE_DTYPE}` |
| Attention backend | `{ATTENTION_BACKEND}` |
| Built image | `{VLLM_IMAGE}` (arch list `{TORCH_CUDA_ARCH_LIST}`) |
| Published image | `{VLLM_STOCK_IMAGE}` (no SM 7.5 — cannot serve) |
| Managed-by tag | `{MANAGED_BY}` |

**The constraint that defines this rig.** G5g needs aarch64 *and* SM 7.5. The
published `vllm/vllm-openai` arm64 manifest is compiled for `8.0 8.7 8.9 9.0
10.0 11.0 12.0`; the amd64 manifest of the same tag is the one carrying 7.5. The
Dockerfile sets no `+PTX`, so nothing JITs. `serving='build'` rebuilds the image
with `--build-arg torch_cuda_arch_list={TORCH_CUDA_ARCH_LIST}`.

**Turing has no bf16 and no fp8.** `--dtype bfloat16` and `--kv-cache-dtype fp8`
are copied throughout the L4 sibling rigs and both are wrong here.

Start with `verify_gpu_arch` — it settles in minutes what the build path costs
hours to discover.
"""


if __name__ == "__main__":
    mcp.run()
