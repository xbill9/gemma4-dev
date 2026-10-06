"""AWS Trainium (trn1) lifecycle and inference MCP server, w8a8 checkpoint.

Forked from tpu-pytorch-trn1-2b and differing in slot 5 only: CHECKPOINT names
xbill9/gemma-4-E2B-it-qat-w8a8-int8, which runs through the native engine
(torch_generate.py, ct_load.py) by hand. The tools deploy the same prebuilt dense
E2B Option-B image as the base rig.

There is no vLLM path here; tpu-vllm-trn1-2b holds that attempt.

boto3 rather than the AWS CLI, so profiles, SSO and roles all work. Remote
administration uses Systems Manager Run Command; no inbound rule or key.
"""

import asyncio
import base64
import logging
import os
import time
from typing import Annotated

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from openai import AsyncOpenAI
from pydantic import Field

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # Keep offline schema/tests usable before `pip install -r`.
    boto3 = None

    class BotoCoreError(Exception):
        """Fallback used only when the optional AWS dependency is absent."""

    class ClientError(Exception):
        """Fallback used only when the optional AWS dependency is absent."""


# This rig's identity: the MCP server name, the log channel, the skill stem and the
# ManagedBy tag all derive from it (NAMING.md). A literal, because the installed
# skill copy lives at .claude/skills/<skill>/mcp/server.py.
RIG_NAME = "tpu-pytorch-trn1-2b-w8a8"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(RIG_NAME)

MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", RIG_NAME)
mcp = MCPServer(MCP_SERVER_NAME)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

# trn1.2xlarge is offered in one us-east-2 zone (us-east-2c) and not at all in
# us-east-1 or us-west-1 as of 2026-10-04, hence the region default.
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-2")
AWS_PROFILE = os.getenv("AWS_PROFILE")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "trn1.2xlarge")
SERVICE_NAME = os.getenv("SERVICE_NAME", RIG_NAME)
SERVE_PORT = int(os.getenv("SERVE_PORT", "8000"))
NEURON_AMI_NAME = os.getenv("NEURON_AMI_NAME", "Deep Learning AMI Neuron (Ubuntu 24.04)*")
MANAGED_BY = RIG_NAME
CONTAINER = "gemma4-neuron"
# The prebuilt Gemma-4 E2B Option-B image: a torch_neuronx-traced graph with the
# neffs and host embeddings baked in, serving OpenAI routes on container port 8080.
# Other single-device builds (xbill9/gemma4-optb-e4b:slim, -12b:slim-devprefill,
# -26b:xlarge) take the same run line; the model is a property of the image.
OPTB_IMAGE = os.getenv("OPTB_IMAGE", "docker.io/xbill9/gemma4-optb:slim")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-4-E2B-it")
# This rig's checkpoint. No tool deploys it: the image above serves the dense
# reference, and the checkpoint runs through torch_generate.py by hand.
CHECKPOINT = os.getenv("CHECKPOINT", "xbill9/gemma-4-E2B-it-qat-w8a8-int8")

NEURON_DEVICES = {"trn1.2xlarge": 1, "trn1.32xlarge": 16, "trn1n.32xlarge": 16}


def _session():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed; run `python3 -m pip install -r requirements.txt`")
    kwargs = {"region_name": AWS_REGION}
    if AWS_PROFILE:
        kwargs["profile_name"] = AWS_PROFILE
    return boto3.Session(**kwargs)


def _client(service: str):
    return _session().client(service)


def _neuron_devices(instance_type: str) -> int:
    return NEURON_DEVICES.get(instance_type, 0)


def _neuron_cores(instance_type: str) -> int:
    return _neuron_devices(instance_type) * 2


def _validate_instance_type(instance_type: str) -> None:
    if not _neuron_devices(instance_type):
        raise ValueError(f"instance_type must be one of {', '.join(NEURON_DEVICES)}")


def _user_data(instance_type: str, image: str = OPTB_IMAGE) -> str:
    """Render idempotent cloud-init that serves one prebuilt single-device image.

    The 16 GB swapfile is insurance rather than a requirement: E2B's neff load
    peaked at 19.53 GB on the 32 GiB trn1.2xlarge. The 26B single-box image
    pulls ~28 GB into the /data volume on first start.
    """
    _validate_instance_type(instance_type)
    return f"""#!/usr/bin/env bash
set -euxo pipefail
if ! swapon --show --noheadings | grep -q /swapfile; then
  fallocate -l 16G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
systemctl enable --now docker
mkdir -p /opt/{RIG_NAME}
cat >/opt/{RIG_NAME}/start.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
docker rm -f {CONTAINER} 2>/dev/null || true
docker run -d --name {CONTAINER} --restart unless-stopped --ipc=host \\
  --device=/dev/neuron0 \\
  -v gemma4-data:/data \\
  -p {SERVE_PORT}:8080 \\
  {image}
SCRIPT
chmod 700 /opt/{RIG_NAME}/start.sh
/opt/{RIG_NAME}/start.sh
"""


async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)


async def _resolve_ami(ec2=None) -> str:
    ec2 = ec2 or _client("ec2")
    result = await _call(
        ec2.describe_images,
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": [NEURON_AMI_NAME]},
            {"Name": "architecture", "Values": ["x86_64"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(result.get("Images", []), key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError(f"No Neuron DLAMI matching {NEURON_AMI_NAME!r} in {AWS_REGION}")
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
            result = await _call(ssm.get_command_invocation, CommandId=command_id, InstanceId=instance_id)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "InvocationDoesNotExist":
                await asyncio.sleep(2)
                continue
            raise
        if result["Status"] in {"Success", "Failed", "TimedOut", "Cancelled"}:
            output = (result.get("StandardOutputContent", "") + result.get("StandardErrorContent", "")).strip()
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


@mcp.tool(title="Generate trn1 deployment configuration", annotations=READ_ONLY)
async def get_deployment_config(
    instance_type: str = INSTANCE_TYPE,
    subnet_id: str = "<subnet-id>",
    security_group_id: str = "<security-group-id>",
    iam_instance_profile: str = "<instance-profile-with-AmazonSSMManagedInstanceCore>",
    image: str = OPTB_IMAGE,
    spot: bool = True,
) -> str:
    """Return cloud-init and an AWS CLI launch command without changing AWS."""
    try:
        script = _user_data(instance_type, image)
        encoded = base64.b64encode(script.encode()).decode()
        market = (
            "--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' " if spot else ""
        )
        return (
            f"### AWS Trainium deployment\n\n```bash\n"
            f"AMI_ID=$(aws ec2 describe-images --region {AWS_REGION} --owners amazon "
            f"--filters 'Name=name,Values={NEURON_AMI_NAME}' 'Name=state,Values=available' "
            "--query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)\n"
            f'aws ec2 run-instances --region {AWS_REGION} --image-id "$AMI_ID" '
            f"--instance-type {instance_type} --subnet-id {subnet_id} "
            f"--security-group-ids {security_group_id} "
            f"--iam-instance-profile Name={iam_instance_profile} "
            f"--metadata-options HttpTokens=required {market}"
            f"--block-device-mappings 'DeviceName=/dev/sda1,Ebs={{VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}}' "
            f"--user-data '{encoded}' --tag-specifications "
            f"'ResourceType=instance,Tags=[{{Key=Name,Value={SERVICE_NAME}}},"
            f"{{Key=ManagedBy,Value={MANAGED_BY}}}]'\n```\n\n"
            f"Serves `{image}` on port {SERVE_PORT} from {_neuron_devices(instance_type)} Neuron device(s) / "
            f"{_neuron_cores(instance_type)} NeuronCore(s); the image uses one device."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Create trn1 instance", annotations=WRITE)
async def create_trn1_instance(
    subnet_id: str,
    security_group_id: str,
    iam_instance_profile: str,
    name: str = SERVICE_NAME,
    instance_type: str = INSTANCE_TYPE,
    image: str = OPTB_IMAGE,
    spot: bool = True,
) -> str:
    """Launch one tagged trn1 instance on the latest regional Neuron DLAMI.

    It serves a prebuilt single-device Gemma-4 Neuron image (default E2B).
    Spot is the default; Trn spot quota starts at 0 vCPUs on a new account,
    so a quota error here is expected until an increase is approved — pass
    spot=False for on-demand.
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
            "MetadataOptions": {"HttpTokens": "required"},
            "UserData": _user_data(instance_type, image),
            "BlockDeviceMappings": [
                {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True}}
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [{"Key": "Name", "Value": name}, {"Key": "ManagedBy", "Value": MANAGED_BY}],
                }
            ],
        }
        if spot:
            args["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}}
        response = await _call(ec2.run_instances, **args)
        instance_id = response["Instances"][0]["InstanceId"]
        market = "spot" if spot else "on-demand"
        return f"✅ Launching `{instance_id}` ({instance_type}, {market}) in `{AWS_REGION}` serving `{image}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List managed trn1 instances", annotations=READ_ONLY)
async def list_trn1_instances() -> str:
    """List instances tagged ManagedBy=tpu-pytorch-trn1-2b-w8a8."""
    try:
        instances = await _instances()
        if not instances:
            return "No managed trn1 instances found."
        rows = ["instance_id\tname\ttype\tstate\tprivate_ip\tpublic_ip"]
        for item in instances:
            tags = {t["Key"]: t["Value"] for t in item.get("Tags", [])}
            rows.append(
                "\t".join(
                    [
                        item["InstanceId"],
                        tags.get("Name", ""),
                        item["InstanceType"],
                        item["State"]["Name"],
                        item.get("PrivateIpAddress", "-"),
                        item.get("PublicIpAddress", "-"),
                    ]
                )
            )
        return "\n".join(rows)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Stop trn1 instance", annotations=DESTRUCTIVE)
async def stop_trn1_instance(instance_id: str) -> str:
    """Stop an on-demand trn1 instance, preserving its EBS volume (spot cannot be stopped)."""
    try:
        await _call(_client("ec2").stop_instances, InstanceIds=[instance_id])
        return f"✅ Stopping `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Start trn1 instance", annotations=WRITE)
async def start_trn1_instance(instance_id: str) -> str:
    """Start a stopped trn1 instance."""
    try:
        await _call(_client("ec2").start_instances, InstanceIds=[instance_id])
        return f"✅ Starting `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Terminate trn1 instance", annotations=DESTRUCTIVE)
async def terminate_trn1_instance(instance_id: str) -> str:
    """Permanently terminate an instance. Root EBS is deleted by default."""
    try:
        await _call(_client("ec2").terminate_instances, InstanceIds=[instance_id])
        return f"✅ Terminating `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check trn1 and Neuron health", annotations=READ_ONLY)
async def verify_neuron_health(instance_id: str) -> str:
    """Use SSM to inspect the Neuron device, the container and the API health."""
    # AWS-RunShellScript executes under sh (dash on Ubuntu), so no bashisms.
    # The SSM agent's PATH omits the Neuron tools directory.
    command = (
        "PATH=$PATH:/opt/aws/neuron/bin; neuron-ls; "
        f"docker inspect --format '{{{{.State.Status}}}}' {CONTAINER}; "
        f"curl -fsS http://127.0.0.1:{SERVE_PORT}/health"
    )
    try:
        return f"### `{instance_id}` Neuron health\n\n```\n{await _ssm(instance_id, command)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get server logs", annotations=READ_ONLY)
async def get_server_logs(instance_id: str, tail: Annotated[int, Field(ge=1, le=5000)] = 200) -> str:
    """Read bounded container logs through SSM, including the per-request [req] timing lines."""
    try:
        output = await _ssm(instance_id, f"docker logs --tail {tail} {CONTAINER} 2>&1")
        return f"```\n{output}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get trn1 endpoint", annotations=READ_ONLY)
async def get_endpoint(instance_id: str) -> str:
    """Resolve the instance address and probe the OpenAI-compatible API.

    With an SSM-only security group nothing is reachable from outside; use
    `aws ssm start-session --document-name AWS-StartPortForwardingSession`.
    """
    try:
        response = await _call(_client("ec2").describe_instances, InstanceIds=[instance_id])
        item = response["Reservations"][0]["Instances"][0]
        host = item.get("PublicIpAddress") or item.get("PrivateIpAddress")
        url = f"http://{host}:{SERVE_PORT}"
        healthy = False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                healthy = (await client.get(f"{url}/health")).is_success
        except httpx.HTTPError:
            pass
        return f"Endpoint: `{url}/v1` — {'healthy' if healthy else 'not reachable from this host'}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Query model", annotations=READ_ONLY)
async def query_model(
    endpoint: str,
    prompt: str,
    max_tokens: Annotated[int, Field(ge=1, le=511)] = 256,
    model: str = MODEL_NAME,
) -> str:
    """Send a prompt to the trn1-hosted OpenAI-compatible endpoint.

    The E2B image's graph is traced at 512 total / 128 prompt tokens, so longer
    prompts are refused by the server rather than truncated silently.
    """
    try:
        client = AsyncOpenAI(base_url=endpoint.rstrip("/") + "/v1", api_key="not-required")
        response = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.0
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check Trainium quotas", annotations=READ_ONLY)
async def check_trn1_quotas() -> str:
    """List EC2 On-Demand and Spot Trainium quota values (vCPUs) in the active region."""
    try:
        quotas = await _call(_client("service-quotas").list_service_quotas, ServiceCode="ec2")
        matches = [
            q
            for q in quotas.get("Quotas", [])
            if "Trn" in q["QuotaName"] and ("On-Demand" in q["QuotaName"] or "Spot" in q["QuotaName"])
        ]
        return (
            "\n".join(f"- {q['QuotaName']}: {q['Value']} vCPUs" for q in matches) or "No Trainium quotas returned."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Help and configuration", annotations=READ_ONLY)
async def get_help() -> str:
    """Show the active AWS/Neuron configuration and operational prerequisites."""
    return (
        "### AWS Trainium (trn1) DevOps agent\n\n"
        f"- Region: `{AWS_REGION}`\n- Instance type: `{INSTANCE_TYPE}`\n"
        f"- Image: `{OPTB_IMAGE}`\n- Checkpoint (native engine, by hand): `{CHECKPOINT}`\n- Port: `{SERVE_PORT}`\n- Tag: `ManagedBy={MANAGED_BY}`\n\n"
        "Serves the prebuilt Gemma-4 Neuron images compiled for inf2; they run unchanged on "
        "trn1's NeuronCore-v2. There is no vLLM path: no vLLM-Neuron release supports both "
        "Trn1 and Gemma-4.\n\n"
        "The instance profile needs `AmazonSSMManagedInstanceCore`. The caller needs EC2, "
        "SSM Run Command and Service Quotas permissions. Launches default to spot; Trn spot "
        "quota starts at 0, so pass spot=False until an increase is approved."
    )


if __name__ == "__main__":
    mcp.run()
