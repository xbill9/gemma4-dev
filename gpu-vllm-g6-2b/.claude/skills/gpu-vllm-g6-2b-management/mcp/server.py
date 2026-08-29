"""EC2 G6 (x86_64 + NVIDIA L4, Ada SM 8.9) lifecycle and inference MCP server.

Like the Inf2 sibling, this uses boto3 rather than shelling out to the AWS CLI so
it works with profiles, environment credentials, IAM roles, and SSO-backed
credential processes, and uses Systems Manager Run Command for remote
administration — no inbound SSH rule or private key.

FORKED FROM `gpu-vllm-g5g-2b` 2026-08-28, and the fork REMOVES that rig's reason
for existing rather than porting it.

The sibling exists because G5g pairs a Graviton2 (aarch64) host with a T4G GPU
(Turing, SM 7.5), and every prebuilt CUDA artifact covers one of those two axes
but not both: `vllm/vllm-openai` publishes an arm64 manifest compiled for
`8.0 8.7 8.9 9.0 10.0 11.0 12.0`, while the amd64 manifest of the same tag is the
one that carries 7.5. The single arch it needed fell in the gap between the two
published images, with no `+PTX` and so no JIT fallback. It pays for that with a
~67-minute from-source build, a CUDA toolkit the DLAMI does not ship, a Rust
toolchain, and an unlanded patch to Triton's attention kernel. See
`gpu-vllm-g5g-2b/docs/turing-aarch64-gap.md`.

**G6 is x86_64 and SM 8.9, so BOTH axes are covered by the stock image.** 8.9 is
in the arch list quoted above and amd64 is the primary manifest. There is
therefore no build, no toolkit, no Rust and no image to bake here — this rig runs
`vllm/vllm-openai` as published. That deletes most of the sibling's code.

The Triton shared-memory ceiling is the other half of the sibling's problem.
Gemma 4's heterogeneous head dims (sliding 256, global 512) force a Triton
attention path whose 512-wide tile wants ~96 KiB of shared memory per block,
against Turing's hard 64 KiB — which is what that unlanded patch works around.
**Ada raises the per-block limit to ~99 KiB**, so the tile is expected to fit
unpatched. NOT VERIFIED ON HARDWARE: this rig has served nothing. It is the first
thing to check, and `ATTENTION_BACKEND` below is the knob.
"""

import asyncio
import base64
import logging
import os
import time

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
RIG_NAME = "gpu-vllm-g6-2b"

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
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "g6.xlarge")
SERVICE_NAME = os.getenv("SERVICE_NAME", "vllm-g6")
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))

# Ada HAS a native bf16 datapath, and fp8 as well. Both were unavailable on the
# T4G sibling, where bfloat16 was refused and fp8 KV did not exist -- so these
# became real tuning knobs at the fork rather than fixed values.
#
# bfloat16 is the default because the checkpoint ships bf16: float16 would make
# vLLM convert every weight on load, and the JAX rig on this exact silicon
# MEASURED that mismatch costing 54% of decode on Turing. Matching the checkpoint
# is the cheap default; see `gpu-jax-g6-2b/CLAUDE.md`, "The dtype tax is gone".
DTYPE = os.getenv("DTYPE", "bfloat16")
# `auto` follows the compute dtype. fp8 KV is newly REACHABLE here and is NOT
# enabled: KV is not the binding constraint for this model (18 KiB/token, so the
# whole cache at 16K context is ~288 MiB against 23 GB of device memory), and
# nothing here has measured its accuracy cost. Turn it on with a measurement, not
# on the grounds that the part supports it.
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto")
# THE SIBLING PINS XFORMERS TO AVOID A TURING-ONLY FAILURE, AND THAT REASON IS
# GONE. Gemma 4's global head_dim of 512 forces a Triton attention path whose
# tile wants ~96 KiB of shared memory per block; Turing caps a block at 64 KiB,
# so the T4G rig carries an unlanded patch shrinking the tile. Ada allows ~99 KiB,
# which should clear it unpatched.
#
# Left EMPTY here on purpose, which means "let vLLM choose". Pinning a backend is
# how the sibling ended up carrying a patch, and this rig has measured nothing --
# an unpinned default that vLLM selects for the actual part is the honest starting
# point. Set VLLM_ATTENTION_BACKEND to pin one once there is a measurement saying
# which. UNVERIFIED ON HARDWARE.
ATTENTION_BACKEND = os.getenv("VLLM_ATTENTION_BACKEND", "")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.90")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "8"))

# THE STOCK IMAGE IS THE IMAGE. On x86_64 + SM 8.9 both axes are covered by the
# published amd64 manifest, so there is nothing to build and VLLM_IMAGE is the
# stock tag rather than a locally-built one. The sibling's VLLM_STOCK_IMAGE /
# VLLM_IMAGE split existed only to distinguish "what you pull" from "what you
# spent 67 minutes compiling"; here they are the same thing and the split is
# deliberately collapsed to a single name.
#
# TORCH_CUDA_ARCH_LIST and VLLM_REF are GONE for the same reason -- nothing on
# this rig compiles, so an arch list and a source ref would be inert settings
# that look meaningful. Do not reintroduce them without a build to feed.
# THE TAG IS NOT THE SIBLING'S STOCK TAG. That rig's VLLM_STOCK_IMAGE was
# v0.27.1, used ONLY to reproduce the SM 7.5 failure -- it never served from it,
# and built its real image from VLLM_REF=v0.27.2rc0. v0.27.1 is BELOW the
# measured floor and is an easy tag to inherit by mistake at a fork.
#
# MEASURED on the sibling: v0.26.0 dies with AmbiguousGlobalPerLayerAttributeError
# against current transformers, because Gemma 4's head_dim is per-layer. The
# per_layer_config handling that fixes it landed in v0.27.2rc0. DO NOT PIN BELOW
# THIS -- the constraint is the MODEL, not the chip, so it carries unchanged.
VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:v0.27.2rc0")

# AWS publishes the x86_64 GPU DLAMI as a public SSM parameter. Prefer it: it is
# single-valued and authoritative, where a describe-images name filter is a fuzzy
# match that can select the wrong image and still boot.
#
# BASE, not PyTorch: this rig serves from a docker image that carries its own
# CUDA and torch, so a PyTorch DLAMI is GBs of image whose entire content is
# unused. The DLAMI only has to supply the NVIDIA driver and docker.
#
# `/latest/` in a DLAMI path is only the newest build WITHIN one PyTorch-and-
# Ubuntu line, and AWS eventually stops rebuilding a line -- the sibling pinned
# `pytorch-2.7-ubuntu-22.04`, which froze at a 2026-05-02 image and reads as
# "track latest" while being a pin to a dead line. This path is the same one
# `gpu-jax-g6-2b` VERIFIED ON HARDWARE 2026-08-28 (driver 595.91.07).
DLAMI_SSM_PARAMETER = os.getenv(
    "DLAMI_SSM_PARAMETER",
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-26.04/latest/ami-id",
)
# Fallback only. It must NOT require "ARM64 AMI" contiguously the way the
# sibling's pattern did -- the base images are named "Deep Learning Base OSS
# Nvidia Driver GPU AMI (Ubuntu 26.04)". Changing the SSM path without changing
# this filter in the same commit is a revert that reports success: the fallback
# quietly resolves the old image.
DLAMI_NAME = os.getenv("DLAMI_NAME", "Deep Learning Base OSS Nvidia Driver GPU*Ubuntu*")

MANAGED_BY = RIG_NAME

# G6 topology: (GPUs, host RAM GiB). Verified against the AWS G6 product page.
#
# NOTE two traps relative to the G5g table this replaces. Host RAM DOUBLED at
# every matching suffix (g6.xlarge is 16 GiB where g5g.xlarge was 8), so a
# host-RAM verdict does not transfer by size name. And g6.16xlarge is
# SINGLE-GPU despite the suffix, where g5g.16xlarge had two -- GPU count is not
# monotonic in the size here, so never infer it from the name.
_G6_SIZES = {
    "g6.xlarge": (1, 16),
    "g6.2xlarge": (1, 32),
    "g6.4xlarge": (1, 64),
    "g6.8xlarge": (1, 128),
    "g6.12xlarge": (4, 192),
    "g6.16xlarge": (1, 256),
    "g6.24xlarge": (4, 384),
    "g6.48xlarge": (8, 768),
}

# Host RAM below this needs a swapfile before the model will load. MEASURED
# 2026-08-13 on g5g.xlarge (7,757 MiB usable), i.e. ON THE SIBLING: loading E2B
# fails with
#
#   RuntimeError: unable to mmap 10246621918 bytes from model.safetensors:
#   Cannot allocate memory (12)
#
# and the container crash-loops on it. The failure is the *mapping*, not
# residency -- the kernel declines to map a 10.2 GB file against 7.5 GiB of RAM
# and no swap, before a single page is faulted in. Adding swap fixed it outright.
#
# The mechanism is a property of the checkpoint and the loader, not the GPU, so
# it carries. THE SIZE NAMES DO NOT: every G6 size has at least 16 GiB of host
# RAM, twice its G5g namesake, so NO G6 SIZE TRIPS THIS GATE and the swap block
# never renders. It is kept rather than deleted because the threshold is a claim
# about the checkpoint (~10.2 GB to map), and a larger checkpoint on a small host
# would need it again.
#
# Consequence worth stating plainly: the swap path is now UNTESTED CODE on this
# rig. The sibling learned that lesson expensively -- a `mkswap -q` busybox flag
# that util-linux rejects sat latent in exactly this block for as long as only
# one unlaunched size rendered it, then killed cloud-init before the install log
# existed.
_SWAP_BELOW_HOST_RAM_GB = 16
_SWAP_GB = 16

# Root volume. The image pull lands here (vllm/vllm-openai is multiple GB), the
# checkpoint downloads here (10.2 GB), and the loader reads it back.
#
# Throughput is set EXPLICITLY because gp3's default is 125 MiB/s. PORTED FROM
# `gpu-jax-g6-2b`, where it is MEASURED rather than assumed: two unrelated load
# stages both landed on ~125 MB/s (read_shards 73.5s / ~139 MB/s, download 87.7s
# / ~116 MB/s), which is the signature of a volume ceiling rather than CPU or
# network -- and raising it took read_shards 73.5s -> 24.7s, a clean 3.0x on the
# same read. That rig serves from a pip install and this one from a docker image,
# so the pull is an ADDITIONAL multi-GB read here and the ceiling should bind at
# least as hard. UNMEASURED ON THIS RIG.
#
# 500 MiB/s is ~4x baseline and still under g6.xlarge's own EBS ceiling
# ("up to" 4.75 Gbps ~= 593 MB/s), so smaller sizes stay instance-bound rather
# than volume-bound. gp3 also requires throughput <= IOPS * 0.25 MiB/s, which
# 6000 IOPS satisfies -- and that rule is enforced at run-instances time, so
# violating it fails a LAUNCH rather than merely slowing a disk.
#
# 200 -> 100 GB: the sibling's get_deployment_config PRINTED VolumeSize=200 while
# its create tool LAUNCHED 100. A copy-pasteable repro command that provisions a
# different volume from the tool it documents is how a manual reproduction fails
# to reproduce. Both now render from these constants.
ROOT_VOLUME_GB = int(os.getenv("ROOT_VOLUME_GB", "100"))
ROOT_VOLUME_THROUGHPUT_MBPS = int(os.getenv("ROOT_VOLUME_THROUGHPUT_MBPS", "500"))
ROOT_VOLUME_IOPS = int(os.getenv("ROOT_VOLUME_IOPS", "6000"))


def _session():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed; run `python3 -m pip install -r requirements.txt`")
    kwargs = {"region_name": AWS_REGION}
    if AWS_PROFILE:
        kwargs["profile_name"] = AWS_PROFILE
    return boto3.Session(**kwargs)


def _client(service: str):
    return _session().client(service)


def _is_g6(instance_type: str) -> bool:
    return instance_type in _G6_SIZES


def _gpu_count(instance_type: str) -> int:
    return _G6_SIZES.get(instance_type, (0, 0))[0]


def _host_memory_gb(instance_type: str) -> int:
    return _G6_SIZES.get(instance_type, (0, 0))[1]


def _needs_swap(instance_type: str) -> bool:
    """True when host RAM is too small to mmap the checkpoint without swap."""
    return 0 < _host_memory_gb(instance_type) < _SWAP_BELOW_HOST_RAM_GB


def _validate_instance_type(instance_type: str) -> None:
    """Only the size list is enforced. Small hosts are supported, not rejected --
    `_user_data` provisions a swapfile for them (see `_SWAP_BELOW_HOST_RAM_GB`)."""
    if not _is_g6(instance_type):
        raise ValueError(f"instance_type must be one of {', '.join(sorted(_G6_SIZES))}")


def _tensor_parallel_size(instance_type: str) -> int:
    return _gpu_count(instance_type)


def _vcpu_count(instance_type: str) -> int:
    """vCPUs for a G6 size.

    Deliberately NOT derived from host RAM. The sibling computed `RAM // 2`,
    which was right for G5g (2 GiB per vCPU) and is wrong for G6 (4 GiB per
    vCPU) -- it would report double. G6 vCPUs are 4x the xlarge multiplier, and
    the metal-less family tops out at 192.
    """
    return _G6_SIZES.get(instance_type, (0, 0))[1] // 4


async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)


def _serve_flags(model: str, instance_type: str) -> str:
    """vLLM flags for Ada (SM 8.9).

    NOTE this rig IS an L4, so the five `gpu-vllm-l4-*` artifact rigs are the same
    silicon -- but their provenance is the weakest in the tree and their flag sets
    were never validated here. Same chip is not same measurement.
    """
    return (
        f"--model {model} --host 0.0.0.0 --port 8000 "
        f"--dtype {DTYPE} --kv-cache-dtype {KV_CACHE_DTYPE} "
        f"--tensor-parallel-size {_tensor_parallel_size(instance_type)} "
        f"--gpu-memory-utilization {GPU_MEMORY_UTILIZATION} "
        f"--max-model-len {MAX_MODEL_LEN} --max-num-seqs {MAX_NUM_SEQS}"
    )


def _user_data(model: str, instance_type: str) -> str:
    """Render idempotent cloud-init that pulls the stock image and serves.

    THERE IS ONLY ONE PATH HERE, and that is the fork's headline. The sibling
    carries two — ``build``, which compiles vLLM for SM 7.5 on the instance over
    ~67 minutes, and ``stock``, which exists purely so the resulting "no kernel
    image is available for execution on the device" failure can be reproduced on
    hardware rather than asserted.

    On SM 8.9 / x86_64 the published image already carries the arch, so ``build``
    has nothing to do and ``stock`` has nothing to fail at. Both were removed
    rather than kept as dead options: a ``serving=`` parameter whose only value is
    the default is a knob that suggests a choice nobody has.

    Still verify rather than assume — `verify_gpu_arch` runs a real matmul on the
    device and answers in minutes. On this rig it should pass where the sibling's
    fails, and that IS the fork's premise. UNVERIFIED ON HARDWARE.
    """
    _validate_instance_type(instance_type)

    flags = _serve_flags(model, instance_type)

    swap = ""
    if _needs_swap(instance_type):
        # Dead code on every current G6 size (all have >= 16 GiB host RAM); kept
        # because the threshold is a claim about the checkpoint, not the host.
        # NOTE `mkswap` takes no `-q` here: that is a busybox flag which
        # util-linux rejects with `invalid option -- 'q'`, and under `set -e` it
        # killed cloud-init on the sibling BEFORE anything logged. Do not re-add.
        swap = f"""if ! swapon --show --noheadings 2>/dev/null | grep -q /swapfile; then
  fallocate -l {_SWAP_GB}G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
"""

    # The attention backend is only exported when explicitly pinned. Passing an
    # empty VLLM_ATTENTION_BACKEND is NOT the same as not setting it -- vLLM sees
    # the variable and can treat "" as a selection rather than falling through to
    # its own dispatch, which is exactly the silent-misconfiguration shape this
    # tree keeps getting bitten by.
    backend_env = f'  -e VLLM_ATTENTION_BACKEND={ATTENTION_BACKEND} \\\n' if ATTENTION_BACKEND else ""

    return f"""#!/usr/bin/env bash
set -euxo pipefail
{swap}systemctl enable --now docker
mkdir -p /opt/{SERVICE_NAME}

# Stage markers, so a stalled launch is attributable rather than a mystery. The
# sibling's build was the longest phase of a deploy and the only untimed one, and
# a spot reclamation mid-install left no record of which step was running.
_T0=$(date +%s)
stage() {{ echo "[stage] $1 +$(( $(date +%s) - _T0 ))s"; }}

stage image-pull-start
docker pull {VLLM_IMAGE}
stage image-pull-done

cat >/opt/{SERVICE_NAME}/start.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
# set +x around the secret: this script can run under `set -x`, and bash traces
# assignments WITH THEIR VALUES. The token must never reach the console log or
# instance metadata, which is readable by anything on the box.
set +x
HF_TOKEN=$(aws secretsmanager get-secret-value --region {AWS_REGION} --secret-id {HF_SECRET_ID} --query SecretString --output text 2>/dev/null || true)
docker rm -f {SERVICE_NAME} 2>/dev/null || true
docker run -d --name {SERVICE_NAME} --restart unless-stopped --ipc=host \\
  --gpus all -e HF_TOKEN="$HF_TOKEN" \\
{backend_env}  -p {VLLM_PORT}:8000 \\
  {VLLM_IMAGE} \\
  {flags}
SCRIPT
chmod 700 /opt/{SERVICE_NAME}/start.sh
/opt/{SERVICE_NAME}/start.sh
stage serving-started
touch /opt/{SERVICE_NAME}/INSTALL_DONE
stage INSTALL_COMPLETE
"""


async def _resolve_ami(ec2=None) -> str:
    """Resolve the x86_64 **GPU** DLAMI for this region.

    Two things have to hold and they are separate: the image must be **x86_64**
    (the architecture flipped at the G5g fork — an arm64 DLAMI cannot boot on a
    G6 at all), and it must carry the NVIDIA driver.

    The SSM public parameter pins both. describe-images is the fallback, and the
    two must be changed together: a name filter that no longer matches the SSM
    path's image family will silently resolve a DIFFERENT image and report
    success. Never hardcode an AMI id — resolve it at launch.
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
            {"Name": "architecture", "Values": ["x86_64"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(result.get("Images", []), key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError(
            f"No x86_64 GPU DLAMI in {AWS_REGION} via SSM ({DLAMI_SSM_PARAMETER}) "
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


@mcp.tool(title="Generate G6 deployment configuration", annotations=READ_ONLY)
async def get_deployment_config(
    model_name: str = MODEL_NAME,
    instance_type: str = INSTANCE_TYPE,
    subnet_id: str = "<subnet-id>",
    security_group_id: str = "<security-group-id>",
    iam_instance_profile: str = "g6-vllm-instance-profile",
    spot: bool = True,
) -> str:
    """Return cloud-init and an AWS CLI launch command without changing AWS.

    The rendered command provisions the SAME root volume the create tool does --
    both read ROOT_VOLUME_*. On the sibling they disagreed (200 printed, 100
    launched), which makes a manual reproduction quietly not reproduce.
    """
    try:
        _validate_instance_type(instance_type)
        script = _user_data(model_name, instance_type)
        encoded = base64.b64encode(script.encode()).decode()
        market = (
            "--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' "
            if spot
            else ""
        )
        note = (
            f"Runs the published `{VLLM_IMAGE}` unchanged — SM 8.9 is in its arch list and "
            "amd64 is its primary manifest, so there is nothing to build here. "
            "Run verify_gpu_arch first anyway: it settles the question in minutes."
        )
        return (
            f"### EC2 G6 deployment ({instance_type}, {_gpu_count(instance_type)}x L4)\n\n```bash\n"
            f"# x86_64 *GPU* DLAMI. The SSM parameter pins both the architecture and the\n"
            f"# NVIDIA driver; never hardcode an AMI id.\n"
            f"AMI_ID=$(aws ssm get-parameter --region {AWS_REGION} "
            f"--name {DLAMI_SSM_PARAMETER} "
            "--query 'Parameter.Value' --output text)\n"
            f'aws ec2 run-instances --region {AWS_REGION} --image-id "$AMI_ID" '
            f"--instance-type {instance_type} --subnet-id {subnet_id} "
            f"--security-group-ids {security_group_id} "
            f"--iam-instance-profile Name={iam_instance_profile} "
            f"{market}"
            f"--block-device-mappings 'DeviceName=/dev/sda1,Ebs={{VolumeSize={ROOT_VOLUME_GB},"
            f"VolumeType=gp3,Throughput={ROOT_VOLUME_THROUGHPUT_MBPS},Iops={ROOT_VOLUME_IOPS},"
            f"DeleteOnTermination=true}}' "
            f"--user-data '{encoded}' --tag-specifications "
            f"'ResourceType=instance,Tags=[{{Key=Name,Value={SERVICE_NAME}}},"
            f"{{Key=ManagedBy,Value={MANAGED_BY}}}]'\n```\n\n"
            f"{note}\n\n"
            f"Serving flags: `{_serve_flags(model_name, instance_type)}`\n"
            f"dtype is `{DTYPE}` — Ada has a native bf16 datapath, unlike the T4G sibling, "
            f"and it matches the checkpoint so vLLM converts nothing on load."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Create G6 instance", annotations=WRITE)
async def create_g6_instance(
    subnet_id: str,
    security_group_id: str,
    iam_instance_profile: str,
    name: str = SERVICE_NAME,
    model_name: str = MODEL_NAME,
    instance_type: str = INSTANCE_TYPE,
    spot: bool = True,
) -> str:
    """Launch one tagged G6 instance using the latest regional x86_64 GPU DLAMI.

    Cloud-init pulls the stock vLLM image and serves. There is no build step and
    no `serving` mode: SM 8.9 is covered by the published image, so the sibling's
    build/stock choice does not exist here.

    Spot is the default; pass spot=False for on-demand. Surface capacity errors
    rather than retrying silently -- G-family spot in us-east-1 is genuinely
    scarce, and `aws ec2 get-spot-placement-scores` picks a size and AZ far more
    cheaply than launching until one succeeds.
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
            "UserData": _user_data(model_name, instance_type),
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": ROOT_VOLUME_GB,
                        "VolumeType": "gp3",
                        "Throughput": ROOT_VOLUME_THROUGHPUT_MBPS,
                        "Iops": ROOT_VOLUME_IOPS,
                        "DeleteOnTermination": True,
                    },
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
            f"Pulling `{VLLM_IMAGE}` and serving — no build. "
            "Follow with get_install_progress, then verify_gpu_arch and verify_model_health."
        )
        return (
            f"✅ Launching `{instance_id}` ({instance_type}, {market}, "
            f"{_gpu_count(instance_type)}x L4) in `{AWS_REGION}`.\n"
            f"AMI: `{args['ImageId']}`\n{tail}"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List managed G6 instances", annotations=READ_ONLY)
async def list_g6_instances() -> str:
    """List instances tagged ManagedBy=gpu-vllm-g6-2b."""
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


@mcp.tool(title="Start G6 instance", annotations=WRITE)
async def start_g6_instance(instance_id: str) -> str:
    """Start a stopped managed instance."""
    try:
        await _call(_client("ec2").start_instances, InstanceIds=[instance_id])
        return f"✅ Starting `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Stop G6 instance", annotations=DESTRUCTIVE)
async def stop_g6_instance(instance_id: str) -> str:
    """Stop a running managed instance. One-time spot instances cannot be
    stopped, only terminated."""
    try:
        await _call(_client("ec2").stop_instances, InstanceIds=[instance_id])
        return f"🛑 Stopping `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Terminate G6 instance", annotations=DESTRUCTIVE)
async def terminate_g6_instance(instance_id: str) -> str:
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

    Defaults to the published image this rig serves from. On SM 8.9 the arch is
    expected to be PRESENT — the opposite of the sibling, where this tool exists
    to confirm an absence. THAT INVERSION IS THE FORK'S PREMISE AND IS UNVERIFIED,
    so run it before believing anything else here.

    It probes with bfloat16, not float16: Ada has a native bf16 datapath and
    `DTYPE` defaults to bfloat16, so a float16 probe would pass without touching
    the datapath the rig actually serves on.
    """
    target = image or VLLM_IMAGE
    probe = (
        "import torch;"
        "print('device:', torch.cuda.get_device_name(0));"
        "print('capability:', torch.cuda.get_device_capability(0));"
        "print('torch arch list:', torch.cuda.get_arch_list());"
        "x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16);"
        "print('matmul ok:', float((x @ x).sum()) == float((x @ x).sum()))"
    )
    command = (
        f"nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true; "
        f'docker run --rm --gpus all --entrypoint python3 {target} -c "{probe}" 2>&1 || true'
    )
    try:
        output = await _ssm(instance_id, command, timeout=600)
        verdict = (
            "\n\n❌ No SM 8.9 in the arch list — this image cannot run on an L4. That "
            "would contradict the published amd64 manifest and is the one result that "
            "would invalidate this rig's premise; check the image tag before concluding it."
            if "8.9" not in output and "arch list" in output
            else "\n\n✅ SM 8.9 is in the arch list and a real bfloat16 matmul executed."
            if "matmul ok: True" in output
            else ""
        )
        if "no kernel image is available" in output:
            verdict = (
                "\n\n❌ `no kernel image is available for execution on the device` — on SM 8.9 "
                "this should NOT happen with the published image. Suspect a wrong tag, an "
                "arm64 manifest pulled by mistake, or a non-L4 GPU."
            )
        return f"### GPU arch probe on `{instance_id}` ({target})\n\n```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get bootstrap progress", annotations=READ_ONLY)
async def get_install_progress(instance_id: str, tail: int = 40) -> str:
    """Tail cloud-init and the image pull. Replaces the sibling's build tracker.

    There is no from-source build here, so this is an image pull and a container
    start — minutes, not the sibling's ~67 minutes.

    PORTED FROM `gpu-jax-g6-2b`, and the ported part is the cloud-init verdict
    rather than the tail. Cloud-init can die BEFORE it writes anything this tool
    would normally read, and the naive rendering of that is `IN PROGRESS` plus
    `no log yet`, forever — which is also exactly what a healthy slow launch
    looks like. A dead bootstrap and a running one must not share a rendering.

    That is not hypothetical: on the JAX sibling a `mkswap -q` busybox flag that
    util-linux rejects failed under `set -e` in the swap block, which renders
    FIRST, so cloud-init died before the install log existed and the instance sat
    there looking busy. Cost a launch rather than a minute.
    """
    try:
        tail = max(1, min(tail, 5000))
        command = (
            f"test -f /opt/{SERVICE_NAME}/INSTALL_DONE && echo 'INSTALL COMPLETE' "
            "|| echo 'INSTALL IN PROGRESS'; "
            "echo '--- cloud-init ---'; "
            "cloud-init status --long 2>&1 || echo 'cloud-init status unavailable'; "
            "echo '--- stages ---'; "
            "grep -F '[stage]' /var/log/cloud-init-output.log 2>/dev/null "
            "|| echo 'no stage markers yet'; "
            "echo '--- cloud-init output ---'; "
            f"tail -n {tail} /var/log/cloud-init-output.log 2>/dev/null "
            "|| echo 'NO CLOUD-INIT OUTPUT LOG'"
        )
        output = await _ssm(instance_id, command)

        # Ordered most-specific first. "status: error" is cloud-init's own
        # verdict and outranks the absence of a log, which is only a symptom.
        if "INSTALL COMPLETE" in output:
            verdict = (
                "\n\n✅ Image pulled and the container started. Next: verify_gpu_arch, "
                "then verify_model_health. Note the container starting is NOT the model "
                "being ready — vLLM still has to download and load the checkpoint."
            )
        elif "status: error" in output:
            verdict = (
                "\n\n❌ cloud-init FAILED — the bootstrap died, nothing is pulling and "
                "nothing will. Read the output above for the failing command; relaunching "
                "reproduces it. This is NOT a slow launch."
            )
        elif "NO CLOUD-INIT OUTPUT LOG" in output and "status: done" in output:
            verdict = (
                "\n\n❌ cloud-init reports done but wrote no output log. The bootstrap "
                "exited before doing its work. Nothing is running."
            )
        elif "NO CLOUD-INIT OUTPUT LOG" in output:
            verdict = (
                "\n\n⏳ cloud-init has not written output yet. Normal for the first minute "
                "or two after launch; if it persists, the bootstrap is stuck early."
            )
        else:
            verdict = "\n\n⏳ Pulling the image and starting the container. No build here."
        return f"```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get vLLM logs", annotations=READ_ONLY)
async def get_vllm_logs(instance_id: str, tail: int = 100) -> str:
    """Tail the vLLM container log."""
    try:
        tail = max(1, min(tail, 5000))
        return f"```\n{await _ssm(instance_id, f'docker logs --tail {tail} {SERVICE_NAME} 2>&1')}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get G6 endpoint", annotations=READ_ONLY)
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


@mcp.tool(title="Check G6 quotas", annotations=READ_ONLY)
async def check_g6_quotas() -> str:
    """Report the On-Demand and Spot G instance vCPU quotas for the region.

    G6 draws on the same 'Running On-Demand G and VT instances' quota as other
    G-family types, counted in vCPUs — a g6.xlarge needs 4 and a g6.2xlarge 8.

    Quota is not capacity. MEASURED 2026-08-28 on the JAX sibling: g6.xlarge spot
    was exhausted in all five us-east-1 AZs with quota to spare. Check
    `aws ec2 get-spot-placement-scores` before launching in a loop.
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
        # G5g had 2 GiB of RAM per vCPU so `RAM // 2` was its vCPU count. G6 has
        # 4 GiB per vCPU, so that shortcut silently DOUBLES every figure here.
        # Read it from the table instead of re-deriving it.
        lines.append(f"`{INSTANCE_TYPE}` needs {_vcpu_count(INSTANCE_TYPE)} vCPUs.")
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Help and configuration", annotations=READ_ONLY)
async def get_help() -> str:
    """Show this rig's resolved configuration and the constraints that shape it."""
    return f"""### {RIG_NAME}

Serving `{MODEL_NAME}` with vLLM on **EC2 G6** — x86_64 host, NVIDIA **L4** GPU
(Ada, SM 8.9, 24 GB nominal / 23034 MiB measured by the JAX sibling).

| Setting | Value |
| --- | --- |
| Region | `{AWS_REGION}` |
| Instance type | `{INSTANCE_TYPE}` ({_gpu_count(INSTANCE_TYPE)}x L4, {_host_memory_gb(INSTANCE_TYPE)} GiB RAM, {_vcpu_count(INSTANCE_TYPE)} vCPU) |
| Tensor parallel | `{_tensor_parallel_size(INSTANCE_TYPE)}` |
| dtype | `{DTYPE}` |
| KV cache dtype | `{KV_CACHE_DTYPE}` |
| Attention backend | `{ATTENTION_BACKEND or "(unpinned — vLLM chooses)"}` |
| Image | `{VLLM_IMAGE}` (published; nothing is built here) |
| Root volume | {ROOT_VOLUME_GB} GB gp3 @ {ROOT_VOLUME_THROUGHPUT_MBPS} MiB/s, {ROOT_VOLUME_IOPS} IOPS |
| Managed-by tag | `{MANAGED_BY}` |

**THIS RIG HAS SERVED NOTHING.** Forked from `gpu-vllm-g5g-2b` 2026-08-28. Every
claim below is arithmetic or inherited from a sibling; none of it is measured
here. `benchmarks/` is deliberately empty.

**The constraint that defined the sibling is GONE, and that is the point.** G5g
needed aarch64 *and* SM 7.5, and the published `vllm/vllm-openai` arm64 manifest
is compiled for `8.0 8.7 8.9 9.0 10.0 11.0 12.0` while the amd64 manifest of the
same tag carries 7.5 — so its one arch fell between the two published images,
with no `+PTX` to JIT from. It pays ~67 minutes of from-source build for that.
**G6 is x86_64 and SM 8.9: both axes are in the published amd64 image, so there
is no build, no CUDA toolkit, no Rust and no Triton patch here.**

**Ada HAS bf16 and fp8, unlike the T4G.** `--dtype bfloat16` is the default here
and matches the checkpoint, so vLLM converts nothing on load. fp8 KV is now
reachable and is NOT enabled — KV is ~18 KiB/token, so the whole cache at 16K is
~288 MiB against 23 GB, and it has no measured accuracy cost here.

**The open question is attention.** Gemma 4's global head_dim of 512 forces a
Triton path whose tile wants ~96 KiB of shared memory per block against Turing's
64 KiB, which is what the sibling's unlanded patch works around. Ada allows
~99 KiB, so it should fit unpatched — UNVERIFIED. `ATTENTION_BACKEND` is left
unpinned so vLLM dispatches for the real part.

Start with `verify_gpu_arch`. It settles in minutes what the sibling spends
hours discovering, and on this rig it is expected to PASS where that one fails —
which is the fork's whole premise and has not been checked.
"""


if __name__ == "__main__":
    mcp.run()
