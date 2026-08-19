"""EC2 G5g (Graviton2 + NVIDIA T4G) lifecycle and inference MCP server, JAX path.

Like the Inf2 sibling, this uses boto3 rather than shelling out to the AWS CLI so
it works with profiles, environment credentials, IAM roles, and SSO-backed
credential processes, and uses Systems Manager Run Command for remote
administration — no inbound SSH rule or private key.

Why this rig exists, next to the vLLM one on identical hardware:

G5g pairs a Graviton2 (aarch64) host with a T4G GPU (Turing, SM 7.5), and the
vLLM path needs both axes at once from artifacts that each cover only one. It
gets there, but only via a ~67-minute from-source build, a CUDA toolkit, Rust,
and an unlanded patch to Triton's attention kernel — because Gemma 4's
heterogeneous head dims force TRITON_ATTN, whose 512-wide global-attention tile
wants ~96 KiB of shared memory against Turing's 64 KiB ceiling.

The JAX path sidesteps all four:

  * jaxlib, jax-cuda12-plugin and jax-cuda12-pjrt publish aarch64 wheels, and
    every CUDA dependency (cublas, cudnn, cuda-runtime, cusolver) does too — so
    pip supplies CUDA and the DLAMI only has to supply the driver. No build.
  * The plugin's arch tables carry sm_75 and its floor is SM 6.0.
  * Attention here is ordinary XLA, not a hand-tiled Triton kernel, so there is
    no per-block shared-memory ceiling to hit and no patch to carry.

What it does NOT sidestep is the same ceiling in a different place: the fused
W4A16 Pallas kernel is tiled for TPU VMEM and needs 550 KiB - 1.1 MiB per block,
so it cannot run on Turing either. This rig therefore serves the dense reference
checkpoint at float16. See docs/turing-aarch64-gap.md and tpu.env.

NOTHING BELOW HAS BEEN MEASURED ON HARDWARE YET. The wheel/arch facts above were
verified against PyPI and by inspecting the plugin binary; that is not the same
as a served token.
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
RIG_NAME = "gpu-jax-g5g-2b"

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
SERVICE_NAME = os.getenv("SERVICE_NAME", "jax-g5g")
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")
JAX_PORT = int(os.getenv("JAX_PORT", "8000"))

# Turing has no bf16 and no fp8 datapath. DTYPE is an override here, not the
# decision: ports/gemma4/jax_e_model.py reads the live device's compute
# capability and picks float16 below SM 8.0. tpu.env explains both.
DTYPE = os.getenv("DTYPE", "float16")
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto")
QUANT_MODE = os.getenv("QUANT_MODE", "fp16")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "1"))

# JAX preallocates this fraction of device memory at first use. NOT the same
# knob as vLLM's --gpu-memory-utilization: there is no engine-managed KV pool
# here, the KV cache is ordinary JAX arrays allocated inside this fraction.
XLA_PYTHON_CLIENT_MEM_FRACTION = os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

# jax 0.11 requires Python >= 3.12 and Ubuntu 22.04 (the DLAMI base) ships 3.10,
# so the bootstrap installs a 3.12 interpreter rather than using the system one.
# Installing into the DLAMI's own PyTorch environment is deliberately avoided —
# it ships its own CUDA libraries and jax[cuda12] brings its own.
JAX_PIP_SPEC = os.getenv("JAX_PIP_SPEC", "jax[cuda12]")
JAX_PYTHON_VERSION = os.getenv("JAX_PYTHON_VERSION", "3.12")
JAX_COMPILATION_CACHE_DIR = os.getenv("JAX_COMPILATION_CACHE_DIR", "/opt/jax-cache")

# What cloud-init installs on the instance, beyond JAX_PIP_SPEC. Kept here rather
# than inline in the bootstrap so requirements-serving.txt can mirror one list;
# tests assert the two agree, because a drifted pair is invisible until a serve.
_SERVING_REQUIREMENTS = (
    "fastapi", "uvicorn", "pydantic", "transformers",
    "safetensors", "huggingface_hub", "numpy",
    # jinja2 is NOT optional here despite not being imported anywhere in this
    # rig: transformers renders the chat template through it, and every serving
    # path goes through apply_chat_template. Without it /health returns 200 and
    # every /v1/chat/completions returns 500 -- measured 2026-08-19. transformers
    # does not pull it in, and it memoizes the availability check at import, so
    # installing it late needs a service restart.
    "jinja2",
)

# Where the serving payload lands on the instance.
APP_DIR = "/opt/jax-g5g"
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

# Host RAM below this needs a swapfile before the model will load. Measured
# 2026-08-13 on g5g.xlarge (7,757 MiB usable): loading E2B fails with
#
#   RuntimeError: unable to mmap 10246621918 bytes from model.safetensors:
#   Cannot allocate memory (12)
#
# and systemd crash-loops on it. The failure is the *mapping*, not residency --
# the kernel declines to map a 10.2 GB file against 7.5 GiB of RAM and no swap,
# before a single page is faulted in. Adding swap fixes it outright: 16 GiB of
# swap took the same instance to a healthy endpoint at 44.24 tok/s, which is
# indistinguishable from the 4xlarge's 43.1 (decode is GPU-bandwidth-bound, so
# vCPU count barely matters once the weights are resident).
#
# This rig previously *rejected* g5g.xlarge on the theory that 8 GiB "cannot
# stage 9.5 GiB of weights". The conclusion was right and the reason was wrong,
# and the fix is swap rather than a bigger instance -- the same remedy
# tpu-pytorch-inf2-2b applies for its neff load.
_SWAP_BELOW_HOST_RAM_GB = 16
_SWAP_GB = 16


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


def _needs_swap(instance_type: str) -> bool:
    """True when host RAM is too small to mmap the checkpoint without swap."""
    return 0 < _host_memory_gb(instance_type) < _SWAP_BELOW_HOST_RAM_GB


def _validate_instance_type(instance_type: str) -> None:
    """Only the size list is enforced. Small hosts are supported, not rejected --
    `_user_data` provisions a swapfile for them (see `_SWAP_BELOW_HOST_RAM_GB`)."""
    if not _is_g5g(instance_type):
        raise ValueError(f"instance_type must be one of {', '.join(sorted(_G5G_SIZES))}")


def _tensor_parallel_size(instance_type: str) -> int:
    return _gpu_count(instance_type)


async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)


def _serve_argv(model: str, instance_type: str) -> str:
    """Arguments for jax_openai_server.py. Deliberately unlike the L4 rigs' set.

    --quant-mode must match the checkpoint, not the chip: a `-w4a16-` export
    carries packed int4 weights and a dense export does not. QUANT_MODE=fp16 in
    tpu.env because MODEL_NAME there is the dense reference build.

    There is no tensor-parallel flag: the JAX engine is single-device. On the
    two-GPU sizes the second T4G idles, which _tensor_parallel_size() reports
    but nothing acts on yet.
    """
    return (
        f"--model {model} --host 0.0.0.0 --port {JAX_PORT} "
        f"--kv-cache-dtype {KV_CACHE_DTYPE} --quant-mode {QUANT_MODE} "
        f"--max-model-len {MAX_MODEL_LEN}"
    )


# The serving payload. These are this rig's own files, shipped to the instance
# over SSM rather than fetched from a registry — there is no published artifact
# for "our JAX Gemma 4 port", and cloning the monorepo would need credentials on
# the box. Gzipped they are ~30 KB of base64, which fits one Run Command; user
# data could not hold them (16 KB limit).
_PAYLOAD_FILES = (
    "jax_openai_server.py",
    "jax_engine.py",
    "ports/gemma4/jax_e_loader.py",
    "ports/gemma4/jax_e_model.py",
)


def _payload_root() -> str:
    """Directory holding the serving payload.

    server.py is also installed as a skill snapshot at
    .claude/skills/<skill>/mcp/server.py, so look up from there too.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.abspath(os.path.join(here, "..", "..", "..", ".."))):
        if all(os.path.exists(os.path.join(cand, f)) for f in _PAYLOAD_FILES):
            return cand
    raise RuntimeError(
        "serving payload not found next to server.py "
        f"(looked for {', '.join(_PAYLOAD_FILES)}). Deploy from the rig directory."
    )


def _payload_tar_b64() -> str:
    """tar.gz of the serving payload, base64-encoded, built deterministically.

    mtime and uid/gid are zeroed so the same sources always produce the same
    string — that is what makes `deploy_jax_server` idempotent and lets a
    redeploy be a no-op you can detect.
    """
    import io as _io
    import tarfile

    root = _payload_root()
    buf = _io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for rel in sorted(_PAYLOAD_FILES):
            info = tar.gettarinfo(os.path.join(root, rel), arcname=rel)
            info.mtime, info.uid, info.gid = 0, 0, 0
            info.uname = info.gname = ""
            with open(os.path.join(root, rel), "rb") as fh:
                tar.addfile(info, fh)
    return base64.b64encode(buf.getvalue()).decode()


def _user_data(model: str, instance_type: str) -> str:
    """Render idempotent cloud-init that installs the JAX runtime.

    It installs and then waits: the serving payload arrives separately via
    `deploy_jax_server`, because it is this rig's own source and does not fit in
    user data. Progress goes to /var/log/jax-install.log and
    {APP_DIR}/INSTALL_DONE appears when the runtime imports and sees the GPU.

    Unlike the vLLM rig there is no `serving` mode: there is no stock-vs-build
    choice to make, because nothing is built.
    """
    _validate_instance_type(instance_type)

    swap = ""
    if _needs_swap(instance_type):
        # Without this the model never loads: the kernel refuses to mmap the
        # 10.2 GB checkpoint on a sub-16 GiB host with no swap. Measured on the
        # vLLM sibling 2026-08-13; the checkpoint and the host are the same here,
        # so the same remedy applies. Idempotent, survives reboot via fstab.
        swap = f"""if ! swapon --show --noheadings 2>/dev/null | grep -q /swapfile; then
  fallocate -l {_SWAP_GB}G /swapfile
  chmod 600 /swapfile
  mkswap -q /swapfile
  swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
"""

    py = f"python{JAX_PYTHON_VERSION}"
    argv = _serve_argv(model, instance_type)
    serving_reqs = " ".join(_SERVING_REQUIREMENTS)

    return f"""#!/usr/bin/env bash
set -euxo pipefail
{swap}mkdir -p {APP_DIR}/app {JAX_COMPILATION_CACHE_DIR}

cat >{APP_DIR}/install.sh <<'INSTEOF'
#!/usr/bin/env bash
set -euxo pipefail

# jax >= 0.11 needs Python 3.12; the Ubuntu 22.04 DLAMI base ships 3.10.
install_runtime() {{
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -y
  apt-get install -y {py} {py}-venv {py}-dev
  curl -sS https://bootstrap.pypa.io/get-pip.py | {py}
  # jax[cuda12] pulls CUDA from pip wheels (aarch64 published for all of them),
  # so the DLAMI only supplies the driver. No CUDA toolkit, no compiler, no Rust.
  {py} -m pip install --upgrade pip setuptools wheel
  {py} -m pip install --upgrade '{JAX_PIP_SPEC}'
  {py} -m pip install --upgrade {serving_reqs}
}}

# Assert the GPU is actually visible to JAX before declaring the install done.
# A driverless ARM64 DLAMI boots fine on a G5g and would otherwise look healthy
# right up until the first token.
verify_gpu() {{
  {py} - <<'PYCHECK'
import jax
devs = jax.devices()
print("jax", jax.__version__, "devices:", devs)
assert devs and devs[0].platform in ("gpu", "cuda"), f"no CUDA device visible to JAX: {{devs}}"
cc = getattr(devs[0], "compute_capability", None)
print("compute_capability:", cc)
PYCHECK
}}

install_runtime
verify_gpu

# Point the unit at the interpreter that actually received the packages.
#
# MEASURED 2026-08-19: the DLAMI already carries /usr/local/bin/python3.12, which
# precedes /usr/bin on PATH, so `python3.12` above installs jax into
# /usr/local/lib/python3.12/site-packages while a hardcoded
# ExecStart=/usr/bin/python3.12 gets the deadsnakes interpreter and dies with
# `ModuleNotFoundError: No module named 'jax'` -- after the install has already
# reported success, because verify_gpu resolves through PATH too.
#
# Rewritten here rather than in the unit template because the resolution can only
# happen after install_runtime has run. Still an absolute path, so systemd is
# happy and ExecStart never depends on the service's PATH.
PY_BIN="$(command -v python{JAX_PYTHON_VERSION})"
sed -i "s|^ExecStart=[^ ]*|ExecStart=$PY_BIN|" /etc/systemd/system/{SERVICE_NAME}.service
systemctl daemon-reload

touch {APP_DIR}/INSTALL_DONE
INSTEOF
chmod 700 {APP_DIR}/install.sh

cat >{APP_DIR}/env <<ENVEOF
MODEL_NAME={model}
KV_CACHE_DTYPE={KV_CACHE_DTYPE}
QUANT_MODE={QUANT_MODE}
MAX_MODEL_LEN={MAX_MODEL_LEN}
JAX_PORT={JAX_PORT}
XLA_PYTHON_CLIENT_MEM_FRACTION={XLA_PYTHON_CLIENT_MEM_FRACTION}
JAX_COMPILATION_CACHE_DIR={JAX_COMPILATION_CACHE_DIR}
PYTHONPATH={APP_DIR}/app
ENVEOF
chmod 600 {APP_DIR}/env

# The HF token is fetched at boot into a root-only EnvironmentFile. It is NEVER
# placed in user data: instance metadata is readable by anything on the box.
#
# xtrace is off across this block on purpose. This script runs under `set -x`,
# and bash traces variable assignments with their values, so leaving it on would
# print the token into /var/log/cloud-init-output.log — readable by anything on
# the instance, which is the exact exposure keeping it out of user data avoids.
set +x
HF=$(aws secretsmanager get-secret-value --region {AWS_REGION} --secret-id {HF_SECRET_ID} --query SecretString --output text 2>/dev/null || true)
if [ -n "$HF" ]; then
  echo "HF_TOKEN=$HF" >>{APP_DIR}/env
fi
unset HF
set -x

cat >/etc/systemd/system/{SERVICE_NAME}.service <<'UNITEOF'
[Unit]
Description=Gemma 4 E2B on T4G via JAX
After=network-online.target

[Service]
Type=simple
EnvironmentFile={APP_DIR}/env
WorkingDirectory={APP_DIR}/app
ExecStart=/usr/bin/{py} {APP_DIR}/app/jax_openai_server.py {argv}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNITEOF
systemctl daemon-reload

nohup bash {APP_DIR}/install.sh >/var/log/jax-install.log 2>&1 &
echo "JAX runtime install started; follow /var/log/jax-install.log, then deploy_jax_server"
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
    iam_instance_profile: str = "g5g-jax-instance-profile",
    spot: bool = True,
) -> str:
    """Return cloud-init and an AWS CLI launch command without changing AWS."""
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
            "Cloud-init installs the JAX runtime only. The serving payload is this "
            "rig's own source and ships separately — run deploy_jax_server once "
            "get_install_progress reports INSTALL COMPLETE."
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
            f"Serving argv: `{_serve_argv(model_name, instance_type)}`\n"
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
    spot: bool = True,
) -> str:
    """Launch one tagged G5g instance using the latest regional ARM64 DLAMI.

    Cloud-init installs Python 3.12 and jax[cuda12] from wheels and asserts JAX
    sees the GPU. It does NOT start serving: the payload is this rig's own
    source, so deploy it with deploy_jax_server once the install finishes.
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
            "UserData": _user_data(model_name, instance_type),
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True},
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
            "Installing the JAX runtime from wheels (no build). Follow with "
            "get_install_progress, then deploy_jax_server to ship the serving code."
        )
        return (
            f"✅ Launching `{instance_id}` ({instance_type}, {market}, "
            f"{_gpu_count(instance_type)}x T4G) in `{AWS_REGION}`.\n{tail}"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List managed G5g instances", annotations=READ_ONLY)
async def list_g5g_instances() -> str:
    """List instances tagged ManagedBy=gpu-jax-g5g-2b."""
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
    """Terminate a managed instance. Permanent, but cheap to redo here: there is
    no built image to lose, only a pip install and the model cache."""
    try:
        await _call(_client("ec2").terminate_instances, InstanceIds=[instance_id])
        return f"🗑️ Terminating `{instance_id}`. Relaunch costs a pip install, not a build."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify GPU compute capability and JAX coverage", annotations=READ_ONLY)
async def verify_gpu_arch(instance_id: str) -> str:
    """Measure whether JAX's CUDA kernels actually cover this GPU.

    This is the rig's central check and the cheapest way to settle the question
    the whole rig turns on. A config flag being accepted proves nothing; a kernel
    either launches or it does not, so this runs a real matmul on the device.

    It reports nvidia-smi's view, JAX's device list and compute capability, the
    dtype the port selects from it, and the result of one fp16 matmul. Note it
    asks JAX rather than torch: the DLAMI's torch carries sm_75 already (measured
    2026-08-12), which says nothing about jaxlib.
    """
    # The reduction accumulates in float32 on purpose. The exact result,
    # 256**3 = 16,777,216, is far past float16's 65,504 max, so summing in
    # float16 overflows to inf and the check can NEVER pass -- on any device.
    # Measured 2026-08-19 on a T4G that was in fact healthy: every element of
    # x @ x was exactly 256.0 and the fp32 sum was exact, while this line
    # reported False. Do not "simplify" the dtype= away.
    #
    # platform is printed on its own labelled line because the CPU-fallback
    # verdict below matches on it; folding it into the device line made that
    # branch unreachable.
    probe = (
        "import jax, jax.numpy as jnp;"
        "d = jax.devices()[0];"
        "print('jax:', jax.__version__);"
        "print('device:', d);"
        "print('platform:', d.platform);"
        "print('capability:', getattr(d, 'compute_capability', None));"
        "x = jnp.ones((256, 256), jnp.float16);"
        "y = x @ x;"
        "print('fp16 matmul ok:', float(y.sum(dtype=jnp.float32)) == 256.0 ** 3)"
    )
    py = f"python{JAX_PYTHON_VERSION}"
    command = (
        "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true; "
        f'{py} -c "{probe}" 2>&1 || true'
    )
    try:
        output = await _ssm(instance_id, command, timeout=600)
        verdict = ""
        if "no kernel image is available" in output:
            verdict = (
                "\n\n❌ `no kernel image is available for execution on the device` — "
                "this jaxlib has no SM 7.5 cubin and no PTX to JIT from. That would "
                "contradict the published arch tables; capture the full log."
            )
        elif "fp16 matmul ok: True" in output:
            verdict = "\n\n✅ JAX reached the GPU and a real fp16 matmul executed."
        elif "platform: cpu" in output:
            verdict = (
                "\n\n❌ JAX fell back to CPU. Either the DLAMI has no NVIDIA driver "
                "(AWS ships driverless ARM64 DLAMIs that boot fine here), or the "
                "CUDA plugin failed to load."
            )
        return f"### GPU probe on `{instance_id}`\n\n```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Deploy the JAX serving payload", annotations=WRITE)
async def deploy_jax_server(instance_id: str, restart: bool = True) -> str:
    """Ship this rig's JAX serving code to the instance and start the service.

    The payload is jax_openai_server.py, jax_engine.py and ports/gemma4/*.py —
    this rig's own source, with no published artifact to pull and no credentials
    on the box to clone the monorepo. It goes over SSM as a gzipped tarball
    (~30 KB of base64); user data could not carry it, at a 16 KB limit.

    Idempotent: the tarball is built deterministically, so redeploying unchanged
    sources writes identical bytes.
    """
    try:
        payload = _payload_tar_b64()
        command = (
            f"set -e; mkdir -p {APP_DIR}/app; "
            f"echo '{payload}' | base64 -d | tar xzf - -C {APP_DIR}/app; "
            f"chmod -R go-w {APP_DIR}/app; "
            f"ls -R {APP_DIR}/app | head -20"
        )
        if restart:
            command += f"; systemctl enable --now {SERVICE_NAME} && systemctl is-active {SERVICE_NAME}"
        output = await _ssm(instance_id, command, timeout=600)
        return (
            f"✅ Deployed {len(_PAYLOAD_FILES)} files ({len(payload) // 1024} KiB base64) "
            f"to `{instance_id}`.\n\n```\n{output}\n```\n"
            + ("Engine init compiles the model; follow with get_jax_logs." if restart else "")
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get JAX runtime install progress", annotations=READ_ONLY)
async def get_install_progress(instance_id: str, tail: int = 40) -> str:
    """Tail the jax[cuda12] install started by cloud-init.

    This is a wheel install, not a build — minutes, not the hours the vLLM
    sibling needs. INSTALL COMPLETE means JAX imported *and* saw the GPU.
    """
    try:
        tail = max(1, min(tail, 5000))
        command = (
            f"test -f {APP_DIR}/INSTALL_DONE && echo 'INSTALL COMPLETE' || echo 'INSTALL IN PROGRESS'; "
            f"tail -n {tail} /var/log/jax-install.log 2>/dev/null || echo 'no install log yet'"
        )
        return f"```\n{await _ssm(instance_id, command)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get JAX server logs", annotations=READ_ONLY)
async def get_jax_logs(instance_id: str, tail: int = 100) -> str:
    """Tail the JAX serving unit's journal.

    systemd, not docker: nothing is containerized on this rig.
    """
    try:
        tail = max(1, min(tail, 5000))
        command = f"journalctl -u {SERVICE_NAME} -n {tail} --no-pager 2>&1"
        return f"```\n{await _ssm(instance_id, command)}\n```"
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
                return f"📡 `http://{host}:{JAX_PORT}/v1`"
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

Serving `{MODEL_NAME}` with **pure JAX** on **EC2 G5g** — AWS Graviton2 (aarch64)
host, NVIDIA **T4G** GPU (Turing, SM 7.5, 15360 MiB measured).

| Setting | Value |
| --- | --- |
| Region | `{AWS_REGION}` |
| Instance type | `{INSTANCE_TYPE}` ({_gpu_count(INSTANCE_TYPE)}x T4G, {_host_memory_gb(INSTANCE_TYPE)} GiB RAM) |
| Runtime | `{JAX_PIP_SPEC}` on Python {JAX_PYTHON_VERSION} |
| dtype | `{DTYPE}` (chosen from the device's compute capability, not from here) |
| KV cache dtype | `{KV_CACHE_DTYPE}` |
| Quant mode | `{QUANT_MODE}` (matches the checkpoint, not the chip) |
| Device mem fraction | `{XLA_PYTHON_CLIENT_MEM_FRACTION}` |
| Service | `{SERVICE_NAME}` (systemd, not docker) |
| Managed-by tag | `{MANAGED_BY}` |

**Why this rig exists.** The vLLM path on identical hardware needs a ~67-minute
from-source build for SM 7.5, a CUDA toolkit, Rust, and an unlanded patch to
Triton's attention kernel. JAX needs none of them: jaxlib and its CUDA plugin
publish aarch64 wheels whose arch tables carry `sm_75`, every CUDA dependency
publishes aarch64 wheels too, and attention here is ordinary XLA rather than a
hand-tiled Triton kernel — so there is no per-block shared-memory ceiling to hit.

**Turing has no bf16 and no fp8.** `bfloat16` does not fail, it *emulates*
through fp32 conversions, which is worse: correct numbers, quiet slowdown. The
port picks `float16` from the live compute capability. fp8 KV is refused outright.

**The ceiling still bites the fused W4A16 kernel.** It is tiled for TPU VMEM and
needs 550 KiB - 1.1 MiB of shared memory per block against Turing's 64 KiB, so
this rig serves the dense reference checkpoint. `check_w4a16_fits_scoped_memory`
refuses at startup rather than at the first token.

**Nothing here has been measured on hardware yet.** The wheel and arch facts were
verified against PyPI and the plugin binary; that is not a served token.

Order of operations: `create_g5g_instance` → `get_install_progress` →
`verify_gpu_arch` → `deploy_jax_server` → `get_jax_logs` → `verify_model_health`.
"""


if __name__ == "__main__":
    mcp.run()
