"""EC2 G4dn (x86_64 + NVIDIA T4) lifecycle and inference MCP server, PyTorch path.

Like the Inf2 sibling, this uses boto3 rather than shelling out to the AWS CLI so
it works with profiles, environment credentials, IAM roles, and SSO-backed
credential processes, and uses Systems Manager Run Command for remote
administration - no inbound SSH rule or private key.

Where this rig sits. It is one corner of a 2x2 over {runtime} x {host}:

                      G5g (Graviton2 aarch64)      G4dn (x86_64)
    pure JAX          gpu-jax-g5g-2b               gpu-jax-g4dn-2b
    PyTorch           gpu-pytorch-g5g-2b           gpu-pytorch-g4dn-2b   <- here

The GPU is the same Turing generation in both columns - T4G on G5g, T4 on G4dn,
both SM 7.5, and both reporting 15360 MiB - so the column is the HOST, not the
chip. Only the host changes across a row, and only the runtime down a column.

THE ROW IS ALREADY ANSWERED, which is what makes this rig's column the
interesting one. gpu-jax-g4dn-2b first served 2026-08-29 and landed exactly on
its G5g sibling: decode 13.1 tok/s against 13.10, tpu_jax_weight_bytes 6.155 GB
against 6.155 GB, and an xprof profile reproducing 54.4% dtype conversion /
32.8% fp32 GEMV / 0.0% TensorCore with roofline peaks identical to three
decimals. The host architecture contributes NOTHING measurable to decode.

So the 86.9% tax is a Turing property, not a Graviton2 one, and the open
question is the one this rig is on the other side of: is it TURING, or is it
XLA? gpu-jax-g4dn-2b is the same host, the same chip, the same checkpoint and
the same dtype policy, which makes this the cleanest runtime A/B in the tree.
Its baseline, to be beaten or matched: 13.1 / 13.2 / 13.1 tok/s at 41 / 521 /
2,057 input tokens, 64 output, median of 3, warmed at the measured shape.

THIS RIG HAS SERVED NOTHING. Forked from `gpu-pytorch-g5g-2b` 2026-08-29 and
retargeted x86_64. `benchmarks/` is empty and every measurement referenced in a
comment below was taken on another rig - see docs/INHERITED.md, which is the
list of what carries and what does not.

What the PyTorch path buys on Turing, and it is a short list because the JAX
sibling already avoids the hard part:

  * torch comes from the AMI, already built for sm_75. Nothing is compiled on
    the instance - no CUDA toolkit, no Rust, no from-source build, and none of
    the vLLM sibling's unlanded Triton patch.
  * attention is transformers' own SDPA, not a hand-tiled kernel, so Turing's
    64 KiB shared-memory ceiling is not in the attention path. That ceiling is
    what forces the vLLM rig onto TRITON_ATTN, whose 512-wide tile for Gemma 4's
    global-attention head wants ~96 KiB per block.
  * transformers owns the KV cache, so the hand-written ring the JAX port
    carries - and the padding-eviction bug in it - has no counterpart here.

What it does not buy is a number. The open question this rig exists to answer is
whether the JAX sibling's decode profile (54.0% of decode in dtype conversion at
0.0% TensorCore utilisation) is a property of Turing or of XLA, and answering it
needs a measurement taken here.
"""

import asyncio
import base64
import hashlib
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
RIG_NAME = "gpu-pytorch-g4dn-2b"

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
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "g4dn.xlarge")
SERVICE_NAME = os.getenv("SERVICE_NAME", "torch-g4dn")
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")
TORCH_PORT = int(os.getenv("TORCH_PORT", "8000"))

# Turing has no bf16 and no fp8 datapath. DTYPE is a RECORD here, not the
# decision: `resolve_compute_dtype()` in torch_openai_server.py reads the live
# device's compute capability and picks float16 below SM 8.0. Asking torch for
# bfloat16 on SM 7.5 does not raise - CUDA emulates it through fp32 - so the
# device is asked rather than trusted. tpu.env explains both.
DTYPE = os.getenv("DTYPE", "float16")
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto")
QUANT_MODE = os.getenv("QUANT_MODE", "fp16")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "1"))

# NOT CARRIED FROM THE JAX SIBLING: PLE_BITS, INT8_LM_HEAD, PREFILL_CHUNK_SIZE
# and XLA_PYTHON_CLIENT_MEM_FRACTION. Every one of them addresses a component of
# `ports/gemma4/`, which this rig does not have - the PLE-table quantiser, the
# hand-rolled prefill chunker, and XLA's device-memory preallocator. transformers
# allocates on demand and quantisation here would go through bitsandbytes or
# torchao, neither of which is installed. They were live constants in the rig
# this was forked from, rendered into the systemd unit for a process that never
# read them; carrying an inert knob is how "the flag was accepted" comes to look
# like evidence.

# TORCH IS NOT IN THIS LIST, AND THAT IS THE POINT. The bootstrap installs INTO
# the DLAMI's own PyTorch environment rather than beside it, which is the exact
# opposite of what the JAX rigs do. There the DLAMI only has to supply a driver
# because pip supplies CUDA; here the image supplies torch itself, already built
# with sm_75 in its arch list. `pip install torch` would replace a vendor build
# known to cover Turing with an upstream wheel whose arch coverage is a separate
# question this rig has not settled - see DLAMI_SSM_PARAMETER below.
TORCH_PIP_SPEC = os.getenv("TORCH_PIP_SPEC", "transformers accelerate")
# The DLAMI's interpreter version, used to NAME the probe, not to install one:
# install.sh locates the image's venv and writes the resolved path to PYTHON_BIN.
# Ubuntu 26.04 carries 3.14 as the system interpreter, but the DLAMI's torch
# lives in its own venv and that is the one the unit must run.
TORCH_PYTHON_VERSION = os.getenv("TORCH_PYTHON_VERSION", "3.14")

# NO COMPILATION CACHE, AND NO S3 SYNC FOR ONE. The rig this was forked from
# carried both, plus a systemd timer pushing /opt/jax-cache to S3 every 10
# minutes. Nothing on this path compiles: there is no torch.compile in
# torch_openai_server.py (see its docstring on why a static-shape buffer is the
# wrong trade on CUDA), so the directory would have stayed empty and the timer
# would have reported a successful sync of nothing, forever. That is the same
# silent-success shape as the JAX rig's own cache bug, where both halves worked
# correctly against a path nothing wrote to. If torch.compile is ever adopted
# here the knob to add is TORCHINDUCTOR_CACHE_DIR, not a JAX one.

# Root volume. The checkpoint downloads here (10.2 GB), the loader reads it back,
# and the pip install and the compilation cache share the same volume.
#
# Throughput is set EXPLICITLY because gp3's default is 125 MiB/s and the two
# dominant load stages sit ON that number rather than near it. MEASURED
# 2026-08-25: read_shards moved the checkpoint in 73.5s (~139 MB/s) and the
# download took 87.7s (~116 MB/s). Two unrelated stages landing on one figure is
# the signature of a volume ceiling, not of CPU or network. Untested as a
# remedy -- the load stages are already timed, so one launch settles it.
#
# 500 MiB/s is ~4x baseline and still under g4dn.xlarge's own EBS ceiling
# ("up to" 4.75 Gbps ~= 593 MB/s), so the smaller sizes stay instance-bound
# rather than volume-bound. gp3 also requires throughput <= IOPS * 0.25 MiB/s,
# which 6000 IOPS satisfies with room to raise throughput to the 1000 cap.
ROOT_VOLUME_GB = int(os.getenv("ROOT_VOLUME_GB", "100"))
ROOT_VOLUME_THROUGHPUT_MBPS = int(os.getenv("ROOT_VOLUME_THROUGHPUT_MBPS", "500"))
ROOT_VOLUME_IOPS = int(os.getenv("ROOT_VOLUME_IOPS", "6000"))

# What cloud-init installs on the instance, beyond TORCH_PIP_SPEC. Kept here rather
# than inline in the bootstrap so requirements-serving.txt can mirror one list;
# tests assert the two agree, because a drifted pair is invisible until a serve.
_SERVING_REQUIREMENTS = (
    # transformers/accelerate ride in TORCH_PIP_SPEC, not here, so the two
    # lists stay disjoint and the mirror file has one entry per install arg.
    "fastapi", "uvicorn", "pydantic",
    "safetensors", "huggingface_hub", "numpy",
    # jinja2 is NOT optional here despite not being imported anywhere in this
    # rig: transformers renders the chat template through it, and every serving
    # path goes through apply_chat_template. Without it /health returns 200 and
    # every /v1/chat/completions returns 500 -- measured 2026-08-19. transformers
    # does not pull it in, and it memoizes the availability check at import, so
    # installing it late needs a service restart.
    "jinja2",
)

# Profiling, installed on the instance alongside the serving deps.
#
# Shipped rather than left to an operator on purpose. The JAX sibling kept its
# profiling deps in a requirements file that was DELIBERATELY excluded from the
# deploy payload, so `docs/profiling-recipes.md` told operators to pip install
# from a path that had never existed on any instance -- xprof "installed" with
# `Could not open requirements file` and the extraction died on
# ModuleNotFoundError, both in logs nobody reads. One provisioning round should
# leave the box able to profile itself.
#
# xprof is the OpenXLA profiler and supplies TensorBoard's profile plugin; torch
# reaches it through `torch.profiler.tensorboard_trace_handler`. Both publish
# x86_64 wheels, which is the easy direction -- it was the aarch64 sibling that
# had to check. Their install is non-fatal -- see install_runtime.
_PROFILING_REQUIREMENTS = ("xprof", "tensorboard")

# Where the serving payload lands on the instance.
APP_DIR = "/opt/torch-g4dn"
# AWS publishes the x86_64 GPU DLAMI as a public SSM parameter. Prefer it: it is
# single-valued and authoritative, where a describe-images name filter is a fuzzy
# match over a much larger set (base images, TensorFlow images, several frozen
# PyTorch lines) in which the newest by CreationDate is frequently not the one
# you want.
#
# A PyTorch DLAMI, NOT the base driver image the JAX rigs use. The reason is the
# whole difference between the two runtimes here: a JAX rig wants a bare driver
# because pip supplies CUDA, and a PyTorch DLAMI would be GBs of image whose
# entire content goes unused. This rig is the reverse - it takes torch FROM the
# image, so the image has to have it.
#
# `/latest/` in a DLAMI parameter path is only the latest build WITHIN that
# PyTorch-version + Ubuntu-version line, and AWS freezes lines it stops
# rebuilding. So the version in the path is a real pin and has to be revisited,
# not assumed to track. VERIFIED 2026-08-29 by enumerating the 60 x86_64 DLAMI
# parameters: `pytorch-2.13-ubuntu-26.04` is the newest line, it resolved to
# ami-0d7cd40a7956dd2c4, and its SSM entry had been rewritten that morning - a
# live line, not a frozen one.
#
# TWO THINGS ARE DELIBERATELY NEWER THAN THE G5G SIBLING, because x86_64 has
# them and arm64 does not: PyTorch 2.13 against its 2.12, and Ubuntu 26.04
# against its 24.04. 26.04 is also what the JAX rigs settled on.
#
# NOT INHERITED, AND DO NOT REINSTATE IT: the g5g rig's comment here argues that
# upstream PyPI wheels omit Turing, so the AMI is the only source of an sm_75
# torch. That was measured for AARCH64 wheels and is not established for x86_64,
# where upstream CUDA wheels have carried sm_75 for years. Taking torch from the
# image is still the right default - it is a vendor build on a vendor driver -
# but the aarch64 argument is not the reason, and repeating it here would turn an
# unverified claim into a cited one. `verify_gpu_arch` settles it on any given
# image in one call; run it before concluding anything about arch coverage.
DLAMI_SSM_PARAMETER = os.getenv(
    "DLAMI_SSM_PARAMETER",
    "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.13-ubuntu-26.04/latest/ami-id",
)
# Fallback only, and it had to be rewritten alongside the SSM path rather than
# carried over. AWS names the two architectures' images in DIFFERENT WORD ORDER:
#
#   arm64   Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch 2.12 (Ubuntu 24.04)
#   x86_64  Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.13 (Ubuntu 26.04)
#
# so the g5g rig's pattern matches ZERO x86_64 images - VERIFIED 2026-08-29 by
# running its filter against this architecture and getting an empty set. Had it
# been carried over unchanged, _resolve_ami would have raised only when SSM was
# also unavailable, which is precisely when the fallback is load-bearing.
DLAMI_NAME = os.getenv("DLAMI_NAME", "Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*")

MANAGED_BY = RIG_NAME

# G4dn topology: (GPUs, host RAM GiB), read from describe_instance_types rather
# than a product page. The ladder is NOT monotonic in the size suffix: 16xlarge
# carries ONE T4 and 12xlarge carries four. Anything deriving GPU count from the
# number in the name is wrong on this family.
_G4DN_SIZES = {
    # instance type: (GPUs, host RAM GiB, vCPUs)
    "g4dn.xlarge": (1, 16, 4),
    "g4dn.2xlarge": (1, 32, 8),
    "g4dn.4xlarge": (1, 64, 16),
    "g4dn.8xlarge": (1, 128, 32),
    "g4dn.12xlarge": (4, 192, 48),
    "g4dn.16xlarge": (1, 256, 64),
    "g4dn.metal": (8, 384, 96),
}

# Host RAM AT OR BELOW this gets a swapfile. INCLUSIVE, and on this family that
# threshold selects exactly one size: g4dn.xlarge, which has exactly 16 GiB and
# is the default.
#
# NOT MEASURED HERE. Both the figure and the reason come from `gpu-jax-g5g-2b`,
# and only one of the two pressures it documents can transfer:
#
#   * Its 8 GiB size could not even mmap the 10.2 GB checkpoint. THIS FAMILY HAS
#     NO 8 GiB SIZE - g4dn starts at 16 - so that failure is unreachable here.
#   * Its 16 GiB size mmapped fine and was then OOM-killed five times at 14.3 GB
#     anon-rss inside the JAX loader's PLE-table quantiser. That code does not
#     exist in this rig either.
#
# So the swapfile is provisioned on a boundary case whose two known causes are
# both absent, and it stays because the remaining pressure is generic and
# untested: 16 GiB of host RAM staging a 10.2 GB checkpoint through
# transformers' loader leaves very little margin, and the failure mode is a
# kernel kill under Restart=on-failure, which reads as a crash-loop rather than
# as memory pressure. Cheap insurance; delete it only with a measurement.
_SWAP_AT_OR_BELOW_HOST_RAM_GB = 16
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


def _is_g4dn(instance_type: str) -> bool:
    return instance_type in _G4DN_SIZES


def _gpu_count(instance_type: str) -> int:
    return _G4DN_SIZES.get(instance_type, (0, 0, 0))[0]


def _host_memory_gb(instance_type: str) -> int:
    return _G4DN_SIZES.get(instance_type, (0, 0, 0))[1]


def _vcpu_count(instance_type: str) -> int:
    """Carried in the table rather than derived. The sibling rig computed vCPUs
    as `host_ram_gb // 2`, which happens to hold on every G5g size and holds on
    NO g4dn size -- a g4dn.xlarge has 16 GiB and 4 vCPUs, not 8. The G-family
    quota is counted in vCPUs, so a derived figure is wrong in the one place the
    number is actually used."""
    return _G4DN_SIZES.get(instance_type, (0, 0, 0))[2]


def _needs_swap(instance_type: str) -> bool:
    """True when host RAM is too small to load the checkpoint without swap.

    Selects exactly one size on this family: g4dn.xlarge, which has exactly
    16 GiB and is the default. Both documented causes are G5g-only and neither
    can occur here -- see _SWAP_AT_OR_BELOW_HOST_RAM_GB for why it stays anyway.
    """
    return 0 < _host_memory_gb(instance_type) <= _SWAP_AT_OR_BELOW_HOST_RAM_GB


def _validate_instance_type(instance_type: str) -> None:
    """Only the size list is enforced. Small hosts are supported, not rejected --
    `_user_data` provisions a swapfile for them (see `_SWAP_AT_OR_BELOW_HOST_RAM_GB`)."""
    if not _is_g4dn(instance_type):
        raise ValueError(f"instance_type must be one of {', '.join(sorted(_G4DN_SIZES))}")


def _tensor_parallel_size(instance_type: str) -> int:
    return _gpu_count(instance_type)


async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)


def _serve_argv(model: str, instance_type: str) -> str:
    """Arguments for torch_openai_server.py -- and ONLY the ones it defines.

    The fork shipped the JAX sibling's set (--kv-cache-dtype, --quant-mode,
    --max-model-len, --ple-bits, --int8-lm-head, --prefill-chunk-size). Not one
    of them exists here: `torch_openai_server.py` defines exactly --model,
    --host, --port and --seq. argparse rejects an unknown flag with exit code 2,
    so the unit would have crash-looped under `Restart=on-failure` from the very
    first start, with the reason only in `journalctl`.

    That is not a cosmetic difference. Those flags name knobs of this repo's own
    JAX port -- PLE quantisation, a fused W4A16 Pallas path, a KV ring. This rig
    is `AutoModelForCausalLM`, which has none of them, so there is nothing to
    forward them to. QUANT_MODE and KV_CACHE_DTYPE survive in `tpu.env` and in
    the unit's environment as a RECORD of what the checkpoint and the device are;
    nothing reads them. Do not re-plumb them into this command on the strength of
    existing -- the JAX-engine knobs that had no reader at all (PLE_BITS,
    INT8_LM_HEAD, PREFILL_CHUNK_SIZE) were dropped in the fork for that reason.

    --seq is the static buffer length and IS this rig's one real serving knob,
    so it comes off MAX_MODEL_LEN rather than the server's own 256 default.

    There is no tensor-parallel flag: the engine is single-device. g4dn.12xlarge
    carries four T4s and g4dn.metal eight; on those sizes every GPU but the first
    idles, which _tensor_parallel_size() reports and nothing acts on yet. Note
    that is a bigger waste than on the G5g sibling, whose largest sizes carry two.
    """
    return (
        f"--model {model} --host 0.0.0.0 --port {TORCH_PORT} "
        f"--seq {MAX_MODEL_LEN}"
    )


# The serving payload. These are this rig's own files, shipped to the instance
# over SSM rather than fetched from a registry — there is no published artifact
# for "our OpenAI-compatible transformers server", and cloning the monorepo
# would need credentials on
# the box. Gzipped they are ~30 KB of base64, which fits one Run Command; user
# data could not hold them (16 KB limit).
_PAYLOAD_FILES = (
    "torch_openai_server.py",
    "torch_generate.py",
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


def _payload_digest(root: str | None = None) -> str:
    """Content digest of the serving payload, as a short hex build id.

    Computed over the payload FILE CONTENTS rather than over the tarball, which
    keeps it non-circular: the digest is written into the tarball, so hashing the
    tarball itself could not work.

    This is the fix for a documented failure. `deploy_torch_server` resolves its
    payload next to server.py, and the MCP server runs from the skill snapshot,
    so on 2026-08-24 a deploy shipped the PREVIOUS `make skill` output, reported
    success, and the instance ran stale code — costing a full measure-and-conclude
    cycle before md5s were compared by hand. The server now reports this id on
    /health, so the comparison is one call instead of a manual hunt.
    """
    root = root or _payload_root()
    digest = hashlib.sha256()
    for rel in sorted(_PAYLOAD_FILES):
        digest.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()[:12]


def _payload_tar_b64() -> str:
    """tar.gz of the serving payload, base64-encoded, built deterministically.

    mtime and uid/gid are zeroed so the same sources always produce the same
    string — that is what makes `deploy_torch_server` idempotent and lets a
    redeploy be a no-op you can detect.

    A PAYLOAD_SHA stamp rides along so the running server can report which
    payload it is executing. It is derived from the same file contents, so it
    does not disturb determinism.
    """
    import gzip as _gzip
    import io as _io
    import tarfile

    root = _payload_root()
    raw = _io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for rel in sorted(_PAYLOAD_FILES):
            info = tar.gettarinfo(os.path.join(root, rel), arcname=rel)
            info.mtime, info.uid, info.gid = 0, 0, 0
            info.uname = info.gname = ""
            with open(os.path.join(root, rel), "rb") as fh:
                tar.addfile(info, fh)
        stamp = _payload_digest(root).encode()
        info = tarfile.TarInfo("PAYLOAD_SHA")
        info.size, info.mtime, info.uid, info.gid, info.mode = len(stamp), 0, 0, 0, 0o644
        info.uname = info.gname = ""
        tar.addfile(info, _io.BytesIO(stamp))

    # gzip's CONTAINER header carries its own MTIME, independent of the tar
    # entries above. Zeroing only the entries -- which is what `mode="w:gz"`
    # left us with -- made this function non-deterministic across a second
    # boundary: two calls in the same second matched, two that straddled one did
    # not. It passed almost every run, which is exactly why it survived.
    buf = _io.BytesIO()
    with _gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(raw.getvalue())
    return base64.b64encode(buf.getvalue()).decode()


def _user_data(model: str, instance_type: str) -> str:
    """Render idempotent cloud-init that installs the PyTorch runtime.

    It installs and then waits: the serving payload arrives separately via
    `deploy_torch_server`, because it is this rig's own source and does not fit in
    user data. Progress goes to /var/log/torch-install.log and
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
  # NOT `mkswap -q`: a busybox flag that util-linux rejects, which under `set -e`
  # kills cloud-init before install.sh is even written -- an empty APP_DIR and no
  # install log. This block renders for the DEFAULT size on this family, so it is
  # on the critical path from the first launch. CLAUDE.md has the history.
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
"""

    py = f"python{TORCH_PYTHON_VERSION}"
    argv = _serve_argv(model, instance_type)
    serving_reqs = " ".join(_SERVING_REQUIREMENTS)
    profiling_reqs = " ".join(_PROFILING_REQUIREMENTS)

    return f"""#!/usr/bin/env bash
set -euxo pipefail
{swap}mkdir -p {APP_DIR}/app

cat >{APP_DIR}/install.sh <<'INSTEOF'
#!/usr/bin/env bash
set -euxo pipefail

# The install is the LONGEST phase of a deployment and was the only one with no
# timings at all -- the model load reports four stages, this reported none. That
# matters most on spot: AWS reclaimed i-0bd73466d5a07a578 21 minutes in, before
# the wheels finished, and nothing recorded which step had been reached. These
# markers are greppable with `grep -F '[stage]' /var/log/torch-install.log`.
_T0=$(date +%s); _TLAST=$_T0
stage() {{
  local now; now=$(date +%s)
  echo "[stage] $1 +$((now - _TLAST))s (total $((now - _T0))s)"
  _TLAST=$now
}}

# This installs a SYSTEM interpreter only so the torch probe below has candidates
# on a non-PyTorch image. On the PyTorch DLAMI this rig defaults to, the
# interpreter that matters is the image's own venv -- see TORCH_PYTHON_VERSION.
install_runtime() {{
  export DEBIAN_FRONTEND=noninteractive
  # Three apt hazards, none of these options optional; all fail the install AND
  # hide it, because INSTALL_DONE is never touched. Lock::Timeout covers
  # unattended-upgrades holding the dpkg lock (apt exits 100 rather than
  # waiting); the Acquire timeouts cover a wedged mirror (apt waits forever
  # rather than erroring); fallback_mirror covers a broken regional mirror.
  # CLAUDE.md has the measurements.
  APT_OPTS="-o DPkg::Lock::Timeout=600 -o Acquire::Retries=3"
  APT_OPTS="$APT_OPTS -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"

  # A regional mirror returning 503 over IPv4 with AAAA-only records, on a host
  # with no IPv6 route, wedged apt-get update for 12 minutes at the first step.
  fallback_mirror() {{
    echo "apt: regional mirror failed; falling back to ports.ubuntu.com" >&2
    sed -i -E 's|https?://[a-z0-9.-]+[.]ec2[.]ports[.]ubuntu[.]com|http://ports.ubuntu.com|g' /etc/apt/sources.list
    apt-get $APT_OPTS update -y
  }}

  apt_run() {{
    apt-get $APT_OPTS "$@" || {{ fallback_mirror; apt-get $APT_OPTS "$@"; }}
  }}

  apt_run update -y
  stage apt-base

  # Use the system interpreter when it is ALREADY the version we want, and only
  # reach for deadsnakes when it is not. On Ubuntu 26.04
  # python{TORCH_PYTHON_VERSION} IS the system python, so this drops a third-party PPA, an
  # `add-apt-repository`, and a second full `apt-get update` off the critical
  # path.
  #
  # THAT IS WHY TORCH_PYTHON_VERSION IS 3.14 HERE AND 3.12 ON THE G5G SIBLING,
  # and the two have to move together with DLAMI_SSM_PARAMETER. deadsnakes
  # publishes python3.14 for jammy and noble ONLY -- not for 26.04 -- so pinning
  # 3.12 against a 26.04 image would miss the system interpreter, take the
  # deadsnakes branch, and fail under `set -e` on a PPA with nothing to offer it.
  # This version does not choose what the SERVICE runs: that is the DLAMI's own
  # venv, located by the probe below. It only decides whether this stage is a
  # no-op or a PPA round trip.
  if command -v {py} >/dev/null 2>&1; then
    echo "{py} is already present at $(command -v {py}); skipping deadsnakes"
    # Still needed even when the interpreter ships with the distro: a source
    # build of any dependency without a wheel for this interpreter needs the
    # headers. Less likely on x86_64 than it was on aarch64, not impossible.
    apt_run install -y {py}-venv {py}-dev || apt_run install -y python3-venv python3-dev
  else
    apt_run install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt_run update -y
    apt_run install -y {py} {py}-venv {py}-dev
  fi
  stage python-{TORCH_PYTHON_VERSION}

  # THE INTERPRETER IS NOT A FREE CHOICE HERE. torch comes from the AMI, and the
  # DLAMI ships it in a venv rather than the system interpreter -- so install
  # into the one that can ALREADY import torch, found by probing because the venv
  # path moves between releases. Guessing wrong yields ModuleNotFoundError at the
  # first token, after the install has reported success.
  TORCH_PY=""
  for cand in /opt/pytorch/bin/python /opt/conda/bin/python \
              /usr/local/bin/{py} "$(command -v {py} || true)" /usr/bin/python3 \
              /opt/*/bin/python /opt/*/*/bin/python; do
    [ -x "$cand" ] || continue
    if "$cand" -c 'import torch' >/dev/null 2>&1; then
      TORCH_PY="$cand"
      echo "torch found in $cand: $("$cand" -c 'import torch; print(torch.__version__, torch.cuda.get_arch_list())')"
      break
    fi
  done
  if [ -z "$TORCH_PY" ]; then
    echo "FATAL: no interpreter on this AMI can import torch." >&2
    echo "This rig takes torch from the DLAMI and never pip-installs it, so an" >&2
    # Single-quoted: backticks inside a double-quoted string are COMMAND
    # SUBSTITUTION, so the double-quoted version of this line would actually run
    # `pip install torch` -- from inside the error path that warns against it.
    echo 'image with no torch is a WRONG AMI, not a missing "pip install torch".' >&2
    echo "Check DLAMI_SSM_PARAMETER really names a PyTorch DLAMI, not a base image." >&2
    echo "Candidates probed:" >&2
    ls -d /opt/*/bin/python /opt/*/*/bin/python 2>/dev/null >&2 || true
    exit 1
  fi
  echo "$TORCH_PY" > {APP_DIR}/PYTHON_BIN
  stage torch-interpreter

  # PEP 668: Ubuntu marks its system interpreter externally-managed from 23.04
  # on, so a system-wide pip install fails outright. Single-purpose serving box,
  # installing into the interpreter systemd runs, and the repo forbids venvs.
  # Harmless and ignored inside the DLAMI venv, which is not marked.
  PIP="$TORCH_PY -m pip install --upgrade --break-system-packages"
  # Conditional: the DLAMI's venv already has pip, and bootstrapping over it is
  # a needless network dependency on the critical path. The system interpreter
  # on a bare base image may not, and get-pip.py runs BEFORE $PIP exists so it
  # needs the PEP 668 flag of its own.
  "$TORCH_PY" -m pip --version >/dev/null 2>&1 || \
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$TORCH_PY" - --break-system-packages
  $PIP pip setuptools wheel
  stage pip-bootstrap
  # UNQUOTED on purpose. TORCH_PIP_SPEC is "transformers accelerate" -- two
  # packages, two arguments. The sibling quoted it because its value was
  # `jax[cuda13]`, one requirement whose brackets the shell would glob; carried
  # over verbatim, the quotes made pip parse "transformers accelerate" as a
  # SINGLE requirement and fail. Do not re-add them without checking the value.
  $PIP {TORCH_PIP_SPEC}
  stage torch-deps
  $PIP {serving_reqs}
  stage serving-deps

  # Profiling and trace-viewing, installed on the box so a profile can be taken
  # without a second provisioning round. NOT fatal: a missing wheel must not cost
  # the serve -- the whole reason install.sh dies loudly elsewhere is that
  # `set -e` used to kill it here. xprof's manylinux_2_35 floor is comfortable on
  # 26.04 (glibc 2.43); on x86_64 the wheel availability question that made this
  # non-fatal on aarch64 barely arises, and it stays non-fatal regardless.
  # The failure is announced rather than swallowed; grep the marker.
  if $PIP {profiling_reqs}; then
    stage profiling-deps
  else
    echo "[stage] profiling-deps FAILED -- xprof/tensorboard unavailable, serving unaffected" >&2
  fi
}}

# Assert the GPU is actually visible to TORCH before declaring the install done.
# An image with a driver but no torch, or a torch with no sm_75 cubin, boots
# fine and would otherwise look healthy right up until the first token.
#
# This imported `jax` until 2026-08-29 -- inherited verbatim from the sibling,
# on a rig that installs no jax. It runs under `set -e`, so ModuleNotFoundError
# killed install.sh here, INSTALL_DONE was never written, and get_install_progress
# reported "INSTALL IN PROGRESS" forever. The rig had served nothing, so nothing
# had ever executed this line.
#
# The matmul is the point, not the import: `torch.cuda.is_available()` is true on
# a driver the arch list does not cover, and the failure then surfaces as a kernel
# error at the first token. sm_75 must be in the arch list AND run.
verify_gpu() {{
  "$(cat {APP_DIR}/PYTHON_BIN)" - <<'PYCHECK'
import torch
print("torch", torch.__version__, "arch_list", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), (
    "no CUDA device visible to torch. On this rig that usually means the AMI is "
    "an AMI with no NVIDIA driver, or an interpreter that is not the DLAMI's."
)
major, minor = torch.cuda.get_device_capability(0)
print("device:", torch.cuda.get_device_name(0), f"sm_{{major}}{{minor}}")
assert f"sm_{{major}}{{minor}}" in torch.cuda.get_arch_list(), (
    f"torch has no sm_{{major}}{{minor}} cubin: {{torch.cuda.get_arch_list()}}"
)
x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
print("fp16 matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
PYCHECK
}}

install_runtime
verify_gpu
stage gpu-verify
# Point the unit at the interpreter install_runtime PROBED, never one re-resolved
# through PATH: the DLAMI can carry a same-version interpreter under
# /usr/local/bin, and the one holding torch is a venv that is not on PATH at all.
# Reading the recorded path keeps install, verification and ExecStart on one
# interpreter instead of three. Done here rather than in the unit template
# because it can only be resolved after install_runtime has run.
PY_BIN="$(cat {APP_DIR}/PYTHON_BIN)"
sed -i "s|^ExecStart=[^ ]*|ExecStart=$PY_BIN|" /etc/systemd/system/{SERVICE_NAME}.service
systemctl daemon-reload

stage unit-rewrite
touch {APP_DIR}/INSTALL_DONE
echo "[stage] INSTALL COMPLETE total $(($(date +%s) - _T0))s"
INSTEOF
chmod 700 {APP_DIR}/install.sh

cat >{APP_DIR}/env <<ENVEOF
MODEL_NAME={model}
RIG_NAME={MANAGED_BY}
KV_CACHE_DTYPE={KV_CACHE_DTYPE}
QUANT_MODE={QUANT_MODE}
MAX_MODEL_LEN={MAX_MODEL_LEN}
TORCH_PORT={TORCH_PORT}
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
# Three failure modes used to render identically -- `2>/dev/null || true` on one
# line meant a missing CLI, a missing secret and a denied GetSecretValue all
# produced exactly "no token", and the visible symptom is a 401 on the
# checkpoint download minutes later. Changing the base image made the first of
# those newly plausible, so they are separated. xtrace stays OFF across the
# whole block: bash traces assignments WITH their values.
if ! command -v aws >/dev/null 2>&1; then
  echo "WARNING: aws CLI not on PATH; cannot read secret {HF_SECRET_ID}." >&2
  echo "WARNING: a gated checkpoint will fail with 401 at download." >&2
else
  HF=$(aws secretsmanager get-secret-value --region {AWS_REGION} --secret-id {HF_SECRET_ID} --query SecretString --output text 2>/dev/null || true)
  if [ -n "$HF" ]; then
    echo "HF_TOKEN=$HF" >>{APP_DIR}/env
    echo "HF token written to {APP_DIR}/env"
  else
    echo "WARNING: secret {HF_SECRET_ID} is empty or unreadable (check the" >&2
    echo "WARNING: instance profile's secretsmanager:GetSecretValue)." >&2
  fi
  unset HF
fi
set -x

cat >/etc/systemd/system/{SERVICE_NAME}.service <<'UNITEOF'
[Unit]
Description=Gemma 4 E2B on NVIDIA T4 via PyTorch
After=network-online.target

[Service]
Type=simple
EnvironmentFile={APP_DIR}/env
WorkingDirectory={APP_DIR}/app
ExecStart=/usr/bin/{py} {APP_DIR}/app/torch_openai_server.py {argv}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNITEOF
systemctl daemon-reload
nohup bash {APP_DIR}/install.sh >/var/log/torch-install.log 2>&1 &
echo "torch runtime install started; follow /var/log/torch-install.log, then deploy_torch_server"
"""


async def _resolve_ami(ec2=None) -> str:
    """Resolve the x86_64 **PyTorch GPU** DLAMI for this region.

    Three things have to hold and they are separate. The image must be x86_64;
    it must carry the NVIDIA driver; and unlike the JAX rigs, it must carry
    **torch itself**, because nothing here pip-installs it. That third
    requirement is why the parameter names a PyTorch line rather than a base
    image, and it is the one a driver-only AMI satisfies while still failing at
    install.sh's torch-interpreter stage.

    The SSM public parameter pins all three. describe-images is the fallback,
    and its pattern is architecture-specific -- see DLAMI_NAME.

    NEVER hardcode an AMI id. The x86_64 ids the legacy tips-tree rigs pinned
    are frozen images from a line AWS stopped rebuilding.
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
            f"No x86_64 PyTorch GPU DLAMI in {AWS_REGION} via SSM ({DLAMI_SSM_PARAMETER}) "
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


# AWS caps each of StandardOutputContent and StandardErrorContent returned by
# get_command_invocation at 24,000 characters. Past that the content is silently
# truncated — which for get_torch_logs means reading a partial journal and
# concluding the error is not there. Detected rather than assumed: we compare
# against the cap and say so.
_SSM_OUTPUT_CAP = 24_000


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
    # Logged at issue time so an invocation that later times out is still
    # findable in the console. Previously the id was bound here and discarded on
    # every failure path, leaving nothing to look up — and on timeout the command
    # is still RUNNING on the box.
    logger.info("ssm send_command id=%s instance=%s timeout=%ds",
                command_id, instance_id, timeout)
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
            stdout = result.get("StandardOutputContent", "")
            stderr = result.get("StandardErrorContent", "")
            output = (stdout + stderr).strip()
            truncated = (
                len(stdout) >= _SSM_OUTPUT_CAP or len(stderr) >= _SSM_OUTPUT_CAP
            )
            if truncated:
                logger.warning("ssm output truncated id=%s", command_id)
                output += (
                    f"\n\n[!] OUTPUT TRUNCATED by SSM at {_SSM_OUTPUT_CAP} characters. "
                    f"This is a partial result — an error may be missing from it. "
                    f"Re-run with a smaller tail, or fetch the full output with "
                    f"`aws ssm get-command-invocation --command-id {command_id} "
                    f"--instance-id {instance_id}`."
                )
            if result["Status"] != "Success":
                raise RuntimeError(
                    f"SSM {result['Status']} (command-id {command_id}): {output}"
                )
            return output
        await asyncio.sleep(2)
    raise TimeoutError(
        f"SSM command did not finish in {timeout}s. It is STILL RUNNING on "
        f"{instance_id}; poll it with command-id {command_id}."
    )


def _error(exc: Exception) -> str:
    """Render an exception for the MCP client, and put the traceback in the log.

    The 18 tool bodies that call this all discard the stack: the client sees one
    line and stderr had nothing at all. logger.exception here covers every call
    site at once — server.py does configure logging, so this actually emits.
    """
    logger.exception("tool call failed: %s", exc)
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


@mcp.tool(title="Generate G4dn deployment configuration", annotations=READ_ONLY)
async def get_deployment_config(
    model_name: str = MODEL_NAME,
    instance_type: str = INSTANCE_TYPE,
    subnet_id: str = "<subnet-id>",
    security_group_id: str = "<security-group-id>",
    iam_instance_profile: str = "g4dn-torch-instance-profile",
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
            "Cloud-init installs the runtime only. The serving payload is this "
            "rig's own source and ships separately — run deploy_torch_server once "
            "get_install_progress reports INSTALL COMPLETE."
        )
        return (
            f"### EC2 G4dn deployment ({instance_type}, {_gpu_count(instance_type)}x T4)\n\n```bash\n"
            f"# x86_64 *PyTorch* GPU DLAMI. The SSM parameter pins the architecture, the\n"
            f"# NVIDIA driver, and torch itself -- this rig never pip-installs torch.\n"
            f"AMI_ID=$(aws ssm get-parameter --region {AWS_REGION} "
            f"--name {DLAMI_SSM_PARAMETER} "
            "--query 'Parameter.Value' --output text)\n"
            f'aws ec2 run-instances --region {AWS_REGION} --image-id "$AMI_ID" '
            f"--instance-type {instance_type} --subnet-id {subnet_id} "
            f"--security-group-ids {security_group_id} "
            f"--iam-instance-profile Name={iam_instance_profile} "
            f"{market}"
            # Rendered from the same constants create_g4dn_instance launches with.
            # This line read VolumeSize=200 while the tool launched 100 and
            # neither carried throughput -- so the copy-pasteable repro command
            # provisioned a different volume from the tool it documents.
            f"--block-device-mappings 'DeviceName=/dev/sda1,Ebs={{VolumeSize={ROOT_VOLUME_GB},"
            f"VolumeType=gp3,Throughput={ROOT_VOLUME_THROUGHPUT_MBPS},Iops={ROOT_VOLUME_IOPS},"
            f"DeleteOnTermination=true}}' "
            f"--user-data '{encoded}' --tag-specifications "
            f"'ResourceType=instance,Tags=[{{Key=Name,Value={SERVICE_NAME}}},"
            f"{{Key=ManagedBy,Value={MANAGED_BY}}}]'\n```\n\n"
            f"{note}\n\n"
            f"Serving argv: `{_serve_argv(model_name, instance_type)}`\n"
            f"dtype is `{DTYPE}`, not bfloat16 — Turing has no bf16 datapath."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Create G4dn instance", annotations=WRITE)
async def create_g4dn_instance(
    subnet_id: str,
    security_group_id: str,
    iam_instance_profile: str,
    name: str = SERVICE_NAME,
    model_name: str = MODEL_NAME,
    instance_type: str = INSTANCE_TYPE,
    spot: bool = True,
) -> str:
    """Launch one tagged G4dn instance using the latest regional x86_64 PyTorch DLAMI.

    Cloud-init resolves the interpreter holding the DLAMI's torch, installs
    transformers/accelerate and the serving deps into it, asserts torch
    sees the GPU. It does NOT start serving: the payload is this rig's own
    source, so deploy it with deploy_torch_server once the install finishes.
    Spot is the default; pass spot=False for on-demand.
    """
    try:
        _validate_instance_type(instance_type)
        if await _instances(name):
            return f"❌ A managed instance named `{name}` already exists."
        ec2 = _client("ec2")
        ami_id = await _resolve_ami(ec2)
        args = {
            "ImageId": ami_id,
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
            "Installing transformers into the DLAMI's torch (no build, no pip torch). "
            "Follow with get_install_progress, then deploy_torch_server to ship the "
            "serving code."
        )
        # The AMI id belongs in the confirmation: a base DLAMI and a PyTorch
        # DLAMI both boot perfectly well here, and the difference only surfaces
        # at install.sh's torch-interpreter stage, which reads as a broken
        # bootstrap rather than a wrong image. Recording which image booted makes
        # that one lookup instead of a guess.
        logger.info("launched instance=%s ami=%s type=%s market=%s",
                    instance_id, ami_id, instance_type, market)
        return (
            f"✅ Launching `{instance_id}` ({instance_type}, {market}, "
            f"{_gpu_count(instance_type)}x T4) in `{AWS_REGION}`.\n"
            f"AMI: `{ami_id}`\n{tail}"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List managed G4dn instances", annotations=READ_ONLY)
async def list_g4dn_instances() -> str:
    """List instances tagged ManagedBy=gpu-pytorch-g4dn-2b."""
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


@mcp.tool(title="Start G4dn instance", annotations=WRITE)
async def start_g4dn_instance(instance_id: str) -> str:
    """Start a stopped managed instance."""
    try:
        await _call(_client("ec2").start_instances, InstanceIds=[instance_id])
        return f"✅ Starting `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Stop G4dn instance", annotations=DESTRUCTIVE)
async def stop_g4dn_instance(instance_id: str) -> str:
    """Stop a running managed instance. One-time spot instances cannot be
    stopped, only terminated."""
    try:
        await _call(_client("ec2").stop_instances, InstanceIds=[instance_id])
        return f"🛑 Stopping `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Terminate G4dn instance", annotations=DESTRUCTIVE)
async def terminate_g4dn_instance(instance_id: str) -> str:
    """Terminate a managed instance. Permanent, but cheap to redo here: there is
    no built image to lose, only a pip install and the model cache."""
    try:
        await _call(_client("ec2").terminate_instances, InstanceIds=[instance_id])
        return f"🗑️ Terminating `{instance_id}`. Relaunch costs a pip install, not a build."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify GPU compute capability and torch coverage", annotations=READ_ONLY)
async def verify_gpu_arch(instance_id: str) -> str:
    """Measure whether torch's CUDA kernels actually cover this GPU.

    This is the rig's central check and the cheapest way to settle the question
    the whole rig turns on. A config flag being accepted proves nothing; a kernel
    either launches or it does not, so this runs a real matmul on the device.

    It reports nvidia-smi's view, torch's version and arch list, the device
    capability, the dtype this rig selects from it, and the result of one fp16
    matmul.

    It asks TORCH, not jax -- inverting the JAX siblings, and not a cosmetic
    swap. Torch here comes from the AMI rather than from pip, so "does this GPU
    have kernels" is a question about the image that booted, and the arch list is
    the direct answer.

    RUN IT BEFORE CONCLUDING ANYTHING ABOUT ARCH COVERAGE ON THIS RIG. The G5g
    sibling can state flatly that upstream wheels omit sm_75, because that was
    measured for aarch64. It is NOT established for x86_64, where upstream CUDA
    wheels have long carried Turing -- so the claim was deliberately not carried
    over, and this tool is what would settle it. Printing the arch list next to
    the measured capability makes it one glance rather than an argument.
    """
    # The reduction accumulates in float32 on purpose. The exact result,
    # 256**3 = 16,777,216, is far past float16's 65,504 max, so summing in
    # float16 overflows to inf and the check can NEVER pass -- on any device.
    # Inherited fix, measured on the G5g sibling 2026-08-19 on a T4G that was in
    # fact healthy: every element of x @ x was exactly 256.0 and the fp32 sum was
    # exact, while this line reported False. Nothing about it is host-specific --
    # it is float16's 65,504 max against a 16,777,216 result. Do not "simplify"
    # the dtype= away.
    #
    # platform is printed on its own labelled line because the CPU-fallback
    # verdict below matches on it; folding it into the device line made that
    # branch unreachable.
    probe = (
        "import torch;"
        "print('torch:', torch.__version__);"
        "print('arch_list:', torch.cuda.get_arch_list());"
        "avail = torch.cuda.is_available();"
        "print('platform:', 'cuda' if avail else 'cpu');"
        "d = torch.device('cuda' if avail else 'cpu');"
        "print('device:', torch.cuda.get_device_name(0) if avail else 'none');"
        "cap = torch.cuda.get_device_capability(0) if avail else None;"
        "print('capability:', cap);"
        "print('compute_dtype:', 'float16' if (cap and cap < (8, 0)) else 'bfloat16');"
        "x = torch.ones((256, 256), dtype=torch.float16, device=d);"
        "y = x @ x;"
        "print('fp16 matmul ok:', float(y.sum(dtype=torch.float32)) == 256.0 ** 3)"
    )
    # Ask the interpreter install.sh recorded, not one re-resolved through PATH:
    # torch lives in the DLAMI's venv, which is generally not on PATH at all, so
    # a bare `python3.12 -c "import torch"` reports a broken GPU on a healthy box.
    # Falls back to PATH only when the marker is absent (i.e. before install).
    py = f"python{TORCH_PYTHON_VERSION}"
    command = (
        "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true; "
        f'PY="$(cat {APP_DIR}/PYTHON_BIN 2>/dev/null || command -v {py})"; '
        'echo "interpreter: $PY"; '
        f'"$PY" -c "{probe}" 2>&1 || true'
    )
    try:
        output = await _ssm(instance_id, command, timeout=600)
        verdict = ""
        if "no kernel image is available" in output:
            verdict = (
                "\n\n❌ `no kernel image is available for execution on the device` — "
                "this torch build has no SM 7.5 cubin. Check `arch_list` above: if "
                "sm_75 is missing, a pip torch has shadowed the DLAMI's."
            )
        elif "fp16 matmul ok: True" in output:
            verdict = "\n\n✅ torch reached the GPU and a real fp16 matmul executed."
        elif "platform: cpu" in output:
            verdict = (
                "\n\n❌ torch fell back to CPU. Either the DLAMI has no NVIDIA driver "
                "(a base DLAMI without one boots fine here), or the "
                "interpreter probed is not the one holding the DLAMI's torch."
            )
        elif "No module named 'torch'" in output:
            verdict = (
                "\n\n❌ that interpreter has no torch at all. This rig never pip-installs "
                "torch; it probes for the DLAMI's. Re-check `install.sh`'s torch-interpreter "
                "stage, or that the AMI really is a PyTorch DLAMI and not the base image."
            )
        return f"### GPU probe on `{instance_id}`\n\n```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Deploy the PyTorch serving payload", annotations=WRITE)
async def deploy_torch_server(instance_id: str, restart: bool = True) -> str:
    """Ship this rig's PyTorch serving code to the instance and start the service.

    The payload is torch_openai_server.py and torch_generate.py —
    this rig's own source, with no published artifact to pull and no credentials
    on the box to clone the monorepo. It goes over SSM as a gzipped tarball
    (~30 KB of base64); user data could not carry it, at a 16 KB limit.

    Idempotent: the tarball is built deterministically, so redeploying unchanged
    sources writes identical bytes.

    With restart=True the unit is RESTARTED, not merely started. `enable --now`
    is a no-op against an already-running unit, so it shipped new files and left
    the old process serving them -- and `is-active` then reported "active", which
    reads as success. Every redeploy silently served stale code.
    """
    try:
        root = _payload_root()
        build_id = _payload_digest(root)
        payload = _payload_tar_b64()
        command = (
            f"set -e; mkdir -p {APP_DIR}/app; "
            f"echo '{payload}' | base64 -d | tar xzf - -C {APP_DIR}/app; "
            f"chmod -R go-w {APP_DIR}/app; "
            f"ls -R {APP_DIR}/app | head -20"
        )
        if restart:
            # `restart` starts a stopped unit and replaces a running one, so it is
            # correct on both the first deploy and every redeploy. Report the PID
            # and start time as well: "active" alone cannot distinguish a fresh
            # process from the one that was already there.
            command += (
                f"; systemctl enable {SERVICE_NAME} >/dev/null 2>&1"
                f"; systemctl restart {SERVICE_NAME}"
                f"; systemctl show {SERVICE_NAME} -p ActiveState -p MainPID -p ExecMainStartTimestamp"
            )
        output = await _ssm(instance_id, command, timeout=600)
        return (
            f"✅ Deployed {len(_PAYLOAD_FILES)} files ({len(payload) // 1024} KiB base64) "
            f"to `{instance_id}`.\n\n"
            # WHICH sources, and their digest. _payload_root() silently picks
            # between the working tree and the skill snapshot; on 2026-08-24 it
            # chose the stale snapshot and nothing said so. Both facts are now in
            # the deploy output, and verify_model_health compares the digest
            # against what the instance reports on /health.
            f"Payload root: `{root}`\n"
            f"Build id: `{build_id}` — verify_model_health checks the running "
            f"server reports this.\n\n"
            f"```\n{output}\n```\n"
            + ("Engine init compiles the model; follow with get_torch_logs." if restart else "")
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get PyTorch runtime install progress", annotations=READ_ONLY)
async def get_install_progress(instance_id: str, tail: int = 40) -> str:
    """Tail the runtime install started by cloud-init, and cloud-init itself.

    This is a wheel install on top of the AMI's torch, not a build — minutes,
    not the hours the vLLM sibling needs. INSTALL COMPLETE means torch imported
    *and* saw the GPU.

    It reports cloud-init's OWN state too, and that is the point rather than a
    nicety. Cloud-init writes install.sh and then backgrounds it, so anything
    that kills cloud-init BEFORE that point leaves no install log at all — and
    this tool used to render that as `INSTALL IN PROGRESS` + `no install log
    yet`, forever, which is also exactly what a healthy slow install looks like.
    A dead bootstrap and a running one must not share a rendering.

    MEASURED 2026-08-26: `mkswap -q` (a busybox flag util-linux rejects) failed
    under `set -e` in the swap block, which renders FIRST, so cloud-init died
    before install.sh existed. The instance sat there looking like it was
    installing. The flag is fixed; this is the fix for not being able to see it.
    """
    try:
        tail = max(1, min(tail, 5000))
        # cloud-init-output.log is tailed only when the install log is absent.
        # When install.sh did start it is the more specific evidence, and both
        # at full length would risk the 24,000-character SSM cap.
        command = (
            f"test -f {APP_DIR}/INSTALL_DONE && echo 'INSTALL COMPLETE' || echo 'INSTALL IN PROGRESS'; "
            "echo '--- cloud-init ---'; "
            "cloud-init status --long 2>&1 || echo 'cloud-init status unavailable'; "
            "echo '--- install log ---'; "
            "if [ -f /var/log/torch-install.log ]; then "
            f"tail -n {tail} /var/log/torch-install.log; "
            "else echo 'NO INSTALL LOG: cloud-init never reached install.sh'; "
            "echo '--- cloud-init output (tail 60) ---'; "
            "tail -n 60 /var/log/cloud-init-output.log 2>/dev/null "
            "|| echo 'no cloud-init output log either'; fi"
        )
        output = await _ssm(instance_id, command)

        # Ordered most-specific first. "status: error" is cloud-init's own
        # verdict and outranks the absence of a log, which is only a symptom.
        if "INSTALL COMPLETE" in output:
            verdict = "\n\n✅ Runtime installed and torch saw the GPU. Next: deploy_torch_server."
        elif "status: error" in output:
            verdict = (
                "\n\n❌ cloud-init FAILED — the bootstrap died, the install is not "
                "running and never will be. Read the cloud-init output above for the "
                "failing command; relaunching will reproduce it. This is NOT a slow install."
            )
        elif "NO INSTALL LOG" in output and "status: done" in output:
            verdict = (
                "\n\n❌ cloud-init finished but never wrote /var/log/torch-install.log, so "
                "the bootstrap exited before backgrounding install.sh. Nothing is "
                "installing. Check the cloud-init output above."
            )
        elif "NO INSTALL LOG" in output:
            verdict = (
                "\n\n⏳ cloud-init is still running and has not reached install.sh yet. "
                "Normal for the first minute or two after launch; if it persists, the "
                "bootstrap is stuck before the install."
            )
        else:
            verdict = "\n\n⏳ Installing. Wheels, not a build — minutes, not hours."
        return f"```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get PyTorch server logs", annotations=READ_ONLY)
async def get_torch_logs(instance_id: str, tail: int = 100) -> str:
    """Tail the PyTorch serving unit's journal.

    systemd, not docker: nothing is containerized on this rig.
    """
    try:
        tail = max(1, min(tail, 5000))
        command = f"journalctl -u {SERVICE_NAME} -n {tail} --no-pager 2>&1"
        return f"```\n{await _ssm(instance_id, command)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get G4dn endpoint", annotations=READ_ONLY)
async def get_endpoint(instance_id: str) -> str:
    """Resolve the instance's OpenAI-compatible base URL. Never hardcoded."""
    try:
        response = await _call(_client("ec2").describe_instances, InstanceIds=[instance_id])
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                host = inst.get("PublicIpAddress") or inst.get("PrivateIpAddress")
                if not host:
                    return f"❌ `{instance_id}` has no reachable address yet."
                return f"📡 `http://{host}:{TORCH_PORT}/v1`"
        return f"❌ `{instance_id}` not found in `{AWS_REGION}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify model health", annotations=READ_ONLY)
async def verify_model_health(instance_id: str) -> str:
    """Check /health, the served build id, and whether the reply was degenerate.

    Uses /v1/chat/completions: raw /v1/completions returns an empty completion on
    `-it` models, so an empty body there is not evidence of a broken deploy.

    It deliberately does NOT pass on "the reply was non-empty". That is the check
    the engineering rules call out by name — on the vLLM sibling a broken deploy
    answered `': ok: ok: ok…'`, and on this rig KV-ring eviction returned a token
    loop with status="success". A non-empty body is not evidence of health.
    Instead this reads `tpu_jax_degenerate_responses_total` either side of its own
    probe, so the verdict comes from the server's own judgement of the full text.

    It also compares the build id the server reports against the digest of the
    local payload, which is what turns a stale deploy from a multi-hour
    investigation into one line of output.
    """
    try:
        endpoint = await get_endpoint(instance_id)
        if not endpoint.startswith("📡"):
            return endpoint
        base = endpoint.strip("📡 `")
        metrics_url = base.replace("/v1", "/metrics")
        async with httpx.AsyncClient(timeout=60) as client:
            health = await client.get(base.replace("/v1", "/health"))
            before, _, _ = _parse_prom((await client.get(metrics_url)).text)
            chat = await client.post(
                f"{base}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                    "max_tokens": 16,
                },
            )
            after, _, _ = _parse_prom((await client.get(metrics_url)).text)

        body = chat.json()
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = body.get("usage") or {}
        key = "tpu_jax_degenerate_responses_total"
        degenerate = after.get(key, 0.0) > before.get(key, 0.0)

        try:
            health_body = health.json()
        except Exception:
            health_body = {}
        served_build = health_body.get("build_id", "unknown")
        try:
            local_build = _payload_digest()
        except Exception:
            local_build = "unavailable"

        ok = (
            health.status_code == 200
            and chat.status_code == 200
            and not degenerate
            and usage.get("completion_tokens", 0) > 0
        )
        status = "✅" if ok else "❌"

        lines = [
            f"{status} health={health.status_code} "
            f"tokens={usage.get('completion_tokens', 0)} reply={text!r}",
            "",
            f"- Degenerate (server's own verdict on the full text): "
            f"**{'YES — token loop' if degenerate else 'no'}**",
            f"- Build id served: `{served_build}`",
        ]
        if local_build not in ("unavailable", served_build) and served_build != "unknown":
            lines.append(
                f"- ⚠️ **STALE DEPLOY**: the local payload digests to `{local_build}`, "
                f"but the instance is serving `{served_build}`. Run `make skill` "
                f"and then deploy_torch_server — deploy ships the SKILL SNAPSHOT, "
                f"not the working tree."
            )
        elif local_build == served_build:
            lines.append(f"- Build id matches the local payload (`{local_build}`).")
        if usage.get("pad_tokens") is not None:
            lines.append(
                f"- Shape: bucket={usage.get('bucket_size')} "
                f"pad={usage.get('pad_tokens')} cold={usage.get('cold_shape')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Query model", annotations=READ_ONLY)
async def query_model(
    instance_id: str, prompt: str, max_tokens: int = 256, stats: bool = True
) -> str:
    """Send a chat completion to the served model.

    With stats=True (the default) the reply carries the token counts the server
    already reports in `usage`, the wall time, and the tok/s they imply.

    That rate is END-TO-END for a single request: it includes prefill and the
    HTTP round trip, so it reads lower than decode throughput and is not the
    number to benchmark on. Prefer get_metrics, whose decode gauge is what both
    benchmark reports compare against.

    It is also meaningless on a cold engine. MEASURED 2026-08-21: the first
    request after init took 18.06 s against 4.50 s warm for the same prompt,
    because XLA compiles per shape bucket. Warm up before believing a number.
    """
    try:
        endpoint = await get_endpoint(instance_id)
        if not endpoint.startswith("📡"):
            return endpoint
        base = endpoint.strip("📡 `")
        async with httpx.AsyncClient(timeout=120) as client:
            started = time.perf_counter()
            response = await client.post(
                f"{base}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
            )
            wall = time.perf_counter() - started
        body = response.json()
        choice = body["choices"][0]
        text = choice["message"]["content"]
        if not stats:
            return text
        usage = body.get("usage") or {}
        completion = usage.get("completion_tokens") or 0
        rate = f"{completion / wall:.2f} tok/s" if completion and wall > 0 else "n/a"
        return (
            f"{text}\n\n---\n"
            f"📡 {completion} completion + {usage.get('prompt_tokens', 0)} prompt tokens "
            f"in {wall:.2f}s — {rate} end-to-end "
            f"(finish: {choice.get('finish_reason')})"
        )
    except Exception as exc:
        return _error(exc)


def _parse_prom(text: str) -> tuple[dict, dict, str | None]:
    """Split a Prometheus exposition into (samples, precision labels, model).

    Pure and offline so the tests can pin it without a served endpoint — the
    rendering below is the part that needs one, the parsing is not.
    """
    samples, precision, served_model = {}, {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        series, _, raw = line.rpartition(" ")
        try:
            value = float(raw)
        except ValueError:
            continue
        name, _, labels = series.partition("{")
        tags = {}
        for part in labels.rstrip("}").split(","):
            key, sep, val = part.partition("=")
            if sep:
                tags[key] = val.strip('"')
        # The model label rides on every series and is identical across them, so
        # it is noise per row — but it is the only record of WHICH checkpoint
        # produced these numbers, so hoist it out rather than drop it. A
        # get_metrics transcript has to be checkable against MODEL_NAME.
        served_model = tags.pop("model", None) or served_model
        if name == "tpu_jax_precision_info":
            precision = tags
            continue
        rendered = ",".join(f'{k}="{v}"' for k, v in sorted(tags.items()))
        samples[f"{name}{{{rendered}}}" if rendered else name] = value
    return samples, precision, served_model


@mcp.tool(title="Get serving metrics", annotations=READ_ONLY)
async def get_metrics(instance_id: str) -> str:
    """Read the serving process's Prometheus metrics, including the decode gauge.

    `tpu_jax_decode_tokens_per_second` is the like-for-like throughput figure
    both of this rig's benchmark reports compare on, because it times decode
    alone. query_model's rate also carries prefill and the HTTP round trip, so
    the two do not agree and the gauge is the one to quote.

    The gauge describes the LAST request only — it is not an average, and it is
    worthless straight after init, when XLA is still compiling. The counters
    below it are cumulative since the process started.
    """
    try:
        endpoint = await get_endpoint(instance_id)
        if not endpoint.startswith("📡"):
            return endpoint
        base = endpoint.strip("📡 `")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(base.replace("/v1", "/metrics"))
        if response.status_code != 200:
            return f"❌ /metrics returned {response.status_code} — is {SERVICE_NAME} serving?"

        samples, precision, served_model = _parse_prom(response.text)
        if not samples:
            return "❌ /metrics returned 200 but exposed no samples."

        lines = [f"### Serving metrics on `{instance_id}`", ""]
        if served_model:
            lines += [f"Served checkpoint: `{served_model}`", ""]
        if precision.get("build_id"):
            lines += [f"Build id: `{precision['build_id']}` "
                      f"(rig `{precision.get('rig', '?')}`)", ""]
        if precision:
            # The resolved dtypes, not the requested ones. On Turing these differ
            # from every sibling rig's, and "auto" hides which way it went.
            kv, want = precision.get("kv_cache_dtype"), precision.get("kv_cache_requested")
            kv_shown = f"{kv} (requested `{want}`)" if want and want != kv else kv
            lines += [
                "| Precision | Resolved |",
                "| --- | --- |",
                f"| Compute dtype | **{precision.get('compute_dtype')}** |",
                f"| Quant mode | **{precision.get('quant_mode')}** |",
                f"| KV cache dtype | **{kv_shown}** |",
                f"| PLE bits | {precision.get('ple_bits')} |",
                f"| int8 lm_head | {precision.get('int8_lm_head')} |",
                f"| Pre-Ampere (no bf16/fp8) | {precision.get('pre_ampere')} |",
                "",
            ]
        lines += ["| Metric | Value |", "| --- | ---: |"]
        for key in sorted(samples):
            value = samples[key]
            shown = f"{value:.2f}" if value % 1 else f"{int(value)}"
            lines.append(f"| `{key}` | {shown} |")

        completions = samples.get("tpu_jax_completion_tokens_total", 0.0)
        latency = samples.get("tpu_jax_latency_seconds_sum", 0.0)
        decode_s = samples.get("tpu_jax_decode_seconds_total", 0.0)
        cold = samples.get("tpu_jax_cold_requests_total", 0.0)
        if completions and decode_s:
            # Decode time alone, so this is like-for-like with the gauge the
            # benchmark reports quote rather than a lower bound. The old figure
            # divided by TOTAL latency, which carries prefill and the HTTP round
            # trip and so could only ever understate decode.
            lines += [
                "",
                f"Cumulative decode: **{completions / decode_s:.2f} tok/s** over "
                f"{int(completions)} completion tokens in {decode_s:.1f}s of decode. "
                "This excludes prefill and HTTP, so it is comparable to the "
                "`tpu_jax_decode_tokens_per_second` gauge.",
            ]
            if cold:
                lines.append(
                    f"⚠️ {int(cold)} request(s) compiled a new shape and are "
                    "averaged in here. Cold requests measure several times slower; "
                    "warm up before quoting this."
                )
        elif completions and latency:
            lines += [
                "",
                f"Cumulative mean: **{completions / latency:.2f} tok/s** over "
                f"{int(completions)} completion tokens in {latency:.1f}s. That average "
                "includes every cold and warm request since start, so it is a "
                "lower bound on warm decode, not a measurement of it.",
            ]
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check G4dn quotas", annotations=READ_ONLY)
async def check_g4dn_quotas() -> str:
    """Report the On-Demand and Spot G instance vCPU quotas for the region.

    G4dn draws on the same 'Running On-Demand G and VT instances' quota as other
    G-family types, counted in vCPUs — a g4dn.xlarge needs 8.
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
        lines.append(f"`{INSTANCE_TYPE}` needs {_vcpu_count(INSTANCE_TYPE)} vCPUs.")
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Help and configuration", annotations=READ_ONLY)
async def get_help() -> str:
    """Show this rig's resolved configuration and the constraints that shape it."""
    return f"""### {RIG_NAME}

Serving `{MODEL_NAME}` with **PyTorch + transformers** on **EC2 G4dn** — an
x86_64 Intel host with an NVIDIA **T4** GPU (Turing, SM 7.5, 15360 MiB, 14.07 GiB usable).

| Setting | Value |
| --- | --- |
| Region | `{AWS_REGION}` |
| Instance type | `{INSTANCE_TYPE}` ({_gpu_count(INSTANCE_TYPE)}x T4, {_vcpu_count(INSTANCE_TYPE)} vCPU, {_host_memory_gb(INSTANCE_TYPE)} GiB host RAM) |
| AMI | x86_64 PyTorch DLAMI via `{DLAMI_SSM_PARAMETER}` |
| Installed on top | `{TORCH_PIP_SPEC}` (torch itself comes from the AMI) |
| dtype | `{DTYPE}` — recorded here, decided on the device |
| Max model len | `{MAX_MODEL_LEN}` (the static buffer, `--seq`) |
| Max concurrent seqs | `{MAX_NUM_SEQS}` |
| Service | `{SERVICE_NAME}` (systemd, not docker) |
| App dir | `{APP_DIR}` |
| Managed-by tag | `{MANAGED_BY}` |
| Root volume | {ROOT_VOLUME_GB} GB gp3 @ {ROOT_VOLUME_THROUGHPUT_MBPS} MiB/s, {ROOT_VOLUME_IOPS} IOPS |

**THIS RIG HAS SERVED NOTHING.** Forked from `gpu-pytorch-g5g-2b` on 2026-08-29
and retargeted from Graviton2 to x86_64. `benchmarks/` is empty. Every number in
this repo's G5g rigs was measured on a different host; `docs/INHERITED.md` is the
list of what carries and what does not.

**Torch comes from the AMI, never from pip.** That inverts the JAX siblings,
where the image supplies only a driver and pip supplies CUDA. Here the image
supplies torch already built for the GPU, so the AMI is a PyTorch DLAMI rather
than the base one, and `install.sh` probes for the interpreter that can import
torch instead of choosing one. Run `verify_gpu_arch` before anything else — it
prints `torch.cuda.get_arch_list()` next to the measured capability and executes
a real fp16 matmul, which is a stronger claim than any flag being accepted.

**Turing has no bf16 and no fp8.** `bfloat16` does not fail, it *emulates*
through fp32, which is worse than an error: correct numbers, quiet slowdown.
`resolve_compute_dtype()` reads the live compute capability and picks `float16`
below SM 8.0, so `DTYPE` above is a record of that policy, not the input to it.

**Nothing here compiles.** No `torch.compile`, so there is no compilation cache
to warm, persist, or sync — the knobs for that were dropped in the fork rather
than carried inert. Warm-up still matters: the first call pays autotune and
allocator growth, so warm up at the shape you intend to measure.

**One GPU is used.** The engine is single-device. `g4dn.12xlarge` carries four
T4s and `g4dn.metal` eight; on those every GPU but the first idles.

Order of operations: `create_g4dn_instance` → `get_install_progress` →
`verify_gpu_arch` → `deploy_torch_server` → `get_torch_logs` → `verify_model_health`.
"""


if __name__ == "__main__":
    mcp.run()
