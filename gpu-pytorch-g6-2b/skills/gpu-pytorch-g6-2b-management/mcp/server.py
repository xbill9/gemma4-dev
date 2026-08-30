"""EC2 G6 (x86_64 + NVIDIA L4) lifecycle and inference MCP server, PyTorch path.

Serves `google/gemma-4-E2B-it` through stock HF transformers on an L4 (Ada,
SM 8.9), under systemd -- not docker, and not vLLM.

Like the Inf2 sibling, this uses boto3 rather than shelling out to the AWS CLI so
it works with profiles, environment credentials, IAM roles, and SSO-backed
credential processes, and uses Systems Manager Run Command for remote
administration -- no inbound SSH rule or private key.

Why this rig exists, next to `gpu-jax-g6-2b` and `gpu-vllm-g6-2b`:

Identical silicon, different runtime, and the runtime is the whole point. This
is the ordinary-PyTorch control: no custom model port, no XLA, no compiled
static shapes, no from-source build. `AutoModelForCausalLM.from_pretrained` and
an HF KV cache. It is the number every other runtime on this chip should be
asked to beat, and the cheapest one to stand up.

Three things it does NOT inherit from the T4G PyTorch sibling, all of which are
consequences of Ada rather than of PyTorch:

  * **bfloat16 is native.** `resolve_compute_dtype` reads the live compute
    capability and picks bfloat16 at SM >= 8.0. On the T4G it picks float16,
    because asking Turing for bfloat16 does not raise -- CUDA emulates through
    fp32 and most of decode silently disappears into conversion.
  * **The DLAMI's torch is a convenience, not a requirement.** On Turing,
    upstream PyPI wheels omit sm_75, so pip torch served on CPU. Ada is sm_89
    and upstream wheels carry it. The PyTorch DLAMI is used to keep a multi-GB
    download off a spot host's critical path -- a choice, not a constraint.
  * **Device memory is 23034 MiB, not 15360.** Do not carry a capacity verdict
    across from a T4G rig.

There is no quantization path here at all: no PLE table, no int8 LM head, no
W4A16. The checkpoint is served dense at the device's native 16-bit dtype, so
`tpu_jax_weight_bytes` is the full ~10.2 GB rather than the JAX rig's 6.155 GB.
That difference is the main reason a JAX-vs-PyTorch comparison on this chip is
NOT a like-for-like runtime comparison -- see CLAUDE.md.

Warm up before recording anything: the first request at a given shape pays
allocator growth and autotune, and cold decode measures several times slower.
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
RIG_NAME = "gpu-pytorch-g6-2b"

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
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "g6.2xlarge")
SERVICE_NAME = os.getenv("SERVICE_NAME", "torch-g6")
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")
TORCH_PORT = int(os.getenv("TORCH_PORT", "8000"))

# DTYPE is an OVERRIDE here, not the decision. `torch_openai_server.py`'s
# resolve_compute_dtype() reads the live device's compute capability and picks
# bfloat16 at SM >= 8.0, float16 below it. The L4 is SM 8.9, so this default
# agrees with what the device picks -- but the device still decides.
DTYPE = os.getenv("DTYPE", "bfloat16")
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto")
QUANT_MODE = os.getenv("QUANT_MODE", "bf16")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "1"))

# INERT ON THIS RIG -- kept only because tpu.env was forked wholesale. There is no
# quantisation path in HF transformers here: no PLE table, no int8 LM head, no
# W4A16. _serve_argv emits none of these and MUST NOT be made to. The block below
# is the JAX sibling's measurement, retained so nobody re-derives it as if it
# applied here.
#
# Quantisation knobs the JAX engine supports and that rig could not reach:
# _serve_argv never emitted them, so the only way to use them was to hand-edit the
# systemd unit. MEASURED 2026-08-23 on a T4G, same prompt, warm:
#
#   dense fp16                      9.26 GB weights, 12.4 tok/s
#   QAT w4a16, ple_bits=0           6.56 GB weights, OOMs on EVERY request
#   QAT w4a16, ple_bits=4           3.05 GB weights, 13.5 tok/s
#
# w4a16 alone does not work here: every request dies allocating 4.52 GiB beside
# 6.56 GB of resident weights. That allocation is NOT identified -- it is not a
# dequantised Linear (largest measures 0.005 GB dequantised); the nearest tensor
# is the unquantised PLE table at 4.698 GB, which is close but not equal. What is
# certain is the remedy: ple_bits=4 shrinks that table 4x (measured saving
# 3.505 GB) and the pair then works, so treat them as a pair on this chip.
PLE_BITS = int(os.getenv("PLE_BITS", "4"))
INT8_LM_HEAD = os.getenv("INT8_LM_HEAD", "1").lower() in ("1", "true", "yes")

# INERT ON THIS RIG. The figures below describe the JAX sibling's jax_engine:
# prefill temporaries LINEAR in the tokens admitted per pass, ~2.13 MB/token, so
# one-shot prefill OOMs at (free HBM / 2.13 MB) tokens. HF transformers does its
# own prefill and takes no such flag.
PREFILL_CHUNK_SIZE = os.getenv("PREFILL_CHUNK_SIZE", "")

# torch is NOT in this spec: it comes from the DLAMI. See the AMI notes below --
# on Ada that is an install-time preference rather than the hard requirement it
# was on the Turing sibling, where upstream wheels omit sm_75 entirely.
#
# TORCH_PYTHON_VERSION only seeds candidate interpreter NAMES. The bootstrap
# probes for the interpreter that can already `import torch` (the DLAMI ships a
# venv, not a system-wide install) and installs into THAT, because a wrong guess
# only surfaces as ModuleNotFoundError after the install reports success.
TORCH_PIP_SPEC = os.getenv("TORCH_PIP_SPEC", "transformers accelerate")
TORCH_PYTHON_VERSION = os.getenv("TORCH_PYTHON_VERSION", "3.14")

# Optional S3 backing for that cache. EMPTY BY DEFAULT, which is exactly the
# behaviour this rig has always had -- nothing below renders unless it is set.
#
# NOTE: the JAX sibling backed an XLA compilation cache to S3 here, restoring it
# at boot and pushing it on a timer. That whole feature was REMOVED at the
# PyTorch fork rather than carried across, because torch compiles nothing on this
# path: there is no XLA cache to persist, so the restore would sync an empty
# prefix and the timer would upload one, both reporting success forever. A
# feature that works correctly against a directory nothing writes to is the same
# silent-success class as a deploy that ships stale code.
#
# The expensive thing to re-fetch on this rig is the 10.2 GB CHECKPOINT, not a
# compilation cache. Caching the HF hub directory would be a real win and is
# deliberately NOT implemented here on the strength of that reasoning alone --
# it is unmeasured, and this rig does not ship unmeasured mechanisms.

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
# 500 MiB/s is ~4x baseline and still under g6.2xlarge's own EBS ceiling
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
# x86_64 wheels. Their install is non-fatal -- see install_runtime.
_PROFILING_REQUIREMENTS = ("xprof", "tensorboard")

# Where the serving payload lands on the instance.
APP_DIR = "/opt/torch-g6"
# AWS publishes the x86_64 GPU DLAMI as a public SSM parameter. Prefer it: it is
# single-valued and authoritative, where a describe-images name filter is a fuzzy
# match over a set that also contains DLAMIs with NO NVIDIA driver and DLAMIs for
# the wrong architecture entirely. Those can be the newest by CreationDate and
# boot perfectly well with no GPU — a failure that looks like a broken runtime
# rather than a wrong AMI.
# The BASE image -- driver only, no PyTorch. Two independent reasons, and the
# second one is why the previous parameter was quietly rotting.
#
# 1. This rig never uses the DLAMI's PyTorch. The bootstrap deliberately does not
#    install into it (it ships its own CUDA libraries and jax brings its own), so
#    a PyTorch DLAMI is multiple GB of image this rig exists to avoid.
# 2. `/latest/` in a DLAMI parameter path is only the latest build WITHIN that
#    PyTorch-version + Ubuntu-version line, and AWS freezes those lines. The old
#    pin (pytorch-2.7-ubuntu-22.04) resolved to an AMI built 2026-05-02 and will
#    never move again -- AWS stopped rebuilding 22.04 after PyTorch 2.7. It read
#    as "track latest" and was in fact a pin to a dead line.
#
# Ubuntu 26.04 rather than 24.04 because it ships **Python 3.14 as the system
# interpreter** (3.14.3, verified in the Ubuntu archive 2026-08-27), which is the
# version this rig wants -- so the deadsnakes PPA leaves the critical path
# entirely. 24.04 ships 3.12 and still needs it. Both carry a newer driver and a
# newer glibc than 22.04, which was sitting exactly ON xprof's manylinux_2_35
# floor.
DLAMI_SSM_PARAMETER = os.getenv(
    "DLAMI_SSM_PARAMETER",
    "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.13-ubuntu-26.04/latest/ami-id",
)
# Fallback only, and it had to change in the same edit as the SSM path above.
#
# The arch is NOT in an x86_64 DLAMI's name. Only the ARM64 images announce
# theirs ("Deep Learning ARM64 AMI ..."); the x86_64 build of the same line is
# just "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.13 (Ubuntu 26.04)".
# So the G5g rig's pattern, arch-substituted, matches NOTHING -- and a fallback
# that matches nothing is not a loud failure, it is a launch that quietly falls
# through to whatever the describe-images call returns next. The architecture is
# pinned by an explicit `architecture: x86_64` filter on the call instead, which
# is where that constraint belongs.
#
# It still requires "Nvidia Driver" in the name, because AWS also ships driverless
# CPU-inference DLAMIs that boot perfectly well and simply have no GPU -- which
# reads as a broken runtime rather than a wrong AMI.
DLAMI_NAME = os.getenv("DLAMI_NAME", "Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*")

MANAGED_BY = RIG_NAME

# G6 topology: (GPUs, host RAM GiB). VERIFIED 2026-08-29 against
# `ec2 describe-instance-types`, not read off a product page or carried over
# from the G5g sibling -- G6 has twice the host RAM at every suffix, has no
# `metal` size at all, and its multi-GPU sizes are NOT the ones G5g put there.
#
# Two traps in this table, both of which a G5g habit gets wrong:
#   * `g6.16xlarge` is SINGLE-GPU despite the suffix. On G5g, 16xlarge was the
#     multi-GPU size. Never infer GPU count from the size name here.
#   * The multi-GPU sizes are 12/24/48xlarge (4, 4, 8 L4s).
# Nothing shards across them regardless: the serving path is single-device.
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

# Host RAM at or below this gets a swapfile before the model will load.
#
# INHERITED THRESHOLD, NOT MEASURED ON G6. The number is carried from
# `gpu-pytorch-g5g-2b` deliberately, but almost every sentence justifying it
# there does NOT transfer, and the two reasons are worth separating:
#
#   * G6 has TWICE the host RAM of G5g at the same suffix. The G5g evidence is
#     about `g5g.xlarge` at 8 GiB and `g5g.2xlarge` at 16 GiB; here those are
#     `g6.xlarge` at 16 GiB and `g6.2xlarge` at 32 GiB. So a verdict about a
#     G5g size name must never be transferred onto the same G6 size name.
#   * The G5g 2xlarge OOM was `quantize_ple_table` upcasting a 4.70 GB PLE
#     table, which is a JAX-rig failure. THIS rig is HF transformers and has no
#     PLE path at all, so that pressure does not exist here.
#
# What DOES carry is the mapping failure, because it is a property of the
# checkpoint and the kernel rather than the runtime or the GPU: without swap the
# kernel refuses to mmap the 10.2 GB checkpoint at all --
#
#   RuntimeError: unable to mmap 10246621918 bytes from model.safetensors:
#   Cannot allocate memory (12)
#
# -- before a single page is faulted in, and systemd then crash-loops on it.
# 10.2 GB of weights against 16 GiB of host RAM is close enough that the gate
# stays inclusive, so `g6.xlarge` is the one G6 size that renders the swap
# block. It has NOT been launched here; per the G5g lesson that a code path
# which only renders for a size nobody launches is untested code, treat the
# swap block as unexercised on G6 until a `g6.xlarge` run says otherwise.
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


def _is_g6(instance_type: str) -> bool:
    return instance_type in _G6_SIZES


def _gpu_count(instance_type: str) -> int:
    return _G6_SIZES.get(instance_type, (0, 0))[0]


def _host_memory_gb(instance_type: str) -> int:
    return _G6_SIZES.get(instance_type, (0, 0))[1]


def _needs_swap(instance_type: str) -> bool:
    """True when host RAM is too small to load the checkpoint without swap.

    Two distinct pressures, both real, and the larger one decides:
      * g6.xlarge (8 GiB) cannot even mmap the 10.2 GB checkpoint.
      * g6.2xlarge (16 GiB) mmaps fine and then dies in quantize_ple_table,
        which needs >15 GiB of host RSS.
    """
    return 0 < _host_memory_gb(instance_type) <= _SWAP_AT_OR_BELOW_HOST_RAM_GB


def _validate_instance_type(instance_type: str) -> None:
    """Only the size list is enforced. Small hosts are supported, not rejected --
    `_user_data` provisions a swapfile for them (see `_SWAP_AT_OR_BELOW_HOST_RAM_GB`)."""
    if not _is_g6(instance_type):
        raise ValueError(f"instance_type must be one of {', '.join(sorted(_G6_SIZES))}")


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
    forward them to. `tpu.env` still carries the keys (QUANT_MODE, PLE_BITS,
    INT8_LM_HEAD, ...) because it was forked wholesale; they are inert here and
    must not be re-plumbed into this command on the strength of existing.

    --seq is the static buffer length and IS this rig's one real serving knob,
    so it comes off MAX_MODEL_LEN rather than the server's own 256 default.

    There is no tensor-parallel flag: the engine is single-device. On the
    multi-GPU sizes (12/24/48xlarge) the extra L4s idle, which _tensor_parallel_size()
    reports
    but nothing acts on yet.
    """
    return (
        f"--model {model} --host 0.0.0.0 --port {TORCH_PORT} "
        f"--seq {MAX_MODEL_LEN}"
    )


# The serving payload. These are this rig's own files, shipped to the instance
# over SSM rather than fetched from a registry — there is no published artifact
# for "our JAX Gemma 4 port", and cloning the monorepo would need credentials on
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
    """Render idempotent cloud-init that installs the serving runtime.

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
  # NOT `mkswap -q`: that is a busybox flag, and util-linux's mkswap (Ubuntu
  # 22.04) rejects it with `invalid option -- 'q'`. Under `set -e` that killed
  # cloud-init before install.sh was even written, so the box came up with an
  # empty /opt/jax-g6 and no install log at all. Latent until 2026-08-26, when
  # making the swap threshold inclusive rendered this block for g6.2xlarge --
  # the rig's DEFAULT size -- for the first time.
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

# jax >= 0.11 needs Python >= 3.12; the Ubuntu 22.04 DLAMI base ships 3.10.
# We install the newest stable line, not the floor -- see TORCH_PYTHON_VERSION.
install_runtime() {{
  export DEBIAN_FRONTEND=noninteractive
  # Three apt hazards, all of which fail the install AND hide it: INSTALL_DONE is
  # never touched, so get_install_progress reports "INSTALL IN PROGRESS"
  # indefinitely instead of surfacing an error.
  #
  # 1. DPkg::Lock::Timeout is not optional. unattended-upgrades runs on every
  #    fresh Ubuntu instance and holds the dpkg frontend lock for minutes;
  #    apt-get then exits 100 immediately rather than waiting. Observed
  #    2026-08-20 on a running G6.
  # 2. Acquire timeouts are not optional either, and cover a DIFFERENT failure:
  #    a wedged mirror, where apt holds the lists lock itself and nothing is
  #    blocking it. Without these, apt waits forever rather than erroring.
  # 3. The regional EC2 mirror can be broken while the canonical one is healthy,
  #    so a hard failure against it is worth one retry elsewhere -- see
  #    fallback_mirror.
  APT_OPTS="-o DPkg::Lock::Timeout=600 -o Acquire::Retries=3"
  APT_OPTS="$APT_OPTS -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"

  # MEASURED 2026-08-21 on i-08639f402a3c3e76b: us-east-1.ec2.ports.ubuntu.com
  # returned 503 over IPv4 and resolved to AAAA records only, on a host with no
  # IPv6 address and no IPv6 default route. apt-get update wedged for 12 minutes
  # at the very first step -- before deadsnakes, before jax. Repointing at
  # ports.ubuntu.com completed the same update in seconds.
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
  # reach for deadsnakes when it is not. On the Ubuntu 26.04 base image
  # python{TORCH_PYTHON_VERSION} IS the system python, so this drops a third-party PPA, an
  # `add-apt-repository`, and a second full `apt-get update` off the critical
  # path. On 22.04/24.04 -- reachable by overriding DLAMI_SSM_PARAMETER -- the
  # deadsnakes branch still runs, so the bootstrap is not tied to one base image.
  if command -v {py} >/dev/null 2>&1; then
    echo "{py} is already present at $(command -v {py}); skipping deadsnakes"
    # Still needed even when the interpreter ships with the distro: a source
    # build of any dependency without an x86_64 wheel needs the headers.
    apt_run install -y {py}-venv {py}-dev || apt_run install -y python3-venv python3-dev
  else
    apt_run install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt_run update -y
    apt_run install -y {py} {py}-venv {py}-dev
  fi
  stage python-{TORCH_PYTHON_VERSION}

  # THE INTERPRETER IS NOT A FREE CHOICE ON THIS RIG, and this is the one place
  # it differs structurally from the JAX sibling.
  #
  # There, `pip install jax[cuda13]` put the runtime into whichever interpreter
  # ran it, so any python3.x would do. Here **torch comes from the AMI**, and the
  # AWS DLAMI does not install it into the system interpreter -- it ships a venv.
  # Installing transformers into /usr/bin/python3.12 and pointing the unit there
  # gets `ModuleNotFoundError: No module named 'torch'` AFTER the install has
  # reported success, which is the same shape as the ExecStart hazard below.
  #
  # So: find the interpreter that can already import torch and install into THAT.
  # Probed rather than hardcoded, because the venv path moves between DLAMI
  # releases and a wrong guess is only visible at the first token.
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
    echo "This rig gets torch from the DLAMI, never from pip: upstream x86_64" >&2
    # Single-quoted: backticks inside a double-quoted string are COMMAND
    # SUBSTITUTION, so the double-quoted version of this line would actually run
    # `pip install torch` -- installing the CPU-only wheel this message warns
    # against, from inside the error path that reports it.
    echo 'wheels omit sm_75, so "pip install torch" would silently serve on CPU.' >&2
    echo "Candidates probed:" >&2
    ls -d /opt/*/bin/python /opt/*/*/bin/python 2>/dev/null >&2 || true
    exit 1
  fi
  echo "$TORCH_PY" > {APP_DIR}/PYTHON_BIN
  stage torch-interpreter

  # PEP 668. Ubuntu marks its system interpreter externally-managed from 23.04
  # on, so a system-wide `pip install` fails outright with
  # `error: externally-managed-environment`. This is a single-purpose serving box
  # installing into the interpreter systemd will run, and the repo forbids
  # virtualenvs, so the override is the honest answer rather than a workaround.
  # Harmless (and ignored) inside the DLAMI venv, which is not marked.
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
  # without a second provisioning round. NOT fatal: xprof publishes x86_64
  # wheels with a manylinux_2_35 floor (24.04 is glibc 2.39, so there is
  # headroom) but a missing wheel must not cost the serve -- the whole reason
  # install.sh dies loudly elsewhere is that `set -e` used to kill it here.
  # The failure is announced rather than swallowed; grep the marker.
  if $PIP {profiling_reqs}; then
    stage profiling-deps
  else
    echo "[stage] profiling-deps FAILED -- xprof/tensorboard unavailable, serving unaffected" >&2
  fi
}}

# Assert the GPU is actually visible to TORCH before declaring the install done.
# A driverless x86_64 DLAMI boots fine on a G6 and would otherwise look healthy
# right up until the first token.
#
# This imported `jax` until 2026-08-29 -- inherited verbatim from the sibling,
# on a rig that installs no jax. It runs under `set -e`, so ModuleNotFoundError
# killed install.sh here, INSTALL_DONE was never written, and get_install_progress
# reported "INSTALL IN PROGRESS" forever. The rig had served nothing, so nothing
# had ever executed this line.
#
# The matmul is the point, not the import: `torch.cuda.is_available()` is true on
# a driver the arch list does not cover, and the failure then surfaces as a kernel
# error at the first token. The device's sm_XX must be in the arch list AND run.
verify_gpu() {{
  "$(cat {APP_DIR}/PYTHON_BIN)" - <<'PYCHECK'
import torch
print("torch", torch.__version__, "arch_list", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), (
    "no CUDA device visible to torch. On this rig that usually means the AMI is "
    "the AMI carries no NVIDIA driver, or a driverless CPU-inference DLAMI was "
    "resolved -- those boot fine and simply have no GPU."
)
major, minor = torch.cuda.get_device_capability(0)
print("device:", torch.cuda.get_device_name(0), f"sm_{{major}}{{minor}}")
# A cubin runs on any device of the SAME MAJOR with a minor >= the one it was
# built for (CUDA binary compatibility). An exact-match test is therefore wrong,
# and it is wrong in the expensive direction: it fails a GPU that works.
#
# MEASURED 2026-08-29 on this rig. The DLAMI's torch 2.13.0+cu130 carries
# ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120'] -- NO sm_89 -- while the L4
# is sm_89. The exact test aborted the install; fp16, bf16 and fp32 matmuls then
# all ran correctly off the sm_86 cubin, and torch.cuda.is_bf16_supported() was
# True. The assertion was inherited from the Turing sibling, where sm_75 IS
# present and an exact match happened to hold.
archs = torch.cuda.get_arch_list()
compat = [a for a in archs
          if a.startswith("sm_") and a[3:].isdigit()
          and int(a[3:-1]) == major and int(a[3:][-1]) <= minor]
assert compat, (
    f"torch has no cubin that can run on sm_{{major}}{{minor}}: {{archs}}. "
    "Needs one of the same major version with an equal or lower minor."
)
print("compatible cubins:", compat)
# The arch list is a claim; a launched kernel is the evidence. Check the dtype
# this rig will actually serve, not just fp16.
dtype = torch.float16 if (major, minor) < (8, 0) else torch.bfloat16
x = torch.ones((256, 256), device="cuda", dtype=dtype)
y = x @ x
torch.cuda.synchronize()
# Accumulate in fp32: 256**3 is far past float16's 65504 max, so a float16 sum
# overflows to inf and the check could never pass on ANY device.
print("matmul ok:", str(dtype), float(y.sum(dtype=torch.float32)) == 256.0 ** 3)
PYCHECK
}}

install_runtime
verify_gpu
stage gpu-verify
# Point the unit at the interpreter that actually has torch AND received the
# packages -- the one install_runtime probed, not one re-resolved through PATH.
#
# MEASURED 2026-08-19 on the sibling (on 3.12, but the hazard is not version-
# specific): the DLAMI may carry an interpreter of the same version under
# /usr/local/bin, which precedes /usr/bin on PATH, so a bare `command -v` and a
# hardcoded `ExecStart=/usr/bin/python3.12` can resolve to DIFFERENT binaries --
# the unit then dies with ModuleNotFoundError after the install has already
# reported success, because verify_gpu resolved through PATH too.
#
# On this rig the same hazard has a second edge: the interpreter holding torch is
# the DLAMI's venv, which is not on PATH as `python3.12` at all. Reading the path
# install_runtime recorded is what keeps install, verification and ExecStart on
# one interpreter instead of three.
#
# Rewritten here rather than in the unit template because the resolution can only
# happen after install_runtime has run. Still an absolute path, so systemd is
# happy and ExecStart never depends on the service's PATH.
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
Description=Gemma 4 E2B on NVIDIA L4 via PyTorch
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
    """Resolve the x86_64 **GPU** DLAMI for this region.

    Two things have to hold and they are separate:

      * **x86_64.** G6 is an x86_64 host. The G5g lineage this rig was retargeted
        from is arm64, and an arm64 AMI cannot boot here at all.
      * **The NVIDIA driver.** AWS also ships driverless CPU-inference DLAMIs.
        They boot perfectly well and simply have no GPU, which reads as a broken
        runtime rather than a wrong AMI.

    The SSM public parameter pins both. describe-images is the fallback, and it
    pins the architecture with an explicit `architecture` filter rather than with
    the name pattern -- the x86_64 images do not carry an arch in their names.
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


@mcp.tool(title="Generate G6 deployment configuration", annotations=READ_ONLY)
async def get_deployment_config(
    model_name: str = MODEL_NAME,
    instance_type: str = INSTANCE_TYPE,
    subnet_id: str = "<subnet-id>",
    security_group_id: str = "<security-group-id>",
    iam_instance_profile: str = "g6-jax-instance-profile",
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
            "Cloud-init installs the serving runtime only. The payload is this "
            "rig's own source and ships separately — run deploy_torch_server once "
            "get_install_progress reports INSTALL COMPLETE."
        )
        return (
            f"### EC2 G6 deployment ({instance_type}, {_gpu_count(instance_type)}x L4)\n\n```bash\n"
            f"# x86_64 *GPU* DLAMI. The SSM parameter pins both the architecture and the\n"
            f"# NVIDIA driver; a name filter can also match driverless x86_64 DLAMIs.\n"
            f"AMI_ID=$(aws ssm get-parameter --region {AWS_REGION} "
            f"--name {DLAMI_SSM_PARAMETER} "
            "--query 'Parameter.Value' --output text)\n"
            f'aws ec2 run-instances --region {AWS_REGION} --image-id "$AMI_ID" '
            f"--instance-type {instance_type} --subnet-id {subnet_id} "
            f"--security-group-ids {security_group_id} "
            f"--iam-instance-profile Name={iam_instance_profile} "
            f"{market}"
            # Rendered from the same constants create_g6_instance launches with.
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
    """Launch one tagged G6 instance using the latest regional x86_64 DLAMI.

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
            "Installing transformers/accelerate into the DLAMI's torch (no build). Follow with "
            "get_install_progress, then deploy_torch_server to ship the serving code."
        )
        # The AMI id belongs in the confirmation: AWS ships driverless x86_64
        # DLAMIs that boot perfectly well on a G6 and simply have no GPU, which
        # reads as a broken runtime rather than a wrong image. Recording which
        # image booted makes that one lookup instead of a guess.
        logger.info("launched instance=%s ami=%s type=%s market=%s",
                    instance_id, ami_id, instance_type, market)
        return (
            f"✅ Launching `{instance_id}` ({instance_type}, {market}, "
            f"{_gpu_count(instance_type)}x L4) in `{AWS_REGION}`.\n"
            f"AMI: `{ami_id}`\n{tail}"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List managed G6 instances", annotations=READ_ONLY)
async def list_g6_instances() -> str:
    """List instances tagged ManagedBy=gpu-pytorch-g6-2b."""
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

    It asks TORCH, not jax -- inverting the sibling, and not a cosmetic swap.
    Torch here comes from the AMI rather than from pip, so "does this GPU have
    kernels" is a question about the image that booted.

    **Do not read the arch list as an exact-match requirement.** A cubin runs on
    any device of the same major version with a minor >= its own. MEASURED
    2026-08-29: the DLAMI's torch 2.13.0+cu130 has no sm_89 at all, the L4 is
    sm_89, and every dtype ran correctly off the sm_86 cubin. The matmul is the
    evidence; the arch list is context.
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
        "import torch;"
        "print('torch:', torch.__version__);"
        "print('arch_list:', torch.cuda.get_arch_list());"
        "avail = torch.cuda.is_available();"
        "print('platform:', 'cuda' if avail else 'cpu');"
        "d = torch.device('cuda' if avail else 'cpu');"
        "print('device:', torch.cuda.get_device_name(0) if avail else 'none');"
        "cap = torch.cuda.get_device_capability(0) if avail else None;"
        "print('capability:', cap);"
        "dt = torch.float16 if (cap and cap < (8, 0)) else torch.bfloat16;"
        "print('compute_dtype:', str(dt).replace('torch.', ''));"
        "print('bf16_supported:', torch.cuda.is_bf16_supported() if avail else False);"
        "x = torch.ones((256, 256), dtype=dt, device=d);"
        "y = x @ x;"
        "print('matmul ok:', float(y.sum(dtype=torch.float32)) == 256.0 ** 3)"
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
                "(AWS ships driverless x86_64 DLAMIs that boot fine here), or the "
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
    """Ship this rig's JAX serving code to the instance and start the service.

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


@mcp.tool(title="Get runtime install progress", annotations=READ_ONLY)
async def get_install_progress(instance_id: str, tail: int = 40) -> str:
    """Tail the runtime install started by cloud-init, and cloud-init itself.

    This is a wheel install, not a build — minutes, not the hours the vLLM
    sibling needs. INSTALL COMPLETE means JAX imported *and* saw the GPU.

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
            verdict = "\n\n✅ Runtime installed and JAX saw the GPU. Next: deploy_torch_server."
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


@mcp.tool(title="Get JAX server logs", annotations=READ_ONLY)
async def get_torch_logs(instance_id: str, tail: int = 100) -> str:
    """Tail the JAX serving unit's journal.

    systemd, not docker: nothing is containerized on this rig.
    """
    try:
        tail = max(1, min(tail, 5000))
        command = f"journalctl -u {SERVICE_NAME} -n {tail} --no-pager 2>&1"
        return f"```\n{await _ssm(instance_id, command)}\n```"
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


@mcp.tool(title="Check G6 quotas", annotations=READ_ONLY)
async def check_g6_quotas() -> str:
    """Report the On-Demand and Spot G instance vCPU quotas for the region.

    G6 draws on the same 'Running On-Demand G and VT instances' quota as other
    G-family types, counted in vCPUs — a g6.2xlarge needs 8.
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

Serving `{MODEL_NAME}` with **stock PyTorch + HF transformers** on **EC2 G6** —
an **x86_64** host with an NVIDIA **L4** GPU (Ada, SM 8.9, 23034 MiB).

| Setting | Value |
| --- | --- |
| Region | `{AWS_REGION}` |
| Instance type | `{INSTANCE_TYPE}` ({_gpu_count(INSTANCE_TYPE)}x L4, {_host_memory_gb(INSTANCE_TYPE)} GiB RAM) |
| Runtime | `{TORCH_PIP_SPEC}` on the DLAMI's torch |
| dtype | `{DTYPE}` (chosen from the device's compute capability, not from here) |
| KV cache dtype | `{KV_CACHE_DTYPE}` |
| Max model len | `{MAX_MODEL_LEN}` |
| Service | `{SERVICE_NAME}` (systemd, not docker) |
| Managed-by tag | `{MANAGED_BY}` |
| Root volume | {ROOT_VOLUME_GB} GB gp3 @ {ROOT_VOLUME_THROUGHPUT_MBPS} MiB/s, {ROOT_VOLUME_IOPS} IOPS |

**Why this rig exists.** It is the *ordinary* runtime on this chip — no custom
model port, no XLA, no compiled static shapes, no from-source build. Just
`AutoModelForCausalLM` and an HF KV cache. That makes it the control the JAX and
vLLM rigs on identical silicon should be measured against, and by far the
cheapest of the three to stand up.

**The device picks the dtype, not this file.** `resolve_compute_dtype` reads the
live compute capability and selects bfloat16 at SM >= 8.0. The L4 is SM 8.9, so
it serves bf16 natively. The guard matters because the failure is silent in the
other direction: bfloat16 on a pre-Ampere GPU does not raise, it emulates
through fp32 — correct numbers, most of decode gone.

**No quantization exists on this path.** No PLE table, no int8 LM head, no
W4A16. The checkpoint is dense at 16-bit, so weight bytes are the full ~10.2 GB
against the JAX rig's 6.155 GB. **A JAX-vs-PyTorch number on this chip is
therefore NOT a like-for-like runtime comparison** — the JAX rig serves 40%
fewer weight bytes, and decode here is bandwidth-bound on exactly those bytes.

**Concurrency is not an axis.** `MAX_NUM_SEQS={MAX_NUM_SEQS}`; the server decodes one
sequence at a time. Sweep context and output length, not concurrency.

Quote the server's `tpu_jax_decode_tokens_per_second` gauge (see `get_metrics`),
not an end-to-end rate: end-to-end carries prefill and the HTTP round trip, so it
falls with context while decode does not. Warm up first — the first request at a
given shape pays allocator growth and autotune.

Order of operations: `create_g6_instance` → `get_install_progress` →
`verify_gpu_arch` → `deploy_torch_server` → `get_torch_logs` → `verify_model_health`.
"""

if __name__ == "__main__":
    mcp.run()
