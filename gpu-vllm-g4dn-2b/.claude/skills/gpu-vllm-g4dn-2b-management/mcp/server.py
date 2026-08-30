"""EC2 G4dn (x86_64 + NVIDIA T4, Turing SM 7.5) lifecycle and inference MCP server.

Like the Inf2 sibling, this uses boto3 rather than shelling out to the AWS CLI so
it works with profiles, environment credentials, IAM roles, and SSO-backed
credential processes, and uses Systems Manager Run Command for remote
administration — no inbound SSH rule or private key.

WHY THIS RIG EXISTS: it isolates the Turing blocker from the aarch64 one.

`gpu-vllm-g5g-2b` hit TWO independent problems at once and had to solve both to
serve a token, which is why it is the hardest rig in this tree:

  1. PACKAGING. G5g needs aarch64 AND SM 7.5 together, and no published CUDA
     artifact has both -- `vllm/vllm-openai`'s arm64 manifest is compiled for
     `8.0 8.7 8.9 9.0 10.0 11.0 12.0` while the amd64 manifest of the SAME TAG
     carries 7.5, and the Dockerfile sets no `+PTX` so nothing JITs. Cost: a
     ~67-minute from-source build, a CUDA toolkit the DLAMI does not ship, a Rust
     toolchain, and a prebuilt AMI to maintain.
  2. TURING SHARED MEMORY. Gemma 4's heterogeneous head dims (sliding 256,
     global 512) force vLLM's Triton attention path, whose 512-wide tile wants
     ~96 KiB of shared memory per block against Turing's hard 64 KiB. Cost: an
     unlanded patch to `triton_unified_attention.py`.

**G4dn is x86_64 and SM 7.5. Problem 1 disappears; problem 2 does not.**

x86_64 means this rig wants the amd64 manifest, and that manifest is exactly the
one carrying 7.5. So there is no build, no toolkit, no Rust and no AMI to bake --
the same deletion `gpu-vllm-g6-2b` got by moving to Ada. But the GPU is still
Turing, so the shared-memory ceiling is untouched, and unlike on Ada it is not a
narrow margin to check: 65,536 < 98,304 is arithmetic, and the same silicon
already produced the failure on the G5g sibling.

THAT COMBINATION IS NEW IN THIS TREE, and it changes how the patch is delivered.
The G5g rig can only get the patch in by compiling vLLM. Here nothing compiles,
so the patch is applied to one pure-Python file inside the published image and a
derived image is built `FROM` it -- seconds, not 67 minutes. `patch_triton_turing.py`
does that, and refuses loudly rather than no-op'ing, because an unpatched file
behind a patched tag surfaces ~10 minutes later as an OutOfResources at engine
start. See docs/turing-shared-memory.md.

NOTHING HERE HAS BEEN RUN ON HARDWARE. Forked from `gpu-vllm-g6-2b` 2026-08-29.
"""

import asyncio
import base64
import gzip
import hashlib
import logging
import os
import time
from pathlib import Path

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
RIG_NAME = "gpu-vllm-g4dn-2b"

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
SERVICE_NAME = os.getenv("SERVICE_NAME", "vllm-g4dn")
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))

# TURING HAS NO bf16 AND NO fp8 DATAPATH, and this is where a fork from the G6
# rig goes wrong in both directions at once. That rig runs Ada (SM 8.9), which
# has both, and defaults to `bfloat16` BECAUSE THE CHECKPOINT IS BF16 -- inheriting
# that here is the single most likely copy-paste error in this file.
#
# float16 is what Turing actually executes. MEASURED on the G5g sibling 2026-08-12:
# bfloat16 does NOT hard-fail -- PyTorch upconverts, and vLLM logs
# `Casting torch.bfloat16 to torch.float16` and proceeds. That makes it a silent
# cost rather than an error, which is worse, and it is why the wrong reason
# ("bf16 fails on Turing") is dangerous: someone tests torch, sees it pass, and
# deletes the guard.
#
# NOTE the consequence, and it is NOT the G6 rig's: the checkpoint ships bf16 and
# the compute dtype here is float16, so vLLM converts every weight at load. On
# this exact silicon `gpu-jax-g4dn-2b`'s lineage MEASURED that class of mismatch
# at 54% of decode -- but that was a JAX loader holding bf16 and converting at
# every USE, per step. vLLM converts ONCE at load. Do not quote the 54% here.
DTYPE = os.getenv("DTYPE", "float16")
# `auto` follows the compute dtype, i.e. float16. fp8 KV is NOT reachable on
# Turing -- there is no datapath, unlike on the G6 sibling where it is available
# and merely unused. int8 is the only low-precision compute win this part has
# (HARDWARE.md), and vLLM's KV cache does not expose it as a dtype.
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto")
# LEFT EMPTY, meaning "let vLLM choose", and the reason is MEASURED rather than
# stylistic. On the G5g sibling 2026-08-12: vLLM v0.27 does not recognize
# VLLM_ATTENTION_BACKEND AT ALL (`Unknown vLLM environment variable detected`),
# and it FORCES `TRITON_ATTN` for Gemma 4 regardless, because only FA4 and Triton
# handle heterogeneous head dims and FA4 is unavailable. Setting it did nothing.
#
# So the backend is not the knob. The tile size inside the forced Triton kernel
# is, and that is what patch_triton_turing.py changes. HARDWARE.md still says the
# Turing-capable backend is XFORMERS; that line predates this measurement.
ATTENTION_BACKEND = os.getenv("VLLM_ATTENTION_BACKEND", "")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.90")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "8"))

# THE PUBLISHED IMAGE IS THE BASE, BUT NOT WHAT SERVES. This is the one place
# this rig differs from BOTH siblings, and collapsing it in either direction is
# wrong:
#
#   * `gpu-vllm-g6-2b` has ONE image name, because SM 8.9 needs no patch at all.
#   * `gpu-vllm-g5g-2b` has a stock/built split, where the built side is 67
#     minutes of compilation for SM 7.5 kernels the arm64 image lacks.
#
# Here the published amd64 image ALREADY has the SM 7.5 kernels, so nothing is
# compiled -- but the Triton attention tile still has to shrink, and that is a
# pure-Python edit. VLLM_PATCHED_IMAGE is a derived tag built `FROM` the stock
# one with a single COPY. Seconds.
#
# v0.28.0, AND THE TAG THIS FORK INHERITED DOES NOT EXIST.
#
# `gpu-vllm-g6-2b` pins `vllm/vllm-openai:v0.27.2rc0` and this rig copied it.
# CHECKED 2026-08-29 against both registries: that tag is a 404 on Docker Hub and
# there is no such git tag in vllm-project/vllm -- the published sequence goes
# v0.27.0, v0.27.0rc1/rc2, v0.27.1, then v0.28.0rc1/rc2, v0.28.0, v0.28.1rc0.
# **Cloud-init would have died at `docker pull`, on the very first stage.**
#
# Note how it survived: the version-floor test asserted the tag was not v0.27.1
# and not v0.26, which a nonexistent tag passes trivially. A floor test that
# never checks the artifact EXISTS is the same shape as this tree's standing rule
# that an accepted flag is not evidence -- one level up, in the test itself.
#
# The FLOOR is unaffected and still holds: v0.26.0 dies with
# AmbiguousGlobalPerLayerAttributeError against current transformers because
# Gemma 4's head_dim is per-layer, and that is a constraint of the MODEL, so it
# applies on every chip in this tree. v0.28.0 clears any v0.27.x floor outright.
#
# v0.28.0 rather than `nightly`: nightly is a MOVING tag, and this rig renders
# deterministic user data on purpose. Checked anyway -- for
# triton_unified_attention.py, `main` is BYTE-IDENTICAL to v0.28.0 and the patch
# applies to both, so nightly would buy nothing and cost reproducibility.
#
# VERIFIED on the real v0.28.0 manifest 2026-08-29, which is this rig's premise:
#   linux/amd64  TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0   <- 7.5 present
#   linux/arm64  TORCH_CUDA_ARCH_LIST=8.0 8.7 8.9 9.0 10.0 11.0 12.0  <- 7.5 absent
VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:v0.28.0")
VLLM_PATCHED_IMAGE = os.getenv("VLLM_PATCHED_IMAGE", "vllm-openai:v0.28.0-sm75-patched")

# Headroom under Turing's hard 65,536 B per block. The tile arithmetic in the
# patch does not account for the kernel's accumulators, so budgeting the full
# limit still overflows. This is the value the G5g sibling ran with.
TURING_SMEM_BUDGET = int(os.getenv("TURING_SMEM_BUDGET", "60000"))

# AWS publishes the x86_64 GPU DLAMI as a public SSM parameter. Prefer it: it is
# single-valued and authoritative, where a describe-images name filter is a fuzzy
# match that can select the wrong image and still boot.
#
# BASE, not PyTorch: this rig serves from a docker image that carries its own
# CUDA and torch, so a PyTorch DLAMI is GBs of image whose entire content is
# unused. The DLAMI only has to supply the NVIDIA driver and docker.
#
# `/latest/` in a DLAMI path is only the newest build WITHIN one PyTorch-and-
# Ubuntu line, and AWS eventually stops rebuilding a line -- `gpu-vllm-g5g-2b`
# pinned `pytorch-2.7-ubuntu-22.04`, which froze at a 2026-05-02 image while
# reading as "track latest". This path is the same one `gpu-jax-g6-2b` VERIFIED
# ON HARDWARE 2026-08-28 (driver 595.91.07, Ubuntu 26.04), and the same one
# `gpu-jax-g4dn-2b` targets -- but it has NOT been verified on a G4dn.
DLAMI_SSM_PARAMETER = os.getenv(
    "DLAMI_SSM_PARAMETER",
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-26.04/latest/ami-id",
)
# Fallback only. It must NOT require "ARM64 AMI" contiguously the way the G5g
# rig's pattern does -- the base images are named "Deep Learning Base OSS Nvidia
# Driver GPU AMI (Ubuntu 26.04)". Changing the SSM path without changing this
# filter in the same commit is a revert that reports success: the fallback
# quietly resolves the old image.
DLAMI_NAME = os.getenv("DLAMI_NAME", "Deep Learning Base OSS Nvidia Driver GPU*Ubuntu*")

MANAGED_BY = RIG_NAME

# G4dn topology: (GPUs, host RAM GiB). Verified against the AWS G4dn product page.
#
# TWO TRAPS, both shared with the G6 table and neither shared with G5g:
#   * Host RAM is DOUBLE its g5g namesake at every suffix -- g4dn.xlarge has
#     16 GiB where g5g.xlarge had 8. The G5g rig's xlarge rejection does NOT
#     carry, and g4dn.xlarge is a reasonable default where g5g.xlarge was not.
#   * GPU count is NOT MONOTONIC in the size: 12xlarge has 4, 16xlarge has 1,
#     metal has 8. Never infer it from the suffix; a wrong tensor-parallel size
#     fails at engine start.
_G4DN_SIZES = {
    "g4dn.xlarge": (1, 16),
    "g4dn.2xlarge": (1, 32),
    "g4dn.4xlarge": (1, 64),
    "g4dn.8xlarge": (1, 128),
    "g4dn.12xlarge": (4, 192),
    "g4dn.16xlarge": (1, 256),
    "g4dn.metal": (8, 384),
}

# Host RAM below this needs a swapfile before the model will load. MEASURED
# 2026-08-13 on g5g.xlarge (7,757 MiB usable), i.e. ON A SIBLING: loading E2B
# fails with
#
#   RuntimeError: unable to mmap 10246621918 bytes from model.safetensors:
#   Cannot allocate memory (12)
#
# and the container crash-loops on it. The failure is the *mapping*, not
# residency -- the kernel declines to map a 10.2 GB file against 7.5 GiB of RAM
# and no swap, before a single page is faulted in.
#
# STRICTLY LESS THAN 16, AND THAT DIFFERS DELIBERATELY FROM `gpu-jax-g4dn-2b`,
# WHICH GATES AT-OR-BELOW 16 ON THE SAME INSTANCE TYPE. That rig OOMs at exactly
# 16 GiB, but in `quantize_ple_table`, which upcasts a 4.70 GB PLE table to
# float32 while the whole tree is resident. That is a property of ITS loader.
# vLLM has no equivalent step, and the G5g rig MEASURED that a 16 GiB host needs
# no swapfile. Do not "harmonise" these two gates: they encode different
# failures, and copying the JAX one here would provision swap nothing needs.
#
# Consequence: NO G4dn SIZE TRIPS THIS GATE, so the swap block never renders and
# is UNTESTED CODE. Kept because the threshold is a claim about the checkpoint
# (~10.2 GB to map), not about the host, and a larger checkpoint would need it.
# The G5g rig learned this expensively: a `mkswap -q` busybox flag that
# util-linux rejects sat latent in exactly this block for as long as only one
# unlaunched size rendered it, then killed cloud-init before any log existed.
_SWAP_BELOW_HOST_RAM_GB = 16
_SWAP_GB = 16

# Root volume. Three multi-GB reads land here: the published image pull, the
# derived-image build, and the 10.2 GB checkpoint download the loader then reads
# back.
#
# Throughput is set EXPLICITLY because gp3's default is 125 MiB/s. PORTED FROM
# `gpu-jax-g5g-2b`, where it is MEASURED rather than assumed: two unrelated load
# stages both landed on ~125 MB/s, which is the signature of a volume ceiling
# rather than CPU or network, and raising it took that rig's read_shards
# 73.5s -> 24.7s, a clean 3.0x on the same read. UNMEASURED HERE.
#
# 500 MiB/s is ~4x baseline and still under g4dn.xlarge's own EBS ceiling, so the
# smaller sizes stay instance-bound rather than volume-bound. gp3 also requires
# throughput <= IOPS * 0.25, and THAT RULE IS ENFORCED AT run-instances TIME --
# violating it fails a LAUNCH, not merely a disk.
#
# get_deployment_config and create_g4dn_instance both render from these. On the
# G5g rig they disagreed (200 printed, 100 launched), which is how a manual
# reproduction quietly fails to reproduce.
ROOT_VOLUME_GB = int(os.getenv("ROOT_VOLUME_GB", "100"))
ROOT_VOLUME_THROUGHPUT_MBPS = int(os.getenv("ROOT_VOLUME_THROUGHPUT_MBPS", "500"))
ROOT_VOLUME_IOPS = int(os.getenv("ROOT_VOLUME_IOPS", "6000"))

# The patch script travels with server.py so the installed skill copy under
# ~/.claude/skills/<skill>/mcp/ can still launch an instance. refresh_skill.py
# copies it for exactly that reason, and a test asserts the copies match.
_PATCH_SCRIPT = "patch_triton_turing.py"
# Must equal patch_triton_turing.MARKER. Duplicated rather than imported so that
# server.py stays importable when only it is installed, and pinned by
# `test_the_marker_matches_the_patch_script` -- if these drift, the in-image
# verification below looks for a string the patch never writes and every launch
# fails at `patch-verified-in-image` with a message blaming the COPY target.
_PATCH_MARKER = "gpu-vllm-g4dn-2b: Turing shared-memory clamp"


def _patch_source() -> str:
    """Read the patch script from beside this file.

    Resolved next to server.py rather than from a fixed path so the rig root and
    the skill snapshot both work. `deploy` from a stale snapshot shipping a stale
    patch is the failure this mirrors from `gpu-jax-g5g-2b`, which lost a whole
    measure-and-conclude cycle to it -- hence `_patch_digest` below and the
    build-id comparison in verify_triton_patch.
    """
    return (Path(__file__).resolve().parent / _PATCH_SCRIPT).read_text()


def _patch_digest() -> str:
    """Short content hash of the patch script, stamped onto the instance.

    Hashes the SOURCE, not the base64 blob: the blob is what gets shipped, so
    hashing it and then shipping the hash inside it would be circular.
    """
    return hashlib.sha256(_patch_source().encode()).hexdigest()[:12]


def _patch_b64() -> str:
    """gzip+base64 the patch script for user data.

    User data is capped at 16 KB and the script is ~9 KB of deliberately
    comment-heavy Python; gzip takes it to roughly a third of that. mtime=0 so
    the encoding is DETERMINISTIC -- an unchanged rig must render byte-identical
    user data, or `get_deployment_config` stops being a reproducible artifact and
    every launch looks like a change.
    """
    raw = _patch_source().encode()
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode()


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
    return _G4DN_SIZES.get(instance_type, (0, 0))[0]


def _host_memory_gb(instance_type: str) -> int:
    return _G4DN_SIZES.get(instance_type, (0, 0))[1]


def _needs_swap(instance_type: str) -> bool:
    """True when host RAM is too small to mmap the checkpoint without swap."""
    return 0 < _host_memory_gb(instance_type) < _SWAP_BELOW_HOST_RAM_GB


def _validate_instance_type(instance_type: str) -> None:
    """Only the size list is enforced. Small hosts are supported, not rejected --
    `_user_data` provisions a swapfile for them (see `_SWAP_BELOW_HOST_RAM_GB`)."""
    if not _is_g4dn(instance_type):
        raise ValueError(f"instance_type must be one of {', '.join(sorted(_G4DN_SIZES))}")


def _tensor_parallel_size(instance_type: str) -> int:
    return _gpu_count(instance_type)


def _vcpu_count(instance_type: str) -> int:
    """vCPUs for a G4dn size.

    Deliberately NOT `RAM // 2`. That shortcut was right for G5g (2 GiB per
    vCPU) and is wrong here: G4dn is 4 GiB per vCPU, so it would report double.
    Same trap the G6 rig hit at its own fork.
    """
    return _G4DN_SIZES.get(instance_type, (0, 0))[1] // 4


async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)


def _serve_flags(model: str, instance_type: str) -> str:
    """vLLM flags for Turing (SM 7.5).

    Deliberately unlike BOTH the five `gpu-vllm-l4-*` artifact rigs and the G6
    sibling, which target SM 8.9 and hardcode `--dtype bfloat16` (and often
    `--kv-cache-dtype fp8`). Neither datapath exists on this part.
    """
    return (
        f"--model {model} --host 0.0.0.0 --port 8000 "
        f"--dtype {DTYPE} --kv-cache-dtype {KV_CACHE_DTYPE} "
        f"--tensor-parallel-size {_tensor_parallel_size(instance_type)} "
        f"--gpu-memory-utilization {GPU_MEMORY_UTILIZATION} "
        f"--max-model-len {MAX_MODEL_LEN} --max-num-seqs {MAX_NUM_SEQS}"
    )


def _user_data(model: str, instance_type: str) -> str:
    """Render idempotent cloud-init: pull, PATCH, build a derived image, serve.

    THE PATCH STAGE IS WHAT MAKES THIS RIG DIFFERENT FROM BOTH SIBLINGS, and the
    shape of it follows directly from which of the two G5g problems survives here.

    `gpu-vllm-g6-2b` has no patch stage: SM 8.9 raises the per-block shared memory
    to ~99 KiB, which clears Triton's ~96 KiB tile. `gpu-vllm-g5g-2b` has no patch
    STAGE either -- its patch is applied by hand to a source tree during a
    ~67-minute compile, because the arm64 image has no SM 7.5 kernels to start
    from and the whole engine has to be rebuilt anyway.

    G4dn is the case where the kernels are already there and only one Python file
    is wrong. So: pull the published image, extract that file, patch it, and
    `docker build` a derived tag with a single COPY. No compiler, no CUDA toolkit,
    no Rust, and no source checkout.

    FIVE THINGS HERE ARE DELIBERATE AND EACH ONE IS A FAILURE THIS TREE HAS SEEN:

    * The in-image path of the file is RESOLVED, never hardcoded. It sits under a
      site-packages directory whose python version moves with the image tag, and
      a hardcoded path silently misses -- so the module is imported and asked
      where it lives.
    * `set -e` plus an explicit `|| exit 1` on the patch. A failed patch must kill
      cloud-init, because the alternative is a derived tag that serves unpatched
      and fails ~10 minutes later as `OutOfResources: shared memory` at engine
      start, having reported success the whole way.
    * The patched file is VERIFIED INSIDE THE BUILT IMAGE, not just on disk. The
      COPY destination is computed, and a wrong destination produces an image
      that builds cleanly and contains an unpatched module.
    * Stage markers, so a stalled launch is attributable. The G5g rig's build was
      the longest phase of a deploy and the only untimed one, and a spot
      reclamation mid-install left no record of which step was running.
    * PATCH_SHA is stamped on the box so `verify_triton_patch` can compare what is
      running against the local script. `gpu-jax-g5g-2b` lost a full
      measure-and-conclude cycle on 2026-08-24 to a deploy that shipped a stale
      payload and reported success.
    """
    _validate_instance_type(instance_type)

    flags = _serve_flags(model, instance_type)
    app_dir = f"/opt/{SERVICE_NAME}"

    swap = ""
    if _needs_swap(instance_type):
        # Dead code on every current G4dn size (all have >= 16 GiB host RAM).
        # NOTE `mkswap` takes no `-q`: that is a busybox flag which util-linux
        # rejects with `invalid option -- 'q'`, and under `set -e` it killed
        # cloud-init on the G5g rig BEFORE anything logged. Do not re-add.
        swap = f"""if ! swapon --show --noheadings 2>/dev/null | grep -q /swapfile; then
  fallocate -l {_SWAP_GB}G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
"""

    # Only exported when explicitly pinned. Passing an empty
    # VLLM_ATTENTION_BACKEND is NOT the same as not setting it -- vLLM sees the
    # variable and can treat "" as a selection rather than falling through to its
    # own dispatch, which is exactly the silent-misconfiguration shape this tree
    # keeps getting bitten by.
    backend_env = f'  -e VLLM_ATTENTION_BACKEND={ATTENTION_BACKEND} \\\n' if ATTENTION_BACKEND else ""

    # WHY THE MODULE-PATH RESOLVE IS SENTINEL-FENCED, kept out here on purpose:
    # user data is capped at 16 KB by EC2 and comments inside the template are
    # billed against it, so the reasoning lives in Python and only the code ships.
    #
    # MEASURED 2026-08-30 -- this killed the rig's FIRST EVER LAUNCH at this exact
    # line. Importing vllm emits INFO logging on STDOUT, and the resolve container
    # deliberately runs WITHOUT `--gpus` (the GPU belongs to the serving container),
    # so the import finds no driver and announces it:
    #
    #   INFO [importing.py:53] Triton is installed but 0 active driver(s) found
    #                          (expected 1). Disabling Triton to prevent runtime errors.
    #   INFO [importing.py:88] Triton not installed or not compatible; ...
    #   /usr/local/lib/python3.12/dist-packages/vllm/.../triton_unified_attention.py
    #
    # A bare $(...) captures all three lines, so the next command became
    # `cat "<2 log lines><path>"` and died with `File name too long`. Because the
    # resolve container never has a GPU, those lines are ALWAYS emitted: this could
    # never have worked on any tag. It stayed latent only because the rig had never
    # been launched -- the arithmetic in CLAUDE.md was all checked, the plumbing
    # was not, and no test could see it because the failure needs a real docker.
    #
    # The sentinel beats `tail -n 1`: logging can interleave on either side of the
    # print, so line position is not a safe key. The `case` guard exists because an
    # empty TARGET makes `cat` write an EMPTY file, which the patch script then
    # refuses for the WRONG REASON -- reporting upstream restructuring when the
    # actual fault is local plumbing.
    return f"""#!/usr/bin/env bash
set -euxo pipefail
{swap}systemctl enable --now docker
mkdir -p {app_dir}

_T0=$(date +%s)
stage() {{ echo "[stage] $1 +$(( $(date +%s) - _T0 ))s"; }}

stage image-pull-start
docker pull {VLLM_IMAGE}
stage image-pull-done

# ---------------------------------------------------------------------------
# Turing shared-memory patch. Gemma 4's global head_dim of 512 forces vLLM's
# Triton attention path, whose tile wants 98,304 B of shared memory per block
# against Turing's hard 65,536. There is no flag; the tile has to shrink.
# ---------------------------------------------------------------------------
echo '{_patch_b64()}' | base64 -d | gunzip > {app_dir}/{_PATCH_SCRIPT}
echo '{_patch_digest()}' > {app_dir}/PATCH_SHA

# Resolved, never hardcoded. Sentinel-fenced: importing vllm logs to STDOUT.
TARGET=$(docker run --rm --entrypoint python3 {VLLM_IMAGE} -c \\
  'import vllm.v1.attention.ops.triton_unified_attention as m; print("__TARGET__" + m.__file__)' \\
  | sed -n 's/^__TARGET__//p' | tail -n 1)
echo "triton_unified_attention.py resolves to $TARGET"
case "$TARGET" in
  /*/triton_unified_attention.py) ;;
  *) echo "FATAL: module path did not resolve (got: '$TARGET')" >&2; exit 1 ;;
esac
stage patch-resolve

docker run --rm --entrypoint cat {VLLM_IMAGE} "$TARGET" > {app_dir}/triton_unified_attention.py

# A failed patch MUST kill cloud-init. Serving unpatched behind a patched tag
# fails ~10 minutes later at engine start, having reported success throughout.
TURING_SMEM_BUDGET={TURING_SMEM_BUDGET} python3 {app_dir}/{_PATCH_SCRIPT} \\
  {app_dir}/triton_unified_attention.py || {{
    echo "FATAL: could not patch triton_unified_attention.py — refusing to serve" >&2
    exit 1
  }}
stage patch-applied

cat >{app_dir}/Dockerfile <<DOCKERFILE
FROM {VLLM_IMAGE}
COPY triton_unified_attention.py $TARGET
DOCKERFILE
docker build -t {VLLM_PATCHED_IMAGE} {app_dir}
stage image-build-done

# Verify the patch INSIDE the built image. A wrong COPY destination builds
# cleanly and leaves the module unpatched.
docker run --rm --entrypoint python3 {VLLM_PATCHED_IMAGE} -c "import vllm.v1.attention.ops.triton_unified_attention as m, sys; sys.exit(0 if '{_PATCH_MARKER}' in open(m.__file__).read() else 1)" || {{
    echo "FATAL: {VLLM_PATCHED_IMAGE} does not contain the clamp — wrong COPY target?" >&2
    exit 1
  }}
stage patch-verified-in-image

cat >{app_dir}/start.sh <<'SCRIPT'
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
  {VLLM_PATCHED_IMAGE} \\
  {flags}
SCRIPT
chmod 700 {app_dir}/start.sh
{app_dir}/start.sh
stage serving-started
touch {app_dir}/INSTALL_DONE
stage INSTALL_COMPLETE
"""


async def _resolve_ami(ec2=None) -> str:
    """Resolve the x86_64 **GPU** DLAMI for this region.

    Two things have to hold and they are separate: the image must be **x86_64**
    (G4dn is an Intel host — the arm64 DLAMI the G5g rig requires cannot boot one
    at all, and this is the axis that flipped at the fork), and it must carry the
    NVIDIA driver.

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
                # The command id is carried into the message on purpose: on a
                # timeout the command is still RUNNING on the box, and it is the
                # only handle left for finding out what happened.
                raise RuntimeError(f"SSM {result['Status']} (command-id {command_id}): {output}")
            return output
        await asyncio.sleep(2)
    raise TimeoutError(f"SSM command did not finish in {timeout}s (command-id {command_id})")


def _error(exc: Exception) -> str:
    logger.exception("tool failed: %s", exc)
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
    iam_instance_profile: str = "g4dn-vllm-instance-profile",
    spot: bool = True,
) -> str:
    """Return cloud-init and an AWS CLI launch command without changing AWS.

    The rendered command provisions the SAME root volume the create tool does --
    both read ROOT_VOLUME_*. On the G5g rig they disagreed (200 printed, 100
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
        return (
            f"### EC2 G4dn deployment ({instance_type}, {_gpu_count(instance_type)}x T4)\n\n```bash\n"
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
            f"Pulls `{VLLM_IMAGE}` (the published amd64 manifest carries SM 7.5, so nothing "
            f"is compiled), patches Triton's attention tile for Turing's 64 KiB shared-memory "
            f"ceiling, and serves from the derived tag `{VLLM_PATCHED_IMAGE}`.\n\n"
            f"Patch script `{_PATCH_SCRIPT}` sha `{_patch_digest()}`.\n\n"
            f"Serving flags: `{_serve_flags(model_name, instance_type)}`\n"
            f"dtype is `{DTYPE}`, NOT bfloat16 — Turing has no bf16 datapath. It does not "
            f"error, it upconverts, which is why the wrong setting is silent here."
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
    """Launch one tagged G4dn instance using the latest regional x86_64 GPU DLAMI.

    Cloud-init pulls the published image, applies the Turing shared-memory patch,
    builds a derived tag and serves from it. There is no from-source build and no
    `serving` mode: SM 7.5 is already in the published amd64 image.

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
        return (
            f"✅ Launching `{instance_id}` ({instance_type}, {market}, "
            f"{_gpu_count(instance_type)}x T4) in `{AWS_REGION}`.\n"
            f"AMI: `{args['ImageId']}`\n"
            f"Patch sha: `{_patch_digest()}` → `{VLLM_PATCHED_IMAGE}`\n"
            "Follow with get_install_progress, then verify_gpu_arch, "
            "verify_triton_patch and verify_model_health."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List managed G4dn instances", annotations=READ_ONLY)
async def list_g4dn_instances() -> str:
    """List instances tagged ManagedBy=gpu-vllm-g4dn-2b."""
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
    """Terminate a managed instance. Permanent.

    Cheap here, unlike on `gpu-vllm-g5g-2b`: that rig loses a locally COMPILED
    SM 7.5 image with the root volume, ~67 minutes of build, which is why it
    maintains a prebuilt AMI and weighs stop against terminate. Nothing is
    compiled here — a relaunch costs an image pull, a seconds-long derived build
    and the model download. Do not import that rig's reasoning.
    """
    try:
        await _call(_client("ec2").terminate_instances, InstanceIds=[instance_id])
        return (
            f"🗑️ Terminating `{instance_id}`. Only the image pull and model cache are lost; "
            "nothing was compiled."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify GPU compute capability and kernel coverage", annotations=READ_ONLY)
async def verify_gpu_arch(instance_id: str, image: str = "") -> str:
    """Measure whether an image's CUDA kernels actually cover this GPU.

    Reports the device's compute capability, the arch list the image's torch was
    compiled for, and one real matmul on the device. A config flag being accepted
    proves nothing; a kernel either launches or it does not.

    ON THIS RIG SM 7.5 IS EXPECTED TO BE PRESENT, and that is the half of the G5g
    problem that G4dn deletes: the published amd64 manifest is compiled for
    `7.5 8.0 8.6 8.9 9.0 10.0 12.0` while the arm64 manifest of the SAME TAG is
    not. Same image, same tag, different answer purely because the host is Intel.

    It probes with float16, not bfloat16. That is deliberate and is the opposite
    of the G6 sibling: Turing has NO bf16 datapath, so a bf16 probe would pass by
    upconversion and tell you nothing about what actually executes.

    PASSING HERE DOES NOT MEAN THE MODEL WILL SERVE. The arch gap and the Triton
    shared-memory ceiling are independent problems, and only the first one is
    gone. Run verify_triton_patch too.
    """
    target = image or VLLM_IMAGE
    probe = (
        "import torch;"
        "print('device:', torch.cuda.get_device_name(0));"
        "print('capability:', torch.cuda.get_device_capability(0));"
        "print('torch arch list:', torch.cuda.get_arch_list());"
        "p = torch.cuda.get_device_properties(0);"
        "print('shared mem per block (static):', p.shared_memory_per_block);"
        "x = torch.randn(256, 256, device='cuda', dtype=torch.float16);"
        "print('fp16 matmul ok:', float((x @ x).sum()) == float((x @ x).sum()))"
    )
    command = (
        f"nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true; "
        f'docker run --rm --gpus all --entrypoint python3 {target} -c "{probe}" 2>&1 || true'
    )
    try:
        output = await _ssm(instance_id, command, timeout=600)
        if "no kernel image is available" in output:
            verdict = (
                "\n\n❌ `no kernel image is available for execution on the device`. On x86_64 "
                "this should NOT happen — the published amd64 manifest carries SM 7.5. Suspect "
                "an arm64 manifest pulled by mistake, a wrong tag, or a non-T4 GPU."
            )
        elif "7.5" not in output and "arch list" in output:
            verdict = (
                "\n\n❌ No SM 7.5 in the arch list. That contradicts the published amd64 "
                "manifest and is the one result that would invalidate this rig's premise; "
                "check the image tag and the host architecture before concluding it."
            )
        elif "fp16 matmul ok: True" in output:
            verdict = (
                "\n\n✅ SM 7.5 is in the arch list and a real float16 matmul executed.\n"
                "**This is only half the question.** The Triton shared-memory ceiling is "
                "independent of kernel coverage — run verify_triton_patch."
            )
        else:
            verdict = ""
        return f"### GPU arch probe on `{instance_id}` ({target})\n\n```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify the Turing shared-memory patch", annotations=READ_ONLY)
async def verify_triton_patch(instance_id: str) -> str:
    """Confirm the derived image is patched and the container is running IT.

    THIS IS THIS RIG'S CENTRAL CHECK, and it exists because three separate things
    can each be true while the deployment still fails at engine start:

      1. The patched image was never built (the patch refused, cloud-init died).
      2. The image was built but the COPY landed somewhere the module is not
         imported from, so it contains an unpatched file and builds cleanly.
      3. The image is patched and the CONTAINER IS RUNNING THE STOCK TAG, because
         a hand-run `docker run` used the wrong name.

    It also compares the sha of the patch script ON THE BOX against the local
    one. `gpu-jax-g5g-2b` lost a full measure-and-conclude cycle on 2026-08-24 to
    a deploy that shipped a stale payload and reported success; that is what
    STALE PATCH below is for.
    """
    local = _patch_digest()
    command = (
        f"cat /opt/{SERVICE_NAME}/PATCH_SHA 2>/dev/null || echo 'PATCH_SHA MISSING'; "
        f"echo '--- image ---'; "
        f"docker image inspect {VLLM_PATCHED_IMAGE} >/dev/null 2>&1 "
        f"&& echo 'PATCHED IMAGE PRESENT' || echo 'PATCHED IMAGE ABSENT'; "
        f"echo '--- module in image ---'; "
        f"docker run --rm --entrypoint python3 {VLLM_PATCHED_IMAGE} -c "
        f"\"import vllm.v1.attention.ops.triton_unified_attention as m; "
        f"t=open(m.__file__).read(); print(m.__file__); "
        f"print('CLAMP PRESENT' if '{_PATCH_MARKER}' in t else 'CLAMP ABSENT')\" 2>&1 || true; "
        f"echo '--- running container ---'; "
        f"docker inspect -f '{{{{.Config.Image}}}}' {SERVICE_NAME} 2>/dev/null "
        f"|| echo 'container {SERVICE_NAME} not running'"
    )
    try:
        output = await _ssm(instance_id, command, timeout=600)
        notes = []
        if "PATCH_SHA MISSING" in output:
            notes.append("❌ No PATCH_SHA on the box — cloud-init never reached the patch stage.")
        elif local not in output:
            notes.append(
                f"❌ STALE PATCH — the box was provisioned with a different "
                f"`{_PATCH_SCRIPT}` than the local `{local}`. Relaunch; the patch "
                "ships in user data, so an existing instance cannot be updated in place."
            )
        if "PATCHED IMAGE ABSENT" in output:
            notes.append(f"❌ `{VLLM_PATCHED_IMAGE}` was never built.")
        if "CLAMP ABSENT" in output:
            notes.append(
                "❌ The image exists but the module is UNPATCHED — the COPY destination did "
                "not match where the module is imported from. Engine start will die with "
                "`OutOfResources: shared memory, Required: 98304, Hardware limit: 65536`."
            )
        if f"container {SERVICE_NAME} not running" in output:
            notes.append("⚠️ No serving container. Nothing is being served yet.")
        elif VLLM_IMAGE in output and VLLM_PATCHED_IMAGE not in output:
            notes.append(
                f"❌ The container is running the STOCK tag `{VLLM_IMAGE}`, not "
                f"`{VLLM_PATCHED_IMAGE}`. It is unpatched whatever the image says."
            )
        if not notes:
            notes.append(
                f"✅ Patch `{local}` applied, present in `{VLLM_PATCHED_IMAGE}`, and that is "
                "the tag the container is running."
            )
        return (
            f"### Turing patch on `{instance_id}` (local sha `{local}`)\n\n"
            f"```\n{output}\n```\n\n" + "\n".join(notes)
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get bootstrap progress", annotations=READ_ONLY)
async def get_install_progress(instance_id: str, tail: int = 40) -> str:
    """Tail cloud-init, the image pull, the patch and the derived build.

    PORTED FROM `gpu-jax-g5g-2b`, and the ported part is the cloud-init VERDICT
    rather than the tail. Cloud-init can die BEFORE it writes anything this tool
    would normally read, and the naive rendering of that is `IN PROGRESS` plus
    `no log yet`, forever — which is also exactly what a healthy slow launch
    looks like. A dead bootstrap and a running one must not share a rendering.

    That is not hypothetical: on that rig a `mkswap -q` busybox flag that
    util-linux rejects failed under `set -e` in the swap block, which renders
    FIRST, so cloud-init died before the install log existed and the instance sat
    there looking busy. It cost a launch rather than a minute.

    The patch stage is called out separately because it is the one that fails on
    a vLLM upgrade: `patch_triton_turing.py` refuses when upstream restructures
    the file, and that refusal is a DELIBERATE hard stop, not a bug.
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
                "\n\n✅ Image pulled, patched, derived image built, container started. Next: "
                "verify_gpu_arch, verify_triton_patch, then verify_model_health. Note the "
                "container starting is NOT the model being ready — vLLM still has to download "
                "and load the checkpoint."
            )
        elif "could not patch triton_unified_attention.py" in output:
            verdict = (
                "\n\n❌ THE PATCH REFUSED TO APPLY, and cloud-init stopped on purpose. "
                f"Upstream has restructured the file in `{VLLM_IMAGE}`. Read the "
                "`patch_triton_turing` error above — it names the anchor or identifier it "
                "could not find. Serving unpatched is not an option on Turing: engine start "
                "dies with `OutOfResources: shared memory`. This is NOT a slow launch."
            )
        elif "does not contain the clamp" in output:
            verdict = (
                "\n\n❌ The derived image built but does not contain the clamp — the COPY "
                "destination did not match the module's import path. This is NOT a slow launch."
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
            verdict = (
                "\n\n⏳ Pulling, patching and building. Nothing is COMPILED here — if this "
                "runs for tens of minutes it is the image pull or the checkpoint download, "
                "not a vLLM build."
            )
        return f"```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get vLLM logs", annotations=READ_ONLY)
async def get_vllm_logs(instance_id: str, tail: int = 100) -> str:
    """Tail the vLLM container log.

    The line to look for on this rig, if the patch did not take:
    `triton.runtime.errors.OutOfResources: shared memory, Required: 98304,
    Hardware limit: 65536`.
    """
    try:
        tail = max(1, min(tail, 5000))
        return f"```\n{await _ssm(instance_id, f'docker logs --tail {tail} {SERVICE_NAME} 2>&1')}\n```"
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
                return f"📡 `http://{host}:{VLLM_PORT}/v1`"
        return f"❌ `{instance_id}` not found in `{AWS_REGION}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify model health", annotations=READ_ONLY)
async def verify_model_health(instance_id: str) -> str:
    """Check /health and confirm the served model returns a usable completion.

    Uses /v1/chat/completions: raw /v1/completions skips the chat template and is
    unreliable on `-it` models, so an empty body there is not evidence either way.

    DO NOT relax this into an emptiness check. MEASURED on `gpu-vllm-g5g-2b`
    2026-08-12: a broken deploy on this lineage answered `': ok: ok: ok…'` —
    degenerate repetition, 16 tokens, non-empty, and completely wrong. The
    degeneracy guard below is crude on purpose; it is there because a body full
    of garbage passes `text.strip()`.
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
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        words = text.split()
        # A reply that is one token repeated is the documented broken-deploy
        # signature here, not a healthy short answer.
        degenerate = len(words) >= 4 and len(set(words)) == 1
        ok = health.status_code == 200 and bool(text.strip()) and not degenerate
        status = "✅" if ok else "❌"
        note = (
            "\n⚠️ DEGENERATE REPETITION — this is the broken-deploy signature on this "
            "lineage, not a healthy answer. Check verify_triton_patch and get_vllm_logs."
            if degenerate
            else ""
        )
        return (
            f"{status} health={health.status_code} tokens="
            f"{body.get('usage', {}).get('completion_tokens', 0)} reply={text!r}{note}"
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


@mcp.tool(title="Check G4dn quotas", annotations=READ_ONLY)
async def check_g4dn_quotas() -> str:
    """Report the On-Demand and Spot G instance vCPU quotas for the region.

    G4dn draws on the same 'Running On-Demand G and VT instances' quota as other
    G-family types, counted in vCPUs — a g4dn.xlarge needs 4.

    Quota is not capacity. MEASURED 2026-08-27/28 elsewhere in this family:
    G-family spot in us-east-1 was exhausted in every AZ but one with quota to
    spare, and the one AZ with capacity was the MOST EXPENSIVE. Price is not a
    proxy for availability. Check `aws ec2 get-spot-placement-scores` before
    launching in a loop.
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
        # G5g had 2 GiB of RAM per vCPU so `RAM // 2` was its vCPU count. G4dn
        # has 4 GiB per vCPU, so that shortcut silently DOUBLES every figure.
        lines.append(f"`{INSTANCE_TYPE}` needs {_vcpu_count(INSTANCE_TYPE)} vCPUs.")
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Help and configuration", annotations=READ_ONLY)
async def get_help() -> str:
    """Show this rig's resolved configuration and the constraints that shape it."""
    return f"""### {RIG_NAME}

Serving `{MODEL_NAME}` with vLLM on **EC2 G4dn** — x86_64 host, NVIDIA **T4** GPU
(Turing, SM 7.5, 16 GB nominal / 15360 MiB measured on the T4G sibling).

| Setting | Value |
| --- | --- |
| Region | `{AWS_REGION}` |
| Instance type | `{INSTANCE_TYPE}` ({_gpu_count(INSTANCE_TYPE)}x T4, {_host_memory_gb(INSTANCE_TYPE)} GiB RAM, {_vcpu_count(INSTANCE_TYPE)} vCPU) |
| Tensor parallel | `{_tensor_parallel_size(INSTANCE_TYPE)}` |
| dtype | `{DTYPE}` |
| KV cache dtype | `{KV_CACHE_DTYPE}` |
| Attention backend | `{ATTENTION_BACKEND or "(unpinned — vLLM forces TRITON_ATTN for this model anyway)"}` |
| Base image | `{VLLM_IMAGE}` (published; nothing is compiled) |
| Served image | `{VLLM_PATCHED_IMAGE}` (derived, Turing tile clamp) |
| Patch sha | `{_patch_digest()}` |
| Shared-memory budget | {TURING_SMEM_BUDGET} B of Turing's 65536 hard limit |
| Root volume | {ROOT_VOLUME_GB} GB gp3 @ {ROOT_VOLUME_THROUGHPUT_MBPS} MiB/s, {ROOT_VOLUME_IOPS} IOPS |
| Managed-by tag | `{MANAGED_BY}` |

**THIS RIG HAS SERVED NOTHING.** Forked from `gpu-vllm-g6-2b` 2026-08-29. Every
claim below is arithmetic or inherited; none of it is measured here.

**It isolates ONE of the two problems `gpu-vllm-g5g-2b` has.**

| | packaging: SM 7.5 in the image? | Turing 64 KiB shared memory |
| --- | --- | --- |
| `gpu-vllm-g5g-2b` (aarch64, SM 7.5) | ❌ arm64 manifest lacks it → ~67-min build | ❌ needs the tile clamp |
| `gpu-vllm-g6-2b` (x86_64, SM 8.9) | ✅ | ✅ ~99 KiB clears the tile (unverified) |
| **this rig** (x86_64, SM 7.5) | **✅ amd64 manifest carries 7.5** | **❌ needs the tile clamp** |

So there is **no build, no CUDA toolkit, no Rust and no AMI to bake** — but the
Triton clamp is still mandatory, and here it is a pure-Python edit plus a
one-line derived image rather than a from-source compile. `verify_triton_patch`
is the check that matters.

**Turing has neither bf16 nor fp8.** `--dtype bfloat16` and `--kv-cache-dtype
fp8` are the defaults across the L4 artifact rigs and the G6 sibling, and both
are wrong here. bfloat16 does not error — PyTorch upconverts and vLLM logs
`Casting torch.bfloat16 to torch.float16` — so the wrong value is silent.

Order: `check_g4dn_quotas` → `create_g4dn_instance` → `get_install_progress`
→ `verify_gpu_arch` → **`verify_triton_patch`** → `verify_model_health`.
"""


if __name__ == "__main__":
    mcp.run()
