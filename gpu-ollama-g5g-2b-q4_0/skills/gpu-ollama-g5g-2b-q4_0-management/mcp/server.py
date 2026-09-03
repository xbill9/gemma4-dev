"""EC2 G5g (Graviton2 + NVIDIA T4G) lifecycle and inference MCP server, Ollama path.

Serves `gemma4:e2b-it-qat` -- Ollama's repackaging of Google's QAT Q4_0 GGUF --
through the Ollama daemon.

Like every G5g sibling this uses boto3 rather than shelling out to the AWS CLI,
and Systems Manager Run Command for remote administration.

WHY THIS RIG EXISTS, NEXT TO `gpu-llamacpp-g5g-2b-q4_0`
-------------------------------------------------------
**They share an engine.** Ollama ships `lib/ollama/libllama.so` and the Gemma 4
graph both execute is upstream `src/models/gemma4.cpp`. Slot 2 names the front
end, not the decoder.

They are still two rigs, because four differences are visible to a benchmark:
the artifact is re-containered, the chat template is Ollama's own, the CUDA
variant is chosen by the daemon, and Ollama carries an experimental Go engine
that may or may not be what served your request. CLAUDE.md has the byte counts.

The practical asymmetry, and the honest reason to keep both: **Ollama hands you a
working aarch64 CUDA binary with native sm_75 SASS. llama.cpp makes you compile
one.** This rig therefore has no build step at all.

THINGS THIS RIG GETS WRONG SILENTLY, all pinned rather than trusted:

  1. **The CUDA variant is auto-selected by driver version.** MEASURED
     2026-09-02 by parsing the fatbins in the v0.33.2 arm64 bundle:
       cuda_v12  native CUBIN for sm_60,61,70,75,80,86,89,90,100,120  + PTX
       cuda_v13  **PTX ONLY, for every arch. No CUBIN at all.**
     A CUDA 13 driver therefore lands on a JIT path whose warm-up would be
     recorded as an engine property. OLLAMA_LLM_LIBRARY pins it.
  2. **OLLAMA_CONTEXT_LENGTH defaults to 0**, which means "4k/32k/256k based on
     VRAM". Two instance sizes would silently get two different contexts. Pinned.
  3. **OLLAMA_KEEP_ALIVE defaults to 5 minutes.** A benchmark that pauses longer
     than that pays a full model reload on its next request and records it as
     latency. Set negative (= forever) here.

There is no `/metrics` endpoint -- verified against `server/routes.go`, which
registers no such route. The decode gauge comes from `/api/generate`'s
`eval_count` / `eval_duration` instead, and `eval_duration` is a Go
`time.Duration`, i.e. **nanoseconds**.

NOTHING HAS BEEN MEASURED ON THIS RIG.
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
RIG_NAME = "gpu-ollama-g5g-2b-q4_0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(RIG_NAME)

MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", RIG_NAME)
mcp = FastMCP(MCP_SERVER_NAME)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")
# THE MODEL SLOT IS AN OLLAMA TAG, not a Hub id, and that is a weaker claim than
# the llama.cpp sibling's.
#
# `gemma4:e2b-it-qat` IS Google's QAT Q4_0 GGUF -- but RE-CONTAINERED. Verified
# 2026-09-02 by pulling the registry manifest and diffing against the Hub:
#
#                     Ollama blob        Google file        delta
#   model          3,349,514,112 B    3,349,516,256 B     -2,144
#   projector        986,833,312 B      986,833,664 B       -352
#   digests        3646b4c1...        fa401b55...         differ
#
# Ollama strips the metadata it owns itself (the GGUF's embedded Google canonical
# chat template, published 2026-07-09) and re-hashes. Tensors are presumed
# identical; the FILE provably is not. So slot 5 `-q4_0` is a claim about what
# Ollama says it packaged, not about a file this rig can hash.
#
# Do NOT "fix" this by pointing the rig at the Hub file through a Modelfile: that
# would make it a llama.cpp rig wearing an Ollama name, and the whole point of the
# pair is to measure the daemon's choices rather than bypass them.
MODEL_NAME = os.getenv("MODEL_NAME", "gemma4:e2b-it-qat")

# The plain `gemma4:e2b` tag is a DIFFERENT artifact -- a single 7.162 GB model
# layer with no projector, against this tag's 3.350 + 0.987 pair. Recorded so
# nobody "simplifies" the tag and silently changes the weights.
MODEL_TAG_ALTERNATIVES = ("gemma4:e2b",)

INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "g5g.2xlarge")
SERVICE_NAME = os.getenv("SERVICE_NAME", "ollama-g5g")
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")

# Ollama's own default is 11434. Pinned to 8000 to match every sibling rig, so
# one sweep command works against all of them.
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8000"))

# Compute dtype is NOT a knob here and there is no DTYPE constant on purpose.
# ggml picks per-kernel: Q4_0 weights are dequantised into the accumulator type
# the CUDA backend chooses for SM 7.5. The fp16-vs-bf16 question that dominates
# the JAX and PyTorch siblings does not arise -- Turing has no bf16 datapath and
# ggml never asks for one. Do not reintroduce DTYPE/QUANT_MODE/PLE_BITS/
# INT8_LM_HEAD: those name knobs of this repo's own JAX port, the PyTorch fork
# carried them inert for a month, and `tpu.env` keeping a key is not evidence
# that anything reads it.

# THERE IS NO N_GPU_LAYERS HERE, and its absence is a real difference rather than
# an oversight. llama-server takes `--n-gpu-layers`; Ollama decides the offload
# itself from its own VRAM estimate, and exposes only OLLAMA_GPU_OVERHEAD as a
# nudge. So the sibling's "partial offload" failure mode is not something this rig
# can prevent by configuration -- it can only be DETECTED, which is what
# verify_gpu_arch's size_vram check does.


# ---------------------------------------------------------------------------
# The runtime. THERE IS NO BUILD ON THIS RIG -- that is the asymmetry against the
# llama.cpp sibling and the main reason both exist.
# ---------------------------------------------------------------------------

# Ollama publishes a GENERIC linux-arm64 bundle that carries CUDA. Verified
# against v0.33.2 (2026-08-27): the tarball is 1543 MB -- LARGER than amd64 --
# and contains cuda_v12/ and cuda_v13/ trees with libggml-cuda.so, libcublas,
# libcublasLt and libcudart alongside libllama.so and libggml-cpu-armv8*.so.
#
# The jetpack5/jetpack6 bundles are SEPARATE and are for Jetson. The binary logs
# "jetpack not detected ... skipping" on a G5g and falls through to these two, so
# do not reach for a jetpack tarball on the strength of "arm64 + NVIDIA".
OLLAMA_VERSION = os.getenv("OLLAMA_VERSION", "v0.33.2")
OLLAMA_TARBALL_URL = os.getenv(
    "OLLAMA_TARBALL_URL",
    f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}"
    "/ollama-linux-arm64.tar.zst",
)

# PIN THE CUDA VARIANT. This is the single most important line in this file.
#
# MEASURED 2026-09-02 by walking the fatbin headers in both libggml-cuda.so and
# decompressing one entry of each kind to identify them (kind=1 is PTX text,
# kind=2 is a CUDA ELF with e_machine=190):
#
#   cuda_v12  CUBIN sm_60 61 70 75 80 86 89 90 100 120   + PTX sm_50..120
#   cuda_v13  **PTX ONLY** sm_75 80 86 87 89 90 100 103 110 120 121, zero CUBIN
#
# Ollama picks between them by DRIVER VERSION. The DLAMI this rig launches
# carries a CUDA 13 driver, so left alone it would select cuda_v13 and JIT every
# kernel from PTX at load. That works -- sm_75 PTX is present -- but the warm-up
# is then a property of the deployment, and it would be recorded as a property of
# the engine.
#
# cuda_v12 has native Turing SASS, so it is what this rig pins. **The llama.cpp
# sibling builds native sm_75 too; pinning here is what makes the pair
# comparable.** Unset this and the comparison silently measures codegen.
OLLAMA_LLM_LIBRARY = os.getenv("OLLAMA_LLM_LIBRARY", "cuda_v12")

# Context. OLLAMA'S DEFAULT IS 0, WHICH MEANS "4k/32k/256k BASED ON VRAM" --
# so an unpinned rig gets a different context on a different instance size, and
# nothing announces it. Pinned to match the llama.cpp sibling's --ctx-size.
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "4096"))

# Parallel request slots. Ollama's own default is already 1, which matches every
# sibling's single-stream baseline -- stated explicitly rather than inherited,
# because a default is not a decision until someone writes it down.
PARALLEL_SLOTS = int(os.getenv("PARALLEL_SLOTS", "1"))

# KEEP THE MODEL LOADED. Ollama unloads after 5 minutes idle by default, so a
# sweep that pauses between cells pays a full reload on the next request and
# records it as latency. Negative means forever.
#
# This has no analogue in any sibling: llama-server, vLLM and our own JAX and
# PyTorch servers all hold the weights for the life of the process. It is purely
# a property of the daemon, which is exactly the kind of thing this rig exists
# to surface.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "-1")

# Where the bundle is extracted. `bin/ollama` and `lib/ollama/` land under here.
APP_DIR = os.getenv("APP_DIR", "/opt/ollama-g5g")

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
# 500 MiB/s is ~4x baseline and still under g5g.2xlarge's own EBS ceiling
# ("up to" 4.75 Gbps ~= 593 MB/s), so the smaller sizes stay instance-bound
# rather than volume-bound. gp3 also requires throughput <= IOPS * 0.25 MiB/s,
# which 6000 IOPS satisfies with room to raise throughput to the 1000 cap.
ROOT_VOLUME_GB = int(os.getenv("ROOT_VOLUME_GB", "100"))
ROOT_VOLUME_THROUGHPUT_MBPS = int(os.getenv("ROOT_VOLUME_THROUGHPUT_MBPS", "500"))
ROOT_VOLUME_IOPS = int(os.getenv("ROOT_VOLUME_IOPS", "6000"))

# apt packages the build needs. Kept here rather than inline in the bootstrap so
# tests can assert on one list; a drifted pair is invisible until a launch.
#
# There is NO pip install anywhere on this rig. That is the structural difference
# from all three GPU siblings: JAX pip-installs its CUDA, PyTorch discovers a
# torch the AMI shipped, and this compiles a C++ binary. So the interpreter
# hazard that dominates the PyTorch sibling -- "the interpreter is discovered,
# not chosen" -- has no analogue here, and neither does PEP 668.
_BUILD_REQUIREMENTS = (
    "build-essential", "cmake", "ccache", "git",
    # libcurl is NOT optional. llama.cpp gates `--hf-repo`/`--hf-file` behind
    # LLAMA_CURL, and without the dev headers cmake silently builds a
    # llama-server that cannot download a model -- the failure surfaces as an
    # unrecognised argument at first start, long after the build reported
    # success. This rig has no other way to fetch the checkpoint.
    "libcurl4-openssl-dev",
)

# Where the serving payload lands on the instance.
# AWS publishes the ARM64 GPU DLAMI as a public SSM parameter. Prefer it: it is
# single-valued and authoritative, where a describe-images name filter is a fuzzy
# match over a set that also contains ARM64 DLAMIs with NO NVIDIA driver (built
# for Graviton CPU inference). Those match a loose "Deep Learning*ARM64*Ubuntu*"
# pattern, can be the newest by CreationDate, and boot perfectly well on a G5g
# with no GPU — a failure that looks like a broken container, not a wrong AMI.
# THE FULL DLAMI, NOT THE BASE DRIVER IMAGE -- and the reason INVERTED at the
# fork, which is worth stating because the value did not change and the argument
# for it did.
#
# The JAX and PyTorch siblings argue for the base image: they never use the
# DLAMI's PyTorch, so multiple GB of it is image they exist to avoid. **That
# reasoning does not survive the move to llama.cpp.** This rig compiles CUDA
# code, so it needs `nvcc`, and nvcc comes with the CUDA TOOLKIT -- which the
# full DLAMI carries and the driver-only base image does not.
#
# The failure if this is repointed at a base image is loud rather than subtle
# (install_runtime probes for nvcc and exits 1), and that is deliberate: the
# alternative -- cmake configuring without CUDA -- yields a llama-server that
# serves correctly on CPU. But loud still costs a launch, so do not "optimise"
# this back to the base image on the strength of a sibling's comment.
#
# apt's `nvidia-cuda-toolkit` is NOT an acceptable substitute. It installs a
# toolkit version unrelated to the driver on the box, which trades a clear
# failure for a silent-wrongness path.
#
# Still worth knowing, inherited and still true: `/latest/` in a DLAMI parameter
# path is only the latest build WITHIN that PyTorch-version + Ubuntu-version
# line, and AWS freezes those lines. The old pin (pytorch-2.7-ubuntu-22.04)
# resolved to an AMI built 2026-05-02 and will never move again. It read as
# "track latest" and was in fact a pin to a dead line -- re-check this parameter
# resolves to something recent before blaming a build failure on llama.cpp.
DLAMI_SSM_PARAMETER = os.getenv(
    "DLAMI_SSM_PARAMETER",
    "/aws/service/deeplearning/ami/arm64/oss-nvidia-driver-gpu-pytorch-2.12-ubuntu-24.04/latest/ami-id",
)
# Fallback only. It still requires the driver in the name so the driverless
# Graviton-CPU images cannot match -- but it no longer requires "ARM64 AMI"
# CONTIGUOUSLY, because the base images are named "Deep Learning ARM64 Base OSS
# Nvidia Driver GPU AMI (Ubuntu 26.04)". The old pattern did not match those at
# all, so leaving it alone while moving the SSM path would have made the fallback
# silently resolve the OLD PyTorch image -- a revert that looks like a success.
DLAMI_NAME = os.getenv("DLAMI_NAME", "Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch*Ubuntu*")

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
# INCLUSIVE. g5g.2xlarge has exactly 16 GiB, and `< 16` gave it no swapfile --
# so the rig provisioned swap for g5g.xlarge and skipped the one size where the
# quantization path actually needs it. MEASURED 2026-08-26: `--ple-bits 8` on a
# 2xlarge was OOM-killed by the kernel five times at 14.3 GB anon-rss under
# Restart=on-failure, because quantize_ple_table upcasts the 4.70 GB PLE table
# to float32 while the full parameter tree is still resident. Adding 16 GiB of
# swap by hand stopped the kills dead. The threshold, not the remedy, was the
# bug.
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


def _is_g5g(instance_type: str) -> bool:
    return instance_type in _G5G_SIZES


def _gpu_count(instance_type: str) -> int:
    return _G5G_SIZES.get(instance_type, (0, 0))[0]


def _host_memory_gb(instance_type: str) -> int:
    return _G5G_SIZES.get(instance_type, (0, 0))[1]


def _needs_swap(instance_type: str) -> bool:
    """True when host RAM is too small to load the checkpoint without swap.

    Two distinct pressures, both real, and the larger one decides:
      * g5g.xlarge (8 GiB) cannot even mmap the 10.2 GB checkpoint.
      * g5g.2xlarge (16 GiB) mmaps fine and then dies in quantize_ple_table,
        which needs >15 GiB of host RSS.
    """
    return 0 < _host_memory_gb(instance_type) <= _SWAP_AT_OR_BELOW_HOST_RAM_GB


def _validate_instance_type(instance_type: str) -> None:
    """Only the size list is enforced. Small hosts are supported, not rejected --
    `_user_data` provisions a swapfile for them (see `_SWAP_AT_OR_BELOW_HOST_RAM_GB`)."""
    if not _is_g5g(instance_type):
        raise ValueError(f"instance_type must be one of {', '.join(sorted(_G5G_SIZES))}")


def _tensor_parallel_size(instance_type: str) -> int:
    return _gpu_count(instance_type)


async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)


# Every OLLAMA_* variable the daemon actually defines, lifted from
# `envconfig/config.go` on 2026-09-02. A test asserts _serve_env() sets nothing
# outside this set.
#
# This is the structural analogue of the llama.cpp sibling's "only emit flags the
# server defines" rule, and it exists for the same reason: the PyTorch fork
# emitted three flags its server did not define and argparse exited 2 on every
# start. **Ollama is worse in one specific way** -- it does not reject an unknown
# environment variable, it IGNORES it. A typo'd OLLAMA_CONTEXT_LEN would serve
# happily at Ollama's VRAM-derived default and nothing anywhere would say so.
_KNOWN_OLLAMA_VARS = frozenset({
    "OLLAMA_AUTH", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_DEBUG",
    "OLLAMA_DEBUG_LOG_REQUESTS", "OLLAMA_EDITOR", "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_GO_TEMPLATE", "OLLAMA_GPU_OVERHEAD", "OLLAMA_HOST",
    "OLLAMA_IGPU_ENABLE", "OLLAMA_KEEP_ALIVE", "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_LLM_LIBRARY", "OLLAMA_LOAD_TIMEOUT", "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_MAX_QUEUE", "OLLAMA_MAX_TRANSFER_STREAMS", "OLLAMA_MODELS",
    "OLLAMA_NOHISTORY", "OLLAMA_NOPRUNE", "OLLAMA_NO_CLOUD",
    "OLLAMA_NUM_PARALLEL", "OLLAMA_ORIGINS", "OLLAMA_REMOTES",
    "OLLAMA_SCHED_SPREAD", "OLLAMA_VULKAN",
})


def _serve_env(model: str, instance_type: str) -> dict:
    """The daemon's configuration -- ALL OF IT.

    **`ollama serve` takes no model arguments.** There is no `--model`, no
    `--ctx-size`, no `--n-gpu-layers`; the unit's ExecStart is bare and every
    decision is an environment variable. That is why this rig has `_serve_env`
    where every sibling has `_serve_argv`, and it is not a cosmetic difference:
    an argv typo fails loudly at argparse, an env typo is silently ignored.

    Four of these are load-bearing and each overrides a default that would
    quietly damage a measurement:

    * `OLLAMA_LLM_LIBRARY` -- pins cuda_v12, which has native sm_75 CUBIN.
      Left unset, a CUDA 13 driver selects cuda_v13, which ships **PTX only for
      every architecture** and JITs at load.
    * `OLLAMA_CONTEXT_LENGTH` -- Ollama's default of 0 means "4k/32k/256k based
      on VRAM", so two instance sizes get two different contexts silently.
    * `OLLAMA_KEEP_ALIVE` -- default 5m. A sweep that pauses longer reloads the
      model and records it as latency. Negative means forever.
    * `OLLAMA_HOST` -- Ollama's own default is 127.0.0.1:11434; this rig binds
      0.0.0.0:8000 so one sweep command works against every sibling.

    `OLLAMA_MODELS` keeps the blob store on the root volume the launch sized for
    it, rather than in root's home.
    """
    return {
        "OLLAMA_HOST": f"0.0.0.0:{LLAMA_PORT}",
        "OLLAMA_MODELS": f"{APP_DIR}/models",
        "OLLAMA_LLM_LIBRARY": OLLAMA_LLM_LIBRARY,
        "OLLAMA_CONTEXT_LENGTH": str(CONTEXT_SIZE),
        "OLLAMA_NUM_PARALLEL": str(PARALLEL_SLOTS),
        "OLLAMA_KEEP_ALIVE": OLLAMA_KEEP_ALIVE,
    }


# What identifies what is actually serving. Two axes, because either alone is
# ambiguous -- and note this is a WEAKER identifier than the llama.cpp sibling's.
# There, `<ref>-sm<arch>` names a binary we built from a pinned source ref. Here
# it names a release tarball and the variant we asked for; whether the daemon
# honoured the pin is checked separately, by verify_gpu_arch.
def _build_id() -> str:
    return f"{OLLAMA_VERSION}-{OLLAMA_LLM_LIBRARY}"


def _user_data(model: str, instance_type: str) -> str:
    """Render idempotent cloud-init that INSTALLS Ollama and starts serving.

    No build, no compiler, no toolkit -- the whole difference from the llama.cpp
    sibling. Download, extract, write a unit, pull a tag, assert the model landed
    in VRAM.

    Progress goes to /var/log/ollama-install.log with `[stage]` markers, and
    an INSTALL_DONE marker under APP_DIR appears only after a real generate has been shown to
    run on the GPU.
    """
    _validate_instance_type(instance_type)

    swap = ""
    if _needs_swap(instance_type):
        # Lower pressure here than on the fp16 siblings -- the blobs total
        # ~4.3 GB, not 10.2 -- but the 1.5 GB tarball is decompressed on the same
        # box and Ollama's loader maps the blob, so the safety net stays.
        swap = f"""if ! swapon --show --noheadings 2>/dev/null | grep -q /swapfile; then
  fallocate -l {_SWAP_GB}G /swapfile
  chmod 600 /swapfile
  # NOT `mkswap -q`: that is a busybox flag and util-linux's mkswap rejects it.
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
"""

    env_lines = "\n".join(f"{k}={v}" for k, v in sorted(_serve_env(model, instance_type).items()))

    return f"""#!/usr/bin/env bash
set -euxo pipefail
{swap}mkdir -p {APP_DIR} {APP_DIR}/models

cat >{APP_DIR}/install.sh <<'INSTEOF'
#!/usr/bin/env bash
set -euxo pipefail

_T0=$(date +%s); _TLAST=$_T0
stage() {{
  local now; now=$(date +%s)
  echo "[stage] $1 +$((now - _TLAST))s (total $((now - _T0))s)"
  _TLAST=$now
}}

# unattended-upgrades RESTARTS SERVICES IT UPGRADES, mid-install, and reports
# Result=success with NRestarts=0 while doing it. On the PyTorch sibling that
# cost two full checkpoint downloads and looked exactly like an OOM.
systemctl stop apt-daily-upgrade.timer apt-daily.timer unattended-upgrades 2>/dev/null || true
systemctl mask apt-daily-upgrade.service apt-daily-upgrade.timer 2>/dev/null || true
stage mask-unattended-upgrades

install_runtime() {{
  export DEBIAN_FRONTEND=noninteractive
  APT_OPTS="-o DPkg::Lock::Timeout=600 -o Acquire::Retries=3"
  APT_OPTS="$APT_OPTS -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"

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
  # zstd only. The bundle is .tar.zst and GNU tar on 24.04 needs the binary to
  # be present; curl is already on the DLAMI. **No compiler, no CUDA toolkit** --
  # that is the whole point of this rig against its llama.cpp sibling.
  apt_run install -y zstd
  stage runtime-deps

  # 1.5 GB. Piped straight into tar rather than staged to disk: the root volume
  # also holds the model blobs, and there is no reason to hold both the archive
  # and its contents at once.
  curl -fsSL {OLLAMA_TARBALL_URL} | tar --use-compress-program=unzstd -x -C {APP_DIR}
  stage download-extract

  test -x {APP_DIR}/bin/ollama || {{
    echo "FATAL: {APP_DIR}/bin/ollama missing after extract." >&2
    echo "The arm64 bundle should contain bin/ollama and lib/ollama/." >&2
    ls -R {APP_DIR} | head -50 >&2
    exit 1
  }}

  # ASSERT THE PINNED CUDA VARIANT IS ACTUALLY IN THE BUNDLE, before anything
  # depends on it. OLLAMA_LLM_LIBRARY bypasses autodetection; it does NOT create
  # a library that is not there, and a bad pin means the daemon silently falls
  # back to whatever it can find -- including CPU.
  if [ ! -d "{APP_DIR}/lib/ollama/{OLLAMA_LLM_LIBRARY}" ]; then
    echo "FATAL: OLLAMA_LLM_LIBRARY={OLLAMA_LLM_LIBRARY} is not in this bundle." >&2
    echo "Present variants:" >&2
    ls -d {APP_DIR}/lib/ollama/*/ 2>/dev/null >&2 || true
    echo "Pinning a variant that does not exist makes the daemon fall back," >&2
    echo "possibly to CPU, with no error. Failing here instead." >&2
    exit 1
  fi
  echo "{OLLAMA_VERSION}-{OLLAMA_LLM_LIBRARY}" > {APP_DIR}/BUILD_ID
  stage verify-bundle
}}

# Assert the model actually LANDED IN VRAM.
#
# This is the analogue of the llama.cpp sibling grepping its built binary's
# --list-devices, and of the PyTorch sibling asserting sm_75 is in torch's arch
# list. Same reason each time: `nvidia-smi` succeeding proves the driver works,
# and Ollama answering proves nothing at all -- it will serve happily on CPU.
#
# `/api/ps` reports size and size_vram per loaded model. **size_vram == 0 means
# CPU.** That is the only signal here that distinguishes a healthy rig from a
# slow one, because Ollama chooses its own offload and cannot be told to fail.
verify_gpu() {{
  nvidia-smi || {{
    echo "FATAL: nvidia-smi failed. On this rig that usually means the AMI is an" >&2
    echo "ARM64 DLAMI built for Graviton CPU inference -- it boots fine, has no GPU." >&2
    exit 1
  }}

  # A generate is what forces a load; /api/ps is empty until something runs.
  curl -fsS --max-time 900 http://127.0.0.1:{LLAMA_PORT}/api/generate \
    -d '{{"model":"{model}","prompt":"ok","stream":false}}' >{APP_DIR}/probe.json
  cat {APP_DIR}/probe.json

  curl -fsS http://127.0.0.1:{LLAMA_PORT}/api/ps >{APP_DIR}/ps.json
  cat {APP_DIR}/ps.json

  python3 - <<'PYEOF'
import json, sys
ps = json.load(open("{APP_DIR}/ps.json"))
models = ps.get("models") or []
if not models:
    sys.exit("FATAL: /api/ps lists no loaded model after a generate.")
m = models[0]
total, vram = m.get("size", 0), m.get("size_vram", 0)
print("loaded:", m.get("name"), "size", total, "size_vram", vram)
if vram == 0:
    sys.exit(
        "FATAL: size_vram is 0 -- the model is on the CPU. Ollama will serve "
        "correctly and slowly. Check that OLLAMA_LLM_LIBRARY names a variant "
        "present in the bundle and that the driver matches it."
    )
if vram < total * 0.9:
    print(
        f"WARNING: only {{vram}}/{{total}} bytes in VRAM -- PARTIAL OFFLOAD. "
        "Ollama decides this itself; it cannot be forced. Do not benchmark this "
        "instance without saying so in the report.",
        file=sys.stderr,
    )
PYEOF
}}

install_runtime

systemctl daemon-reload
systemctl enable --now {SERVICE_NAME}.service
stage service-start

# Wait for the daemon before pulling: `enable --now` returns as soon as the
# process starts, not when it is listening.
#
# THE TIMEOUT IS FATAL. It used to fall through to `ollama pull` against a dead
# port, so a crash-looping daemon produced a confusing pull failure two stages
# later instead of naming itself here. MEASURED 2026-09-02: the $HOME bug above
# crash-looped the unit and this loop simply expired and carried on.
_ready=""
for _ in $(seq 1 60); do
  curl -fsS http://127.0.0.1:{LLAMA_PORT}/ >/dev/null 2>&1 && {{ _ready=1; break; }}
  sleep 2
done
if [ -z "$_ready" ]; then
  echo "FATAL: ollama did not bind {LLAMA_PORT} within 120s." >&2
  systemctl status {SERVICE_NAME} --no-pager -l >&2 || true
  journalctl -u {SERVICE_NAME} -n 40 --no-pager >&2 || true
  exit 1
fi
stage daemon-ready

# Pull through the DAEMON's own client so it lands in OLLAMA_MODELS rather than
# root's home -- the CLI reads the same env file the unit does.
env $(cat {APP_DIR}/env | xargs) {APP_DIR}/bin/ollama pull {model}
stage pull-{model}

verify_gpu
stage gpu-verify

touch {APP_DIR}/INSTALL_DONE
echo "[stage] INSTALL COMPLETE total $(($(date +%s) - _T0))s"
INSTEOF
chmod 700 {APP_DIR}/install.sh

cat >{APP_DIR}/env <<ENVEOF
{env_lines}
# HOME IS NOT OPTIONAL. MEASURED 2026-09-02 on i-0db1233beb07f0da7: systemd gives
# a unit a minimal environment with no $HOME, and `ollama serve` exits 1 with
#
#   Error: $HOME is not defined
#
# on every start. Restart=on-failure then crash-loops it -- 13 restarts in ~2
# minutes -- while the bootstrap's daemon-ready poll sat on a dead port.
#
# Setting OLLAMA_MODELS does NOT substitute for it: Ollama reads $HOME for its
# own config and signing key regardless of where the blob store lives.
HOME=/root
RIG_NAME={MANAGED_BY}
MODEL_NAME={model}
OLLAMA_VERSION={OLLAMA_VERSION}
ENVEOF
chmod 600 {APP_DIR}/env

# The HF token is fetched at boot into a root-only EnvironmentFile, never into
# user data -- instance metadata is readable by anything on the box.
#
# **Ollama does not use it.** It pulls from its own registry, not from Hugging
# Face, so this block is inert on this rig as configured. Kept anyway: it costs
# nothing, it survives a repoint at a Modelfile importing a Hub GGUF, and the
# xtrace discipline below is the part worth never losing.
#
# xtrace is off across the block on purpose: the script runs under `set -x` and
# bash traces variable assignments WITH their values.
set +x
if ! command -v aws >/dev/null 2>&1; then
  echo "WARNING: aws CLI not on PATH; cannot read secret {HF_SECRET_ID}." >&2
else
  HF=$(aws secretsmanager get-secret-value --region {AWS_REGION} --secret-id {HF_SECRET_ID} --query SecretString --output text 2>/dev/null || true)
  if [ -n "$HF" ]; then
    echo "HF_TOKEN=$HF" >>{APP_DIR}/env
  else
    echo "WARNING: secret {HF_SECRET_ID} is empty or unreadable." >&2
  fi
  unset HF
fi
set -x

# ExecStart is BARE. `ollama serve` takes no model arguments -- every decision is
# in the EnvironmentFile above. An argv typo would fail loudly; an env typo is
# silently ignored by the daemon, which is why _serve_env is validated against
# the daemon's own variable list in tests rather than trusted.
cat >/etc/systemd/system/{SERVICE_NAME}.service <<'UNITEOF'
[Unit]
Description=Gemma 4 E2B Q4_0 on T4G via Ollama
After=network-online.target

[Service]
Type=simple
EnvironmentFile={APP_DIR}/env
WorkingDirectory={APP_DIR}
ExecStart={APP_DIR}/bin/ollama serve
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNITEOF
systemctl daemon-reload

nohup bash {APP_DIR}/install.sh >/var/log/ollama-install.log 2>&1 &
echo "Ollama install started; follow /var/log/ollama-install.log"
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


# AWS caps each of StandardOutputContent and StandardErrorContent returned by
# get_command_invocation at 24,000 characters. Past that the content is silently
# truncated — which for get_llama_logs means reading a partial journal and
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
            f"Cloud-init does EVERYTHING here: it downloads the Ollama "
            f"{OLLAMA_VERSION} arm64 bundle, asserts `{OLLAMA_LLM_LIBRARY}` is in it, "
            "starts the daemon, pulls the tag, and asserts the model landed in VRAM. "
            "**No build, no compiler, no CUDA toolkit.** There is no separate deploy "
            "step. Watch get_install_progress for INSTALL COMPLETE."
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
            # Rendered from the same constants create_g5g_instance launches with.
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
            "Serving configuration (`ollama serve` takes NO arguments; all of it "
            "is environment):\n```\n"
            + "\n".join(f"{k}={v}" for k, v in sorted(_serve_env(model_name, instance_type).items()))
            + f"\n```\nBuild: `{_build_id()}`."
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

    Cloud-init does the whole job here: install zstd, download and extract the
    Ollama arm64 bundle, assert the pinned CUDA variant is present, start the
    daemon, pull the model tag, then assert the model actually landed in VRAM.
    **It starts serving on its own** — there is no deploy step and no build.

    Faster to a serving endpoint than the llama.cpp sibling, which compiles, and
    the spot exposure is correspondingly smaller: a reclaim costs a 1.5 GB
    download and a 4.3 GB pull, not a compile as well. Pass spot=False for
    on-demand when a run matters.
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
            "Downloading the Ollama arm64 bundle (no build). Follow with "
            "get_install_progress; it starts serving by itself."
        )
        # The AMI id belongs in the confirmation: AWS ships driverless ARM64
        # DLAMIs that boot perfectly well on a G5g and simply have no GPU, which
        # reads as a broken runtime rather than a wrong image. Recording which
        # image booted makes that one lookup instead of a guess.
        logger.info("launched instance=%s ami=%s type=%s market=%s",
                    instance_id, ami_id, instance_type, market)
        return (
            f"✅ Launching `{instance_id}` ({instance_type}, {market}, "
            f"{_gpu_count(instance_type)}x T4G) in `{AWS_REGION}`.\n"
            f"AMI: `{ami_id}`\n{tail}"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List managed G5g instances", annotations=READ_ONLY)
async def list_g5g_instances() -> str:
    """List instances tagged ManagedBy=gpu-pytorch-g5g-2b."""
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


@mcp.tool(title="Verify the GPU and that the model is in VRAM", annotations=READ_ONLY)
async def verify_gpu_arch(instance_id: str) -> str:
    """Check the driver sees a T4G AND that Ollama actually put the model on it.

    Three claims, and only the third one matters:

    * `nvidia-smi` works — proves the AMI carries a driver.
    * Ollama answers — **proves nothing.** It serves correctly on CPU.
    * `/api/ps` reports `size_vram > 0` — the real check.

    **Ollama chooses its own offload and cannot be told to fail.** The llama.cpp
    sibling can at least demand `--n-gpu-layers 999` and assert on the built
    binary's device list; here the daemon decides, so the only honest test is
    where the bytes ended up after a load. A partial offload — `size_vram` well
    under `size` — is a working, slow server with nothing in any log to find.

    This also reports which CUDA variant was pinned. Left unpinned on a CUDA 13
    driver Ollama selects `cuda_v13`, which carries **PTX for every architecture
    and native SASS for none**, so every kernel is JIT-compiled at load.
    """
    command = (
        "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true; "
        f"echo '--- build id ---'; cat {APP_DIR}/BUILD_ID 2>/dev/null || echo '(not installed yet)'; "
        f"echo '--- pinned variant ---'; grep OLLAMA_LLM_LIBRARY {APP_DIR}/env 2>/dev/null || true; "
        f"echo '--- variants present in the bundle ---'; ls -d {APP_DIR}/lib/ollama/*/ 2>/dev/null || true; "
        f"echo '--- loaded models ---'; curl -fsS http://127.0.0.1:{LLAMA_PORT}/api/ps 2>&1 || true"
    )
    try:
        output = await _ssm(instance_id, command, timeout=600)
        verdict = ""
        vram = None
        try:
            import json as _json
            blob = output[output.index('{"models"'):]
            models = _json.loads(blob[:blob.rindex("}") + 1]).get("models") or []
            if models:
                vram, total = models[0].get("size_vram", 0), models[0].get("size", 0)
        except Exception:
            pass

        if "not installed yet" in output:
            verdict = "\n\n⏳ Not installed yet — check `get_install_progress`."
        elif vram is None:
            verdict = (
                "\n\n⚠️ No model is loaded, so the VRAM check could not run. Ollama "
                "unloads on idle (this rig sets OLLAMA_KEEP_ALIVE=-1, so an empty "
                "list here means nothing has been generated since the last restart). "
                "Send one `query_model` call and re-run."
            )
        elif vram == 0:
            verdict = (
                "\n\n❌ **`size_vram` is 0 — the model is on the CPU.** Ollama will "
                "serve correctly and slowly, and nothing will log an error. Check that "
                f"`OLLAMA_LLM_LIBRARY={OLLAMA_LLM_LIBRARY}` names a variant listed above "
                "and that the driver matches it. **Do not benchmark this instance.**"
            )
        elif total and vram < total * 0.9:
            verdict = (
                f"\n\n⚠️ **PARTIAL OFFLOAD**: {vram}/{total} bytes in VRAM. Ollama "
                "decides this itself and cannot be forced. The rig will serve; any "
                "number off it must say this in the report."
            )
        else:
            verdict = (
                f"\n\n✅ Model resident in VRAM ({vram} bytes) with "
                f"`{OLLAMA_LLM_LIBRARY}` pinned. This is the combination the rig requires."
            )
        return f"### GPU probe on `{instance_id}`\n\n```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Restart the Ollama daemon", annotations=WRITE)
async def restart_ollama(instance_id: str) -> str:
    """Restart the serving unit and report what came back up.

    **This rig has no deploy step and no payload** — the daemon is a published
    release tarball and the weights come from Ollama's registry. Nothing of ours
    reaches the box, so the stale-payload trap that bit the PyTorch sibling
    (deploying through the registered MCP server ships the PREVIOUS `make skill`
    output) has nothing to act on here.

    **A restart drops the loaded model.** With `OLLAMA_KEEP_ALIVE=-1` the daemon
    otherwise holds it forever, so the first request after this call pays a full
    load and will read as a latency spike. Warm up before measuring anything.

    RESTART, not `enable --now`: the latter is a no-op against a running unit,
    which is how a sibling silently served stale code while `is-active` reported
    success.
    """
    try:
        command = (
            f"set -e; systemctl restart {SERVICE_NAME}; sleep 5; "
            f"systemctl is-active {SERVICE_NAME} || true; "
            f"cat {APP_DIR}/BUILD_ID 2>/dev/null || true; "
            f"systemctl show {SERVICE_NAME} -p NRestarts -p ExecMainStatus -p Result"
        )
        output = await _ssm(instance_id, command, timeout=600)
        return (
            f"### Restarted `{SERVICE_NAME}` on `{instance_id}`\n\n```\n{output}\n```\n\n"
            "⚠️ The loaded model was dropped — the next request pays a full load. "
            "Warm up before measuring.\n\n"
            "⚠️ `Result=success` with `NRestarts=0` does NOT mean nothing went wrong: "
            "unattended-upgrades restarts services it upgrades and reports exactly that. "
            "The bootstrap masks it, but check the PID actually changed if a restart "
            "looks unexplained."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get Ollama install progress", annotations=READ_ONLY)
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
            "if [ -f /var/log/ollama-install.log ]; then "
            f"tail -n {tail} /var/log/ollama-install.log; "
            "else echo 'NO INSTALL LOG: cloud-init never reached install.sh'; "
            "echo '--- cloud-init output (tail 60) ---'; "
            "tail -n 60 /var/log/cloud-init-output.log 2>/dev/null "
            "|| echo 'no cloud-init output log either'; fi"
        )
        output = await _ssm(instance_id, command)

        # Ordered most-specific first. "status: error" is cloud-init's own
        # verdict and outranks the absence of a log, which is only a symptom.
        if "INSTALL COMPLETE" in output:
            verdict = (
                "\n\n✅ Built, the binary saw a CUDA device, and the unit is enabled. "
                "Next: verify_model_health. Note the first start downloads 3.35 GB "
                "before it binds a port."
            )
        elif "status: error" in output:
            verdict = (
                "\n\n❌ cloud-init FAILED — the bootstrap died, the install is not "
                "running and never will be. Read the cloud-init output above for the "
                "failing command; relaunching will reproduce it. This is NOT a slow install."
            )
        elif "NO INSTALL LOG" in output and "status: done" in output:
            verdict = (
                "\n\n❌ cloud-init finished but never wrote /var/log/ollama-install.log, so "
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


@mcp.tool(title="Get Ollama daemon logs", annotations=READ_ONLY)
async def get_llama_logs(instance_id: str, tail: int = 100) -> str:
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
                return f"📡 `http://{host}:{LLAMA_PORT}/v1`"
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
    Ollama has no degeneracy counter, so the verdict is computed locally — see
    the heuristic below. Weaker, and labelled as weaker, but it still cannot pass
    a `': ok: ok: ok…'` reply.

    **Ollama's health endpoint is `GET /`**, which returns the plain text
    "Ollama is running". There is no `/health` and no `/metrics`; the OpenAI shim
    at `/v1` does not add one. So this checks `/` for liveness and reads the
    decode rate off `/api/generate`'s own timings, which doubles as a "did this
    box actually get a GPU" signal — a CPU-resident model answers correctly at a
    rate that gives it away immediately.
    """
    try:
        endpoint = await get_endpoint(instance_id)
        if not endpoint.startswith("📡"):
            return endpoint
        base = endpoint.strip("📡 `")
        root = base.removesuffix("/v1")
        async with httpx.AsyncClient(timeout=120) as client:
            # `GET /` is Ollama's liveness check. It answers 200 with
            # "Ollama is running" as soon as the daemon binds -- BEFORE any model
            # is loaded -- so it is necessary and nowhere near sufficient.
            health = await client.get(f"{root}/")
            # The native endpoint, not the OpenAI shim: `/api/generate` returns
            # eval_count/eval_duration, and the shim drops them.
            gen = await client.post(
                f"{root}/api/generate",
                json={"model": MODEL_NAME,
                      "prompt": "Reply with the single word: ok",
                      "stream": False, "options": {"num_predict": 16}},
            )

        body = gen.json()
        text = body.get("response", "")
        # Ollama has NO degeneracy counter, so this rig detects a token loop
        # itself rather than pretending the server does.
        #
        # The siblings read `tpu_jax_degenerate_responses_total` either side of
        # the probe, because their own server computes it. Dropping that check
        # entirely was the tempting move and would have been wrong: the vLLM
        # sibling on this exact silicon was measured answering `': ok: ok: ok…'`,
        # which is non-empty, 200, and worthless. **Never health-check by testing
        # for a non-empty response.**
        #
        # The heuristic is deliberately crude and deliberately local: a short
        # reply whose tokens are almost all repeats of one another. It can miss;
        # it cannot produce a false ✅ from an empty-or-not test.
        words = text.split()
        degenerate = len(words) >= 6 and len(set(words)) <= max(2, len(words) // 4)

        # `GET /` returns plain text, not JSON, and carries no build information.
        # So the served build is what the bootstrap stamped -- a weaker claim than
        # the siblings' (they read it out of the running process), labelled as
        # such rather than quietly equated.
        served_build = "(Ollama reports none; see /api/version)"
        local_build = _build_id()

        ok = (
            health.status_code == 200
            and gen.status_code == 200
            and not degenerate
            and (body.get("eval_count") or 0) > 0
        )
        status = "✅" if ok else "❌"

        lines = [
            f"{status} liveness={health.status_code} "
            f"tokens={body.get('eval_count', 0)} reply={text!r}",
            "",
            f"- Degenerate (local heuristic, not a server verdict): "
            f"**{'YES — token loop' if degenerate else 'no'}**",
            f"- Build configured here: `{local_build}`",
            f"- Build reported by the server: `{served_build}`",
        ]
        rate = _decode_rate(body)
        if rate:
            # THIS REQUEST, not a cumulative counter -- Ollama has none. Stated
            # every time it is printed, because a sibling's identically-shaped
            # line means something different.
            lines.append(f"- Decode on THIS request: **{rate:.2f} tok/s**")
            # A CPU-resident model is the failure this rig is most likely to
            # ship, and it passes every other check in this function. The exact
            # threshold does not matter: the gap is roughly an order of
            # magnitude, not a few percent. Warm-up noise cannot cross it.
            if rate < 3.0:
                lines.append(
                    "- ⚠️ **That rate is consistent with a CPU-resident model.** Run "
                    "`verify_gpu_arch` and read `size_vram` — Ollama serves correctly "
                    "on CPU and logs nothing about it."
                )
        load_ms = (body.get("load_duration") or 0) / 1e6
        if load_ms > 100:
            lines.append(
                f"- ⚠️ `load_duration` {load_ms:.0f} ms — the model was not resident. "
                "With `OLLAMA_KEEP_ALIVE=-1` that should only follow a restart."
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


# THERE IS NO /metrics ENDPOINT ON THIS RIG, and no `_parse_prom` with it.
#
# Verified 2026-09-02 against `server/routes.go`: Ollama registers `/`,
# `/api/version`, `/api/status`, `/api/ps`, `/api/generate`, `/api/chat`,
# `/api/tags`, `/api/show`, the `/v1/*` OpenAI shims and more -- and **no
# Prometheus route of any kind.**
#
# So where every sibling SCRAPES a counter, this rig PROBES. That is a weaker
# instrument and the difference is worth stating rather than papering over:
#
#   scrape (llama.cpp, JAX, PyTorch)   cumulative over every request since start
#   probe  (here)                      one request, right now
#
# A probe cannot report what happened during a sweep; it can only measure a fresh
# request. `sweep.py` is unaffected -- it already computes the client-side
# inter-token statistic and never needed a server counter -- but do not treat the
# number this tool returns as a whole-run figure the way `get_metrics` output
# from a sibling can be treated.

# Timings Ollama returns on /api/generate, from `api.Metrics` in api/types.go.
# **Every duration is a Go time.Duration, which marshals to NANOSECONDS.**
# Dividing by 1e6 instead of 1e9 yields a decode rate 1000x too low and looks
# like a catastrophically broken rig rather than a unit bug.
_NS_PER_S = 1e9


def _decode_rate(body: dict) -> float | None:
    """Decode tok/s from an /api/generate response. Prefill excluded.

    `eval_count` / `eval_duration` is generation alone; `prompt_eval_*` is
    prefill and `total_duration` carries both plus load. This is the field pair
    that makes an Ollama number comparable to the siblings'
    `tpu_jax_decode_tokens_per_second` and to llama.cpp's
    `llamacpp:tokens_predicted_*`.
    """
    tokens = body.get("eval_count") or 0
    ns = body.get("eval_duration") or 0
    return tokens / (ns / _NS_PER_S) if tokens and ns else None


def _prefill_rate(body: dict) -> float | None:
    """Prefill tok/s. Reported separately and NEVER folded into decode."""
    tokens = body.get("prompt_eval_count") or 0
    ns = body.get("prompt_eval_duration") or 0
    return tokens / (ns / _NS_PER_S) if tokens and ns else None


@mcp.tool(title="Probe serving timings", annotations=READ_ONLY)
async def get_metrics(instance_id: str, prompt: str = "Explain gradient descent.",
                      max_tokens: int = 128) -> str:
    """Run one generate and report its decode rate, prefill rate and load time.

    **This PROBES rather than scrapes** — Ollama has no `/metrics` endpoint (see
    the note above `_decode_rate`). The figure describes the request this call
    just made, not the run so far.

    Quote the decode rate. `total_duration` carries prefill, sampling and the
    HTTP round trip; the PyTorch sibling measured decode and end-to-end
    disagreeing by up to 36% on the same rows.

    `load_duration` is worth watching on its own: a non-trivial value means the
    model was not resident, which with `OLLAMA_KEEP_ALIVE=-1` should only happen
    right after a restart. If it recurs, the keep-alive is not taking effect and
    every sweep cell is paying a reload.
    """
    try:
        endpoint = await get_endpoint(instance_id)
        if not endpoint.startswith("📡"):
            return endpoint
        base = endpoint.strip("📡 `").removesuffix("/v1")
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{base}/api/generate",
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False,
                      "options": {"num_predict": max_tokens}},
            )
        if response.status_code != 200:
            return f"❌ /api/generate returned {response.status_code} — is {SERVICE_NAME} serving?"
        body = response.json()

        decode = _decode_rate(body)
        prefill = _prefill_rate(body)
        load_ms = (body.get("load_duration") or 0) / 1e6
        total_ms = (body.get("total_duration") or 0) / 1e6

        lines = [
            f"### Timing probe on `{instance_id}`",
            "",
            f"Build: `{_build_id()}` — Ollama {OLLAMA_VERSION}, "
            f"`OLLAMA_LLM_LIBRARY={OLLAMA_LLM_LIBRARY}`.",
            f"Model: `{MODEL_NAME}`",
            "",
            "| Measure | Value |",
            "| --- | ---: |",
            f"| Generated tokens | {body.get('eval_count', 0)} |",
            f"| **Decode** | **{decode:.2f} tok/s** |" if decode else "| Decode | — |",
            f"| Prefill | {prefill:.1f} tok/s |" if prefill else "| Prefill | — |",
            f"| Load | {load_ms:.1f} ms |",
            f"| Total (incl. prefill + HTTP) | {total_ms:.1f} ms |",
            "",
            "**One request, not a cumulative counter** — Ollama exposes no "
            "/metrics. Prefill is listed separately and must never be added in.",
        ]
        if load_ms > 100:
            lines.append(
                f"\n⚠️ `load_duration` is {load_ms:.0f} ms — the model was not resident. "
                "With `OLLAMA_KEEP_ALIVE=-1` that should only follow a restart. If it "
                "recurs, keep-alive is not taking effect and every sweep cell is paying "
                "a reload."
            )
        if decode and decode < 3.0:
            lines.append(
                "\n⚠️ **That rate is consistent with a CPU-only load.** Run "
                "`verify_gpu_arch` and check `size_vram` — Ollama serves correctly "
                "on CPU and logs nothing."
            )
        return "\n".join(lines)
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
    env = "\n".join(f"{k}={v}" for k, v in sorted(_serve_env(MODEL_NAME, INSTANCE_TYPE).items()))
    return f"""### {RIG_NAME}

Serving `{MODEL_NAME}` with **Ollama** on **EC2 G5g** — AWS Graviton2 (aarch64)
host, NVIDIA **T4G** GPU (Turing, SM 7.5, 15360 MiB).

**Its sibling `gpu-llamacpp-g5g-2b-q4_0` runs the same engine.** Ollama links
llama.cpp; the Gemma 4 graph both execute is upstream `src/models/gemma4.cpp`.
Slot 2 names the front end, not the decoder. Read THE NOTE in `CLAUDE.md` before
comparing them — four differences are visible to a benchmark.

| Setting | Value |
| --- | --- |
| Region | `{AWS_REGION}` |
| Instance type | `{INSTANCE_TYPE}` ({_gpu_count(INSTANCE_TYPE)}x T4G, {_host_memory_gb(INSTANCE_TYPE)} GiB RAM) |
| Runtime | Ollama `{OLLAMA_VERSION}` arm64 bundle — **downloaded, not built** |
| CUDA variant | `{OLLAMA_LLM_LIBRARY}` (pinned) |
| Build id | `{_build_id()}` |
| Service | `{SERVICE_NAME}` (systemd, not docker) |
| Managed-by tag | `{MANAGED_BY}` |
| Root volume | {ROOT_VOLUME_GB} GB gp3 @ {ROOT_VOLUME_THROUGHPUT_MBPS} MiB/s, {ROOT_VOLUME_IOPS} IOPS |

**`ollama serve` takes no arguments.** All configuration is environment:

```
{env}
```

**Why the CUDA variant is pinned.** MEASURED 2026-09-02 by walking the fatbin
headers in the v0.33.2 arm64 bundle and decompressing one entry of each kind:

```
cuda_v12   CUBIN sm_60 61 70 75 80 86 89 90 100 120  + PTX sm_50..120
cuda_v13   PTX ONLY for every arch — zero CUBIN
```

Ollama selects between them by driver version. This rig's DLAMI carries a CUDA 13
driver, so left alone it would pick `cuda_v13` and JIT every kernel from PTX at
load — warm-up that would be recorded as a property of the engine. `cuda_v12` has
native Turing SASS, and the llama.cpp sibling builds native sm_75 too, so pinning
is what makes the pair comparable.

**Two other defaults that would damage a measurement**, both overridden above:
`OLLAMA_CONTEXT_LENGTH` defaults to 0, meaning "4k/32k/256k based on VRAM" — two
instance sizes would silently get two contexts. `OLLAMA_KEEP_ALIVE` defaults to
5m, so a sweep pausing longer reloads the model and records it as latency.

**There is no `/metrics` endpoint.** Verified against `server/routes.go`: Ollama
registers no Prometheus route. `get_metrics` therefore **probes** — it runs one
generate and reads `eval_count` / `eval_duration` from the response. Those are Go
`time.Duration`s, i.e. **nanoseconds**. A probe measures one request, not a run.

**The failure this rig ships silently is a CPU-resident model.** Ollama chooses
its own offload and cannot be told to fail, so the only honest check is where the
bytes ended up: `verify_gpu_arch` reads `size_vram` from `/api/ps`, and zero
means CPU. Ollama will serve correctly and slowly and log nothing.

**NOTHING HAS BEEN MEASURED ON THIS RIG.** Every figure above is a property of
the bundle or a sibling's number, labelled as such.

Order of operations: `create_g5g_instance` → `get_install_progress` →
`verify_gpu_arch` → `verify_model_health` → `get_metrics`. No deploy step.
"""


if __name__ == "__main__":
    mcp.run()
