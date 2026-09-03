"""EC2 G5g (Graviton2 + NVIDIA T4G) lifecycle and inference MCP server, llama.cpp path.

Serves `google/gemma-4-E2B-it-qat-q4_0-gguf` -- Google's QAT checkpoint in its
native Q4_0 packing -- through `llama-server`, built from source on the box.

Like every G5g sibling this uses boto3 rather than shelling out to the AWS CLI,
and Systems Manager Run Command for remote administration: no inbound SSH rule
and no private key.

WHY THIS RIG EXISTS
-------------------
It is the only rig here that serves 4-bit weights on a GPU. Every other route
was checked and closed (see the root QUANTIZATION.md):

  * vLLM 0.26.0 has no `gguf` module at all -- on CUDA or TPU.
  * No GGUF reader exists anywhere in the JAX ecosystem.
  * transformers 5.12.1 reads the file but dequantizes to fp32 (a 9.4 GB host
    transient on one tensor) and silently drops 35 `layer_scalar` tensors.

llama.cpp is what is left, and it is not a consolation prize: the file streams
1.407 GB per decode step against the fp16 rig's 4.514 GB, and drops resident
weights from 10.209 GB to 3.35 GB on a 14.07 GB chip. **The residency is the
win, not the bandwidth** -- decode at B=1 on this silicon is launch-bound, and
the sibling measured a 3.5 GB weight cut moving throughput by 0.0%. What the
headroom buys is batching, which the PyTorch sibling measured at 7.84x.

THREE THINGS THIS RIG CAN GET WRONG SILENTLY, all asserted rather than trusted:

  1. **A CPU-only build.** `-DGGML_CUDA=ON` failing does not fail the build --
     it produces a working `llama-server` that serves at single-digit tok/s.
     verify_gpu greps the built binary's own device list for CUDA.
  2. **The wrong SM.** llama.cpp defaults to a broad arch list; this pins
     `CMAKE_CUDA_ARCHITECTURES=75` because T4G is the only target and a JIT
     fallback would be measured as a runtime property.
  3. **No nvcc.** The DLAMI ships one; if it ever stops, the build must die
     loudly rather than fall back to CPU. install_runtime probes and exits 1.

NOTHING HAS BEEN MEASURED ON THIS RIG YET. Every figure above is arithmetic from
the artifact's own tensor table or a number measured on a *sibling* and labelled
as such. Do not quote any of it as this rig's result.
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
RIG_NAME = "gpu-llamacpp-g5g-2b-q4_0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(RIG_NAME)

MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", RIG_NAME)
mcp = FastMCP(MCP_SERVER_NAME)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")
# The MODEL SLOT IS A REPO AND A FILE, not a repo alone. A GGUF repo can hold
# several quantisations plus an mmproj, so naming only the repo would leave
# llama-server to pick -- and slot 5 of this rig's name is a claim about which
# file. Both are passed to `llama-server --hf-repo/--hf-file`, which downloads
# and caches on the box; there is no rig-owned payload to ship.
#
# Verified on the Hub 2026-09-02: gemma-4-E2B_q4_0-it.gguf is 3,349,516,256
# bytes, sha256 fa401b55... 541 tensors, Q4_0 x275 / F32 x263 / Q6_K x2 / F16 x1.
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-E2B-it-qat-q4_0-gguf")
MODEL_FILE = os.getenv("MODEL_FILE", "gemma-4-E2B_q4_0-it.gguf")
# The vision/audio projector ships as a SEPARATE 986,833,664-byte file. Empty by
# default: this rig serves text, and loading it costs ~1 GB of the budget the
# Q4_0 packing exists to free. Set it only for a multimodal run.
MMPROJ_FILE = os.getenv("MMPROJ_FILE", "")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "g5g.2xlarge")
SERVICE_NAME = os.getenv("SERVICE_NAME", "llamacpp-g5g")
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8000"))

# Compute dtype is NOT a knob here and there is no DTYPE constant on purpose.
# ggml picks per-kernel: Q4_0 weights are dequantised into the accumulator type
# the CUDA backend chooses for SM 7.5. The fp16-vs-bf16 question that dominates
# the JAX and PyTorch siblings does not arise -- Turing has no bf16 datapath and
# ggml never asks for one. Do not reintroduce DTYPE/QUANT_MODE/PLE_BITS/
# INT8_LM_HEAD: those name knobs of this repo's own JAX port, the PyTorch fork
# carried them inert for a month, and `tpu.env` keeping a key is not evidence
# that anything reads it.

# Context BOUND, not a padded buffer: llama.cpp allocates a KV cache of exactly
# this size at startup and nothing recompiles per shape. E2B's KV is ~18 KiB per
# token, so 4096 tokens is ~74 MiB against a 14.07 GB chip -- KV is not what
# sizes this rig, and a context limit must never be derived from KV arithmetic
# here. See MODELS.md.
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "4096"))

# Parallel decode slots. llama.cpp splits CONTEXT_SIZE across them, so raising
# this divides the per-sequence context rather than adding memory.
#
# Defaults to 1 to match the siblings' single-stream baseline, which is the only
# configuration the family's existing numbers can be compared against. The
# PyTorch sibling measured 7.84x from batching at B=8 for 0.258 GB, and this rig
# has ~6.9 GB more headroom than it did -- but raise this deliberately, in a run
# whose report says so, not as a default that quietly invalidates a comparison.
PARALLEL_SLOTS = int(os.getenv("PARALLEL_SLOTS", "1"))

# Offload every layer. 999 is llama.cpp's idiom for "all of them"; a partial
# offload is the silent-slowness failure this rig is most likely to hit after a
# CPU-only build, because it also looks like a working server.
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "999"))

# ---------------------------------------------------------------------------
# The build. There are no wheels to install on this path -- see install_runtime.
# ---------------------------------------------------------------------------

# llama.cpp publishes NO prebuilt Linux aarch64 CUDA binary. Verified against
# release b10760 (2026-09-02): the arm64 Linux assets are `-bin-ubuntu-arm64`
# (CPU) and `-bin-ubuntu-vulkan-arm64`; every `-cuda-` Linux asset is x64, and
# the only arm64 CUDA build is for Windows. So this rig compiles, and that is a
# real difference from the Ollama sibling, which ships a working aarch64 CUDA
# binary with native sm_75 SASS already in it.
LLAMA_CPP_REPO = os.getenv("LLAMA_CPP_REPO", "https://github.com/ggml-org/llama.cpp")

# PINNED, not `master`. An unpinned build makes the engine a moving variable
# across two runs that are supposed to differ only in the thing being measured,
# and llama.cpp cuts several releases a day (b10754..b10760 all landed on
# 2026-09-02). Bump it deliberately and record the bump in the run's REPORT.md.
LLAMA_CPP_REF = os.getenv("LLAMA_CPP_REF", "b10760")

# T4G is SM 7.5 and it is the ONLY target, so the build is pinned to it rather
# than to llama.cpp's default arch list. Two reasons, and the second is the one
# that matters for measurement: a narrow list builds several times faster on a
# Graviton2, and a broad list can leave the actual device on a JIT'd PTX path
# whose warm-up would be recorded as a runtime property of the engine.
#
# This is the same hazard the Ollama sibling meets from the other direction: its
# cuda_v13 bundle carries PTX for every arch and native SASS for none.
CUDA_ARCH = os.getenv("CUDA_ARCH", "75")

# Where the checkout and the build tree live. Kept on the root volume with the
# model cache, which is why ROOT_VOLUME_GB is sized for both.
BUILD_DIR = os.getenv("BUILD_DIR", "/opt/llamacpp-build")

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
APP_DIR = os.getenv("APP_DIR", "/opt/llamacpp-g5g")
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


def _serve_argv(model: str, instance_type: str) -> str:
    """Arguments for `llama-server` -- and ONLY the ones it defines.

    The PyTorch fork of this rig shipped the JAX sibling's flag set verbatim
    (--quant-mode, --ple-bits, --int8-lm-head, ...). None existed on the target
    server, argparse exits 2 on an unknown flag, and the unit crash-looped from
    the very first start with the reason only in `journalctl`. That class of bug
    is why this function exists separately and is unit-tested against the flag
    list rather than written inline.

    Four things worth knowing about the choices here:

    * `--hf-repo` + `--hf-file` name BOTH halves of the artifact. A GGUF repo can
      hold several quantisations plus an mmproj, so naming the repo alone would
      let llama-server pick -- and slot 5 of this rig's directory name is a claim
      about which file.
    * `--n-gpu-layers 999` offloads everything. A PARTIAL offload is the second
      silent-slowness failure on this rig (the first is a CPU-only build): the
      server starts, answers correctly, and runs several times slower.
    * `--ctx-size` is split across `--parallel` slots by llama.cpp, so raising
      the slot count divides per-sequence context rather than adding memory.
    * `--metrics` is NOT optional. It is off by default, and without it
      /metrics 404s and `get_metrics` has nothing to read. The decode gauge is
      the number this whole rig family compares on.

    There is no tensor-parallel flag: one T4G. On the two-GPU sizes the second
    idles, which _tensor_parallel_size() reports but nothing acts on.
    """
    argv = (
        f"--hf-repo {model} --hf-file {MODEL_FILE} "
        f"--host 0.0.0.0 --port {LLAMA_PORT} "
        f"--ctx-size {CONTEXT_SIZE} --parallel {PARALLEL_SLOTS} "
        f"--n-gpu-layers {N_GPU_LAYERS} --metrics"
    )
    if MMPROJ_FILE:
        argv += f" --mmproj-file {MMPROJ_FILE}"
    return argv


# THERE IS NO SERVING PAYLOAD ON THIS RIG, and that is a real difference rather
# than an omission.
#
# The JAX and PyTorch siblings ship their own `*_openai_server.py` over SSM,
# because "our Gemma 4 port" is not a published artifact. Here the server IS the
# published artifact: `llama-server` is built from a pinned llama.cpp ref and
# fetches its own checkpoint through `--hf-repo`. So `_PAYLOAD_FILES`,
# `_payload_root`, `_payload_digest` and `_payload_tar_b64` are all deleted
# rather than left returning empty -- a payload digest of nothing would still
# have been reported as a build id, and it would have been wrong.
#
# What identifies a build here is LLAMA_CPP_REF plus CUDA_ARCH, which the
# bootstrap writes to {APP_DIR}/BUILD_ID and `get_deployment_config` reports.
def _build_id() -> str:
    """Short identifier for what is actually serving.

    Two axes, because either alone is ambiguous: the same llama.cpp ref built
    for a different arch is a different binary, and the Ollama sibling's whole
    hazard is that its arch coverage is chosen for it.
    """
    return f"{LLAMA_CPP_REF}-sm{CUDA_ARCH}"


def _user_data(model: str, instance_type: str) -> str:
    """Render idempotent cloud-init that BUILDS llama.cpp and starts serving.

    Unlike every sibling this has no separate deploy step: there is no rig-owned
    payload, and llama-server fetches its own checkpoint. One provisioning round
    takes the instance from empty to serving.

    Progress goes to /var/log/llamacpp-install.log with `[stage]` markers, and
    an INSTALL_DONE marker under APP_DIR appears only after the built binary has been shown to
    see a CUDA device. That ordering is the point: a CPU-only build produces a
    perfectly working llama-server, so "it starts" proves nothing.
    """
    _validate_instance_type(instance_type)

    swap = ""
    if _needs_swap(instance_type):
        # Kept from the siblings, though the pressure here is lower: the Q4_0
        # file is 3.35 GB where theirs is 10.2, so the mmap that failed on a
        # 7.5 GiB host has far more room. Retained anyway because the build is
        # the new memory peak -- nvcc on a Graviton2 with -j$(nproc) is what
        # will hit this, not the model load.
        swap = f"""if ! swapon --show --noheadings 2>/dev/null | grep -q /swapfile; then
  fallocate -l {_SWAP_GB}G /swapfile
  chmod 600 /swapfile
  # NOT `mkswap -q`: that is a busybox flag and util-linux's mkswap rejects it
  # with `invalid option -- 'q'`. Under `set -e` that killed cloud-init before
  # install.sh was even written.
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
"""

    argv = _serve_argv(model, instance_type)
    build_reqs = " ".join(_BUILD_REQUIREMENTS)

    return f"""#!/usr/bin/env bash
set -euxo pipefail
{swap}mkdir -p {APP_DIR} {BUILD_DIR} {APP_DIR}/models

cat >{APP_DIR}/install.sh <<'INSTEOF'
#!/usr/bin/env bash
set -euxo pipefail

# The build is the longest phase and on spot it is the one that gets reclaimed
# out from under you -- the sibling lost an instance 21 minutes in. Greppable
# with `grep -F '[stage]' /var/log/llamacpp-install.log`.
_T0=$(date +%s); _TLAST=$_T0
stage() {{
  local now; now=$(date +%s)
  echo "[stage] $1 +$((now - _TLAST))s (total $((now - _T0))s)"
  _TLAST=$now
}}

# unattended-upgrades RESTARTS SERVICES IT UPGRADES, mid-install, and reports
# Result=success with NRestarts=0 while doing it. On the PyTorch sibling that
# cost two full checkpoint downloads and looked exactly like an OOM (it is not:
# an OOM kill is SIGKILL and cannot report success). Masked FIRST, before
# anything long-running starts, because it is also what holds the dpkg lock.
systemctl stop apt-daily-upgrade.timer apt-daily.timer unattended-upgrades 2>/dev/null || true
systemctl mask apt-daily-upgrade.service apt-daily-upgrade.timer 2>/dev/null || true
stage mask-unattended-upgrades

install_runtime() {{
  export DEBIAN_FRONTEND=noninteractive
  # Three apt hazards, all of which fail the install AND hide it, because
  # INSTALL_DONE is never touched and get_install_progress then reports
  # "INSTALL IN PROGRESS" indefinitely rather than surfacing an error.
  APT_OPTS="-o DPkg::Lock::Timeout=600 -o Acquire::Retries=3"
  APT_OPTS="$APT_OPTS -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"

  # MEASURED on the sibling 2026-08-21: us-east-1.ec2.ports.ubuntu.com returned
  # 503 over IPv4 and resolved to AAAA only, on a host with no IPv6 route.
  # apt-get update wedged for 12 minutes at the very first step.
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
  apt_run install -y {build_reqs}
  stage build-deps

  # NVCC IS DISCOVERED, AND IT IS NOT WHERE ANYONE LOOKS FIRST.
  #
  # MEASURED 2026-09-02 on i-05005979aff5a0df9, the rig's first launch. The
  # ARM64 PyTorch DLAMI carries **no /usr/local/cuda at all** and no nvcc on
  # PATH. The original probe checked exactly those two places, so it exited 1
  # and killed the install -- correctly, but for the wrong reason.
  #
  # The toolkit is there. It ships as a PIP PACKAGE inside the DLAMI's PyTorch
  # venv:
  #
  #   /opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc   (13.2)
  #                                            .../cu13/include/       headers
  #                                            .../cu13/lib/           libcudart.so.13,
  #                                                                    libcublas.so.13, ...
  #
  # **This is the same shape as the PyTorch sibling's central hazard**, one
  # level down. There, torch comes from the AMI so the INTERPRETER is discovered
  # rather than chosen. Here the TOOLKIT comes from the AMI, at a path that
  # moves with the DLAMI's Python version. Do not hardcode this path; glob it.
  #
  # apt's nvidia-cuda-toolkit is still NOT the answer: its candidate on 24.04 is
  # 12.0 against a CUDA 13 driver (595.71.05 measured), so it would build against
  # a different CUDA line than the runtime the box actually has.
  NVCC=""
  for cand in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc \
              /opt/*/lib/python*/site-packages/nvidia/cu*/bin/nvcc \
              /opt/*/lib/python*/site-packages/nvidia/cuda_nvcc/bin/nvcc \
              "$(command -v nvcc || true)"; do
    [ -x "$cand" ] && {{ NVCC="$cand"; break; }}
  done
  if [ -z "$NVCC" ]; then
    echo "FATAL: no nvcc anywhere on this AMI, so llama.cpp cannot be built with CUDA." >&2
    echo "A CPU-only build would SERVE CORRECTLY and be several times slower," >&2
    echo "so this fails loudly rather than producing a misleading rig." >&2
    echo "Looked in /usr/local/cuda*, the DLAMI pip trees, and PATH:" >&2
    ls -d /usr/local/cuda* /opt/*/lib/python*/site-packages/nvidia/* 2>/dev/null >&2 || true
    exit 1
  fi
  # CUDA root is two levels up from bin/nvcc, whichever layout matched.
  CUDA_ROOT="$(dirname "$(dirname "$NVCC")")"
  echo "nvcc: $NVCC (root $CUDA_ROOT, $("$NVCC" --version | tail -2 | head -1))"

  # PIP WHEELS SHIP VERSIONED SONAMES ONLY -- libcublas.so.13 exists, plain
  # libcublas.so does not, and cmake's FindCUDAToolkit wants the bare name.
  # MEASURED: without these links the configure fails to find cuBLAS even though
  # the library is right there. Creating them is idempotent and touches only the
  # DLAMI's own tree.
  for l in cublas cublasLt cudart nvJitLink cusparse cusolver; do
    for v in "$CUDA_ROOT"/lib/lib$l.so.*; do
      [ -e "$v" ] && ln -sf "$v" "$CUDA_ROOT/lib/lib$l.so" && break
    done
  done
  # libcuda.so (the DRIVER stub, not the toolkit) comes from the driver package
  # at /usr/lib/aarch64-linux-gnu and is already present -- do not link it here.
  export CUDACXX="$NVCC"
  export CUDAToolkit_ROOT="$CUDA_ROOT"
  export LD_LIBRARY_PATH="$CUDA_ROOT/lib:/usr/lib/aarch64-linux-gnu${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
  echo "$CUDA_ROOT" > {APP_DIR}/CUDA_ROOT
  stage nvcc-probe

  git clone --depth 1 --branch {LLAMA_CPP_REF} {LLAMA_CPP_REPO} {BUILD_DIR}/llama.cpp
  stage clone-{LLAMA_CPP_REF}

  # -DGGML_CUDA=ON is REQUIRED to fail the build if CUDA cannot be configured.
  # cmake's default behaviour on a missing toolkit is to carry on without it.
  # CMAKE_CUDA_ARCHITECTURES pins sm_{CUDA_ARCH} -- see CUDA_ARCH for why a broad
  # list is worse here than a narrow one.
  # LLAMA_CURL is what makes --hf-repo/--hf-file exist at all.
  cmake -S {BUILD_DIR}/llama.cpp -B {BUILD_DIR}/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH} \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF
  stage cmake-configure

  cmake --build {BUILD_DIR}/build --config Release --target llama-server -j"$(nproc)"
  stage cmake-build

  install -m 0755 {BUILD_DIR}/build/bin/llama-server {APP_DIR}/llama-server
  # ggml ships its backends as shared objects next to the binary; copy them or
  # llama-server loads with no CUDA backend registered and falls back to CPU --
  # the same silent slowness, arriving one step later.
  cp -a {BUILD_DIR}/build/bin/*.so* {APP_DIR}/ 2>/dev/null || true
  echo "{LLAMA_CPP_REF}-sm{CUDA_ARCH}" > {APP_DIR}/BUILD_ID
  # Point the unit at the CUDA libs the probe found. Done HERE rather than in the
  # unit template because the path is only known after install_runtime has run --
  # exactly the reason the PyTorch sibling rewrites its ExecStart at this point.
  sed -i "s|^LD_LIBRARY_PATH=.*|LD_LIBRARY_PATH={APP_DIR}:$CUDA_ROOT/lib:/usr/lib/aarch64-linux-gnu|" \
    {APP_DIR}/env
  stage install-binary
}}

# Assert the BUILT BINARY sees a CUDA device, before declaring the install done.
#
# This is the analogue of the PyTorch sibling's sm_75 arch-list assertion, and it
# exists for the same reason: `nvidia-smi` succeeding proves the driver works,
# not that the thing we compiled can use it. A CPU-only llama-server passes every
# check short of this one -- it starts, binds, and answers correctly.
#
# --list-devices is the binary's own view, so it cannot be satisfied by a
# different CUDA installation elsewhere on the box.
verify_gpu() {{
  nvidia-smi || {{
    echo "FATAL: nvidia-smi failed. On this rig that usually means the AMI is an" >&2
    echo "ARM64 DLAMI built for Graviton CPU inference -- it boots fine, has no GPU." >&2
    exit 1
  }}
  LD_LIBRARY_PATH={APP_DIR} {APP_DIR}/llama-server --list-devices | tee {APP_DIR}/devices.txt
  if ! grep -qi 'CUDA' {APP_DIR}/devices.txt; then
    echo "FATAL: the llama-server we just built lists no CUDA device." >&2
    echo "It would serve correctly on CPU at a fraction of the speed, which is" >&2
    echo "why this is fatal rather than a warning." >&2
    exit 1
  fi
}}

install_runtime
verify_gpu
stage gpu-verify

systemctl daemon-reload
systemctl enable --now {SERVICE_NAME}.service
stage service-start

touch {APP_DIR}/INSTALL_DONE
echo "[stage] INSTALL COMPLETE total $(($(date +%s) - _T0))s"
INSTEOF
chmod 700 {APP_DIR}/install.sh

cat >{APP_DIR}/env <<ENVEOF
# The CUDA runtime libraries live in the DLAMI's pip tree, not on the default
# loader path, so llama-server cannot start without this. install.sh rewrites the
# placeholder with the root the nvcc probe actually discovered -- the same
# discipline as the PyTorch sibling rewriting ExecStart after probing for torch.
LD_LIBRARY_PATH={APP_DIR}:/usr/lib/aarch64-linux-gnu
MODEL_NAME={model}
MODEL_FILE={MODEL_FILE}
RIG_NAME={MANAGED_BY}
CONTEXT_SIZE={CONTEXT_SIZE}
PARALLEL_SLOTS={PARALLEL_SLOTS}
LLAMA_PORT={LLAMA_PORT}
LLAMA_CPP_REF={LLAMA_CPP_REF}
CUDA_ARCH={CUDA_ARCH}
LD_LIBRARY_PATH={APP_DIR}
HF_HOME={APP_DIR}/models
ENVEOF
chmod 600 {APP_DIR}/env

# The HF token is fetched at boot into a root-only EnvironmentFile. It is NEVER
# placed in user data: instance metadata is readable by anything on the box.
#
# This checkpoint is UNGATED (verified 2026-09-02: gated=False, 508k downloads),
# so the token is not strictly needed to fetch it. Kept anyway -- it costs
# nothing, it survives a repoint at a gated checkpoint, and removing a security
# control because today's model does not need it is how the next one leaks.
#
# xtrace is off across this block on purpose: this script runs under `set -x`,
# and bash traces variable assignments WITH their values.
set +x
if ! command -v aws >/dev/null 2>&1; then
  echo "WARNING: aws CLI not on PATH; cannot read secret {HF_SECRET_ID}." >&2
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

# ExecStart is absolute and needs no post-install rewrite: the binary lands at a
# path this script chose. That whole class of hazard on the PyTorch sibling --
# install, verify and ExecStart resolving to three different interpreters -- does
# not exist when the artifact is a binary rather than a script plus a runtime.
#
# Restart=on-failure will NOT catch the failure this rig is most likely to hit.
# A CPU-only build does not fail; verify_gpu is what catches that, before the
# unit is ever enabled.
cat >/etc/systemd/system/{SERVICE_NAME}.service <<'UNITEOF'
[Unit]
Description=Gemma 4 E2B Q4_0 on T4G via llama.cpp
After=network-online.target

[Service]
Type=simple
EnvironmentFile={APP_DIR}/env
WorkingDirectory={APP_DIR}
ExecStart={APP_DIR}/llama-server {argv}
Restart=on-failure
RestartSec=10
# The first start downloads 3.35 GB before it binds a port. systemd's default
# start timeout would kill it mid-download on a slow link and then restart it,
# re-downloading from zero -- which is precisely how the sibling lost two full
# checkpoint pulls to a different cause.
TimeoutStartSec=1800

[Install]
WantedBy=multi-user.target
UNITEOF
systemctl daemon-reload

nohup bash {APP_DIR}/install.sh >/var/log/llamacpp-install.log 2>&1 &
echo "llama.cpp build started; follow /var/log/llamacpp-install.log"
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
            "Cloud-init does EVERYTHING here: it builds llama.cpp from "
            f"`{LLAMA_CPP_REF}` with `CMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}`, verifies the "
            "built binary sees a CUDA device, then enables the unit. There is no "
            "separate deploy step and no rig-owned payload — llama-server fetches "
            "its own checkpoint. Watch get_install_progress for INSTALL COMPLETE."
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
            f"Serving argv: `{_serve_argv(model_name, instance_type)}`\n"
            f"Build: `{_build_id()}`. There is no dtype flag — ggml picks per "
            "kernel, and Turing has no bf16 datapath for it to pick."
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

    Cloud-init does the whole job here: apt build deps, probe for nvcc, clone
    llama.cpp at the pinned LLAMA_CPP_REF, build llama-server with CUDA pinned to
    CUDA_ARCH, assert the BUILT BINARY lists a CUDA device, then enable the
    unit. **It starts serving on its own** — there is no deploy step, because
    there is no rig-owned payload and llama-server fetches its own checkpoint.

    The build is minutes on a Graviton2, so the reclamation window matters more
    here than on the siblings: spot is still the default, but a reclaim mid-build
    loses the whole thing rather than a pip install. Pass spot=False for
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
            "Building llama.cpp from source (there is no prebuilt aarch64 CUDA "
            "binary). Follow with get_install_progress; it starts serving by itself."
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


@mcp.tool(title="Verify GPU and the built binary's CUDA backend", annotations=READ_ONLY)
async def verify_gpu_arch(instance_id: str) -> str:
    """Check the driver sees a T4G AND that OUR llama-server can use it.

    Those are two claims and only the second one matters. `nvidia-smi` succeeding
    proves the AMI carries a driver; it says nothing about whether the binary we
    compiled has a CUDA backend. **A CPU-only llama.cpp build starts, binds,
    answers correctly, and is several times slower** — there is no error anywhere
    to find, which is why this rig asserts on the binary's own device list rather
    than on the driver.

    That is the direct analogue of the PyTorch sibling asserting `sm_75 in
    torch.cuda.get_arch_list()` rather than trusting `torch.cuda.is_available()`,
    and it exists for the same reason: the check that passes on a broken box is
    the one everybody reaches for first.
    """
    command = (
        "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true; "
        f"echo '--- build id ---'; cat {APP_DIR}/BUILD_ID 2>/dev/null || echo '(not built yet)'; "
        f"echo '--- llama-server devices ---'; "
        f"LD_LIBRARY_PATH={APP_DIR} {APP_DIR}/llama-server --list-devices 2>&1 || true"
    )
    try:
        output = await _ssm(instance_id, command, timeout=600)
        verdict = ""
        if "7.5" in output and "CUDA" in output:
            verdict = (
                "\n\n✅ Driver reports SM 7.5 and the built llama-server lists a CUDA "
                "device. This is the combination the rig requires."
            )
        elif "not built yet" in output:
            verdict = (
                "\n\n⏳ The binary does not exist yet — the build is still running or "
                "failed. Check `get_install_progress`."
            )
        elif "CUDA" not in output:
            verdict = (
                "\n\n❌ llama-server lists NO CUDA device. Either the AMI is a driverless "
                "ARM64 DLAMI (AWS ships those; they boot fine on a G5g and have no GPU), "
                "or the build configured without CUDA. **It will still serve, correctly "
                "and slowly** — do not benchmark this box. Re-check the `nvcc-probe` and "
                "`cmake-configure` stages in /var/log/llamacpp-install.log."
            )
        elif "7.5" not in output:
            verdict = (
                f"\n\n⚠️ The device is not SM 7.5, but the build pinned "
                f"`CMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}`. Those must agree or the kernels "
                "will not load. Set CUDA_ARCH to match and rebuild."
            )
        return f"### GPU probe on `{instance_id}`\n\n```\n{output}\n```{verdict}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Restart the llama-server unit", annotations=WRITE)
async def restart_llama_server(instance_id: str) -> str:
    """Restart the serving unit and report what came back up.

    **This rig has no deploy step, and that is a structural difference from every
    sibling rather than a missing feature.** The JAX and PyTorch rigs ship their
    own `*_openai_server.py` over SSM because "our Gemma 4 port" is not a
    published artifact; here the server IS the artifact, built from a pinned
    llama.cpp ref, and it fetches its own checkpoint. There is nothing to ship.

    So the entire `make skill` / payload-root trap that bit the PyTorch sibling —
    deploying through the registered MCP server ships the PREVIOUS `make skill`
    output — cannot happen on this rig. The only way to change what runs here is
    to change LLAMA_CPP_REF or the serve flags and relaunch.

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
            "⚠️ `Result=success` with `NRestarts=0` does NOT mean nothing went wrong — "
            "unattended-upgrades restarts services it upgrades and reports exactly that. "
            "The bootstrap masks it, but check the PID actually changed if a restart "
            "looks unexplained."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get llama.cpp build progress", annotations=READ_ONLY)
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
            "if [ -f /var/log/llamacpp-install.log ]; then "
            f"tail -n {tail} /var/log/llamacpp-install.log; "
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
                "\n\n❌ cloud-init finished but never wrote /var/log/llamacpp-install.log, so "
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


@mcp.tool(title="Get llama-server logs", annotations=READ_ONLY)
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
    llama-server has no degeneracy counter of its own, so unlike the siblings the
    verdict here is computed locally — see the heuristic below. Weaker, and
    labelled as weaker, but it still cannot pass a `': ok: ok: ok…'` reply.

    It also reads the decode rate either side of its own probe, so a health check
    doubles as a "did this box actually get a GPU" signal: a CPU-only build
    answers this correctly at a rate that gives it away immediately.
    """
    try:
        endpoint = await get_endpoint(instance_id)
        if not endpoint.startswith("📡"):
            return endpoint
        base = endpoint.strip("📡 `")
        metrics_url = base.replace("/v1", "/metrics")
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
            after, _ = _parse_prom((await client.get(metrics_url)).text)

        body = chat.json()
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = body.get("usage") or {}
        # llama-server has NO degeneracy counter, so this rig detects a token
        # loop itself rather than pretending the server does.
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

        try:
            health_body = health.json()
        except Exception:
            health_body = {}
        # llama-server's /health is a bare {"status":"ok"} with no build field, so
        # the served build comes from the file the bootstrap stamped. That is a
        # weaker claim than the siblings' (they read it out of the running
        # process) and is labelled as such below rather than quietly equated.
        served_build = health_body.get("build_id") or "(not reported by llama-server)"
        local_build = _build_id()

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
            f"- Degenerate (local heuristic, not a server verdict): "
            f"**{'YES — token loop' if degenerate else 'no'}**",
            f"- Build configured here: `{local_build}`",
            f"- Build reported by the server: `{served_build}`",
        ]
        rate = _decode_rate(after)
        if rate:
            lines.append(f"- Cumulative decode: **{rate:.2f} tok/s**")
            # A CPU-only llama.cpp build is the failure this rig is most likely to
            # ship, and it passes every other check in this function. The exact
            # threshold does not matter: the gap is roughly an order of magnitude,
            # not a few percent. Warm-up noise cannot cross it.
            if rate < 3.0:
                lines.append(
                    "- ⚠️ **That rate is consistent with a CPU-only build.** Run "
                    "`verify_gpu_arch` — a llama.cpp built without CUDA serves "
                    "correctly and slowly, and nothing else here will catch it."
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


def _parse_prom(text: str) -> tuple[dict, str | None]:
    """Split a Prometheus exposition into (samples, model).

    Pure and offline so tests can pin it without a served endpoint.

    **The metric names here are llama.cpp's, not this family's, and that is the
    one place this rig cannot follow the house convention.** Every sibling emits
    `tpu_jax_decode_tokens_per_second` — wrong as description, right as an
    identifier, because every benchmark report in the family compares on that
    exact string. This rig cannot: llama-server's exposition is fixed at
    `llamacpp:*` and there is no server of ours in the path to rename it.

    So the translation happens HERE, in `_decode_rate`, and the rig's reports
    quote the translated number. Do not "fix" this by renaming llama.cpp's
    metrics in a scraper — the raw names are what an operator sees in
    `curl /metrics`, and a shim that disagrees with the wire is worse than one
    that admits the mapping.
    """
    samples, served_model = {}, None
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
        served_model = tags.pop("model", None) or served_model
        rendered = ",".join(f'{k}="{v}"' for k, v in sorted(tags.items()))
        samples[f"{name}{{{rendered}}}" if rendered else name] = value
    return samples, served_model


# The mapping from llama.cpp's exposition onto the gauge this family compares on.
#
# llama-server separates prefill from decode at the source, which is exactly the
# split the house convention exists to preserve:
#
#   llamacpp:prompt_tokens_total            <- prefill, NOT this
#   llamacpp:prompt_seconds_total           <- prefill, NOT this
#   llamacpp:tokens_predicted_total         <- decode tokens
#   llamacpp:tokens_predicted_seconds_total <- decode seconds
#
# tokens_predicted / tokens_predicted_seconds is therefore like-for-like with
# `tpu_jax_decode_tokens_per_second`. `llamacpp:predicted_tokens_seconds` is a
# gauge of the same thing but describes only the last request; the counters are
# cumulative and are what a benchmark should quote.
_DECODE_TOKENS = "llamacpp:tokens_predicted_total"
_DECODE_SECONDS = "llamacpp:tokens_predicted_seconds_total"
_PREFILL_TOKENS = "llamacpp:prompt_tokens_total"
_PREFILL_SECONDS = "llamacpp:prompt_seconds_total"


def _decode_rate(samples: dict) -> float | None:
    """Cumulative decode tok/s, prefill and HTTP excluded.

    This is THE number for this rig. The PyTorch sibling measured end-to-end and
    decode disagreeing by up to 36% on the same rows, which is why the family
    quotes the gauge and not the round trip.
    """
    tokens = samples.get(_DECODE_TOKENS, 0.0)
    seconds = samples.get(_DECODE_SECONDS, 0.0)
    return tokens / seconds if tokens and seconds else None


@mcp.tool(title="Get serving metrics", annotations=READ_ONLY)
async def get_metrics(instance_id: str) -> str:
    """Read llama-server's Prometheus metrics and derive the decode rate.

    Requires `--metrics`, which is off by default in llama.cpp and which
    `_serve_argv` therefore always passes. Without it this returns a 404, not an
    empty exposition.

    The derived decode figure excludes prefill and HTTP, so it is comparable to
    the `tpu_jax_decode_tokens_per_second` gauge every sibling report quotes —
    see the mapping note above `_decode_rate`. `query_model`'s rate is not.
    """
    try:
        endpoint = await get_endpoint(instance_id)
        if not endpoint.startswith("📡"):
            return endpoint
        base = endpoint.strip("📡 `")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(base.replace("/v1", "/metrics"))
        if response.status_code == 404:
            return (
                "❌ /metrics returned 404. llama-server disables it by default — the "
                "unit must pass `--metrics`. Check the rendered ExecStart."
            )
        if response.status_code != 200:
            return f"❌ /metrics returned {response.status_code} — is {SERVICE_NAME} serving?"

        samples, served_model = _parse_prom(response.text)
        if not samples:
            return "❌ /metrics returned 200 but exposed no samples."

        lines = [f"### Serving metrics on `{instance_id}`", ""]
        if served_model:
            lines += [f"Served checkpoint: `{served_model}`", ""]
        lines += [
            f"Build: `{_build_id()}` — llama.cpp `{LLAMA_CPP_REF}`, "
            f"`CMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}`.",
            "",
            "| Metric | Value |", "| --- | ---: |",
        ]
        for key in sorted(samples):
            value = samples[key]
            shown = f"{value:.2f}" if value % 1 else f"{int(value)}"
            lines.append(f"| `{key}` | {shown} |")

        decode = _decode_rate(samples)
        if decode:
            lines += [
                "",
                f"**Decode: {decode:.2f} tok/s** over "
                f"{int(samples.get(_DECODE_TOKENS, 0))} generated tokens in "
                f"{samples.get(_DECODE_SECONDS, 0.0):.1f}s of decode. Prefill and HTTP "
                "excluded, so this is the figure comparable to the siblings' "
                "`tpu_jax_decode_tokens_per_second`.",
            ]
        prefill_t = samples.get(_PREFILL_TOKENS, 0.0)
        prefill_s = samples.get(_PREFILL_SECONDS, 0.0)
        if prefill_t and prefill_s:
            lines.append(
                f"Prefill: {prefill_t / prefill_s:.1f} tok/s over {int(prefill_t)} prompt "
                "tokens — reported separately on purpose. Never add it to the line above."
            )
        slots = samples.get("llamacpp:n_busy_slots_per_decode")
        if slots is not None:
            lines.append(
                f"\nAverage busy slots per decode: **{slots:.2f}** against "
                f"`--parallel {PARALLEL_SLOTS}`. A figure near 1 on a multi-slot run means "
                "the load generator never actually concurrent-loaded the server, which is "
                "how a batching measurement quietly becomes a single-stream one."
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
    mm = f"`{MMPROJ_FILE}`" if MMPROJ_FILE else "not loaded (text-only)"
    return f"""### {RIG_NAME}

Serving `{MODEL_NAME}` / `{MODEL_FILE}` with **llama.cpp** on **EC2 G5g** — AWS
Graviton2 (aarch64) host, NVIDIA **T4G** GPU (Turing, SM 7.5, 15360 MiB).

**The only rig here serving 4-bit weights on a GPU.** Every other route is
closed: vLLM 0.26.0 has no `gguf` module at all (CUDA or TPU), no GGUF reader
exists in the JAX ecosystem, and transformers 5.12.1 reads the file but
dequantizes to fp32 and silently drops 35 `layer_scalar` tensors. See the root
`QUANTIZATION.md`.

| Setting | Value |
| --- | --- |
| Region | `{AWS_REGION}` |
| Instance type | `{INSTANCE_TYPE}` ({_gpu_count(INSTANCE_TYPE)}x T4G, {_host_memory_gb(INSTANCE_TYPE)} GiB RAM) |
| Engine | llama.cpp `{LLAMA_CPP_REF}`, built from source |
| CUDA arch | `sm_{CUDA_ARCH}` (pinned; not a JIT/PTX fallback) |
| Build id | `{_build_id()}` |
| GGUF file | `{MODEL_FILE}` |
| Projector | {mm} |
| Context | {CONTEXT_SIZE} tokens across {PARALLEL_SLOTS} slot(s) |
| GPU layers | {N_GPU_LAYERS} (all) |
| Service | `{SERVICE_NAME}` (systemd, not docker) |
| Managed-by tag | `{MANAGED_BY}` |
| Root volume | {ROOT_VOLUME_GB} GB gp3 @ {ROOT_VOLUME_THROUGHPUT_MBPS} MiB/s, {ROOT_VOLUME_IOPS} IOPS |

**It compiles, and that is not incidental.** llama.cpp publishes no prebuilt
Linux aarch64 CUDA binary — verified against release b10760 (2026-09-02): the
arm64 Linux assets are CPU and Vulkan, every `-cuda-` Linux asset is x64, and
the only arm64 CUDA build targets Windows. The Ollama sibling ships a working
aarch64 CUDA binary instead; that asymmetry is the main reason both rigs exist.

**Three silent failures, all asserted rather than trusted.** A CPU-only build
serves correctly and slowly; a partial GPU offload does the same; a broad arch
list can leave the device on a JIT'd PTX path whose warm-up gets recorded as an
engine property. `verify_gpu_arch` greps the built binary's own `--list-devices`,
and `verify_model_health` flags a decode rate consistent with CPU.

**What Q4_0 buys, and what it does not.** Streamed bytes drop 4.514 GB → 1.407 GB
and residency 10.209 GB → 3.35 GB on a 14.07 GB chip. **Do not turn that into a
throughput prediction.** Decode at B=1 on this silicon is launch-bound, not
bandwidth-bound: the sibling cut 3.5 GB of weights for 0.0%. The headroom is what
pays for batching — 7.84x measured on the PyTorch sibling. Raise `PARALLEL_SLOTS`
in a run whose report says so, never as a default.

**NOTHING HAS BEEN MEASURED ON THIS RIG.** Every figure above is either arithmetic
from the artifact's tensor table or a sibling's number, labelled as such. Quote
the decode rate `get_metrics` derives from `llamacpp:tokens_predicted_*`, never
an end-to-end rate — the PyTorch sibling measured the two disagreeing by 36%.

Order of operations: `create_g5g_instance` → `get_install_progress` →
`verify_gpu_arch` → `verify_model_health` → `get_metrics`. There is no deploy
step; cloud-init builds and starts the service on its own.
"""


if __name__ == "__main__":
    mcp.run()
