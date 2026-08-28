import asyncio
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, Optional

import httpx
from google.cloud import secretmanager
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from openai import AsyncOpenAI
from pydantic import Field

# This rig's identity. The directory name is the single identifier everything else derives
# from — the MCP server name, the log channel, and the zone-status cache directory. Sibling
# rigs share a GCP project and zone, so a shared constant here is how one rig ends up
# answering for, or clobbering the state of, another. It is a literal rather than
# basename(__file__) because the installed skill copy lives at
# .claude/skills/<skill>/mcp/server.py, where deriving from the path would yield "mcp".
RIG_NAME = "gce-jaxrust-v6e1-2b"

# Setup logging
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(RIG_NAME)

# Initialize FastMCP server. The name has to match the key the client registers this
# server under, because that key prefixes every tool — mcp__<key>__find_tpu. Every
# sibling rig used to answer to "tpu-devops", so with more than one registered you could
# not tell which rig a tool call would reach. It now defaults to the rig directory name;
# MCP_SERVER_NAME overrides it, and project-setup.sh passes the key it registered.
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", RIG_NAME)
mcp = FastMCP(MCP_SERVER_NAME)

# Annotation presets — hints that let clients (e.g. permission layers) auto-allow
# reads and require confirmation before destructive calls.
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

# --- Configuration ---


def _resolve_project_id() -> str:
    """GOOGLE_CLOUD_PROJECT env var, falling back to the active gcloud config."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if project:
        return project
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        project = result.stdout.strip()
    except Exception:
        project = ""
    return "" if project == "(unset)" else project


PROJECT_ID = _resolve_project_id()
if not PROJECT_ID:
    logger.warning(
        "No GCP project configured: set GOOGLE_CLOUD_PROJECT or run "
        "`gcloud config set project <id>`. All gcloud-backed tools will fail until then."
    )
# europe-west4-a is where a single v6e chip has actually been granted on this path,
# first try and with no queueing. Quota is REGIONAL and capacity is ZONAL, and they
# diverge sharply: three of five zones checked held full quota and no chips at all.
# So this is a starting point, not a guarantee — `find_tpu_vm` sweeps siblings, and
# `probe_zone_capacity` tells a stockout apart from a quota wall before you wait.
ZONE = os.getenv("GOOGLE_CLOUD_ZONE", "europe-west4-a")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "europe-west4")
# A v6e-1 chip has 32GB of HBM, so the default payload is an E2B-class checkpoint.
# Raise it with MODEL_NAME on bigger shapes.
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-E2B-it")
# Secret Manager secret holding the Hugging Face token. The startup script fetches it by
# id at boot, so a rotated or per-project secret only needs this to change.
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "hf-token")
# Documentation on this path: ACCELERATOR_TYPE is the Cloud TPU API's spelling, kept so
# the rig name and the benchmark reports line up. What Compute Engine actually consumes
# is the MACHINE TYPE derived from it (_gce_machine_type); `gcloud compute instances
# create` would reject --accelerator-type outright.
ACCELERATOR_TYPE = os.getenv("ACCELERATOR_TYPE", "v6e-1")
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
# On this path the instance IS the node: no Queued Resource indirection and no derived
# <resource_id>-node. Defaults to the rig name so two rigs in one project and zone cannot
# collide. Set it only to reach an instance created under an older name — a rename
# orphans it and it keeps billing.
INSTANCE_NAME = os.getenv("INSTANCE_NAME", RIG_NAME)

# --- Compute Engine provisioning ---------------------------------------------------
# The four ways to ask Compute Engine for a chip. These are this rig's lowercase labels;
# _provisioning_flags() maps them to gcloud's SCREAMING_CASE --provisioning-model values.
#   flex-start  queues via Dynamic Workload Scheduler (up to a 2h wait), then runs
#               uninterrupted for --max-run-duration (10 min to 7 days). Cheapest on v6e.
#   spot        never queues — it fails fast and names the reason, which is what makes
#               it the capacity probe (see `probe_zone_capacity`). Costs MORE than
#               flex-start on v6e, and is reclaimable with ~30s notice.
#   on-demand   full price, no run limit, the only model that spends family quota first.
#   reservation-bound  binds to a future reservation; has no queued-resource equivalent.
PROVISIONING_MODEL = os.getenv("PROVISIONING_MODEL", "flex-start")
# How long a flex-start request stays queued, and how long the VM may run once granted.
# Unlike the TPU API, --max-run-duration is not flex-start's alone here: spot and
# on-demand take it too, paired with --instance-termination-action=DELETE.
REQUEST_VALID_FOR = os.getenv("REQUEST_VALID_FOR", "2h")
MAX_RUN_DURATION = os.getenv("MAX_RUN_DURATION", "4h")
# The image default is 10GB, which cannot hold the vLLM TPU image. Undersizing fails
# late — after a clean boot, mid-pull, a long way from the flag that caused it.
BOOT_DISK_SIZE_GB = int(os.getenv("BOOT_DISK_SIZE_GB", "200"))
# Replaces the TPU API's --runtime-version. Pin the FAMILY, never a dated build: images
# ship roughly weekly and every superseded build goes DEPRECATED within the fortnight.
IMAGE_FAMILY = os.getenv("IMAGE_FAMILY", "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e")
IMAGE_PROJECT = os.getenv("IMAGE_PROJECT", "ubuntu-os-accelerator-images")
# Required for RESERVATION_BOUND and ignored by the other three: gcloud accepts that
# model only alongside --reservation-affinity=specific, so the model alone is not a
# complete request. Create the reservation out of band; it is a separate resource.
RESERVATION_NAME = os.getenv("RESERVATION_NAME", "")

# --- Compute Engine quota: the trap this path is most often lost to ------------------
# The two control planes meter against DISJOINT pools. TPU API quota does not come with
# you, and the regional view most people reach for cannot see the v6e metrics at all:
# `gcloud compute regions describe` returns only the older v5-era ids. v6e lives in the
# Cloud Quotas API and has to be asked for by name.
#
# WHICH ID GOVERNS WHICH MODEL — and this is the counterintuitive part:
#   flex-start  -> GCE_SPOT_QUOTA_ID first, GCE_QUOTA_ID as a documented FALLBACK
#   spot        -> GCE_SPOT_QUOTA_ID   (the preemptible pool)
#   on-demand   -> GCE_QUOTA_ID        (regional, family-wide, tpu_family=CT6E)
#
# Flex-start spends PREEMPTIBLE quota even though it is not preemptible in behaviour —
# once granted it runs uninterrupted. Google's provisioning-models page states it, and
# the second sentence matters as much as the first: "If your project lacks preemptible
# quota, then standard quota is consumed." So a region is usable for flex-start if
# EITHER pool has room; never write a region off on one listing alone.
#
# Their DEFAULTS ARE OPPOSITE, which is what makes a single reading misleading: a region
# absent from the family listing inherits 0, a region absent from the preemptible
# listing inherits 1536. Regions have been written off for a day on that alone.
#
# There is no non-preemptible per-generation v6e id at all — no TPU-V6E-per-project-region
# exists — which is why on-demand falls back to the generic family quota. v4/v5e/v5p each
# publish their own dedicated pair, so this is a v6e and TPU7x quirk, not a rule.
GCE_QUOTA_ID = os.getenv("GCE_QUOTA_ID", "TPUS-PER-TPU-FAMILY-per-project-region")
GCE_SPOT_QUOTA_ID = os.getenv("GCE_SPOT_QUOTA_ID", "PREEMPTIBLE-TPU-V6E-per-project-region")
GCE_TPU_FAMILY = os.getenv("GCE_TPU_FAMILY", "CT6E")


def _family(accelerator: str) -> str:
    """Normalizes an accelerator string to its family ('v5litepod-1' -> 'v5e')."""
    family = accelerator.lower().split("-")[0]
    return "v5e" if family == "v5litepod" else family


# This rig's lowercase provisioning labels -> the gcloud enum. `flex-start` and
# `FLEX_START` are the same request to different APIs; keeping the mapping in one place
# is what stops a second, drifting copy appearing. gcloud validates the enum client-side,
# so a wrong value here is the one failure on this path that costs nothing.
_PROVISIONING_MODELS = {
    "flex-start": "FLEX_START",
    "spot": "SPOT",
    "on-demand": "STANDARD",
    "reservation-bound": "RESERVATION_BOUND",
}


def _provisioning_flags(model: str, request_valid_for: str, max_run_duration: str) -> list[str]:
    """gcloud flags for one provisioning model. Raises ValueError on an unknown label.

    --request-valid-for-duration is flex-start's queue knob and is rejected by the other
    models, so it is not applied unconditionally. --max-run-duration is applied to all of
    them: on Compute Engine (unlike the TPU API) it is not flex-start's alone, and pairing
    it with instance-termination-action=DELETE is what makes a demo box clean up after
    itself. RESERVATION_BOUND is incomplete without --reservation-affinity, so that is
    attached here rather than left to the caller.
    """
    gcloud_model = _PROVISIONING_MODELS.get(model)
    if gcloud_model is None:
        raise ValueError(f"unknown provisioning model '{model}'; expected one of {sorted(_PROVISIONING_MODELS)}")
    flags = [f"--provisioning-model={gcloud_model}"]
    if gcloud_model == "FLEX_START":
        flags.append(f"--request-valid-for-duration={request_valid_for}")
    if gcloud_model == "RESERVATION_BOUND":
        if not RESERVATION_NAME:
            raise ValueError(
                "provisioning_model='reservation-bound' needs RESERVATION_NAME set to an existing reservation"
            )
        flags += ["--reservation-affinity=specific", f"--reservation={RESERVATION_NAME}"]
    if max_run_duration:
        flags += [f"--max-run-duration={max_run_duration}", "--instance-termination-action=DELETE"]
    return flags


# Bare JAX dev VMs (workload="jax"). No docker, no vLLM, no HF token.
# Ubuntu 22.04's system interpreter is 3.10, which pins JAX to an old release;
# the startup script installs this version from deadsnakes instead and pip-installs
# into it directly (no venv, per this repo's standard).
JAX_PYTHON_VERSION = os.getenv("JAX_PYTHON_VERSION", "3.13")
# libtpu resolves from the JAX releases index, not PyPI — the startup script
# passes -f for it. Pin here (e.g. "jax[tpu]==0.11.0") for reproducible runs.
JAX_PIP_SPEC = os.getenv("JAX_PIP_SPEC", "jax[tpu]")
# Best-effort extras installed after the TPU stack; failure is non-fatal.
# CPU debug boxes: correctness work off the TPU. Memory decides what you can do,
# vCPUs decide how long you wait. These figures are MEASURED peak RSS, not derived
# from weight sizes — XLA:CPU allocates far more than the parameters occupy:
#
#   E2B  load + short generation   26 GiB peak   (6.6 GiB of weights)
#   31B  load only                 48 GiB peak   (19 GiB of weights)
#   31B  load + one forward pass   >64 GiB       (OOM-killed on a 64 GiB box)
#
# So the useful tiers are 64 GiB to inspect a 31B parameter tree and 128 GiB to
# actually run it. 32 GiB only covers E2B. Guessing from checkpoint size
# underestimates this by roughly 2x; do not.
#
# e2-highmem-16 (16 vCPU / 128 GiB) is the default because it is the cheapest SKU
# that runs a 31B forward pass, and CPU passes on a 31B are slow enough that the
# cores matter. Prefer Spot for short sessions, but note a preemption takes the
# checkpoint download with it — a 23 GB re-fetch.
CPU_DEBUG_MACHINE_TYPE = os.getenv("CPU_DEBUG_MACHINE_TYPE", "e2-highmem-16")
CPU_DEBUG_PIP_SPEC = os.getenv(
    "CPU_DEBUG_PIP_SPEC",
    "jax numpy scipy ml_dtypes safetensors huggingface_hub transformers sentencepiece 'jinja2>=3.1'",
)
JAX_PIP_EXTRAS = os.getenv(
    "JAX_PIP_EXTRAS",
    "numpy scipy ml_dtypes safetensors huggingface_hub transformers sentencepiece 'jinja2>=3.1'",
)

# --- Rust/XLA dev VMs (workload="jaxrust") -----------------------------------
# The serving path here is Rust: the graph is built in rlx's JAX-shaped IR,
# lowered to HLO, and executed through libtpu's PJRT plugin. Python appears on
# these VMs only as the delivery mechanism for the libtpu wheel, which is not
# published anywhere but the JAX release index.
#
# Pinned, not floating. Two of the crates in the engine are 2024-edition, so the
# distro rustc (1.75 on Ubuntu 22.04) cannot build them at all — and "whatever
# rustup installs today" is not a thing to discover in the middle of a flex-start
# window that took two hours to arrive.
RUST_TOOLCHAIN = os.getenv("RUST_TOOLCHAIN", "1.90.0")
# libtpu alone, --no-deps: no JAX, no jaxlib. Pin here for a reproducible run.
LIBTPU_SPEC = os.getenv("LIBTPU_SPEC", "libtpu")
# Cargo features for the engine build. `gemma` pulls in rlx-gemma, which is
# GPL-3.0-only while the rest of this rig is permissive — see rust/NOTICE.md
# before shipping a binary built with it. Drop it to build the probe and the
# server shell alone; the capacity path does not need the model.
JAXRUST_CARGO_FEATURES = os.getenv("JAXRUST_CARGO_FEATURES", "tpu,gemma")
# Where the engine source lands on the VM, and where its binaries end up.
JAXRUST_REMOTE_DIR = os.getenv("JAXRUST_REMOTE_DIR", "~/gemma4-engine")

# find_tpu records per-zone provisioning outcomes here so later sweeps can skip
# zones that never delivered capacity. Lives outside the skill directory so
# reinstalls (`make skill`, project-setup.sh) don't wipe the learned state.
#
# Keyed by RIG_NAME, not shared: the sibling rigs request different accelerator types, and
# a zone that rejects one says nothing about another. Sharing one file meant a failure
# recorded by one rig silently suppressed a different rig's scan of that zone. Override the
# whole path with TPU_ZONES_STATUS_FILE.
STATUS_FILE = os.getenv(
    "TPU_ZONES_STATUS_FILE",
    os.path.join(os.path.expanduser("~"), ".cache", RIG_NAME, "tpu_zones_status.md"),
)

# --- Helper Functions ---


async def run_command(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Runs a shell command asynchronously."""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return process.returncode or 0, stdout.decode().strip(), stderr.decode().strip()
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        stdout, stderr = await process.communicate()
        return -1, stdout.decode().strip(), f"Timeout after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def _zone(zone: Optional[str]) -> str:
    """Resolves an optional zone argument against the current global default.

    Tools take `zone=None` rather than a `zone=ZONE` default so that when
    `find_tpu_vm` moves the global ZONE to wherever capacity was found, follow-up
    calls without an explicit zone target the new zone (import-time defaults
    would keep pointing at the old one).
    """
    return zone or ZONE


def _build_ssh_cmd(remote_cmd: str, instance_name: Optional[str], zone: str) -> tuple[list[str], str]:
    """Builds the gcloud SSH argv for the serving host, and returns (argv, target).

    A ct6e-* instance is an ordinary Compute Engine instance that happens to carry a TPU,
    so `gcloud compute tpus tpu-vm ssh` cannot reach it — it is the call site most often
    left behind in a migration, because every tool you reach for *after* something has
    gone wrong (container management, log tailing, journalctl, benchmarks) goes through
    here. One builder, so there is one place to get it right.
    """
    target = instance_name or INSTANCE_NAME
    argv = [
        "gcloud",
        "compute",
        "ssh",
        target,
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--command",
        remote_cmd,
    ]
    return argv, target


async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    try:
        response = await asyncio.to_thread(client.access_secret_version, request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception:
        return None


# Startup-script templates are hand-maintained inside the skill, next to the
# *deployed* copy of this file. When server.py runs from the repo root instead
# (tests, `python3 server.py`, an MCP registration pointing at the source), the
# templates are not siblings — so search the skill directory too rather than
# failing at VM-creation time with a confusing "no such file".
_TEMPLATE_SEARCH_DIRS = (
    os.path.dirname(os.path.abspath(__file__)),
    # Derived, not spelled out: the skill directory carries the rig name, so a rename that
    # updates RIG_NAME keeps this search path pointing at the right place.
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".claude", "skills", f"{RIG_NAME}-management", "mcp"),
)


def _find_rust_workspace() -> str:
    """Absolute path to the `rust/` workspace, or "" if it is not beside us.

    Same problem `_read_template` solves: this file runs either from the rig root
    or from the installed skill copy four directories down, and the engine source
    only ever lives at the rig root.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "rust"),
        os.path.abspath(os.path.join(here, "..", "..", "..", "..", "rust")),
    ):
        if os.path.isdir(os.path.join(candidate, "gemma4-engine")):
            return candidate
    return ""


def _build_scp_cmd(local: str, remote: str, instance_name: Optional[str], zone: str) -> tuple[list[str], str]:
    """gcloud SCP argv for the serving host. Same reason as `_build_ssh_cmd`:
    `gcloud compute tpus tpu-vm scp` cannot reach a ct6e-* instance."""
    target = instance_name or INSTANCE_NAME
    argv = [
        "gcloud",
        "compute",
        "scp",
        local,
        f"{target}:{remote}",
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
    ]
    return argv, target


def _read_template(filename: str) -> str:
    """Returns the contents of a startup-script template.

    Raises RuntimeError naming every location searched; callers must not create
    infrastructure with a missing or broken startup script.
    """
    tried = []
    for directory in _TEMPLATE_SEARCH_DIRS:
        path = os.path.join(directory, filename)
        tried.append(path)
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    return f.read()
            except OSError as e:
                raise RuntimeError(f"Cannot read startup script template {path}: {e}") from e
    raise RuntimeError(f"Startup script template '{filename}' not found. Searched: " + ", ".join(tried))


def _get_formatted_startup_script(model_name: str, zone: str, tp_size: Optional[int] = None) -> str:
    """Formats the startup script template. The Hugging Face token is NOT interpolated —
    the template fetches it from Secret Manager at boot, so it never lands in instance
    metadata (readable by anyone with compute.instances.get) or on local disk.

    Raises RuntimeError if the template is missing or malformed; callers must not
    create infrastructure with a broken startup script.
    """
    template = _read_template("startup_script_template.sh")
    try:
        return template.format(
            project_id=PROJECT_ID,
            zone=zone,
            model_name=model_name,
            hf_secret_id=HF_SECRET_ID,
            tp_size=tp_size if tp_size is not None else TENSOR_PARALLEL_SIZE,
            limit_mm_per_prompt_env='export VLLM_LIMIT_MM_PER_PROMPT=\'{"image":4,"audio":1}\'',
        )
    except Exception as e:
        raise RuntimeError(f"Cannot render startup_script_template.sh: {e}") from e


def _get_cpu_debug_startup_script(zone: str) -> str:
    """Formats the CPU debug startup script: the same JAX stack as the TPU VMs
    minus libtpu, plus safetensors/transformers for loading real checkpoints.

    Raises RuntimeError if the template is missing or malformed.
    """
    template = _read_template("startup_script_cpu_template.sh")
    try:
        return template.format(
            project_id=PROJECT_ID,
            zone=zone,
            python_version=JAX_PYTHON_VERSION,
            pip_spec=CPU_DEBUG_PIP_SPEC,
        )
    except Exception as e:
        raise RuntimeError(f"Cannot render startup_script_cpu_template.sh: {e}") from e


def _get_jax_startup_script(zone: str) -> str:
    """Formats the bare JAX startup script: installs a current CPython from
    deadsnakes plus jax[tpu] (libtpu from the JAX releases index) on the bare VM
    and asserts a TPU device is visible — no docker, no vLLM, no HF token.

    Raises RuntimeError if the template is missing or malformed; callers must not
    create infrastructure with a broken startup script.
    """
    template = _read_template("startup_script_jax_template.sh")
    try:
        return template.format(
            project_id=PROJECT_ID,
            zone=zone,
            python_version=JAX_PYTHON_VERSION,
            jax_pip_spec=JAX_PIP_SPEC,
            jax_pip_extras=JAX_PIP_EXTRAS,
        )
    except Exception as e:
        raise RuntimeError(f"Cannot render startup_script_jax_template.sh: {e}") from e


def _get_jaxrust_startup_script(zone: str) -> str:
    """Formats the Rust/XLA startup script: a pinned Rust toolchain, the build
    dependencies pjrt-sys needs (protoc and clang, both of which fail late and
    obscurely when missing), and libtpu — asserting that the .so it installed
    actually exports GetPjrtApi.

    It stops at a prepared environment. Nothing is fetched, built or served here;
    `deploy_jaxrust_engine` does that over SSH once the VM is up.

    Raises RuntimeError if the template is missing or malformed; callers must not
    create infrastructure with a broken startup script.
    """
    template = _read_template("startup_script_jaxrust_template.sh")
    try:
        return template.format(
            project_id=PROJECT_ID,
            zone=zone,
            rust_toolchain=RUST_TOOLCHAIN,
            libtpu_spec=LIBTPU_SPEC,
        )
    except Exception as e:
        raise RuntimeError(f"Cannot render startup_script_jaxrust_template.sh: {e}") from e


def _write_startup_script(content: str) -> str:
    """Writes a startup script to a private (0600) temp file and returns its path.
    Callers must unlink it after the gcloud call completes."""
    fd, path = tempfile.mkstemp(prefix="tpu-startup-", suffix=".sh")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


async def discover_vllm_url() -> Optional[str]:
    """Finds the URL of a RUNNING TPU VM serving vLLM.

    Never hardcode an endpoint: discovery is dynamic (RUNNING TPU instance -> external
    or internal IP -> :8000). Two field shapes moved with the control plane and neither
    throws when you get it wrong — `status: RUNNING` replaced `state: READY`, and the
    external IP moved from networkEndpoints[].accessConfig.externalIp to
    networkInterfaces[].accessConfigs[].natIP. A stale copy of either just quietly
    finds nothing.
    """
    gce_cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={PROJECT_ID}",
        f"--filter={_GCE_TPU_FILTER} AND status=RUNNING",
        "--format=value(name,networkInterfaces[0].accessConfigs[0].natIP,networkInterfaces[0].networkIP)",
    ]
    rc, stdout, _ = await run_command(gce_cmd)
    if rc == 0 and stdout:
        for line in stdout.splitlines():
            parts = line.split()
            name = parts[0] if parts else "?"
            for ip in parts[1:]:
                if ip:
                    url = f"http://{ip}:8000"
                    logger.info(f"📡 Found RUNNING TPU VM {name} at {url}")
                    return url
    return None


async def _get_served_model_id(url: str) -> Optional[str]:
    """The model id vLLM actually loaded — the only id it answers to. Deploy-time
    model_name overrides mean this can differ from the configured MODEL_NAME."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/v1/models")
        if res.status_code == 200:
            data = res.json().get("data", [])
            if data:
                return data[0].get("id")
    except Exception:
        pass
    return None


async def get_vllm_client() -> tuple[AsyncOpenAI, str]:
    """Returns an AsyncOpenAI client for the vLLM service plus the model id it is
    actually serving (falling back to MODEL_NAME if /v1/models is unreachable)."""
    url = await discover_vllm_url()
    if not url:
        raise Exception(f"No active vLLM service found (zone {ZONE}).")
    model_id = await _get_served_model_id(url) or MODEL_NAME
    return AsyncOpenAI(base_url=f"{url}/v1", api_key="not-needed"), model_id


@mcp.tool(title="Verify model health", annotations=READ_ONLY)
async def verify_model_health() -> str:
    """Runs a deep logic check with latency reporting."""
    try:
        client, model_id = await get_vllm_client()
        start_time = time.monotonic()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, is the model working?"}],
            model=model_id,
            max_tokens=10,
        )
        end_time = time.monotonic()
        latency = end_time - start_time
        response_content = chat_completion.choices[0].message.content

        if response_content:
            return (
                f"✅ Model health check PASSED.\n"
                f"Response: '{response_content[:50]}...'\n"
                f"Latency: {latency:.2f} seconds."
            )
        else:
            return "❌ Model health check FAILED: Empty response."
    except Exception as e:
        return f"❌ Model health check FAILED: {e}"


@mcp.tool(title="Save Hugging Face token", annotations=WRITE)
async def save_hf_token(token: str) -> str:
    """Saves a Hugging Face API token to GCP Secret Manager as secret 'hf-token'.
    Note: the token passes through the conversation; for maximum privacy the user
    can instead run `echo -n <token> | gcloud secrets versions add hf-token --data-file=-`
    themselves (after creating the secret once)."""
    client = secretmanager.SecretManagerServiceClient()
    secret_parent = f"projects/{PROJECT_ID}/secrets/{HF_SECRET_ID}"

    try:
        try:
            # Check if the secret already exists
            await asyncio.to_thread(client.get_secret, request={"name": secret_parent})
        except Exception:
            # If not, create it
            await asyncio.to_thread(
                client.create_secret,
                request={
                    "parent": f"projects/{PROJECT_ID}",
                    "secret_id": HF_SECRET_ID,
                    "secret": {"replication": {"automatic": {}}},
                },
            )

        # Add the new version
        response = await asyncio.to_thread(
            client.add_secret_version,
            request={"parent": secret_parent, "payload": {"data": token.encode("UTF-8")}},
        )
    except Exception as e:
        return f"❌ Failed to save token to Secret Manager: {e}"
    return f"✅ Token saved. Version: {response.name}"


# --- Compute Engine TPU VM instances (this rig's only provisioning path) ------------
# The Cloud TPU API is no longer under active development, and TPU7x and later are
# Compute Engine or GKE only. This rig provisions with `gcloud compute instances create`
# and holds no queued-resource path at all — every tool below talks to Compute Engine.
#
# accelerator -> (GCE machine type, chips). Only the generations with a real Compute
# Engine create path appear here.
_GCE_MACHINE_TYPES = {
    "v6e-1": ("ct6e-standard-1t", 1),
    "v6e-4": ("ct6e-standard-4t", 4),
    "v6e-8": ("ct6e-standard-8t", 8),
    "v5p-8": ("ct5p-hightpu-4t", 4),
}
# v5e is deliberately absent, and its absence is verified rather than assumed.
# `ct5lp-hightpu-*` machine types DO exist in the catalog, in 26 zones; the shared OS
# image family is even named for v5e and there is a Compute Engine v5-lite quota id.
# None of that is evidence — creating one is rejected at validation with
#   This user agent is not allowed to use the machine type [ct5lp-hightpu-1t].
# which is not a quota error and not a does-not-exist error. Those artefacts exist
# because the TPU API and GKE are implemented on Compute Engine underneath. Catalog
# presence is not creatability: do not re-derive a v5e entry from the machine-type list.
_GCE_NO_CE_PATH = {"v5e": "the Cloud TPU API (queued resources) — Compute Engine refuses ct5lp-* machine types"}
_GCE_IMAGE_FLAGS = [
    f"--image-project={IMAGE_PROJECT}",
    f"--image-family={IMAGE_FAMILY}",
]
_GCE_TPU_FILTER = "machineType~'ct6e|ct5p'"


def _gce_machine_type(accelerator: str) -> Optional[tuple[str, int]]:
    """(machine type, chips) for an accelerator, accepting either v5e spelling.
    None means this rig cannot create it — see `_unsupported_accelerator_message`."""
    key = accelerator.lower()
    if key.startswith("v5litepod-"):
        key = "v5e-" + key.split("-", 1)[1]
    return _GCE_MACHINE_TYPES.get(key)


def _unsupported_accelerator_message(accelerator: str) -> str:
    """Explains a refusal, distinguishing 'no Compute Engine path exists for this
    generation' from 'that is not a shape'. The first is not fixable by picking another
    zone or asking for quota, which is exactly what the error text otherwise invites."""
    elsewhere = _GCE_NO_CE_PATH.get(_family(accelerator))
    if elsewhere:
        return (
            f"❌ `{accelerator}` has no Compute Engine provisioning path — it is reachable only through "
            f"{elsewhere}. This is a property of the generation, not of this zone or your quota, so "
            "retrying elsewhere will not help. Supported here: " + ", ".join(sorted(_GCE_MACHINE_TYPES)) + "."
        )
    return f"❌ Unsupported accelerator '{accelerator}'. Supported: " + ", ".join(sorted(_GCE_MACHINE_TYPES)) + "."


# HBM per chip, by family. The reserve is what is gone before a single weight is
# loaded: libtpu/XLA runtime plus the activation working set, measured at
# 2.0 + 1.5 GB on a v6e-1 (see deploy.md). What is left is the weight budget.
_CHIP_HBM_GB = {"v6e": 32.0, "v5e": 16.0, "v5p": 96.0}
_CHIP_RESERVED_GB = 3.5
# bf16 weight footprint in GB. E2B is the measured 10.21 GB from deploy.md; the
# others are 2 bytes/param at nominal size.
_BF16_WEIGHTS_GB = {"E2B": 10.2, "E4B": 16.0, "12B": 24.0, "26B": 52.0, "31B": 62.0}
# Single-host shapes that actually exist, so the floor names a size you can create.
_HOST_SHAPES = (1, 4, 8)


def _min_chips_for_model(model: str, accelerator: str = "") -> int:
    """Rough single-host chip floor for a checkpoint on this accelerator's family.
    Prevents the expensive failure mode of a big model on a small accelerator: VM
    boot + image pull + ~10 min of loading before an inevitable OOM.

    Sized on the target chip's real HBM rather than a fixed 32 GB, because a v5e chip
    has half a v6e's: a bf16 12B fits one v6e chip and OOMs on a v5e-1. Quantized
    variants (QAT/int4/AWQ/GPTQ/FP8) are ~4x smaller but are not exempt — a 26B int4
    is still 13 GB, which does not fit a v5e-1 either. Unknown checkpoints are never
    blocked."""
    weights_gb = next((gb for marker, gb in _BF16_WEIGHTS_GB.items() if marker in model), 0.0)
    if any(q in model.lower() for q in ("qat", "int4", "q4", "awq", "gptq", "fp8")):
        weights_gb /= 4
    budget_gb = _CHIP_HBM_GB.get(_family(accelerator or ACCELERATOR_TYPE), 32.0) - _CHIP_RESERVED_GB
    if weights_gb <= 0 or budget_gb <= 0:
        return 1
    needed = weights_gb / budget_gb
    if needed > _HOST_SHAPES[-1]:
        return math.ceil(needed)  # larger than any single host: always blocks
    return next(n for n in _HOST_SHAPES if n >= needed)


@mcp.tool(title="Create TPU VM instance", annotations=WRITE)
async def create_tpu_vm_instance(
    instance_name: Optional[str] = None,
    zone: Optional[str] = None,
    accelerator: Annotated[
        str, Field(description="TPU type: v6e-1/4/8 or v5p-8 (v5e has no Compute Engine path)")
    ] = ACCELERATOR_TYPE,
    model_name: Optional[str] = None,
    boot_disk_size_gb: int = BOOT_DISK_SIZE_GB,
    max_run_duration: str = MAX_RUN_DURATION,
    request_valid_for: str = REQUEST_VALID_FOR,
    provisioning_model: Annotated[
        Literal["flex-start", "spot", "on-demand", "reservation-bound"],
        Field(
            description="How to ask for the chip. flex-start queues (cheapest on v6e); spot fails fast; "
            "on-demand is ~2x; reservation-bound needs RESERVATION_NAME"
        ),
    ] = PROVISIONING_MODEL,
    workload: Annotated[
        Literal["jaxrust", "jax", "vllm"],
        Field(
            description="'jaxrust' installs a Rust toolchain + libtpu for the Rust/XLA engine "
            "(the default on this rig); 'jax' installs a bare jax[tpu] dev environment; "
            "'vllm' serves model_name via docker"
        ),
    ] = "jaxrust",
) -> str:
    """Creates a TPU VM as a Compute Engine instance — this rig's only provisioning path.

    workload='jaxrust' (the default) installs a pinned Rust toolchain, the build
    dependencies pjrt-sys needs, and libtpu — no docker, no HF token, and no Python in
    the serving path. It prepares an environment and stops; follow with
    `deploy_jaxrust_engine` to build and start the engine. workload='jax' installs the
    Python jax[tpu] stack instead, which is what the parity oracle runs on.
    workload='vllm' auto-starts Gemma 4 serving via the vLLM startup script.

    Three flags below are load-bearing and every one of them fails LATE, long after the
    flag that caused it: --scopes=cloud-platform (without it the VM boots fine and then
    spins for 30 minutes failing to read Secret Manager, which reads as a token problem),
    --boot-disk-size (the image default is 10GB and cannot hold the vLLM image, so it
    dies mid-pull after a clean boot), and --maintenance-policy=TERMINATE (required — a
    TPU instance cannot live-migrate).

    Note what the return value does NOT mean: an instance is RUNNING the moment the VM
    boots, before the startup script has installed anything. Follow with
    `wait_for_jax_ready` or `wait_for_vllm_ready`, which read the serial log."""
    zone = _zone(zone)
    instance_name = instance_name or INSTANCE_NAME
    shape = _gce_machine_type(accelerator)
    if shape is None:
        return _unsupported_accelerator_message(accelerator)
    machine_type, chips = shape
    try:
        provisioning_flags = _provisioning_flags(provisioning_model, request_valid_for, max_run_duration)
    except ValueError as e:
        return f"❌ Aborted: {e}"

    selected_model = model_name or MODEL_NAME
    if workload in ("jax", "jaxrust"):
        # No model to size and no gated weights to fetch: skip the chip-count
        # check and the HF token requirement, neither of which applies.
        try:
            startup_script_content = (
                _get_jaxrust_startup_script(zone) if workload == "jaxrust" else _get_jax_startup_script(zone)
            )
        except RuntimeError as e:
            return f"❌ Aborted: {e}"
    else:
        min_chips = _min_chips_for_model(selected_model, accelerator)
        if chips < min_chips:
            return (
                f"❌ `{selected_model}` needs ~{min_chips} chips but `{accelerator}` has {chips} — "
                "it would OOM after a long load. Pick a larger accelerator or a smaller model."
            )

        token = await get_secret()
        if not token:
            return "❌ Aborted: 'hf-token' secret missing. Save one with `save_hf_token` first."

        try:
            startup_script_content = _get_formatted_startup_script(selected_model, zone, tp_size=chips)
        except RuntimeError as e:
            return f"❌ Aborted: {e}"
    script_file = _write_startup_script(startup_script_content)

    create_cmd = [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance_name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        *provisioning_flags,
        *_GCE_IMAGE_FLAGS,
        "--maintenance-policy=TERMINATE",
        f"--boot-disk-size={boot_disk_size_gb}GB",
        "--scopes=cloud-platform",
        f"--metadata-from-file=startup-script={script_file}",
    ]
    logger.info(f"Executing gcloud command: {' '.join(shlex.quote(c) for c in create_cmd)}")
    # Flex-start creation blocks until capacity is granted or the request expires.
    try:
        rc, stdout, stderr = await run_command(create_cmd, timeout=590)
    finally:
        try:
            os.unlink(script_file)
        except OSError:
            pass
    if rc != 0:
        if stderr.startswith("Timeout after"):
            return (
                f"⏳ gcloud gave up after ~10 min, but the {provisioning_model} request for `{instance_name}` "
                f"(valid for {request_valid_for}) may still be PENDING server-side — the VM can still "
                f"appear and bill later. Check with `list_tpu_vm_instances`; if you no longer want it, "
                f"delete it with `destroy_tpu_vm_instance` once it appears.\n\n"
                f"PENDING means **either** no quota **or** no capacity, and the two are identical from "
                f"outside — a create does not report which. Run `probe_zone_capacity('{zone}')` to tell "
                f"them apart: it is free and takes seconds."
            )
        hint = ""
        if "not allowed to use the machine type" in stderr:
            hint = (
                f"\n\nThis is the 'no Compute Engine path' refusal, not a quota or capacity problem — "
                f"`{machine_type}` exists in the catalog but cannot be created. Retrying elsewhere will not help."
            )
        elif "TPUS_PER_TPU_FAMILY" in stderr or "quota" in stderr.lower():
            hint = (
                f"\n\nCheck BOTH metrics before believing this — {provisioning_model} spends "
                f"`{GCE_SPOT_QUOTA_ID}` first and falls back to `{GCE_QUOTA_ID}`, and their defaults are "
                f"opposite (absent from the family listing = 0, absent from the preemptible listing = 1536). "
                f"`get_zones_with_available_quota` reads both; `gcloud compute regions describe` reads "
                f"neither — it only carries the older v5-era metrics."
            )
        elif "stockout" in stderr.lower() or "does not have enough resources" in stderr.lower():
            hint = (
                f"\n\nCapacity, not quota. Capacity is ZONAL while quota is regional, so a sibling zone in "
                f"{REGION} costs nothing to try — `find_tpu_vm` sweeps them."
            )
        return f"❌ Creation failed: {stderr}{hint}"
    lifecycle = (
        f"the VM self-deletes at max-run-duration ({max_run_duration})"
        if max_run_duration
        else "the VM runs until you delete it"
    )
    if workload == "jaxrust":
        return (
            f"🚀 TPU VM `{instance_name}` ({machine_type}, {chips} chip(s), {provisioning_model}) created in "
            f"{zone}; installing Rust {RUST_TOOLCHAIN} + `{LIBTPU_SPEC}`. **It is RUNNING but not ready** — "
            f"RUNNING means the VM booted, nothing more. Follow with `wait_for_jaxrust_ready`, then "
            f"`deploy_jaxrust_engine` (which builds the engine and runs the probe); {lifecycle}.\n{stdout}"
        )
    if workload == "jax":
        return (
            f"🚀 TPU VM `{instance_name}` ({machine_type}, {chips} chip(s), {provisioning_model}) created in "
            f"{zone}; installing Python {JAX_PYTHON_VERSION} + `{JAX_PIP_SPEC}`. **It is RUNNING but not "
            f"ready** — RUNNING means the VM booted, nothing more. Follow with `wait_for_jax_ready` "
            f"(or `get_tpu_vm_serial_log`), then `verify_jax_tpu`; {lifecycle}.\n{stdout}"
        )
    return (
        f"🚀 TPU VM `{instance_name}` ({machine_type}, {chips} chip(s), {provisioning_model}) created in "
        f"{zone}; vLLM is starting `{selected_model}` (tp={chips}). **RUNNING is not ready** — model load "
        f"takes ~10 min. Follow with `wait_for_vllm_ready` or `get_tpu_vm_serial_log`; {lifecycle}.\n{stdout}"
    )


@mcp.tool(title="List TPU VM instances", annotations=READ_ONLY)
async def list_tpu_vm_instances(zone: Optional[str] = None) -> str:
    """Lists GCE TPU VM instances (ct5lp/ct6e/ct5p machine types) across all zones, or one zone."""
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={PROJECT_ID}",
        f"--filter={_GCE_TPU_FILTER}" + (f" AND zone:{zone}" if zone else ""),
        "--format=table(name,zone,machineType.basename(),status,networkInterfaces[0].networkIP,networkInterfaces[0].accessConfigs[0].natIP)",
    ]
    rc, stdout, stderr = await run_command(cmd)
    if rc != 0:
        return f"❌ Failed to list TPU VM instances: {stderr}"
    return stdout if stdout else "No GCE TPU VM instances found."


@mcp.tool(title="Destroy TPU VM instance", annotations=DESTRUCTIVE)
async def destroy_tpu_vm_instance(instance_name: str, zone: Optional[str] = None) -> str:
    """Deletes a GCE TPU VM instance. Flex-start bills until deletion — confirm with the
    user before destroying anything they may still need."""
    zone = _zone(zone)
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "delete",
        instance_name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        "--quiet",
    ]
    rc, _, stderr = await run_command(cmd, timeout=300)
    if rc != 0:
        return f"❌ Deletion failed: {stderr}"
    return f"🗑️ TPU VM `{instance_name}` deleted from {zone}. Billing for it has stopped."


@mcp.tool(title="Get TPU VM serial log", annotations=READ_ONLY)
async def get_tpu_vm_serial_log(
    instance_name: str, zone: Optional[str] = None, tail: Annotated[int, Field(ge=1, le=1000)] = 40
) -> str:
    """Tails the serial-console output of a GCE TPU VM. SSH to TPU VMs is often blocked by
    firewall policy, so this is the primary way to watch startup-script/vLLM boot progress.
    Success markers: 'JAXRUST-BOOTLOADER: TPU environment ready.' (workload='jaxrust'),
    'JAX-BOOTLOADER: TPU environment ready.' (workload='jax') or
    'vLLM application startup complete.' (workload='vllm')."""
    zone = _zone(zone)
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "get-serial-port-output",
        instance_name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
    ]
    rc, stdout, stderr = await run_command(cmd, timeout=120)
    if rc != 0:
        return f"❌ Failed to read serial console: {stderr}"
    lines = stdout.splitlines()
    # Show output from the most recent startup-script run when a marker is present.
    # Every workload announces itself, so scan for any of the three banners.
    for i in range(len(lines) - 1, -1, -1):
        if (
            "Starting Rust/XLA TPU Bootloader" in lines[i]
            or "Starting JAX TPU Bootloader" in lines[i]
            or "Starting vLLM Bootloader" in lines[i]
        ):
            lines = lines[i:]
            break
    return "\n".join(lines[-tail:]) if lines else "No serial output available yet."


async def _get_instance_ips(instance_name: str, zone: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (external_ip, internal_ip) of a GCE instance."""
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        instance_name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        "--format=value(networkInterfaces[0].accessConfigs[0].natIP,networkInterfaces[0].networkIP)",
    ]
    rc, stdout, _ = await run_command(cmd)
    if rc != 0 or not stdout:
        return None, None
    parts = stdout.split()
    external = parts[0] if len(parts) > 1 else None
    internal = parts[-1] if parts else None
    return external, internal


@mcp.tool(title="Get TPU VM endpoint", annotations=READ_ONLY)
async def get_tpu_vm_endpoint(instance_name: str, zone: Optional[str] = None) -> str:
    """Returns the vLLM endpoint URLs of a GCE TPU VM and probes their health. Port 8000
    is frequently unreachable from outside the VPC (firewall) even when serving is healthy —
    if both probes fail, check `get_tpu_vm_serial_log` for the startup-complete marker."""
    zone = _zone(zone)
    external, internal = await _get_instance_ips(instance_name, zone)
    if not external and not internal:
        return f"❌ Could not resolve IPs for `{instance_name}` in {zone}."

    report = []
    async with httpx.AsyncClient(timeout=5) as client:
        for label, ip in (("external", external), ("internal", internal)):
            if not ip:
                continue
            url = f"http://{ip}:8000"
            try:
                res = await client.get(f"{url}/health")
                status = "🟢 healthy" if res.status_code == 200 else f"⚠️ HTTP {res.status_code}"
            except Exception:
                status = "🔴 unreachable (may be firewall, not the service)"
            report.append(f"- {label}: `{url}` — {status}")
    return f"### Endpoints for `{instance_name}`\n" + "\n".join(report)


@mcp.tool(title="Wait for vLLM ready", annotations=READ_ONLY)
async def wait_for_vllm_ready(
    instance_name: str = INSTANCE_NAME,
    zone: Optional[str] = None,
    timeout_minutes: Annotated[int, Field(ge=1, le=30)] = 15,
) -> str:
    """Polls a GCE flex-start TPU VM every 30s until vLLM serving is ready, checking
    the health endpoint and the serial-console startup marker. One call replaces
    manual serial-log polling; model load typically takes ~10 min. Also fails fast
    if the startup script logs an ERROR (e.g. Secret Manager access denied)."""
    zone = _zone(zone)
    deadline = time.monotonic() + timeout_minutes * 60
    last_signal = "no serial output yet"
    while True:
        external, internal = await _get_instance_ips(instance_name, zone)
        for ip in (external, internal):
            if not ip:
                continue
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    res = await client.get(f"http://{ip}:8000/health")
                if res.status_code == 200:
                    return f"🟢 vLLM is serving on `{instance_name}` at http://{ip}:8000 (health check passed)."
            except Exception:
                pass

        serial = await get_tpu_vm_serial_log(instance_name, zone=zone, tail=40)
        if "vLLM application startup complete." in serial:
            return (
                f"🟢 vLLM startup complete on `{instance_name}` (serial-console marker). "
                "Port 8000 isn't reachable from here — likely firewall; serving is up inside the VPC."
            )
        error_lines = [line for line in serial.splitlines() if line.startswith("ERROR:")]
        if error_lines:
            return f"❌ Startup failed on `{instance_name}`:\n" + "\n".join(error_lines)
        if not serial.startswith("❌") and serial.strip():
            last_signal = serial.splitlines()[-1]

        if time.monotonic() >= deadline:
            return (
                f"⏳ Not ready after {timeout_minutes} min. Last serial output: {last_signal}\n"
                "Keep watching with `get_tpu_vm_serial_log`."
            )
        await asyncio.sleep(30)


@mcp.tool(title="Create CPU debug VM", annotations=WRITE)
async def create_cpu_debug_vm(
    instance_name: str = "gemma-debug",
    zone: Optional[str] = None,
    machine_type: Annotated[
        str,
        Field(
            description="GCE machine type. Memory gates what loads, vCPUs gate how long you wait. "
            "e2-highmem-16 = 16 vCPU/128 GiB runs a 31B; 64 GiB only loads one; 32 GiB is E2B only"
        ),
    ] = CPU_DEBUG_MACHINE_TYPE,
    boot_disk_size_gb: Annotated[
        int, Field(ge=50, description="Checkpoints are large: E2B 8.3 GB, 31B W4A16 23.3 GB, MoE ~14 GB")
    ] = 100,
    spot: Annotated[bool, Field(description="Spot is ~70% cheaper and preemption is harmless for a debug box")] = True,
    max_run_duration: Annotated[
        Optional[str],
        Field(description="Auto-delete after this long, e.g. '8h'. Omit to leave it running."),
    ] = None,
) -> str:
    """Creates a plain CPU VM for correctness work — no TPU, no accelerator cost.

    The JAX engine runs unchanged on the CPU backend (`w4a16_impl` defaults to the
    reference path, Pallas auto-switches to interpret mode), so architecture and
    loader bugs reproduce here at a fraction of TPU rates. Use the TPU only for
    questions whose answer is a time or a memory ceiling.

    Memory is the spec that decides what you can debug:
      * ~8 GiB  — E2B/E4B dense
      * ~32 GiB — E2B only (26 GiB measured peak for load + short generation).
      * ~64 GiB — enough to LOAD the 31B and inspect its parameter tree
                  (48 GiB measured peak), but a forward pass OOMs.
      * ~128 GiB — enough to RUN the 31B on CPU.
    These are measured peak RSS. XLA:CPU allocates roughly 2x what the weights
    occupy, so sizing from checkpoint bytes will leave you short.
    Host RAM does NOT predict HBM: XLA:CPU allocates differently and can use
    virtual memory, so a model that loads here may still OOM on a v6e-1.

    Spot instances use termination-action=STOP so the boot disk (and its cached
    checkpoints) survives preemption — restart the instance to carry on.
    """
    zone = _zone(zone)
    try:
        startup_script_content = _get_cpu_debug_startup_script(zone)
    except RuntimeError as e:
        return f"❌ Aborted: {e}"
    script_file = _write_startup_script(startup_script_content)

    create_cmd = [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance_name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        "--image-project=ubuntu-os-cloud",
        "--image-family=ubuntu-2204-lts",
        f"--boot-disk-size={boot_disk_size_gb}GB",
        "--boot-disk-type=pd-balanced",
        # cloud-platform so the box can read the hf-token secret the same way the
        # TPU VMs do; the default compute SA already holds secretAccessor.
        "--scopes=cloud-platform",
        f"--metadata-from-file=startup-script={script_file}",
    ]
    if spot:
        create_cmd += ["--provisioning-model=SPOT", "--instance-termination-action=STOP"]
    if max_run_duration:
        create_cmd += [f"--max-run-duration={max_run_duration}", "--instance-termination-action=DELETE"]

    logger.info(f"Executing gcloud command: {' '.join(shlex.quote(c) for c in create_cmd)}")
    rc, stdout, stderr = await run_command(create_cmd, timeout=600)
    if rc != 0:
        return f"❌ Creation failed: {stderr}"

    kind = "Spot" if spot else "on-demand"
    life = f"; self-deletes at {max_run_duration}" if max_run_duration else "; runs until you delete it"
    return (
        f"🖥️ CPU debug VM `{instance_name}` ({machine_type}, {kind}) created in {zone}"
        f"{life}. Installing Python {JAX_PYTHON_VERSION} + `{CPU_DEBUG_PIP_SPEC}`.\n"
        f"Follow with `wait_for_jax_ready(instance_name='{instance_name}')`, then "
        f"`destroy_tpu_vm_instance('{instance_name}')` when done "
        f"(it deletes any GCE instance, TPU or not).\n{stdout}"
    )


@mcp.tool(title="Wait for Rust/XLA TPU ready", annotations=READ_ONLY)
async def wait_for_jaxrust_ready(
    instance_name: str = INSTANCE_NAME,
    zone: Optional[str] = None,
    timeout_minutes: Annotated[int, Field(ge=1, le=30)] = 15,
) -> str:
    """Polls a workload='jaxrust' VM every 20s until its startup script reports the
    Rust + libtpu environment ready (serial-console marker), failing fast on the
    script's FAILED marker rather than waiting out the timeout.

    Ready here means: a pinned Rust toolchain is installed, protoc and clang are
    present, and a libtpu.so exporting GetPjrtApi is on disk with LIBTPU_PATH
    pointing at it. It does NOT mean anything has been compiled or that the chip
    has executed a single instruction — `deploy_jaxrust_engine` settles that by
    building and running the probe."""
    zone = _zone(zone)
    deadline = time.monotonic() + timeout_minutes * 60
    last_signal = "no serial output yet"

    def emitted(serial: str, marker: str) -> bool:
        """True only if the script *printed* the marker — see `wait_for_jax_ready`
        for why trace lines have to be skipped."""
        for line in serial.splitlines():
            body = line.split("startup-script:", 1)[-1].strip()
            if body.startswith("+"):
                continue
            if marker in body:
                return True
        return False

    while True:
        serial = await get_tpu_vm_serial_log(instance_name, zone=zone, tail=60)
        if emitted(serial, "JAXRUST-BOOTLOADER: TPU environment ready."):
            return (
                f"🟢 Rust + libtpu ready on `{instance_name}`. Nothing is built yet — "
                "follow with `deploy_jaxrust_engine`."
            )
        if emitted(serial, "JAXRUST-BOOTLOADER: FAILED"):
            detail = [
                ln for ln in serial.splitlines() if "JAXRUST-BOOTLOADER: ERROR" in ln or ln.startswith("ERROR:")
            ]
            return f"❌ Rust/XLA startup failed on `{instance_name}`:\n" + "\n".join(
                detail or ["see `get_tpu_vm_serial_log`"]
            )
        if not serial.startswith("❌") and serial.strip():
            last_signal = serial.splitlines()[-1]
        if time.monotonic() >= deadline:
            return (
                f"⏳ Not ready after {timeout_minutes} min. Last serial output: {last_signal}\n"
                "Keep watching with `get_tpu_vm_serial_log`."
            )
        await asyncio.sleep(20)


@mcp.tool(title="Build the Rust engine on the VM", annotations=WRITE)
async def deploy_jaxrust_engine(
    instance_name: str = INSTANCE_NAME,
    zone: Optional[str] = None,
    features: str = JAXRUST_CARGO_FEATURES,
    build_timeout_minutes: Annotated[int, Field(ge=5, le=90)] = 45,
) -> str:
    """Uploads the `rust/` workspace, builds it on the VM in release mode, and runs
    `xla-probe` — which compiles a StableHLO matmul on the chip and checks the
    numbers that come back.

    The probe is the point. A build that succeeds proves the toolchain works; it
    says nothing about whether libtpu can talk to a chip, and this rig's standing
    rule is that a thing being accepted is not evidence it did anything. The probe
    is the cheapest artefact that fails when the accelerator does not work.

    `target/` is excluded from the upload — it is the largest directory in the
    workspace and the VM is the only host whose build output matters."""
    zone = _zone(zone)
    workspace = _find_rust_workspace()
    if not workspace:
        return (
            "❌ Cannot find the `rust/` workspace next to this server. Run this tool from a "
            "checkout of the rig, not from a bare skill install."
        )

    fd, tarball = tempfile.mkstemp(prefix="jaxrust-", suffix=".tar.gz")
    os.close(fd)
    try:
        rc, _, stderr = await run_command(
            [
                "tar",
                "czf",
                tarball,
                "-C",
                os.path.dirname(workspace),
                "--exclude=target",
                "--exclude=.git",
                os.path.basename(workspace),
            ],
            timeout=300,
        )
        if rc != 0:
            return f"❌ Could not package the workspace: {stderr}"

        argv, target = _build_scp_cmd(tarball, "/tmp/jaxrust.tar.gz", instance_name, zone)
        rc, _, stderr = await run_command(argv, timeout=600)
        if rc != 0:
            return f"❌ Upload to `{target}` failed: {stderr}"
    finally:
        try:
            os.unlink(tarball)
        except OSError:
            pass

    remote = (
        ". /etc/profile.d/jaxrust.sh && set -e && "
        f"rm -rf {JAXRUST_REMOTE_DIR} && mkdir -p {JAXRUST_REMOTE_DIR} && "
        f"tar xzf /tmp/jaxrust.tar.gz -C {JAXRUST_REMOTE_DIR} --strip-components=1 && "
        f"cd {JAXRUST_REMOTE_DIR} && "
        "cargo build --release -p xla-probe && "
        f"cargo build --release -p gemma4-engine --no-default-features --features {shlex.quote(features)} && "
        "./target/release/xla-probe"
    )
    argv, target = _build_ssh_cmd(remote, instance_name, zone)
    rc, stdout, stderr = await run_command(argv, timeout=build_timeout_minutes * 60)
    output = "\n".join(part for part in (stdout, stderr) if part).strip()
    if rc != 0:
        hint = ""
        if "Could not find `protoc`" in output:
            hint = "\n\n`protoc` is missing — the startup script installs it, so this VM was not booted with the jaxrust workload."
        elif "GetPjrtApi" in output or "no TPU PJRT plugin found" in output:
            hint = "\n\nlibtpu did not resolve. Check LIBTPU_PATH in /etc/profile.d/jaxrust.sh."
        return f"❌ Build or probe failed on `{target}`:\n```\n{output}\n```{hint}"
    return (
        f"✅ Engine built and the probe passed on `{target}` (features `{features}`):\n"
        f"```\n{output}\n```\n"
        "Start serving with `manage_jaxrust_server` action='start'."
    )


@mcp.tool(title="Verify Rust sees the TPU", annotations=READ_ONLY)
async def verify_rust_tpu(
    instance_name: str = INSTANCE_NAME,
    zone: Optional[str] = None,
) -> str:
    """Re-runs `xla-probe` over SSH: loads the PJRT plugin, lists the devices with
    their HBM stats, compiles a StableHLO matmul and checks the result.

    The Rust analogue of `verify_jax_tpu`, and deliberately stricter than it.
    `dlopen` on libtpu.so succeeds on a host with no chip, exactly as `import jax`
    succeeds with no TPU backend — so this asserts on a computed value rather than
    on a device list."""
    zone = _zone(zone)
    remote = f". /etc/profile.d/jaxrust.sh && {JAXRUST_REMOTE_DIR}/target/release/xla-probe"
    argv, target = _build_ssh_cmd(remote, instance_name, zone)
    rc, stdout, stderr = await run_command(argv, timeout=300)
    output = "\n".join(part for part in (stdout, stderr) if part).strip()
    if rc != 0:
        if "No such file" in output:
            return f"❌ No probe binary on `{target}` — run `deploy_jaxrust_engine` first.\n{output}"
        return f"❌ Rust cannot execute on the TPU on `{target}`:\n{output}"
    return f"🟢 Rust compiled and ran StableHLO on the TPU on `{target}`:\n{output}"


@mcp.tool(title="Manage the Rust engine process", annotations=DESTRUCTIVE)
async def manage_jaxrust_server(
    action: Literal["start", "stop", "status", "logs"],
    instance_name: str = INSTANCE_NAME,
    zone: Optional[str] = None,
    model_path: str = "",
    max_model_len: int = 8192,
) -> str:
    """Starts, stops, inspects or tails the `gemma4-engine` process on the VM.

    'start' needs `model_path` — a directory or file holding the checkpoint on the
    VM. The engine does not bind its port until the weights are loaded and the
    graph is compiled, so a port that is not answering yet is a load in progress,
    not a crash; `logs` tells the two apart."""
    zone = _zone(zone)
    binary = f"{JAXRUST_REMOTE_DIR}/target/release/gemma4-engine"
    log = "/tmp/gemma4-engine.log"
    if action == "start":
        if not model_path:
            return "❌ `model_path` is required to start the engine — it is the checkpoint on the VM."
        cmd = (
            f". /etc/profile.d/jaxrust.sh && nohup {binary} "
            f"--weights {shlex.quote(model_path)} --max-model-len {max_model_len} "
            f"--device tpu --model-name {shlex.quote(MODEL_NAME)} "
            f"> {log} 2>&1 & echo started $!"
        )
    elif action == "stop":
        cmd = "pkill -f gemma4-engine && echo stopped || echo 'no engine process running'"
    elif action == "status":
        cmd = "pgrep -af gemma4-engine || echo 'not running'; curl -s -m 3 localhost:8000/health || echo 'port 8000 not answering'"
    else:
        cmd = f"tail -n 80 {log} 2>/dev/null || echo 'no log at {log}'"

    argv, target = _build_ssh_cmd(cmd, instance_name, zone)
    rc, stdout, stderr = await run_command(argv, timeout=180)
    output = "\n".join(part for part in (stdout, stderr) if part).strip()
    if rc != 0:
        return f"❌ `{action}` failed on `{target}`:\n{output}"
    suffix = ""
    if action == "start":
        suffix = (
            "\n\n**Started is not ready.** The engine compiles the graph before it binds "
            "port 8000; watch `manage_jaxrust_server` action='logs' for `JAXRUST-SERVER: listening`."
        )
    return f"✅ `{action}` on `{target}`:\n```\n{output}\n```{suffix}"


@mcp.tool(title="Wait for JAX TPU ready", annotations=READ_ONLY)
async def wait_for_jax_ready(
    instance_name: str = INSTANCE_NAME,
    zone: Optional[str] = None,
    timeout_minutes: Annotated[int, Field(ge=1, le=30)] = 10,
) -> str:
    """Polls a workload='jax' flex-start TPU VM every 20s until its startup script
    reports the JAX environment ready (serial-console marker). Fails fast on the
    script's FAILED marker instead of waiting out the timeout — the script only
    emits the ready marker after asserting a TPU device is actually visible."""
    zone = _zone(zone)
    deadline = time.monotonic() + timeout_minutes * 60
    last_signal = "no serial output yet"

    def emitted(serial: str, marker: str) -> bool:
        """True only if the script *printed* the marker.

        The startup script runs under `set -x`, so the shell echoes each command
        before running it. A traced line can contain a marker string verbatim
        (the ERR trap's own definition contains the FAILED marker), which would
        otherwise read as a failure on a healthy boot. Trace lines are prefixed
        with '+', so skip them.
        """
        for line in serial.splitlines():
            body = line.split("startup-script:", 1)[-1].strip()
            if body.startswith("+"):
                continue
            if marker in body:
                return True
        return False

    while True:
        serial = await get_tpu_vm_serial_log(instance_name, zone=zone, tail=60)
        if emitted(serial, "environment ready."):
            return (
                f"🟢 JAX environment ready on `{instance_name}`. "
                "Verify with `verify_jax_tpu`; run workloads over SSH with "
                f"`python{JAX_PYTHON_VERSION}`."
            )
        if emitted(serial, "JAX-BOOTLOADER: FAILED"):
            detail = [ln for ln in serial.splitlines() if "JAX-BOOTLOADER: ERROR" in ln or ln.startswith("ERROR:")]
            return f"❌ JAX startup failed on `{instance_name}`:\n" + "\n".join(
                detail or ["see `get_tpu_vm_serial_log`"]
            )
        if not serial.startswith("❌") and serial.strip():
            last_signal = serial.splitlines()[-1]
        if time.monotonic() >= deadline:
            return (
                f"⏳ Not ready after {timeout_minutes} min. Last serial output: {last_signal}\n"
                "Keep watching with `get_tpu_vm_serial_log`."
            )
        await asyncio.sleep(20)


@mcp.tool(title="Verify JAX sees the TPU", annotations=READ_ONLY)
async def verify_jax_tpu(
    instance_name: str = INSTANCE_NAME,
    zone: Optional[str] = None,
) -> str:
    """Re-runs the JAX TPU check on the VM over SSH: reports jax/jaxlib/libtpu
    versions and the device list, and fails if no TPU device is visible.

    Importing jax succeeds even with no TPU backend, so this asserts on
    jax.devices() rather than on the import."""
    zone = _zone(zone)
    probe = (
        "import jax, sys;"
        "import importlib.metadata as md;"
        "devs = jax.devices();"
        "print('jax', jax.__version__, '| libtpu', md.version('libtpu') if any(d.platform=='tpu' for d in devs) else 'n/a');"
        "print('devices:', devs);"
        "sys.exit(0 if any(d.platform=='tpu' for d in devs) else 1)"
    )
    remote = f"python{JAX_PYTHON_VERSION} -c {shlex.quote(probe)}"
    argv, target = _build_ssh_cmd(remote, instance_name, zone)
    rc, stdout, stderr = await run_command(argv, timeout=180)
    output = "\n".join(part for part in (stdout, stderr) if part).strip()
    if rc != 0:
        return f"❌ JAX cannot see the TPU on `{target}`:\n{output}"
    return f"🟢 JAX sees the TPU on `{target}`:\n{output}"


async def _quota_regions(quota_id: str) -> dict[str, float]:
    """Region -> limit for one Compute Engine quota metric, dropping zero entries.

    `gcloud compute regions describe` cannot answer this: its quota list only carries
    the older v5-era metrics (TPU_LITE_PODSLICE_V5 and friends), so it reports
    confidently and wrongly about v6e. The newer metrics live in the Cloud Quotas API
    and have to be asked for by name, one call per metric.
    """
    cmd = [
        "gcloud",
        "beta",
        "quotas",
        "info",
        "list",
        "--service=compute.googleapis.com",
        f"--project={PROJECT_ID}",
        f"--filter=quotaId:{quota_id}",
        "--format=json",
    ]
    rc, stdout, stderr = await run_command(cmd)
    if rc != 0:
        logger.error(f"Failed to retrieve quota info for {quota_id}: {stderr}")
        return {}
    try:
        quota_data = json.loads(stdout)
    except Exception:
        return {}

    regions: dict[str, float] = {}
    for info in quota_data:
        for dim_info in info.get("dimensionsInfos", []):
            raw = dim_info.get("details", {}).get("value")
            if raw in (None, "", "0"):
                continue
            try:
                limit = float(raw)
            except (TypeError, ValueError):
                continue
            if limit <= 0:
                continue
            dim_map = dim_info.get("dimensions", {})
            # The family metric is dimensioned by region AND tpu_family; a listing for
            # another family says nothing about ours.
            family = dim_map.get("tpu_family")
            if family and family.upper() != GCE_TPU_FAMILY.upper():
                continue
            locations = [dim_map.get("region") or dim_map.get("zone")]
            if not locations[0]:
                locations = dim_info.get("applicableLocations", [])
            for loc in locations:
                if loc:
                    regions[loc] = max(regions.get(loc, 0.0), limit)
    return regions


async def _zones_publishing_machine_type(machine_type: str) -> list[str]:
    """Zones whose catalog carries this machine type.

    Quota is a ceiling, not an allocation, and the catalog is not creatability — but the
    intersection of the two is still the only sane place to start a sweep, and an unset
    quota reads identically to a zero one so a blank tells you nothing about the hardware.
    """
    cmd = [
        "gcloud",
        "compute",
        "machine-types",
        "list",
        f"--project={PROJECT_ID}",
        f"--filter=name={machine_type}",
        "--format=value(zone)",
    ]
    rc, stdout, _ = await run_command(cmd, timeout=120)
    if rc != 0 or not stdout:
        return []
    return sorted({line.strip() for line in stdout.splitlines() if line.strip()})


async def _candidate_zones(accelerator: str, provisioning_model: str) -> list[str]:
    """Zones worth attempting: the machine type is published there AND its region holds
    quota on whichever pool this provisioning model actually spends.

    Flex-start reads as usable if EITHER pool has room, because it spends the preemptible
    quota first and falls back to the family quota. Reading one metric alone is how live
    regions get written off.
    """
    shape = _gce_machine_type(accelerator)
    if shape is None:
        return []
    machine_type, _ = shape
    if provisioning_model == "on-demand":
        quota_ids = [GCE_QUOTA_ID]
    elif provisioning_model == "spot":
        quota_ids = [GCE_SPOT_QUOTA_ID]
    else:  # flex-start, reservation-bound
        quota_ids = [GCE_SPOT_QUOTA_ID, GCE_QUOTA_ID]

    regions: set[str] = set()
    for quota_id in quota_ids:
        regions |= set(await _quota_regions(quota_id))
    zones = await _zones_publishing_machine_type(machine_type)
    return [z for z in zones if z.rsplit("-", 1)[0] in regions]


@mcp.tool(title="List regions with TPU quota", annotations=READ_ONLY)
async def get_zones_with_available_quota(
    quota_id: Annotated[
        Optional[str],
        Field(description="One Compute Engine quota id; omit to read both metrics that matter"),
    ] = None,
) -> str:
    """Reports Compute Engine TPU quota by region, for BOTH metrics unless one is named.

    Reading only one is the most expensive mistake on this path, because their defaults
    are opposite: a region absent from the family listing inherits 0, a region absent
    from the preemptible listing inherits 1536. A region that looks dead in one listing
    can hold full headroom in the other, and flex-start — which spends the preemptible
    pool first and falls back to the family pool — only needs one of them.

    Quota is permission to ask, not reserved hardware: use `probe_zone_capacity` to find
    out whether the chips are actually there."""
    ids = [quota_id] if quota_id else [GCE_SPOT_QUOTA_ID, GCE_QUOTA_ID]
    spends = {
        GCE_SPOT_QUOTA_ID: "spent by flex-start (first) and spot",
        GCE_QUOTA_ID: f"spent by on-demand, and by flex-start as fallback; tpu_family={GCE_TPU_FAMILY}",
    }
    output = ["### 📊 Compute Engine TPU quota by region\n"]
    for qid in ids:
        regions = await _quota_regions(qid)
        note = spends.get(qid, "")
        output.append(f"**`{qid}`**" + (f" — {note}" if note else ""))
        if not regions:
            output.append("- No region holds a non-zero limit on this metric.\n")
            continue
        for region, limit in sorted(regions.items()):
            output.append(f"- `{region}`: {limit:g}")
        output.append("")
    output.append(
        "A region missing from a listing is not necessarily zero — the two metrics inherit "
        "opposite defaults (family 0, preemptible 1536), and an unset value reads exactly "
        "like a zero one."
    )
    return "\n".join(output)


@mcp.tool(title="Probe a zone for real capacity", annotations=WRITE)
async def probe_zone_capacity(
    zone: Optional[str] = None,
    accelerator: Annotated[str, Field(description="TPU type to probe for")] = ACCELERATOR_TYPE,
    instance_name: Annotated[str, Field(description="Throwaway name for the probe instance")] = "capacity-probe",
) -> str:
    """Tells a stockout apart from a quota wall, in seconds and for free.

    A create that sits in PENDING means EITHER no quota OR no capacity, and from the
    outside the two are identical — the create never reports which. Spot is the
    discriminator because it does not queue: it fails fast and names the reason. Spot and
    flex-start draw on the same preemptible pool, so this probes the ZONE rather than
    your entitlement.

    Reading the result:
      * `stockout` — the hardware is not there. No amount of quota will help; try a
        sibling zone (capacity is zonal, quota is regional, and they diverge sharply).
      * provisioned — capacity exists, so a stuck flex-start request is a quota problem;
        the probe instance is deleted immediately.
      * a quota error — the answer is quota, and `get_zones_with_available_quota` says
        which pool.

    Availability moves faster than you can test against it: a zone has provisioned an
    instance and then refused the next request a minute later. Treat this as a reading,
    not a reservation."""
    zone = _zone(zone)
    shape = _gce_machine_type(accelerator)
    if shape is None:
        return _unsupported_accelerator_message(accelerator)
    machine_type, _ = shape

    create_cmd = [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance_name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        "--provisioning-model=SPOT",
        # A backstop, not the cleanup path: the probe is deleted explicitly below. This
        # only matters if that delete fails — without it a leaked probe instance bills
        # indefinitely. 10m is the floor Compute Engine accepts.
        "--max-run-duration=10m",
        "--instance-termination-action=DELETE",
        *_GCE_IMAGE_FLAGS,
        "--maintenance-policy=TERMINATE",
        "--boot-disk-size=50GB",
        "--no-service-account",
        "--no-scopes",
    ]
    async def _delete_probe() -> tuple[int, str]:
        """Best-effort teardown. Called on EVERY path that could have left an instance,
        not just the success path.

        This used to hang off `rc == 0` alone, and on 2026-08-28 that leaked a real
        instance: the create exceeded the 180s timeout below, `run_command` returned
        non-zero, and the function returned "neither stockout nor quota" while a
        ct6e-standard-1t sat in STAGING and billed. A create that times out has not
        necessarily failed — it has only stopped being watched.
        """
        return (
            await run_command(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "delete",
                    instance_name,
                    f"--project={PROJECT_ID}",
                    f"--zone={zone}",
                    "--quiet",
                ],
                timeout=300,
            )
        )[::2]

    logger.info(f"Probing {zone} for {machine_type} capacity via a SPOT create...")
    # 300s, not 180s: a ct6e spot create in a zone that HAS capacity was measured taking
    # longer than 180s to return, which turned a successful probe into an inconclusive
    # one and leaked the instance behind it.
    rc, stdout, stderr = await run_command(create_cmd, timeout=300)
    if rc == 0:
        # Capacity existed. Delete immediately — this is a probe, not a deployment.
        del_rc, del_err = await _delete_probe()
        cleanup = (
            "Probe instance deleted."
            if del_rc == 0
            else f"⚠️ **The probe instance is still running and billing** — delete `{instance_name}` "
            f"in {zone} by hand: {del_err}"
        )
        return (
            f"🟢 `{zone}` HAS {machine_type} capacity right now — a spot create succeeded. {cleanup}\n\n"
            f"So a flex-start request stuck in PENDING here is a **quota** problem, not scarcity: "
            f"read both metrics with `get_zones_with_available_quota`. Note capacity moves within "
            f"minutes, so this is a reading and not a reservation."
        )
    blob = f"{stdout}\n{stderr}".lower()
    if "stockout" in blob or "does not have enough resources" in blob:
        return (
            f"🔴 `{zone}` has NO {machine_type} capacity — reason: stockout. Quota cannot fix this.\n\n"
            f"Capacity is zonal while quota is regional, so a sibling zone in the same region costs "
            f"nothing to try — `find_tpu_vm` sweeps them. Nothing was created.\n\n{stderr}"
        )
    if "not allowed to use the machine type" in blob:
        return _unsupported_accelerator_message(accelerator) + f"\n\n{stderr}"
    if "quota" in blob:
        return (
            f"🟡 `{zone}` refused the probe on QUOTA, so capacity is untested here. This probe spends "
            f"`{GCE_SPOT_QUOTA_ID}` — the same pool flex-start draws on first — so flex-start will hit "
            f"the same wall unless `{GCE_QUOTA_ID}` has room to fall back on. Read both with "
            f"`get_zones_with_available_quota`.\n\n{stderr}"
        )
    # Neither a recognised failure nor a success — most often a create that outran its
    # timeout. An instance may well exist and be billing, so tear it down before
    # reporting, and say what was found rather than implying nothing was created.
    del_rc, del_err = await _delete_probe()
    if del_rc == 0:
        aftermath = (
            f"\n\n**An instance did exist and has been deleted.** A create that outruns its timeout "
            f"has not failed — it has only stopped being watched, and reaching this branch after a "
            f"successful delete is itself weak evidence that `{zone}` HAS capacity. Re-probe to "
            f"confirm, or go straight to a flex-start create."
        )
    else:
        aftermath = (
            f"\n\nNo instance was left behind (delete said: {del_err.strip() or 'nothing to delete'}). "
            f"The `--max-run-duration=10m` backstop on the probe would have caught it regardless."
        )
    return f"⚠️ Probe of `{zone}` failed for a reason that is neither stockout nor quota:\n{stderr}{aftermath}"


async def _update_status_file(zone: str, success_str: str, detail_str: str) -> None:
    try:
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        if not os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "w") as f:
                f.write(
                    "# TPU zone provisioning status\n\n"
                    "- **Successful Zone:** (none yet)\n\n"
                    "| Zone | Attempted | Started | Details |\n"
                    "| --- | --- | --- | --- |\n"
                )
        with open(STATUS_FILE, "r") as f:
            content = f.read()

        if success_str == "Yes":
            content = re.sub(
                r"- \*\*Successful Zone:\*\*.*", f"- **Successful Zone:** `{zone}` (Started, reached ACTIVE)", content
            )

        lines = content.splitlines()
        new_lines = []
        updated = False
        for line in lines:
            if f"**{zone}**" in line:
                new_line = f"| **{zone}** | Yes | {success_str} | {detail_str} |"
                new_lines.append(new_line)
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            new_lines.append(f"| **{zone}** | Yes | {success_str} | {detail_str} |")

        with open(STATUS_FILE, "w") as f:
            f.write("\n".join(new_lines) + "\n")
    except Exception as e:
        logger.error(f"Error updating status file: {e}")


@mcp.tool(title="Find TPU VM capacity across zones", annotations=WRITE)
async def find_tpu_vm(
    instance_name: Optional[str] = None,
    accelerator: Annotated[str, Field(description="TPU type: v6e-1/4/8 or v5p-8")] = ACCELERATOR_TYPE,
    zones: Annotated[
        Optional[list[str]],
        Field(
            description="Zones to try in order; defaults to zones publishing the machine type in a region with quota"
        ),
    ] = None,
    model_name: Optional[str] = None,
    per_zone_wait: Annotated[
        str, Field(description="How long each zone's flex-start request stays queued, e.g. '5m'")
    ] = "5m",
    provisioning_model: Annotated[
        Literal["flex-start", "spot", "on-demand", "reservation-bound"],
        Field(description="How to ask for the chip in each zone"),
    ] = PROVISIONING_MODEL,
    workload: Annotated[
        Literal["jax", "vllm"],
        Field(
            description="'jax' installs a bare jax[tpu] dev environment (no docker, no HF token); "
            "'vllm' serves model_name via docker"
        ),
    ] = "jax",
) -> str:
    """Tries TPU VM creation across zones until one grants capacity.

    This is the sweep worth running rather than picking a zone by hand, because holding
    quota does not mean the hardware is there — single v6e chips have been scarce in most
    zones checked, with three of five holding full regional quota and no chips at all.
    Capacity is zonal while quota is regional, so a sibling zone in the same region costs
    nothing to try.

    Each attempt's flex-start request expires after `per_zone_wait`, so a sweep does not
    stack pending requests across zones. On success the server's default zone moves to the
    winner and failed zones are recorded so later sweeps skip them. Follow up with
    `wait_for_jax_ready` (or `wait_for_vllm_ready` for workload='vllm') — a created
    instance is RUNNING long before it is ready."""
    instance_name = instance_name or INSTANCE_NAME
    if _gce_machine_type(accelerator) is None:
        return _unsupported_accelerator_message(accelerator)
    # Candidates are the intersection of "the catalog publishes this machine type here"
    # and "the region holds quota on the pool this provisioning model spends".
    candidate_zones = zones or await _candidate_zones(accelerator, provisioning_model)
    if not candidate_zones:
        return (
            f"❌ No candidate zones for `{accelerator}` under {provisioning_model}. Check quota with "
            "`get_zones_with_available_quota` (both metrics), or pass `zones` explicitly — an unset "
            "quota value reads exactly like a zero one, so an empty listing is not proof of nothing."
        )

    # Zones a previous sweep found dead. Recorded per rig, because the rigs request
    # different accelerator types and a zone that refuses one says nothing about another.
    skipped_zones = set()
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                for line in f.read().splitlines():
                    match = re.search(r"\|\s*\*\*([a-zA-Z0-9-]+)\*\*\s*\|\s*([^|]+)\|\s*No\s*\|", line)
                    if match:
                        skipped_zones.add(match.group(1).strip())
        except Exception as e:
            logger.error(f"Error parsing status file: {e}")

    attempts = []
    for zone in candidate_zones:
        if zone in skipped_zones:
            attempts.append(f"- **{zone}**: ⏭️ skipped (recorded as failed in {STATUS_FILE})")
            continue
        logger.info(f"Attempting {provisioning_model} TPU VM {instance_name} in {zone}...")
        result = await create_tpu_vm_instance(
            instance_name=instance_name,
            zone=zone,
            accelerator=accelerator,
            model_name=model_name,
            request_valid_for=per_zone_wait,
            provisioning_model=provisioning_model,
            workload=workload,
        )
        attempts.append(f"- **{zone}**: {result.splitlines()[0]}")
        if result.startswith("🚀"):
            await _update_status_file(zone, "Yes", f"Granted {accelerator} under {provisioning_model}.")
            global ZONE
            ZONE = zone
            return f"✅ TPU VM secured in `{zone}` (now the default zone).\n\n{result}\n\n**Attempts:**\n" + "\n".join(
                attempts
            )
        if result.startswith("⏳"):
            # gcloud timed out with the request possibly still pending in this zone;
            # stop rather than stack pending capacity requests across zones.
            return f"{result}\n\n**Attempts:**\n" + "\n".join(attempts)
        await _update_status_file(zone, "No", result.splitlines()[0][:200])

    return (
        f"❌ No zone granted {provisioning_model} capacity for `{accelerator}`.\n"
        "**Attempts:**\n" + "\n".join(attempts) + "\n\n"
        "Next: `probe_zone_capacity` on the most promising zone tells a stockout apart from a "
        "quota wall in seconds. Availability also moves within minutes, so a sweep that came up "
        "empty is worth repeating rather than treated as settled."
    )


@mcp.tool(title="Manage vLLM container", annotations=DESTRUCTIVE)
async def manage_vllm_docker(
    action: Literal["start", "stop", "restart", "status", "log", "rm"] = "start",
    instance_name: Annotated[
        Optional[str], Field(description="TPU VM instance to target; defaults to this rig's instance")
    ] = None,
    zone: Optional[str] = None,
    model_name: Annotated[
        Optional[str], Field(description="Hugging Face model ID; defaults to the configured MODEL_NAME")
    ] = None,
    load_format: Annotated[
        Optional[str],
        Field(
            description="vLLM load format, e.g. 'tpu_streaming_loader' or 'runai_streamer'; auto-picked from model size"
        ),
    ] = None,
    max_model_len: Annotated[
        Optional[int], Field(ge=1, description="Context length override; auto-picked from model size")
    ] = None,
    gpu_memory_utilization: Annotated[
        Optional[float], Field(gt=0, le=1, description="Memory utilization fraction; auto-picked from model size")
    ] = None,
) -> str:
    """Manages the vLLM Docker container on the serving TPU VM over SSH.

    Note the accelerator image ships **no Docker on PATH at first boot** — the startup
    script installs it. If you reach a VM that never ran that script, `docker` is simply
    not there, and the error says `command not found` rather than anything about images.

    'start' with any serving parameter (model_name, load_format, max_model_len,
    gpu_memory_utilization) REPLACES the container so the new configuration takes
    effect — this is how you switch the served model. A plain 'start' just restarts
    the existing container (or creates one with defaults)."""
    zone = _zone(zone)
    selected_model = model_name or MODEL_NAME
    config_given = any(p is not None for p in (model_name, load_format, max_model_len, gpu_memory_utilization))
    # Auto-detect defaults based on model name
    is_large = "26B" in selected_model or "31B" in selected_model
    resolved_load_format = load_format or ("tpu_streaming_loader" if is_large else "runai_streamer")
    resolved_max_model_len = int(max_model_len or (16384 if is_large else 65536))
    resolved_gpu_memory_utilization = float(gpu_memory_utilization or (0.80 if is_large else 0.90))

    # Use the nightly image for latest fixes. String args are shell-quoted because
    # this whole command line is executed remotely via `ssh --command`.
    docker_image = "vllm/vllm-tpu:nightly"
    docker_run_cmd = (
        f"sudo docker run --name vllm-gemma4 --privileged --net=host -d "
        f"-v /dev/shm:/dev/shm --shm-size 10gb "
        f"-e HF_HOME=/dev/shm "
        f"-e HF_HUB_DISABLE_XET=1 "
        f"-e HF_HUB_ENABLE_HF_TRANSFER=0 "
        f"-e XLA_PYTHON_CLIENT_MEM_FRACTION={resolved_gpu_memory_utilization} "
        f"-e XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"-e HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token) "
        f"{docker_image} vllm serve {shlex.quote(selected_model)} "
        f"--tensor-parallel-size {TENSOR_PARALLEL_SIZE} --disable_chunked_mm_input --max-model-len {resolved_max_model_len} "
        f"--gpu-memory-utilization {resolved_gpu_memory_utilization} "
        f"--max_num_batched_tokens 4096 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 "
        f"--load-format {shlex.quote(resolved_load_format)} "
        f'--limit-mm-per-prompt \'{{"image":0,"audio":0}}\''
    )

    # A plain start reuses an existing container; with explicit serving params the
    # container must be recreated or the params would be silently ignored.
    start_cmd = (
        f"sudo docker rm -f vllm-gemma4 >/dev/null 2>&1; {docker_run_cmd}"
        if config_given
        else f"sudo docker start vllm-gemma4 || {docker_run_cmd}"
    )
    commands = {
        "start": start_cmd,
        "stop": "sudo docker stop vllm-gemma4",
        "restart": "sudo docker restart vllm-gemma4",
        "status": "sudo docker ps -a --filter name=vllm-gemma4",
        "log": "sudo docker logs --tail 100 vllm-gemma4",
        "rm": "sudo docker rm -f vllm-gemma4",
    }
    ssh_cmd, target = _build_ssh_cmd(commands[action], instance_name, zone)

    # 'start' may fall back to `docker run`, which pulls the vLLM image (~5 min)
    # when it isn't cached — don't kill the client mid-pull.
    timeout = 600 if action in ("start", "restart") else 60
    rc, out, err = await run_command(ssh_cmd, timeout=timeout)
    if rc != 0:
        return f"""⚠️ Docker {action} failed on {target}, but the TPU itself remains safe.
Error: {err}"""
    return f"""✅ Docker {action} command executed on {target}.
{out}"""


# $/chip-hour by (family, provisioning model), read from the Cloud Billing Catalog for
# europe-west4 on 2026-08-11. Two things here are counterintuitive enough to be worth
# stating rather than deriving:
#
#   * SPOT COSTS MORE THAN FLEX-START ON v6e. It does in us-east5 too ($1.78 vs $1.40),
#     and the ordering does not invert on v5e either ($0.607 vs $0.60). "Spot" does not
#     mean "cheapest" here — read the rate rather than assuming.
#   * RESERVATION_BOUND has NO LIST RATE AT ALL. What it costs is whatever the
#     reservation was priced at, so it is deliberately absent below rather than modelled
#     as on-demand, which would quietly invent a number.
_CHIP_HOUR_RATES = {
    "v6e": {"flex-start": 1.35, "spot": 1.78, "on-demand": 2.97},
    "v5p": {"flex-start": 0.60, "spot": 1.20, "on-demand": 2.40},
    "v5e": {"flex-start": 0.60, "spot": 0.607, "on-demand": 1.20},
}


@mcp.tool(title="Estimate deployment cost", annotations=READ_ONLY)
async def estimate_deployment_cost(
    hours: Annotated[float, Field(gt=0)] = 1.0,
    tpu_type: Literal["v6e", "v5p", "v5e"] = "v6e",
    topology: Annotated[str, Field(pattern=r"^\d+(x\d+)*$", description="Chip grid, e.g. '2x4'")] = "1",
    provisioning_model: Annotated[
        Literal["flex-start", "spot", "on-demand", "reservation-bound"],
        Field(description="Which rate to apply; reservation-bound has no list rate"),
    ] = PROVISIONING_MODEL,
) -> str:
    """Estimates the cost of a TPU deployment. `topology` is a chip grid like '2x4'.

    Rates were read from the Cloud Billing Catalog for europe-west4 on 2026-08-11 and are
    not live — re-read them before using this for real budgeting. Note that spot is the
    DEARER option on v6e, so the cheap-sounding model is not the cheap one."""
    try:
        chips = math.prod(int(part) for part in topology.lower().split("x"))
        if chips <= 0:
            raise ValueError("topology dimensions must be positive")
    except ValueError as e:
        return f"❌ Invalid topology '{topology}': {e}. Expected a chip grid like '2x4'."

    if provisioning_model == "reservation-bound":
        return (
            "### 💸 No list rate for `reservation-bound`.\n"
            "Reservation-bound capacity is billed at whatever the reservation was priced at — the "
            "billing catalog publishes no SKU for it. Read the reservation rather than substituting "
            "the on-demand rate, which would overstate it by an unknown amount."
        )

    family_rates = _CHIP_HOUR_RATES.get(tpu_type, _CHIP_HOUR_RATES["v6e"])
    rate = family_rates[provisioning_model]
    total_cost = chips * rate * hours
    cheapest = min(family_rates, key=family_rates.get)
    note = ""
    if provisioning_model != cheapest:
        saving = (rate - family_rates[cheapest]) * chips * hours
        note = (
            f"\n\n`{cheapest}` is cheaper on {tpu_type} (${family_rates[cheapest]:.2f}/chip-hr) — ${saving:.2f} less."
        )
    return (
        f"### 💸 Estimated Cost: `${total_cost:.2f}` for `{hours}h` on `{chips}` chip `{tpu_type}` "
        f"({provisioning_model}, ${rate:.2f}/chip-hr, europe-west4 rates read 2026-08-11).{note}"
    )


@mcp.tool(title="Generate the deployment command", annotations=READ_ONLY)
async def get_deployment_command(
    instance_name: str = INSTANCE_NAME,
    accelerator: str = ACCELERATOR_TYPE,
    workload: Literal["jaxrust", "jax", "vllm"] = "jaxrust",
    provisioning_model: Literal["flex-start", "spot", "on-demand", "reservation-bound"] = PROVISIONING_MODEL,
) -> str:
    """Emits the `gcloud compute instances create` command this rig would run, for
    pasting into a terminal or a runbook.

    Every flag below is the Compute Engine spelling, and the mapping from the Cloud TPU
    API is not a rename in every case:

      --accelerator-type=v6e-1        ->  --machine-type=ct6e-standard-1t
      --runtime-version=v2-alpha-...  ->  --image-family=... plus --image-project
      --valid-until-duration          ->  --request-valid-for-duration
      --provisioning-model=flex-start ->  --provisioning-model=FLEX_START (SCREAMING_CASE)
      the QR produced a <id>-node     ->  the instance IS the node

    The Hugging Face token is never interpolated: the startup script fetches it from
    Secret Manager on the VM at boot, so it stays out of instance metadata (which anyone
    with compute.instances.get can read)."""
    shape = _gce_machine_type(accelerator)
    if shape is None:
        return _unsupported_accelerator_message(accelerator)
    machine_type, chips = shape
    try:
        flags = _provisioning_flags(provisioning_model, REQUEST_VALID_FOR, MAX_RUN_DURATION)
    except ValueError as e:
        return f"❌ {e}"
    script = {
        "jaxrust": "startup_script_jaxrust_template.sh",
        "jax": "startup_script_jax_template.sh",
        "vllm": "startup_script_template.sh",
    }[workload]
    lines = [
        "gcloud compute instances create " + instance_name,
        f"  --project={PROJECT_ID or 'YOUR_PROJECT'}",
        f"  --zone={ZONE}",
        f"  --machine-type={machine_type}",
        *[f"  {flag}" for flag in flags],
        *[f"  {flag}" for flag in _GCE_IMAGE_FLAGS],
        "  --maintenance-policy=TERMINATE",
        f"  --boot-disk-size={BOOT_DISK_SIZE_GB}GB",
        "  --scopes=cloud-platform",
        f"  --metadata-from-file=startup-script={script}",
    ]
    cmd = " \\\n".join(lines)
    prereq = (
        ""
        if workload in ("jax", "jaxrust")
        else f"\nRequires the `{HF_SECRET_ID}` secret (save one with `save_hf_token`) and "
        "`roles/secretmanager.secretAccessor` on the VM's service account.\n"
    )
    return (
        f"```bash\n{cmd}\n```\n"
        f"{chips} chip(s), workload `{workload}`.{prereq}\n"
        "`--scopes=cloud-platform`, `--boot-disk-size` and `--maintenance-policy=TERMINATE` are all "
        "required and all fail late — see `create_tpu_vm_instance` for what each looks like when missing."
    )


@mcp.tool(title="System status dashboard", annotations=READ_ONLY)
async def get_system_status() -> str:
    """High-level dashboard: TPU VM instances across all zones, plus vLLM health.

    Read RUNNING carefully. It means the VM booted — not that anything is serving, and
    not that the startup script even succeeded. A dead boot reports RUNNING indefinitely
    and nothing distinguishes it from a healthy one except the serial log or the port."""
    vms_str = await list_tpu_vm_instances()
    health = "🔴 Offline"
    url = await discover_vllm_url()
    if url:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"{url}/health", timeout=2)
                if res.status_code == 200:
                    health = f"🟢 Online ({url})"
        except Exception:
            pass

    if "🟢" in health:
        next_step = "Use `query_gemma4` to interact with the model."
    elif "RUNNING" in vms_str:
        next_step = (
            "A VM is RUNNING but serving isn't reachable — that is the expected state during boot, "
            "and also what a dead startup script looks like forever. Read `get_tpu_vm_serial_log`, "
            "or wait on it with `wait_for_jax_ready` / `wait_for_vllm_ready`."
        )
    else:
        next_step = "Call `create_tpu_vm_instance` (or `find_tpu_vm` to sweep zones) to provision infrastructure."

    return (
        f"### 🌀 System Status ({ZONE})\n"
        f"- **vLLM Health:** {health}\n\n"
        f"**TPU VM instances (all zones):**\n```\n{vms_str}\n```\n"
        f"**👉 Next Step:** {next_step}"
    )


@mcp.tool(title="Get vLLM endpoint", annotations=READ_ONLY)
async def get_vllm_endpoint() -> str:
    """Returns the active vLLM service URL if available."""
    url = await discover_vllm_url()
    if url:
        return f"🟢 vLLM is Online at: {url}"
    return "❌ No RUNNING TPU VM with a reachable vLLM service found."


@mcp.tool(title="Query the served model", annotations=READ_ONLY)
async def query_gemma4(prompt: str, include_stats: bool = False) -> str:
    """Queries the self-hosted model (the configured MODEL_NAME) on the active TPU
    deployment. With include_stats=True, streams the response and also reports TTFT,
    total generation time, and tokens/second."""
    logger.info(f"Querying model with prompt: '{prompt[:50]}...'")
    try:
        client, model_id = await get_vllm_client()

        if not include_stats:
            chat_completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
            )
            response = chat_completion.choices[0].message.content or "No response from model."
            logger.info(f"Model response: '{response[:100]}...'")
            return response

        start_time = time.monotonic()
        ttft = None
        response_content = ""
        total_tokens = 0

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id,
            stream=True,
        )

        async for chunk in stream:
            if ttft is None:
                ttft = time.monotonic() - start_time

            content = getattr(chunk.choices[0].delta, "content", None) or getattr(
                chunk.choices[0].delta, "reasoning", None
            )
            if content:
                response_content += content
                total_tokens += 1  # Rough token count

        end_time = time.monotonic()
        total_time = end_time - start_time

        if not response_content:
            return "❌ Model returned an empty response."

        tokens_per_second = total_tokens / (total_time - ttft) if ttft and total_time > ttft else 0

        stats_report = (
            f"### 📊 Performance Stats\n"
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
        logger.error(f"Error querying model: {e}")
        return f"❌ An error occurred while querying the model: {e}"


BENCH_RESULT_MARKER = "---BENCH-RESULT-JSON---"


def _sweep_point_from_bench_result(result: dict, concurrency: int) -> dict:
    """Maps a `vllm bench serve --save-result` JSON dump to a `throughput.sweep[]`
    entry of benchmarks/serving-report.schema.json. The full dump is embedded under
    `raw` minus its list-valued keys (per-request arrays, one element per prompt)."""

    def _stats(metric: str) -> dict:
        out = {}
        for stat in ("mean", "median", "p90", "p99"):
            v = result.get(f"{stat}_{metric}_ms")
            if isinstance(v, (int, float)):
                out[stat] = round(v, 2)
        return out

    point: dict = {"concurrency": concurrency}
    for key, src in (
        ("request_rate_rps", "request_throughput"),
        ("output_tok_per_s", "output_throughput"),
        ("total_tok_per_s", "total_token_throughput"),
    ):
        v = result.get(src)
        if isinstance(v, (int, float)):
            point[key] = round(v, 2)
    for key, metric in (("ttft_ms", "ttft"), ("tpot_ms", "tpot"), ("itl_ms", "itl")):
        stats = _stats(metric)
        if stats:
            point[key] = stats
    tpot_median = point.get("tpot_ms", {}).get("median")
    if tpot_median:
        point["per_stream_tok_per_s"] = round(1000 / tpot_median, 1)
    point["raw"] = {k: v for k, v in result.items() if not isinstance(v, list)}
    return point


@mcp.tool(title="Run vLLM benchmark", annotations=WRITE)
async def run_vllm_benchmark(
    instance_name: Annotated[
        Optional[str], Field(description="TPU VM instance to target; defaults to this rig's instance")
    ] = None,
    zone: Optional[str] = None,
    backend: str = "vllm",
    model: Optional[str] = None,
    dataset_name: str = "random",
    num_prompts: int = 100,
    random_input_len: int = 1024,
    random_output_len: int = 128,
    max_concurrency: Optional[int] = None,
    save_result: Annotated[
        bool,
        Field(
            description="Also capture the benchmark's --save-result JSON and return it as a "
            "throughput.sweep[] entry for benchmarks/serving-report.schema.json"
        ),
    ] = False,
) -> str:
    """Runs vLLM's internal benchmark tool in a separate container on the serving TPU VM.
    `model` defaults to the model actually being served (else MODEL_NAME). With
    save_result=True, the result JSON is fetched back and mapped to a sweep-point
    entry of the repo's serving-report schema, ready to paste into a report."""
    zone = _zone(zone)
    if model is None:
        url = await discover_vllm_url()
        model = (await _get_served_model_id(url) if url else None) or MODEL_NAME
    # String args are shell-quoted because the command runs remotely via `ssh --command`.
    benchmark_cmd = (
        "vllm bench serve "
        f"--backend {shlex.quote(backend)} "
        f"--model {shlex.quote(model)} "
        f"--dataset-name {shlex.quote(dataset_name)} "
        f"--num-prompts {int(num_prompts)} "
        f"--random-input-len {int(random_input_len)} "
        f"--random-output-len {int(random_output_len)}"
    )
    if max_concurrency:
        benchmark_cmd += f" --max-concurrency {int(max_concurrency)}"

    # /dev/shm is bind-mounted into the container, so a result file written there
    # survives the --rm container and can be read back in the same SSH session.
    result_name = f"vllm-bench-{os.urandom(4).hex()}.json"
    if save_result:
        benchmark_cmd += f" --save-result --result-dir /dev/shm --result-filename {result_name}"

    # We run the benchmark in a new container to not interfere with the serving container
    docker_cmd = (
        "sudo docker run --rm --privileged --net=host "
        "-v /dev/shm:/dev/shm --shm-size 10gb "
        "-e HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token) "
        f"vllm/vllm-tpu:nightly {benchmark_cmd}"
    )
    remote_cmd = docker_cmd
    if save_result:
        remote_cmd += (
            f" && echo {BENCH_RESULT_MARKER} && sudo cat /dev/shm/{result_name} && sudo rm -f /dev/shm/{result_name}"
        )

    ssh_cmd, target = _build_ssh_cmd(remote_cmd, instance_name, zone)

    rc, out, err = await run_command(ssh_cmd, timeout=600)  # Increased timeout for benchmark
    if rc != 0:
        return f"""⚠️ Benchmark failed on {target}.
Error: {err}
Output: {out}"""
    if not save_result:
        return f"""✅ Benchmark completed on {target}:
{out}"""

    bench_stdout, sep, result_json = out.partition(BENCH_RESULT_MARKER)
    if not sep:
        return f"""⚠️ Benchmark ran on {target} but no result JSON came back:
{out}"""
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        return f"""⚠️ Benchmark ran on {target} but the result JSON did not parse ({e}):
{result_json.strip()[:2000]}"""
    concurrency = int(max_concurrency) if max_concurrency else int(num_prompts)
    point = _sweep_point_from_bench_result(result, concurrency)
    return (
        f"✅ Benchmark completed on {target}.\n\n"
        "throughput.sweep[] entry (benchmarks/serving-report.schema.json):\n"
        f"```json\n{json.dumps(point, indent=2)}\n```\n\n"
        f"Benchmark output:\n{bench_stdout.strip()}"
    )


@mcp.tool(title="Get vLLM container logs", annotations=READ_ONLY)
async def get_vllm_docker_logs(
    instance_name: Annotated[
        Optional[str], Field(description="TPU VM instance to target; defaults to this rig's instance")
    ] = None,
    zone: Optional[str] = None,
    tail: Annotated[int, Field(ge=1, le=5000)] = 100,
) -> str:
    """Retrieves the last `tail` lines of the vLLM Docker container's logs from the
    serving TPU VM. Bounded — the full log of a long-serving container can run to
    megabytes."""
    log_cmd = f"sudo docker logs vllm-gemma4 --tail {int(tail)}"

    ssh_cmd, target = _build_ssh_cmd(log_cmd, instance_name, _zone(zone))

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Failed to get Docker logs from {target}.
Error: {err}"""
    return f"""✅ Docker logs from {target}:
{out}"""


@mcp.tool(title="Get TPU system logs", annotations=READ_ONLY)
async def get_tpu_system_logs(
    instance_name: Annotated[
        Optional[str], Field(description="TPU VM instance to target; defaults to this rig's instance")
    ] = None,
    zone: Optional[str] = None,
    service: Annotated[str, Field(description="systemd unit name, e.g. 'docker'")] = "docker",
    tail: Annotated[int, Field(ge=1, le=5000)] = 100,
) -> str:
    """Retrieves systemd logs for a specific service from the serving TPU VM."""
    log_cmd = f"journalctl -u {shlex.quote(service)} -n {int(tail)}"

    ssh_cmd, target = _build_ssh_cmd(log_cmd, instance_name, _zone(zone))

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Failed to get system logs from {target}.
Error: {err}"""
    return f"""✅ System logs for '{service}' from {target}:
{out}"""


def _log_entry_host(log_entry: dict) -> str:
    """Names the host a log entry came from.

    Compute Engine labels its entries instance_name/instance_id; the TPU API used
    node_id. Both are read so entries predating the migration still render.
    """
    labels = log_entry.get("resource", {}).get("labels", {})
    return labels.get("instance_name") or labels.get("instance_id") or labels.get("node_id") or "N/A"


async def _fetch_cloud_logging_logs(log_filter: str, limit: int) -> tuple[bool, int, str]:
    """Fetches Cloud Logging entries. Returns (fetch_ok, entry_count, formatted_text).

    The structured status exists so callers can tell "the fetch failed" apart from
    "the logs mention the word error" — log *content* must never be mistaken for a
    fetch failure.
    """
    cmd = ["gcloud", "logging", "read", log_filter, f"--project={PROJECT_ID}", f"--limit={limit}", "--format=json"]
    rc, out, err = await run_command(cmd)
    if rc != 0:
        return False, 0, f"❌ Failed to fetch Cloud Logs: {err}"

    try:
        logs = json.loads(out)
    except Exception:
        return True, -1, f"### ☁️ Cloud Logs (raw)\n```\n{out}\n```"

    formatted_logs = "\n".join(
        f"[{log_entry.get('timestamp')}] "
        f"{_log_entry_host(log_entry)} - "
        f"{log_entry.get('textPayload', log_entry.get('jsonPayload', {}))}"
        for log_entry in logs
    )
    return True, len(logs), f"### ☁️ Cloud Logs (filter: `{log_filter}`)\n```\n{formatted_logs}\n```"


@mcp.tool(title="Get Cloud Logging logs", annotations=READ_ONLY)
async def get_cloud_logging_logs(
    log_filter: str = 'resource.type="gce_instance"', limit: Annotated[int, Field(ge=1, le=500)] = 20
) -> str:
    """Fetches logs from Google Cloud Logging.

    A TPU VM created through Compute Engine logs as `gce_instance`, not `tpu_worker` —
    the old filter matches nothing here and returns cleanly, so it reads as "no errors"
    rather than as a wrong query."""
    _, _, text = await _fetch_cloud_logging_logs(log_filter, limit)
    return text


@mcp.tool(title="Analyze TPU error logs", annotations=READ_ONLY)
async def analyze_cloud_logging(minutes: Annotated[int, Field(ge=1, le=10080)] = 60) -> str:
    """Summarizes recent TPU errors from Cloud Logging using the self-hosted Gemma 4 model."""
    # Cloud Logging filters need an RFC3339 timestamp; relative durations like
    # "-PT60M" are not valid filter syntax.
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # gce_instance, not tpu_worker: a ct6e-* instance is an ordinary Compute Engine
    # instance to Cloud Logging. The old filter returns zero entries without erroring.
    log_filter = f'resource.type="gce_instance" severity>=ERROR timestamp>="{cutoff}"'
    fetch_ok, entry_count, logs_result = await _fetch_cloud_logging_logs(log_filter, 10)

    if not fetch_ok:
        return f"❌ Cannot analyze: the Cloud Logging fetch itself failed.\n{logs_result}"
    if entry_count == 0:
        return f"✅ No TPU error logs (severity>=ERROR) in the last {minutes} minutes — nothing to analyze."

    prompt = (
        f"Here are the recent TPU error logs:\n{logs_result}\n\n"
        "Please analyze these logs, identify the root cause of the failures, and suggest remediations."
    )
    summary = await query_gemma4(prompt)
    if summary.startswith("❌"):
        return (
            f"⚠️ Fetched {entry_count} error log entries but the self-hosted model is unavailable "
            f"to analyze them:\n{summary}\n\n{logs_result}"
        )
    return f"### 🔍 Log Analysis Summary\n\n{summary}\n\n{logs_result}"


# Metric families worth surfacing by default; the raw /metrics dump is mostly
# histogram buckets and runs to tens of KB.
_KEY_METRIC_NAMES = (
    "vllm_requests_running",
    "vllm_requests_swapped",
    "vllm_requests_waiting",
    "vllm_tpu_cache_usage_perc",
    "process_resident_memory_bytes",
)


def _filter_key_metrics(metrics_text: str) -> list[str]:
    return [
        line
        for line in metrics_text.splitlines()
        if not line.startswith("#") and any(name in line for name in _KEY_METRIC_NAMES)
    ]


@mcp.tool(title="Get model & engine details", annotations=READ_ONLY)
async def get_model_details() -> str:
    """
    Retrieves detailed information about the running model, vLLM engine, and versions.

    Provides a verbose report including:
    - Model ID and details from the vLLM engine.
    - vLLM version and build information.
    - Health status.
    - Key performance metrics.
    """
    url = await discover_vllm_url()
    if not url:
        return "❌ No RUNNING TPU VM with a reachable vLLM service found."

    report = f"### 🧩 Model & vLLM Engine Details ({url})\n\n"

    async with httpx.AsyncClient(timeout=10) as client:
        # 1. Get Model Details from /v1/models
        try:
            models_res = await client.get(f"{url}/v1/models")
            if models_res.status_code == 200:
                models_data = models_res.json()
                report += "**Model Information (`/v1/models`):**\n"
                report += f"```json\n{json.dumps(models_data, indent=2)}\n```\n"
            else:
                report += f"⚠️ Could not fetch model details. Status: {models_res.status_code}\n"
        except Exception as e:
            report += f"❌ Error fetching model details: {e}\n"

        # 2. Get vLLM Version from /version
        try:
            version_res = await client.get(f"{url}/version")
            if version_res.status_code == 200:
                version_data = version_res.json()
                report += "**vLLM Version (`/version`):**\n"
                report += f"- Version: `{version_data.get('version', 'N/A')}`\n\n"
            else:
                report += f"⚠️ Could not fetch vLLM version. Status: {version_res.status_code}\n\n"
        except Exception as e:
            report += f"❌ Error fetching vLLM version: {e}\n\n"

        # 3. Get Health Status from /health
        try:
            health_res = await client.get(f"{url}/health")
            if health_res.status_code == 200:
                report += "**Health Status (`/health`):**\n- Status: `Healthy` ✅\n\n"
            else:
                report += (
                    f"**Health Status (`/health`):**\n- Status: `Unhealthy` ❌ (Code: {health_res.status_code})\n\n"
                )
        except Exception as e:
            report += f"❌ Error fetching health status: {e}\n\n"

        # 4. Get Metrics from /metrics
        try:
            metrics_res = await client.get(f"{url}/metrics")
            if metrics_res.status_code == 200:
                report += "**Key vLLM Metrics (`/metrics`):**\n"
                key_metrics = _filter_key_metrics(metrics_res.text)
                if key_metrics:
                    report += "```\n" + "\n".join(key_metrics) + "\n```\n"
                else:
                    report += "Metrics endpoint available, but no key metrics found in snippet.\n"
            else:
                report += "⚠️ Metrics endpoint not available or failed.\n"
        except Exception as e:
            report += f"❌ Error fetching metrics: {e}\n"

    return report


@mcp.tool(title="Help & configuration", annotations=READ_ONLY)
async def get_help() -> str:
    """Provides help text and summarizes the configuration options and all available SRE/DevOps tools for this TPU VM MCP server."""
    return (
        f"### 🛠️ {RIG_NAME} — Gemma 4 on TPU via Compute Engine\n\n"
        "This rig provisions **only** through Compute Engine (`gcloud compute instances create`). "
        "It holds no queued-resource path: the Cloud TPU API is no longer under active development, "
        "and TPU7x and later are Compute Engine or GKE only.\n\n"
        "Configure it with these environment variables:\n\n"
        f"- **`GOOGLE_CLOUD_PROJECT`**: Your GCP Project ID (falls back to the active gcloud config).\n"
        f"  - *Current Value:* `{PROJECT_ID or '(not set)'}`\n"
        f"- **`GOOGLE_CLOUD_ZONE`**: The default zone (`find_tpu_vm` moves it to wherever capacity lands).\n"
        f"  - *Current Value:* `{ZONE}`\n"
        f"- **`GOOGLE_CLOUD_REGION`**: The region quota is metered against.\n"
        f"  - *Current Value:* `{REGION}`\n"
        f"- **`MODEL_NAME`**: Default Hugging Face repository or path.\n"
        f"  - *Current Value:* `{MODEL_NAME}`\n"
        f"- **`ACCELERATOR_TYPE`**: Documentation only on this path — the machine type derived from it "
        "is what Compute Engine consumes. v6e-1/4/8 and v5p-8 have a Compute Engine path; **v5e does not**.\n"
        f"  - *Current Value:* `{ACCELERATOR_TYPE}` -> `{(_gce_machine_type(ACCELERATOR_TYPE) or ('(no CE path)', 0))[0]}`\n"
        f"- **`INSTANCE_NAME`**: The instance this rig manages. The instance IS the node here — no "
        "Queued Resource indirection, no derived `<id>-node`.\n"
        f"  - *Current Value:* `{INSTANCE_NAME}`\n"
        f"- **`PROVISIONING_MODEL`**: flex-start | spot | on-demand | reservation-bound. Flex-start is "
        "cheapest on v6e and **spot is dearer**; read `estimate_deployment_cost` rather than assuming.\n"
        f"  - *Current Value:* `{PROVISIONING_MODEL}`\n"
        f"- **`REQUEST_VALID_FOR`** / **`MAX_RUN_DURATION`**: how long a flex-start request queues "
        "(2h cap for a standalone VM), and how long the VM may run once granted (10 min to 7 days).\n"
        f"  - *Current Values:* `{REQUEST_VALID_FOR}` / `{MAX_RUN_DURATION}`\n"
        f"- **`BOOT_DISK_SIZE_GB`**: the image default is 10GB and cannot hold the vLLM image.\n"
        f"  - *Current Value:* `{BOOT_DISK_SIZE_GB}`\n"
        f"- **`IMAGE_FAMILY`** / **`IMAGE_PROJECT`**: replaces the TPU API's `--runtime-version`. Pin the "
        "family, never a dated build — superseded builds go DEPRECATED within a fortnight.\n"
        f"  - *Current Values:* `{IMAGE_FAMILY}` / `{IMAGE_PROJECT}`\n"
        f"- **`GCE_SPOT_QUOTA_ID`** / **`GCE_QUOTA_ID`**: the two metrics that gate creation. Flex-start "
        "spends the **preemptible** one first and falls back to the family one, and their defaults are "
        "opposite (family absent = 0, preemptible absent = 1536).\n"
        f"  - *Current Values:* `{GCE_SPOT_QUOTA_ID}` / `{GCE_QUOTA_ID}` (tpu_family=`{GCE_TPU_FAMILY}`)\n"
        f"- **`RUST_TOOLCHAIN`**: rustup toolchain installed on workload='jaxrust' VMs. Pinned — two "
        "engine crates are 2024-edition, which Ubuntu 22.04's rustc 1.75 cannot build at all.\n"
        f"  - *Current Value:* `{RUST_TOOLCHAIN}`\n"
        f"- **`LIBTPU_SPEC`**: the libtpu package installed with --no-deps on workload='jaxrust' VMs. "
        "No JAX, no jaxlib: the Rust engine dlopens the .so directly.\n"
        f"  - *Current Value:* `{LIBTPU_SPEC}`\n"
        f"- **`JAXRUST_CARGO_FEATURES`**: cargo features for the engine build. `gemma` pulls in "
        "rlx-gemma, which is **GPL-3.0-only** while the rest of this rig is permissive — see "
        "`rust/NOTICE.md`. Drop it to build the probe and server shell alone.\n"
        f"  - *Current Value:* `{JAXRUST_CARGO_FEATURES}`\n"
        f"- **`JAXRUST_REMOTE_DIR`**: where the engine source and binaries live on the VM.\n"
        f"  - *Current Value:* `{JAXRUST_REMOTE_DIR}`\n"
        f"- **`JAX_PYTHON_VERSION`**: CPython installed from deadsnakes on workload='jax' VMs "
        "(Ubuntu 22.04's system 3.10 pins JAX to an old release).\n"
        f"  - *Current Value:* `{JAX_PYTHON_VERSION}`\n"
        f"- **`JAX_PIP_SPEC`**: JAX package spec for workload='jax' VMs; libtpu resolves from the "
        "JAX releases index. Pin it (e.g. `jax[tpu]==0.11.0`) for reproducible runs.\n"
        f"  - *Current Value:* `{JAX_PIP_SPEC}`\n"
        f"- **`JAX_PIP_EXTRAS`**: best-effort extras installed after the TPU stack (non-fatal).\n"
        f"  - *Current Value:* `{JAX_PIP_EXTRAS}`\n"
        f"- **`CPU_DEBUG_MACHINE_TYPE`**: machine type for `create_cpu_debug_vm`. Memory is the "
        "spec that matters — 128 GiB to run a 31B on CPU, 64 GiB to only load it.\n"
        f"  - *Current Value:* `{CPU_DEBUG_MACHINE_TYPE}`\n"
        f"- **`CPU_DEBUG_PIP_SPEC`**: packages installed on CPU debug boxes (no libtpu).\n"
        f"  - *Current Value:* `{CPU_DEBUG_PIP_SPEC}`\n"
        "A Hugging Face token must exist as Secret Manager secret `hf-token` "
        "(save one with `save_hf_token`) before creating a workload='vllm' VM "
        "(workload='jaxrust' and workload='jax' VMs don't need it).\n\n"
        "---\n\n"
        "### ⚠️ Three things that do not fail loudly\n\n"
        "- **RUNNING is not ready.** An instance is RUNNING the moment the VM boots, before the startup "
        "script has installed anything. A dead boot says RUNNING indefinitely. Only the serial log or "
        "the port tells you the difference — `wait_for_jaxrust_ready` / `wait_for_jax_ready` / "
        "`wait_for_vllm_ready` read them.\n"
        "- **PENDING is either quota or capacity**, and a create never says which. "
        "`probe_zone_capacity` settles it in seconds, for free.\n"
        "- **Quota is permission to ask, not reserved hardware.** Zones have held full regional quota "
        "and no chips at all.\n\n"
        "---\n\n"
        "### 🧰 Available MCP Tools\n\n"
        "#### 🐳 Capacity & Lifecycle\n"
        "- **`create_tpu_vm_instance`**: Creates a TPU VM via Compute Engine; workload='jaxrust' "
        "(default) installs a Rust toolchain + libtpu, workload='jax' installs the Python jax[tpu] "
        "stack, workload='vllm' auto-starts serving.\n"
        "- **`find_tpu_vm`**: Sweeps candidate zones attempting creation until one grants capacity, "
        "recording the failures so later sweeps skip them.\n"
        "- **`probe_zone_capacity`**: Fires a throwaway SPOT create to tell a stockout apart from a "
        "quota wall — spot does not queue, so it fails fast and names the reason.\n"
        "- **`get_zones_with_available_quota`**: Compute Engine TPU quota by region, for BOTH metrics.\n"
        "- **`wait_for_jaxrust_ready`**: Polls the serial marker until the Rust + libtpu environment is "
        "ready (or failed). Ready means installed, not compiled and not proven.\n"
        "- **`deploy_jaxrust_engine`**: Uploads `rust/`, builds it release-mode on the VM, and runs "
        "`xla-probe` — which compiles StableHLO on the chip and checks the result.\n"
        "- **`verify_rust_tpu`**: Re-runs the probe. Stricter than `verify_jax_tpu`: it asserts on a "
        "computed value, not on a device list.\n"
        "- **`manage_jaxrust_server`**: start / stop / status / logs for the `gemma4-engine` process.\n"
        "- **`wait_for_jax_ready`**: Polls the serial marker until the JAX environment is ready (or failed).\n"
        "- **`verify_jax_tpu`**: Re-runs the JAX device check over SSH (asserts jax.devices() sees a TPU).\n"
        "- **`wait_for_vllm_ready`**: Polls health endpoint + serial marker until serving is up (~10 min loads).\n"
        "- **`create_cpu_debug_vm`**: Plain CPU VM (default Spot e2-highmem-16/128 GiB) running the "
        "same JAX stack minus libtpu — correctness work off the TPU, for cents an hour.\n"
        "- **`list_tpu_vm_instances`**: Lists TPU VM instances (ct6e/ct5p) with IPs and status.\n"
        "- **`destroy_tpu_vm_instance`**: Deletes a TPU VM instance (stops billing).\n"
        "- **`get_tpu_vm_serial_log`**: Tails a TPU VM's serial console (the primary progress signal; "
        "SSH to TPU VMs is often firewalled).\n"
        "- **`get_tpu_vm_endpoint`**: Resolves and health-probes a TPU VM's vLLM endpoint.\n"
        "- **`get_deployment_command`**: The `gcloud compute instances create` command, for a runbook.\n"
        "- **`estimate_deployment_cost`**: Cost by provisioning model (spot is dearer than flex-start on v6e).\n"
        "- **`find_gpu`**: GPU VMs, Cloud Run GPU services, and GPU quota in the project.\n\n"
        "#### 🚀 Serving\n"
        "- **`manage_vllm_docker`**: start/stop/restart/status/log/rm for the vLLM container over SSH.\n"
        "- **`get_vllm_endpoint`**: Active vLLM service URL.\n"
        "- **`save_hf_token`**: Securely saves a Hugging Face API token to Secret Manager.\n\n"
        "#### 📊 Monitoring & Logs\n"
        "- **`get_system_status`**: High-level status dashboard of TPU VM state and vLLM service.\n"
        "- **`verify_model_health`**: Verifies model inference health with a simple prompt.\n"
        "- **`get_model_details`**: Model, vLLM version, health, and key metrics report.\n"
        "- **`get_metrics`**: Raw Prometheus metrics from the vLLM /metrics endpoint.\n"
        "- **`get_vllm_docker_logs`**: Logs from the vLLM Docker container on the TPU VM.\n"
        "- **`get_tpu_system_logs`**: systemd logs for a service on the TPU VM.\n"
        "- **`get_cloud_logging_logs`**: Cloud Logging, filtered on `gce_instance` (not `tpu_worker`).\n"
        "- **`analyze_cloud_logging`**: Summarizes recent errors using the self-hosted Gemma 4 model.\n\n"
        "#### 📈 Inference & Benchmarking\n"
        "- **`query_gemma4`**: Queries the served model (include_stats=True adds TTFT/throughput).\n"
        "- **`run_vllm_benchmark`**: Runs `vllm bench serve` in a separate container on the VM; "
        "save_result=True returns a serving-report sweep-point JSON.\n"
        "- **`get_help`**: This help text."
    )


@mcp.tool(title="Get vLLM metrics", annotations=READ_ONLY)
async def get_metrics(raw: bool = False) -> str:
    """Fetches Prometheus metrics from the running vLLM service's /metrics endpoint.
    Returns the key serving metrics by default; raw=True returns the full dump
    (large — mostly histogram buckets)."""
    url = await discover_vllm_url()
    if not url:
        return "❌ No RUNNING TPU VM with a reachable vLLM service found."

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{url}/metrics")
            if res.status_code != 200:
                return f"❌ Failed to fetch metrics. Status code: {res.status_code}\nResponse: {res.text}"
            if raw:
                return res.text
            key_metrics = _filter_key_metrics(res.text)
            if not key_metrics:
                return "Metrics endpoint reachable, but no key serving metrics found (use raw=True for the full dump)."
            return "\n".join(key_metrics)
    except Exception as e:
        return f"❌ Error connecting to vLLM metrics endpoint: {e}"


@mcp.tool(title="Find GPU resources", annotations=READ_ONLY)
async def find_gpu(
    service: str = "compute.googleapis.com",
    quota_id: str = "NVIDIA-L4-GPUS-per-project-zone",
) -> str:
    """
    Finds available GPU resources (GCE VMs, Cloud Run services, and zones with available GPU quota) in the GCP project.
    """
    # 1. Fetch GPU VM instances
    gce_cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={PROJECT_ID}",
        "--format=json(name,zone,machineType,status,guestAccelerators)",
    ]
    rc_g, stdout_g, stderr_g = await run_command(gce_cmd)
    gpu_vms = []
    if rc_g == 0 and stdout_g:
        try:
            instances = json.loads(stdout_g)
            for inst in instances:
                guest_acc = inst.get("guestAccelerators", [])
                machine_type = inst.get("machineType", "")
                is_gpu = len(guest_acc) > 0 or any(x in machine_type.lower() for x in ["g2-", "a2-", "a3-"])
                if is_gpu:
                    zone = inst.get("zone", "").split("/")[-1]
                    mtype = machine_type.split("/")[-1]
                    acc_info = []
                    for acc in guest_acc:
                        acc_type = acc.get("acceleratorType", "").split("/")[-1]
                        acc_count = acc.get("acceleratorCount", 1)
                        acc_info.append(f"{acc_count}x {acc_type}")
                    acc_str = ", ".join(acc_info) if acc_info else "Yes"
                    gpu_vms.append(
                        {
                            "name": inst.get("name"),
                            "zone": zone,
                            "machine_type": mtype,
                            "status": inst.get("status"),
                            "accelerators": acc_str,
                        }
                    )
        except Exception as e:
            logger.error(f"Error parsing GCE VMs: {e}")

    # 2. Fetch Cloud Run services
    run_cmd = [
        "gcloud",
        "run",
        "services",
        "list",
        f"--project={PROJECT_ID}",
        "--format=json(metadata.name,status.address.url,spec.template.spec.containers)",
    ]
    rc_r, stdout_r, stderr_r = await run_command(run_cmd)
    gpu_services = []
    if rc_r == 0 and stdout_r:
        try:
            services = json.loads(stdout_r)
            for svc in services:
                metadata = svc.get("metadata", {})
                name = metadata.get("name", "")
                status = svc.get("status", {})
                url = status.get("address", {}).get("url", "")
                spec = svc.get("spec", {})
                containers = spec.get("template", {}).get("spec", {}).get("containers", [])
                has_gpu = False
                gpu_count = 0
                for container in containers:
                    resources = container.get("resources", {})
                    limits = resources.get("limits", {})
                    if "run.googleapis.com/gpu" in limits:
                        has_gpu = True
                        gpu_count = limits["run.googleapis.com/gpu"]
                if has_gpu or name.startswith("gpu-"):
                    gpu_services.append(
                        {
                            "name": name,
                            "url": url,
                            "gpus": f"{gpu_count}x nvidia-l4" if gpu_count else "1x nvidia-l4 (Estimated)",
                        }
                    )
        except Exception as e:
            logger.error(f"Error parsing Cloud Run services: {e}")

    # 3. Fetch GPU quotas
    quota_cmd = [
        "gcloud",
        "beta",
        "quotas",
        "info",
        "list",
        f"--service={service}",
        f"--project={PROJECT_ID}",
        f"--filter=quotaId:{quota_id}",
        "--format=json",
    ]
    rc_q, stdout_q, stderr_q = await run_command(quota_cmd)
    gpu_quotas = []
    if rc_q == 0 and stdout_q:
        try:
            quota_data = json.loads(stdout_q)
            for info in quota_data:
                dimensions_infos = info.get("dimensionsInfos", [])
                for dim_info in dimensions_infos:
                    details = dim_info.get("details", {})
                    limit_val = details.get("value")
                    if limit_val and limit_val != "0":
                        dim_map = dim_info.get("dimensions", {})
                        zone_val = dim_map.get("zone") or dim_map.get("region")
                        if zone_val:
                            gpu_quotas.append((zone_val, limit_val))
                        else:
                            locations = dim_info.get("applicableLocations", [])
                            for loc in locations:
                                gpu_quotas.append((loc, limit_val))
            gpu_quotas = sorted(list(set(gpu_quotas)))
        except Exception as e:
            logger.error(f"Error parsing GPU quotas: {e}")

    # Build report
    report = []
    report.append("# 🚀 GCP GPU Resource Discovery Report")
    report.append(f"**Project:** `{PROJECT_ID}`\n")

    report.append("## 🖥️ Compute Engine GPU VMs")
    if gpu_vms:
        report.append("| VM Name | Zone | Machine Type | Status | Accelerator(s) |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        for vm in gpu_vms:
            report.append(
                f"| **{vm['name']}** | `{vm['zone']}` | `{vm['machine_type']}` | `{vm['status']}` | {vm['accelerators']} |"
            )
    else:
        report.append("_No GPU VM instances found in the project._")
    report.append("")

    report.append("## 🐳 Cloud Run GPU Services")
    if gpu_services:
        report.append("| Service Name | GPU Configuration | Active Endpoint URL |")
        report.append("| :--- | :--- | :--- |")
        for svc in gpu_services:
            report.append(f"| **{svc['name']}** | `{svc['gpus']}` | [{svc['url']}]({svc['url']}) |")
    else:
        report.append("_No Cloud Run GPU services found in the project._")
    report.append("")

    report.append("## 📊 Available GPU Quotas (nvidia-l4)")
    if gpu_quotas:
        report.append("| Zone | Limit (Value) |")
        report.append("| :--- | :--- |")
        for zone_name, limit in gpu_quotas:
            limit_display = "Default (-1)" if limit == "-1" else limit
            report.append(f"| `{zone_name}` | {limit_display} |")
    else:
        report.append("_No zones found with available NVIDIA L4 GPU quota._")

    return "\n".join(report)


if __name__ == "__main__":
    mcp.run()
