import asyncio
import json
import logging
import os
import re
import shlex
import sys
import tempfile
import time
from typing import NamedTuple, Optional

import httpx
from google.cloud import secretmanager
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

# Setup logging
# The rig directory name is this deployment's identity — it names the MCP server, names the
# log channel, and is the default id for every resource the tools provision. Sibling rigs
# share a project and a zone, so deriving from the directory is what keeps one rig's tools
# off another rig's capacity, and its log lines distinguishable from another rig's.
# NAMING.md constrains the directory to lowercase/digits/hyphens, which is a valid GCP id.
RIG_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(RIG_NAME)

# --- Configuration ---
# tpu.env is the single source of truth for the deployment parameters. load_dotenv does
# not overwrite variables that are already set, so a real environment variable (or an
# MCP client's env block) still wins over the file.
#
# This has to run before FastMCP is constructed: MCP_SERVER_NAME is read below, and a value
# set in tpu.env would otherwise arrive too late to name the server.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpu.env"))
except Exception as e:  # python-dotenv missing or unreadable file — fall back to defaults
    logger.warning(f"Could not load tpu.env, using environment and defaults only: {e}")

# Name this server advertises over MCP. It has to match the key the client registers it
# under (mcp_config.json / .mcp.json), because that key is what prefixes every tool —
# mcp__<key>__find_tpu. Every sibling rig used to answer to "tpu-devops", so with more than
# one registered you could not tell which rig a tool call would reach. Defaulting to the rig
# directory makes the prefix name the rig; MCP_SERVER_NAME overrides it for a client that
# has already registered this server under something else.
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", RIG_NAME)

# Initialize FastMCP server
mcp = FastMCP(MCP_SERVER_NAME)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
ZONE = os.getenv("GOOGLE_CLOUD_ZONE", "us-east5-b")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-east5")
# The reference bf16 31B release. 62 GB on disk, 57.7 GiB resident, ~7.2 GiB per chip at
# TENSOR_PARALLEL_SIZE=8. tpu.env carries the full sizing note; the constraint on this
# checkpoint is KV, not weights.
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
# Secret Manager secret holding the Hugging Face token. The startup script fetches it by
# id at boot, so a rotated or per-project secret only needs this to change.
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "hf-token")
# Default GCE instance name for every tool. On this path the instance *is* the node — there
# is no Queued Resource indirection and no derived "<id>-node", which is the single biggest
# simplification the Compute Engine path buys. Follows the directory, but tpu.env still wins
# — pin INSTANCE_NAME there when a name has to outlive a rename, because a rename otherwise
# orphans whatever is already provisioned.
INSTANCE_NAME = os.getenv("INSTANCE_NAME", RIG_NAME)
# Back-compat alias: the forked tools and tests spell this RESOURCE_ID. Same value, and on
# this path it names an instance rather than a queued resource.
RESOURCE_ID = INSTANCE_NAME
# Force GOOGLE_APPLICATION_CREDENTIALS and HOME if missing in environment.
# An MCP client can launch this with no HOME, and both gcloud and ADC need one. Ask the
# passwd database for the account's real home rather than baking in one machine's path —
# expanduser("~") falls back to pwd.getpwuid() precisely when HOME is unset.
if "HOME" not in os.environ:
    os.environ["HOME"] = os.path.expanduser("~")
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    default_adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if os.path.exists(default_adc_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_adc_path

# ACCELERATOR_TYPE is documentation on this path — the Cloud TPU API's spelling of the slice,
# kept so reports and the directory name line up with the twin rig. What gcloud actually
# consumes is MACHINE_TYPE below. Never pass this to `gcloud compute instances create`.
ACCELERATOR_TYPE = os.getenv("ACCELERATOR_TYPE", "v6e-8")

# The Compute Engine machine type is the real accelerator request. `ct6e-standard-8t` is
# EIGHT v6e chips on ONE host, 360 vCPU / 1440 GB, published in the same 19 zones as the
# 1-chip shape (verified 2026-08-19). Single host means every collective travels the on-board
# ICI. There is a second family spelled `<name>-tpu` (identical vCPU, memory and zone
# coverage; `guestAcceleratorType: tpu-v6e` instead of `ct6e`) — see ../HARDWARE.md. They are
# not known to be interchangeable, so the exact string is config.
MACHINE_TYPE = os.getenv("MACHINE_TYPE", "ct6e-standard-8t")


def _chips_in_machine_type(machine_type: str) -> Optional[int]:
    """Chip count from the trailing `-<N>t` of a TPU machine type, or None.

    `ct6e-standard-8t` and `ct6e-standard-8t-tpu` both give 8; `ct5p-hightpu-4t` gives 4.
    Derived rather than configured because the two must never disagree — this rig bills and
    spends quota per chip, so a stale literal here understates both by a factor of eight.
    CHIP_COUNT overrides it for a future shape that stops encoding the count in its name.
    """
    m = re.search(r"-(\d+)t(?:-tpu)?$", machine_type)
    return int(m.group(1)) if m else None


# Chips in one instance, and the mesh they form. Quota is metered in chips and every catalog
# rate is per chip-hour, so both the quota arithmetic and `estimate_deployment_cost` scale by
# this. v6e-8 is a 2x4 mesh; the v6e-1 fork this rig came from was 1x1, and that single number
# is the whole difference in the cost and quota story.
CHIP_COUNT = int(os.getenv("CHIP_COUNT") or _chips_in_machine_type(MACHINE_TYPE) or 1)
TOPOLOGY = os.getenv("TOPOLOGY", "2x4")
# Replaces the TPU API's --runtime-version. Ubuntu 22.04 / kernel 6.8, shared across
# v5e/v5p/v6e, preloaded with the TPU runtime, drivers and agents. Pin the *family*, never a
# dated build: images ship roughly weekly and every superseded build goes DEPRECATED.
IMAGE_FAMILY = os.getenv("IMAGE_FAMILY", "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e")
IMAGE_PROJECT = os.getenv("IMAGE_PROJECT", "ubuntu-os-accelerator-images")
# The image default is 10 GB, which cannot hold the vLLM TPU image. Undersizing this fails
# late — after boot, during the docker pull — so it is a default rather than a caller's problem.
BOOT_DISK_SIZE_GB = os.getenv("BOOT_DISK_SIZE_GB", "200")

# Quota ids, and the trap that defines this rig: **the two control planes meter against
# different pools.** The TPU API ids below are what the twin tpu-vllm-v6e8-2b uses; they
# are kept only so estimate/compare tooling can report both. Compute Engine does not consult
# them. Verified 2026-08-10 — the project holds 512 v6e chips in us-east5 under the TPU API
# and **no stated CT6E value** under Compute Engine.
TPU_QUOTA_ID = os.getenv("TPU_QUOTA_ID", "TPUV6EPerProjectPerZoneForTPUAPI")
TPU_SPOT_QUOTA_ID = os.getenv("TPU_SPOT_QUOTA_ID", "TPUV6EPreemptiblePerProjectPerZoneForTPUAPI")
# What Compute Engine actually meters.
#
#   FLEX_START -> GCE_SPOT_QUOTA_ID first, GCE_QUOTA_ID as FALLBACK
#   SPOT       -> GCE_SPOT_QUOTA_ID  (the preemptible pool)
#   STANDARD   -> GCE_QUOTA_ID       (regional, family-wide, tpu_family=CT6E)
#
# FLEX-START SPENDS THE PREEMPTIBLE QUOTA FIRST AND FALLS BACK TO THE STANDARD ONE. Google's
# https://docs.cloud.google.com/compute/docs/instances/provisioning-models states it in full:
#   "When you create a Flex-start VM, preemptible quota is consumed. If your project lacks
#    preemptible quota, then standard quota is consumed."
# Counterintuitive, since flex-start is not preemptible in behaviour — once granted it runs
# uninterrupted. Corrected twice: this file first grouped flex-start with on-demand (wrong),
# then said it spends preemptible and NOT the family quota (also wrong — the family quota is
# the documented fallback). A flex-start zone is usable if EITHER pool has room.
#
# The preemptible id is used at REGION scope. The per-zone spelling exists but carries no
# entries in this project; the per-region one holds the values.
GCE_QUOTA_ID = os.getenv("GCE_QUOTA_ID", "TPUS-PER-TPU-FAMILY-per-project-region")
GCE_SPOT_QUOTA_ID = os.getenv("GCE_SPOT_QUOTA_ID", "PREEMPTIBLE-TPU-V6E-per-project-region")
GCE_TPU_FAMILY = os.getenv("GCE_TPU_FAMILY", "CT6E")
# Tensor parallel degree. 8 uses the whole slice, and on this checkpoint it is not optional:
# 57.7 GiB of weights do not fit one 31.24 GiB chip, so TP=1 and TP=2 cannot boot at all.
#
# The KV cache shards across KV heads, not weight columns, so a layer's cache shards
# min(TP, num_kv_heads) ways and the runtime pads the head count up to TP when it falls short.
# ../MODELS.md gives the 31B two geometries straddling 8: its 50 sliding layers carry 16 KV
# heads and shard exactly, its 10 full layers carry 4 and pad to 8. That is a 2x KV penalty on
# a sixth of the layers — 960 KiB/token against an ideal 880, a 9.1% overhead. Do not carry
# E2B's version of this note over: E2B was full MQA (num_key_value_heads=1), every layer at the
# limit, and TP=8 multiplied its KV by 8. See tpu.env and
# When_TP_Crosses_the_KV_Head_Count_v6e8.pdf, which measures this crossing on this checkpoint.
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "8"))

# How an instance asks for capacity. These are gcloud's `--provisioning-model` values for
# `compute instances create`, and they are SCREAMING_CASE where the TPU API's are lowercase
# hyphenated — flex-start/FLEX_START is the same request to a different API. RESERVATION_BOUND
# exists on this path and has no Queued Resource equivalent.
PROVISIONING_MODELS = ("flex-start", "spot", "on-demand", "reservation-bound")
PROVISIONING_MODEL = os.getenv("PROVISIONING_MODEL", "flex-start")
# How long to wait for flex-start capacity, and how long the VM may run once granted.
REQUEST_VALID_FOR = os.getenv("REQUEST_VALID_FOR", "2h")
MAX_RUN_DURATION = os.getenv("MAX_RUN_DURATION", "4h")
# The reservation RESERVATION_BOUND consumes. Empty by default because this project holds no
# future reservation; a calendar-mode one is created out of band with
# `gcloud compute future-reservations create --reservation-mode=CALENDAR`, and is pinned to
# the zone it was reserved in. Only ever read for provisioning_model='reservation-bound'.
RESERVATION_NAME = os.getenv("RESERVATION_NAME", "")

# Cloud Billing Catalog service id for Compute Engine, which is where the TPU SKUs live.
COMPUTE_BILLING_SERVICE_ID = "6F81-5844-456A"
_SKU_CACHE: dict[str, list] = {}

# How each provisioning model is spelled in SKU descriptions, anchored at the start so the
# Reserved/Commitment/Capacity-Optimized variants of the same chip do not match. The catalog
# calls flex-start "DWS Defined Duration" (Dynamic Workload Scheduler) and drops the "Tpu"
# prefix there, and calls spot "Preemptible". The three spellings hold for v6e exactly as
# they did for v5e (`TpuV6e running in Columbus`, `DWS Defined Duration V6e running in
# Columbus`, `TpuV6e attached to Spot Preemptible VMs running in Columbus`), so tpu_type is
# the only thing a chip retarget changes here. Verified against the live catalog 2026-08-07.
_SKU_DESCRIPTION_PATTERNS = {
    "on-demand": r"^Tpu{fam} running in ",
    "flex-start": r"^DWS Defined Duration {fam} running in ",
    "spot": r"^Tpu{fam} attached to Spot Preemptible VMs",
}
_SKU_USAGE_TYPES = {"on-demand": "OnDemand", "flex-start": "OnDemand", "spot": "Preemptible"}
LOCAL_DOCKER_IMAGE = os.getenv("LOCAL_DOCKER_IMAGE", "")

# Serving parameters. Both deployment paths (the boot-time startup script and
# manage_vllm_docker) read these, so a container recreated by hand serves the same
# config the queued resource booted with.
VLLM_IMAGE = "vllm/vllm-tpu:nightly"
# Per-request ceiling, not an allocation. tpu.env is the source of truth (32768); this
# default matches it so a bare `python3 server.py` serves the same config.
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "32768")
MAX_NUM_BATCHED_TOKENS = os.getenv("MAX_NUM_BATCHED_TOKENS", "4096")
LIMIT_MM_PER_PROMPT = os.getenv("LIMIT_MM_PER_PROMPT", '{"image":4,"audio":1}')

# GCS object holding a pre-staged Hugging Face cache tar for MODEL_NAME, or "" to pull from
# Hugging Face at boot as before. Purely an optimization: the startup script verifies the
# object is readable and falls back to the online pull on any failure. See
# `stage_model_to_gcs.sh` and the "Warm the Hugging Face cache" block in the template.
MODEL_GCS_URI = os.getenv("MODEL_GCS_URI", "")

# Empty by default: gcloud then uses the project's default network. Set these only if
# the project really has a custom VPC — and remember subnetworks are regional, so a
# named subnet exists in one region only and will break find_tpu's cross-zone sweep.
TPU_NETWORK = os.getenv("TPU_NETWORK", "")
TPU_SUBNETWORK = os.getenv("TPU_SUBNETWORK", "")

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
        except Exception:
            pass
        return -1, "", f"Timeout after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


async def _render_tool_catalog() -> str:
    """Renders the live tool list as markdown by introspecting the MCP registry.

    Built from the registered tools rather than a hardcoded list so it can never
    drift out of sync with the @mcp.tool() decorators below.
    """
    lines = []
    for tool in sorted(await mcp.list_tools(), key=lambda t: t.name):
        summary = (tool.description or "").strip().splitlines()
        first_line = summary[0].strip() if summary else "No description."
        lines.append(f"- **`{tool.name}`**: {first_line}")
    return "\n".join(lines)


async def _get_node_id(instance_name: str, zone: str = ZONE) -> Optional[str]:
    """Confirms an instance exists and returns its name.

    On the Queued Resource path this had real work to do — a QR holds a node spec whose
    nodeId is derived (`<resource_id>-node`), so the id you asked for and the node you got
    were different strings, and three separate helpers existed to reconcile them. Here the
    instance *is* the node, so this collapses to an existence check and is kept only so the
    call sites forked from the sibling rig keep working.
    """
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        instance_name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        "--format=value(name)",
    ]
    rc, name, _ = await run_command(cmd)
    return name.strip() if rc == 0 and name.strip() else None


async def _get_node_ip(node_id: str, zone: str = ZONE) -> Optional[str]:
    """Gets the external or internal IP of a TPU instance."""
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        node_id,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        "--format=value(networkInterfaces[0].accessConfig[0].natIP)",
    ]
    rc, ip, _ = await run_command(cmd)
    if rc == 0 and ip.strip():
        return ip.strip()

    # Fallback to internal IP if external is not found
    cmd[-1] = "value(networkInterfaces[0].networkIP)"
    rc, ip, _ = await run_command(cmd)
    return ip.strip() if rc == 0 and ip.strip() else None


async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from Secret Manager."""
    rc, stdout, stderr = await run_command(
        ["gcloud", "secrets", "versions", "access", "latest", f"--secret={secret_id}"]
    )
    if rc == 0:
        return stdout.strip()
    logger.error(f"Failed to access secret {secret_id} via gcloud (exit code {rc}): {stderr}")
    return None


async def _get_formatted_startup_script(model_name: str, zone: str = ZONE) -> str:
    """Formats the startup script with necessary values.

    The HF token is deliberately NOT a placeholder: the rendered script is uploaded as
    instance metadata, so it fetches the secret at boot with the VM's own credentials.
    """
    template_path = os.path.join(os.path.dirname(__file__), "startup_script_template.sh")
    try:
        with open(template_path, "r") as f:
            template = f.read()
        return template.format(
            project_id=PROJECT_ID,
            zone=zone,
            model_name=model_name,
            hf_secret_id=HF_SECRET_ID,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
            model_gcs_uri=MODEL_GCS_URI,
        )
    except Exception as e:
        logger.error(f"Error formatting startup script: {e}")
        return f"""#!/bin/bash
echo 'Error loading template: {e}'"""


def _vllm_serve_flags(
    mm_limit: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    max_model_len: Optional[int] = None,
    kv_cache_dtype: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    extra_flags: str = "",
) -> str:
    """The vLLM serve flags for Gemma 4 on TPU, shared by every deployment path.

    mm_limit: the already-quoted --limit-mm-per-prompt value. Defaults to single quotes,
    which is wrong inside an outer single-quoted argument — pass the double-quoted,
    backslash-escaped form there instead.

    The remaining arguments are per-run overrides for a sweep, and every one of them
    defaults to None meaning "use the `tpu.env` value, or omit the flag entirely". With
    no overrides this renders exactly what it rendered before they existed, so the boot
    path is unchanged unless a caller opts in. This is still the ONE place serving flags
    are built — do not grow a second list somewhere for the sweep's benefit.

    - tensor_parallel_size: the crossing variable. 4 and 8 are the only degrees that boot
      this checkpoint (57.7 GiB of weights need >= 4 chips), so a sweep is 4 vs 8.
    - kv_cache_dtype: emitted only when passed. Zimbres 2026 s6.3 measured the *automatic*
      fp8 path on this checkpoint announcing fp8 in a startup banner and then allocating
      bf16 anyway, so passing it explicitly is the only route that acts — and the
      allocation line, not the banner, is what says whether it did.
    - gpu_memory_utilization: vLLM's default is 0.9. Emitted only when passed.
    """
    if mm_limit is None:
        mm_limit = f"'{LIMIT_MM_PER_PROMPT}'"
    flags = (
        f"--max-model-len {max_model_len if max_model_len is not None else MAX_MODEL_LEN} "
        f"--tensor-parallel-size "
        f"{tensor_parallel_size if tensor_parallel_size is not None else TENSOR_PARALLEL_SIZE} "
        f"--disable_chunked_mm_input "
        f"--max_num_batched_tokens {MAX_NUM_BATCHED_TOKENS} "
        f"--limit-mm-per-prompt {mm_limit} "
        f"--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4"
    )
    if kv_cache_dtype is not None:
        flags += f" --kv-cache-dtype {kv_cache_dtype}"
    if gpu_memory_utilization is not None:
        flags += f" --gpu-memory-utilization {gpu_memory_utilization}"
    if extra_flags:
        flags += f" {extra_flags.strip()}"
    return flags


def _provisioning_flags(
    provisioning_model: str,
    max_run_duration: Optional[str] = None,
    request_valid_for: Optional[str] = None,
    reservation_name: Optional[str] = None,
) -> list[str]:
    """The gcloud flags that select how a Compute Engine instance asks for capacity.

    This is the one place the two control planes diverge in vocabulary, and every
    difference here is a real difference rather than a spelling:

    - **Values are SCREAMING_CASE.** `queued-resources create` takes
      `--provisioning-model=flex-start`; `instances create` takes `FLEX_START`. Passing
      the TPU API's spelling is rejected client-side by gcloud's enum validation, which
      is the one failure mode on this path that does *not* cost a round trip.
    - **`--max-run-duration` is not flex-start-only here.** On the TPU API it is; on
      Compute Engine it is a general instance-scheduling flag, so spot and on-demand can
      have an automatic stop too. Pairing it with
      `--instance-termination-action=DELETE` is what makes a demo VM clean up after
      itself, which the Queued Resource path could not do outside flex-start.
    - **`--request-valid-for-duration` replaces `--valid-until-duration`** and is
      documented as the FLEX_START wait knob specifically — how long to keep asking, not
      how long to run.
    - **RESERVATION_BOUND has no Queued Resource equivalent.** It consumes a calendar or
      dense-deployment reservation for that reservation's whole duration. gcloud accepts the
      model only when the instance targets a *specific* reservation, and
      `--reservation-affinity` defaults to `any` — so the model alone is not a complete
      request and `reservation_name` is effectively required for it. Callers validate that;
      here a missing name just omits the affinity pair rather than emitting a half-formed
      `--reservation=`.

    Nothing here is verified by a successful creation — see this rig's CLAUDE.md.
    """
    run_for = max_run_duration or MAX_RUN_DURATION
    valid_for = request_valid_for or REQUEST_VALID_FOR
    stop_flags = [f"--max-run-duration={run_for}", "--instance-termination-action=DELETE"]

    # Every instance carries rig=<RIG_NAME>. Sibling rigs provision ct6e instances into this
    # same project and zone, and unlike a Queued Resource id an instance name is not forced
    # to encode its owner — so the label is what lets teardown tell ours from theirs.
    def labelled(purpose: str) -> str:
        return f"--labels=rig={RIG_NAME},purpose={purpose}"

    if provisioning_model == "spot":
        return ["--provisioning-model=SPOT", *stop_flags, labelled("spot")]
    if provisioning_model == "on-demand":
        return ["--provisioning-model=STANDARD", *stop_flags, labelled("on-demand")]
    if provisioning_model == "reservation-bound":
        # No stop flags: the instance runs for the reservation's whole duration by
        # definition, so a --max-run-duration would contradict the model.
        reservation = reservation_name or RESERVATION_NAME
        affinity = ["--reservation-affinity=specific", f"--reservation={reservation}"] if reservation else []
        return ["--provisioning-model=RESERVATION_BOUND", *affinity, labelled("reservation-bound")]
    return [
        "--provisioning-model=FLEX_START",
        f"--request-valid-for-duration={valid_for}",
        *stop_flags,
        labelled("flex-start"),
    ]


def _quota_id_for(provisioning_model: str) -> str:
    """Maps a provisioning model to a **Cloud TPU API** quota id (TPU_QUOTA_ID / TPU_SPOT_QUOTA_ID).

    TPU API only — used when reporting the sibling rig's pools. **Do not read this as the
    Compute Engine rule, which is different**: there, flex-start spends the *preemptible*
    quota alongside spot, and only on-demand draws on the family quota. See GCE_QUOTA_ID.
    """
    return TPU_SPOT_QUOTA_ID if provisioning_model == "spot" else TPU_QUOTA_ID


async def _list_queued_resources_json(zone: str) -> Optional[list]:
    """Lists the Queued Resources in a zone. Returns None if the gcloud call failed.

    Retained on the Compute Engine rig **only to detect cross-path collisions**: the
    twin tpu-vllm-v6e8-2b provisions QRs into this same project and zone, and a chip it
    holds is a chip this rig cannot get — eight of them, at this slice size. Nothing here
    creates or deletes one.
    """
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "list",
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--format=json",
    ]
    rc, stdout, stderr = await run_command(cmd)
    if rc != 0:
        logger.error(f"Failed to list queued resources in {zone}: {stderr}")
        return None
    try:
        return json.loads(stdout)
    except Exception:
        return []


# Shell prelude that makes a Docker-dependent remote command safe on this path.
#
# THE IMAGE SHIPS NO DOCKER. `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` has no `docker` on
# PATH at first boot, unlike the Cloud TPU API's runtime versions (v2-alpha-tpuv6e), which
# do. Verified 2026-08-10 on the first instance this rig ever created: the boot-time startup
# script died in its pull loop with `sudo: docker: command not found`, and every recovery
# tool below would have died the same way — which is exactly when they are reached for.
#
# The startup script installs Docker itself now, so this prelude is a no-op on a healthy
# boot. It exists for the case that matters: an instance whose startup script failed, where
# `manage_vllm_docker start` is the natural next move and must not fail for the same reason.
_ENSURE_DOCKER = (
    "if ! command -v docker > /dev/null 2>&1; then "
    "sudo apt-get update -qq && "
    "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io && "
    "sudo systemctl enable --now docker; fi"
)


def _ssh_command(instance_name: str, remote_cmd: str, zone: str = ZONE, ensure_docker: bool = False) -> list[str]:
    """Builds the `gcloud compute ssh` argv for a command on one of this rig's instances.

    **This is `compute ssh`, not `compute tpus tpu-vm ssh`.** A ct6e instance is an ordinary
    Compute Engine instance that happens to carry a TPU; it does not appear in the TPU API's
    node list at all, so the sibling rig's `tpu-vm ssh` cannot reach it and fails with a
    not-found on a VM that is plainly RUNNING. Four tools here shelled to `tpu-vm ssh` after
    the fork — `manage_vllm_docker`, `run_vllm_benchmark`, `get_vllm_docker_logs`,
    `get_tpu_system_logs` — none of which discovery covered, because the test that pins this
    rig off the TPU API only inspected the discovery path.

    No `--tunnel-through-iap`: these instances get an external IP and plain SSH reaches them.

    Set ensure_docker for any command that invokes `docker`; see `_ENSURE_DOCKER`.
    """
    if ensure_docker:
        remote_cmd = f"{_ENSURE_DOCKER}; {remote_cmd}"
    return [
        "gcloud",
        "compute",
        "ssh",
        instance_name,
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--command",
        remote_cmd,
    ]


async def _create_tpu_instance(
    instance_name: str,
    zone: str,
    provisioning_model: str = PROVISIONING_MODEL,
    machine_type: str = MACHINE_TYPE,
    reservation_name: Optional[str] = None,
) -> tuple[bool, str]:
    """Renders the startup script and issues the `compute instances create` call.

    The rendered script carries no secret — it reads 'hf-token' from Secret Manager at
    boot — but it is still written to a private temp file and removed afterwards. That
    property is unchanged from the Queued Resource path and matters more here, because the
    script is uploaded as instance metadata either way.

    Two flags are load-bearing and easy to drop:

    - **`--scopes=cloud-platform`** — without it the booted VM cannot reach Secret Manager
      and the startup script spins for its full 30-minute retry before giving up. The QR
      path got a workable default scope set; `instances create` does not.
    - **`--maintenance-policy=TERMINATE`** — TPU instances cannot live-migrate.
    """
    if provisioning_model not in PROVISIONING_MODELS:
        return (
            False,
            f"❌ Aborted: unknown provisioning_model '{provisioning_model}'. Use one of {PROVISIONING_MODELS}.",
        )

    # Caught here rather than at the API: RESERVATION_BOUND without a specific reservation is
    # rejected server-side after the startup script has already been rendered and uploaded,
    # and the message you get back does not name the missing flag.
    if provisioning_model == "reservation-bound" and not (reservation_name or RESERVATION_NAME):
        return (
            False,
            "❌ Aborted: 'reservation-bound' must name the reservation it consumes. Pass "
            "reservation_name=, or set RESERVATION_NAME in tpu.env. Create one first with "
            "`gcloud compute future-reservations create <name> --reservation-mode=CALENDAR "
            "--tpu-version=V6E --chip-count=1 --workload-type=SERVING --planning-status=SUBMITTED "
            "--require-specific-reservation`, and note it is pinned to the zone it was reserved in.",
        )

    # Pre-flight only: confirm the secret exists before spending scarce Flex-start
    # capacity on a VM that would spin for 30 minutes and then fail. The value is
    # checked for presence and discarded — it is never written into the script.
    if not await get_secret():
        return False, f"❌ Aborted: '{HF_SECRET_ID}' secret missing or unreadable."

    script_content = await _get_formatted_startup_script(MODEL_NAME, zone=zone)
    fd, script_file = tempfile.mkstemp(prefix="vllm-startup-", suffix=".sh")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script_content)

        create_cmd = [
            "gcloud",
            "compute",
            "instances",
            "create",
            instance_name,
            f"--project={PROJECT_ID}",
            f"--zone={zone}",
            f"--machine-type={machine_type}",
            f"--image-family={IMAGE_FAMILY}",
            f"--image-project={IMAGE_PROJECT}",
            "--maintenance-policy=TERMINATE",
            f"--boot-disk-size={BOOT_DISK_SIZE_GB}GB",
            "--scopes=cloud-platform",
            f"--metadata-from-file=startup-script={script_file}",
            *_provisioning_flags(provisioning_model, reservation_name=reservation_name),
        ]
        if TPU_NETWORK:
            create_cmd.append(f"--network={TPU_NETWORK}")
        if TPU_SUBNETWORK:
            create_cmd.append(f"--subnetwork={TPU_SUBNETWORK}")

        logger.info(f"Executing gcloud command: {' '.join(shlex.quote(c) for c in create_cmd)}")
        # A flex-start create blocks until capacity is granted or the request expires, so
        # the default 60s timeout is far too short. Cap below the client's patience and
        # treat a timeout as "still pending", never as a failure.
        rc, _, err = await run_command(create_cmd, timeout=590)
    finally:
        try:
            os.unlink(script_file)
        except OSError:
            pass

    if rc != 0:
        if err.startswith("Timeout after"):
            return False, (
                f"⏳ gcloud gave up waiting, but the {provisioning_model} request for `{instance_name}` may "
                f"still be pending server-side and can still produce a billing VM. Check "
                f"`list_tpu_instances` before retrying, and destroy it if you no longer want it."
            )
        hint = ""
        if "TPUS_PER_TPU_FAMILY" in err or "tpus_per_tpu_family" in err:
            hint = (
                f" (this is the Compute Engine quota `{GCE_QUOTA_ID}` for family {GCE_TPU_FAMILY} in "
                f"{REGION} — a *different pool* from the TPU API quota the twin rig uses. "
                "Request it on the Compute Engine quota page, not the TPU API one. Note this "
                "family quota governs on-demand only; flex-start and spot spend "
                f"`{GCE_SPOT_QUOTA_ID}` instead. Quota is counted in CHIPS and `{machine_type}` "
                f"spends `{CHIP_COUNT}` of them per instance, so a limit that looked ample for "
                "the single-chip rigs divides by that here.)"
            )
        return False, f"❌ Creation failed: {err}{hint}"

    lifetime = {
        "flex-start": f"Self-terminates and deletes after {MAX_RUN_DURATION}.",
        "spot": f"⚠️ Preemptible with ~30s notice; also set to delete at {MAX_RUN_DURATION}.",
        "on-demand": f"⚠️ Bills at the full rate; set to delete at {MAX_RUN_DURATION}.",
        "reservation-bound": (
            f"⚠️ Bound to reservation `{reservation_name or RESERVATION_NAME}`; runs for that "
            "reservation's whole duration — no automatic stop."
        ),
    }[provisioning_model]
    return True, (
        f"🚀 Instance {instance_name} ({machine_type}) creation initiated in {zone} "
        f"({provisioning_model}) with startup script. {lifetime}"
    )


async def _list_tpu_vm_nodes(zone: str = ZONE) -> list[dict]:
    """Lists every TPU-bearing Compute Engine instance in the zone.

    The sibling rig listed **TPU VM nodes** here, because a Queued Resource and a
    hand-provisioned tpu-vm both land in that namespace. Instances created with a `ct6e-*`
    machine type do **not** appear in `gcloud compute tpus tpu-vm list` at all — they are
    ordinary Compute Engine instances that happen to carry a TPU — so discovery has to look
    somewhere else entirely. That is the sharpest operational difference between the two
    paths, and it is why the two rigs cannot share a discovery helper.
    """
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={PROJECT_ID}",
        f"--zones={zone}",
        "--filter=machineType~'ct6e|ct5p'",
        "--format=json",
    ]
    rc, stdout, _ = await run_command(cmd)
    if rc != 0 or not stdout:
        return []
    try:
        nodes = json.loads(stdout)
    except json.JSONDecodeError as e:
        logger.error(f"Could not parse instances list: {e}")
        return []
    return nodes if isinstance(nodes, list) else []


def _node_ip(node: dict) -> Optional[str]:
    """External IP of a listed instance, falling back to the internal one."""
    for iface in node.get("networkInterfaces") or []:
        for access in iface.get("accessConfigs") or []:
            if access.get("natIP"):
                return str(access["natIP"])
        if iface.get("networkIP"):
            return str(iface["networkIP"])
    return None


async def _active_qr_node_ids(zone: str = ZONE) -> set[str]:
    """Node ids belonging to an ACTIVE Queued Resource in the zone.

    Nothing this rig creates appears here — a `ct6e-*` instance has no Queued Resource
    behind it. It is kept so discovery can *exclude* the sibling rig's nodes and so
    `get_system_status` can say plainly that the chip in this zone is held by the other
    control plane. Never treat a hit here as one of ours.
    """
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "list",
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        "--format=json",
    ]
    rc, stdout, _ = await run_command(cmd)
    if rc != 0 or not stdout:
        return set()
    try:
        resources = json.loads(stdout)
    except json.JSONDecodeError:
        return set()
    node_ids = set()
    for res in resources:
        if res.get("state", {}).get("state") != "ACTIVE":
            continue
        for spec in (res.get("tpu") or {}).get("nodeSpec") or []:
            if spec.get("nodeId"):
                node_ids.add(spec["nodeId"])
    return node_ids


async def _probe_vllm(url: str, timeout: float = 5.0) -> Optional[str]:
    """Returns the served model id if a vLLM OpenAI server answers at url, else None."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.get(f"{url}/v1/models")
            if res.status_code != 200:
                return None
            data = res.json().get("data") or []
    except Exception:
        return None
    return str(data[0].get("id", "unknown")) if data else "unknown"


def _is_this_rig(node_name: str) -> bool:
    """Whether an instance name is one this rig provisions.

    The `<name>-` prefix match is kept from the sibling rig even though this path never
    derives a `-node` suffix, because it still catches sweep-created instances named
    `<INSTANCE_NAME>-<zone>` by find_tpu.
    """
    return node_name == INSTANCE_NAME or node_name.startswith(f"{INSTANCE_NAME}-")


class VllmNode(NamedTuple):
    """A TPU node discovery picked out, and whether vLLM answered on it."""

    name: str
    url: str
    serving: bool


class _Candidate(NamedTuple):
    sort_key: tuple
    name: str
    url: str
    mine: bool


async def _discover_vllm_node() -> Optional[VllmNode]:
    """Finds the Compute Engine TPU instance serving vLLM in ZONE.

    Ranks this rig's own instances first, then RUNNING ones, then probes each on
    /v1/models; the first that answers wins. If none answers, an instance of ours that is
    still booting is returned anyway (serving=False) so callers can poll it — but an
    instance that is not ours is never returned unprobed, since sibling rigs share this
    zone and one of theirs is not ours to talk to.

    The state string differs from the sibling rig: a Compute Engine instance is `RUNNING`,
    a TPU VM node is `READY`. Copying the sibling's `READY` check here would demote every
    healthy instance to the bottom of the ranking rather than erroring, so it would only
    show up as discovery picking the wrong node when two are up.
    """
    nodes = await _list_tpu_vm_nodes()
    if not nodes:
        return None
    qr_node_ids = await _active_qr_node_ids()

    candidates: list[_Candidate] = []
    for node in nodes:
        name = (node.get("name") or "").split("/")[-1]
        ip = _node_ip(node)
        if not name or not ip:
            continue
        mine = _is_this_rig(name)
        sort_key = (
            0 if mine else 1,
            0 if node.get("status") == "RUNNING" else 1,
            name,
        )
        candidates.append(_Candidate(sort_key, name, f"http://{ip}:8000", mine))
    candidates.sort(key=lambda c: c.sort_key)

    for cand in candidates:
        model_id = await _probe_vllm(cand.url)
        if model_id:
            # A name that is also an ACTIVE QR node belongs to the sibling TPU-API rig.
            origin = "⚠️ sibling rig's queued resource" if cand.name in qr_node_ids else "GCE TPU instance"
            logger.info(f"📡 Found vLLM serving {model_id} on {cand.name} ({origin}) at {cand.url}")
            return VllmNode(cand.name, cand.url, True)

    for cand in candidates:
        if cand.mine:
            logger.info(f"📡 Instance {cand.name} is up at {cand.url} but vLLM is not answering yet.")
            return VllmNode(cand.name, cand.url, False)

    logger.info(f"No Compute Engine TPU instance in {ZONE} is serving vLLM.")
    return None


async def discover_vllm_url() -> Optional[str]:
    """Finds the URL of a running vLLM service in ZONE."""
    node = await _discover_vllm_node()
    return node.url if node else None


async def _resolve_node_id(resource_id: str, zone: str = ZONE) -> Optional[str]:
    """Resolves an id the tools were given to an actual instance name.

    Tried in order: an instance by that exact name; then the instance currently serving
    vLLM, so the default id still reaches a running deployment named by hand.

    The sibling rig needed a third step here — describe the Queued Resource, read its
    derived `<id>-node` — because the id you asked for was never the name you got. That
    whole class of mismatch does not exist on this path, so this function is two lookups
    instead of three and cannot return a name the caller did not effectively ask for.
    """
    if await _get_node_id(resource_id, zone):
        return resource_id

    serving = await _discover_vllm_node()
    if serving and serving.serving:
        logger.info(f"No instance named {resource_id}; falling back to {serving.name}, which is serving vLLM.")
        return serving.name
    return None


async def get_vllm_client() -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for the vLLM service."""
    url = await discover_vllm_url()
    if not url:
        raise Exception(f"No Compute Engine TPU instance in {ZONE} is serving vLLM.")
    return AsyncOpenAI(base_url=f"{url}/v1", api_key="not-needed")


@mcp.tool()
async def verify_model_health() -> str:
    """Runs a deep logic check with latency reporting."""
    try:
        client = await get_vllm_client()
        start_time = time.monotonic()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, is the model working?"}],
            model=MODEL_NAME,
            max_tokens=10,
        )
        end_time = time.monotonic()
        latency = end_time - start_time
        response_content = chat_completion.choices[0].message.content

        if response_content:
            return (
                f"✅ Model health check PASSED.\\n"
                f"Response: '{response_content[:50]}...\\n'"
                f"Latency: {latency:.2f} seconds."
            )
        else:
            return "❌ Model health check FAILED: Empty response."
    except Exception as e:
        return f"❌ Model health check FAILED: {e}"


@mcp.tool()
async def save_hf_token(token: str) -> str:
    """Securely saves a Hugging Face API token to GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    secret_parent = f"projects/{PROJECT_ID}/secrets/{HF_SECRET_ID}"

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
    return f"✅ Token saved. Version: {response.name}"


@mcp.tool()
async def get_vllm_deployment_config(service_name: str = INSTANCE_NAME, model_name: str = MODEL_NAME) -> str:
    """Generates the gcloud command for an eight-chip v6e vLLM deployment on Compute Engine.

    This is the copy-pasteable form of what `create_tpu_instance` does, so it carries the
    same Docker install: the image family below ships **no** `docker` binary, and without
    the install line the inline startup script fails on its first command while the
    instance settles into a healthy-looking RUNNING. The full boot path uses
    `startup_script_template.sh` instead, which has the retry loops this one-liner omits.
    """
    # The token is read on the VM at runtime — never interpolated into the returned text.
    # The whole startup script is one single-quoted argument, so the JSON value has to be
    # double-quoted and backslash-escaped rather than single-quoted.
    escaped_mm = '"' + LIMIT_MM_PER_PROMPT.replace('"', '\\"') + '"'
    cmd = (
        f"gcloud compute instances create {service_name} \\\n"
        f"  --machine-type={MACHINE_TYPE} \\\n"
        f"  --image-family={IMAGE_FAMILY} \\\n"
        f"  --image-project={IMAGE_PROJECT} \\\n"
        f"  --maintenance-policy=TERMINATE \\\n"
        f"  --boot-disk-size={BOOT_DISK_SIZE_GB}GB \\\n"
        f"  --scopes=cloud-platform \\\n"
        f"  {' '.join(_provisioning_flags(PROVISIONING_MODEL))} \\\n"
        f"  --zone={ZONE} \\\n"
        f"  --project={PROJECT_ID} \\\n"
        f"  --metadata=startup-script='#!/bin/bash\\n"
        f"{_ENSURE_DOCKER}\\n"
        f"docker run -t --rm --name vllm-gemma4 --privileged --net=host "
        f"-v /dev/shm:/dev/shm --shm-size 10gb "
        f"-e HF_HOME=/dev/shm "
        f"-e HF_TOKEN=$(gcloud secrets versions access latest --secret={HF_SECRET_ID}) "
        f"{VLLM_IMAGE} vllm serve {model_name} {_vllm_serve_flags(mm_limit=escaped_mm)}'"
    )
    return cmd


@mcp.tool()
async def get_vllm_tpu_deployment_config() -> str:
    """Generates GKE manifests for TPU-based deployments.

    The `google.com/tpu` limit is a **chip** request, so it tracks `CHIP_COUNT` (8 here), not
    `TENSOR_PARALLEL_SIZE`. The two happen to be equal on this rig and are not the same
    quantity: TP is a sharding choice, the limit is how much hardware the pod is handed. A
    multi-chip node pool also needs the topology selectors below — a bare limit of 8 on a
    1-chip node pool is unschedulable, which is a much clearer failure than the silent
    mis-sizing that dropping them causes.
    """
    manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-gemma4-tpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-gemma4-tpu
  template:
    metadata:
      labels:
        app: vllm-gemma4-tpu
    spec:
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
        cloud.google.com/gke-tpu-topology: "{TOPOLOGY}"
      containers:
      - name: vllm-container
        image: vllm/vllm-tpu:nightly
        resources:
          limits:
            google.com/tpu: "{CHIP_COUNT}"
        env:
        - name: MODEL_NAME
          value: {MODEL_NAME}
        - name: TENSOR_PARALLEL_SIZE
          value: "{TENSOR_PARALLEL_SIZE}"
"""
    return manifest


# --- MCP Tools ---


@mcp.tool()
async def destroy_tpu_instance(instance_name: str, zone: str = ZONE) -> str:
    """Deletes a Compute Engine TPU instance.

    Simpler than the Queued Resource teardown it replaces, and the difference is structural
    rather than cosmetic. A QR delete needed `--force` because an ACTIVE resource owns a
    node the API refuses to drop out from under it, so the two-object lifecycle leaked into
    the teardown call. An instance is one object: delete it and it is gone.

    Still asynchronous, and a TPU instance takes a while to release its chip — poll
    `list_tpu_instances` rather than assuming the zone is clear on return.
    """
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "delete",
        instance_name,
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--quiet",
    ]
    rc, stdout, stderr = await run_command(cmd, timeout=300)
    if rc != 0:
        return f"❌ Failed to delete instance {instance_name}: {stderr}"
    return f"🗑️ Deletion of {instance_name} initiated: {stdout}"


@mcp.tool()
async def manage_tpu_instance(
    instance_name: str = INSTANCE_NAME, zone: str = ZONE, provisioning_model: str = PROVISIONING_MODEL
) -> str:
    """DESTRUCTIVE. Ensures the primary instance exists and deletes this rig's *other* TPU instances in the zone.

    **Deliberately narrower than the sibling rig's `manage_queued_resource`, which deletes
    every Queued Resource in the zone that is not the named primary.** That was tolerable
    when a QR id encoded its owning rig by convention; an instance name does not, and three
    rigs now provision `ct6e-*` instances into this project. So this only deletes instances
    carrying `rig=<this rig>` — anything unlabelled or belonging to a sibling is reported
    and left alone. Do not "fix" this to match the sibling.

    provisioning_model is only consulted if the primary has to be created. An existing
    primary keeps whatever model it was created with — this tool does not convert one.
    """
    instances = await _list_tpu_vm_nodes(zone)

    deleted: list[str] = []
    skipped: list[str] = []
    primary = None

    for inst in instances:
        name = (inst.get("name") or "").split("/")[-1]
        status = inst.get("status", "UNKNOWN")
        owner = (inst.get("labels") or {}).get("rig")

        if name == instance_name:
            if status in ("TERMINATED", "SUSPENDED"):
                logger.info(f"Primary instance {name} is {status}. Deleting to recreate.")
                await destroy_tpu_instance(name, zone=zone)
                deleted.append(f"{name} ({status})")
            else:
                primary = inst
        elif owner == RIG_NAME:
            logger.info(f"Deleting redundant instance owned by this rig: {name}")
            await destroy_tpu_instance(name, zone=zone)
            deleted.append(name)
        else:
            skipped.append(f"{name} (rig={owner or 'unlabelled'})")

    note = f" Left alone: {skipped}." if skipped else ""
    if not primary:
        _, msg = await _create_tpu_instance(instance_name, zone, provisioning_model)
        return f"{msg} Cleaned up: {deleted}.{note}"

    return f"✅ Primary instance {instance_name} is {primary.get('status')}. Cleaned up: {deleted}.{note}"


@mcp.tool()
async def create_tpu_instance(
    instance_name: str = INSTANCE_NAME,
    zone: str = ZONE,
    provisioning_model: str = PROVISIONING_MODEL,
    machine_type: str = MACHINE_TYPE,
    reservation_name: str = "",
) -> str:
    """Creates a TPU instance on Compute Engine in the given zone. Non-destructive.

    Other instances in the zone are left alone. If an instance with this exact name already
    exists it is reported and nothing is created; only a TERMINATED one under that same
    name is deleted so it can be recreated.

    provisioning_model is one of:
      * 'flex-start' (default) — `FLEX_START`, queues for scarce capacity via Dynamic
        Workload Scheduler, then self-deletes at max-run-duration.
      * 'spot' — `SPOT`, reclaimable with ~30s notice. Metered by
        `PREEMPTIBLE-TPU-V6E-per-project-region`, the same pool flex-start spends. In
        us-east5 v6e spot lists *dearer* than flex-start; read `estimate_deployment_cost`
        rather than assuming.
      * 'on-demand' — `STANDARD`, full price, no preemption.
      * 'reservation-bound' — `RESERVATION_BOUND`, consumes a calendar or dense-deployment
        reservation. **No Queued Resource equivalent** — this model exists only here. It
        needs `reservation_name` (or `RESERVATION_NAME` in tpu.env): gcloud accepts the model
        only alongside `--reservation-affinity=specific`, and a calendar reservation is
        pinned to the zone it was reserved in, so `zone` has to match it.

    reservation_name is read only for 'reservation-bound' and ignored by every other model.

    Note this blocks for up to ~10 minutes on flex-start while gcloud waits for capacity.
    """
    existing = {(i.get("name") or "").split("/")[-1]: i for i in await _list_tpu_vm_nodes(zone)}

    if instance_name in existing:
        status = existing[instance_name].get("status", "UNKNOWN")
        if status != "TERMINATED":
            return f"✅ Instance {instance_name} already exists in {zone} and is {status}. Nothing created."
        logger.info(f"Instance {instance_name} is {status}. Deleting it so it can be recreated.")
        await destroy_tpu_instance(instance_name, zone=zone)
        _, msg = await _create_tpu_instance(
            instance_name, zone, provisioning_model, machine_type, reservation_name or None
        )
        return f"{msg} (replaced the previous {status} instance of the same name)"

    _, msg = await _create_tpu_instance(instance_name, zone, provisioning_model, machine_type, reservation_name or None)
    return msg


async def _get_zones_with_available_quota_list(
    service: str = "tpu.googleapis.com",
    quota_id: str = TPU_QUOTA_ID,
) -> list[str]:
    """Helper to retrieve a list of GCP zones that have a non-zero quota for a specific metric."""
    cmd = [
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
    rc, stdout, stderr = await run_command(cmd)
    if rc != 0:
        logger.error(f"Failed to retrieve quota info: {stderr}")
        return []
    try:
        quota_data = json.loads(stdout)
    except Exception:
        return []

    zones = []
    for info in quota_data:
        dimensions_infos = info.get("dimensionsInfos", [])
        for dim_info in dimensions_infos:
            details = dim_info.get("details", {})
            limit_val = details.get("value")
            if limit_val and limit_val != "0":
                dim_map = dim_info.get("dimensions", {})
                zone_val = dim_map.get("zone") or dim_map.get("region")
                if zone_val:
                    zones.append(zone_val)
                else:
                    locations = dim_info.get("applicableLocations", [])
                    for loc in locations:
                        zones.append(loc)
    return sorted(list(set(zones)))


@mcp.tool()
async def get_zones_with_available_quota(
    service: str = "compute.googleapis.com",
    quota_id: Optional[str] = None,
    provisioning_model: str = PROVISIONING_MODEL,
) -> str:
    """
    Retrieves the zones/regions with a non-zero Compute Engine TPU quota.

    **The service defaults to `compute.googleapis.com`, not `tpu.googleapis.com`** — that is
    the whole point of this rig. The two control planes meter against different pools and
    holding one buys you nothing on the other. Verified 2026-08-10: this project held 512
    v6e chips in us-east5 under `TPUV6EPerProjectPerZoneForTPUAPI` and nothing at all for
    family CT6E under the Compute Engine quota this path consumes. (us-east5 CT6E has since
    been granted 32; the point about the pools being disjoint stands.)

    **The two Compute Engine ids split by provisioning model, not by preemptibility of
    behaviour.** flex-start and spot spend `GCE_SPOT_QUOTA_ID`; only on-demand goes straight
    to the *regional, family-wide* `GCE_QUOTA_ID` dimensioned by (region, tpu_family).

    **For flex-start the family quota is a documented fallback**, so this tool's answer is a
    primary-pool answer, not a verdict: "When you create a Flex-start VM, preemptible quota
    is consumed. If your project lacks preemptible quota, then standard quota is consumed."
    A flex-start zone is usable if EITHER pool has room, so check both before concluding.

    Their defaults are opposite — a region absent from the family quota inherits 0, one
    absent from the preemptible quota inherits 1536 — so reading only one listing writes off
    regions that are in fact usable.

    Non-zero quota still does NOT mean the zone will accept the request — it is a ceiling,
    not an offer of capacity. See this rig's CLAUDE.md on the three gates.

    Args:
        service: The GCP service to query. Pass 'tpu.googleapis.com' to inspect the
            sibling rig's pool for comparison.
        quota_id: The specific quota ID to filter by. Defaults to the one matching
            provisioning_model and service.
        provisioning_model: 'flex-start' (default), 'spot', 'on-demand', or
            'reservation-bound'.
    """
    if quota_id is None:
        if service.startswith("tpu."):
            quota_id = _quota_id_for(provisioning_model)
        else:
            # flex-start and spot both spend the PREEMPTIBLE pool first; only on-demand
            # goes straight to the family quota. Reported id is the PRIMARY one — for
            # flex-start the family quota is a documented fallback, so a zone showing zero
            # here may still be usable. The caller is told so below.
            quota_id = GCE_QUOTA_ID if provisioning_model == "on-demand" else GCE_SPOT_QUOTA_ID
    zones = await _get_zones_with_available_quota_list(service, quota_id)
    if not zones:
        return (
            f"No zones/locations found with a non-zero quota limit for `{quota_id}` on `{service}`.\n\n"
            "⚠️ For the Compute Engine ids this is the expected result in regions where the quota has "
            "never been requested — an unset family quota reads the same as a zero one here. It does "
            "**not** mean the hardware is absent; check `machine-types list` for that.\n\n"
            f"⚠️ For **flex-start** this is not a verdict either: `{GCE_SPOT_QUOTA_ID}` is only the "
            f"pool it draws on *first*, and `{GCE_QUOTA_ID}` is the documented fallback. Check that "
            "one too before concluding a region is unusable."
        )

    output = [f"### 📊 Zones with quota for `{quota_id}` (`{service}`)\n"]
    for zone in zones:
        output.append(f"- Zone/Region `{zone}`")
    return "\n".join(output)


def _status_model(detail: str) -> str:
    """Reads the provisioning model back out of a tpu_zones_status.md detail cell.

    Rows written before the models were configurable carry no tag; they all recorded
    flex-start attempts, so that is the right reading for an untagged row.
    """
    detail = detail.strip()
    for model in PROVISIONING_MODELS:
        if detail.startswith(f"[{model}]"):
            return model
    return "flex-start"


async def _update_status_file(zone: str, success_str: str, detail_str: str) -> None:
    status_file = os.path.join(os.path.dirname(__file__), "tpu_zones_status.md")
    if not os.path.exists(status_file):
        return
    try:
        with open(status_file, "r") as f:
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

        with open(status_file, "w") as f:
            f.write("\n".join(new_lines) + "\n")
    except Exception as e:
        logger.error(f"Error updating status file: {e}")


async def _zones_with_machine_type(machine_type: str = MACHINE_TYPE) -> list[str]:
    """Zones where the TPU machine type is published.

    This is the Compute Engine analogue of the TPU API's `accelerator-types list`, and it is
    the right gate-1 signal for this path. For v6e the two agree exactly (18 zones each,
    2026-08-10); for v5p they disagree in one zone. See ../HARDWARE.md.
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
    rc, stdout, stderr = await run_command(cmd, timeout=120)
    if rc != 0:
        logger.error(f"Failed to list zones for {machine_type}: {stderr}")
        return []
    return sorted({z.strip() for z in stdout.splitlines() if z.strip()})


@mcp.tool()
async def find_tpu(
    resource_id: str = INSTANCE_NAME,
    service: str = "compute.googleapis.com",
    quota_id: Optional[str] = None,
    provisioning_model: str = PROVISIONING_MODEL,
    machine_type: str = MACHINE_TYPE,
) -> str:
    """
    Finds a zone offering the TPU machine type and attempts to create an instance there until successful.

    **The zone source differs from the sibling rig, because it has to.** That rig sweeps
    zones with non-zero TPU API quota, which is zonal. Both Compute Engine TPU quotas are
    *regional*, so neither can produce a zone list. So the sweep is driven by where the
    machine type is published, which is the stronger gate-1 signal anyway.

    provisioning_model is one of 'flex-start' (default), 'spot', 'on-demand', or
    'reservation-bound'. The sweep is non-destructive: it only touches the named resource_id.
    """
    if provisioning_model not in PROVISIONING_MODELS:
        return f"❌ Aborted: unknown provisioning_model '{provisioning_model}'. Use one of {PROVISIONING_MODELS}."

    zones = await _zones_with_machine_type(machine_type)
    if not zones:
        return f"❌ Aborted: no zone publishes machine type `{machine_type}`."
    logger.info(f"Zones offering {machine_type}: {zones}")

    # Parse flat status file to skip zones where TPU could not be started. A failure is only
    # evidence about the provisioning model that produced it — a zone can refuse flex-start
    # and still serve spot fine — so only same-model failures are skipped.
    skipped_zones = set()
    status_file = os.path.join(os.path.dirname(__file__), "tpu_zones_status.md")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                content = f.read()

            for line in content.splitlines():
                # Matches lines like: | **zone-name** | Yes | No | [spot] reason...
                match = re.search(r"\|\s*\*\*([a-zA-Z0-9-]+)\*\*\s*\|\s*([^|]+)\|\s*No\s*\|([^|]*)\|", line)
                if match and _status_model(match.group(3)) == provisioning_model:
                    zone_name = match.group(1).strip()
                    skipped_zones.add(zone_name)
            logger.info(f"Skipping {provisioning_model} zones (marked as failed in status file): {list(skipped_zones)}")
        except Exception as e:
            logger.error(f"Error parsing status file: {e}")

    logger.info(f"Zones to attempt: {zones}")

    attempts = []
    for zone in zones:
        if zone in skipped_zones:
            logger.info(f"Skipping zone {zone} as it is marked as failed in status file.")
            attempts.append(f"- **Zone {zone}**: ⏭️ Skipped (previously failed according to status file)")
            continue

        logger.info(f"Attempting to create {provisioning_model} instance {resource_id} in zone {zone}...")
        result = await create_tpu_instance(instance_name=resource_id, zone=zone, provisioning_model=provisioning_model)

        if result.startswith("❌"):
            attempts.append(f"- **Zone {zone}**: {result}")
            reason = result.replace("❌ Creation failed:", "").strip()
            await _update_status_file(zone, "No", f"[{provisioning_model}] {reason}")
            continue
        if result.startswith("⏳"):
            # Timed out client-side with the request possibly still live. Recording this as
            # a zone failure would be wrong twice over: it is a capacity outcome (gate 3),
            # which is never cached, and the instance may yet appear and bill.
            attempts.append(f"- **Zone {zone}**: {result}")
            continue

        # No separate "wait for ACTIVE" phase. A Queued Resource is a request object that
        # transitions ACCEPTED -> PROVISIONING -> ACTIVE and hands you a node at the end, so
        # the sibling rig had to poll a second resource to learn whether it got hardware.
        # `instances create` returns only once the instance exists, so a zero exit already
        # means capacity was granted; all that is left is to watch it boot.
        success = False
        poll_start = time.time()
        while time.time() - poll_start < 300:
            state_cmd = [
                "gcloud",
                "compute",
                "instances",
                "describe",
                resource_id,
                f"--zone={zone}",
                f"--project={PROJECT_ID}",
                "--format=value(status)",
            ]
            rc_s, stdout_s, stderr_s = await run_command(state_cmd)
            if rc_s == 0:
                current_state = stdout_s.strip()
                logger.info(f"Instance {resource_id} status in {zone}: {current_state}")
                if current_state == "RUNNING":
                    success = True
                    break
                if current_state in ("TERMINATED", "SUSPENDED"):
                    logger.info(f"Instance {resource_id} reached terminal state: {current_state}")
                    break
            else:
                logger.warning(f"Failed to check status: {stderr_s or stdout_s}")
            await asyncio.sleep(15)

        if success:
            await _update_status_file(zone, "Yes", f"[{provisioning_model}] Instance created and RUNNING.")
            attempts.append(f"- **Zone {zone}**: ✅ Successfully created and RUNNING.")

            # Dynamically update global ZONE variable
            global ZONE
            ZONE = zone

            return (
                f"✅ Successfully secured a TPU instance in zone `{zone}`!\n\n"
                f"**Creation Output:**\n{result}\n\n"
                f"**Attempts Log:**\n" + "\n".join(attempts)
            )
        else:
            logger.info(f"Instance in {zone} never reached RUNNING. Deleting it...")
            await destroy_tpu_instance(resource_id, zone=zone)
            timeout_msg = "Created but never reached RUNNING within 5 minutes."
            await _update_status_file(zone, "No", f"[{provisioning_model}] {timeout_msg}")
            attempts.append(f"- **Zone {zone}**: ❌ {timeout_msg}")

    return "❌ Failed to start a TPU instance in any zone. Attempted zones:\n" + "\n".join(attempts)


@mcp.tool()
async def manage_vllm_docker(
    resource_id: str = RESOURCE_ID,
    action: str = "start",
    tensor_parallel_size: Optional[int] = None,
    max_model_len: Optional[int] = None,
    kv_cache_dtype: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    extra_flags: str = "",
) -> str:
    """Manages the vLLM Docker container on the TPU VM.

    The override arguments re-serve the SAME instance under different flags, which is
    what makes a sweep affordable: the 62 GB checkpoint lives in the host's /dev/shm and
    is bind-mounted in, so recreating the container reloads from RAM instead of pulling
    from Hugging Face again. One instance, one pull, N configurations.

    **Passing any override forces a recreate, and that is not a convenience.** `docker
    start` on an existing container replays the arguments it was BUILT with and silently
    ignores new ones — so without this, a TP=4 sweep step against a container built at
    TP=8 would report success, serve at 8, and produce a fully plausible wrong number.
    That is exactly the advertised-versus-implemented gap this rig keeps documenting, so
    the overrides remove the container first and rebuild it.

    Verify the result at the allocation line in `log`, never at this tool's return value.
    """
    node_id = await _resolve_node_id(resource_id)
    if not node_id:
        return f"❌ Could not find an instance named {resource_id} in {ZONE}."

    overrides = {
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "kv_cache_dtype": kv_cache_dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "extra_flags": extra_flags or None,
    }
    active = {k: v for k, v in overrides.items() if v is not None}

    # Same image and serve flags the boot-time startup script uses, so a container
    # recreated here matches what the instance originally booted with.
    serve_flags = _vllm_serve_flags(
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        extra_flags=extra_flags,
    )
    docker_run_cmd = (
        f"sudo docker run --name vllm-gemma4 --privileged --net=host -d "
        f"-v /dev/shm:/dev/shm --shm-size 10gb "
        f"-e HF_HOME=/dev/shm -e HF_TOKEN=$(gcloud secrets versions access latest --secret={HF_SECRET_ID}) "
        f"{VLLM_IMAGE} vllm serve {MODEL_NAME} {serve_flags}"
    )
    # `docker start` would reuse the old argv, so an override has to rebuild.
    recreate_cmd = f"sudo docker rm -f vllm-gemma4 > /dev/null 2>&1 || true; {docker_run_cmd}"

    commands = {
        "start": recreate_cmd if active else f"sudo docker start vllm-gemma4 || {docker_run_cmd}",
        "stop": "sudo docker stop vllm-gemma4",
        "restart": recreate_cmd if active else "sudo docker restart vllm-gemma4",
        "status": "sudo docker ps -a --filter name=vllm-gemma4",
        "log": "sudo docker logs --tail 100 vllm-gemma4",
        "rm": "sudo docker rm -f vllm-gemma4",
    }

    if active and action not in ("start", "restart"):
        return (
            f"❌ Overrides ({', '.join(sorted(active))}) only apply to `start` or `restart`, "
            f"not `{action}` — they change how the container is built. Re-run with action='restart'."
        )

    ssh_cmd = _ssh_command(node_id, commands.get(action, commands["status"]), ensure_docker=True)

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Docker {action} failed, but instance {resource_id} remains safe.
Error: {err}"""
    note = ""
    if active:
        applied = ", ".join(f"{k}={v}" for k, v in sorted(active.items()))
        note = (
            f"\n🔁 Container REBUILT with overrides: {applied}\n"
            f"⏳ It is reloading the model from /dev/shm and recompiling — not serving yet.\n"
            f"📡 Confirm the override actually took effect in the ALLOCATION line "
            f"(`action='log'`), not here."
        )
    return f"""✅ Docker {action} command executed on {node_id}.{note}
{out}"""


@mcp.tool()
async def list_tpu_instances(zone: str = ZONE) -> str:
    """Lists all TPU-bearing Compute Engine instances in a specific zone."""
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--zones={zone}",
        f"--project={PROJECT_ID}",
        "--filter=machineType~'ct6e|ct5p'",
        "--format=table(name, status, machineType.basename(), labels.rig, creationTimestamp)",
    ]
    rc, out, err = await run_command(cmd)
    if rc != 0:
        return f"❌ List failed: {err}"
    return f"""### 📋 TPU instances in {zone}
```
{out or "(none)"}
```"""


@mcp.tool()
async def list_queued_resources(zone: str = ZONE) -> str:
    """Lists Queued Resources in a zone — the *sibling* rig's resources, not this one's.

    Nothing this rig creates appears here. It is kept because both rigs share a project and
    a zone and compete for the same physical chips, so "who is holding the v6e" is a
    question this rig has to be able to answer about the other control plane.
    """
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "list",
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--format=table(name, state.state, node_id, accelerator_type, create_time)",
    ]
    rc, out, err = await run_command(cmd)
    if rc == 0:
        return f"""### 📋 Queued Resources in {zone} (Cloud TPU API — not this rig's)
```
{out or "(none)"}
```"""
    else:
        return f"❌ List failed: {err}"


@mcp.tool()
async def describe_tpu_instance(instance_name: str = INSTANCE_NAME, zone: str = ZONE) -> str:
    """Provides detailed information about a specific TPU instance."""
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        instance_name,
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--format=json",
    ]
    rc, out, err = await run_command(cmd)
    if rc != 0:
        return f"❌ Describe failed: {err}"
    try:
        data = json.loads(out)
        scheduling = data.get("scheduling") or {}
        return (
            f"### 🔍 Detail: {instance_name}\n"
            f"- **Status:** `{data.get('status', 'UNKNOWN')}`\n"
            f"- **Machine type:** `{str(data.get('machineType', 'N/A')).split('/')[-1]}`\n"
            f"- **Provisioning model:** `{scheduling.get('provisioningModel', 'STANDARD')}`\n"
            f"- **Max run duration:** `{scheduling.get('maxRunDuration', 'none')}`\n"
            f"- **Termination action:** `{scheduling.get('instanceTerminationAction', 'none')}`\n"
            f"- **Owning rig:** `{(data.get('labels') or {}).get('rig', 'unlabelled')}`\n"
            f"- **Full Data:**\n```json\n{json.dumps(data, indent=2)}\n```"
        )
    except Exception:
        return f"""### 🔍 Detail: {instance_name}
```
{out}
```"""


@mcp.tool()
async def get_reservation_status(instance_name: str = INSTANCE_NAME) -> str:
    """Checks the lifecycle state and run bound of a TPU instance."""
    return await describe_tpu_instance(instance_name)


@mcp.tool()
async def check_tpu_availability(instance_name: str = INSTANCE_NAME) -> str:
    """Simple check to see whether a TPU instance has reached RUNNING.

    `RUNNING` is this path's `ACTIVE`. Note it is a weaker claim than the Queued Resource
    state was: a QR reached ACTIVE only once its node was up, whereas an instance is
    RUNNING from the moment the VM boots — long before the startup script has pulled the
    vLLM image or loaded the model. Use `verify_model_health` for readiness.
    """
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        instance_name,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--format=value(status)",
    ]
    rc, state, err = await run_command(cmd)
    if rc != 0:
        return f"❌ Check failed: {err}"
    is_running = state.strip() == "RUNNING"
    return (
        f"### 🧊 TPU instance availability: {instance_name}\n"
        f"- **Status:** `{state.strip()}`\n"
        f"- **Booted:** {'✅ Yes' if is_running else '⏳ No'}"
        + ("\n- ℹ️ RUNNING means the VM booted, not that vLLM is serving." if is_running else "")
    )


def _parse_topology(topology: str) -> Optional[int]:
    """Chip count from a topology string like '1x1' or '2x4x4'. None if it isn't one."""
    parts = topology.lower().split("x")
    if not all(p.strip().isdigit() for p in parts) or not parts:
        return None
    chips = 1
    for p in parts:
        chips *= int(p)
    return chips or None


async def _fetch_compute_skus() -> tuple[Optional[list], str]:
    """Fetches the Compute Engine SKU catalog from the Cloud Billing Catalog API.

    TPU SKUs live under the Compute Engine service. The catalog has no server-side
    description filter, so this pages through all ~32k SKUs and callers match locally.
    Cached for the life of the process — published prices do not move within a session.
    """
    if _SKU_CACHE.get("skus"):
        return _SKU_CACHE["skus"], ""

    rc, token, err = await run_command(["gcloud", "auth", "print-access-token"])
    if rc != 0 or not token:
        return None, f"could not get an access token ({err or 'gcloud auth print-access-token failed'})"

    skus: list = []
    page_token = None
    base = f"https://cloudbilling.googleapis.com/v1/services/{COMPUTE_BILLING_SERVICE_ID}/skus"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                params = {"pageSize": "5000", "currencyCode": "USD"}
                if page_token:
                    params["pageToken"] = page_token
                res = await client.get(base, params=params, headers={"Authorization": f"Bearer {token}"})
                if res.status_code != 200:
                    return None, f"Billing Catalog API returned HTTP {res.status_code}"
                payload = res.json()
                skus.extend(payload.get("skus", []))
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break
    except Exception as e:
        return None, f"Billing Catalog API request failed ({e})"

    _SKU_CACHE["skus"] = skus
    return skus, ""


async def _lookup_tpu_rate(
    tpu_type: str, provisioning_model: str, region: str
) -> tuple[Optional[float], str, Optional[str]]:
    """Looks up the published USD rate for one TPU chip-hour. Returns (rate, unit, sku_description).

    rate is None when the catalog has no SKU for that combination — this returns nothing
    rather than falling back to a guess, because a confident wrong price is worse than
    no price. Note published list rates ignore any negotiated or committed-use discount
    on the billing account, so a real invoice can be lower.

    reservation-bound has no catalog rate of its own: what it costs is whatever the
    reservation it consumes was priced at, which is a property of that reservation and not
    of the chip in the region. Saying so is the honest answer; picking the on-demand SKU
    because it is the closest match would invent a number.
    """
    if provisioning_model not in _SKU_DESCRIPTION_PATTERNS:
        return (
            None,
            "",
            (
                f"`{provisioning_model}` has no list rate in the billing catalog. Its cost is set by the "
                "reservation it consumes — read that reservation, not the public SKUs."
            ),
        )

    skus, err = await _fetch_compute_skus()
    if skus is None:
        return None, "", err

    pattern = re.compile(_SKU_DESCRIPTION_PATTERNS[provisioning_model].format(fam=re.escape(tpu_type)), re.I)
    usage_type = _SKU_USAGE_TYPES[provisioning_model]

    for sku in skus:
        if region not in sku.get("serviceRegions", []):
            continue
        if sku.get("category", {}).get("usageType") != usage_type:
            continue
        description = sku.get("description", "")
        if not pattern.match(description):
            continue
        try:
            expression = sku["pricingInfo"][0]["pricingExpression"]
            unit_price = expression["tieredRates"][-1]["unitPrice"]
            rate = int(unit_price.get("units", 0)) + unit_price.get("nanos", 0) / 1e9
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"SKU '{description}' has an unreadable price: {e}")
            continue
        return rate, expression.get("usageUnit", "h"), description

    return None, "", f"no `{provisioning_model}` SKU published for `{tpu_type}` in `{region}`"


@mcp.tool()
async def estimate_deployment_cost(
    hours: float = 1.0,
    tpu_type: str = "v6e",
    topology: str = TOPOLOGY,
    provisioning_model: str = PROVISIONING_MODEL,
    region: str = REGION,
) -> str:
    """Estimates the cost of a TPU deployment from live Google Cloud published pricing.

    Rates come from the Cloud Billing Catalog API, not a table in this file — an earlier
    hardcoded table was wrong by 10x and there is no way to notice that from inside the
    code. If the catalog has no matching SKU this reports that instead of guessing.

    **Every catalog rate is per chip-hour and this rig's slice is eight chips.** The default
    topology is `TOPOLOGY` (2x4 for v6e-8), so the figure is the whole slice, not a chip —
    8x what the single-chip rigs quote for the same hours. Pass `topology="1x1"` only to
    price one chip on purpose.

    Requires a usable gcloud access token and the Cloud Billing API enabled on the project.
    """
    if provisioning_model not in PROVISIONING_MODELS:
        return f"❌ Unknown provisioning_model '{provisioning_model}'. Use one of {PROVISIONING_MODELS}."

    chips = _parse_topology(topology)
    if chips is None:
        return f"❌ Could not read a chip count from topology `{topology}`. Expected something like `1x1` or `2x4`."

    rate, unit, detail = await _lookup_tpu_rate(tpu_type, provisioning_model, region)
    if rate is None:
        return (
            f"❌ No published rate found: {detail}.\n"
            f"Nothing is estimated rather than guessing — check "
            f"https://cloud.google.com/tpu/pricing for `{tpu_type}` in `{region}`."
        )

    total_cost = chips * rate * hours
    lines = [
        f"### 💸 Estimated Cost: `${total_cost:.2f}` for `{hours}h` on `{chips}` chip `{tpu_type}` "
        f"({provisioning_model}) in `{region}`",
        f"- **Rate:** `${rate:.4f}` per chip-{unit} × `{chips}` chips × `{hours}h`",
        f"- **SKU:** {detail}",
    ]
    if chips != CHIP_COUNT:
        lines.append(
            f"- ⚠️ Topology `{topology}` is `{chips}` chips, but this rig provisions "
            f"`{MACHINE_TYPE}` = `{CHIP_COUNT}` chips. Multiply by `{CHIP_COUNT / chips:.2g}` "
            f"for what a `make deploy-tpu` here actually bills."
        )
    if provisioning_model == "spot":
        lines.append("- ⚠️ Spot can be preempted mid-run, so billed hours may be shorter than requested.")
    elif provisioning_model == "flex-start":
        lines.append("- Flex-start self-terminates at `--max-run-duration` (4h here), capping the bill.")
    else:
        lines.append("- ⚠️ On-demand has no run bound — this bills until you destroy the resource.")
    lines.append("_List price from the Cloud Billing Catalog; committed-use or negotiated discounts are not applied._")
    return "\n".join(lines)


@mcp.tool()
async def get_system_status() -> str:
    """Provides a high-level dashboard of system status."""
    health = "🔴 Offline"
    node = await _discover_vllm_node()
    if node:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"{node.url}/health", timeout=2)
                if res.status_code == 200:
                    health = f"🟢 Online ({node.url} on `{node.name}`)"
        except Exception:
            pass
        if "🔴" in health:
            health = f"🟡 Instance `{node.name}` up at {node.url}, vLLM not answering"

    instances = await _list_tpu_vm_nodes()
    if instances:
        rows = [
            f"- `{(n.get('name') or '').split('/')[-1]}` — {n.get('status', 'UNKNOWN')}, "
            f"{str(n.get('machineType', 'unknown')).split('/')[-1]}, "
            f"rig={(n.get('labels') or {}).get('rig', 'unlabelled')}, ip {_node_ip(n) or 'none'}"
            for n in instances
        ]
        instances_str = "**🖥️ TPU instances (Compute Engine):**\n" + "\n".join(rows)
    else:
        instances_str = "**🖥️ TPU instances (Compute Engine):** none"

    # The sibling rig provisions into this same zone through the Cloud TPU API, and a chip it
    # holds is a chip this rig cannot get. Surfacing it here is the difference between "no
    # capacity" and "the other rig has it", which are the same symptom and different fixes.
    qr_str = await list_queued_resources()
    contention = ""
    if "ACTIVE" in qr_str:
        contention = "\n- ⚠️ The sibling TPU-API rig holds an ACTIVE queued resource in this zone."

    if "🟢" in health:
        next_step = "Use `query_queued_gemma4` to interact with the model."
    elif node:
        next_step = "An instance is up but vLLM is not answering — use `manage_vllm_docker` to start the service."
    else:
        next_step = "Call `create_tpu_instance` to provision infrastructure."

    return (
        f"### 🌀 System Status ({ZONE})\n- **vLLM Health:** {health}{contention}\n"
        f"{instances_str}\n{qr_str}\n**👉 Next Step:** {next_step}"
    )


@mcp.tool()
async def get_vllm_endpoint() -> str:
    """Returns the active vLLM service URL if available."""
    url = await discover_vllm_url()
    if url:
        return f"🟢 vLLM is Online at: {url}"
    return f"❌ No TPU node in {ZONE} is serving vLLM (checked queued resources and standalone TPU VMs)."


@mcp.tool()
async def get_deployed_endpoint() -> str:
    """Returns the raw URL of the active vLLM service."""
    url = await discover_vllm_url()
    return url if url else "None"


@mcp.tool()
async def query_queued_gemma4(prompt: str) -> str:
    """Queries the self-hosted Gemma 4 model on whichever TPU node is serving it."""
    logger.info(f"Querying model with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
        )
        response = chat_completion.choices[0].message.content or "No response from model."
        logger.info(f"Model response: '{response[:100]}...'")
        return response or "No response from model."
    except Exception as e:
        logger.error(f"Error querying model: {e}")
        return f"❌ An error occurred while querying the model: {e}"


@mcp.tool()
async def query_queued_gemma4_with_stats(prompt: str) -> str:
    """
    Queries the self-hosted Gemma 4 model and returns detailed performance statistics.

    This tool provides:
    - The full model response.
    - Time to First Token (TTFT).
    - Total generation time.
    - Tokens per second.
    """
    logger.info(f"Querying model with stats with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()

        start_time = time.monotonic()
        ttft = None
        response_content = ""
        total_tokens = 0

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
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
        logger.error(f"Error querying model with stats: {e}")
        return f"❌ An error occurred while querying the model with stats: {e}"


BENCH_RESULT_MARKER = "---BENCH-RESULT-JSON---"


def _sweep_point_from_bench_result(result: dict, concurrency: int, input_len: int, output_len: int) -> dict:
    """Maps a `vllm bench serve --save-result` dump to a `throughput.sweep[]` entry of
    benchmarks/serving-report.schema.json v1.1.

    input_len is passed in rather than read from the dump: vLLM records `random_input_len`
    as null there, and it is the second axis of a 2-D sweep, so losing it would collapse
    the cell key. The full dump goes under `raw` minus its list-valued keys, which are
    per-request arrays with one element per prompt."""

    def _stats(metric: str) -> dict:
        out = {}
        for stat in ("mean", "median", "p90", "p99"):
            v = result.get(f"{stat}_{metric}_ms")
            if isinstance(v, (int, float)):
                out[stat] = round(v, 2)
        return out

    point: dict = {
        "concurrency": concurrency,
        "input_len": input_len,
        "output_len": output_len,
        "status": "ok",
    }
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


@mcp.tool()
async def run_vllm_benchmark(
    resource_id: str = RESOURCE_ID,
    backend: str = "vllm",
    model: str = MODEL_NAME,
    dataset_name: str = "random",
    num_prompts: int = 100,
    random_input_len: int = 1024,
    random_output_len: int = 128,
    max_concurrency: Optional[int] = None,
    save_result: bool = False,
) -> str:
    """Runs vLLM's internal benchmark tool inside the container on the TPU VM.

    With save_result=True the run's --save-result JSON is fetched back and returned as a
    ready-made throughput.sweep[] entry for benchmarks/serving-report.schema.json, one call
    per sweep point."""
    node_id = await _resolve_node_id(resource_id)
    if not node_id:
        return (
            f"❌ Could not find a TPU node for {resource_id} in {ZONE} "
            "(no ACTIVE queued resource and no standalone TPU VM by that name)."
        )

    benchmark_cmd = (
        "vllm bench serve "
        f"--backend {backend} "
        f"--model {model} "
        f"--dataset-name {dataset_name} "
        f"--num-prompts {num_prompts} "
        f"--random-input-len {random_input_len} "
        f"--random-output-len {random_output_len}"
    )
    if max_concurrency:
        benchmark_cmd += f" --max-concurrency {max_concurrency}"

    # /dev/shm is already bind-mounted into the container, so a result written there outlives
    # the --rm container and can be read back in the same SSH session.
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

    ssh_cmd = _ssh_command(node_id, remote_cmd, ensure_docker=True)

    rc, out, err = await run_command(ssh_cmd, timeout=600)  # Increased timeout for benchmark
    if rc != 0:
        return f"""⚠️ Benchmark failed on {node_id}.
Error: {err}
Output: {out}"""
    if not save_result:
        return f"""✅ Benchmark completed on {node_id}:
{out}"""

    bench_stdout, sep, result_json = out.partition(BENCH_RESULT_MARKER)
    if not sep:
        return f"""⚠️ Benchmark ran on {node_id} but no result JSON came back:
{out}"""
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        return f"""⚠️ Benchmark ran on {node_id} but the result JSON did not parse ({e}):
{result_json.strip()[:2000]}"""
    concurrency = int(max_concurrency) if max_concurrency else int(num_prompts)
    point = _sweep_point_from_bench_result(result, concurrency, int(random_input_len), int(random_output_len))
    return (
        f"✅ Benchmark completed on {node_id}.\n\n"
        "throughput.sweep[] entry (benchmarks/serving-report.schema.json):\n"
        f"```json\n{json.dumps(point, indent=2)}\n```\n\n"
        f"Benchmark output:\n{bench_stdout.strip()}"
    )


@mcp.tool()
async def get_vllm_docker_logs(resource_id: str = RESOURCE_ID, tail: Optional[int] = None) -> str:
    """Retrieves logs from the vLLM Docker container on the TPU VM."""
    node_id = await _resolve_node_id(resource_id)
    if not node_id:
        return f"❌ Could not find an instance named {resource_id} in {ZONE}."

    log_cmd = "sudo docker logs vllm-gemma4"
    if tail:
        log_cmd += f" --tail {tail}"

    ssh_cmd = _ssh_command(node_id, log_cmd, ensure_docker=True)

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Failed to get Docker logs from {node_id}.
Error: {err}"""
    return f"""✅ Docker logs from {node_id}:
{out}"""


@mcp.tool()
async def get_tpu_system_logs(
    resource_id: str = RESOURCE_ID, service: str = "docker", tail: Optional[int] = None
) -> str:
    """Retrieves systemd logs for a specific service from the TPU VM."""
    node_id = await _resolve_node_id(resource_id)
    if not node_id:
        return f"❌ Could not find an instance named {resource_id} in {ZONE}."

    log_cmd = f"journalctl -u {service} -n {tail or 100}"

    # No ensure_docker: journalctl is not a Docker command. Note `service="docker"` is the
    # default and reads the daemon's own unit log — which on a fresh boot may not exist at
    # all, because the unit is only installed once something installs Docker.
    ssh_cmd = _ssh_command(node_id, log_cmd)

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Failed to get system logs from {node_id}.
Error: {err}"""
    return f"""✅ System logs for '{service}' from {node_id}:
{out}"""


@mcp.tool()
async def get_cloud_logging_logs(log_filter: str = 'resource.type="tpu_worker"', limit: int = 20) -> str:
    """Fetches logs from Google Cloud Logging."""
    cmd = ["gcloud", "logging", "read", log_filter, f"--project={PROJECT_ID}", f"--limit={limit}", "--format=json"]
    rc, out, err = await run_command(cmd)
    if rc != 0:
        return f"❌ Failed to fetch Cloud Logs: {err}"

    try:
        logs = json.loads(out)
        formatted_logs = "\n".join(
            [
                f"[{log_entry.get('timestamp')}] {log_entry.get('resource', {}).get('labels', {}).get('node_id', 'N/A')} - "
                f"{log_entry.get('textPayload', log_entry.get('jsonPayload', {}))}"
                for log_entry in logs
            ]
        )
        return f"### ☁️ Cloud Logs (filter: `{log_filter}`)\n```\n{formatted_logs}\n```"
    except Exception:
        return f"### ☁️ Cloud Logs (raw)\n```\n{out}\n```"


@mcp.tool()
async def analyze_cloud_logging(minutes: int = 60) -> str:
    """Summarizes TPU-related errors using the self-hosted Gemma 4 model."""
    log_filter = f'resource.type="tpu_worker" severity>=ERROR timestamp>="-PT{minutes}M"'
    logs_result = await get_cloud_logging_logs(log_filter=log_filter, limit=10)

    if "error" in logs_result.lower() or "failed" in logs_result.lower() or "```\n\n```" in logs_result:
        prompt = "Provide a summary of common TPU node issues (e.g. out of memory, VM preemption) and their standard remediations."
    else:
        prompt = (
            f"Here are the recent TPU error logs:\n{logs_result}\n\n"
            "Please analyze these logs, identify the root cause of the failures, and suggest remediations."
        )

    try:
        summary = await query_queued_gemma4(prompt)
        return f"### 🔍 Log Analysis Summary\n\n{summary}"
    except Exception as e:
        return f"❌ Failed to analyze logs: {e}"


@mcp.tool()
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
        return f"❌ No TPU node in {ZONE} is serving vLLM (checked queued resources and standalone TPU VMs)."

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
                metrics_lines = metrics_res.text.splitlines()
                key_metrics = [
                    line
                    for line in metrics_lines
                    if "vllm_requests_running" in line
                    or "vllm_requests_swapped" in line
                    or "vllm_requests_waiting" in line
                    or "vllm_tpu_cache_usage_perc" in line
                    or "process_resident_memory_bytes" in line
                ]
                if key_metrics:
                    report += "```\n" + "\n".join(key_metrics) + "\n```\n"
                else:
                    report += "Metrics endpoint available, but no key metrics found in snippet.\n"
            else:
                report += "⚠️ Metrics endpoint not available or failed.\n"
        except Exception as e:
            report += f"❌ Error fetching metrics: {e}\n"

    return report


@mcp.tool()
async def get_help() -> str:
    """Provides help text and summarizes the configuration options and all available SRE/DevOps tools for this TPU Cloud Run/VM MCP server."""
    return (
        "### 🛠️ TPU Gemma 4 SRE Agent Help & Configuration\n\n"
        "You can configure this MCP server using the following environment variables:\n\n"
        f"- **`GOOGLE_CLOUD_PROJECT`**: Your GCP Project ID.\n"
        f"  - *Current Value:* `{PROJECT_ID}`\n"
        f"- **`GOOGLE_CLOUD_ZONE`**: The GCP Zone for deployment.\n"
        f"  - *Current Value:* `{ZONE}`\n"
        f"- **`GOOGLE_CLOUD_REGION`**: The GCP Region for network resources.\n"
        f"  - *Current Value:* `{REGION}`\n"
        f"- **`MODEL_NAME`**: Default Hugging Face repository or path.\n"
        f"  - *Current Value:* `{MODEL_NAME}`\n"
        f"- **`ACCELERATOR_TYPE`**: TPU Accelerator type (documentation only on this path).\n"
        f"  - *Current Value:* `{ACCELERATOR_TYPE}`\n"
        f"- **`MACHINE_TYPE`**: What Compute Engine actually provisions.\n"
        f"  - *Current Value:* `{MACHINE_TYPE}` (`{CHIP_COUNT}` chips, topology `{TOPOLOGY}`)\n"
        f"- **`TENSOR_PARALLEL_SIZE`**: Tensor parallel size for serving.\n"
        f"  - *Current Value:* `{TENSOR_PARALLEL_SIZE}`\n\n"
        "### ℹ️ Active Mode Summary\n"
        "The server is running in **TPU** mode targeting TPU VM resources.\n\n"
        "---\n\n"
        "### 🧰 Available MCP Tools\n\n"
        "Below is a summary of the tools exposed by this SRE/DevOps agent:\n\n" + await _render_tool_catalog()
    )


@mcp.tool()
async def get_metrics() -> str:
    """
    Fetches raw Prometheus metrics from the running vLLM service's /metrics endpoint.
    """
    url = await discover_vllm_url()
    if not url:
        return f"❌ No TPU node in {ZONE} is serving vLLM (checked queued resources and standalone TPU VMs)."

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{url}/metrics")
            if res.status_code == 200:
                return res.text
            else:
                return f"❌ Failed to fetch metrics. Status code: {res.status_code}\nResponse: {res.text}"
    except Exception as e:
        return f"❌ Error connecting to vLLM metrics endpoint: {e}"


@mcp.tool()
async def get_active_models() -> str:
    """Gets the active resource usage (actively loaded models, sizes, CPU/GPU status, context size) via ollama ps."""
    if "ollama" not in LOCAL_DOCKER_IMAGE.lower():
        return "❌ Active resource usage (ollama ps) is only supported on Ollama backend."

    cmd = ["docker", "exec", "gemma4", "ollama", "ps"]
    rc, out, err = await run_command(cmd, timeout=30)
    if rc != 0:
        return f"⚠️ Failed to check active models.\nError: {err}\nOutput: {out}"
    return f"### 📊 Active Loaded Models:\n\n```\n{out}\n```"


@mcp.tool()
async def get_model_show_details(model_name: str) -> str:
    """Gets deep model parameters, architecture, license, and config details via ollama show <model_name>."""
    if "ollama" not in LOCAL_DOCKER_IMAGE.lower():
        return "❌ Deep model details (ollama show) are only supported on Ollama backend."

    cmd = ["docker", "exec", "gemma4", "ollama", "show", model_name]
    rc, out, err = await run_command(cmd, timeout=30)
    if rc != 0:
        return f"⚠️ Failed to get model details for {model_name}.\nError: {err}\nOutput: {out}"
    return f"### 🧩 Model Details for `{model_name}`:\n\n```\n{out}\n```"


@mcp.tool()
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
