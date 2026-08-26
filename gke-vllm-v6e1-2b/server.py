import asyncio
import base64
import json
import logging
import os
import re
import string
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
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-E2B-it")
# Secret Manager secret holding the Hugging Face token. The startup script fetches it by
# id at boot, so a rotated or per-project secret only needs this to change.
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "hf-token")
# There is no instance name on this path and no Queued Resource id either — the names that
# matter are the cluster and the node pool (GKE_CLUSTER_NAME / GKE_NODE_POOL below), both
# derived from the rig directory the same way. INSTANCE_NAME is kept ONLY as the default
# those derive from and for tpu.env compatibility with the sibling rigs; nothing here
# provisions an instance. A rename orphans whatever is already provisioned, so pin the
# cluster name in tpu.env before renaming the directory.
INSTANCE_NAME = os.getenv("INSTANCE_NAME", RIG_NAME)
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

# ACCELERATOR_TYPE is documentation on this path — the Cloud TPU API's spelling of the chip,
# kept so reports and the directory name line up with the sibling rig. What gcloud actually
# consumes is MACHINE_TYPE below. Never pass this to `gcloud compute instances create`.
ACCELERATOR_TYPE = os.getenv("ACCELERATOR_TYPE", "v6e-1")

# The Compute Engine machine type is the real accelerator request. `ct6e-standard-1t` is one
# v6e chip, 44 vCPU / 176 GB. There is a second family spelled `<name>-tpu` (identical vCPU,
# memory and zone coverage; `guestAcceleratorType: tpu-v6e` instead of `ct6e`) — see
# ../HARDWARE.md. They are not known to be interchangeable, so the exact string is config.
MACHINE_TYPE = os.getenv("MACHINE_TYPE", "ct6e-standard-1t")
# Replaces the TPU API's --runtime-version. Ubuntu 22.04 / kernel 6.8, shared across
# v5e/v5p/v6e, preloaded with the TPU runtime, drivers and agents. Pin the *family*, never a
# dated build: images ship roughly weekly and every superseded build goes DEPRECATED.
IMAGE_FAMILY = os.getenv("IMAGE_FAMILY", "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e")
IMAGE_PROJECT = os.getenv("IMAGE_PROJECT", "ubuntu-os-accelerator-images")
# The image default is 10 GB, which cannot hold the vLLM TPU image. Undersizing this fails
# late — after boot, during the docker pull — so it is a default rather than a caller's problem.
BOOT_DISK_SIZE_GB = os.getenv("BOOT_DISK_SIZE_GB", "200")

# Quota ids, and the trap that defines this rig: **the two control planes meter against
# different pools.** The TPU API ids below are what the sibling tpu-vllm-v6e1-2b uses; they
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
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

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
# the rendered manifest and `_vllm_serve_flags`) read these, so the pod, the shell path's
# manifest and a hand-run container all serve the same config.
VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-tpu:nightly")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "16384")
MAX_NUM_BATCHED_TOKENS = os.getenv("MAX_NUM_BATCHED_TOKENS", "4096")
LIMIT_MM_PER_PROMPT = os.getenv("LIMIT_MM_PER_PROMPT", '{"image":4,"audio":1}')

# Empty by default: gcloud then uses the project's default network. Set these only if
# the project really has a custom VPC — and remember subnetworks are regional, so a
# named subnet exists in one region only and will break find_tpu's cross-zone sweep.
TPU_NETWORK = os.getenv("TPU_NETWORK", "")
TPU_SUBNETWORK = os.getenv("TPU_SUBNETWORK", "")

# --- GKE: the control plane this rig provisions through ---------------------------------
#
# The accelerator here is a property of a NODE POOL, not of an instance. Nothing below has
# an analogue in the sibling gce-vllm-v6e1-2b rig, and the Compute Engine constants above
# (IMAGE_FAMILY, BOOT_DISK_SIZE_GB, MAX_RUN_DURATION) do not apply to a node pool at all —
# they are kept only because estimate/compare tooling reports both paths.
GKE_CLUSTER_NAME = os.getenv("GKE_CLUSTER_NAME", RIG_NAME)
GKE_NODE_POOL = os.getenv("GKE_NODE_POOL", "tpu-v6e-1")
# A TPU node pool has to sit in the zone that has the chips, so the cluster is zonal.
GKE_LOCATION = os.getenv("GKE_LOCATION", ZONE)
GKE_NUM_NODES = os.getenv("GKE_NUM_NODES", "1")

# TWO TOPOLOGY VALUES, AND THEY ARE NOT THE SAME THING. Verified 2026-08-25 on this rig's
# first node-pool create, which passed --tpu-topology=1x1 and was refused outright:
#
#   400: TPU topology can't be specified with single-host TPU slice pool;
#        please remove the tpu_topology from the node pool creation request
#
# ct6e-standard-1t at one node IS the slice — there is no topology to describe. The flag
# belongs to MULTI-host slices, where it says how several nodes are wired into one. What
# makes this quiet is that GKE then labels the node cloud.google.com/gke-tpu-topology=1x1
# anyway, so the value is real as a SELECTOR and rejected as a CREATE FLAG.
TPU_TOPOLOGY = os.getenv("TPU_TOPOLOGY", "1x1")  # pod nodeSelector label
GKE_TPU_TOPOLOGY = os.getenv("GKE_TPU_TOPOLOGY", "")  # multi-host create flag; empty = single-host
GKE_TPU_ACCELERATOR = os.getenv("GKE_TPU_ACCELERATOR", "tpu-v6e-slice")
GKE_NODE_DISK_SIZE_GB = os.getenv("GKE_NODE_DISK_SIZE_GB", "200")
# The small pool that runs kube-dns and metrics-server. Keeping the system workloads off the
# TPU node is why it exists: a v6e node is billed by the chip and must not be kept alive by
# CoreDNS after the model pod is gone.
GKE_SYSTEM_MACHINE_TYPE = os.getenv("GKE_SYSTEM_MACHINE_TYPE", "e2-standard-4")
GKE_SYSTEM_DISK_SIZE_GB = os.getenv("GKE_SYSTEM_DISK_SIZE_GB", "50")
# Pin the CHANNEL, never a version string. v6e needs a recent control plane and every pinned
# version goes stale; rapid gave 1.36.3-gke.1537000 on 2026-08-25.
GKE_RELEASE_CHANNEL = os.getenv("GKE_RELEASE_CHANNEL", "rapid")
# LoadBalancer matches the sibling rigs' network path — an external IP on :8000 — which is
# what keeps a benchmark comparable across the three control planes. It is UNAUTHENTICATED.
GKE_SERVICE_TYPE = os.getenv("GKE_SERVICE_TYPE", "LoadBalancer")
GKE_NODE_PROVISIONING = os.getenv("GKE_NODE_PROVISIONING", "on-demand")

# Kubernetes object names. The manifest template is shared with the `make gke-*` shell path
# so the serving flags cannot drift between the two.
K8S_DEPLOYMENT = os.getenv("K8S_DEPLOYMENT", "vllm-gemma4")
K8S_SERVICE = os.getenv("K8S_SERVICE", "vllm-gemma4")
HF_K8S_SECRET = os.getenv("HF_K8S_SECRET", "hf-token")
VLLM_SHM_SIZE = os.getenv("VLLM_SHM_SIZE", "16Gi")
MANIFEST_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gke", "vllm-gemma4.yaml.tmpl")

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


async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from Secret Manager."""
    # --project is load-bearing and its absence fails confusingly: without it gcloud uses the
    # machine's DEFAULT project, which on this workstation is an expired qwiklabs lab, and
    # Secret Manager answers "Permission denied on resource project qwiklabs-..." naming a
    # project nothing here ever mentions. Inherited from the fork; the shell path always
    # passed it, which is why only the MCP tool hit it.
    rc, stdout, stderr = await run_command(
        ["gcloud", "secrets", "versions", "access", "latest", f"--secret={secret_id}", f"--project={PROJECT_ID}"]
    )
    if rc == 0:
        return stdout.strip()
    logger.error(f"Failed to access secret {secret_id} via gcloud (exit code {rc}): {stderr}")
    return None


def _vllm_serve_flags(mm_limit: Optional[str] = None) -> str:
    """The vLLM serve flags for Gemma 4 on TPU, shared by every deployment path.

    mm_limit: the already-quoted --limit-mm-per-prompt value. Defaults to single quotes,
    which is wrong inside an outer single-quoted argument — pass the double-quoted,
    backslash-escaped form there instead.
    """
    if mm_limit is None:
        mm_limit = f"'{LIMIT_MM_PER_PROMPT}'"
    return (
        f"--max-model-len {MAX_MODEL_LEN} "
        f"--tensor-parallel-size {TENSOR_PARALLEL_SIZE} "
        f"--disable_chunked_mm_input "
        f"--max_num_batched_tokens {MAX_NUM_BATCHED_TOKENS} "
        f"--limit-mm-per-prompt {mm_limit} "
        f"--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4"
    )


def _quota_id_for(provisioning_model: str) -> str:
    """Maps a provisioning model to a **Cloud TPU API** quota id (TPU_QUOTA_ID / TPU_SPOT_QUOTA_ID).

    TPU API only — used when reporting the sibling rig's pools. **Do not read this as the
    Compute Engine rule, which is different**: there, flex-start spends the *preemptible*
    quota alongside spot, and only on-demand draws on the family quota. See GCE_QUOTA_ID.
    """
    return TPU_SPOT_QUOTA_ID if provisioning_model == "spot" else TPU_QUOTA_ID


_ENSURE_DOCKER = (
    "if ! command -v docker > /dev/null 2>&1; then "
    "sudo apt-get update -qq && "
    "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io && "
    "sudo systemctl enable --now docker; fi"
)


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


class VllmNode(NamedTuple):
    """What discovery found, and whether vLLM answered on it.

    `name` is the vLLM pod on this path, not a machine: the unit that serves the model is a
    pod, and it can be rescheduled onto a different node without the endpoint changing.
    """

    name: str
    url: str
    serving: bool


# --- GKE plumbing -----------------------------------------------------------------------


def _cluster_context(cluster_name: str = "", location: str = "") -> str:
    """The kubeconfig context name `get-credentials` writes for a cluster.

    Every kubectl call is pinned to this explicitly rather than trusting the current
    context. The current context is machine-global state shared with every other cluster
    the user has ever fetched credentials for, so an unpinned `kubectl delete` is one
    stale context away from acting on somebody else's cluster.
    """
    return f"gke_{PROJECT_ID}_{location or GKE_LOCATION}_{cluster_name or GKE_CLUSTER_NAME}"


async def _ensure_cluster_credentials(cluster_name: str = "", location: str = "") -> tuple[bool, str]:
    """Fetches kubeconfig credentials for the cluster. Idempotent and cheap to repeat."""
    cluster = cluster_name or GKE_CLUSTER_NAME
    loc = location or GKE_LOCATION
    cmd = [
        "gcloud",
        "container",
        "clusters",
        "get-credentials",
        cluster,
        f"--location={loc}",
        f"--project={PROJECT_ID}",
    ]
    rc, _, err = await run_command(cmd, timeout=120)
    if rc != 0:
        return False, f"❌ Could not fetch credentials for cluster `{cluster}` in {loc}: {err.strip()}"
    return True, ""


def _kubectl(args: list[str], cluster_name: str = "", location: str = "") -> list[str]:
    """kubectl argv pinned to this rig's cluster context."""
    return ["kubectl", f"--context={_cluster_context(cluster_name, location)}", *args]


async def _kubectl_json(args: list[str], cluster_name: str = "", location: str = "", timeout: int = 60):
    """Runs a kubectl command with -o json and parses it. Returns None on any failure."""
    rc, out, err = await run_command(_kubectl([*args, "-o", "json"], cluster_name, location), timeout=timeout)
    if rc != 0 or not out:
        logger.info(f"kubectl {' '.join(args)} failed: {err.strip() or out.strip()}")
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        logger.error(f"Could not parse kubectl output: {e}")
        return None


async def _cluster_exists(cluster_name: str = "", location: str = "") -> bool:
    cmd = [
        "gcloud",
        "container",
        "clusters",
        "describe",
        cluster_name or GKE_CLUSTER_NAME,
        f"--location={location or GKE_LOCATION}",
        f"--project={PROJECT_ID}",
        "--format=value(status)",
    ]
    rc, _, _ = await run_command(cmd, timeout=120)
    return rc == 0


async def _node_pool_exists(node_pool: str = "", cluster_name: str = "", location: str = "") -> bool:
    cmd = [
        "gcloud",
        "container",
        "node-pools",
        "describe",
        node_pool or GKE_NODE_POOL,
        f"--cluster={cluster_name or GKE_CLUSTER_NAME}",
        f"--location={location or GKE_LOCATION}",
        f"--project={PROJECT_ID}",
        "--format=value(status)",
    ]
    rc, _, _ = await run_command(cmd, timeout=120)
    return rc == 0


def _node_pool_provisioning_flags(
    provisioning_model: str, num_nodes: str = "", reservation_name: Optional[str] = None
) -> list[str]:
    """The `node-pools create` flags that select how the pool asks for capacity.

    Third vocabulary for the same four ideas, and every one of them is spelled differently
    again. `queued-resources create` takes `--provisioning-model=flex-start`; `instances
    create` takes `--provisioning-model=FLEX_START`; a node pool takes a bare `--flex-start`
    with no --provisioning-model flag at all. This is the single place the GKE mapping
    lives — do not add a second one.

    Two differences that are more than spelling:

    - **Flex-start on a node pool is an autoscaling shape, not just a flag.** It creates the
      pool at zero nodes and lets the autoscaler pull capacity when a pod needs it, which is
      why the node count moves into --total-max-nodes. A plain --num-nodes with --flex-start
      is not the documented form.
    - **There is no --max-run-duration and no termination action.** A Compute Engine
      instance can be told to delete itself; a node pool cannot. Nothing here stops billing
      on a timer — `destroy_tpu_node_pool` is the only stop, which is why this rig's
      teardown tool matters more than the twin's did.
    """
    nodes = num_nodes or GKE_NUM_NODES
    if provisioning_model == "spot":
        return ["--spot", f"--num-nodes={nodes}"]
    if provisioning_model == "on-demand":
        return [f"--num-nodes={nodes}"]
    if provisioning_model == "reservation-bound":
        reservation = reservation_name or RESERVATION_NAME
        affinity = ["--reservation-affinity=specific", f"--reservation={reservation}"] if reservation else []
        return [*affinity, f"--num-nodes={nodes}"]
    return [
        "--flex-start",
        "--enable-autoscaling",
        "--num-nodes=0",
        "--total-min-nodes=0",
        f"--total-max-nodes={nodes}",
        "--location-policy=ANY",
        "--reservation-affinity=none",
        "--no-enable-autorepair",
    ]


def _topology_flags() -> list[str]:
    """--tpu-topology, but only for a multi-host slice. See GKE_TPU_TOPOLOGY."""
    if not GKE_TPU_TOPOLOGY:
        return []
    return [f"--tpu-topology={GKE_TPU_TOPOLOGY}", "--placement-type=COMPACT"]


def _render_vllm_manifest(model_name: str = "", service_type: str = "") -> str:
    """Renders the Deployment + Service YAML from the template the shell path also uses.

    ONE TEMPLATE, TWO CALLERS. `gke/gke-deploy.sh` renders this file with envsubst and these
    tools render it with string.Template; both read the same tpu.env values. That is
    deliberate — the serving flags are what has to stay identical across the three v6e-1
    rigs, and a second hardcoded copy here is exactly how they would drift apart.
    """
    with open(MANIFEST_TEMPLATE) as f:
        template = string.Template(f.read())
    return template.substitute(
        MODEL_NAME=model_name or MODEL_NAME,
        MAX_MODEL_LEN=MAX_MODEL_LEN,
        TENSOR_PARALLEL_SIZE=TENSOR_PARALLEL_SIZE,
        MAX_NUM_BATCHED_TOKENS=MAX_NUM_BATCHED_TOKENS,
        TPU_TOPOLOGY=TPU_TOPOLOGY,
        GKE_TPU_ACCELERATOR=GKE_TPU_ACCELERATOR,
        GKE_SERVICE_TYPE=service_type or GKE_SERVICE_TYPE,
        VLLM_IMAGE=VLLM_IMAGE,
        VLLM_SHM_SIZE=VLLM_SHM_SIZE,
        HF_K8S_SECRET=HF_K8S_SECRET,
    )


async def _kubectl_apply(manifest: str, cluster_name: str = "", location: str = "") -> tuple[bool, str]:
    """Writes a manifest to a private temp file and applies it.

    A file rather than stdin because run_command uses create_subprocess_exec and never a
    shell, so there is no pipe to write into. Mode 0600 because the HF token's Secret goes
    through here.
    """
    fd, path = tempfile.mkstemp(prefix="vllm-gemma4-", suffix=".yaml")
    try:
        os.write(fd, manifest.encode())
        os.close(fd)
        os.chmod(path, 0o600)
        rc, out, err = await run_command(_kubectl(["apply", "-f", path], cluster_name, location), timeout=180)
        return rc == 0, (out or err).strip()
    finally:
        os.unlink(path)


async def _service_endpoint(cluster_name: str = "", location: str = "") -> Optional[str]:
    """The URL vLLM is reachable on, or None.

    **This is the discovery difference that matters most on this path.** The two sibling
    rigs read an IP off the machine: a Queued Resource's node, or a Compute Engine
    instance's networkInterfaces[].accessConfigs[].natIP. Neither is where the model is
    listening here. A GKE node *does* appear in `gcloud compute instances list` — so the
    sibling's discovery call succeeds and returns the wrong object rather than failing —
    but the model is behind a Service, and only the Service knows the address.
    """
    svc = await _kubectl_json(["get", "svc", K8S_SERVICE], cluster_name, location)
    if not svc:
        return None
    ingress = ((svc.get("status") or {}).get("loadBalancer") or {}).get("ingress") or []
    for entry in ingress:
        ip = entry.get("ip") or entry.get("hostname")
        if ip:
            return f"http://{ip}:8000"
    # ClusterIP (or a LoadBalancer still being assigned) is not reachable from here. Say so
    # by returning None rather than handing back an address that will hang.
    logger.info(f"Service {K8S_SERVICE} has no external address yet (type {(svc.get('spec') or {}).get('type')}).")
    return None


async def _vllm_pod(cluster_name: str = "", location: str = "") -> Optional[dict]:
    """The vLLM pod, if one exists."""
    pods = await _kubectl_json(["get", "pods", "-l", f"app={K8S_DEPLOYMENT}"], cluster_name, location)
    items = (pods or {}).get("items") or []
    return items[0] if items else None


def _pod_summary(pod: dict) -> str:
    """One line: phase, readiness and where it landed."""
    name = (pod.get("metadata") or {}).get("name", "unknown")
    phase = (pod.get("status") or {}).get("phase", "Unknown")
    node = (pod.get("spec") or {}).get("nodeName", "unscheduled")
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    ready = bool(statuses and statuses[0].get("ready"))
    state = "ready" if ready else "not ready"
    if statuses and not ready:
        waiting = (statuses[0].get("state") or {}).get("waiting") or {}
        if waiting.get("reason"):
            state = f"not ready ({waiting['reason']})"
    return f"`{name}` — {phase}, {state}, on `{node}`"


async def _discover_vllm_node() -> Optional[VllmNode]:
    """Finds the vLLM endpoint on this rig's cluster.

    Returns serving=True only once /v1/models answers. A pod that exists but is not ready
    is returned with serving=False so callers can poll it — the model pull, weight load and
    XLA precompile take about ten minutes on this path, and `Running` says nothing about
    any of them. That is the same "RUNNING is a weaker claim than ACTIVE" trap the Compute
    Engine twin documents, one layer further down: here even a Ready *node* tells you
    nothing, because the model lives in a pod scheduled onto it afterwards.
    """
    url = await _service_endpoint()
    pod = await _vllm_pod()
    if not url:
        if pod:
            return VllmNode((pod.get("metadata") or {}).get("name", K8S_DEPLOYMENT), "", False)
        return None
    model_id = await _probe_vllm(url)
    name = (pod.get("metadata") or {}).get("name", K8S_DEPLOYMENT) if pod else K8S_SERVICE
    if model_id:
        logger.info(f"📡 Found vLLM serving {model_id} at {url} (pod {name})")
        return VllmNode(name, url, True)
    logger.info(f"📡 Service {K8S_SERVICE} is up at {url} but vLLM is not answering yet.")
    return VllmNode(name, url, False)


async def discover_vllm_url() -> Optional[str]:
    """Finds the URL of the vLLM Service, if it is serving."""
    node = await _discover_vllm_node()
    return node.url if node and node.url else None


async def get_vllm_client() -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for the vLLM service."""
    url = await discover_vllm_url()
    if not url:
        raise Exception(f"No vLLM Service is answering on cluster `{GKE_CLUSTER_NAME}` in {GKE_LOCATION}.")
    return AsyncOpenAI(base_url=f"{url}/v1", api_key="not-needed")


# --- Provisioning -----------------------------------------------------------------------


@mcp.tool()
async def create_gke_cluster(
    cluster_name: str = GKE_CLUSTER_NAME,
    location: str = GKE_LOCATION,
    system_machine_type: str = GKE_SYSTEM_MACHINE_TYPE,
    release_channel: str = GKE_RELEASE_CHANNEL,
) -> str:
    """Creates the zonal GKE cluster that TPU node pools attach to. Idempotent.

    The cluster carries a small default pool for system workloads only; the chips arrive
    with `create_tpu_node_pool`. Takes roughly ten minutes and creates nothing billable
    beyond the system node and the cluster fee.
    """
    if await _cluster_exists(cluster_name, location):
        return f"✅ Cluster `{cluster_name}` already exists in {location}."

    cmd = [
        "gcloud",
        "container",
        "clusters",
        "create",
        cluster_name,
        f"--project={PROJECT_ID}",
        f"--location={location}",
        f"--release-channel={release_channel}",
        "--num-nodes=1",
        f"--machine-type={system_machine_type}",
        f"--disk-size={GKE_SYSTEM_DISK_SIZE_GB}",
        f"--labels=rig={RIG_NAME}",
    ]
    rc, out, err = await run_command(cmd, timeout=1200)
    if rc != 0:
        return f"❌ Cluster creation failed in {location}: {err.strip() or out.strip()}"
    return f"✅ Cluster `{cluster_name}` created in {location} ({release_channel} channel).\n\n{out.strip()}"


@mcp.tool()
async def create_tpu_node_pool(
    node_pool: str = GKE_NODE_POOL,
    cluster_name: str = GKE_CLUSTER_NAME,
    location: str = GKE_LOCATION,
    machine_type: str = MACHINE_TYPE,
    provisioning_model: str = GKE_NODE_PROVISIONING,
    num_nodes: str = GKE_NUM_NODES,
    reservation_name: Optional[str] = None,
) -> str:
    """Creates the TPU node pool — the chips. Idempotent, and non-destructive.

    provisioning_model is one of 'on-demand' (default), 'spot', 'flex-start' or
    'reservation-bound'; see `_node_pool_provisioning_flags` for what each spells.

    Note what is NOT here: no --tpu-topology for a single-host slice (the API refuses it),
    no image family, no boot disk image, no --max-run-duration. A node pool has no
    self-destruct, so nothing stops the bill except `destroy_tpu_node_pool`.
    """
    if provisioning_model not in PROVISIONING_MODELS:
        return f"❌ Aborted: unknown provisioning_model '{provisioning_model}'. Use one of {PROVISIONING_MODELS}."
    if not await _cluster_exists(cluster_name, location):
        return f"❌ No cluster `{cluster_name}` in {location}. Call `create_gke_cluster` first."
    if await _node_pool_exists(node_pool, cluster_name, location):
        return f"✅ Node pool `{node_pool}` already exists in cluster `{cluster_name}` ({location})."

    cmd = [
        "gcloud",
        "container",
        "node-pools",
        "create",
        node_pool,
        f"--project={PROJECT_ID}",
        f"--location={location}",
        f"--cluster={cluster_name}",
        f"--node-locations={location}",
        f"--machine-type={machine_type}",
        *_topology_flags(),
        *_node_pool_provisioning_flags(provisioning_model, num_nodes, reservation_name),
        f"--disk-size={GKE_NODE_DISK_SIZE_GB}",
    ]
    rc, out, err = await run_command(cmd, timeout=1200)
    if rc != 0:
        return f"❌ Node pool creation failed in {location}: {err.strip() or out.strip()}"
    return (
        f"✅ Node pool `{node_pool}` created in `{cluster_name}` ({location}): "
        f"{num_nodes}x {machine_type}, {provisioning_model}.\n\n{out.strip()}"
    )


@mcp.tool()
async def provision_gke_tpu(
    cluster_name: str = GKE_CLUSTER_NAME,
    location: str = GKE_LOCATION,
    provisioning_model: str = GKE_NODE_PROVISIONING,
) -> str:
    """Cluster plus TPU node pool in one call — everything before the model. Idempotent."""
    cluster_result = await create_gke_cluster(cluster_name=cluster_name, location=location)
    if cluster_result.startswith("❌"):
        return cluster_result
    pool_result = await create_tpu_node_pool(
        cluster_name=cluster_name, location=location, provisioning_model=provisioning_model
    )
    next_step = "Call `deploy_vllm` to start the model." if not pool_result.startswith("❌") else ""
    return f"{cluster_result}\n{pool_result}\n\n👉 {next_step}".rstrip()


@mcp.tool()
async def deploy_vllm(
    cluster_name: str = GKE_CLUSTER_NAME,
    location: str = GKE_LOCATION,
    model_name: str = MODEL_NAME,
    service_type: str = GKE_SERVICE_TYPE,
) -> str:
    """Deploys vLLM onto the TPU node pool: HF token Secret, then Deployment + Service.

    The token is read from Secret Manager and written into a Kubernetes Secret through a
    0600 temp file — never as a --from-literal argument, which would put it in the process
    table for every user on the machine.
    """
    ok, err = await _ensure_cluster_credentials(cluster_name, location)
    if not ok:
        return err

    token = await get_secret(HF_SECRET_ID)
    if not token:
        return f"❌ Aborted: Secret Manager secret `{HF_SECRET_ID}` is missing or unreadable. Use `save_hf_token`."

    secret_manifest = (
        "apiVersion: v1\nkind: Secret\nmetadata:\n"
        f"  name: {HF_K8S_SECRET}\ntype: Opaque\ndata:\n"
        f"  token: {base64.b64encode(token.encode()).decode()}\n"
    )
    ok, detail = await _kubectl_apply(secret_manifest, cluster_name, location)
    if not ok:
        return f"❌ Could not create the `{HF_K8S_SECRET}` Secret: {detail}"

    try:
        manifest = _render_vllm_manifest(model_name=model_name, service_type=service_type)
    except (OSError, KeyError, ValueError) as e:
        return f"❌ Could not render {MANIFEST_TEMPLATE}: {e}"
    ok, detail = await _kubectl_apply(manifest, cluster_name, location)
    if not ok:
        return f"❌ Could not apply the vLLM manifest: {detail}"

    return (
        f"🚀 Deployed `{K8S_DEPLOYMENT}` to cluster `{cluster_name}` ({location}).\n"
        f"- model: `{model_name}`, max-model-len {MAX_MODEL_LEN}, TP {TENSOR_PARALLEL_SIZE}\n"
        f"- image: `{VLLM_IMAGE}`, service type `{service_type}`\n\n{detail}\n\n"
        "👉 The image pull, weight load and XLA precompile take about ten minutes. Poll with "
        "`get_system_status`, watch `get_vllm_pod_logs`, and use `verify_model_health` for readiness — "
        "a Running pod is not a served model."
    )


@mcp.tool()
async def manage_vllm_deployment(action: str = "restart", replicas: int = 1) -> str:
    """Restarts, scales, or deletes the vLLM Deployment.

    action: 'restart' (rolling restart — re-pulls nothing, but re-runs the whole ten-minute
    load), 'scale' (to `replicas`; more than one needs more than one TPU node), 'delete'
    (removes the Deployment and Service, leaves the node pool and its bill running), or
    'status'.
    """
    ok, err = await _ensure_cluster_credentials()
    if not ok:
        return err
    if action == "status":
        return await get_system_status()
    if action == "restart":
        args = ["rollout", "restart", f"deployment/{K8S_DEPLOYMENT}"]
    elif action == "scale":
        args = ["scale", f"deployment/{K8S_DEPLOYMENT}", f"--replicas={int(replicas)}"]
    elif action == "delete":
        args = ["delete", "deployment,service", "-l", f"app={K8S_DEPLOYMENT}"]
    else:
        return f"❌ Unknown action '{action}'. Use restart, scale, delete or status."

    rc, out, err_out = await run_command(_kubectl(args), timeout=180)
    if rc != 0:
        return f"❌ `kubectl {' '.join(args)}` failed: {err_out.strip() or out.strip()}"
    suffix = ""
    if action == "delete":
        suffix = "\n\n⚠️ The node pool is still running and still billing — `destroy_tpu_node_pool` stops that."
    return f"✅ {out.strip()}{suffix}"


@mcp.tool()
async def get_vllm_pod_logs(tail: int = 200, previous: bool = False) -> str:
    """Logs from the vLLM pod — the GKE equivalent of the twin rig's docker logs.

    previous=True reads the last terminated container, which is the only place a crash
    that happened before the current restart is still visible.
    """
    ok, err = await _ensure_cluster_credentials()
    if not ok:
        return err
    pod = await _vllm_pod()
    if not pod:
        return f"❌ No pod for `{K8S_DEPLOYMENT}` on cluster `{GKE_CLUSTER_NAME}`. Has `deploy_vllm` run?"
    name = (pod.get("metadata") or {}).get("name", "")
    args = ["logs", name, f"--tail={int(tail)}"]
    if previous:
        args.append("--previous")
    rc, out, err_out = await run_command(_kubectl(args), timeout=120)
    if rc != 0:
        return f"⚠️ Could not read logs from `{name}`: {err_out.strip()}"
    return f"✅ Logs from `{name}` (last {tail} lines):\n{out}"


@mcp.tool()
async def get_system_status() -> str:
    """Dashboard: cluster, TPU nodes, the vLLM pod, the Service, and what to do next."""
    if not await _cluster_exists():
        return (
            f"### 🌀 System Status ({GKE_LOCATION})\n"
            f"- **Cluster:** ❌ `{GKE_CLUSTER_NAME}` does not exist\n"
            "**👉 Next Step:** call `provision_gke_tpu` to create the cluster and a TPU node pool."
        )
    ok, err = await _ensure_cluster_credentials()
    if not ok:
        return err

    pool_exists = await _node_pool_exists()
    nodes = await _kubectl_json(["get", "nodes", "-l", f"cloud.google.com/gke-tpu-accelerator={GKE_TPU_ACCELERATOR}"])
    node_rows = []
    for node in (nodes or {}).get("items") or []:
        meta = node.get("metadata") or {}
        alloc = (node.get("status") or {}).get("allocatable") or {}
        conditions = (node.get("status") or {}).get("conditions") or []
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
        node_rows.append(
            f"- `{meta.get('name')}` — {'Ready' if ready else 'NotReady'}, "
            f"google.com/tpu: {alloc.get('google.com/tpu', '0')}, "
            f"pool `{(meta.get('labels') or {}).get('cloud.google.com/gke-nodepool', '?')}`"
        )
    nodes_str = "**🖥️ TPU nodes:**\n" + ("\n".join(node_rows) if node_rows else "- none")

    pod = await _vllm_pod()
    pod_str = f"**📦 vLLM pod:** {_pod_summary(pod) if pod else 'none deployed'}"

    url = await _service_endpoint()
    if url:
        model_id = await _probe_vllm(url)
        health = f"🟢 Online at {url} (serving `{model_id}`)" if model_id else f"🟡 {url} up, vLLM not answering yet"
    else:
        health = "🔴 No external Service address"

    if url and "🟢" in health:
        next_step = "Use `query_queued_gemma4` to interact with the model."
    elif pod:
        next_step = "The pod is still loading — watch `get_vllm_pod_logs`."
    elif pool_exists:
        next_step = "Node pool is up with no model on it — call `deploy_vllm`."
    else:
        next_step = "Cluster exists but has no TPU node pool — call `create_tpu_node_pool`."

    return (
        f"### 🌀 System Status ({GKE_LOCATION})\n"
        f"- **Cluster:** ✅ `{GKE_CLUSTER_NAME}`\n"
        f"- **TPU node pool:** {'✅ ' + GKE_NODE_POOL if pool_exists else '❌ none'}\n"
        f"- **vLLM Health:** {health}\n"
        f"{nodes_str}\n{pod_str}\n**👉 Next Step:** {next_step}"
    )


@mcp.tool()
async def list_tpu_node_pools(cluster_name: str = GKE_CLUSTER_NAME, location: str = GKE_LOCATION) -> str:
    """Lists the node pools on this rig's cluster, with their machine types and sizes."""
    cmd = [
        "gcloud",
        "container",
        "node-pools",
        "list",
        f"--cluster={cluster_name}",
        f"--location={location}",
        f"--project={PROJECT_ID}",
        "--format=json",
    ]
    rc, out, err = await run_command(cmd, timeout=120)
    if rc != 0:
        return f"❌ Could not list node pools on `{cluster_name}` ({location}): {err.strip()}"
    try:
        pools = json.loads(out) if out else []
    except json.JSONDecodeError:
        return f"⚠️ Node pool list did not parse:\n{out[:1000]}"
    if not pools:
        return f"ℹ️ Cluster `{cluster_name}` has no node pools."
    rows = [
        f"- `{p.get('name')}` — {(p.get('config') or {}).get('machineType', '?')}, "
        f"{p.get('initialNodeCount', '?')} node(s), {p.get('status', '?')}"
        for p in pools
    ]
    return f"**Node pools on `{cluster_name}` ({location}):**\n" + "\n".join(rows)


@mcp.tool()
async def destroy_tpu_node_pool(
    node_pool: str = GKE_NODE_POOL, cluster_name: str = GKE_CLUSTER_NAME, location: str = GKE_LOCATION
) -> str:
    """Deletes the TPU node pool — the only thing that stops the chip bill.

    Deliberately narrower than deleting the cluster: the cluster and its system pool
    survive, so a redeploy is one `create_tpu_node_pool` away. Flex-start and spot capacity
    can take a long time to come back, so do not call this speculatively.
    """
    cmd = [
        "gcloud",
        "container",
        "node-pools",
        "delete",
        node_pool,
        f"--cluster={cluster_name}",
        f"--location={location}",
        f"--project={PROJECT_ID}",
        "--quiet",
    ]
    rc, out, err = await run_command(cmd, timeout=900)
    if rc != 0:
        return f"❌ Could not delete node pool `{node_pool}`: {err.strip() or out.strip()}"
    return f"🗑️ Node pool `{node_pool}` deleted from `{cluster_name}` ({location}). The chip is released."


@mcp.tool()
async def destroy_gke_cluster(cluster_name: str = GKE_CLUSTER_NAME, location: str = GKE_LOCATION) -> str:
    """Deletes the whole cluster, including every node pool on it.

    Only ever deletes the named cluster. It does not sweep the project for others, which is
    the same split the twin rig keeps between `create_tpu_instance` and its teardown: one
    rig's tools must not remove a sibling's capacity.
    """
    cmd = [
        "gcloud",
        "container",
        "clusters",
        "delete",
        cluster_name,
        f"--location={location}",
        f"--project={PROJECT_ID}",
        "--quiet",
    ]
    rc, out, err = await run_command(cmd, timeout=1200)
    if rc != 0:
        return f"❌ Could not delete cluster `{cluster_name}`: {err.strip() or out.strip()}"
    return f"🗑️ Cluster `{cluster_name}` deleted from {location}."


@mcp.tool()
async def get_vllm_deployment_config(model_name: str = MODEL_NAME, service_type: str = GKE_SERVICE_TYPE) -> str:
    """Prints the exact Deployment + Service YAML these tools apply, for review or `kubectl apply` by hand."""
    try:
        manifest = _render_vllm_manifest(model_name=model_name, service_type=service_type)
    except (OSError, KeyError, ValueError) as e:
        return f"❌ Could not render {MANIFEST_TEMPLATE}: {e}"
    return f"**Rendered manifest** (`{MANIFEST_TEMPLATE}`):\n```yaml\n{manifest}\n```"


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
    topology: str = "1x1",
    provisioning_model: str = PROVISIONING_MODEL,
    region: str = REGION,
) -> str:
    """Estimates the cost of a TPU deployment from live Google Cloud published pricing.

    Rates come from the Cloud Billing Catalog API, not a table in this file — an earlier
    hardcoded table was wrong by 10x and there is no way to notice that from inside the
    code. If the catalog has no matching SKU this reports that instead of guessing.

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
    # EVERY model bills until the node pool is deleted. The twin rig can say that flex-start
    # "self-terminates at --max-run-duration", because a Compute Engine instance carries one.
    # A node pool does not: there is no run bound and no termination action on this path, so
    # repeating the twin's caveat here would be a confident wrong statement about money.
    if provisioning_model == "spot":
        lines.append("- ⚠️ Spot can be preempted mid-run, so billed hours may be shorter than requested.")
    elif provisioning_model == "flex-start":
        lines.append(
            "- ⚠️ Flex-start caps how long capacity is *granted*, not how long you are billed: a node pool "
            "has no `--max-run-duration`. It bills until `destroy_tpu_node_pool`."
        )
    else:
        lines.append("- ⚠️ On-demand has no run bound — this bills until `destroy_tpu_node_pool`.")
    lines.append(f"- Not counted here: the `{GKE_SYSTEM_MACHINE_TYPE}` system node and the GKE cluster management fee.")
    lines.append("_List price from the Cloud Billing Catalog; committed-use or negotiated discounts are not applied._")
    return "\n".join(lines)


@mcp.tool()
async def get_vllm_endpoint() -> str:
    """Returns the active vLLM service URL if available."""
    url = await discover_vllm_url()
    if url:
        return f"🟢 vLLM is Online at: {url}"
    return (
        f"❌ No vLLM Service is answering on cluster `{GKE_CLUSTER_NAME}` in {GKE_LOCATION}. Check `get_system_status`."
    )


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
async def find_tpu(
    node_pool: str = GKE_NODE_POOL,
    cluster_name: str = GKE_CLUSTER_NAME,
    provisioning_model: str = GKE_NODE_PROVISIONING,
    machine_type: str = MACHINE_TYPE,
    create_cluster_if_missing: bool = False,
) -> str:
    """Sweeps zones for TPU capacity and creates the node pool in the first that gives it.

    **A zone sweep costs more here than on either sibling path, and the tool is shaped
    around that.** A Queued Resource or a Compute Engine instance can be attempted in any
    zone for free; a node pool can only be created inside a cluster, and a cluster is
    zonal — so attempting zone N means having a cluster in zone N, which is ten minutes and
    a standing management fee. Hence `create_cluster_if_missing`, which is False by default:
    the sweep then only attempts zones where this rig already has a cluster, and reports the
    rest as candidates. Pass True when actually hunting capacity; clusters this call creates
    are deleted again if their node pool fails, so a failed sweep leaves nothing behind.

    Gate 1 is still `machine-types list` rather than quota, for the same reason as the
    Compute Engine twin: both TPU quotas on this path are regional and cannot produce a zone
    list. Quota is a ceiling, not an allocation — a zone with 1536 chips of quota and no
    hardware fails exactly like a zone with none.
    """
    if provisioning_model not in PROVISIONING_MODELS:
        return f"❌ Aborted: unknown provisioning_model '{provisioning_model}'. Use one of {PROVISIONING_MODELS}."

    zones = await _zones_with_machine_type(machine_type)
    if not zones:
        return f"❌ Aborted: no zone publishes machine type `{machine_type}`."
    # The configured location first: it is the zone most likely to already hold a cluster.
    zones = sorted(zones, key=lambda z: (z != GKE_LOCATION, z))
    logger.info(f"Zones offering {machine_type}: {zones}")

    skipped_zones = set()
    status_file = os.path.join(os.path.dirname(__file__), "tpu_zones_status.md")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                content = f.read()
            for line in content.splitlines():
                match = re.search(r"\|\s*\*\*([a-zA-Z0-9-]+)\*\*\s*\|\s*([^|]+)\|\s*No\s*\|([^|]*)\|", line)
                if match and _status_model(match.group(3)) == provisioning_model:
                    skipped_zones.add(match.group(1).strip())
            logger.info(f"Skipping {provisioning_model} zones marked failed: {sorted(skipped_zones)}")
        except Exception as e:
            logger.error(f"Error parsing status file: {e}")

    attempts = []
    for zone in zones:
        if zone in skipped_zones:
            attempts.append(f"- **Zone {zone}**: ⏭️ Skipped (previously failed according to status file)")
            continue

        created_cluster_here = False
        if not await _cluster_exists(cluster_name, zone):
            if not create_cluster_if_missing:
                attempts.append(
                    f"- **Zone {zone}**: ⏭️ No cluster here (pass create_cluster_if_missing=True to build one)"
                )
                continue
            logger.info(f"Creating cluster {cluster_name} in {zone}...")
            cluster_result = await create_gke_cluster(cluster_name=cluster_name, location=zone)
            if cluster_result.startswith("❌"):
                attempts.append(f"- **Zone {zone}**: {cluster_result}")
                continue
            created_cluster_here = True

        logger.info(f"Attempting {provisioning_model} node pool {node_pool} in {zone}...")
        result = await create_tpu_node_pool(
            node_pool=node_pool, cluster_name=cluster_name, location=zone, provisioning_model=provisioning_model
        )
        if result.startswith("❌"):
            attempts.append(f"- **Zone {zone}**: {result}")
            reason = result.replace("❌ Node pool creation failed", "").strip(" :in")
            await _update_status_file(zone, "No", f"[{provisioning_model}] {reason}")
            if created_cluster_here:
                # Leave nothing behind: this cluster exists only because the sweep made it.
                logger.info(f"Node pool failed in {zone}; deleting the cluster this sweep created.")
                await destroy_gke_cluster(cluster_name=cluster_name, location=zone)
            continue

        await _update_status_file(zone, "Yes", f"[{provisioning_model}] Node pool created.")
        attempts.append(f"- **Zone {zone}**: ✅ {result.splitlines()[0]}")
        global GKE_LOCATION
        GKE_LOCATION = zone
        return (
            f"✅ Secured TPU capacity in zone `{zone}`!\n\n"
            f"**Node pool:**\n{result}\n\n"
            f"**Attempts Log:**\n" + "\n".join(attempts) + "\n\n👉 Call `deploy_vllm` to start the model."
        )

    return "❌ No zone gave TPU capacity. Attempted zones:\n" + "\n".join(attempts)


@mcp.tool()
async def run_vllm_benchmark(
    backend: str = "vllm",
    model: str = MODEL_NAME,
    dataset_name: str = "random",
    num_prompts: int = 100,
    random_input_len: int = 1024,
    random_output_len: int = 128,
    max_concurrency: Optional[int] = None,
    save_result: bool = False,
) -> str:
    """Runs vLLM's own benchmark against the deployed model, from inside the pod.

    The twin rig starts a SECOND container beside the serving one over SSH. There is no
    SSH here and no second container: the benchmark runs with `kubectl exec` in the serving
    pod itself, against localhost. The load is TPU-bound and the client is not, so sharing
    the pod's CPU with the server does not distort the numbers the way sharing a chip would.

    With save_result=True the run's --save-result JSON comes back as a ready-made
    throughput.sweep[] entry for benchmarks/serving-report.schema.json, one call per point.
    """
    ok, err = await _ensure_cluster_credentials()
    if not ok:
        return err
    pod = await _vllm_pod()
    if not pod:
        return f"❌ No pod for `{K8S_DEPLOYMENT}` on cluster `{GKE_CLUSTER_NAME}`. Has `deploy_vllm` run?"
    pod_name = (pod.get("metadata") or {}).get("name", "")

    bench = (
        "vllm bench serve "
        f"--backend {backend} "
        f"--base-url http://localhost:8000 "
        f"--model {model} "
        f"--dataset-name {dataset_name} "
        f"--num-prompts {num_prompts} "
        f"--random-input-len {random_input_len} "
        f"--random-output-len {random_output_len}"
    )
    if max_concurrency:
        bench += f" --max-concurrency {max_concurrency}"

    result_name = f"vllm-bench-{os.urandom(4).hex()}.json"
    if save_result:
        bench += f" --save-result --result-dir /dev/shm --result-filename {result_name}"
        bench += f" && echo {BENCH_RESULT_MARKER} && cat /dev/shm/{result_name} && rm -f /dev/shm/{result_name}"

    args = ["exec", pod_name, "--", "sh", "-c", bench]
    rc, out, err_out = await run_command(_kubectl(args), timeout=900)
    if rc != 0:
        return f"⚠️ Benchmark failed in pod `{pod_name}`.\nError: {err_out}\nOutput: {out}"
    if not save_result:
        return f"✅ Benchmark completed in `{pod_name}`:\n{out}"

    bench_stdout, sep, result_json = out.partition(BENCH_RESULT_MARKER)
    if not sep:
        return f"⚠️ Benchmark ran in `{pod_name}` but no result JSON came back:\n{out}"
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        return f"⚠️ Benchmark ran in `{pod_name}` but the result JSON did not parse ({e}):\n{result_json.strip()[:2000]}"
    concurrency = int(max_concurrency) if max_concurrency else int(num_prompts)
    point = _sweep_point_from_bench_result(result, concurrency, int(random_input_len), int(random_output_len))
    return (
        f"✅ Benchmark completed in `{pod_name}`.\n\n"
        "throughput.sweep[] entry (benchmarks/serving-report.schema.json):\n"
        f"```json\n{json.dumps(point, indent=2)}\n```\n\n"
        f"Benchmark output:\n{bench_stdout.strip()}"
    )


@mcp.tool()
async def get_tpu_node_diagnostics(node_name: str = "") -> str:
    """Describes the TPU node and the pod's recent events — where a scheduling failure shows up.

    This replaces the twin rig's journalctl-over-SSH tool. The failures it is reached for
    are different in kind: not "the docker daemon died" but "the pod is Pending because
    nothing satisfies its nodeSelector", which never appears in a container log at all.
    """
    ok, err = await _ensure_cluster_credentials()
    if not ok:
        return err
    if not node_name:
        nodes = await _kubectl_json(
            ["get", "nodes", "-l", f"cloud.google.com/gke-tpu-accelerator={GKE_TPU_ACCELERATOR}"]
        )
        items = (nodes or {}).get("items") or []
        if not items:
            return f"❌ No node carries `cloud.google.com/gke-tpu-accelerator={GKE_TPU_ACCELERATOR}` on `{GKE_CLUSTER_NAME}`."
        node_name = ((items[0].get("metadata") or {}).get("name")) or ""

    rc, node_out, node_err = await run_command(_kubectl(["describe", "node", node_name]), timeout=120)
    if rc != 0:
        return f"⚠️ Could not describe node `{node_name}`: {node_err.strip()}"
    rc_e, events_out, _ = await run_command(
        _kubectl(["get", "events", "--sort-by=.lastTimestamp", "--field-selector", "type!=Normal"]), timeout=120
    )
    events = events_out.strip() if rc_e == 0 else "(events unavailable)"
    return f"**Node `{node_name}`:**\n```\n{node_out[-6000:]}\n```\n\n**Warning events:**\n```\n{events[-3000:]}\n```"


@mcp.tool()
async def get_cloud_logging_logs(
    log_filter: str = 'resource.type="k8s_container" resource.labels.container_name="vllm"', limit: int = 20
) -> str:
    """Fetches logs from Google Cloud Logging.

    The default filter is `k8s_container`, not the twin rig's `tpu_worker`: nothing this rig
    runs is a TPU-API worker, and a pod's stdout is exported under the container resource.
    """
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
        return f"❌ No vLLM Service is answering on cluster `{GKE_CLUSTER_NAME}` in {GKE_LOCATION}. Check `get_system_status`."

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
        f"- **`ACCELERATOR_TYPE`**: TPU Accelerator type.\n"
        f"  - *Current Value:* `{ACCELERATOR_TYPE}`\n"
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
        return f"❌ No vLLM Service is answering on cluster `{GKE_CLUSTER_NAME}` in {GKE_LOCATION}. Check `get_system_status`."

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
