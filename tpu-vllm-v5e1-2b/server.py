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
ZONE = os.getenv("GOOGLE_CLOUD_ZONE", "us-west4-a")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-west4")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-E2B-it")
# Secret Manager secret holding the Hugging Face token. The startup script fetches it by
# id at boot, so a rotated or per-project secret only needs this to change.
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "hf-token")
# Default Queued Resource id for every tool; its node is derived as <RESOURCE_ID>-node.
# Follows the directory, but tpu.env still wins — pin RESOURCE_ID there when a name has to
# outlive a rename, because a rename otherwise orphans whatever is already provisioned.
RESOURCE_ID = os.getenv("RESOURCE_ID", RIG_NAME)
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

ACCELERATOR_TYPE = os.getenv("ACCELERATOR_TYPE", "v5litepod-1")
# v5e is called "v5litepod" by the TPU API; the matching Flex-start runtime is v2-alpha-tpuv5-lite.
TPU_RUNTIME_VERSION = os.getenv("TPU_RUNTIME_VERSION", "v2-alpha-tpuv5-lite")
TPU_QUOTA_ID = os.getenv("TPU_QUOTA_ID", "TPUV5sLitepodPerProjectPerZoneForTPUAPI")
# Spot capacity is metered by its own preemptible quota, so a zone sweep for spot has to ask
# about this id instead of TPU_QUOTA_ID. Both ids confirmed present on the TPU API 2026-08-06.
TPU_SPOT_QUOTA_ID = os.getenv("TPU_SPOT_QUOTA_ID", "TPUV5sPreemptibleLitepodPerProjectPerZoneForTPUAPI")
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

# How a Queued Resource asks for capacity. flex-start is the historical default and the only
# model this rig has provisioned in production — note it is accepted for v5litepod-1 in
# us-west4-a only (see CLAUDE.md), which is the reason the other two exist.
PROVISIONING_MODELS = ("flex-start", "spot", "on-demand")
PROVISIONING_MODEL = os.getenv("PROVISIONING_MODEL", "flex-start")

# Cloud Billing Catalog service id for Compute Engine, which is where the TPU SKUs live.
COMPUTE_BILLING_SERVICE_ID = "6F81-5844-456A"
_SKU_CACHE: dict[str, list] = {}

# How each provisioning model is spelled in SKU descriptions, anchored at the start so the
# Reserved/Commitment/Capacity-Optimized variants of the same chip do not match. The catalog
# calls flex-start "DWS Defined Duration" (Dynamic Workload Scheduler) and drops the "Tpu"
# prefix there, and calls spot "Preemptible". Verified against the live catalog 2026-08-06.
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
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "16384")
MAX_NUM_BATCHED_TOKENS = os.getenv("MAX_NUM_BATCHED_TOKENS", "4096")
LIMIT_MM_PER_PROMPT = os.getenv("LIMIT_MM_PER_PROMPT", '{"image":4,"audio":1}')

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


async def _get_node_id(resource_id: str) -> Optional[str]:
    """Retrieves the node ID for a given Queued Resource."""
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        resource_id,
        f"--project={PROJECT_ID}",
        f"--zone={ZONE}",
        "--format=value(tpu.nodeSpec[0].nodeId)",
    ]
    rc, node_id, _ = await run_command(cmd)
    return node_id.strip() if rc == 0 and node_id else None


async def _get_node_ip(node_id: str) -> Optional[str]:
    """Gets the external or internal IP of a TPU node."""
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        node_id,
        f"--project={PROJECT_ID}",
        f"--zone={ZONE}",
        "--format=value(networkEndpoints[0].accessConfig.externalIp)",
    ]
    rc, ip, _ = await run_command(cmd)
    if rc == 0 and ip:
        return ip.strip()

    # Fallback to internal IP if external is not found
    cmd[-1] = "value(networkEndpoints[0].ipAddress)"
    rc, ip, _ = await run_command(cmd)
    return ip.strip() if rc == 0 and ip else None


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
        )
    except Exception as e:
        logger.error(f"Error formatting startup script: {e}")
        return f"""#!/bin/bash
echo 'Error loading template: {e}'"""


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


def _provisioning_flags(provisioning_model: str) -> list[str]:
    """The gcloud flags that select how a Queued Resource asks for capacity.

    Only flex-start passes --max-run-duration: gcloud documents that flag as flex-start
    only. A spot or on-demand node therefore has no automatic stop — it runs until it is
    preempted or destroyed, so it keeps billing after a demo ends.
    """
    if provisioning_model == "spot":
        return ["--spot", "--labels=purpose=spot"]
    if provisioning_model == "on-demand":
        return ["--labels=purpose=on-demand"]
    return ["--provisioning-model=flex-start", "--max-run-duration=4h", "--labels=purpose=flex-start"]


def _quota_id_for(provisioning_model: str) -> str:
    """Spot draws on the separate preemptible quota; flex-start and on-demand share TPU_QUOTA_ID."""
    return TPU_SPOT_QUOTA_ID if provisioning_model == "spot" else TPU_QUOTA_ID


async def _list_queued_resources_json(zone: str) -> Optional[list]:
    """Lists the Queued Resources in a zone. Returns None if the gcloud call failed."""
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


async def _create_queued_resource(
    resource_id: str, zone: str, provisioning_model: str = PROVISIONING_MODEL
) -> tuple[bool, str]:
    """Renders the startup script and issues the queued-resources create call.

    The rendered script carries no secret — it reads 'hf-token' from Secret Manager at
    boot — but it is still written to a private temp file and removed afterwards.
    """
    if provisioning_model not in PROVISIONING_MODELS:
        return (
            False,
            f"❌ Aborted: unknown provisioning_model '{provisioning_model}'. Use one of {PROVISIONING_MODELS}.",
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
            "alpha",
            "compute",
            "tpus",
            "queued-resources",
            "create",
            resource_id,
            f"--zone={zone}",
            f"--runtime-version={TPU_RUNTIME_VERSION}",
            f"--node-id={resource_id}-node",
            # Bounds how long the *request* stays queued, independent of provisioning model.
            "--valid-until-duration=4h",
            f"--project={PROJECT_ID}",
            f"--accelerator-type={ACCELERATOR_TYPE}",
            f"--metadata-from-file=startup-script={script_file}",
            *_provisioning_flags(provisioning_model),
        ]
        if TPU_NETWORK:
            create_cmd.append(f"--network={TPU_NETWORK}")
        if TPU_SUBNETWORK:
            create_cmd.append(f"--subnetwork={TPU_SUBNETWORK}")

        logger.info(f"Executing gcloud command: {' '.join(shlex.quote(c) for c in create_cmd)}")
        rc, _, err = await run_command(create_cmd)
    finally:
        try:
            os.unlink(script_file)
        except OSError:
            pass

    if rc != 0:
        return False, f"❌ Creation failed: {err}"

    lifetime = {
        "flex-start": "Self-terminates after 4h.",
        "spot": "⚠️ Preemptible with ~30s notice and has no max-run-duration — destroy it when you are done.",
        "on-demand": "⚠️ No max-run-duration — it bills until you destroy it.",
    }[provisioning_model]
    return True, (
        f"🚀 Resource {resource_id} creation initiated in {zone} ({provisioning_model}) with startup script. {lifetime}"
    )


async def _list_tpu_vm_nodes(zone: str = ZONE) -> list[dict]:
    """Lists every TPU VM node in the zone, however it was provisioned.

    A Queued Resource creates a node in this same namespace (which is why _get_node_ip
    can `tpu-vm describe` it), so one call covers both provisioning paths — the QR path
    and a directly created node like the ones `make deploy-tpu-spot` makes.
    """
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "list",
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        "--format=json",
    ]
    rc, stdout, _ = await run_command(cmd)
    if rc != 0 or not stdout:
        return []
    try:
        nodes = json.loads(stdout)
    except json.JSONDecodeError as e:
        logger.error(f"Could not parse tpu-vm list: {e}")
        return []
    return nodes if isinstance(nodes, list) else []


def _node_ip(node: dict) -> Optional[str]:
    """External IP of a listed node, falling back to the internal one."""
    for endpoint in node.get("networkEndpoints") or []:
        ip = (endpoint.get("accessConfig") or {}).get("externalIp") or endpoint.get("ipAddress")
        if ip:
            return str(ip)
    return None


async def _active_qr_node_ids(zone: str = ZONE) -> set[str]:
    """Node ids belonging to an ACTIVE Queued Resource in the zone."""
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
    """Whether a node name is one this rig provisions (RESOURCE_ID or <RESOURCE_ID>-node)."""
    return node_name == RESOURCE_ID or node_name.startswith(f"{RESOURCE_ID}-")


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
    """Finds the TPU node serving vLLM in ZONE, whichever way it was provisioned.

    Both provisioning paths end at a TPU VM node, so this lists nodes rather than Queued
    Resources: the old QR-only sweep was blind to a directly created node, and read a
    perfectly healthy spot VM as "no TPU". Candidates are ranked with this rig's own names
    first and then probed on /v1/models; the first node actually answering wins. If none
    answers, a node of ours that is still booting is returned anyway (serving=False) so
    callers can poll it — but a node that is not ours is never returned unprobed, since
    sibling rigs share this zone and one of theirs is not ours to talk to.
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
            0 if node.get("state") == "READY" else 1,
            0 if name in qr_node_ids else 1,
            name,
        )
        candidates.append(_Candidate(sort_key, name, f"http://{ip}:8000", mine))
    candidates.sort(key=lambda c: c.sort_key)

    for cand in candidates:
        model_id = await _probe_vllm(cand.url)
        if model_id:
            origin = "queued resource" if cand.name in qr_node_ids else "standalone TPU VM"
            logger.info(f"📡 Found vLLM serving {model_id} on {cand.name} ({origin}) at {cand.url}")
            return VllmNode(cand.name, cand.url, True)

    for cand in candidates:
        if cand.mine:
            logger.info(f"📡 Node {cand.name} is up at {cand.url} but vLLM is not answering yet.")
            return VllmNode(cand.name, cand.url, False)

    logger.info(f"No TPU node in {ZONE} is serving vLLM.")
    return None


async def discover_vllm_url() -> Optional[str]:
    """Finds the URL of a running vLLM service in ZONE, Queued Resource or not."""
    node = await _discover_vllm_node()
    return node.url if node else None


async def _resolve_node_id(resource_id: str, zone: str = ZONE) -> Optional[str]:
    """Resolves an id the tools were given to an actual TPU VM node name.

    Tried in order: the node of a Queued Resource by that id; a TPU VM by that exact name
    or by <id>-node (a directly created node has no Queued Resource to describe); and last,
    the node currently serving vLLM, so the default resource_id still reaches a running
    deployment whose node was named by hand or by an earlier convention.
    """
    node_id = await _get_node_id(resource_id)
    if node_id:
        return node_id

    names = {(n.get("name") or "").split("/")[-1] for n in await _list_tpu_vm_nodes(zone)}
    for candidate in (resource_id, f"{resource_id}-node"):
        if candidate in names:
            logger.info(f"No queued resource {resource_id}; using standalone TPU VM {candidate}.")
            return candidate

    serving = await _discover_vllm_node()
    if serving and serving.serving:
        logger.info(f"No node named {resource_id}; falling back to {serving.name}, which is serving vLLM.")
        return serving.name
    return None


async def get_vllm_client() -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for the vLLM service."""
    url = await discover_vllm_url()
    if not url:
        raise Exception(f"No TPU node in {ZONE} is serving vLLM (checked queued resources and standalone TPU VMs).")
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
async def get_vllm_deployment_config(service_name: str = RESOURCE_ID, model_name: str = MODEL_NAME) -> str:
    """Generates the gcloud command for a single-host TPU v5e vLLM deployment."""
    # The token is read on the VM at runtime — never interpolated into the returned text.
    # The whole startup script is one single-quoted argument, so the JSON value has to be
    # double-quoted and backslash-escaped rather than single-quoted.
    escaped_mm = '"' + LIMIT_MM_PER_PROMPT.replace('"', '\\"') + '"'
    cmd = (
        f"gcloud alpha compute tpus tpu-vm create {service_name} \\\n"
        f"  --accelerator-type={ACCELERATOR_TYPE} \\\n"
        f"  --version={TPU_RUNTIME_VERSION} \\\n"
        f"  --zone={ZONE} \\\n"
        f"  --project={PROJECT_ID} \\\n"
        f"  --metadata=startup-script='#!/bin/bash\\n"
        f"docker run -t --rm --name vllm-gemma4 --privileged --net=host "
        f"-v /dev/shm:/dev/shm --shm-size 10gb "
        f"-e HF_HOME=/dev/shm "
        f"-e HF_TOKEN=$(gcloud secrets versions access latest --secret={HF_SECRET_ID}) "
        f"{VLLM_IMAGE} vllm serve {model_name} {_vllm_serve_flags(mm_limit=escaped_mm)}'"
    )
    return cmd


@mcp.tool()
async def get_vllm_tpu_deployment_config() -> str:
    """Generates GKE manifests for TPU-based deployments."""
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
      containers:
      - name: vllm-container
        image: vllm/vllm-tpu:nightly
        resources:
          limits:
            google.com/tpu: "{TENSOR_PARALLEL_SIZE}"
        env:
        - name: MODEL_NAME
          value: {MODEL_NAME}
"""
    return manifest


# --- MCP Tools ---


@mcp.tool()
async def destroy_queued_resource(resource_id: str, zone: str = ZONE) -> str:
    """Safely deletes a Queued Resource and its node."""
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "delete",
        resource_id,
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--async",
        "--quiet",
    ]
    rc, stdout, stderr = await run_command(cmd)
    if rc != 0:
        return f"❌ Failed to delete resource {resource_id}: {stderr}"
    return f"🗑️ Deletion of {resource_id} initiated: {stdout}"


@mcp.tool()
async def manage_queued_resource(
    resource_id: str = RESOURCE_ID, zone: str = ZONE, provisioning_model: str = PROVISIONING_MODEL
) -> str:
    """DESTRUCTIVE. Ensures the primary Queued Resource exists and DELETES every other one in the zone.

    Only call this when the user has explicitly asked to clean up a zone. To create a
    resource without touching anything else, use create_tpu_queued_resource.

    provisioning_model is only consulted if the primary has to be created: one of
    'flex-start' (default), 'spot', or 'on-demand'. An existing primary keeps whatever
    model it was created with — this tool does not convert one in place.
    """
    resources = await _list_queued_resources_json(zone)
    if resources is None:
        return "❌ Failed to list resources."

    redundant_deleted = []
    primary_res = None

    for res in resources:
        name = res.get("name", "").split("/")[-1]
        state = res.get("state", {}).get("state", "UNKNOWN")

        if name == resource_id:
            if state in ["FAILED", "SUSPENDED"]:
                logger.info(f"Primary resource {name} is {state}. Deleting to recreate.")
                await destroy_queued_resource(name, zone=zone)
                redundant_deleted.append(f"{name} (Failed)")
            else:
                primary_res = res
        else:
            logger.info(f"Deleting redundant resource: {name}")
            await destroy_queued_resource(name, zone=zone)
            redundant_deleted.append(name)

    if not primary_res:
        ok, msg = await _create_queued_resource(resource_id, zone, provisioning_model)
        return f"{msg} Cleaned up: {redundant_deleted}"

    state = primary_res.get("state", {}).get("state", "UNKNOWN")
    return f"✅ Primary resource {resource_id} is {state}. Cleaned up: {redundant_deleted}"


@mcp.tool()
async def create_tpu_queued_resource(
    resource_id: str = RESOURCE_ID, zone: str = ZONE, provisioning_model: str = PROVISIONING_MODEL
) -> str:
    """Creates a TPU Queued Resource in the given zone. Non-destructive.

    Other Queued Resources in the zone are left alone. If a resource with this exact id
    already exists it is reported and nothing is created; only a FAILED/SUSPENDED
    resource under that same id is deleted, so it can be recreated.

    provisioning_model is one of:
      * 'flex-start' (default) — queues for scarce capacity, self-terminates after 4h.
        Only accepted for v5litepod-1 in us-west4-a; other zones reject it at the API.
      * 'spot' — cheapest, draws on the separate preemptible quota, and can be reclaimed
        with ~30s notice. No max-run-duration, so destroy it when you are done.
      * 'on-demand' — standard capacity at full price, no preemption, no run bound.
    """
    resources = await _list_queued_resources_json(zone)
    if resources is None:
        return "❌ Failed to list resources."

    for res in resources:
        if res.get("name", "").split("/")[-1] != resource_id:
            continue
        state = res.get("state", {}).get("state", "UNKNOWN")
        if state not in ["FAILED", "SUSPENDED"]:
            return f"✅ Resource {resource_id} already exists in {zone} and is {state}. Nothing created."

        logger.info(f"Resource {resource_id} is {state}. Deleting it so it can be recreated.")
        await destroy_queued_resource(resource_id, zone=zone)
        _, msg = await _create_queued_resource(resource_id, zone, provisioning_model)
        return f"{msg} (replaced the previous {state} resource of the same id)"

    _, msg = await _create_queued_resource(resource_id, zone, provisioning_model)
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
    service: str = "tpu.googleapis.com",
    quota_id: Optional[str] = None,
    provisioning_model: str = PROVISIONING_MODEL,
) -> str:
    """
    Retrieves a list of GCP zones that have a non-zero quota for a specific metric.

    Non-zero quota does NOT mean the zone will accept the request — flex-start is refused
    for v5litepod-1 outside us-west4-a in zones that all report quota. Quota is a ceiling,
    not an offer of capacity.

    Args:
        service: The GCP service to query (defaults to 'tpu.googleapis.com').
        quota_id: The specific quota ID to filter by. Defaults to the one matching
            provisioning_model; pass a value to override.
        provisioning_model: 'flex-start' (default), 'spot', or 'on-demand'. Spot is metered
            by a separate preemptible quota, so this picks a different id.
    """
    quota_id = quota_id or _quota_id_for(provisioning_model)
    zones = await _get_zones_with_available_quota_list(service, quota_id)
    if not zones:
        return f"No zones/locations found with non-zero quota limit for `{quota_id}`."

    output = [f"### 📊 Available Zones with Quota for `{quota_id}`\n"]
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


@mcp.tool()
async def find_tpu(
    resource_id: str = RESOURCE_ID,
    service: str = "tpu.googleapis.com",
    quota_id: Optional[str] = None,
    provisioning_model: str = PROVISIONING_MODEL,
) -> str:
    """
    Finds a zone with available quota and attempts to create the TPU queued resource in that zone until successful.

    provisioning_model is one of 'flex-start' (default), 'spot', or 'on-demand', and also
    selects which quota the zone sweep reads — spot has its own preemptible quota. The
    sweep is non-destructive: it only touches the named resource_id.
    """
    if provisioning_model not in PROVISIONING_MODELS:
        return f"❌ Aborted: unknown provisioning_model '{provisioning_model}'. Use one of {PROVISIONING_MODELS}."

    quota_id = quota_id or _quota_id_for(provisioning_model)
    zones = await _get_zones_with_available_quota_list(service, quota_id)
    if not zones:
        return f"❌ Aborted: No zones found with non-zero quota for `{quota_id}`."

    # Parse flat status file to skip zones where TPU could not be started. A failure is only
    # evidence about the provisioning model that produced it — flex-start is refused outside
    # us-west4-a in zones where spot is fine — so only same-model failures are skipped.
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

    logger.info(f"Zones with available quota: {zones}")

    attempts = []
    for zone in zones:
        if zone in skipped_zones:
            logger.info(f"Skipping zone {zone} as it is marked as failed in status file.")
            attempts.append(f"- **Zone {zone}**: ⏭️ Skipped (previously failed according to status file)")
            continue

        logger.info(f"Attempting to create {provisioning_model} queued resource {resource_id} in zone {zone}...")
        result = await create_tpu_queued_resource(
            resource_id=resource_id, zone=zone, provisioning_model=provisioning_model
        )

        if result.startswith("❌"):
            attempts.append(f"- **Zone {zone}**: {result}")
            reason = result.replace("❌ Creation failed:", "").strip()
            await _update_status_file(zone, "No", f"[{provisioning_model}] {reason}")
            continue

        # Wait up to 3 minutes (180s) or 10 minutes (600s) if it becomes PROVISIONING
        logger.info(f"Waiting for queued resource {resource_id} in zone {zone} to become ACTIVE...")
        success = False
        poll_start = time.time()
        timeout = 180
        extended = False
        while time.time() - poll_start < timeout:
            await asyncio.sleep(15)
            state_cmd = [
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "queued-resources",
                "describe",
                resource_id,
                f"--zone={zone}",
                f"--project={PROJECT_ID}",
                "--format=value(state.state)",
            ]
            rc_s, stdout_s, stderr_s = await run_command(state_cmd)
            if rc_s == 0:
                current_state = stdout_s.strip()
                logger.info(f"Queued resource {resource_id} state in {zone}: {current_state}")
                if current_state == "ACTIVE":
                    success = True
                    break
                elif current_state == "PROVISIONING" and not extended:
                    logger.info("Resource is PROVISIONING. Extending timeout to 10 minutes (600 seconds) from start.")
                    timeout = 600
                    extended = True
                elif current_state in ["FAILED", "SUSPENDED"]:
                    logger.info(f"Queued resource {resource_id} reached failed/suspended state: {current_state}")
                    break
            else:
                logger.warning(f"Failed to check state: {stderr_s or stdout_s}")

        if success:
            await _update_status_file(
                zone, "Yes", f"[{provisioning_model}] Successfully started and reached ACTIVE state."
            )
            attempts.append(f"- **Zone {zone}**: ✅ Successfully created and reached ACTIVE state.")

            # Dynamically update global ZONE variable
            global ZONE
            ZONE = zone

            return (
                f"✅ Successfully initiated and secured TPU in zone `{zone}`!\n\n"
                f"**Creation Output:**\n{result}\n\n"
                f"**Attempts Log:**\n" + "\n".join(attempts)
            )
        else:
            logger.info(f"Timed out or failed waiting for TPU in {zone} to become ACTIVE. Deleting queued resource...")
            await destroy_queued_resource(resource_id, zone=zone)
            timeout_msg = (
                "Timed out waiting 10 minutes to reach ACTIVE state (reached PROVISIONING)."
                if extended
                else "Timed out waiting 3 minutes to reach ACTIVE state."
            )
            await _update_status_file(zone, "No", f"[{provisioning_model}] {timeout_msg}")
            attempts.append(f"- **Zone {zone}**: ❌ {timeout_msg}")

    return "❌ Failed to start TPU in any zone. Attempted zones:\n" + "\n".join(attempts)


@mcp.tool()
async def manage_vllm_docker(resource_id: str = RESOURCE_ID, action: str = "start") -> str:
    """Manages the vLLM Docker container on the TPU VM."""
    node_id = await _resolve_node_id(resource_id)
    if not node_id:
        return (
            f"❌ Could not find a TPU node for {resource_id} in {ZONE} "
            "(no ACTIVE queued resource and no standalone TPU VM by that name)."
        )

    # Same image and serve flags the boot-time startup script uses, so a container
    # recreated here matches what the queued resource originally booted with.
    docker_run_cmd = (
        f"sudo docker run --name vllm-gemma4 --privileged --net=host -d "
        f"-v /dev/shm:/dev/shm --shm-size 10gb "
        f"-e HF_HOME=/dev/shm -e HF_TOKEN=$(gcloud secrets versions access latest --secret={HF_SECRET_ID}) "
        f"{VLLM_IMAGE} vllm serve {MODEL_NAME} {_vllm_serve_flags()}"
    )

    commands = {
        "start": f"sudo docker start vllm-gemma4 || {docker_run_cmd}",
        "stop": "sudo docker stop vllm-gemma4",
        "restart": "sudo docker restart vllm-gemma4",
        "status": "sudo docker ps -a --filter name=vllm-gemma4",
        "log": "sudo docker logs --tail 100 vllm-gemma4",
        "rm": "sudo docker rm -f vllm-gemma4",
    }

    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        node_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command",
        commands.get(action, commands["status"]),
    ]

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Docker {action} failed, but reservation {resource_id} remains safe.
Error: {err}"""
    return f"""✅ Docker {action} command executed on {node_id}.
{out}"""


@mcp.tool()
async def list_queued_resources(zone: str = ZONE) -> str:
    """Lists all Queued Resources in a specific zone."""
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
        return f"""### 📋 Queued Resources in {zone}
```
{out}
```"""
    else:
        return f"❌ List failed: {err}"


@mcp.tool()
async def describe_queued_resource(resource_id: str = RESOURCE_ID, zone: str = ZONE) -> str:
    """Provides detailed information about a specific Queued Resource."""
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        resource_id,
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--format=json",
    ]
    rc, out, err = await run_command(cmd)
    if rc != 0:
        return f"❌ Describe failed: {err}"
    try:
        data = json.loads(out)
        state = data.get("state", {}).get("state", "UNKNOWN")
        node_id = data.get("tpu", {}).get("nodeSpec", [{}])[0].get("nodeId", "N/A")
        return (
            f"### 🔍 Detail: {resource_id}\n"
            f"- **State:** `{state}`\n"
            f"- **Node ID:** `{node_id}`\n"
            f"- **Full Data:**\n```json\n{json.dumps(data, indent=2)}\n```"
        )
    except Exception:
        return f"""### 🔍 Detail: {resource_id}
```
{out}
```"""


@mcp.tool()
async def get_reservation_status(resource_id: str = RESOURCE_ID) -> str:
    """Checks the lifecycle state and expiry time of a Queued Resource."""
    # This function can be simplified if `describe_queued_resource` is sufficient
    return await describe_queued_resource(resource_id)


@mcp.tool()
async def check_tpu_availability(resource_id: str) -> str:
    """Simple check to see if a Queued Resource has reached ACTIVE state."""
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        resource_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--format=value(state.state)",
    ]
    rc, state, err = await run_command(cmd)
    if rc != 0:
        return f"❌ Check failed: {err}"
    is_active = state.strip() == "ACTIVE"
    return (
        f"### 🧊 TPU Availability: {resource_id}\n"
        f"- **State:** `{state.strip()}`\n"
        f"- **Available:** {'✅ Yes' if is_active else '⏳ No'}"
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
    """
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
    tpu_type: str = "v5e",
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
    resources_str = await list_queued_resources()
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
            health = f"🟡 Node `{node.name}` up at {node.url}, vLLM not answering"

    # Queued resources alone do not describe the zone: a directly created node (make
    # deploy-tpu-spot, or anything provisioned by hand) has no queued resource behind it
    # and used to be missing from this dashboard entirely.
    nodes = await _list_tpu_vm_nodes()
    if nodes:
        rows = [
            f"- `{(n.get('name') or '').split('/')[-1]}` — {n.get('state', 'UNKNOWN')}, "
            f"{n.get('acceleratorType', 'unknown type')}, ip {_node_ip(n) or 'none'}"
            for n in nodes
        ]
        nodes_str = "**🖥️ TPU VM Nodes:**\n" + "\n".join(rows)
    else:
        nodes_str = "**🖥️ TPU VM Nodes:** none"

    if "🟢" in health:
        next_step = "Use `query_queued_gemma4` to interact with the model."
    elif node:
        next_step = "A node is up but vLLM is not answering — use `manage_vllm_docker` to start the service."
    elif "ACTIVE" in resources_str:
        next_step = "Use `manage_vllm_docker` to start the service on the ACTIVE resource."
    else:
        next_step = "Call `manage_queued_resource` to provision infrastructure."

    return (
        f"### 🌀 System Status ({ZONE})\n- **vLLM Health:** {health}\n"
        f"{resources_str}\n{nodes_str}\n**👉 Next Step:** {next_step}"
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

    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        node_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command",
        remote_cmd,
    ]

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
        return (
            f"❌ Could not find a TPU node for {resource_id} in {ZONE} "
            "(no ACTIVE queued resource and no standalone TPU VM by that name)."
        )

    log_cmd = "sudo docker logs vllm-gemma4"
    if tail:
        log_cmd += f" --tail {tail}"

    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        node_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command",
        log_cmd,
    ]

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
        return (
            f"❌ Could not find a TPU node for {resource_id} in {ZONE} "
            "(no ACTIVE queued resource and no standalone TPU VM by that name)."
        )

    log_cmd = f"journalctl -u {service} -n {tail or 100}"

    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        node_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command",
        log_cmd,
    ]

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
