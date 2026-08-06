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

# Setup logging
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vllm-devops-agent")

# Initialize FastMCP server. The name stays generic — model and topology come
# from env vars, so they don't belong in the server identity.
mcp = FastMCP("tpu-devops-agent")

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
ZONE = os.getenv("GOOGLE_CLOUD_ZONE", "europe-west4-a")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "europe-west4")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
HF_SECRET_ID = "hf-token"
ACCELERATOR_TYPE = os.getenv("ACCELERATOR_TYPE", "v6e-8")
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "8"))
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
    "jax numpy scipy ml_dtypes safetensors huggingface_hub transformers",
)
JAX_PIP_EXTRAS = os.getenv(
    "JAX_PIP_EXTRAS",
    "numpy scipy ml_dtypes safetensors huggingface_hub",
)

# find_tpu records per-zone provisioning outcomes here so later sweeps can skip
# zones that never delivered capacity. Lives outside the skill directory so
# reinstalls (`make skill`, project-setup.sh) don't wipe the learned state.
STATUS_FILE = os.path.join(os.path.expanduser("~"), ".cache", "tpu-devops", "tpu_zones_status.md")

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
    `find_tpu` moves the global ZONE to wherever capacity was found, follow-up
    calls without an explicit zone target the new zone (import-time defaults
    would keep pointing at the old one).
    """
    return zone or ZONE


async def _get_node_id(resource_id: str, zone: Optional[str] = None) -> Optional[str]:
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
        f"--zone={_zone(zone)}",
        "--format=value(tpu.nodeSpec[0].nodeId)",
    ]
    rc, node_id, _ = await run_command(cmd)
    return node_id.strip() if rc == 0 and node_id else None


async def _build_tpu_ssh_cmd(
    remote_cmd: str, resource_id: str, instance_name: Optional[str], zone: str
) -> tuple[Optional[list[str]], str]:
    """Builds the gcloud SSH argv for either serving host: a GCE flex-start TPU VM
    when instance_name is given, else the queued resource's node. Returns
    (argv, target_name); argv is None with an error message in target_name."""
    if instance_name:
        argv = [
            "gcloud",
            "compute",
            "ssh",
            instance_name,
            f"--zone={zone}",
            f"--project={PROJECT_ID}",
            "--command",
            remote_cmd,
        ]
        return argv, instance_name
    node_id = await _get_node_id(resource_id, zone)
    if not node_id:
        return None, (
            f"❌ Could not find node for resource {resource_id}. Ensure it is ACTIVE, "
            "or pass instance_name to target a GCE flex-start TPU VM instead."
        )
    argv = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        node_id,
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--command",
        remote_cmd,
    ]
    return argv, node_id


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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".claude", "skills", "tpu-management", "mcp"),
)


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
    raise RuntimeError(
        f"Startup script template '{filename}' not found. Searched: " + ", ".join(tried)
    )


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


def _write_startup_script(content: str) -> str:
    """Writes a startup script to a private (0600) temp file and returns its path.
    Callers must unlink it after the gcloud call completes."""
    fd, path = tempfile.mkstemp(prefix="tpu-startup-", suffix=".sh")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


async def discover_vllm_url() -> Optional[str]:
    """Finds the URL of an ACTIVE Queued Resource vLLM service."""
    list_cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "list",
        f"--project={PROJECT_ID}",
        f"--zone={ZONE}",
        "--format=json",
    ]
    rc, stdout, _ = await run_command(list_cmd)
    if rc != 0 or not stdout:
        return None

    try:
        resources = json.loads(stdout)
        for res in resources:
            if res.get("state", {}).get("state") == "ACTIVE":
                resource_id = res.get("name", "").split("/")[-1]
                node_id = await _get_node_id(resource_id)
                if node_id:
                    ip = await _get_node_ip(node_id)
                    if ip:
                        url = f"http://{ip}:8000"
                        logger.info(f"📡 Found ACTIVE Queued Resource {resource_id} at {url}")
                        return url
    except Exception as e:
        logger.error(f"Discovery error: {e}")

    # Fallback: RUNNING GCE flex-start TPU VM instances (the recommended v6e/v5p path).
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
                    logger.info(f"📡 Found RUNNING GCE TPU VM {name} at {url}")
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


@mcp.tool(title="Generate TPU VM deployment command", annotations=READ_ONLY)
async def get_vllm_deployment_config(service_name: str = "vllm-gemma4-qr", model_name: str = MODEL_NAME) -> str:
    """Generates the gcloud command for a single-host TPU v6e vLLM deployment.

    The startup script fetches the Hugging Face token from Secret Manager on the VM
    at boot, so the secret never appears in this output or in instance metadata.
    """
    startup_script = (
        "#!/bin/bash\n"
        f"HF_TOKEN=$(gcloud secrets versions access latest --secret={HF_SECRET_ID} --project={PROJECT_ID})\n"
        "docker run -t --rm --name vllm-gemma4 --privileged --net=host \\\n"
        "  -v /dev/shm:/dev/shm --shm-size 10gb \\\n"
        '  -e HF_TOKEN="$HF_TOKEN" \\\n'
        f"  vllm/vllm-tpu:nightly vllm serve {model_name} \\\n"
        f"  --max-model-len 16384 --tensor-parallel-size {TENSOR_PARALLEL_SIZE} --disable_chunked_mm_input"
    )
    cmd = (
        f"gcloud alpha compute tpus tpu-vm create {service_name} \\\n"
        f"  --accelerator-type={ACCELERATOR_TYPE} \\\n"
        f"  --version=v2-alpha-tpuv6e \\\n"
        f"  --zone={ZONE} \\\n"
        f"  --project={PROJECT_ID} \\\n"
        f"  --metadata=startup-script='{startup_script}'"
    )
    return (
        f"```bash\n{cmd}\n```\n"
        f"Requires the `{HF_SECRET_ID}` secret (save one with `save_hf_token`) and "
        "`roles/secretmanager.secretAccessor` on the VM's service account."
    )


# --- MCP Tools ---


@mcp.tool(title="Destroy queued resource", annotations=DESTRUCTIVE)
async def destroy_queued_resource(resource_id: str, zone: Optional[str] = None) -> str:
    """Safely deletes a Queued Resource and its node. Zone defaults to the server's
    current zone (which `find_tpu` updates when it secures capacity elsewhere)."""
    zone = _zone(zone)
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


async def _create_queued_resource(
    resource_id: str,
    zone: str,
    reserved: bool,
    model_name: Optional[str],
) -> str:
    """Creates a single Queued Resource with the vLLM startup script. Non-destructive:
    never touches other resources."""
    selected_model = model_name or MODEL_NAME
    try:
        accel_chips = int(ACCELERATOR_TYPE.rsplit("-", 1)[-1])
    except ValueError:
        accel_chips = None
    min_chips = _min_chips_for_model(selected_model)
    if accel_chips is not None and accel_chips < min_chips:
        return (
            f"❌ `{selected_model}` needs ~{min_chips} chips but `{ACCELERATOR_TYPE}` has {accel_chips} — "
            "it would OOM after a long load. Pick a larger accelerator or a smaller model."
        )

    token = await get_secret()
    if not token:
        return "❌ Aborted: 'hf-token' secret missing. Save one with `save_hf_token` first."
    try:
        startup_script_content = _get_formatted_startup_script(selected_model, zone)
    except RuntimeError as e:
        return f"❌ Aborted: {e}"
    script_file = _write_startup_script(startup_script_content)

    create_cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "create",
        resource_id,
        f"--zone={zone}",
        "--runtime-version=v2-alpha-tpuv6e",
        f"--node-id={resource_id}-node",
        f"--project={PROJECT_ID}",
        f"--accelerator-type={ACCELERATOR_TYPE}",
        f"--metadata-from-file=startup-script={script_file}",
    ]
    if reserved:
        create_cmd.append("--reserved")
    else:
        create_cmd.extend([
            "--provisioning-model=flex-start",
            "--max-run-duration=4h",
            "--valid-until-duration=4h",
            "--labels=purpose=flex-start",
        ])

    logger.info(f"Executing gcloud command: {' '.join(shlex.quote(c) for c in create_cmd)}")
    try:
        rc, _, err = await run_command(create_cmd)
    finally:
        try:
            os.unlink(script_file)
        except OSError:
            pass

    if rc != 0:
        return f"❌ Creation failed: {err}"
    return f"🚀 Queued Resource {resource_id} creation initiated in {zone} with startup script."


@mcp.tool(title="Manage primary queued resource (deletes others)", annotations=DESTRUCTIVE)
async def manage_queued_resource(
    resource_id: str = "vllm-gemma4-qr",
    zone: Optional[str] = None,
    reserved: bool = False,
    model_name: Optional[str] = None,
) -> str:
    """Ensures the primary Queued Resource exists — and DELETES every other queued
    resource in the zone (plus the primary itself if FAILED/SUSPENDED, so it can be
    recreated). Destructive by design; confirm with the user before running it in a
    zone that may hold resources they want kept. To create without any cleanup, use
    `create_tpu_queued_resource`."""
    zone = _zone(zone)
    list_cmd = [
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
    rc, stdout, stderr = await run_command(list_cmd)
    if rc != 0:
        return f"❌ Failed to list resources: {stderr}"

    try:
        resources = json.loads(stdout)
    except Exception:
        resources = []

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
        result = await _create_queued_resource(resource_id, zone, reserved, model_name)
        return f"{result} Cleaned up: {redundant_deleted}"

    state = primary_res.get("state", {}).get("state", "UNKNOWN")
    return f"✅ Primary resource {resource_id} is {state}. Cleaned up: {redundant_deleted}"


@mcp.tool(title="Create queued resource", annotations=WRITE)
async def create_tpu_queued_resource(
    resource_id: str = "vllm-gemma4-qr",
    zone: Optional[str] = None,
    reserved: bool = False,
    model_name: Optional[str] = None,
) -> str:
    """Creates a TPU Queued Resource (Flex-start or reserved) with the vLLM startup
    script. Non-destructive: unlike `manage_queued_resource`, it never deletes other
    resources — if one with this ID already exists it just reports its state."""
    zone = _zone(zone)
    describe_cmd = [
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
    rc, state, _ = await run_command(describe_cmd)
    if rc == 0 and state.strip():
        return (
            f"✅ Queued Resource {resource_id} already exists in {zone} (state: {state.strip()}). "
            "Delete it first with `destroy_queued_resource` to recreate."
        )
    return await _create_queued_resource(resource_id, zone, reserved, model_name)


# --- GCE flex-start TPU VM instances (recommended path for v6e / v5p) ---------------
# Google is migrating TPU provisioning to GCE machine types; the queued-resources API
# is legacy (v5e-only going forward). These tools manage TPU VMs created via
# `gcloud compute instances create --provisioning-model=FLEX_START`.

# accelerator -> (GCE machine type, chips)
_GCE_MACHINE_TYPES = {
    "v6e-1": ("ct6e-standard-1t", 1),
    "v6e-4": ("ct6e-standard-4t", 4),
    "v6e-8": ("ct6e-standard-8t", 8),
    "v5p-8": ("ct5p-hightpu-4t", 4),
}
_GCE_IMAGE_FLAGS = [
    "--image-project=ubuntu-os-accelerator-images",
    "--image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
]
_GCE_TPU_FILTER = "machineType~'ct6e|ct5p'"


def _min_chips_for_model(model: str) -> int:
    """Rough single-host chip floor for full-precision checkpoints (~32GB HBM per
    v6e chip). Prevents the expensive failure mode of a big model on a small
    accelerator: VM boot + image pull + ~10 min of loading before an inevitable OOM.
    Quantized variants (QAT/int4/AWQ/GPTQ/FP8) are ~4x smaller — never blocked."""
    lowered = model.lower()
    if any(q in lowered for q in ("qat", "int4", "q4", "awq", "gptq", "fp8")):
        return 1
    for marker, chips in (("31B", 4), ("26B", 4)):
        if marker in model:
            return chips
    return 1  # 12B and below fit on a single chip


@mcp.tool(title="Create flex-start TPU VM", annotations=WRITE)
async def create_tpu_vm_instance(
    instance_name: Optional[str] = None,
    zone: Optional[str] = None,
    accelerator: Annotated[str, Field(description="TPU type: v6e-1, v6e-4, v6e-8, or v5p-8")] = ACCELERATOR_TYPE,
    model_name: Optional[str] = None,
    boot_disk_size_gb: int = 200,
    max_run_duration: str = "4h",
    request_valid_for: str = "2h",
    workload: Annotated[
        Literal["vllm", "jax"],
        Field(description="'vllm' serves model_name via docker; 'jax' installs a bare jax[tpu] dev environment (no docker, no HF token)"),
    ] = "vllm",
) -> str:
    """Creates a flex-start TPU VM as a GCE instance (recommended path for v6e/v5p).
    workload='vllm' auto-starts Gemma 4 serving via the vLLM startup script;
    workload='jax' instead installs a bare jax[tpu] environment and asserts the TPU
    is visible — no docker, no HF token, much faster boot.
    Boot disk defaults to 200GB because the image default (10GB) cannot hold the
    vLLM TPU image."""
    zone = _zone(zone)
    instance_name = instance_name or ("jax-tpu-vm" if workload == "jax" else "vllm-gemma4-vm")
    if accelerator not in _GCE_MACHINE_TYPES:
        supported = ", ".join(sorted(_GCE_MACHINE_TYPES))
        return f"❌ Unsupported accelerator '{accelerator}'. Supported: {supported}"
    machine_type, chips = _GCE_MACHINE_TYPES[accelerator]

    selected_model = model_name or MODEL_NAME
    if workload == "jax":
        # No model to size and no gated weights to fetch: skip the chip-count
        # check and the HF token requirement, neither of which applies.
        try:
            startup_script_content = _get_jax_startup_script(zone)
        except RuntimeError as e:
            return f"❌ Aborted: {e}"
    else:
        min_chips = _min_chips_for_model(selected_model)
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
        "--provisioning-model=FLEX_START",
        f"--request-valid-for-duration={request_valid_for}",
        f"--max-run-duration={max_run_duration}",
        "--instance-termination-action=DELETE",
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
                f"⏳ gcloud gave up after ~10 min, but the flex-start request for `{instance_name}` "
                f"(valid for {request_valid_for}) may still be PENDING server-side — the VM can still "
                f"appear and bill later. Check with `list_tpu_vm_instances`; if you no longer want it, "
                f"delete it with `destroy_tpu_vm_instance` once it appears."
            )
        hint = ""
        if "TPUS_PER_TPU_FAMILY" in stderr:
            hint = (
                f" (per-region TPU family quota is 0 in {zone} — "
                "find alternatives with `get_zones_with_available_quota`)"
            )
        return f"❌ Creation failed: {stderr}{hint}"
    if workload == "jax":
        return (
            f"🚀 Flex-start TPU VM `{instance_name}` ({machine_type}, {chips} chip(s)) created in {zone}; "
            f"installing Python {JAX_PYTHON_VERSION} + `{JAX_PIP_SPEC}`. Follow with `wait_for_jax_ready` "
            f"or `get_tpu_vm_serial_log`, then `verify_jax_tpu`; the VM self-deletes at "
            f"max-run-duration ({max_run_duration}).\n{stdout}"
        )
    return (
        f"🚀 Flex-start TPU VM `{instance_name}` ({machine_type}, {chips} chip(s)) created in {zone}; "
        f"vLLM is starting `{selected_model}` (tp={chips}). Model load can take ~10 min — follow progress "
        f"with `get_tpu_vm_serial_log` and note the VM self-deletes at max-run-duration ({max_run_duration}).\n{stdout}"
    )


@mcp.tool(title="List TPU VM instances", annotations=READ_ONLY)
async def list_tpu_vm_instances(zone: Optional[str] = None) -> str:
    """Lists GCE TPU VM instances (ct6e/ct5p machine types) across all zones, or one zone."""
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
    Success marker: 'vLLM application startup complete.'"""
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
    for i in range(len(lines) - 1, -1, -1):
        if "Starting Queued vLLM Bootloader" in lines[i]:
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
    instance_name: str = "vllm-gemma4-vm",
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
        Field(description="GCE machine type. Memory gates what loads, vCPUs gate how long you wait. "
                          "e2-highmem-16 = 16 vCPU/128 GiB runs a 31B; 64 GiB only loads one; 32 GiB is E2B only"),
    ] = CPU_DEBUG_MACHINE_TYPE,
    boot_disk_size_gb: Annotated[
        int, Field(ge=50, description="Checkpoints are large: E2B 8.3 GB, 31B W4A16 23.3 GB, MoE ~14 GB")
    ] = 100,
    spot: Annotated[
        bool, Field(description="Spot is ~70% cheaper and preemption is harmless for a debug box")
    ] = True,
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
        "gcloud", "compute", "instances", "create", instance_name,
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
        create_cmd += [f"--max-run-duration={max_run_duration}",
                       "--instance-termination-action=DELETE"]

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


@mcp.tool(title="Wait for JAX TPU ready", annotations=READ_ONLY)
async def wait_for_jax_ready(
    instance_name: str = "jax-tpu-vm",
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
            return (
                f"❌ JAX startup failed on `{instance_name}`:\n" + "\n".join(detail or ["see `get_tpu_vm_serial_log`"])
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
    instance_name: str = "jax-tpu-vm",
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
    argv, target = await _build_tpu_ssh_cmd(remote, "", instance_name, zone)
    if argv is None:
        return target
    rc, stdout, stderr = await run_command(argv, timeout=180)
    output = "\n".join(part for part in (stdout, stderr) if part).strip()
    if rc != 0:
        return f"❌ JAX cannot see the TPU on `{target}`:\n{output}"
    return f"🟢 JAX sees the TPU on `{target}`:\n{output}"


async def _get_zones_with_available_quota_list(
    service: str = "tpu.googleapis.com",
    quota_id: str = "TPUV6EPerProjectPerZoneForTPUAPI",
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


@mcp.tool(title="List zones with quota", annotations=READ_ONLY)
async def get_zones_with_available_quota(
    service: Annotated[str, Field(description="GCP service to query")] = "tpu.googleapis.com",
    quota_id: Annotated[str, Field(description="Quota ID to filter by")] = "TPUV6EPerProjectPerZoneForTPUAPI",
) -> str:
    """Retrieves the GCP zones that have a non-zero quota limit for a specific metric."""
    zones = await _get_zones_with_available_quota_list(service, quota_id)
    if not zones:
        return f"No zones/locations found with non-zero quota limit for `{quota_id}`."

    output = [f"### 📊 Available Zones with Quota for `{quota_id}`\n"]
    for zone in zones:
        output.append(f"- Zone/Region `{zone}`")
    return "\n".join(output)


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


@mcp.tool(title="Find TPU capacity across zones", annotations=DESTRUCTIVE)
async def find_tpu(
    resource_id: str = "vllm-gemma4-qr",
    service: Annotated[str, Field(description="GCP service to query for quota")] = "tpu.googleapis.com",
    quota_id: Annotated[str, Field(description="Quota ID to filter zones by")] = "TPUV6EPerProjectPerZoneForTPUAPI",
) -> str:
    """Sweeps every zone with available quota, creating the queued resource and polling
    until one reaches ACTIVE. Side effects: deletes the created resource in each zone
    that fails or times out; on success switches the server's default zone to the
    winning zone; records failed zones in ~/.cache/tpu-devops/tpu_zones_status.md and
    skips them on later sweeps."""
    zones = await _get_zones_with_available_quota_list(service, quota_id)
    if not zones:
        return f"❌ Aborted: No zones found with non-zero quota for `{quota_id}`."

    # Parse flat status file to skip zones where TPU could not be started
    skipped_zones = set()
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                content = f.read()
            for line in content.splitlines():
                # Matches lines like: | **zone-name** | Yes | No | ...
                match = re.search(r"\|\s*\*\*([a-zA-Z0-9-]+)\*\*\s*\|\s*([^|]+)\|\s*No\s*\|", line)
                if match:
                    zone_name = match.group(1).strip()
                    skipped_zones.add(zone_name)
            logger.info(f"Skipping zones (marked as failed in status file): {list(skipped_zones)}")
        except Exception as e:
            logger.error(f"Error parsing status file: {e}")

    logger.info(f"Zones with available quota: {zones}")

    attempts = []
    for zone in zones:
        if zone in skipped_zones:
            logger.info(f"Skipping zone {zone} as it is marked as failed in status file.")
            attempts.append(f"- **Zone {zone}**: ⏭️ Skipped (previously failed according to status file)")
            continue

        logger.info(f"Attempting to create TPU queued resource {resource_id} in zone {zone}...")
        result = await create_tpu_queued_resource(resource_id=resource_id, zone=zone)

        if result.startswith("❌"):
            attempts.append(f"- **Zone {zone}**: {result}")
            reason = result.replace("❌ Creation failed:", "").strip()
            await _update_status_file(zone, "No", reason)
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
            await _update_status_file(zone, "Yes", "Successfully started and reached ACTIVE state.")
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
            await _update_status_file(zone, "No", timeout_msg)
            attempts.append(f"- **Zone {zone}**: ❌ {timeout_msg}")

    return "❌ Failed to start TPU in any zone. Attempted zones:\n" + "\n".join(attempts)


@mcp.tool(title="Find flex-start TPU VM capacity", annotations=WRITE)
async def find_tpu_vm(
    instance_name: Optional[str] = None,
    accelerator: Annotated[str, Field(description="TPU type: v6e-1, v6e-4, v6e-8, or v5p-8")] = ACCELERATOR_TYPE,
    zones: Annotated[
        Optional[list[str]],
        Field(description="Zones to try in order; defaults to the zones with TPU quota"),
    ] = None,
    model_name: Optional[str] = None,
    per_zone_wait: Annotated[
        str, Field(description="How long each zone's flex-start request stays valid, e.g. '5m'")
    ] = "5m",
    workload: Annotated[
        Literal["vllm", "jax"],
        Field(description="'vllm' serves model_name via docker; 'jax' installs a bare jax[tpu] dev environment (no docker, no HF token)"),
    ] = "vllm",
) -> str:
    """GCE counterpart of `find_tpu`: tries flex-start TPU VM creation across zones
    until one grants capacity. TPUS_PER_TPU_FAMILY quota is only discoverable by
    attempting creation, so quota failures fail fast and the sweep moves on; each
    attempt's request expires after `per_zone_wait` so no pending requests pile up.
    On success, switches the server's default zone to the winning zone. Follow up
    with `wait_for_vllm_ready` (or `wait_for_jax_ready` for workload='jax')."""
    instance_name = instance_name or ("jax-tpu-vm" if workload == "jax" else "vllm-gemma4-vm")
    candidate_zones = zones or await _get_zones_with_available_quota_list()
    if not candidate_zones:
        return "❌ No candidate zones: no zones with TPU quota found — pass `zones` explicitly."

    attempts = []
    for zone in candidate_zones:
        logger.info(f"Attempting flex-start TPU VM {instance_name} in {zone}...")
        result = await create_tpu_vm_instance(
            instance_name=instance_name,
            zone=zone,
            accelerator=accelerator,
            model_name=model_name,
            request_valid_for=per_zone_wait,
            workload=workload,
        )
        attempts.append(f"- **{zone}**: {result.splitlines()[0]}")
        if result.startswith("🚀"):
            global ZONE
            ZONE = zone
            return (
                f"✅ Flex-start TPU VM secured in `{zone}` (now the default zone).\n\n{result}\n\n"
                "**Attempts:**\n" + "\n".join(attempts)
            )
        if result.startswith("⏳"):
            # gcloud timed out with the request possibly still pending in this zone;
            # stop so we don't stack pending capacity requests across zones.
            return f"{result}\n\n**Attempts:**\n" + "\n".join(attempts)

    return "❌ No zone granted flex-start capacity.\n**Attempts:**\n" + "\n".join(attempts)


@mcp.tool(title="Manage vLLM container", annotations=DESTRUCTIVE)
async def manage_vllm_docker(
    resource_id: str = "vllm-gemma4-qr",
    action: Literal["start", "stop", "restart", "status", "log", "rm"] = "start",
    instance_name: Annotated[
        Optional[str], Field(description="GCE flex-start TPU VM name; targets it instead of a queued resource")
    ] = None,
    zone: Optional[str] = None,
    model_name: Annotated[
        Optional[str], Field(description="Hugging Face model ID; defaults to the configured MODEL_NAME")
    ] = None,
    load_format: Annotated[
        Optional[str],
        Field(description="vLLM load format, e.g. 'tpu_streaming_loader' or 'runai_streamer'; auto-picked from model size"),
    ] = None,
    max_model_len: Annotated[
        Optional[int], Field(ge=1, description="Context length override; auto-picked from model size")
    ] = None,
    gpu_memory_utilization: Annotated[
        Optional[float], Field(gt=0, le=1, description="Memory utilization fraction; auto-picked from model size")
    ] = None,
) -> str:
    """Manages the vLLM Docker container on the serving host. Targets the queued
    resource's node by default, or a GCE flex-start TPU VM when instance_name is given.

    'start' with any serving parameter (model_name, load_format, max_model_len,
    gpu_memory_utilization) REPLACES the container so the new configuration takes
    effect — this is how you switch the served model. A plain 'start' just restarts
    the existing container (or creates one with defaults)."""
    zone = _zone(zone)
    selected_model = model_name or MODEL_NAME
    config_given = any(
        p is not None for p in (model_name, load_format, max_model_len, gpu_memory_utilization)
    )
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
    ssh_cmd, target = await _build_tpu_ssh_cmd(commands[action], resource_id, instance_name, zone)
    if ssh_cmd is None:
        return target

    # 'start' may fall back to `docker run`, which pulls the vLLM image (~5 min)
    # when it isn't cached — don't kill the client mid-pull.
    timeout = 600 if action in ("start", "restart") else 60
    rc, out, err = await run_command(ssh_cmd, timeout=timeout)
    if rc != 0:
        return f"""⚠️ Docker {action} failed on {target}, but the TPU itself remains safe.
Error: {err}"""
    return f"""✅ Docker {action} command executed on {target}.
{out}"""


@mcp.tool(title="List queued resources", annotations=READ_ONLY)
async def list_queued_resources(zone: Optional[str] = None) -> str:
    """Lists all Queued Resources in a specific zone (defaults to the current zone)."""
    zone = _zone(zone)
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


@mcp.tool(title="Describe queued resource", annotations=READ_ONLY)
async def describe_queued_resource(resource_id: str = "vllm-gemma4-qr", zone: Optional[str] = None) -> str:
    """Provides detailed information about a specific Queued Resource, including its
    lifecycle state and expiry."""
    zone = _zone(zone)
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
        full = json.dumps(data, indent=2)
        if len(full) > 4000:
            full = full[:4000] + "\n... (truncated)"
        return (
            f"### 🔍 Detail: {resource_id}\n"
            f"- **State:** `{state}`\n"
            f"- **Node ID:** `{node_id}`\n"
            f"- **Full Data:**\n```json\n{full}\n```"
        )
    except Exception:
        return f"""### 🔍 Detail: {resource_id}
```
{out}
```"""


@mcp.tool(title="Estimate deployment cost", annotations=READ_ONLY)
async def estimate_deployment_cost(
    hours: Annotated[float, Field(gt=0)] = 1.0,
    tpu_type: Literal["v6e", "v5e", "v5p"] = "v6e",
    topology: Annotated[str, Field(pattern=r"^\d+(x\d+)*$", description="Chip grid, e.g. '2x4'")] = "2x4",
    is_flex: bool = True,
) -> str:
    """Estimates the cost of a TPU deployment. `topology` is a chip grid like '2x4'."""
    # Approximate flex-start $/chip-hour as of mid-2026; on-demand modeled as 2x.
    # Update against the pricing page before relying on these for real budgeting.
    rates = {"v6e": 1.35, "v5e": 0.12, "v5p": 0.60}
    rate = rates.get(tpu_type, rates["v6e"]) * (1 if is_flex else 2)

    try:
        chips = math.prod(int(part) for part in topology.lower().split("x"))
        if chips <= 0:
            raise ValueError("topology dimensions must be positive")
    except ValueError as e:
        return f"❌ Invalid topology '{topology}': {e}. Expected a chip grid like '2x4'."

    total_cost = chips * rate * hours
    return (
        f"### 💸 Estimated Cost: `${total_cost:.2f}` for `{hours}h` on `{chips}` chip `{tpu_type}` "
        f"({'Flex-start' if is_flex else 'On-demand'})."
    )


@mcp.tool(title="System status dashboard", annotations=READ_ONLY)
async def get_system_status() -> str:
    """High-level dashboard covering both serving paths: GCE flex-start TPU VMs
    (all zones) and Queued Resources (current zone), plus vLLM health."""
    vms_str, resources_str = await asyncio.gather(list_tpu_vm_instances(), list_queued_resources())
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
        next_step = "Use `query_queued_gemma4` to interact with the model."
    elif "RUNNING" in vms_str or "ACTIVE" in resources_str:
        next_step = (
            "Capacity is up but serving isn't reachable — check `get_tpu_vm_serial_log` / "
            "`wait_for_vllm_ready` (GCE VM) or `manage_vllm_docker` with action='status'."
        )
    else:
        next_step = "Call `create_tpu_vm_instance` (or `find_tpu_vm` to sweep zones) to provision infrastructure."

    return (
        f"### 🌀 System Status ({ZONE})\n"
        f"- **vLLM Health:** {health}\n\n"
        f"**GCE flex-start TPU VMs (all zones):**\n```\n{vms_str}\n```\n"
        f"{resources_str}\n"
        f"**👉 Next Step:** {next_step}"
    )


@mcp.tool(title="Get vLLM endpoint", annotations=READ_ONLY)
async def get_vllm_endpoint() -> str:
    """Returns the active vLLM service URL if available."""
    url = await discover_vllm_url()
    if url:
        return f"🟢 vLLM is Online at: {url}"
    return "❌ No ACTIVE Queued Resource with a reachable vLLM service found."


@mcp.tool(title="Query the served model", annotations=READ_ONLY)
async def query_queued_gemma4(prompt: str, include_stats: bool = False) -> str:
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
    resource_id: str = "vllm-gemma4-qr",
    instance_name: Annotated[
        Optional[str], Field(description="GCE flex-start TPU VM name; targets it instead of a queued resource")
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
    """Runs vLLM's internal benchmark tool in a separate container on the serving host
    (queued resource node by default, or a GCE flex-start TPU VM via instance_name).
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
            f" && echo {BENCH_RESULT_MARKER}"
            f" && sudo cat /dev/shm/{result_name}"
            f" && sudo rm -f /dev/shm/{result_name}"
        )

    ssh_cmd, target = await _build_tpu_ssh_cmd(remote_cmd, resource_id, instance_name, zone)
    if ssh_cmd is None:
        return target

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
    resource_id: str = "vllm-gemma4-qr",
    instance_name: Annotated[
        Optional[str], Field(description="GCE flex-start TPU VM name; targets it instead of a queued resource")
    ] = None,
    zone: Optional[str] = None,
    tail: Annotated[int, Field(ge=1, le=5000)] = 100,
) -> str:
    """Retrieves the last `tail` lines of the vLLM Docker container's logs from the
    serving host (queued resource node by default, or a GCE flex-start TPU VM via
    instance_name). Bounded — the full log of a long-serving container can run to
    megabytes."""
    log_cmd = f"sudo docker logs vllm-gemma4 --tail {int(tail)}"

    ssh_cmd, target = await _build_tpu_ssh_cmd(log_cmd, resource_id, instance_name, _zone(zone))
    if ssh_cmd is None:
        return target

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Failed to get Docker logs from {target}.
Error: {err}"""
    return f"""✅ Docker logs from {target}:
{out}"""


@mcp.tool(title="Get TPU system logs", annotations=READ_ONLY)
async def get_tpu_system_logs(
    resource_id: str = "vllm-gemma4-qr",
    instance_name: Annotated[
        Optional[str], Field(description="GCE flex-start TPU VM name; targets it instead of a queued resource")
    ] = None,
    zone: Optional[str] = None,
    service: Annotated[str, Field(description="systemd unit name, e.g. 'docker'")] = "docker",
    tail: Annotated[int, Field(ge=1, le=5000)] = 100,
) -> str:
    """Retrieves systemd logs for a specific service from the serving host (queued
    resource node by default, or a GCE flex-start TPU VM via instance_name)."""
    log_cmd = f"journalctl -u {shlex.quote(service)} -n {int(tail)}"

    ssh_cmd, target = await _build_tpu_ssh_cmd(log_cmd, resource_id, instance_name, _zone(zone))
    if ssh_cmd is None:
        return target

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Failed to get system logs from {target}.
Error: {err}"""
    return f"""✅ System logs for '{service}' from {target}:
{out}"""


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
        f"[{log_entry.get('timestamp')}] {log_entry.get('resource', {}).get('labels', {}).get('node_id', 'N/A')} - "
        f"{log_entry.get('textPayload', log_entry.get('jsonPayload', {}))}"
        for log_entry in logs
    )
    return True, len(logs), f"### ☁️ Cloud Logs (filter: `{log_filter}`)\n```\n{formatted_logs}\n```"


@mcp.tool(title="Get Cloud Logging logs", annotations=READ_ONLY)
async def get_cloud_logging_logs(
    log_filter: str = 'resource.type="tpu_worker"', limit: Annotated[int, Field(ge=1, le=500)] = 20
) -> str:
    """Fetches logs from Google Cloud Logging."""
    _, _, text = await _fetch_cloud_logging_logs(log_filter, limit)
    return text


@mcp.tool(title="Analyze TPU error logs", annotations=READ_ONLY)
async def analyze_cloud_logging(minutes: Annotated[int, Field(ge=1, le=10080)] = 60) -> str:
    """Summarizes recent TPU errors from Cloud Logging using the self-hosted Gemma 4 model."""
    # Cloud Logging filters need an RFC3339 timestamp; relative durations like
    # "-PT60M" are not valid filter syntax.
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_filter = f'resource.type="tpu_worker" severity>=ERROR timestamp>="{cutoff}"'
    fetch_ok, entry_count, logs_result = await _fetch_cloud_logging_logs(log_filter, 10)

    if not fetch_ok:
        return f"❌ Cannot analyze: the Cloud Logging fetch itself failed.\n{logs_result}"
    if entry_count == 0:
        return f"✅ No TPU error logs (severity>=ERROR) in the last {minutes} minutes — nothing to analyze."

    prompt = (
        f"Here are the recent TPU error logs:\n{logs_result}\n\n"
        "Please analyze these logs, identify the root cause of the failures, and suggest remediations."
    )
    summary = await query_queued_gemma4(prompt)
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
        return "❌ No ACTIVE Queued Resource with a reachable vLLM service found."

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
        "### 🛠️ TPU Gemma 4 SRE Agent Help & Configuration\n\n"
        "You can configure this MCP server using the following environment variables:\n\n"
        f"- **`GOOGLE_CLOUD_PROJECT`**: Your GCP Project ID (falls back to the active gcloud config).\n"
        f"  - *Current Value:* `{PROJECT_ID or '(not set)'}`\n"
        f"- **`GOOGLE_CLOUD_ZONE`**: The default GCP zone (`find_tpu` moves it to wherever capacity lands).\n"
        f"  - *Current Value:* `{ZONE}`\n"
        f"- **`GOOGLE_CLOUD_REGION`**: The GCP Region for network resources.\n"
        f"  - *Current Value:* `{REGION}`\n"
        f"- **`MODEL_NAME`**: Default Hugging Face repository or path.\n"
        f"  - *Current Value:* `{MODEL_NAME}`\n"
        f"- **`ACCELERATOR_TYPE`**: TPU Accelerator type.\n"
        f"  - *Current Value:* `{ACCELERATOR_TYPE}`\n"
        f"- **`TENSOR_PARALLEL_SIZE`**: Tensor parallel size for serving.\n"
        f"  - *Current Value:* `{TENSOR_PARALLEL_SIZE}`\n"
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
        "(save one with `save_hf_token`) before vLLM resource creation "
        "(workload='jax' VMs don't need it).\n\n"
        "---\n\n"
        "### 🧰 Available MCP Tools\n\n"
        "#### 🐳 Capacity & Lifecycle — GCE flex-start TPU VMs (recommended for v6e/v5p)\n"
        "- **`create_tpu_vm_instance`**: Creates a flex-start TPU VM via GCE; workload='vllm' auto-starts "
        "serving, workload='jax' installs a bare jax[tpu] dev environment instead.\n"
        "- **`find_tpu_vm`**: Sweeps zones attempting flex-start creation until one grants capacity "
        "(same workload choice).\n"
        "- **`wait_for_vllm_ready`**: Polls health endpoint + serial marker until serving is up (~10 min loads).\n"
        "- **`create_cpu_debug_vm`**: Plain CPU VM (default Spot e2-highmem-16/128 GiB) running the "
        "same JAX stack minus libtpu — correctness work off the TPU, for cents an hour.\n"
        "- **`wait_for_jax_ready`**: Polls the serial marker until the JAX environment is ready (or failed).\n"
        "- **`verify_jax_tpu`**: Re-runs the JAX device check over SSH (asserts jax.devices() sees a TPU).\n"
        "- **`list_tpu_vm_instances`**: Lists GCE TPU VM instances (ct6e/ct5p) with IPs and status.\n"
        "- **`destroy_tpu_vm_instance`**: Deletes a GCE TPU VM instance (stops flex-start billing).\n"
        "- **`get_tpu_vm_serial_log`**: Tails a GCE TPU VM's serial console (boot/vLLM progress when SSH is blocked).\n"
        "- **`get_tpu_vm_endpoint`**: Resolves and health-probes a GCE TPU VM's vLLM endpoint.\n\n"
        "#### 🧊 Capacity & Lifecycle — Queued Resources (legacy API)\n"
        "- **`find_tpu`**: Sweeps every zone with quota until a queued resource reaches ACTIVE.\n"
        "- **`create_tpu_queued_resource`**: Creates one queued resource; never deletes others.\n"
        "- **`manage_queued_resource`**: Ensures the primary exists — ⚠️ DELETES all other queued resources in the zone.\n"
        "- **`destroy_queued_resource`**: Deletes a queued resource and its node.\n"
        "- **`list_queued_resources`** / **`describe_queued_resource`**: Inspect state.\n"
        "- **`get_zones_with_available_quota`**: Zones with non-zero quota for a metric.\n"
        "- **`find_gpu`**: GPU VMs, Cloud Run GPU services, and GPU quota in the project.\n"
        "- **`estimate_deployment_cost`**: Rough cost estimate for a TPU deployment.\n\n"
        "#### 🚀 Serving\n"
        "- **`manage_vllm_docker`**: start/stop/restart/status/log/rm for the vLLM container "
        "(queued node by default; GCE VM via instance_name).\n"
        "- **`get_vllm_endpoint`**: Active vLLM service URL.\n"
        "- **`get_vllm_deployment_config`**: gcloud one-liner for a single-host TPU vLLM deployment.\n"
        "- **`save_hf_token`**: Securely saves a Hugging Face API token to Secret Manager.\n\n"
        "#### 📊 Monitoring & Logs\n"
        "- **`get_system_status`**: High-level status dashboard of TPU node health and vLLM service.\n"
        "- **`verify_model_health`**: Verifies model inference health with a simple prompt.\n"
        "- **`get_model_details`**: Model, vLLM version, health, and key metrics report.\n"
        "- **`get_metrics`**: Raw Prometheus metrics from the vLLM /metrics endpoint.\n"
        "- **`get_vllm_docker_logs`**: Logs from the vLLM Docker container on the TPU VM.\n"
        "- **`get_tpu_system_logs`**: systemd logs for a service on the TPU VM.\n"
        "- **`get_cloud_logging_logs`**: Fetches logs from Google Cloud Logging.\n"
        "- **`analyze_cloud_logging`**: Summarizes recent TPU errors using the self-hosted Gemma 4 model.\n\n"
        "#### 📈 Inference & Benchmarking\n"
        "- **`query_queued_gemma4`**: Queries the served model (include_stats=True adds TTFT/throughput).\n"
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
        return "❌ No ACTIVE Queued Resource with a reachable vLLM service found."

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
