"""MCP server for the MI300X droplet that serves Gemma 4 E2B through vLLM.

gpu-vllm-mi300x-2b is one AMD Instinct MI300X serving `google/gemma-4-E2B-it`
under vLLM in Docker. There is no AMD GPU on the workstation this server runs
on: the card lives on a DigitalOcean GPU droplet reached through AMD Developer
Cloud (devcloud.amd.com), which is DigitalOcean underneath — same v2 API, same
droplet ids, token issued from the "My AMD Team" account. Every ROCm command
and every `docker` invocation in here is executed over SSH, and every lifecycle
operation goes through the DigitalOcean v2 API. Nothing in this file touches a
local GPU, and nothing should be added that assumes one.

STATUS 2026-09-16: serving. vLLM 0.29.1rc1.dev187+gaf1c01499, torch
2.12.0+rocm10.0.0, transformers 5.17.0, on gfx942 (MI300X VF, 191.69 GiB).
9,026,017 tokens of KV cache, 275.45x concurrency at 32768. Text, thinking,
tool calling and vision verified live against the endpoint.

THE NEWEST VENDOR IMAGE CANNOT LOAD THIS MODEL. `rocm/vllm` at
rocm10.0.0_..._vllm_0.27.0 (2026-08-27, the newest AMD publishes) dies in
config parsing, before the GPU is touched:

    AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
    and may vary across layers.

Gemma 4 runs 256-wide heads on its sliding-attention layers and 512 on its
full-attention ones. Transformers >= 5.15 reports `head_dim` as a per-layer
attribute and *raises* on a global read; vLLM's generic `get_head_size()` asks
for it with `getattr(..., 0)`, and the default never applies because the
accessor raises rather than returning nothing. Upstream fixes this with
`Gemma4ModelArchConfigConvertor`, which that build predates. This is a
version-pairing bug, not a ROCm one — the same image runs bf16 GEMMs on gfx942
correctly. `check_image` reads the convertor registry out of a candidate image
so this is a free question rather than a 62 GB one.

The pip route named in vLLM's own Gemma 4 recipe
(wheels.vllm.ai/rocm/nightly/rocm721) is staler still — 0.20.2rc1.dev15, older
than the build that fails. The container nightlies are the only current route,
and VLLM_IMAGE points at one.

THE OFFICIAL IMAGE'S ENTRYPOINT IS ALREADY `["vllm","serve"]`. The model id is
the first argument; passing `vllm serve <model>` again is an unrecognised-
arguments error. AMD's `rocm/vllm` images have no entrypoint and do need the
subcommand. `deploy_vllm` handles both, keyed off the image name.

AUDIO IS OFF AND CANNOT BE TURNED ON HERE. E2B carries a conformer audio
encoder, but no ROCm vLLM image ships the `vllm[audio]` extras — `librosa` and
`soundfile` are absent from all three tried. `--limit-mm-per-prompt` therefore
sets `audio: 0`, which also skips allocating encoder memory for a path that
cannot be reached. Serving audio needs a derived image, not a flag.

Deliberately absent: create and destroy. This server can find, start, stop,
reach and interrogate a droplet that already exists, and it is scoped by tag so
it cannot act on one it was not pointed at. Provisioning an MI300 droplet and
destroying one are the two operations whose cost of being wrong is measured in
dollars per hour and in lost local state, so they stay a deliberate human step.

BILLING DOES NOT STOP WHEN A DROPLET IS POWERED OFF. DigitalOcean bills a
powered-off droplet at the full hourly rate, because the resources stay
reserved for it. `stop_droplet` saves nothing.
"""

import asyncio
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / "tpu.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The registered key prefixes every tool name
# (mcp__gpu-vllm-mi300x-2b__serving_status), so it must match the directory or
# two loaded rigs are indistinguishable at the call site. NAMING.md holds the
# rule; the default is derived rather than written out.
RIG_NAME = PROJECT_DIR.name
MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", RIG_NAME)

DO_API_BASE = os.environ.get("DO_API_BASE", "https://api.digitalocean.com/v2")

# Every lookup is filtered by this tag, so the server can only ever see the
# droplets somebody deliberately tagged for it. An untagged droplet in the same
# account is invisible here, which is the point: a typo in a droplet id cannot
# power-cycle an unrelated machine.
DROPLET_TAG = os.environ.get("DROPLET_TAG", "gemma")

SSH_USER = os.environ.get("SSH_USER", "root")
SSH_KEY = os.environ.get("SSH_KEY", "")
SSH_PORT = os.environ.get("SSH_PORT", "22")
REMOTE_WORKDIR = os.environ.get("REMOTE_WORKDIR", "/opt/gpu-vllm-mi300x-2b")

# Serving. Read from tpu.env, which is committed and carries no secrets.
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai-rocm:nightly-rocm100")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-4-E2B-it")
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
VLLM_CONTAINER = os.environ.get("VLLM_CONTAINER", "vllm")
HF_CACHE = os.environ.get("HF_CACHE", "/opt/hf-cache")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "32768")
GPU_MEMORY_UTILIZATION = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
LIMIT_MM_PER_PROMPT = os.environ.get("LIMIT_MM_PER_PROMPT", '{"image": 4, "audio": 0}')
CHAT_TEMPLATE = os.environ.get("CHAT_TEMPLATE", "/app/vllm/examples/tool_chat_template_gemma4.jinja")

# The render groups on the Debian 13 GPU image. Passed as --group-add so the
# container can open /dev/kfd without running privileged.
RENDER_GIDS = os.environ.get("RENDER_GIDS", "44,991")

mcp = MCPServer(MCP_SERVER_NAME)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

# A 64x64 red/blue checkerboard, 158 bytes. Embedded rather than fetched so the
# vision probe needs no network from the droplet and the expected answer is
# known before the model gives one.
CHECKERBOARD_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAZUlEQVR4nO3PMQoAMAgEQf//adPan"
    "hCwGEhplpvqquiF5/27n54DAAAAAAAAAAAAAMwftwblfQAAAAAAAAAAAACAbf/coHQRAAAAAAAAAA"
    "AAAMAacG1Q3gcAAAAAAAAAAAAAWPYfuOzw4kINlWAAAAAASUVORK5CYII="
)


def _error(exc: Exception) -> str:
    """Render an exception as the markdown a tool returns instead of raising."""
    if isinstance(exc, httpx.HTTPError):
        return f"❌ DigitalOcean API unreachable: {exc}"
    return f"❌ {exc}"


def _token() -> str:
    """Read the API token, preferring the name doctl and Terraform use.

    The token is never read from tpu.env — that file is committed. It comes
    from the environment or from .env, which is gitignored and mode 0600.
    """
    token = os.environ.get("DIGITALOCEAN_ACCESS_TOKEN") or os.environ.get("DIGITALOCEAN_TOKEN")
    if not token:
        raise RuntimeError(
            "DIGITALOCEAN_ACCESS_TOKEN is unset. Put it in `.env` (gitignored, mode 0600) "
            "or export it — never in `tpu.env`, which is committed."
        )
    return token


async def _api(method: str, path: str, payload: Optional[dict] = None, timeout: int = 30) -> dict:
    """Call the DigitalOcean v2 API and return the decoded body.

    Raises RuntimeError carrying the API's own error message, which is far more
    useful than a bare status code: DigitalOcean puts the reason in `message`.
    """
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Content-Type": "application/json",
    }
    url = path if path.startswith("http") else f"{DO_API_BASE}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(method, url, headers=headers, json=payload)

    if resp.status_code == 401:
        raise RuntimeError("DigitalOcean rejected the token (401). Is DIGITALOCEAN_ACCESS_TOKEN current?")
    if resp.status_code == 429:
        remaining = resp.headers.get("ratelimit-remaining", "?")
        reset = resp.headers.get("ratelimit-reset", "?")
        raise RuntimeError(f"DigitalOcean rate limit hit (429). remaining={remaining} reset={reset}")
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("message", resp.text[:300])
        except ValueError:
            detail = resp.text[:300]
        raise RuntimeError(f"DigitalOcean {resp.status_code}: {detail}")
    if not resp.content:
        return {}
    return resp.json()


async def _paged(path: str, key: str) -> list[dict]:
    """Collect every page of a list endpoint. per_page=200 is the API maximum."""
    sep = "&" if "?" in path else "?"
    url = f"{path}{sep}per_page=200"
    items: list[dict] = []
    while url:
        body = await _api("GET", url)
        items.extend(body.get(key, []))
        url = body.get("links", {}).get("pages", {}).get("next", "")
    return items


async def _droplets() -> list[dict]:
    """Every droplet carrying DROPLET_TAG."""
    return await _paged(f"/droplets?tag_name={DROPLET_TAG}", "droplets")


async def _resolve(droplet: str) -> dict:
    """Find one tagged droplet by numeric id or by name.

    Accepting the name matters because the id is a nine-digit number nobody
    remembers, and mistyping one is exactly the class of error the tag scope
    exists to contain.
    """
    found = await _droplets()
    if not found:
        raise RuntimeError(
            f"No droplets tagged `{DROPLET_TAG}`. Tag the droplet in the DigitalOcean "
            f"console, or set DROPLET_TAG in `tpu.env` to the tag it already has."
        )
    wanted = droplet.strip()
    for item in found:
        if str(item.get("id")) == wanted or item.get("name") == wanted:
            return item
    names = ", ".join(f"`{d.get('name')}` ({d.get('id')})" for d in found)
    raise RuntimeError(f"No droplet `{wanted}` tagged `{DROPLET_TAG}`. Tagged droplets: {names}")


def _public_ip(droplet: dict) -> Optional[str]:
    """Pull the public IPv4 out of a droplet record.

    Resolved per call rather than pinned in tpu.env: a droplet that is rebuilt
    or moved comes back with a different address, and a stale hardcoded IP is
    indistinguishable from a machine that is merely still booting.
    """
    for iface in droplet.get("networks", {}).get("v4", []):
        if iface.get("type") == "public":
            return iface.get("ip_address")
    return None


def _ssh_argv(ip: str, command: Optional[str] = None) -> list[str]:
    """Build an ssh argv that fails fast instead of asking a question.

    BatchMode=yes turns a missing key into an error rather than a password
    prompt, and accept-new adopts an unknown host key rather than an
    interactive yes/no. Either prompt would hang this tool until it timed out,
    with no output to explain why.
    """
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-p",
        SSH_PORT,
    ]
    if SSH_KEY:
        argv += ["-i", os.path.expanduser(SSH_KEY)]
    argv.append(f"{SSH_USER}@{ip}")
    if command is not None:
        argv.append(command)
    return argv


async def run_command(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run a command with no shell. Never shell=True — see CLAUDE.md."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )
    except asyncio.TimeoutError:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"


async def _reachable(droplet: dict) -> tuple[Optional[str], Optional[str]]:
    """Return (ip, reason_it_is_not_usable). Exactly one is None."""
    status = droplet.get("status")
    if status != "active":
        return None, f"droplet is `{status}`, not `active`. Run `start_droplet` first."
    ip = _public_ip(droplet)
    if not ip:
        return (
            None,
            "droplet is active but has no public IPv4 yet. Networking is still coming up.",
        )
    return ip, None


async def _remote(droplet: str, command: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run one command on the rig's droplet over SSH.

    Raises if the droplet cannot be reached, so callers can assume the tuple
    describes the command rather than the connection.
    """
    item = await _resolve(droplet)
    ip, why_not = await _reachable(item)
    if why_not:
        raise RuntimeError(f"Cannot reach `{item.get('name')}`: {why_not}")
    return await run_command(_ssh_argv(ip, command), timeout=timeout)


async def _curl_endpoint(droplet: str, path: str, payload: Optional[dict] = None, timeout: int = 120) -> tuple[int, str, str]:
    """Call the vLLM endpoint from inside the droplet, not across the internet.

    The serving port is published on the droplet's own interface and there is
    no promise a firewall lets the workstation reach it. Going over SSH to
    127.0.0.1 tests the thing that actually matters — whether vLLM is serving —
    rather than whether a port happens to be open to the world.
    """
    url = f"http://127.0.0.1:{VLLM_PORT}{path}"
    if payload is None:
        cmd = f"curl -s -m {timeout - 10} {shlex.quote(url)}"
    else:
        body = shlex.quote(json.dumps(payload))
        cmd = f"curl -s -m {timeout - 10} {shlex.quote(url)} -H 'Content-Type: application/json' -d {body}"
    return await _remote(droplet, cmd, timeout=timeout)


def _truncate(text: str, limit: int = 6000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text or "(no output)"
    return text[:limit] + f"\n… truncated at {limit} chars."


def _is_official_image(image: str) -> bool:
    """True when the image already has `vllm serve` as its ENTRYPOINT.

    vllm/vllm-openai-rocm sets ENTRYPOINT ["vllm","serve"], so the model id is
    the first argument. AMD's rocm/vllm images have no entrypoint and need the
    subcommand spelled out. Getting this backwards is an unrecognised-arguments
    error on one side and a "vllm: command not found" on the other.
    """
    return image.split(":", 1)[0].startswith("vllm/")


def _serve_argv() -> list[str]:
    """The docker argv that starts the model server, as a list."""
    argv = [
        "docker",
        "run",
        "-d",
        "--name",
        VLLM_CONTAINER,
        "--restart",
        "unless-stopped",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
    ]
    for gid in RENDER_GIDS.split(","):
        gid = gid.strip()
        if gid:
            argv += ["--group-add", gid]
    argv += [
        "--security-opt",
        "seccomp=unconfined",
        "--security-opt",
        "label=disable",
        "-v",
        f"{HF_CACHE}:/root/.cache/huggingface",
        "-p",
        f"{VLLM_PORT}:{VLLM_PORT}",
        VLLM_IMAGE,
    ]
    if not _is_official_image(VLLM_IMAGE):
        argv += ["vllm", "serve"]
    argv += [
        VLLM_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        VLLM_PORT,
        "--max-model-len",
        MAX_MODEL_LEN,
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "gemma4",
        "--tool-call-parser",
        "gemma4",
        "--chat-template",
        CHAT_TEMPLATE,
        "--limit-mm-per-prompt",
        LIMIT_MM_PER_PROMPT,
        "--async-scheduling",
    ]
    return argv


@mcp.tool(title="List managed droplets", annotations=READ_ONLY)
async def list_droplets() -> str:
    """List every droplet tagged for this rig, with status, size and address."""
    try:
        found = await _droplets()
        if not found:
            return f"📡 No droplets tagged `{DROPLET_TAG}`."
        lines = [
            f"📡 {len(found)} droplet(s) tagged `{DROPLET_TAG}`.",
            "",
            "| Name | ID | Status | Size | Region | Public IPv4 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in found:
            lines.append(
                f"| `{item.get('name')}` | {item.get('id')} | {item.get('status')} "
                f"| {item.get('size_slug')} | {item.get('region', {}).get('slug', '-')} "
                f"| {_public_ip(item) or '-'} |"
            )
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Show droplet status", annotations=READ_ONLY)
async def droplet_status(droplet: str) -> str:
    """Report one tagged droplet's power state, size, address and hourly cost."""
    try:
        item = await _resolve(droplet)
        size = item.get("size", {})
        hourly = size.get("price_hourly")
        ip, why_not = await _reachable(item)
        lines = [
            f"📡 **{item.get('name')}** ({item.get('id')})",
            "",
            f"- status: `{item.get('status')}`",
            f"- size: `{item.get('size_slug')}` — {size.get('vcpus', '?')} vCPU, "
            f"{size.get('memory', 0) // 1024} GB RAM, {size.get('disk', '?')} GB disk",
            f"- region: {item.get('region', {}).get('slug', '-')}",
            f"- public IPv4: {ip or '-'}",
            f"- cost: {f'${hourly:.2f}/hr' if hourly else 'unknown'} — **billed while powered off, too**",
        ]
        if why_not:
            lines += ["", f"❌ Not usable right now: {why_not}"]
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Start droplet", annotations=WRITE)
async def start_droplet(droplet: str) -> str:
    """Power on a tagged droplet."""
    try:
        item = await _resolve(droplet)
        if item.get("status") == "active":
            return f"✅ `{item.get('name')}` is already active."
        body = await _api("POST", f"/droplets/{item.get('id')}/actions", {"type": "power_on"})
        action = body.get("action", {})
        return (
            f"✅ Power-on requested for `{item.get('name')}` — action {action.get('id')}, "
            f"status `{action.get('status')}`. Poll with `action_status`."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Stop droplet", annotations=DESTRUCTIVE)
async def stop_droplet(droplet: str, graceful: bool = True) -> str:
    """Power off a tagged droplet. This does NOT stop DigitalOcean billing it.

    A powered-off droplet keeps its resources reserved and keeps costing the
    full hourly rate. Only destroying it stops the meter, and this server
    cannot destroy one. Stop a droplet to quiesce it, never to save money.
    """
    try:
        item = await _resolve(droplet)
        kind = "shutdown" if graceful else "power_off"
        body = await _api("POST", f"/droplets/{item.get('id')}/actions", {"type": kind})
        action = body.get("action", {})
        return (
            f"✅ `{kind}` requested for `{item.get('name')}` — action {action.get('id')}.\n\n"
            "📡 Billing continues at the full hourly rate while it is off."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Reboot droplet", annotations=DESTRUCTIVE)
async def reboot_droplet(droplet: str) -> str:
    """Reboot a tagged droplet — the fix for a missing /dev/kfd on a fresh one.

    On a freshly provisioned droplet `amdgpu` fails to bind during provisioning
    and unloads, leaving a card `lspci` can see and no ROCm process can use.
    Nothing needs installing. One reboot is the whole remedy.
    """
    try:
        item = await _resolve(droplet)
        body = await _api("POST", f"/droplets/{item.get('id')}/actions", {"type": "reboot"})
        action = body.get("action", {})
        return f"✅ Reboot requested for `{item.get('name')}` — action {action.get('id')}."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check action status", annotations=READ_ONLY)
async def action_status(action_id: str) -> str:
    """Report the status of a DigitalOcean action id returned by another tool."""
    try:
        body = await _api("GET", f"/actions/{action_id}")
        action = body.get("action", {})
        icon = "✅" if action.get("status") == "completed" else "📡"
        return (
            f"{icon} Action {action.get('id')} — `{action.get('type')}` is "
            f"`{action.get('status')}` (started {action.get('started_at')})."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get SSH command", annotations=READ_ONLY)
async def ssh_command(droplet: str) -> str:
    """Print the ssh command for a tagged droplet, with the address resolved."""
    try:
        item = await _resolve(droplet)
        ip, why_not = await _reachable(item)
        if why_not:
            return f"❌ Cannot reach `{item.get('name')}`: {why_not}"
        return f"✅ ```\n{' '.join(shlex.quote(part) for part in _ssh_argv(ip))}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Run a command on the droplet", annotations=WRITE)
async def run_on_droplet(droplet: str, command: str, timeout: int = 300) -> str:
    """Run one shell command on the rig's droplet over SSH and return its output.

    The command is handed to the remote shell as a single argument; the local
    side never invokes a shell at all.
    """
    try:
        code, out, err = await _remote(droplet, command, timeout=timeout)
        icon = "✅" if code == 0 else "❌"
        return f"{icon} `{command}` exited {code}.\n\n```\n{_truncate(out or err)}\n```"
    except Exception as exc:
        return _error(exc)


def _summarize_rocm_smi(raw: str) -> Optional[str]:
    """Turn `rocm-smi --json` output into a table, or None if it isn't that."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or not data:
        return None

    def pick(card: dict, *names: str) -> str:
        for name in names:
            for key, value in card.items():
                if key.lower().replace(" ", "") == name:
                    return str(value)
        return "-"

    rows = ["| Card | Product | GPU use % | VRAM used % |", "| --- | --- | --- | --- |"]
    count = 0
    for card, values in sorted(data.items()):
        if not isinstance(values, dict):
            continue
        count += 1
        rows.append(
            f"| {card} | {pick(values, 'cardseries', 'devicename', 'productname')} "
            f"| {pick(values, 'gpuuse(%)', 'gpuuse')} | {pick(values, 'gpumemoryuse(%)', 'memoryuse(%)')} |"
        )
    if not count:
        return None
    rows.append("")
    rows.append(f"📡 {count} GPU(s) reported by rocm-smi.")
    return "\n".join(rows)


@mcp.tool(title="Show GPU status on the droplet", annotations=READ_ONLY)
async def gpu_status(droplet: str) -> str:
    """Report the MI300X on the rig's droplet, and say why if it is not there.

    NEITHER rocm-smi NOR amd-smi SETS A USEFUL EXIT CODE. With the driver
    uninitialised, `rocm-smi` prints "Driver not initialized" to stderr,
    prints nothing to stdout, and exits 0. The parsed output decides here; the
    exit code is not consulted at all.
    """
    try:
        _, out, err = await _remote(droplet, "rocm-smi --json", timeout=60)
        table = _summarize_rocm_smi(out)
        if table:
            return f"✅ GPU reporting on `{droplet}`.\n\n{table}"

        _, kfd, _ = await _remote(droplet, "test -e /dev/kfd && echo present || echo missing", timeout=30)
        if "missing" in kfd:
            return (
                "❌ `/dev/kfd` is missing, so no ROCm process can use the card.\n\n"
                "📡 On a freshly provisioned droplet this is expected: `amdgpu` failed to bind "
                "during provisioning and unloaded. **Reboot it once** — nothing needs installing. "
                "Use `reboot_droplet`."
            )
        return f"❌ rocm-smi produced no parseable output.\n\n```\n{_truncate(err or out, 2000)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check an image before pulling it", annotations=READ_ONLY)
async def check_image(droplet: str, image: Optional[str] = None) -> str:
    """Ask a candidate vLLM image whether it can actually load this checkpoint.

    Two questions, both free, both answered from a container that is already
    local or is pulled once:

    1. Does its vLLM register `Gemma4ModelArchConfigConvertor`? Without it the
       image raises `AmbiguousGlobalPerLayerAttributeError` on `head_dim`
       during config parsing and never reaches the GPU. `rocm/vllm` at vLLM
       0.27.1.dev5 fails exactly here.
    2. Does its torch carry `gfx942`? `torch.cuda.get_arch_list()` returns `[]`
       with no devices mapped in, which reads like "no kernels for you" and is
       nothing of the sort — so the devices are mapped in for the check.

    Import `vllm.config` before reaching into `vllm.transformers_utils`, or a
    circular-import ImportError makes a working image look broken.
    """
    target = image or VLLM_IMAGE
    probe = (
        "import vllm, torch, transformers; "
        "import vllm.config; "
        "from vllm.transformers_utils.model_arch_config_convertor import "
        "MODEL_ARCH_CONFIG_CONVERTORS as M; "
        "print('vllm', vllm.__version__); "
        "print('torch', torch.__version__); "
        "print('transformers', transformers.__version__); "
        "print('convertor', M.get('gemma4')); "
        "print('archs', [a for a in torch.cuda.get_arch_list() if 'gfx94' in a])"
    )
    gids = " ".join(f"--group-add {g.strip()}" for g in RENDER_GIDS.split(",") if g.strip())
    cmd = (
        f"docker run --rm --device=/dev/kfd --device=/dev/dri {gids} "
        f"--security-opt seccomp=unconfined --security-opt label=disable "
        f"--entrypoint python3 {shlex.quote(target)} -c {shlex.quote(probe)}"
    )
    try:
        code, out, err = await _remote(droplet, cmd, timeout=900)
        text = out.strip()
        ok = "convertor" in text and "None" not in text.split("convertor", 1)[-1].splitlines()[0]
        has_arch = "gfx94" in text
        icon = "✅" if (code == 0 and ok and has_arch) else "❌"
        verdict = []
        if code != 0:
            verdict.append("the probe itself failed")
        else:
            verdict.append("carries the Gemma 4 convertor" if ok else "**lacks `Gemma4ModelArchConfigConvertor`** — it cannot load Gemma 4")
            verdict.append("carries gfx942 kernels" if has_arch else "**no gfx94x in the arch list**")
        return (
            f"{icon} `{target}`: {'; '.join(verdict)}.\n\n```\n{_truncate(out or err, 2000)}\n```"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Pull the serving image", annotations=WRITE)
async def pull_image(droplet: str, image: Optional[str] = None) -> str:
    """Pull the vLLM image onto the droplet. 35-62 GB, so it is not quick."""
    target = image or VLLM_IMAGE
    try:
        code, out, err = await _remote(droplet, f"docker pull {shlex.quote(target)}", timeout=3600)
        icon = "✅" if code == 0 else "❌"
        return f"{icon} `docker pull {target}` exited {code}.\n\n```\n{_truncate(out or err, 1500)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Cache the model weights", annotations=WRITE)
async def download_weights(droplet: str, model: Optional[str] = None) -> str:
    """Cache the checkpoint into HF_CACHE on the droplet, before serving it.

    The weights are image-independent, so caching them on the host and mounting
    the directory in means swapping images costs nothing in re-download. E2B is
    Apache-2.0 and ungated, so no token is needed.
    """
    target = model or VLLM_MODEL
    probe = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({target!r}, max_workers=8); print('DONE')"
    )
    cmd = (
        f"docker run --rm -v {shlex.quote(HF_CACHE)}:/root/.cache/huggingface "
        f"--entrypoint python3 {shlex.quote(VLLM_IMAGE)} -c {shlex.quote(probe)}"
    )
    try:
        code, out, err = await _remote(droplet, cmd, timeout=3600)
        icon = "✅" if code == 0 and "DONE" in out else "❌"
        return f"{icon} Cached `{target}` into `{HF_CACHE}`.\n\n```\n{_truncate(out or err, 1500)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Start the model server", annotations=WRITE)
async def deploy_vllm(droplet: str, replace: bool = False) -> str:
    """Start vLLM serving the checkpoint, with the Gemma 4 flags this rig needs.

    Set `replace=True` to remove an existing container of the same name first.
    A running container holds the card's memory, so a second one asking for
    0.90 of it fails on free VRAM rather than on anything interesting.
    """
    try:
        if replace:
            await _remote(droplet, f"docker rm -f {shlex.quote(VLLM_CONTAINER)}", timeout=120)
        argv = _serve_argv()
        cmd = " ".join(shlex.quote(part) for part in argv)
        code, out, err = await _remote(droplet, cmd, timeout=600)
        if code != 0:
            return f"❌ `docker run` exited {code}.\n\n```\n{_truncate(err or out, 2000)}\n```"
        return (
            f"✅ Started `{VLLM_CONTAINER}` on `{droplet}` serving `{VLLM_MODEL}`.\n\n"
            f"- image: `{VLLM_IMAGE}`\n"
            f"- context: {MAX_MODEL_LEN}, gpu-memory-utilization {GPU_MEMORY_UTILIZATION}\n"
            f"- multimodal: `{LIMIT_MM_PER_PROMPT}` — audio is 0 because no ROCm image ships `vllm[audio]`\n\n"
            "📡 Loading takes minutes: weights, then torch.compile, then graph capture. "
            "Poll `serving_status`; read `server_logs` if it does not come up."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Stop the model server", annotations=DESTRUCTIVE)
async def stop_vllm(droplet: str, remove: bool = True) -> str:
    """Stop the serving container and release the card's memory."""
    try:
        cmd = f"docker rm -f {shlex.quote(VLLM_CONTAINER)}" if remove else f"docker stop {shlex.quote(VLLM_CONTAINER)}"
        code, out, err = await _remote(droplet, cmd, timeout=180)
        icon = "✅" if code == 0 else "❌"
        return f"{icon} `{cmd}` exited {code}.\n\n```\n{_truncate(out or err, 800)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Show serving status", annotations=READ_ONLY)
async def serving_status(droplet: str) -> str:
    """Report the container state and whether the endpoint answers.

    A running container is not a serving model: vLLM spends minutes loading
    weights and capturing graphs after `docker run` returns. Both facts are
    reported separately so "up but not ready" is distinguishable from "down".
    """
    try:
        _, ps, _ = await _remote(
            droplet,
            f"docker ps -a --filter name=^/{VLLM_CONTAINER}$ --format '{{{{.Status}}}}|{{{{.Image}}}}'",
            timeout=60,
        )
        ps = ps.strip()
        if not ps:
            return f"❌ No container named `{VLLM_CONTAINER}` on `{droplet}`. Run `deploy_vllm`."
        status, _, image = ps.partition("|")

        code, out, _ = await _curl_endpoint(droplet, "/v1/models", timeout=30)
        served = "-"
        ready = False
        try:
            body = json.loads(out)
            served = ", ".join(f"`{m.get('id')}`" for m in body.get("data", []))
            ready = bool(body.get("data"))
        except ValueError:
            pass

        icon = "✅" if ready else "📡"
        lines = [
            f"{icon} `{VLLM_CONTAINER}` on `{droplet}`",
            "",
            f"- container: {status}",
            f"- image: `{image}`",
            f"- endpoint: {'answering' if ready else 'not answering yet'} on `127.0.0.1:{VLLM_PORT}`",
            f"- served: {served}",
        ]
        if not ready and code == 0:
            lines += ["", "📡 Container is up but the API is not serving yet — still loading. Check `server_logs`."]
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Fetch model server logs", annotations=READ_ONLY)
async def server_logs(droplet: str, lines: int = 80, grep: Optional[str] = None) -> str:
    """Tail the serving container's logs, optionally filtered.

    The ROCm runtime prints a `queue_controller.cpp` warning on every queue
    creation that is benign on a VF and drowns everything else, so it is
    filtered out unless a grep is given.
    """
    try:
        cmd = f"docker logs --tail {int(lines) * 4} {shlex.quote(VLLM_CONTAINER)} 2>&1"
        if grep:
            cmd += f" | grep -E {shlex.quote(grep)}"
        else:
            cmd += " | grep -v queue_controller"
        cmd += f" | tail -n {int(lines)}"
        code, out, err = await _remote(droplet, cmd, timeout=120)
        icon = "✅" if code == 0 else "❌"
        return f"{icon} Last {lines} log lines from `{VLLM_CONTAINER}`.\n\n```\n{_truncate(out or err)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Query the served model", annotations=WRITE)
async def query_model(
    droplet: str,
    prompt: str,
    max_tokens: int = 512,
    enable_thinking: bool = False,
) -> str:
    """Send one chat completion to the endpoint and return the answer.

    Thinking is on in this chat template even when it is not asked for — a
    plain prompt spends reasoning tokens — so the reasoning text and the token
    accounting are both reported rather than silently dropped.
    """
    payload: dict = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    try:
        _, out, err = await _curl_endpoint(droplet, "/v1/chat/completions", payload, timeout=180)
        try:
            body = json.loads(out)
        except ValueError:
            return f"❌ Endpoint did not return JSON.\n\n```\n{_truncate(out or err, 1500)}\n```"
        if "error" in body:
            return f"❌ {body['error']}"
        message = body["choices"][0]["message"]
        usage = body.get("usage", {})
        details = usage.get("completion_tokens_details") or {}
        lines = [f"✅ `{VLLM_MODEL}` answered in {usage.get('completion_tokens', '?')} tokens."]
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            lines += ["", f"**Thinking** ({details.get('reasoning_tokens', '?')} tokens):", "", "> " + reasoning.strip().replace("\n", "\n> ")]
        lines += ["", "**Answer:**", "", (message.get("content") or "(empty)").strip()]
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify every advertised capability", annotations=WRITE)
async def verify_capabilities(droplet: str) -> str:
    """Exercise text, thinking, tool calling and vision against the live endpoint.

    A flag being accepted is not evidence it did anything, so each capability is
    probed with a request whose correct answer is known in advance. Audio is not
    probed: no ROCm vLLM image ships `librosa`/`soundfile`, so it would fail for
    a packaging reason and tell you nothing about the model.
    """
    results: list[tuple[str, str, str]] = []
    try:
        # Text.
        _, out, _ = await _curl_endpoint(
            droplet,
            "/v1/chat/completions",
            {
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": "In one sentence: what GPU architecture is an AMD MI300X?"}],
                "max_tokens": 160,
                "temperature": 0,
            },
            timeout=180,
        )
        try:
            body = json.loads(out)
            text = (body["choices"][0]["message"].get("content") or "").strip()
            results.append(("text", "✅" if text else "❌", text[:110] or "(empty)"))
        except (ValueError, KeyError, IndexError):
            results.append(("text", "❌", _truncate(out, 110)))

        # Thinking.
        _, out, _ = await _curl_endpoint(
            droplet,
            "/v1/chat/completions",
            {
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": "A snail climbs 3 feet a day up a 20-foot well and slides 2 back each night. How many days to the top?"}],
                "max_tokens": 1200,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            timeout=240,
        )
        try:
            body = json.loads(out)
            msg = body["choices"][0]["message"]
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            tokens = (body.get("usage", {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            results.append(("thinking", "✅" if reasoning else "❌", f"{len(reasoning)} chars, {tokens} reasoning tokens"))
        except (ValueError, KeyError, IndexError):
            results.append(("thinking", "❌", _truncate(out, 110)))

        # Tool calling.
        _, out, _ = await _curl_endpoint(
            droplet,
            "/v1/chat/completions",
            {
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": "What is the weather in Reykjavik right now? Use the tool."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string", "description": "City name"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "max_tokens": 600,
                "temperature": 0,
            },
            timeout=240,
        )
        try:
            body = json.loads(out)
            choice = body["choices"][0]
            calls = choice["message"].get("tool_calls") or []
            detail = f"{choice.get('finish_reason')} → " + ", ".join(
                f"{c['function']['name']}{c['function']['arguments']}" for c in calls
            ) if calls else f"no tool call ({choice.get('finish_reason')})"
            results.append(("tool calling", "✅" if calls else "❌", detail[:110]))
        except (ValueError, KeyError, IndexError):
            results.append(("tool calling", "❌", _truncate(out, 110)))

        # Vision.
        _, out, _ = await _curl_endpoint(
            droplet,
            "/v1/chat/completions",
            {
                "model": VLLM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in one sentence: what colors and pattern?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{CHECKERBOARD_PNG_B64}"}},
                        ],
                    }
                ],
                "max_tokens": 200,
                "temperature": 0,
            },
            timeout=240,
        )
        try:
            body = json.loads(out)
            text = (body["choices"][0]["message"].get("content") or "").strip()
            seen = "red" in text.lower() and "blue" in text.lower()
            results.append(("vision", "✅" if seen else "❌", text[:110] or "(empty)"))
        except (ValueError, KeyError, IndexError):
            results.append(("vision", "❌", _truncate(out, 110)))

        passed = sum(1 for _, icon, _ in results if icon == "✅")
        lines = [
            f"{'✅' if passed == len(results) else '❌'} {passed}/{len(results)} capabilities verified on `{droplet}`.",
            "",
            "| Capability | | Result |",
            "| --- | --- | --- |",
        ]
        for name, icon, detail in results:
            lines.append(f"| {name} | {icon} | {detail.replace('|', '/')} |")
        lines += ["", "📡 Audio is not probed — no ROCm vLLM image ships the `vllm[audio]` extras."]
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Triage the server logs with the served model", annotations=WRITE)
async def analyze_logs(droplet: str, lines: int = 120) -> str:
    """Feed the serving logs to the model itself and ask it what is wrong.

    The rig triaging itself only works while the endpoint answers, which is
    exactly when it is least needed — so a dead endpoint returns the raw tail
    rather than an error about the analysis.
    """
    try:
        _, raw, _ = await _remote(
            droplet,
            f"docker logs --tail {int(lines) * 4} {shlex.quote(VLLM_CONTAINER)} 2>&1 "
            f"| grep -v queue_controller | tail -n {int(lines)}",
            timeout=120,
        )
        if not raw.strip():
            return f"❌ No logs from `{VLLM_CONTAINER}`. Is it running? Try `serving_status`."

        prompt = (
            "You are an SRE triaging a vLLM server on an AMD MI300X. Read these log lines and "
            "answer in under 150 words: is the server healthy, and if not, what is the single most "
            "likely cause and the next command to run?\n\n" + raw[-6000:]
        )
        _, out, _ = await _curl_endpoint(
            droplet,
            "/v1/chat/completions",
            {
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 700,
                "temperature": 0,
            },
            timeout=240,
        )
        try:
            body = json.loads(out)
            verdict = (body["choices"][0]["message"].get("content") or "").strip()
        except (ValueError, KeyError, IndexError):
            return (
                "📡 The endpoint could not analyse its own logs, so here is the raw tail.\n\n"
                f"```\n{_truncate(raw)}\n```"
            )
        return f"📡 `{VLLM_MODEL}` on its own logs:\n\n{verdict}\n\n<details><summary>Log tail</summary>\n\n```\n{_truncate(raw, 3000)}\n```\n\n</details>"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List this server's tools", annotations=READ_ONLY)
async def get_help() -> str:
    """List the tools this server exposes and the configuration it is running with."""
    tools = await mcp.list_tools()
    lines = [
        f"📡 **{MCP_SERVER_NAME}** — one MI300X serving `{VLLM_MODEL}` through vLLM.",
        "",
        f"Scoped to droplets tagged `{DROPLET_TAG}`; SSH as `{SSH_USER}`; checkout at `{REMOTE_WORKDIR}`.",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| `VLLM_IMAGE` | `{VLLM_IMAGE}` |",
        f"| `VLLM_MODEL` | `{VLLM_MODEL}` |",
        f"| `VLLM_PORT` | `{VLLM_PORT}` |",
        f"| `MAX_MODEL_LEN` | `{MAX_MODEL_LEN}` |",
        f"| `GPU_MEMORY_UTILIZATION` | `{GPU_MEMORY_UTILIZATION}` |",
        f"| `LIMIT_MM_PER_PROMPT` | `{LIMIT_MM_PER_PROMPT}` |",
        f"| `HF_CACHE` | `{HF_CACHE}` |",
        "",
    ]
    for tool in tools:
        lines.append(f"- **{tool.name}** — {(tool.description or '').splitlines()[0]}")
    lines += [
        "",
        (
            "No create or destroy tools, by design: both are dollar-per-hour decisions, so they stay "
            "a deliberate step in the DigitalOcean console."
        ),
        "",
        "Powering a droplet off does not stop DigitalOcean billing it.",
        "",
        (
            "`rocm/vllm` at vLLM 0.27.0 cannot load Gemma 4 — it lacks `Gemma4ModelArchConfigConvertor` "
            "and raises on `head_dim`. Run `check_image` before changing `VLLM_IMAGE`."
        ),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
