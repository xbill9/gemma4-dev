"""Local LiteRT-LM lifecycle and inference MCP server — CPU only, Gemma 4 E2B, QAT grid int4.

A `local` rig: there is no control plane. The CPU is in the machine. Nothing here
provisions capacity, waits for it, discovers an endpoint, or releases anything
(NAMING.md, "`local` is the absence of a control plane").

SISTER: local-llamacpp-i71360p-2b-q4_0, same host. Both serve Google's QAT weights on
their trained 4-bit grid, every weight matrix and both embedding tables included. Slot
2 (runtime) differs, and so does slot 5: LiteRT stores each block's scale at bf16
precision where Q4_0 stores fp16, so the levels are identical and the scales are not
(benchmarks/runs/2026-10-15-qat-grid-bf16-scales-i71360p).

RUNTIME: `litert-lm serve` (pip package litert-lm), an OpenAI-compatible server that
loads models lazily, on the first request that names them. Its settings come from a
config JSON keyed by model id; this rig writes its own (`run/litertlm-config.json`,
rendered from tpu.env) and passes it with --config, so nothing in ~/.litert-lm is read
or written. The model id of a file named `model.litertlm` is its directory name with
`--` read as `/`; requests name the model by its absolute path.

CPU ONLY, AND ENFORCED RATHER THAN CONFIGURED: `backend` is written as "cpu" by this
file, not read from tpu.env, and attest.py reports `mixed` if a GPU driver is mapped.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from attest import EXPECTED_DEVICE, attest_port, describe, mismatch
from attest import pid_owning_port as _pid_owning_port

RIG_DIR = Path(__file__).resolve().parent
load_dotenv(RIG_DIR / "tpu.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RIG_NAME = RIG_DIR.name
MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", RIG_NAME)

MODEL_NAME = os.environ.get("MODEL_NAME", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
LITERT_LM_BIN = os.environ.get("LITERT_LM_BIN", "litert-lm")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8092")
ENDPOINT = os.environ.get("ENDPOINT", f"http://{HOST}:{PORT}")
MAX_NUM_TOKENS = os.environ.get("MAX_NUM_TOKENS", "8192")
CPU_THREAD_COUNT = os.environ.get("CPU_THREAD_COUNT", "8")
THINKING = os.environ.get("THINKING", "false")
COMPILE_CACHE = os.environ.get("COMPILE_CACHE", "disk")

# Deliberately not read from the environment. See the module docstring.
BACKEND = "cpu"

RUN_DIR = RIG_DIR / "run"
PID_FILE = RUN_DIR / "litert-lm.pid"
LOG_FILE = RUN_DIR / "litert-lm.log"
CONFIG_FILE = RUN_DIR / "litertlm-config.json"

mcp = MCPServer(MCP_SERVER_NAME)


def model_id(path: str = "") -> str:
    """The id `litert-lm` gives a file: for `.../a--b/model.litertlm` it is `a/b`."""
    p = Path(path or MODEL_PATH)
    if p.name == "model.litertlm":
        return p.parent.name.replace("--", "/")
    return p.name


def render_config() -> dict:
    """The --config JSON. Must agree with `make config`; a test enforces it."""
    return {
        "models": {
            model_id(): {
                "backend": BACKEND,
                "cpu_thread_count": int(CPU_THREAD_COUNT),
                "max_num_tokens": int(MAX_NUM_TOKENS),
                "thinking": THINKING.lower() == "true",
                "cache": COMPILE_CACHE,
            }
        }
    }


def write_config() -> Path:
    RUN_DIR.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(render_config(), indent=1) + "\n")
    return CONFIG_FILE


async def run_command(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run a command with no shell. Never shell=True — see CLAUDE.md."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    except asyncio.TimeoutError:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"


def _pidfile_pid() -> Optional[int]:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if Path(f"/proc/{pid}").exists() else None


def _read_pid() -> Optional[int]:
    """The serving pid: the pid file first, then whoever holds PORT (`make serve` writes none)."""
    pid = _pidfile_pid()
    if pid is not None:
        return pid
    try:
        return _pid_owning_port(int(PORT))
    except ValueError:
        return None


def _parse_cpulist(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _cpu_facts() -> dict:
    cpuinfo = _read_text("/proc/cpuinfo")
    model, flags = "unknown", set()
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "model name" and model == "unknown":
            model = value.strip()
        elif key == "flags" and not flags:
            flags = set(value.split())
    meminfo = {}
    for line in _read_text("/proc/meminfo").splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if parts and parts[0].isdigit():
            meminfo[key.strip()] = int(parts[0])
    return {
        "model": model,
        "logical": os.cpu_count() or 0,
        "p_cores": _parse_cpulist(_read_text("/sys/devices/cpu_core/cpus")),
        "e_cores": _parse_cpulist(_read_text("/sys/devices/cpu_atom/cpus")),
        "simd": sorted(f for f in flags if f.startswith(("avx", "amx"))),
        "mem_total_gib": meminfo.get("MemTotal", 0) / 2**20,
        "mem_available_gib": meminfo.get("MemAvailable", 0) / 2**20,
    }


@mcp.tool()
async def cpu_status() -> str:
    """Report the local CPU: model, hybrid P/E topology, SIMD flags, RAM total/available."""
    facts = _cpu_facts()
    body = [
        f"📡 **CPU** — `{RIG_NAME}`",
        "",
        f"- **Model:** {facts['model']}",
        f"- **Logical CPUs:** {facts['logical']}",
    ]
    if facts["p_cores"] or facts["e_cores"]:
        body.append(f"- **Hybrid:** {len(facts['p_cores'])} P-core threads, {len(facts['e_cores'])} E-core threads")
    body += [
        f"- **SIMD:** {' '.join(facts['simd']) or 'none reported'}",
        f"- **RAM:** {facts['mem_available_gib']:.2f} GiB available of {facts['mem_total_gib']:.2f} GiB",
        f"- **Threads configured:** `cpu_thread_count` {CPU_THREAD_COUNT} (see tpu.env)",
        "- **CPU affinity:** not set. The kernel places XNNPACK's threads.",
    ]
    return "\n".join(body)


@mcp.tool()
async def model_info() -> str:
    """Report the configured .litertlm, where it is, and how it was built."""
    path = Path(MODEL_PATH) if MODEL_PATH else None
    if path is None:
        return "❌ MODEL_PATH is unset. It is set in `tpu.env`, which is the source of truth."
    if not path.exists():
        return f"❌ Model file not found: `{path}`\n\nRun `make model` (builds it from the QAT checkpoint)."
    return (
        f"📡 **Model** — `{RIG_NAME}`\n\n"
        f"- **Name:** `{MODEL_NAME}`\n"
        f"- **Model id (litert-lm):** `{model_id()}`\n"
        f"- **Path:** `{path}`\n"
        f"- **On disk:** {path.stat().st_size / 1e9:.2f} GB\n"
        f"- **Encoding slot:** `q4wi4b32` — every FULLY_CONNECTED and EMBEDDING_LOOKUP weight is "
        f"LiteRT dynamic int4 in blocks of 32, each block's scale the trained QAT step at bf16 "
        f"precision. Text only.\n\n"
        f"Run `make info` to re-derive the storage from the artifact rather than trusting this."
    )


def _server_command() -> list[str]:
    """The `litert-lm serve` argv. Must carry the same flags as `make serve`; a test enforces it."""
    return [LITERT_LM_BIN, "serve", "--config", str(CONFIG_FILE), "--host", HOST, "--port", str(PORT)]


def _spawn_detached(cmd: list[str]) -> int:
    """Start the server so it OUTLIVES this process (asyncio's transport kills live children)."""
    with open(LOG_FILE, "ab") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, cwd=RIG_DIR
        )
    return proc.pid


@mcp.tool()
async def start_model_server() -> str:
    """Start `litert-lm serve` on the local CPU. No-op if it is already running.

    The model loads on the FIRST request, not here: `model_server_status` turns ✅ as soon
    as the port answers, and the first `query_model` pays the load.
    """
    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        return f"❌ Model not found at `{MODEL_PATH}`. Run `make model`, or set `MODEL_PATH` in `tpu.env`."
    existing = _read_pid()
    if existing is not None:
        return f"✅ Already running (pid {existing}) at {ENDPOINT}. Use `stop_model_server` first to restart."
    write_config()
    cmd = _server_command()
    pid = _spawn_detached(cmd)
    PID_FILE.write_text(str(pid))
    return (
        f"📡 Started litert-lm serve (pid {pid}) → {ENDPOINT}\n\n"
        f"```\n{' '.join(cmd)}\n```\n\n"
        f"Config `{CONFIG_FILE}`. Poll `model_server_status`; log at `{LOG_FILE}`."
    )


@mcp.tool()
async def stop_model_server() -> str:
    """Stop the running `litert-lm serve`. Teardown is complete — nothing is billed here."""
    pid = _read_pid()
    if pid is None:
        PID_FILE.unlink(missing_ok=True)
        return "✅ Not running."
    discovered = _pidfile_pid() is None
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return f"❌ Could not signal pid {pid}: {exc}"
    PID_FILE.unlink(missing_ok=True)
    origin = f" — found on port {PORT}, not started through this server" if discovered else ""
    return f"✅ Sent SIGTERM to litert-lm serve (pid {pid}){origin}."


@mcp.tool()
async def model_server_status() -> str:
    """Check whether `litert-lm serve` is up, and whether the process serving is THIS arm."""
    pid = _read_pid()
    try:
        att = attest_port(int(PORT), model_id())
    except ValueError:
        att = {"serving": False, "port": PORT}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{ENDPOINT}/v1/models")
    except httpx.HTTPError as exc:
        if pid is None:
            return f"❌ litert-lm serve is not running. Endpoint would be {ENDPOINT}."
        return f"📡 Process is up (pid {pid}) but {ENDPOINT} is not answering yet ({exc})."
    if resp.status_code != 200:
        return f"📡 Reachable at {ENDPOINT} (pid {pid}) but `/v1/models` → {resp.status_code}."
    wrong_arm = mismatch(att)
    if wrong_arm:
        return (
            f"❌ **Answering, but it is not this arm.** {wrong_arm}\n\n"
            f"A number taken now would be labelled with the wrong device or runtime."
        )
    return (
        f"✅ Serving at {ENDPOINT} (pid {pid}). `/v1/models` → 200.\n\n"
        f"Arm attested **{att['device']}** — {describe(att)}\n\n"
        f"The model loads on the first request that names it."
    )


@mcp.tool()
async def query_model(prompt: str, max_tokens: int = 1024, thinking: bool = False) -> str:
    """Send a chat completion to the local endpoint and return the reply.

    `thinking=False` sends `reasoning_effort: "none"`, so Gemma 4 answers directly. The
    first call after a start also loads the model, so it is slower than the rest.
    """
    try:
        wrong_arm = mismatch(attest_port(int(PORT), model_id()))
    except ValueError:
        wrong_arm = None
    if wrong_arm:
        return (
            f"❌ Refusing to query: {wrong_arm}. This rig is the `{EXPECTED_DEVICE}` arm. "
            f"Use `model_server_status` for the full attestation."
        )
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "reasoning_effort": "high" if thinking else "none",
    }
    try:
        async with httpx.AsyncClient(timeout=900) as client:
            resp = await client.post(f"{ENDPOINT}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            return f"❌ {resp.status_code} from {ENDPOINT}: {resp.text[:500]}"
        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usage = data.get("usage", {})
        if not text and reasoning:
            return (
                f"📡 **Reasoning only — no answer yet.** `finish_reason: {choice.get('finish_reason')}`. "
                f"Re-run with a larger `max_tokens` (currently {max_tokens}).\n\n"
                f"<details>\n\n{reasoning[:800]}\n\n</details>"
            )
        parts = ["✅ **Reply**", "", text, "", "---"]
        if reasoning:
            parts.append(f"_(plus {len(reasoning)} chars of reasoning, suppressed)_")
        parts.append(
            f"prompt {usage.get('prompt_tokens', '?')} tok · completion {usage.get('completion_tokens', '?')} tok"
        )
        return "\n".join(parts)
    except httpx.HTTPError as exc:
        return f"❌ Could not reach {ENDPOINT}: {exc}. Is litert-lm serve running?"
    except (KeyError, IndexError, ValueError) as exc:
        return f"❌ Unexpected response shape from {ENDPOINT}: {exc}"


@mcp.tool()
async def attest_arm() -> str:
    """Report which process is answering on the endpoint and on what device, read from /proc."""
    try:
        att = attest_port(int(PORT), model_id())
    except ValueError:
        return f"❌ PORT is not an integer: `{PORT}`"
    if not att["serving"]:
        return f"📡 Nothing is listening on port {PORT}. Expected arm: **{EXPECTED_DEVICE}** (`{RIG_NAME}`)."
    verdict = mismatch(att)
    lines = [
        f"{'❌' if verdict else '✅'} **Arm attested: {att['device']}** (this rig expects **{EXPECTED_DEVICE}**)",
        "",
        f"- **pid:** {att['pid']}",
        f"- **litert-lm serve:** {att['litertlm_serve']}",
        f"- **config:** `{att['config']}`",
        f"- **backend for `{model_id()}`:** {att['backend']}",
        f"- **cpu_thread_count:** {att['cpu_thread_count']} · **max_num_tokens:** {att['max_num_tokens']}",
        f"- **GPU libraries mapped:** {', '.join(att['gpu_libs']) if att['gpu_libs'] else 'none'}",
    ]
    if verdict:
        lines += ["", f"⚠️  {verdict}"]
    return "\n".join(lines)


@mcp.tool()
async def get_help() -> str:
    """List the tools this rig exposes."""
    tools = await mcp.list_tools()
    lines = [f"📡 **{MCP_SERVER_NAME}** — local LiteRT-LM rig, CPU only, Gemma 4 E2B QAT int4", ""]
    for tool in tools:
        lines.append(f"- **{tool.name}** — {(tool.description or '').splitlines()[0]}")
    lines += ["", "No provisioning tools, by design: the hardware is local and there is no control plane."]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
