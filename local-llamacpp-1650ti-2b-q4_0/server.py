"""Local llama.cpp lifecycle and inference MCP server — GTX 1650 Ti, Gemma 4 E2B q4_0.

THIS IS THE FIRST `local` RIG IN THE TREE, and the shape of it is different from
every sibling in one way that matters more than any of the code below: there is
no control plane. The card is in the machine. Nothing here provisions capacity,
waits for it, discovers an endpoint, or releases anything.

So the tools that dominate a sibling's server.py — find_tpu, create_*_queued_resource,
manage_queued_resource, the zone-status skip list — have no analogue and are
deliberately absent. If a `find_*` tool ever appears in this file, the rig has the
wrong name (NAMING.md, "`local` is the absence of a control plane").

What is left is the half that is actually the same everywhere: start the model
server, check it, ask it something, and report what the hardware is doing.

MEMORY, BECAUSE IT IS THE ONLY REAL CONSTRAINT HERE: the artifact is 3.35 GB on
disk but only ~1.31 GiB has to be resident, because per_layer_token_embd (1.93 GB,
58% of the file) is created with TENSOR_READ_LAZY in llama.cpp's
src/models/gemma4.cpp and served by GET_ROWS out of the mmap. Full offload fits a
4 GiB card with ~2.3 GiB to spare. Do not "fix" a memory worry by lowering
N_GPU_LAYERS or passing --no-mmap; the second one breaks the mechanism outright.
See CLAUDE.md.

STATUS 2026-09-03: NOTHING HAS BEEN SERVED. llama.cpp is built at 95ef7fc and the
model file is on disk; no token has been generated through this server.
"""

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

RIG_DIR = Path(__file__).resolve().parent
load_dotenv(RIG_DIR / "tpu.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The registered key prefixes every tool name, so it must match the directory or
# two loaded rigs are indistinguishable at the call site (root CLAUDE.md).
RIG_NAME = RIG_DIR.name
MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", RIG_NAME)

MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it-qat-q4_0-gguf")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8080")
ENDPOINT = os.environ.get("ENDPOINT", f"http://{HOST}:{PORT}")
N_GPU_LAYERS = os.environ.get("N_GPU_LAYERS", "99")
CONTEXT_SIZE = os.environ.get("CONTEXT_SIZE", "8192")
KV_CACHE_TYPE = os.environ.get("KV_CACHE_TYPE", "f16")

RUN_DIR = RIG_DIR / "run"
PID_FILE = RUN_DIR / "llama-server.pid"
LOG_FILE = RUN_DIR / "llama-server.log"

mcp = FastMCP(MCP_SERVER_NAME)


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


def _read_pid() -> Optional[int]:
    """The running llama-server's pid, or None if it is not up.

    Checked against /proc rather than trusted, because a stale pid file outlives
    a Ctrl-C and there is no control plane here to ask for the truth.
    """
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if Path(f"/proc/{pid}").exists() else None


@mcp.tool()
async def gpu_status() -> str:
    """Report the local GPU: name, compute capability, VRAM total/used/free, driver."""
    rc, out, err = await run_command([
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader",
    ], timeout=30)
    if rc != 0:
        return f"❌ nvidia-smi failed (rc={rc}): {err.strip() or out.strip()}"

    line = out.strip()
    body = [f"📡 **GPU** — `{RIG_NAME}`", "", f"```\n{line}\n```", ""]

    # sm_75 is shared with the T4 rigs and the cards are NOT equivalent: the T4 is
    # TU104 and has tensor cores, TU116/TU117 has none. Say so at the point of use
    # rather than hoping the reader checks CLAUDE.md.
    if "1650" in line or "1660" in line:
        body.append(
            "⚠️  GTX 16-series (TU116/TU117): compute capability 7.5 but **no tensor "
            "cores**. Do not compare throughput against the T4-based `g4dn`/`g5g` "
            "rigs on the strength of a matching compute capability."
        )
    return "\n".join(body)


@mcp.tool()
async def model_info() -> str:
    """Report the configured checkpoint, where it is, and the resident-vs-lazy split."""
    path = Path(MODEL_PATH) if MODEL_PATH else None
    if path is None:
        return "❌ MODEL_PATH is unset. It is set in `tpu.env`, which is the source of truth."
    if not path.exists():
        return f"❌ Model file not found: `{path}`\n\nSet `MODEL_PATH` in `tpu.env`."

    size_gb = path.stat().st_size / 1e9
    return (
        f"📡 **Model** — `{RIG_NAME}`\n\n"
        f"- **Name:** `{MODEL_NAME}`\n"
        f"- **Path:** `{path}`\n"
        f"- **On disk:** {size_gb:.2f} GB\n"
        f"- **Quantization slot:** `q4_0` — but the dominant tensor type is **Q6_K**. "
        f"Both embedding tensors are Q6_K (2.257 GB of 3.334 GB); only the ~1.08 GB "
        f"transformer body is actually Q4_0.\n"
        f"- **Resident on GPU:** ~1.31 GiB. `per_layer_token_embd` (1.93 GB, 58% of the "
        f"file) is `TENSOR_READ_LAZY` and is served by GET_ROWS out of the mmap.\n\n"
        f"Run `inspect_gguf.py` to re-derive the split from the artifact rather than "
        f"trusting these numbers."
    )


@mcp.tool()
async def start_model_server(context_size: Optional[str] = None) -> str:
    """Start llama-server on the local GPU. No-op if it is already running."""
    if not LLAMA_SERVER_BIN or not Path(LLAMA_SERVER_BIN).exists():
        return f"❌ llama-server not found at `{LLAMA_SERVER_BIN}`. Set `LLAMA_SERVER_BIN` in `tpu.env`."
    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        return f"❌ Model not found at `{MODEL_PATH}`. Set `MODEL_PATH` in `tpu.env`."

    existing = _read_pid()
    if existing is not None:
        return f"✅ Already running (pid {existing}) at {ENDPOINT}. Use `stop_model_server` first to restart."

    RUN_DIR.mkdir(exist_ok=True)
    cmd = [
        LLAMA_SERVER_BIN,
        "-m", MODEL_PATH,
        "--host", HOST,
        "--port", str(PORT),
        "-ngl", str(N_GPU_LAYERS),
        "-c", str(context_size or CONTEXT_SIZE),
        "-ctk", KV_CACHE_TYPE,
        "-ctv", KV_CACHE_TYPE,
    ]
    # NOTE: no --no-mmap, ever. TENSOR_READ_LAZY "requires mmap for now", so
    # disabling it forces the 1.93 GB per-layer embedding tensor to be
    # materialised and turns a comfortable fit into an OOM.

    with open(LOG_FILE, "ab") as log:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=log, stderr=log, start_new_session=True
        )
    PID_FILE.write_text(str(proc.pid))
    return (
        f"📡 Started llama-server (pid {proc.pid}) → {ENDPOINT}\n\n"
        f"```\n{' '.join(cmd)}\n```\n\n"
        f"Loading is not instant. Poll `model_server_status`; log at `{LOG_FILE}`."
    )


@mcp.tool()
async def stop_model_server() -> str:
    """Stop the running llama-server. Teardown is complete — nothing is billed here."""
    pid = _read_pid()
    if pid is None:
        PID_FILE.unlink(missing_ok=True)
        return "✅ Not running."
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return f"❌ Could not signal pid {pid}: {exc}"
    PID_FILE.unlink(missing_ok=True)
    return f"✅ Sent SIGTERM to llama-server (pid {pid}). VRAM is released on exit."


@mcp.tool()
async def model_server_status() -> str:
    """Check whether llama-server is up and serving at the known local endpoint."""
    pid = _read_pid()
    if pid is None:
        return f"❌ llama-server is not running. Endpoint would be {ENDPOINT}."
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{ENDPOINT}/health")
        if resp.status_code == 200:
            return f"✅ Serving at {ENDPOINT} (pid {pid}). `/health` → 200."
        return f"📡 Process is up (pid {pid}) but `/health` → {resp.status_code}. Still loading?"
    except httpx.HTTPError as exc:
        return f"📡 Process is up (pid {pid}) but {ENDPOINT} is not answering yet ({exc}). Still loading?"


@mcp.tool()
async def query_model(prompt: str, max_tokens: int = 1024) -> str:
    """Send a chat completion to the local endpoint and return the reply.

    Uses /v1/chat/completions, not /v1/completions — raw completions return an
    empty string on `-it` checkpoints (root CLAUDE.md).

    GEMMA 4 IS A REASONING MODEL AND THIS IS THE SECOND WAY TO GET AN EMPTY
    STRING HERE. llama.cpp routes the thinking block to `reasoning_content` and
    leaves `content` empty until it closes. MEASURED 2026-09-03: "Name three TPU
    generations" spent 1274 characters reasoning before writing 22 characters of
    answer, so at max_tokens=64 the reply is `finish_reason: length` with an
    EMPTY content and a truncated thought. That reads as a broken deploy and is
    not one — hence the 1024 default, and the explicit report below.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(f"{ENDPOINT}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            return f"❌ {resp.status_code} from {ENDPOINT}: {resp.text[:500]}"
        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        usage = data.get("usage", {})
        timings = data.get("timings", {})

        if not text and reasoning:
            return (
                f"📡 **Reasoning only — no answer yet.** `finish_reason: "
                f"{choice.get('finish_reason')}` after {usage.get('completion_tokens', '?')} tokens, "
                f"all of them thinking.\n\n"
                f"This is Gemma 4 reasoning, not a broken server. Re-run with a larger "
                f"`max_tokens` (currently {max_tokens}).\n\n"
                f"<details>\n\n{reasoning[:800]}\n\n</details>"
            )

        parts = ["✅ **Reply**", "", text, "", "---"]
        if reasoning:
            parts.append(f"_(plus {len(reasoning)} chars of reasoning, suppressed)_")
        parts.append(
            f"prompt {usage.get('prompt_tokens', '?')} tok · "
            f"completion {usage.get('completion_tokens', '?')} tok"
            + (f" · {timings['predicted_per_second']:.1f} tok/s" if "predicted_per_second" in timings else "")
        )
        return "\n".join(parts)
    except httpx.HTTPError as exc:
        return f"❌ Could not reach {ENDPOINT}: {exc}. Is llama-server running?"
    except (KeyError, IndexError, ValueError) as exc:
        return f"❌ Unexpected response shape from {ENDPOINT}: {exc}"


@mcp.tool()
async def get_help() -> str:
    """List the tools this rig exposes."""
    tools = await mcp.list_tools()
    lines = [f"📡 **{MCP_SERVER_NAME}** — local llama.cpp rig, GTX 1650 Ti, Gemma 4 E2B q4_0", ""]
    for tool in tools:
        lines.append(f"- **{tool.name}** — {(tool.description or '').splitlines()[0]}")
    lines += [
        "",
        "No provisioning tools, by design: the hardware is local and there is no "
        "control plane to call. See NAMING.md, \"`local` is the absence of a control plane\".",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
