"""Local llama.cpp lifecycle and inference MCP server — CPU only, Gemma 4 E2B q4_0.

A `local` rig: there is no control plane. The CPU is in the machine. Nothing here
provisions capacity, waits for it, discovers an endpoint, or releases anything, so
the tools that dominate a cloud sibling's server.py — find_tpu,
create_*_queued_resource, manage_queued_resource, the zone-status skip list —
are deliberately absent (NAMING.md, "`local` is the absence of a control plane").

CPU ONLY, AND ENFORCED RATHER THAN CONFIGURED. Retargeted 2026-09-15 from a
byte-identical copy of local-llamacpp-1650ti-2b-q4_0. `-ngl 0` is hardcoded, the
child gets an empty CUDA_VISIBLE_DEVICES, and `start_model_server` refuses a
llama-server binary built with a GPU backend. A CPU number from a process that
could have offloaded to a GPU is not a CPU number.

MEMORY: per_layer_token_embd (1.93 GB, 58% of the file) is created with
TENSOR_READ_LAZY in llama.cpp's src/models/gemma4.cpp and served by GET_ROWS out
of the mmap. On a GPU that decided whether the model fit the card; on a CPU it
decides how much of the file stays hot in the page cache. --no-mmap breaks the
mechanism outright either way. See CLAUDE.md.

THIS RIG IS ONE ARM OF A CONTROL. `local-llamacpp-1650ti-2b-q4_0` is the other:
same GGUF, same llama.cpp checkout, same port, same harness, same prompts, run
alternately so that the device is the only thing that differs. Nothing in an HTTP
response says which arm answered, so the arm is read off the running process --
see attest.py, which every status and query path here goes through.

STATUS 2026-09-22: SERVING on llama.cpp f95b0d9, rebuilt from clean after the host
moved to Debian sid (gcc 16.2, CUDA 13.4). This arm lists no compute devices; the
GPU arm lists CUDA0.

THE HOST CPU WAS MISIDENTIFIED UNTIL 2026-09-22. This rig documented an i7-1360P
with a hybrid 4P+8E topology; the machine is a homogeneous 6-core/12-thread
i7-10750H with no avx_vnni. The 18-cell thread/affinity sweep was interpreted
through that wrong die and is quarantined -- THREADS/THREADS_BATCH are now derived
from real topology (6/12) and spot-checked, not swept. The host also throttles hard
enough that an identical config re-run cold moved 19% on decode, so no absolute
figure from this rig is quotable without a cooldown protocol.
"""

import asyncio
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
CONTEXT_SIZE = os.environ.get("CONTEXT_SIZE", "8192")
KV_CACHE_TYPE = os.environ.get("KV_CACHE_TYPE", "f16")
FLASH_ATTENTION = os.environ.get("FLASH_ATTENTION", "1")
# 6 = physical cores, 12 = logical, on this host's i7-10750H. These mirror tpu.env,
# which is the source of truth; they were 4/8, derived from a 4P+8E die this
# machine is not. Keep them agreeing with tpu.env.
THREADS = os.environ.get("THREADS", "6")
THREADS_BATCH = os.environ.get("THREADS_BATCH", "12")
PARALLEL_SLOTS = os.environ.get("PARALLEL_SLOTS", "1")
METRICS = os.environ.get("METRICS", "0")

# Deliberately not read from the environment. See the module docstring.
N_GPU_LAYERS = "0"

# ggml backends that would put work on something other than the CPU. A build
# links one statically (visible to ldd) or ships it as a shared library next to
# the binary (visible to a directory listing, and dlopen'd at runtime where ldd
# cannot see it), so both are checked.
GPU_BACKEND_MARKERS = ("ggml-cuda", "ggml-vulkan", "ggml-sycl", "ggml-hip", "ggml-musa",
                       "ggml-cann", "ggml-opencl", "ggml-metal", "libcudart", "libcublas")

RUN_DIR = RIG_DIR / "run"
PID_FILE = RUN_DIR / "llama-server.pid"
LOG_FILE = RUN_DIR / "llama-server.log"

mcp = MCPServer(MCP_SERVER_NAME)


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
    """The pid this server recorded when it started llama-server, if still alive.

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


def _read_pid() -> Optional[int]:
    """The running llama-server's pid, or None if nothing is serving.

    Two sources, in order: the pid file `start_model_server` writes, then the
    process actually holding the listening socket on PORT.

    THE PID FILE ALONE IS NOT ENOUGH, AND IT FAILS IN BOTH DIRECTIONS. It goes
    stale when a server dies, and it is simply ABSENT whenever the server was
    started any other way — which is the normal case here: `make serve` is
    foreground by design and writes no pid file at all.
    """
    pid = _pidfile_pid()
    if pid is not None:
        return pid
    try:
        return _pid_owning_port(int(PORT))
    except ValueError:  # PORT unparseable; nothing to discover against
        return None


def _parse_cpulist(text: str) -> list[int]:
    """Expand a kernel cpulist ("0-7,12") into logical cpu ids."""
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
    """CPU model, SIMD flags, hybrid topology and RAM, read from /proc and /sys."""
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
            meminfo[key.strip()] = int(parts[0])  # kB

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
        body.append(f"- **Hybrid:** {len(facts['p_cores'])} P-core threads, "
                    f"{len(facts['e_cores'])} E-core threads")
    body += [
        f"- **SIMD:** {' '.join(facts['simd']) or 'none reported'}",
        f"- **RAM:** {facts['mem_available_gib']:.2f} GiB available of {facts['mem_total_gib']:.2f} GiB",
        f"- **Threads configured:** decode `-t {THREADS}`, prefill `-tb {THREADS_BATCH}` "
        f"(DERIVED from this host's topology — physical cores for decode, logical for "
        f"prefill — and spot-checked 2026-09-22, NOT swept)",
        "- **CPU affinity:** not set. The 2026-09-16 sweep that called it the largest "
        "lever is QUARANTINED: it described a 4P+8E die this machine is not. Whether "
        "pinning helps here is open.",
        "- **Measurement health:** this host throttles hard, and an identical config "
        "re-run cold moved 19% on decode. Run-to-run drift exceeds every lever measured "
        "so far — no absolute t/s from this rig is quotable without a cooldown protocol.",
    ]
    if not any(f.startswith("avx512") for f in facts["simd"]):
        body += ["", "⚠️  No AVX-512. llama.cpp takes its AVX2 kernels here; do not compare "
                     "against a number from an AVX-512 host on the strength of the same build flags."]
    return "\n".join(body)


@mcp.tool()
async def model_info() -> str:
    """Report the configured checkpoint, where it is, and the resident-vs-lazy split."""
    path = Path(MODEL_PATH) if MODEL_PATH else None
    if path is None:
        return "❌ MODEL_PATH is unset. It is set in `tpu.env`, which is the source of truth."
    if not path.exists():
        return f"❌ Model file not found: `{path}`\n\nRun `make download`, or set `MODEL_PATH` in `tpu.env`."

    size_gb = path.stat().st_size / 1e9
    return (
        f"📡 **Model** — `{RIG_NAME}`\n\n"
        f"- **Name:** `{MODEL_NAME}`\n"
        f"- **Path:** `{path}`\n"
        f"- **On disk:** {size_gb:.2f} GB\n"
        f"- **Quantization slot:** `q4_0` — but the dominant tensor type is **Q6_K**. "
        f"Both embedding tensors are Q6_K (2.257 GB of 3.334 GB); only the ~1.08 GB "
        f"transformer body is actually Q4_0.\n"
        f"- **Touched every token:** ~1.31 GiB. `per_layer_token_embd` (1.93 GB, 58% of the "
        f"file) is `TENSOR_READ_LAZY` and is served by GET_ROWS out of the mmap, a few rows "
        f"per token.\n\n"
        f"Run `inspect_gguf.py` to re-derive the split from the artifact rather than "
        f"trusting these numbers."
    )


async def _gpu_backends(binary: str) -> list[str]:
    """GPU ggml backends a llama-server binary would load, from its directory and ldd."""
    found: set[str] = set()
    bin_dir = Path(binary).resolve().parent
    try:
        for entry in bin_dir.iterdir():
            found.update(m for m in GPU_BACKEND_MARKERS if m in entry.name)
    except OSError:
        pass
    rc, out, _ = await run_command(["ldd", binary], timeout=30)
    if rc == 0:
        found.update(m for m in GPU_BACKEND_MARKERS if m in out)
    return sorted(found)


def _server_env() -> dict[str, str]:
    """The child environment: this process's, with every CUDA device hidden."""
    return {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


def _server_command(context_size: Optional[str] = None) -> list[str]:
    """The llama-server argv. Must carry the same flags as `make serve`.

    A test enforces that parity: on 2026-09-10 the GPU sibling's list lacked -fa,
    -t and --parallel and the MCP-started server came up with 4 slots and 6
    threads.
    """
    cmd = [
        LLAMA_SERVER_BIN,
        "-m", MODEL_PATH,
        "--host", HOST,
        "--port", str(PORT),
        "-ngl", N_GPU_LAYERS,
        "-c", str(context_size or CONTEXT_SIZE),
        "-ctk", KV_CACHE_TYPE,
        "-ctv", KV_CACHE_TYPE,
        "-fa", FLASH_ATTENTION,
        "-t", THREADS,
        "-tb", THREADS_BATCH,
        # llama.cpp splits -c across slots, and its default is more than one.
        "--parallel", PARALLEL_SLOTS,
    ]
    # llama.cpp serves /metrics only when asked; without this it answers 501.
    if METRICS == "1":
        cmd.append("--metrics")
    # NOTE: no --no-mmap, ever. TENSOR_READ_LAZY "requires mmap for now".
    return cmd


def _spawn_detached(cmd: list[str]) -> int:
    """Start llama-server as a daemon that OUTLIVES this process, and return its pid.

    DELIBERATELY NOT asyncio.create_subprocess_exec, which is what `run_command`
    uses and what this did until 2026-09-16. Asyncio's subprocess transport KILLS
    a live child when it is torn down: BaseSubprocessTransport.__del__ calls
    close(), and close() calls _proc.kill() on a child that has not exited.
    `start_new_session=True` does not save it -- MEASURED 2026-09-16, a spawned
    `sleep 60` is dead within a second of the interpreter exiting.

    So a server started through this tool did not survive, while `make serve` (a
    plain exec) was unaffected. That is very likely why "no pid file" is
    documented here as the NORMAL case: the path that writes one could not leave
    a server running.

    subprocess.Popen only warns when collected with a live child; it does not
    kill it. Still no shell -- the rule in CLAUDE.md is against shell=True, not
    against the module.
    """
    with open(LOG_FILE, "ab") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True, env=_server_env(),
        )
    return proc.pid


@mcp.tool()
async def start_model_server(context_size: Optional[str] = None) -> str:
    """Start llama-server on the local CPU. No-op if it is already running."""
    if not LLAMA_SERVER_BIN or not Path(LLAMA_SERVER_BIN).exists():
        return (f"❌ llama-server not found at `{LLAMA_SERVER_BIN}`. Run `make build`, "
                f"or set `LLAMA_SERVER_BIN` in `tpu.env`.")
    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        return f"❌ Model not found at `{MODEL_PATH}`. Run `make download`, or set `MODEL_PATH` in `tpu.env`."

    existing = _read_pid()
    if existing is not None:
        return f"✅ Already running (pid {existing}) at {ENDPOINT}. Use `stop_model_server` first to restart."

    backends = await _gpu_backends(LLAMA_SERVER_BIN)
    if backends:
        return (
            f"❌ `{LLAMA_SERVER_BIN}` is a GPU build ({', '.join(backends)}). This rig is CPU only. "
            f"Point `LLAMA_SERVER_BIN` at a `GGML_CUDA=OFF` build — `make build` makes one."
        )

    RUN_DIR.mkdir(exist_ok=True)
    cmd = _server_command(context_size)
    pid = _spawn_detached(cmd)
    PID_FILE.write_text(str(pid))
    return (
        f"📡 Started llama-server (pid {pid}) → {ENDPOINT}\n\n"
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
    # `_read_pid` also finds a server this process did not start, so this stops
    # a `make serve` too.
    discovered = _pidfile_pid() is None
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return f"❌ Could not signal pid {pid}: {exc}"
    PID_FILE.unlink(missing_ok=True)
    origin = f" — found on port {PORT}, not started through this server" if discovered else ""
    return f"✅ Sent SIGTERM to llama-server (pid {pid}){origin}. Memory is released on exit."


@mcp.tool()
async def model_server_status() -> str:
    """Check whether llama-server is up, and whether the process serving is THIS arm."""
    # /health DECIDES whether something is up. Gating on the pid first is what
    # made the GPU sibling return ❌ against a healthy server on 2026-09-08.
    #
    # BUT UP IS NOT THE SAME QUESTION AS OURS. The GPU twin serves the same model
    # on this same port, and until 2026-09-16 this tool answered ✅ against it --
    # annotating "not started through this server" while still leading with the
    # tick. In a control that is the whole failure mode, so the device now
    # decides the prefix.
    pid = _read_pid()
    if pid is None:
        who = "pid unknown"
    elif _pidfile_pid() == pid:
        who = f"pid {pid}"
    else:
        who = f"pid {pid}, found on port {PORT} — not started through this server, so `{LOG_FILE}` may not be its log"

    try:
        att = attest_port(int(PORT))
    except ValueError:
        att = {"serving": False, "port": PORT}
    wrong_arm = mismatch(att)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{ENDPOINT}/health")
    except httpx.HTTPError as exc:
        if pid is None:
            return f"❌ llama-server is not running. Endpoint would be {ENDPOINT}."
        return f"📡 Process is up ({who}) but {ENDPOINT} is not answering yet ({exc}). Still loading?"

    if resp.status_code != 200:
        return f"📡 Reachable at {ENDPOINT} ({who}) but `/health` → {resp.status_code}. Still loading?"

    if wrong_arm:
        return (
            f"❌ **Healthy, but it is not this arm.** {wrong_arm}\n\n"
            f"`/health` → 200 at {ENDPOINT}, so something IS serving — it is the "
            f"`{att.get('device')}` arm, and this rig is `{EXPECTED_DEVICE}`. A number "
            f"taken now would be labelled with the wrong device.\n\n"
            f"Stop it (`stop_model_server`) and start this arm, or measure it from the "
            f"rig it belongs to."
        )
    return (f"✅ Serving at {ENDPOINT} ({who}). `/health` → 200.\n\n"
            f"Arm attested **{att['device']}** — {describe(att)}")


@mcp.tool()
async def query_model(prompt: str, max_tokens: int = 1024) -> str:
    """Send a chat completion to the local endpoint and return the reply.

    Uses /v1/chat/completions, not /v1/completions — raw completions return an
    empty string on `-it` checkpoints (root CLAUDE.md).

    GEMMA 4 IS A REASONING MODEL AND THIS IS THE SECOND WAY TO GET AN EMPTY
    STRING HERE. llama.cpp routes the thinking block to `reasoning_content` and
    leaves `content` empty until it closes. MEASURED 2026-09-03 on the GPU
    sibling: "Name three TPU generations" spent 1274 characters reasoning before
    22 characters of answer. Hence the 1024 default and the explicit report below.

    The 900 s timeout is sized for a CPU, not measured on one: 1024 tokens of
    thinking at a single-digit decode rate is minutes, not seconds.
    """
    # The twin serves the same model on this same port, so "did I get a reply"
    # does not establish which device produced it. Attest before spending 900 s.
    try:
        wrong_arm = mismatch(attest_port(int(PORT)))
    except ValueError:
        wrong_arm = None
    if wrong_arm:
        return (f"❌ Refusing to query: {wrong_arm}. This rig is the `{EXPECTED_DEVICE}` arm. "
                f"Use `model_server_status` for the full attestation.")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
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
async def attest_arm() -> str:
    """Report which binary is answering on the endpoint, read from /proc — not from config.

    The control's one silent failure: this rig and `local-llamacpp-1650ti-2b-q4_0`
    serve the same GGUF on the same port, and an HTTP response says nothing about
    which device produced it. Everything below is read off the live process, so a
    disagreement between `tpu.env` and reality shows up rather than being assumed
    away.
    """
    try:
        att = attest_port(int(PORT))
    except ValueError:
        return f"❌ PORT is not an integer: `{PORT}`"
    if not att["serving"]:
        return (f"📡 Nothing is listening on port {PORT}. Expected arm: "
                f"**{EXPECTED_DEVICE}** (`{RIG_NAME}`).")

    verdict = mismatch(att)
    lines = [
        f"{'❌' if verdict else '✅'} **Arm attested: {att['device']}** "
        f"(this rig expects **{EXPECTED_DEVICE}**)",
        "",
        f"- **pid:** {att['pid']}",
        f"- **exe:** `{att['exe']}`",
        f"- **sha256:** `{att['exe_sha256'][:16]}…` — the pairing identity. Two arms are "
        f"comparable only if they were built from one commit; the hash is what actually ran.",
        f"- **`-ngl`:** {att['n_gpu_layers']}",
        f"- **GPU libraries mapped:** {', '.join(att['gpu_libs']) if att['gpu_libs'] else 'none'}",
        f"- **CUDA_VISIBLE_DEVICES:** "
        f"{'(empty — devices hidden)' if att['cuda_visible_devices'] == '' else att['cuda_visible_devices'] or '(unset)'}",
        f"- **model:** `{att['model_arg']}`",
        f"- **threads:** `-t {att['threads']}` `-tb {att['threads_batch']}` · **ctx:** {att['ctx_size']}",
    ]
    if verdict:
        lines += ["", f"⚠️  {verdict}"]
    if att["device"] == "mixed":
        lines += ["", "⚠️  **mixed**: a GPU backend is mapped into the process AND `-ngl` is 0. "
                      "That computes on the CPU but is not a clean CPU arm — the device is "
                      "initialised and llama.cpp can still move large prefill batches onto it."]
    return "\n".join(lines)


@mcp.tool()
async def get_help() -> str:
    """List the tools this rig exposes."""
    tools = await mcp.list_tools()
    lines = [f"📡 **{MCP_SERVER_NAME}** — local llama.cpp rig, CPU only, Gemma 4 E2B q4_0", ""]
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
