"""Local vLLM-on-CPU lifecycle MCP server — no accelerator, Gemma 4 E2B.

STATUS 2026-09-04: scaffolded, nothing served, and the budget is CLOSE rather
than hopeless. `check_host_capacity` has the live arithmetic.

WHAT THIS RIG IS FOR. It is the runtime control for `local-jax-cpu-2b` — same
host, same checkpoint, no chip — and the only place a vLLM number could be
attributed entirely to the engine rather than to silicon.

THERE IS NO CONTROL PLANE. No EC2 launch, no AMI resolution, no spot handling, no
SSM, no Secrets Manager, no systemd unit, no boto3, and no deploy tool: the
payload runs in place. `tests/test_server.py::TestNoCloudControlPlane` asserts
the ABSENCE of that vocabulary and is the most load-bearing class in the suite —
this directory was a verbatim copy of `gpu-jax-g4dn-2b` until 2026-09-04, and a
fork of a cloud rig keeps passing its own tests while describing hardware that
does not exist. Dead cloud code here is worse than unused; it reads as live
configuration.

MEMORY IS THE ONLY REAL CONSTRAINT, AND ITS FAILURE MODE IS WORSE THAN A CLOUD
RIG'S. Exceeding a quota is refused at the API. Exceeding host RAM is *accepted*
and paid for in swap, so a thrashing serve is indistinguishable from a loading
one — this host has 15.4 GB of free swap to thrash into. That is why
`check_host_capacity` is a tool rather than a note, and why `start_vllm_server`
REFUSES rather than starting.

THREE THINGS AN EARLIER VERSION OF THIS FILE ASSERTED AND GOT WRONG. All three
were stated as findings when they were assumptions, and all three made the rig
look less feasible than it is. Corrected 2026-09-04 against primary sources:

  1. "No AVX512-BF16, so the backend upcasts bf16 to fp32 and the weights
     double." FALSE. vLLM's `CpuPlatform.supported_dtypes` returns
     [bfloat16, float16, float32] for x86 unconditionally — its own comment
     reads "x86/aarch64 CPU has supported both bf16 and fp16 natively". AVX512-BF16
     governs the speed of a bf16 datapath, not the dtype's byte count. The
     headline shortfall was inflated 2x by this.
  2. "AVX2 needs a from-source build with AVX512 disabled", framed as a hack.
     MISLEADING. `cmake/cpu_extension.cmake` carries first-class
     CXX_COMPILE_FLAGS_AVX2 beside the AVX512 set and dispatches between them.
     The only hard x86 requirement is gcc/g++ >= 12.3, and this host has 14.2.
     The build is from source only because no CPU wheel is published.
  3. "E2B under vLLM needs ~2.9 GB of w4a16 weights." FALSE, and off by ~3x. The
     real artifact is 8.32 GB (MEASURED from the Hub; `@MODELS.md` records 8.15 GB
     resident) because the checkpoint's own `ignore` list keeps the vision tower
     and the embeddings at bf16 — only the linears are packed. A `weights / 4`
     estimate is not usable for this family.

WHAT SURVIVES ALL THREE CORRECTIONS: this rig still cannot fit the GTX 1650 Ti's
4096 MiB, and by a wider margin than first stated — 8.15 GB of w4a16 weights, not
2.9. The reason is structural: vLLM holds the 2.349 B-parameter per-layer
embedding table resident, where llama.cpp creates it with TENSOR_READ_LAZY and
gathers rows out of the mmap so it never reaches the device at all.
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

RIG_NAME = RIG_DIR.name
MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", RIG_NAME)

MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it")
MODEL_SAFETENSORS_BYTES = int(os.environ.get("MODEL_SAFETENSORS_BYTES", "10246621918"))
MODEL_TOWERS_BYTES = int(os.environ.get("MODEL_TOWERS_BYTES", "951000000"))
MODEL_W4A16_NAME = os.environ.get("MODEL_W4A16_NAME", "google/gemma-4-E2B-it-qat-w4a16-ct")
MODEL_W4A16_BYTES = int(os.environ.get("MODEL_W4A16_BYTES", "8316306646"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "1"))
KVCACHE_SPACE_GIB = int(os.environ.get("VLLM_CPU_KVCACHE_SPACE", "1"))
OMP_THREADS_BIND = os.environ.get("VLLM_CPU_OMP_THREADS_BIND", "")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8001")
ENDPOINT = os.environ.get("ENDPOINT", f"http://{HOST}:{PORT}")

RUN_DIR = RIG_DIR / "run"
PID_FILE = RUN_DIR / "vllm.pid"
LOG_FILE = RUN_DIR / "vllm.log"

GIB = 1024 ** 3
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
    """The running server's pid, or None. Checked against /proc rather than
    trusted: a stale pid file outlives a Ctrl-C and there is no control plane
    here to ask for the truth."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if Path(f"/proc/{pid}").exists() else None


def _meminfo() -> dict:
    """Live host memory, in bytes.

    MemAvailable, not MemFree. MemFree excludes reclaimable page cache and on a
    desktop reads catastrophically low for no reason; MemAvailable is the
    kernel's own estimate of what a new allocation can actually get, and it is
    the number this rig's decisions turn on.
    """
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            out[key] = int(rest.strip().split()[0]) * 1024
    return out


def _cpu_flags() -> set:
    """The ISA flags of core 0, from /proc/cpuinfo."""
    flags = set()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    return flags


def _capacity() -> dict:
    """The whole of this rig's go/no-go arithmetic, in one place.

    WEIGHTS ARE THE TERM CONFIGURATION CANNOT MOVE, which is what makes this a
    verdict rather than a tuning exercise. MAX_MODEL_LEN and the KV pool are
    adjustable; 10.25 GB of safetensors is not, short of a quantized checkpoint
    that vLLM's CPU backend can actually load.
    """
    mem = _meminfo()
    flags = _cpu_flags()
    has_avx512 = any(f.startswith("avx512") for f in flags)
    has_avx512_bf16 = "avx512_bf16" in flags or "amx_bf16" in flags

    # THERE IS NO fp32 UPCAST ON x86, and an earlier version of this function
    # asserted one and doubled the headline figure. CORRECTED 2026-09-04 against
    # vLLM's own source: `CpuPlatform.supported_dtypes` returns
    # [bfloat16, float16, float32] for x86 unconditionally — the comment above it
    # reads "x86/aarch64 CPU has supported both bf16 and fp16 natively". AVX512-BF16
    # governs whether there is a fast bf16 *datapath*, not whether the dtype is
    # accepted, and it does not change how many bytes the weights occupy.
    weights = MODEL_SAFETENSORS_BYTES
    kv = KVCACHE_SPACE_GIB * GIB
    # Activations and the process itself. Deliberately coarse and deliberately
    # not zero: a budget that lands exactly on the line is a budget that swaps.
    overhead = 1 * GIB
    need = weights + kv + overhead
    avail = mem.get("MemAvailable", 0)
    total = mem.get("MemTotal", 0)
    return {
        "available": avail,
        "total": total,
        "swap_free": mem.get("SwapFree", 0),
        "has_avx512": has_avx512,
        "has_avx512_bf16": has_avx512_bf16,
        "weights": weights,
        "kv": kv,
        "overhead": overhead,
        "need": need,
        "fits": need < avail,
        "shortfall": max(0, need - avail),
        # A live desktop's MemAvailable is not the machine's ceiling. When the
        # budget fits within TOTAL but not within what is free right now, the
        # remedy is to close things, not to give up on the rig.
        "fits_if_freed": need < total,
        "headroom_if_freed": max(0, total - need),
    }


@mcp.tool()
async def check_host_capacity() -> str:
    """Decide whether this host can serve the configured checkpoint in RAM.

    READ THIS BEFORE BUILDING vLLM. The CPU backend is a from-source build; the
    arithmetic here is free and settles whether the build could pay.

    The failure mode this guards against is not an error. Exceeding host RAM is
    accepted by the kernel and paid for in swap, so an over-budget serve looks
    exactly like a slow one — it never gets worse than "still loading".

    IT ALSO DISTINGUISHES "does not fit this machine" FROM "does not fit right
    now", because those have completely different remedies and an earlier version
    of this tool conflated them.
    """
    c = _capacity()
    g = lambda b: f"{b / 1e9:.2f} GB"  # noqa: E731

    lines = [
        f"📡 **Host capacity** — `{RIG_NAME}`", "",
        f"- RAM total {g(c['total'])} · **available {g(c['available'])}** · swap free {g(c['swap_free'])}",
        f"- AVX512: {'yes' if c['has_avx512'] else 'no'} · "
        f"AVX512-BF16/AMX: {'yes' if c['has_avx512_bf16'] else 'no'} — "
        f"neither changes the byte count; see `_capacity`.",
        "",
        f"**Budget for `{MODEL_NAME}`**", "",
        "| term | |", "| :--- | ---: |",
        f"| weights ({DTYPE}) | {g(c['weights'])} |",
        f"| KV pool (`VLLM_CPU_KVCACHE_SPACE={KVCACHE_SPACE_GIB}`) | {g(c['kv'])} |",
        f"| activations + process | {g(c['overhead'])} |",
        f"| **needed** | **{g(c['need'])}** |",
        f"| available now | {g(c['available'])} |",
        f"| machine total | {g(c['total'])} |",
        "",
    ]

    if c["fits"]:
        lines.append(f"✅ **Fits now**, {g(c['available'] - c['need'])} to spare. Re-check "
                     f"immediately before a load — this is a live desktop.")
    elif c["fits_if_freed"]:
        lines.append(
            f"⚠️  **Does not fit RIGHT NOW — short by {g(c['shortfall'])} — but it fits this "
            f"MACHINE**, with {g(c['headroom_if_freed'])} spare against the {g(c['total'])} total. "
            f"The remedy is to free {g(c['shortfall'])} (close a browser) and re-check, not to "
            f"abandon the rig.\n\n"
            f"**Do not start until it fits.** The kernel would accept the allocation and satisfy "
            f"it from the {g(c['swap_free'])} of swap, and a thrashing serve is indistinguishable "
            f"from a loading one."
        )
    else:
        lines.append(
            f"❌ **Over by {g(c['shortfall'])} even against the machine's full {g(c['total'])}.** "
            f"No amount of freeing helps; this needs a smaller checkpoint."
        )

    # The alternative artifact, always shown, because the choice of checkpoint is
    # the largest lever this rig has and it is invisible from the budget above.
    alt_need = MODEL_W4A16_BYTES + c["kv"] + c["overhead"]
    lines += [
        "", "**The quantized route, which is the real lever here**", "",
        f"`{MODEL_W4A16_NAME}` is **{g(MODEL_W4A16_BYTES)}** — MEASURED from the Hub "
        f"2026-09-04, ungated, `compressed-tensors` / `pack-quantized`, 4-bit int, "
        f"group size 32, symmetric.",
        "",
        "| | weights | needed | vs available |",
        "| :--- | ---: | ---: | :--- |",
        f"| `{MODEL_NAME.split('/')[-1]}` | {g(c['weights'])} | {g(c['need'])} | "
        f"{'fits' if c['fits'] else 'short ' + g(c['shortfall'])} |",
        f"| `{MODEL_W4A16_NAME.split('/')[-1]}` | {g(MODEL_W4A16_BYTES)} | {g(alt_need)} | "
        f"{'fits' if alt_need < c['available'] else 'short ' + g(alt_need - c['available'])} |",
        "",
        "**It saves less than 4-bit suggests, and the reason is in the checkpoint's own "
        "`ignore` list**: the vision tower and the embeddings stay bf16, so only the linear "
        "layers are packed. 10.25 → 8.32 GB is a 19% cut, not 75%. `@MODELS.md` records the "
        "resident figure as 8.15 GB and it is right; do not size this from a `weights ÷ 4` "
        "estimate, which under-predicts by ~3x.",
        "",
        f"- Trimming `MAX_MODEL_LEN` ({MAX_MODEL_LEN}) or the KV pool moves {g(c['kv'])} of a "
        f"{g(c['need'])} budget. Real, and second-order next to the checkpoint choice.",
        f"- Dropping the vision and audio towers is worth {g(MODEL_TOWERS_BYTES)} "
        f"(476 M params, MEASURED). They are in the w4a16 `ignore` list, so on that checkpoint "
        f"they are bf16 and the saving is larger in relative terms.",
        "- **vLLM has no equivalent of the JAX sibling's `PLE_BITS=4`**, which takes "
        "`local-jax-cpu-2b` to 5.752 GB by quantizing the per-layer embedding table — a gather, "
        "never a matmul, so it costs 0.0% decode. That remains the cheapest route to a CPU "
        "baseline on this host.",
    ]
    return "\n".join(lines)


@mcp.tool()
async def verify_cpu_backend() -> str:
    """Turn an install into evidence rather than a flag that was accepted.

    `VLLM_TARGET_DEVICE=cpu` being set at build time is not proof the result has
    a CPU backend, and the PyPI wheel imports fine while carrying only CUDA
    kernels. This asks the installed package what platform it actually resolved.
    """
    code = (
        "import vllm, sys;"
        "print('version', vllm.__version__);"
        "from vllm.platforms import current_platform as p;"
        "print('platform', p.device_name if hasattr(p,'device_name') else type(p).__name__);"
        "print('is_cpu', getattr(p, 'is_cpu', lambda: None)())"
    )
    rc, out, err = await run_command(["python3", "-c", code], timeout=120)
    if rc == 127:
        return "❌ python3 not found."
    if rc != 0:
        return (
            f"❌ vLLM is not importable, or has no CPU platform.\n\n```\n{(err or out).strip()[:800]}\n```\n\n"
            f"There is **no published CPU wheel** — `wheels.vllm.ai/cpu` returns 404 and the PyPI "
            f"`vllm` wheel is the CUDA build. A CPU vLLM is a source build:\n\n"
            f"```\ngit clone https://github.com/vllm-project/vllm && cd vllm\n"
            f"VLLM_TARGET_DEVICE=cpu VLLM_CPU_DISABLE_AVX512=1 pip install -e .\n```\n\n"
            f"**Run `check_host_capacity` first.** On this host it says the result could not serve."
        )
    return f"✅ **vLLM CPU backend**\n\n```\n{out.strip()}\n```"


@mcp.tool()
async def start_vllm_server() -> str:
    """Start the vLLM OpenAI-compatible server on the CPU.

    REFUSES when the host cannot hold the weights. That is the one decision this
    rig takes away from the operator, and it is deliberate: the alternative is a
    serve that swaps, and a swapping serve reports nothing — it just never
    finishes loading.
    """
    existing = _read_pid()
    if existing is not None:
        return f"✅ Already running (pid {existing}) at {ENDPOINT}. Use `stop_vllm_server` first."

    c = _capacity()
    if not c["fits"]:
        return (
            f"❌ **Refusing to start: over budget by {c['shortfall'] / 1e9:.2f} GB.**\n\n"
            f"Needs {c['need'] / 1e9:.2f} GB, {c['available'] / 1e9:.2f} GB available. "
            f"Run `check_host_capacity` for the breakdown and for what does and does not help.\n\n"
            f"This is not a safety margin being cautious — the kernel would accept the "
            f"allocation and satisfy it from swap."
        )

    RUN_DIR.mkdir(exist_ok=True)
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", HOST, "--port", str(PORT),
        "--device", "cpu",
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
    ]
    env = dict(os.environ)
    env["VLLM_CPU_KVCACHE_SPACE"] = str(KVCACHE_SPACE_GIB)
    if OMP_THREADS_BIND:
        env["VLLM_CPU_OMP_THREADS_BIND"] = OMP_THREADS_BIND

    with open(LOG_FILE, "ab") as log:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=log, stderr=log, env=env, start_new_session=True
        )
    PID_FILE.write_text(str(proc.pid))
    return (
        f"📡 Started vLLM (pid {proc.pid}) → {ENDPOINT}\n\n```\n{' '.join(cmd)}\n```\n\n"
        f"Loading {MODEL_SAFETENSORS_BYTES / 1e9:.2f} GB from disk is not instant. Poll "
        f"`server_status`; log at `{LOG_FILE}`."
    )


@mcp.tool()
async def stop_vllm_server() -> str:
    """Stop the running server. Teardown is complete — nothing is billed here."""
    pid = _read_pid()
    if pid is None:
        PID_FILE.unlink(missing_ok=True)
        return "✅ Not running."
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return f"❌ Could not signal pid {pid}: {exc}"
    PID_FILE.unlink(missing_ok=True)
    return f"✅ Sent SIGTERM to vLLM (pid {pid}). RAM is released on exit."


@mcp.tool()
async def server_status() -> str:
    """Check whether the server is up, and report live RAM alongside it.

    RAM IS REPORTED HERE ON PURPOSE. On an accelerator rig "still loading" and
    "thrashing" look different; here they do not, and the only way to tell is
    that available memory has collapsed.
    """
    pid = _read_pid()
    mem = _meminfo()
    ram = (f"RAM available {mem.get('MemAvailable', 0) / 1e9:.2f} GB · "
           f"swap free {mem.get('SwapFree', 0) / 1e9:.2f} GB")
    if pid is None:
        return f"❌ Not running. Endpoint would be {ENDPOINT}.\n\n{ram}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{ENDPOINT}/health")
        if resp.status_code == 200:
            return f"✅ Serving at {ENDPOINT} (pid {pid}).\n\n{ram}"
        return f"📡 Process up (pid {pid}), `/health` → {resp.status_code}.\n\n{ram}"
    except httpx.HTTPError as exc:
        return (
            f"📡 Process up (pid {pid}) but {ENDPOINT} is not answering yet ({exc}).\n\n{ram}\n\n"
            f"If available RAM has collapsed, this is swapping rather than loading."
        )


@mcp.tool()
async def query_model(prompt: str, max_tokens: int = 1024) -> str:
    """Send a chat completion to the local endpoint.

    /v1/chat/completions, not /v1/completions — raw completions return an empty
    string on `-it` checkpoints (root CLAUDE.md).

    GEMMA 4 IS A REASONING MODEL, which is a second and unrelated way to get an
    empty reply: the thinking block is emitted first and a small `max_tokens`
    truncates mid-thought. MEASURED on the llama.cpp sibling 2026-09-03: 1274
    characters of reasoning before 22 characters of answer. Hence 1024.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    try:
        async with httpx.AsyncClient(timeout=1800) as client:
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
                f"📡 **Reasoning only — no answer yet.** `finish_reason: "
                f"{choice.get('finish_reason')}` after {usage.get('completion_tokens', '?')} tokens, "
                f"all of them thinking. Re-run with a larger `max_tokens` (currently {max_tokens})."
            )
        parts = ["✅ **Reply**", "", text, "", "---",
                 f"prompt {usage.get('prompt_tokens', '?')} tok · "
                 f"completion {usage.get('completion_tokens', '?')} tok"]
        return "\n".join(parts)
    except httpx.HTTPError as exc:
        return f"❌ Could not reach {ENDPOINT}: {exc}. Is the server running?"
    except (KeyError, IndexError, ValueError) as exc:
        return f"❌ Unexpected response shape from {ENDPOINT}: {exc}"


@mcp.tool()
async def get_help() -> str:
    """List the tools this rig exposes."""
    tools = await mcp.list_tools()
    lines = [f"📡 **{MCP_SERVER_NAME}** — local vLLM on CPU, Gemma 4 E2B. No accelerator.", ""]
    for tool in tools:
        lines.append(f"- **{tool.name}** — {(tool.description or '').splitlines()[0]}")
    lines += [
        "",
        "No provisioning tools, by design: there is no control plane and no chip to find. "
        "See NAMING.md, \"`local` is the absence of a control plane\".",
        "",
        "**Start with `check_host_capacity`.** As of 2026-09-04 it refuses on this host, and "
        "`start_vllm_server` refuses with it. The reachable CPU baseline on this machine is "
        "`local-jax-cpu-2b`, whose `PLE_BITS=4` lever has no vLLM equivalent.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
