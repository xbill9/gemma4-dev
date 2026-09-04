"""Local Ollama lifecycle and inference MCP server — GTX 1650 Ti, Gemma 4 E2B q4_0.

THIS IS A `local` RIG, so there is no control plane: the card is in the machine,
nothing here provisions capacity, waits for it, discovers an endpoint or releases
anything. The tools that dominate a cloud sibling's server.py — find_tpu,
create_*_queued_resource, manage_queued_resource, the zone-status skip list —
have no analogue and are deliberately absent. If a `find_*` tool ever appears
here the rig has the wrong name (NAMING.md, "`local` is the absence of a control
plane").

AND IT IS AN `ollama` RIG, which is the half that differs from
local-llamacpp-1650ti-2b-q4_0. The two serve the same weights on the same card
through the same engine — Ollama links llama.cpp as a library — so the pair is an
A/B on the DAEMON's choices, and this file exists mostly to make those choices
visible. Four of them are measured and load-bearing:

  1. THE STOCK TAG LOADS A PROJECTOR AND THIS RIG NO LONGER SERVES IT.
     `gemma4:e2b-it-qat` bundles a 986 MB mmproj and the daemon passes --mmproj
     unconditionally, so a vision+audio encoder is resident even for a pure-text
     workload. MEASURED 2026-09-04: 2762 MiB against the llama.cpp sibling's
     1618 MiB at the same 8192 context — 28% of a 4096 MiB card spent on
     capability this rig never exercises.

     MODEL_NAME therefore points at `gemma4:e2b-it-qat-text`, the SAME Ollama
     blob with the projector layer dropped (modelfiles/text-only.Modelfile,
     `make text-only`). MEASURED: 1612 MiB — 6 MiB BELOW the sibling. The stock
     tag is kept on disk as MODEL_STOCK_TAG because it is what
     gpu-ollama-g5g-2b-q4_0 serves, so it is the reference for any comparison
     against that rig.

     It is not only the encoder: with the projector loaded the daemon also asked
     llama-server for 2048 tokens/slot when told 1024, so 32 slots did not fit at
     all (26/36 layers offloaded, 512 MiB of KV on the host). Projector-free, the
     same request is honoured at 1024/slot and 32 slots fit in 2106 MiB.
  2. OLLAMA'S OWN SIZE GAUGE UNDER-REPORTS, WHICHEVER MODEL IS LOADED. `/api/ps`
     reported 1.7 GB against nvidia-smi's 2762 MiB for the stock tag, and 1.5 GB
     against 1612 MiB for the variant. verify_model_resident therefore reads BOTH
     and interprets the gap against the model's declared capabilities rather than
     against a hardcoded number.
  3. THERE IS NO N_GPU_LAYERS. llama-server takes --n-gpu-layers; Ollama decides
     the offload from its own VRAM estimate. Partial offload cannot be prevented
     by configuration here, only detected — which is what the `processor` field
     in verify_model_resident is for.
  4. THE CHAT TEMPLATE IS OLLAMA'S, NOT THE GGUF'S. The daemon starts its child
     with `--no-jinja --chat-template chatml` and then applies its own Go
     renderer/parser (`renderer=gemma4 parser=gemma4`). Thinking behaviour
     differs from the sibling's as a result — see query_model.

MEMORY: the E2B GGUF is 3.35 GB on disk but only ~1.31 GiB of it has to be
resident, because per_layer_token_embd (58% of the file) is created with
TENSOR_READ_LAZY and served by GET_ROWS out of the mmap. That is a property of
llama.cpp's Gemma 4 graph and it holds here too, since this is the same engine.
It is why 4096 MiB is enough for a checkpoint whose file is 3.35 GB.
"""

import asyncio
import json
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

# An Ollama TAG, not a Hub id. See tpu.env for why that is a weaker claim than
# the llama.cpp sibling's MODEL_PATH, and why pointing this at the Hub file
# through a Modelfile would delete the point of the pair.
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma4:e2b-it-qat-text")
# The upstream tag MODEL_NAME is derived from. Read here only to name it in
# diagnostics — this rig never serves it by default, because the projector it
# carries costs 1150 MiB of a 4096 MiB card.
MODEL_STOCK_TAG = os.environ.get("MODEL_STOCK_TAG", "gemma4:e2b-it-qat")
OLLAMA_BIN = os.environ.get("OLLAMA_BIN", "")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8000")
ENDPOINT = os.environ.get("ENDPOINT", f"http://{HOST}:{PORT}")
OLLAMA_LLM_LIBRARY = os.environ.get("OLLAMA_LLM_LIBRARY", "cuda_v12")
OLLAMA_CONTEXT_LENGTH = os.environ.get("OLLAMA_CONTEXT_LENGTH", "8192")
OLLAMA_NUM_PARALLEL = os.environ.get("OLLAMA_NUM_PARALLEL", "1")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "-1")
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", "")

RUN_DIR = RIG_DIR / "run"
PID_FILE = RUN_DIR / "ollama.pid"
LOG_FILE = RUN_DIR / "ollama.log"

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
    """The running daemon's pid, or None. Checked against /proc rather than
    trusted: a stale pid file outlives a Ctrl-C and there is no control plane
    here to ask for the truth."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if Path(f"/proc/{pid}").exists() else None


def _daemon_env() -> dict:
    """The environment the daemon is started with.

    HOME IS DELIBERATELY INHERITED AND MUST BE PRESENT. `ollama serve` exits 1
    with "Error: $HOME is not defined", and gpu-ollama-g5g-2b-q4_0 lost a
    provisioning cycle to exactly that: systemd gives a unit a minimal
    environment, the daemon crash-looped 13 times in two minutes, and the
    bootstrap's readiness poll timed out and carried on anyway. Nothing here runs
    under systemd, so HOME comes from the invoking shell — but the test that
    pins it stays, because the failure is silent at the point it is caused.
    """
    env = dict(os.environ)
    env.update({
        "OLLAMA_HOST": f"{HOST}:{PORT}",
        "OLLAMA_LLM_LIBRARY": OLLAMA_LLM_LIBRARY,
        "OLLAMA_CONTEXT_LENGTH": OLLAMA_CONTEXT_LENGTH,
        "OLLAMA_NUM_PARALLEL": OLLAMA_NUM_PARALLEL,
        "OLLAMA_KEEP_ALIVE": OLLAMA_KEEP_ALIVE,
    })
    if OLLAMA_MODELS:
        env["OLLAMA_MODELS"] = OLLAMA_MODELS
    return env


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

    _, apps, _ = await run_command([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ], timeout=30)
    if apps.strip():
        body += ["**Compute apps:**", "", f"```\n{apps.strip()}\n```", ""]

    # sm_75 is shared with the T4 rigs and the cards are NOT equivalent: the T4 is
    # TU104 and has tensor cores, TU116/TU117 has none. Say so at the point of
    # use rather than hoping the reader checks CLAUDE.md.
    if "1650" in line or "1660" in line:
        body.append(
            "⚠️  GTX 16-series (TU116/TU117): compute capability 7.5 but **no tensor "
            "cores**. Do not compare throughput against the T4-based `g4dn`/`g5g` "
            "rigs on the strength of a matching compute capability."
        )
    return "\n".join(body)


@mcp.tool()
async def start_daemon() -> str:
    """Start `ollama serve` on the local endpoint. No-op if it is already running."""
    if not OLLAMA_BIN or not Path(OLLAMA_BIN).exists():
        return f"❌ ollama not found at `{OLLAMA_BIN}`. Set `OLLAMA_BIN` in `tpu.env`."

    existing = _read_pid()
    if existing is not None:
        return f"✅ Already running (pid {existing}) at {ENDPOINT}. Use `stop_daemon` first to restart."

    env = _daemon_env()
    if not env.get("HOME"):
        return "❌ HOME is not set. `ollama serve` exits 1 without it — see `_daemon_env`."

    RUN_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "ab") as log:
        proc = await asyncio.create_subprocess_exec(
            OLLAMA_BIN, "serve", stdout=log, stderr=log, env=env, start_new_session=True
        )
    PID_FILE.write_text(str(proc.pid))
    return (
        f"📡 Started ollama serve (pid {proc.pid}) → {ENDPOINT}\n\n"
        f"- `OLLAMA_LLM_LIBRARY={OLLAMA_LLM_LIBRARY}` — pinned; unset it and the "
        f"driver selects cuda_v13, which may JIT every kernel from PTX at load.\n"
        f"- `OLLAMA_CONTEXT_LENGTH={OLLAMA_CONTEXT_LENGTH}` × "
        f"`OLLAMA_NUM_PARALLEL={OLLAMA_NUM_PARALLEL}` — Ollama MULTIPLIES these. "
        f"llama.cpp DIVIDES its `-c` across slots. Same total, opposite convention.\n"
        f"- `OLLAMA_KEEP_ALIVE={OLLAMA_KEEP_ALIVE}` — negative means never unload.\n\n"
        f"The model is not loaded until the first request. Log at `{LOG_FILE}`."
    )


@mcp.tool()
async def stop_daemon() -> str:
    """Stop the running daemon. Teardown is complete — nothing is billed here."""
    pid = _read_pid()
    if pid is None:
        PID_FILE.unlink(missing_ok=True)
        return "✅ Not running."
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return f"❌ Could not signal pid {pid}: {exc}"
    PID_FILE.unlink(missing_ok=True)
    return f"✅ Sent SIGTERM to ollama (pid {pid}). VRAM is released when the child exits."


@mcp.tool()
async def daemon_status() -> str:
    """Check whether the daemon is up and answering at the known local endpoint."""
    pid = _read_pid()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{ENDPOINT}/api/version")
        if resp.status_code == 200:
            version = resp.json().get("version", "?")
            owned = f"pid {pid}" if pid else "not started by this rig"
            return f"✅ Ollama {version} answering at {ENDPOINT} ({owned})."
        return f"📡 {ENDPOINT}/api/version → {resp.status_code}."
    except httpx.HTTPError as exc:
        return f"❌ {ENDPOINT} is not answering ({exc}). Start it with `start_daemon`."


def _has_projector(capabilities: list) -> bool:
    """Whether the loaded model declares the multimodal capabilities that make the
    daemon pass --mmproj. This is the honest test: the projector is a LAYER on the
    model, so the capability list is what decides, not the tag name."""
    return any(c in ("vision", "audio") for c in capabilities or ())


@mcp.tool()
async def model_info() -> str:
    """Report what Ollama thinks the configured tag is: architecture, quantization, capabilities."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{ENDPOINT}/api/show", json={"model": MODEL_NAME})
        if resp.status_code != 200:
            return f"❌ /api/show → {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
    except httpx.HTTPError as exc:
        return f"❌ Could not reach {ENDPOINT}: {exc}"

    details = data.get("details", {})
    info = data.get("model_info", {})
    caps = data.get("capabilities", [])

    body = [
        f"📡 **Model** — `{RIG_NAME}`",
        "",
        f"- **Tag:** `{MODEL_NAME}` — an Ollama tag, **not** a Hub id.",
        f"- **Architecture:** {details.get('family', '?')} · "
        f"**{details.get('parameter_size', '?')}** · "
        f"quantization `{details.get('quantization_level', '?')}`",
        f"- **Trained context:** {info.get('gemma4.context_length', '?')}",
        f"- **Capabilities:** {', '.join(caps) or '—'}",
        "",
    ]

    # The verdict is READ OFF the capabilities rather than assumed from the tag
    # name, because the expensive mistake here is serving the stock tag by
    # accident and nothing announces it — `ollama ps` does not count the
    # projector, so the memory is simply gone.
    if _has_projector(caps):
        body.append(
            "⚠️  **This model carries the projector.** `vision`/`audio` in that "
            "list are not free: the tag bundles a 986 MB mmproj that the daemon "
            "loads unconditionally — **~1154 MiB of a 4096 MiB card**, plus a "
            "larger minimum per-slot context that cost another 192 MiB of KV at "
            "32 slots. MEASURED 2026-09-04: 2762 MiB resident against 1612 for "
            "the projector-free variant.\n\n"
            "If this rig is serving it, that is a regression: `MODEL_NAME` should "
            "be `gemma4:e2b-it-qat-text`. Rebuild with `make text-only`."
        )
    else:
        body.append(
            f"✅ **Projector-free.** No `vision`/`audio`, so no mmproj is loaded: "
            f"**1612 MiB** resident at 8192 context, against **2762 MiB** for the "
            f"stock `{MODEL_STOCK_TAG}` and **1618 MiB** for the llama.cpp "
            f"sibling serving the same weights. Same model blob by digest, same "
            f"renderer, same sampling — only the projector layer is dropped."
        )
    body += ["", "Run `verify_model_resident` for the measured split."]
    return "\n".join(body)


@mcp.tool()
async def verify_model_resident() -> str:
    """Assert the model is fully on the GPU, and cross-check Ollama's gauge against the driver.

    OLLAMA'S OWN `size` UNDER-REPORTS WHAT THE PROCESS HOLDS, so this tool reads
    both, and it reads the model's declared capabilities so the gap can be
    interpreted instead of guessed. MEASURED 2026-09-04 on this card at 8192
    context:

        stock gemma4:e2b-it-qat        /api/ps 1.7 GB   nvidia-smi 2762 MiB
        gemma4:e2b-it-qat-text         /api/ps 1.5 GB   nvidia-smi 1612 MiB

    The gap is the CUDA context plus the compute buffers, and it SCALES WITH SLOT
    COUNT — MEASURED 181 MiB at `-np 1` and 291 MiB at `-np 32`, both
    projector-free. So the discriminator is order of magnitude, not a threshold:
    a couple of hundred MiB is normal, ~1150 MiB is an mmproj.

    The `processor` field is the one that must say 100% GPU. There is no
    N_GPU_LAYERS on this rig, so partial offload cannot be prevented here, only
    detected; a "60%/40% CPU/GPU" reading is a silent throughput cliff, not an
    error anything raises — MEASURED at 32 slots with the stock tag, where it
    halved single-stream decode.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{ENDPOINT}/api/ps")
        if resp.status_code != 200:
            return f"❌ /api/ps → {resp.status_code}: {resp.text[:300]}"
        models = resp.json().get("models", [])
    except httpx.HTTPError as exc:
        return f"❌ Could not reach {ENDPOINT}: {exc}"

    if not models:
        return (
            "📡 No model is loaded. Ollama loads on first request, so this is the "
            "expected state after `start_daemon`. Send one query and re-check."
        )

    # Capabilities decide how to read the gap below. Best-effort: a failure here
    # must not cost the caller the residency verdict, which is the point of the tool.
    caps = []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            show = await client.post(f"{ENDPOINT}/api/show", json={"model": MODEL_NAME})
        if show.status_code == 200:
            caps = show.json().get("capabilities", [])
    except (httpx.HTTPError, ValueError):
        pass

    lines = [f"📡 **Residency** — `{RIG_NAME}`", ""]
    for m in models:
        size = m.get("size", 0)
        vram = m.get("size_vram", 0)
        pct = (vram / size * 100) if size else 0.0
        verdict = "✅ fully on GPU" if size and vram >= size else "❌ PARTIAL OFFLOAD"
        lines.append(
            f"- `{m.get('name')}` — Ollama reports {size / 1e9:.2f} GB, "
            f"{vram / 1e9:.2f} GB in VRAM ({pct:.0f}%) — {verdict}"
        )
        lines.append(f"  context {m.get('context_length', '?')}")

    rc, out, _ = await run_command([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ], timeout=30)
    if rc == 0 and out.strip():
        lines += ["", "**What the driver says the process actually holds:**", "",
                  f"```\n{out.strip()}\n```", ""]
        if _has_projector(caps):
            lines.append(
                "⚠️  This model declares `vision`/`audio`, so the driver figure "
                "should EXCEED Ollama's `size` by roughly the projector "
                "(~1154 MiB) — and that memory is being spent on a capability "
                "this rig does not use. Serve `gemma4:e2b-it-qat-text` instead "
                "(`make text-only`)."
            )
        else:
            lines.append(
                "Projector-free, so the driver figure should exceed Ollama's "
                "`size` by the CUDA context and compute buffers only — MEASURED "
                "181 MiB at 1 slot and 291 MiB at 32, so it scales with slots and "
                "a couple of hundred MiB is normal. **A gap near 1150 MiB means a "
                "projector-carrying model is loaded** — check which tag answered, "
                "not which one you asked for."
            )
        lines.append(
            "Ollama's `size` never counts the projector, so it cannot be used to "
            "size this rig either way. Read the driver."
        )
    return "\n".join(lines)


@mcp.tool()
async def query_model(prompt: str, max_tokens: int = 1024, think: bool = True) -> str:
    """Send a chat request to the local daemon and return the reply.

    GEMMA 4 REASONS, AND OLLAMA HANDLES THAT DIFFERENTLY FROM THE llama.cpp
    SIBLING — three ways, all MEASURED 2026-09-04 on the same prompt ("Name three
    TPU generations"), and all three can look like a broken server:

      /api/chat think=true    278 tok  ->  916 chars thinking + 152 chars content
      /api/chat think=false   490 tok  ->  0 chars thinking + 2290 chars content
      /api/generate (default) 278 tok  ->  thinking GENERATED AND DISCARDED
      /v1/chat/completions     -        ->  field is `reasoning`, NOT the
                                            sibling's `reasoning_content`

    The third is the trap. At max_tokens=128 the thinking block has not closed,
    so `/api/generate` returns `done_reason: length` with BOTH fields empty —
    128 tokens generated and zero characters returned. The llama.cpp sibling at
    least exposes the partial thought. Hence the 1024 default, `think=True`, and
    the explicit report below.

    Budget tokens accordingly in any benchmark: a 128-token limit on this model
    measures the thinking phase and nothing else, which makes the tok/s real and
    the task fictional.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
        "options": {"num_predict": max_tokens},
    }
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(f"{ENDPOINT}/api/chat", json=payload)
        if resp.status_code != 200:
            return f"❌ {resp.status_code} from {ENDPOINT}: {resp.text[:500]}"
        data = resp.json()
    except httpx.HTTPError as exc:
        return f"❌ Could not reach {ENDPOINT}: {exc}. Is the daemon running?"
    except json.JSONDecodeError as exc:
        return f"❌ Unexpected response shape from {ENDPOINT}: {exc}"

    message = data.get("message", {})
    text = message.get("content") or ""
    thinking = message.get("thinking") or ""
    done_reason = data.get("done_reason", "?")
    eval_count = data.get("eval_count")

    if not text:
        return (
            f"📡 **No answer text.** `done_reason: {done_reason}` after "
            f"{eval_count} tokens"
            + (f", {len(thinking)} chars of them thinking.\n\n" if thinking else
               ", and the thinking was discarded rather than returned.\n\n")
            + f"This is Gemma 4 reasoning, not a broken daemon. Re-run with a larger "
              f"`max_tokens` (currently {max_tokens}).\n\n"
            + (f"<details>\n\n{thinking[:800]}\n\n</details>" if thinking else "")
        )

    parts = ["✅ **Reply**", "", text, "", "---"]
    if thinking:
        parts.append(f"_(plus {len(thinking)} chars of thinking, suppressed)_")
    parts.append(f"{eval_count} tokens · `done_reason: {done_reason}`")
    rate = _decode_rate(data)
    if rate is not None:
        parts.append(
            f"⚠️  {rate:.2f} tok/s — this is Ollama's SERVER-SIDE gauge "
            f"(`eval_count / eval_duration`). It excludes prefill and HTTP and is "
            f"**not** the same statistic as a client-side stream rate. Do not "
            f"difference it against a sibling's number."
        )
    return "\n".join(parts)


_NS_PER_S = 1e9


def _decode_rate(body: dict) -> Optional[float]:
    """Ollama's own decode gauge, in tok/s. NOT comparable to a client-side rate."""
    count, duration = body.get("eval_count"), body.get("eval_duration")
    if not count or not duration:
        return None
    return count / (duration / _NS_PER_S)


def _prefill_rate(body: dict) -> Optional[float]:
    """Ollama's own prefill gauge, in tok/s. Meaningless on a short prompt, where
    it is dominated by fixed overhead: MEASURED 130 t/s on a 21-token prompt
    against ~318 t/s on a 652-token one, on the same loaded model."""
    count, duration = body.get("prompt_eval_count"), body.get("prompt_eval_duration")
    if not count or not duration:
        return None
    return count / (duration / _NS_PER_S)


@mcp.tool()
async def get_metrics(prompt: str = "Explain gradient descent.", max_tokens: int = 512) -> str:
    """Probe the daemon's own timing gauges for one request.

    READ THE WARNING IN THE OUTPUT BEFORE QUOTING ANY OF THIS. These are
    server-side nanosecond counters. `benchmarks/` holds the client-side stream
    statistic that is comparable across rigs; this tool is for checking that the
    daemon is behaving, not for producing a number anyone publishes.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(f"{ENDPOINT}/api/generate", json=payload)
        if resp.status_code != 200:
            return f"❌ {resp.status_code} from {ENDPOINT}: {resp.text[:500]}"
        data = resp.json()
    except httpx.HTTPError as exc:
        return f"❌ Could not reach {ENDPOINT}: {exc}"

    decode = _decode_rate(data)
    prefill = _prefill_rate(data)
    load_ms = (data.get("load_duration") or 0) / 1e6
    return (
        f"📡 **Timings** — `{RIG_NAME}`\n\n"
        f"- prompt {data.get('prompt_eval_count', '?')} tok"
        + (f" · {prefill:.1f} t/s prefill\n" if prefill else "\n")
        + f"- output {data.get('eval_count', '?')} tok"
        + (f" · {decode:.2f} tok/s decode\n" if decode else "\n")
        + f"- load {load_ms:.0f} ms · `done_reason: {data.get('done_reason', '?')}`\n\n"
        f"⚠️  **These are Ollama's own gauges, not a benchmark.** `eval_count / "
        f"eval_duration` excludes prefill and HTTP entirely; the vLLM and JAX "
        f"siblings' figures come from client-side statistics that include both. "
        f"Differencing the two is the mistake gpu-ollama-g5g-2b-q4_0's CLAUDE.md "
        f"records as \"do not publish 3x vLLM off this\".\n\n"
        f"Prefill is meaningless on a short prompt — it is fixed overhead there."
    )


@mcp.tool()
async def get_help() -> str:
    """List the tools this rig exposes."""
    tools = await mcp.list_tools()
    lines = [f"📡 **{MCP_SERVER_NAME}** — local Ollama rig, GTX 1650 Ti, Gemma 4 E2B q4_0", ""]
    for tool in tools:
        lines.append(f"- **{tool.name}** — {(tool.description or '').splitlines()[0]}")
    lines += [
        "",
        "No provisioning tools, by design: the hardware is local and there is no "
        "control plane to call. See NAMING.md, \"`local` is the absence of a control plane\".",
        "",
        "Paired with `local-llamacpp-1650ti-2b-q4_0`: same weights, same card, same "
        "engine, different daemon. What differs is the artifact container, the chat "
        "template and the CUDA variant selection.",
        "",
        f"Serving `{MODEL_NAME}` — the projector-free variant of "
        f"`{MODEL_STOCK_TAG}`, which is worth 1150 MiB on a 4096 MiB card. "
        f"`model_info` says which one actually answered.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
