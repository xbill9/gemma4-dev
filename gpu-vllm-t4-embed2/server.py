"""vLLM-on-an-attached-T4 lifecycle MCP server — EmbeddingGemma 2, one Tesla T4.

STATUS 2026-10-07: SERVING AND MEASURED. vLLM 0.31.1rc1.dev63+gd169c5a90 (the
nightly; support for this model is in no release yet) serves
`google/embeddinggemma-2` on `/v1/embeddings` at float32, text + image, and
matches sentence-transformers to 1-cos <= 7.0e-9.
`benchmarks/runs/2026-10-07-first-serve-t4/REPORT.md` is the record.

WHAT THIS RIG IS. An EMBEDDING model, not a generative one — the first in this
tree. vLLM's pooling runner, `/v1/embeddings`, no chat completions, no KV cache
(the encoder is bidirectional, `AttentionType.ENCODER_ONLY`). Slot 4 is `embed2`
because the checkpoint has no Gemma 4 size: one file loads as 271M to 744M
parameters depending on which towers are enabled (@NAMING.md).

THE T4 IS THE SAME ONE `gpu-vllm-t4-2b` USES, and the host facts there hold here:
no control plane, three filesystems, two interpreters, 7.8 GB of host RAM. Read
that rig's CLAUDE.md for them; this file repeats only what the code depends on.

THREE THINGS ARE DIFFERENT, AND EACH ONE BREAKS A SIBLING ASSUMPTION:

  * float32, NOT float16. Google says float16 is incompatible with this model,
    and Turing has no bf16 datapath (vLLM silently casts bf16 to fp16 on SM 7.5,
    so asking for bf16 would get the forbidden dtype without an error).
  * A DIFFERENT TRITON KERNEL BREAKS. Encoder-only attention goes through
    `triton_prefill_attention.context_attention_fwd`, not the unified decoder
    kernel the siblings clamp. At float32 it needs 69,632 B of shared memory at
    head_dim 256 and fits no tile at all at head_dim 512. See
    `patch_triton_encoder_turing.py`.
  * ITS OWN PYTHONUSERBASE, `/opt1/pyuser-embed2`. The siblings' `/opt1/pyuser`
    holds vLLM 0.29 with their clamp applied; this model needs the nightly and
    transformers 5.19, and installing those there would break both siblings and
    silently revert their patch.
"""

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

RIG_DIR = Path(__file__).resolve().parent
load_dotenv(RIG_DIR / "tpu.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RIG_NAME = RIG_DIR.name
MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", RIG_NAME)

MODEL_NAME = os.environ.get("MODEL_NAME", "google/embeddinggemma-2")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "")
MODEL_SAFETENSORS_BYTES = int(os.environ.get("MODEL_SAFETENSORS_BYTES", "1488915288"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

GPU_TOTAL_MIB = int(os.environ.get("GPU_TOTAL_MIB", "15360"))
COMPUTE_CAPABILITY = os.environ.get("COMPUTE_CAPABILITY", "7.5")
DTYPE = os.environ.get("DTYPE", "float32")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
RUNNER = os.environ.get("RUNNER", "pooling")
LIMIT_MM_PER_PROMPT = os.environ.get("LIMIT_MM_PER_PROMPT", '{"image":1,"audio":0,"video":0}')
HF_OVERRIDES = os.environ.get("HF_OVERRIDES", '{"matryoshka_dimensions":[768,512,256,128]}')

# The model's trained task prefixes, from its config_sentence_transformers.json.
# vLLM DOES NOT ADD THEM — `/v1/embeddings` embeds exactly the string it is sent —
# so the tools below prepend them, and a client that skips them gets embeddings
# from a distribution the model was not trained to produce.
# Constants, not env settings: each ends in a space that an env file would trim or
# a Makefile `export` would wrap in literal quotes.
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8000")
ENDPOINT = os.environ.get("ENDPOINT", f"http://{HOST}:{PORT}")
VLLM_PIP_SPEC = os.environ.get("VLLM_PIP_SPEC", "vllm")

# THE INTERPRETER IS CONFIGURATION, NOT `python3` (see gpu-vllm-t4-2b): pyenv's
# python3 has its site-packages on the nearly-full root disk. Every subprocess
# this module runs uses PYTHON_BIN and _child_env() — never a bare "python3".
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")
PYTHONUSERBASE = os.environ.get("PYTHONUSERBASE", "")
PIP_CACHE_DIR = os.environ.get("PIP_CACHE_DIR", "")
TMPDIR = os.environ.get("TMPDIR", "")

RUN_DIR = RIG_DIR / "run"
PID_FILE = RUN_DIR / "vllm.pid"
LOG_FILE = RUN_DIR / "vllm.log"

GIB = 1024**3

# ARITHMETIC allowance for the install (torch + nvidia-cu13 wheels + vLLM), as in
# the sibling. MEASURED for this user base 2026-10-07: see `check_host_capacity`.
VLLM_INSTALL_BYTES = int(os.environ.get("VLLM_INSTALL_BYTES", str(7 * 10**9)))

_PATCH_SCRIPT = "patch_triton_encoder_turing.py"
# Must equal patch_triton_encoder_turing.SENTINEL. Duplicated rather than imported
# so this module stays importable on a host with no vLLM; pinned by a test.
_PATCH_SENTINEL = "Turing encoder-attention clamp"
_TRITON_MODULE = "vllm.v1.attention.ops.triton_prefill_attention"

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


def _child_env() -> dict:
    """The environment every probe and subprocess here must run under.

    PYTHONUSERBASE is the load-bearing one, and on this rig it is ALSO what keeps
    the generative siblings' install out of the picture: without it the target
    interpreter would see `/opt1/pyuser`'s vLLM 0.29, which cannot load this model.
    """
    env = dict(os.environ)
    for key, value in (
        ("PYTHONUSERBASE", PYTHONUSERBASE),
        ("PIP_CACHE_DIR", PIP_CACHE_DIR),
        ("TMPDIR", TMPDIR),
    ):
        if value:
            env[key] = value
    return env


async def _probe(code: str, timeout: int = 600) -> tuple[int, str, str]:
    """Run a snippet under the TARGET interpreter, not under this process."""
    try:
        proc = await asyncio.create_subprocess_exec(
            PYTHON_BIN,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_child_env(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace"))
    except asyncio.TimeoutError:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "", f"not found: {PYTHON_BIN}"


def _site_packages_dir() -> Optional[Path]:
    """Where a `pip install --user` for PYTHON_BIN would land (derived, not asked)."""
    if not PYTHONUSERBASE:
        return None
    base = Path(PYTHONUSERBASE)
    candidates = sorted(base.glob("lib/python3.*/site-packages"))
    return candidates[-1] if candidates else base


def _read_pid() -> Optional[int]:
    """The running server's pid, or None — checked against /proc, never trusted."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if Path(f"/proc/{pid}").exists() else None


def _meminfo() -> dict:
    """Live host memory, in bytes. MemAvailable, never MemFree."""
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            out[key] = int(rest.strip().split()[0]) * 1024
    return out


def _disk_free(path) -> tuple[int, int]:
    """(free, total) bytes on the filesystem holding `path`. ALWAYS PER TARGET.

    Walks up to the nearest existing ancestor, so it answers for a directory that
    does not exist yet. On this host `/`, `/tmp` and `/opt1` are three filesystems.
    """
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return usage.free, usage.total


def _hf_cache_dir() -> Path:
    """Where the checkpoint will land."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface"


def _gpu_vram_bytes() -> int:
    """Total device memory, from `tpu.env` — the capacity arithmetic must run with no driver."""
    return GPU_TOTAL_MIB * 1024 * 1024


def _loaded_weight_bytes(file_bytes: int) -> int:
    """Device bytes for the weights at DTYPE. The file is bf16; float32 doubles it.

    An upper bound: it counts every tower, and LIMIT_MM_PER_PROMPT skips loading
    the disabled ones. MEASURED 2026-10-07: 1.03 GiB text only, 1.63 GiB with the
    vision tower, against 2.77 GiB for the whole file at float32.
    """
    return file_bytes * 2 if DTYPE == "float32" else file_bytes


def _capacity(model_bytes: Optional[int] = None) -> dict:
    """The rig's go/no-go arithmetic: three disks and VRAM, each on its own path.

    VRAM IS NOT CLOSE TO BINDING and the report says so. There is no KV cache to
    size — encoder-only attention keeps none — so the budget is weights plus
    activations, and activations at 8,192 tokens are the open term.

    HOST RAM IS THE TERM TO WATCH and it is still not a hard one (see the sibling
    for why). MEASURED 2026-10-07 with the vision tower on: 6.3 GB of 7.4 used while
    serving, and a sentence-transformers reference loaded beside it was OOM-killed.
    """
    weights_file = model_bytes if model_bytes is not None else MODEL_SAFETENSORS_BYTES
    weights_loaded = _loaded_weight_bytes(weights_file)
    mem = _meminfo()
    vram = _gpu_vram_bytes()

    site = _site_packages_dir()
    install_target = site if site is not None else Path(shutil.which(PYTHON_BIN) or "/usr")
    install_free, install_total = _disk_free(install_target)
    build_target = Path(TMPDIR) if TMPDIR else Path(tempfile.gettempdir())
    build_free, build_total = _disk_free(build_target)
    cache = _hf_cache_dir()
    weights_free, weights_total = _disk_free(cache)

    already = _vllm_files_present()
    install_need = 0 if already else VLLM_INSTALL_BYTES
    build_need = 0 if already else VLLM_INSTALL_BYTES // 3

    # 1 GiB for activations and the CUDA context: deliberately coarse, not measured.
    vram_overhead = 1 * GIB
    vram_need = weights_loaded + vram_overhead

    terms = [
        ("install disk", install_need, install_free, str(install_target)),
        ("build disk (TMPDIR)", build_need, build_free, str(build_target)),
        ("weights disk", weights_file, weights_free, str(cache)),
        ("VRAM", vram_need, vram, f"Tesla T4 {GPU_TOTAL_MIB} MiB"),
    ]
    short = [(name, need - free, where) for name, need, free, where in terms if need > free]

    return {
        "weights": weights_file,
        "weights_loaded": weights_loaded,
        "terms": terms,
        "short": short,
        "binding": short[0][0] if short else None,
        "fits": not short,
        "already_installed": already,
        "install_target": str(install_target),
        "build_target": str(build_target),
        "cache": str(cache),
        "ram_avail": mem.get("MemAvailable", 0),
        "ram_total": mem.get("MemTotal", 0),
        "swap_total": mem.get("SwapTotal", 0),
        "vram": vram,
        "vram_need": vram_need,
    }


def _vllm_files_present() -> bool:
    """Whether vLLM's files are on disk for the target interpreter.

    FOR THE DISK BUDGET ONLY — presence is never evidence vLLM works or that it is
    a version that knows this model. `verify_gpu_arch` asks the interpreter.
    """
    site = _site_packages_dir()
    if site is None:
        return False
    return (site / "vllm").is_dir()


def _g(byte_count: int) -> str:
    return f"{byte_count / 1e9:.2f} GB"


@mcp.tool()
async def check_host_capacity() -> str:
    """Decide whether this host can install and serve the configured checkpoint.

    RUN THIS FIRST. Free arithmetic over three filesystems and the GPU. Names the
    binding term, because the remedies (a redirected install, a bigger disk, a
    smaller checkpoint) are unrelated.
    """
    c = _capacity()
    lines = [
        f"📡 **Host capacity** — `{RIG_NAME}`",
        "",
        f"- GPU: Tesla T4, compute capability {COMPUTE_CAPABILITY}, {GPU_TOTAL_MIB} MiB ({_g(c['vram'])})",
        f"- Host RAM: total {_g(c['ram_total'])} · available {_g(c['ram_avail'])} · swap {_g(c['swap_total'])}",
        f"- Target interpreter: `{PYTHON_BIN}`"
        + (f", user base `{PYTHONUSERBASE}`" if PYTHONUSERBASE else " (no PYTHONUSERBASE set)"),
        "- vLLM files present for it: "
        + ("yes — **not evidence it works**, see `verify_gpu_arch`" if c["already_installed"] else "**no**"),
        "",
        f"**Budget for `{MODEL_NAME}` at {DTYPE}** ({_g(c['weights'])} on disk, "
        f"≤ {_g(c['weights_loaded'])} on the device with every tower loaded)",
        "",
        "| term | on | needed | available | verdict |",
        "| :--- | :--- | ---: | ---: | :--- |",
    ]
    for name, need, free, where in c["terms"]:
        verdict = "fits" if need <= free else f"**short {_g(need - free)}**"
        if need == 0:
            verdict = "already paid"
        lines.append(f"| {name} | `{where}` | {_g(need)} | {_g(free)} | {verdict} |")
    lines.append(f"| host RAM | — | ~6.3 GB MEASURED with vision on | {_g(c['ram_avail'])} | the term to watch |")
    lines.append("")

    if c["fits"]:
        lines.append(
            "✅ **Every term fits.** There is no KV cache to size — the encoder is bidirectional "
            "and keeps none — so VRAM is weights plus activations. Next: `verify_gpu_arch`, then "
            "`apply_encoder_patch`."
        )
    else:
        binding = c["short"][0]
        lines.append(
            f"❌ **{binding[0].upper()} BINDS — short by {_g(binding[1])}** on `{binding[2]}`."
            + ("" if len(c["short"]) == 1 else f" ({len(c['short'])} terms are short; this is the first.)")
        )

    if not c["already_installed"]:
        lines += [
            "",
            "**The install, with this host's redirections and THIS RIG'S OWN user base**",
            "",
            "Not `/opt1/pyuser`: that holds the generative siblings' vLLM 0.29 with their clamp "
            "applied, and this model needs a nightly plus transformers 5.19. The pinned set is "
            "`benchmarks/runs/2026-10-07-first-serve-t4/evidence/requirements-lock.txt`.",
            "",
            "```bash",
            f"PYTHONUSERBASE={PYTHONUSERBASE or '<big-volume>/pyuser-embed2'} \\",
            f"PIP_CACHE_DIR={PIP_CACHE_DIR or '<big-volume>/pipcache'} \\",
            f"TMPDIR={TMPDIR or '<big-volume>/tmp'} \\",
            f"  {PYTHON_BIN} -m pip install --user --break-system-packages --no-deps \\",
            "  --extra-index-url https://download.pytorch.org/whl/cu130 \\",
            "  -r benchmarks/runs/2026-10-07-first-serve-t4/evidence/requirements-lock.txt",
            "```",
            "",
            "**This rig will not run that for you.** It is several GB and the operator's call.",
        ]

    lines += [
        "",
        f"**Host RAM is the term to watch.** {_g(c['ram_total'])} total; MEASURED 2026-10-07 at "
        "6.3 GB used while serving with the vision tower. A second copy of the model in this "
        "host's RAM (a sentence-transformers reference, say) is OOM-killed — stop the server "
        "first. `--limit-mm-per-prompt` with `image: 0` skips the tower and the load.",
    ]
    return "\n".join(lines)


@mcp.tool()
async def verify_gpu_arch() -> str:
    """Ask the TARGET interpreter whether its stack can serve this model on this GPU.

    Three questions, each a different failure: does the CUDA torch cover SM 7.5,
    does a float32 matmul execute, and is the installed vLLM/transformers new
    enough to know `EmbeddingGemma2Model` at all. The third is the one the shared
    `/opt1/pyuser` would fail — it is vLLM 0.29.
    """
    rc, smi, _ = await run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        timeout=60,
    )
    smi_out = smi.strip() if rc == 0 else f"(nvidia-smi unavailable: rc={rc})"

    rc, out, err = await _probe(
        "import torch;"
        "print('python', __import__('sys').version.split()[0]);"
        "print('torch', torch.__version__);"
        "print('torch.version.cuda', torch.version.cuda);"
        "print('arch list:', torch.cuda.get_arch_list());"
        "print('device:', torch.cuda.get_device_name(0));"
        "print('capability:', torch.cuda.get_device_capability(0));"
        "x = torch.randn(256, 256, device='cuda', dtype=torch.float32);"
        "print('fp32 matmul ok:', bool(torch.isfinite((x @ x).sum())));"
        "import transformers, vllm;"
        "print('transformers', transformers.__version__);"
        "print('vllm', vllm.__version__);"
        "from vllm.model_executor.models.registry import ModelRegistry;"
        "print('EmbeddingGemma2Model registered:', 'EmbeddingGemma2Model' in ModelRegistry.get_supported_archs());"
        "import transformers.models.embedding_gemma2;"
        "print('transformers embedding_gemma2: present')"
    )
    body = (out + err).strip()

    if rc == 127:
        return f"❌ `{PYTHON_BIN}` not found. `PYTHON_BIN` in `tpu.env` names the interpreter."

    notes = []
    if "+cpu" in body or "torch.version.cuda None" in body:
        notes.append("❌ **CPU build of torch.** It cannot use the T4; its empty arch list is not an answer.")
    elif "No module named 'torch'" in body or "No module named 'vllm'" in body:
        notes.append(f"⚠️ Not installed for `{PYTHON_BIN}`. `check_host_capacity` prints the install.")
    elif "No module named 'transformers.models.embedding_gemma2'" in body:
        notes.append(
            "❌ **transformers is too old for this model** — `embedding_gemma2` first ships in "
            "5.19.0. vLLM's model file imports it, so the engine cannot load."
        )
    elif "registered: False" in body:
        notes.append(
            "❌ **This vLLM does not know `EmbeddingGemma2Model`.** Support merged 2026-10-06 "
            "(`02b83919aa`) and is in no release; are you probing `/opt1/pyuser` (vLLM 0.29) "
            "instead of this rig's user base?"
        )
    elif "fp32 matmul ok: True" in body and "present" in body:
        if "sm_75" in body:
            notes.append(
                "✅ SM 7.5 kernels, a float32 matmul on the device, and a vLLM + transformers that "
                "know this model. **Half the question:** the encoder attention kernel still needs "
                "`apply_encoder_patch`."
            )
        else:
            notes.append("❌ **No SM 7.5 in the arch list.** Check the probe ran against the intended interpreter.")

    return (
        f"### GPU arch probe — `{RIG_NAME}` via `{PYTHON_BIN}`\n\n"
        f"```\n{smi_out}\n\n{body[:1800]}\n```\n\n" + "\n\n".join(notes)
    )


@mcp.tool()
async def apply_encoder_patch() -> str:
    """Make vLLM's encoder-only Triton attention fit Turing's 64 KiB per block.

    Without it engine start dies during profiling with `OutOfResources: shared
    memory, Required: 69632, Hardware limit: 65536` (MEASURED 2026-10-07). The
    target is THIS RIG'S site-packages (`PYTHONUSERBASE`), resolved through the
    interpreter rather than hardcoded, and a vLLM reinstall there reverts it.
    """
    rc, out, err = await _probe(f"import {_TRITON_MODULE} as m; print('__TARGET__' + m.__file__)", timeout=300)
    if rc != 0:
        blob = (err or out).strip()
        return (
            f"❌ Could not resolve `{_TRITON_MODULE}`.\n\n```\n{blob[:800]}\n```\n\n"
            f"Run `verify_gpu_arch` — it says whether this is a missing install or a broken one."
        )
    target = ""
    for line in out.splitlines():
        if line.startswith("__TARGET__"):
            target = line[len("__TARGET__") :].strip()
    if not target:
        return f"❌ Resolved no path for `{_TRITON_MODULE}`.\n\n```\n{out.strip()[:800]}\n```"

    try:
        proc = await asyncio.create_subprocess_exec(
            PYTHON_BIN,
            str(RIG_DIR / _PATCH_SCRIPT),
            target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_child_env(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        rc = proc.returncode or 0
    except (asyncio.TimeoutError, OSError) as exc:
        return f"❌ Could not run the patch: {exc}"

    body = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
    if rc == 0:
        return f"✅ **Patched** `{target}`\n\n```\n{body[:1500]}\n```\n\nConfirm with `verify_encoder_patch`."
    return (
        f"❌ **The patch REFUSED** — it never silently no-ops.\n\n```\n{body[:2000]}\n```\n\n"
        f"Upstream has probably restructured `context_attention_fwd`; read the patch script's "
        f"docstring before changing an anchor."
    )


@mcp.tool()
async def verify_encoder_patch() -> str:
    """Confirm the clamp is present in the module the target interpreter imports."""
    rc, out, err = await _probe(
        f"import {_TRITON_MODULE} as m;"
        "t = open(m.__file__).read();"
        "print('__FILE__' + m.__file__);"
        f"print('CLAMP ' + ('PRESENT' if {_PATCH_SENTINEL!r} in t else 'ABSENT'));"
        f"print('OCCURRENCES', t.count({_PATCH_SENTINEL!r}))",
        timeout=300,
    )
    if rc != 0:
        return f"❌ Could not import `{_TRITON_MODULE}`.\n\n```\n{(err or out).strip()[:800]}\n```"
    if "CLAMP ABSENT" in out:
        return (
            f"❌ **UNPATCHED.**\n\n```\n{out.strip()}\n```\n\n"
            f"Engine start will die with `OutOfResources: shared memory, Required: 69632, "
            f"Hardware limit: 65536`. Run `apply_encoder_patch`."
        )
    if "OCCURRENCES 1" not in out:
        return (
            f"⚠️  **Patched MORE THAN ONCE.**\n\n```\n{out.strip()}\n```\n\n"
            f"Reinstall vLLM in this rig's user base and apply once."
        )
    return f"✅ **Patched**, in the site-packages `{PYTHON_BIN}` imports.\n\n```\n{out.strip()}\n```"


def _serve_cmd() -> list[str]:
    """The engine argv. One place, so the tool, `make serve` and the tests agree."""
    cmd = [
        PYTHON_BIN,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--runner",
        RUNNER,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--dtype",
        DTYPE,
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--limit-mm-per-prompt",
        LIMIT_MM_PER_PROMPT,
        "--hf-overrides",
        HF_OVERRIDES,
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    return cmd


@mcp.tool()
async def start_vllm_server() -> str:
    """Start the vLLM embedding server (`/v1/embeddings`) on the attached T4.

    REFUSES unless vLLM is installed for this rig's interpreter, every capacity
    term fits, and the encoder clamp is POSITIVELY confirmed — not merely "not
    reported absent", which an import failure would also satisfy.

    MEASURED 2026-10-07: 96 s to healthy text only, 177 s with the vision tower.
    The generative T4 rigs use the same GPU and port: stop theirs first.
    """
    existing = _read_pid()
    if existing is not None:
        return f"✅ Already running (pid {existing}) at {ENDPOINT}. `stop_vllm_server` first."

    if not _vllm_files_present():
        return f"❌ vLLM is not installed for `{PYTHON_BIN}` under `{PYTHONUSERBASE}`. Run `check_host_capacity`."

    c = _capacity()
    if not c["fits"]:
        return f"❌ **Refusing to start: {c['binding']} binds.** Run `check_host_capacity`."

    patch_state = await verify_encoder_patch()
    if "✅ **Patched**" not in patch_state:
        return (
            "❌ **Refusing to start: the encoder clamp is not confirmed present.**\n\n"
            "float32 encoder attention needs 69,632 B of shared memory against Turing's 65,536, "
            "and the engine dies during profiling.\n\n`verify_encoder_patch` said:\n\n" + patch_state
        )

    RUN_DIR.mkdir(exist_ok=True)
    cmd = _serve_cmd()
    # subprocess.Popen, NOT asyncio.create_subprocess_exec, and this is the fix for
    # the sibling's `make serve`: asyncio kills its children when the event loop's
    # interpreter exits, so `python3 -c "asyncio.run(start_vllm_server())"` lost the
    # engine about a second after launch (gpu-vllm-t4-2b, 2026-09-21). A Popen child
    # in its own session outlives the caller. Still argv, never shell=True.
    with open(LOG_FILE, "ab") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, env=_child_env()
        )
    PID_FILE.write_text(str(proc.pid))
    return (
        f"📡 Started vLLM (pid {proc.pid}) → {ENDPOINT}\n\n```\n{' '.join(cmd)}\n```\n\n"
        f"Poll `server_status`; log at `{LOG_FILE}`. Then `verify_embeddings`."
    )


@mcp.tool()
async def stop_vllm_server() -> str:
    """Stop the running server. The VM and its T4 stay billed — this releases nothing."""
    pid = _read_pid()
    if pid is None:
        PID_FILE.unlink(missing_ok=True)
        return "✅ Not running."
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return f"❌ Could not signal pid {pid}: {exc}"
    PID_FILE.unlink(missing_ok=True)
    return f"✅ Sent SIGTERM to vLLM (pid {pid}). VRAM is released on exit; the VM stays billed."


@mcp.tool()
async def server_status() -> str:
    """Check whether the server is up, with live VRAM and the served model beside it.

    The model id is reported because the generative T4 rigs serve on the same port:
    a healthy `:8000` is not evidence it is THIS model.
    """
    pid = _read_pid()
    rc, smi, _ = await run_command(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ],
        timeout=60,
    )
    vram = f"VRAM {smi.strip()}" if rc == 0 else "VRAM unavailable (no nvidia-smi)"

    if pid is None:
        return f"❌ Not running (no pid file). Endpoint would be {ENDPOINT}.\n\n{vram}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{ENDPOINT}/health")
            models = await client.get(f"{ENDPOINT}/v1/models") if resp.status_code == 200 else None
        if resp.status_code == 200:
            served = [m.get("id") for m in models.json().get("data", [])] if models is not None else []
            return f"✅ Serving {served} at {ENDPOINT} (pid {pid}).\n\n{vram}"
        return f"📡 Process up (pid {pid}), `/health` → {resp.status_code}.\n\n{vram}"
    except (httpx.HTTPError, ValueError) as exc:
        return (
            f"📡 Process up (pid {pid}) but {ENDPOINT} is not answering yet ({exc}).\n\n{vram}\n\n"
            f"If the process vanishes, check `{LOG_FILE}` for `OutOfResources` (the clamp) and "
            f"`dmesg` for an OOM kill (host RAM)."
        )


async def _embed(inputs: list[str], dimensions: Optional[int] = None) -> list[list[float]]:
    """POST to /v1/embeddings and return the vectors in input order."""
    payload: dict = {"model": MODEL_NAME, "input": inputs}
    if dimensions:
        payload["dimensions"] = dimensions
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(f"{ENDPOINT}/v1/embeddings", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} from {ENDPOINT}: {resp.text[:400]}")
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@mcp.tool()
async def embed_text(texts: list[str], kind: str = "query", dimensions: Optional[int] = None) -> str:
    """Embed text with the model's trained task prefix, and summarise the vectors.

    kind: `query` (prefix `task: search result | query: `), `document` (prefix
    `title: none | text: `) or `raw` (no prefix — only if the caller already
    applied one). dimensions: 768 (default), 512, 256 or 128 — the Matryoshka sizes
    `HF_OVERRIDES` declares; anything else is rejected by the server.
    """
    prefix = {"query": QUERY_PREFIX, "document": DOCUMENT_PREFIX, "raw": ""}.get(kind)
    if prefix is None:
        return f"❌ kind must be query, document or raw, not {kind!r}."
    try:
        vectors = await _embed([prefix + t for t in texts], dimensions)
    except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as exc:
        return f"❌ {exc}"
    lines = [f"✅ **{len(vectors)} embedding(s)**, kind `{kind}`", ""]
    for text, vec in zip(texts, vectors):
        norm = _dot(vec, vec) ** 0.5
        head = ", ".join(f"{x:+.4f}" for x in vec[:4])
        lines.append(f"- dim {len(vec)} · norm {norm:.4f} · [{head}, …] — {text[:60]!r}")
    return "\n".join(lines)


@mcp.tool()
async def rank_documents(query: str, documents: list[str], dimensions: Optional[int] = None) -> str:
    """Rank documents against a query by cosine similarity, with the trained prefixes.

    The vectors are unit-normalised by the model's own Normalize module, so the dot
    product IS the cosine. Expect high scores across the board — Google notes
    unrelated sentences sit near 0.6-0.7 — so read the ORDER, not the magnitude.
    """
    try:
        vectors = await _embed([QUERY_PREFIX + query] + [DOCUMENT_PREFIX + d for d in documents], dimensions)
    except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as exc:
        return f"❌ {exc}"
    q, docs = vectors[0], vectors[1:]
    ranked = sorted(((_dot(q, v), d) for v, d in zip(docs, documents)), reverse=True)
    lines = [f"✅ **Ranked {len(documents)} document(s)** for {query!r}", ""]
    lines += [f"{i}. `{s:.4f}` — {d[:80]!r}" for i, (s, d) in enumerate(ranked, 1)]
    return "\n".join(lines)


@mcp.tool()
async def verify_embeddings() -> str:
    """Live sanity check: dimension, unit norm, ranking and Matryoshka truncation.

    Not a parity check — that needs a sentence-transformers reference, which does
    not fit in host RAM beside a running server (benchmarks/runs/.../scripts/parity.py
    does it in two phases). This catches a wrong model on the port, a missing task
    prefix, a broken pooler and a missing `HF_OVERRIDES`.
    """
    checks = []
    try:
        full = await _embed(
            [
                QUERY_PREFIX + "Which GPU has no bfloat16 datapath?",
                DOCUMENT_PREFIX + "The Tesla T4 is a Turing part with no bfloat16 support.",
                DOCUMENT_PREFIX + "Sourdough needs a starter, flour, water and salt.",
            ]
        )
        q, rel, irr = full
        checks.append((len(q) == EMBEDDING_DIM, f"dimension {len(q)} (expected {EMBEDDING_DIM})"))
        checks.append((abs(_dot(q, q) - 1.0) < 1e-3, f"unit norm ({_dot(q, q) ** 0.5:.5f})"))
        checks.append((_dot(q, rel) > _dot(q, irr), f"relevant {_dot(q, rel):.4f} > irrelevant {_dot(q, irr):.4f}"))
        short = (await _embed([QUERY_PREFIX + "Which GPU has no bfloat16 datapath?"], 256))[0]
        prefix = q[:256]
        cos = _dot(short, prefix) / (_dot(prefix, prefix) ** 0.5)
        checks.append(
            (len(short) == 256 and cos > 0.9999, f"dimensions=256 is the renormalised prefix (cos {cos:.6f})")
        )
    except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as exc:
        checks.append((False, f"request failed: {exc}"))
    ok = all(passed for passed, _ in checks)
    lines = [("✅" if ok else "❌") + f" **Embedding checks** — `{MODEL_NAME}` at {ENDPOINT}", ""]
    lines += [("- ✅ " if passed else "- ❌ ") + msg for passed, msg in checks]
    return "\n".join(lines)


@mcp.tool()
async def get_help() -> str:
    """List the tools this rig exposes."""
    tools = await mcp.list_tools()
    lines = [
        f"📡 **{MCP_SERVER_NAME}** — EmbeddingGemma 2 on one attached Tesla T4 (SM {COMPUTE_CAPABILITY}), "
        f"vLLM pooling runner, `/v1/embeddings`.",
        "",
    ]
    for tool in tools:
        lines.append(f"- **{tool.name}** — {(tool.description or '').splitlines()[0]}")
    lines += [
        "",
        "**The order matters.** `check_host_capacity` → `verify_gpu_arch` → `apply_encoder_patch` → "
        "`verify_encoder_patch` → `start_vllm_server` → `verify_embeddings`.",
        "",
        "**An embedding model, not a chat model.** There is no completion endpoint; "
        "`embed_text` and `rank_documents` add the model's trained task prefixes, which vLLM "
        "does not. **Not an A/B twin** of the generative T4 rigs that share this GPU.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
