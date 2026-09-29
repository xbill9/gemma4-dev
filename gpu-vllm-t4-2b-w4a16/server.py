"""vLLM-on-an-attached-T4 lifecycle MCP server — Gemma 4 E2B, one Tesla T4.

STATUS 2026-09-17: SCAFFOLDED. Nothing installed, nothing served, nothing
measured. `check_host_capacity` has the live arithmetic and currently refuses.

WHAT THIS RIG IS. One NVIDIA Tesla T4 that is ALREADY ATTACHED to the Compute
Engine VM this process runs on. Platform slot `gpu` — "a general-purpose GPU
attached to a VM, any cloud" (@NAMING.md).

THERE IS NO CONTROL PLANE, AND THAT IS NOT THE SAME CLAIM AS `local`. Nothing
here launches an instance, resolves an AMI, bids on spot capacity, runs SSM,
reads a secret manager, writes a systemd unit, or imports boto3 — the GPU exists
whether or not this code runs, so the entire provisioning half of the EC2
siblings is absent. But the machine is a GCE VM rather than a workstation, and
@NAMING.md settles which slot that is in as many words: "SSH into a cloud VM you
provisioned and it keeps that VM's platform value." So this is the FIRST `gpu`
rig here with no provisioning tools, and it is still not a `local` rig.
`tests/test_server.py::TestNoCloudControlPlane` asserts the absence of that
vocabulary and is the most load-bearing class in the suite: a fork of a cloud rig
keeps passing its own tests while describing hardware that does not exist.

THREE FILESYSTEMS, AND A SINGLE `df` LIES ABOUT ALL OF THEM. This host's root
disk has 4.23 GB free, `/tmp` has 3.88 GB, and `/opt1` has 249.20 GB — and
`~/.cache` is a SYMLINK into `/opt1`, so the checkpoint has room while a default
`pip install` does not. The first draft of this rig read `df /`, concluded "disk
binds", and was wrong in both directions at once. `check_host_capacity` measures
each target path separately and says which term binds, because the remedies are
unrelated: a redirected install, a bigger boot disk, a bigger machine type and a
smaller checkpoint are four different actions.

VRAM IS NOT THE CONSTRAINT AND SHOULD NOT BE TREATED AS ONE.
`gpu-vllm-g4dn-2b` MEASURED 9.8 GiB of weights inside 15.0 GiB of usable HBM on
this same part and still got a 329,579-token KV pool.

THE INSTALL THAT IS ALREADY HERE IS BROKEN, AND INSTRUCTIVELY SO.
`/opt1/pyuser` (the user base for `/usr/bin/python3.13`) holds a CUDA vLLM
against **torch 2.11.0+cpu** — `torch.version.cuda` is None, the arch list is
empty, and `import vllm` raises `ImportError: libcudart.so.13`. Two packages
present, neither usable. That is why `_vllm_files_present()` feeds only the disk
budget and never the verdict, and why `verify_gpu_arch` asks the interpreter
instead of the filesystem.

TWO THINGS THE SIBLINGS ESTABLISHED THAT THIS RIG MUST NOT RE-DERIVE:

  * float16, not bfloat16. Turing has no bf16 datapath. State the reason
    precisely, because the wrong reason invites deleting the guard: bf16 does NOT
    fail — PyTorch upconverts and vLLM logs `Casting torch.bfloat16 to
    torch.float16` and proceeds (MEASURED on `gpu-vllm-g5g-2b` 2026-08-12). A
    silent cost is worse than an error.
  * The Triton tile clamp. Gemma 4's full-attention layers are 512 wide, only FA4
    or Triton handle heterogeneous head dims, FA4 is unavailable, so vLLM FORCES
    TRITON_ATTN and its tile wants 98,304 B against Turing's 65,536 limit. The
    backend is not overridable and `VLLM_ATTENTION_BACKEND` is not even a
    recognized variable. See docs/turing-on-a-gce-t4.md.

WHAT IS GENUINELY UNVERIFIED, AND IT DECIDES THE RIG: whether the PyPI `vllm`
wheel ships SM 7.5 kernels. For the docker image the answer is published and
known (amd64 carries 7.5, arm64 does not). A wheel is a different artifact, and
nobody in this tree has read its arch list. `verify_gpu_arch` asks the installed
build rather than assuming, and `start_vllm_server` refuses until it has.
"""

import asyncio
import hashlib
import logging
import os
import shutil
import signal
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

MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it")
MODEL_SAFETENSORS_BYTES = int(os.environ.get("MODEL_SAFETENSORS_BYTES", "10246621918"))
MODEL_W4A16_NAME = os.environ.get("MODEL_W4A16_NAME", "google/gemma-4-E2B-it-qat-w4a16-ct")
MODEL_W4A16_BYTES = int(os.environ.get("MODEL_W4A16_BYTES", "8316306646"))

GPU_TOTAL_MIB = int(os.environ.get("GPU_TOTAL_MIB", "15360"))
COMPUTE_CAPABILITY = os.environ.get("COMPUTE_CAPABILITY", "7.5")
DTYPE = os.environ.get("DTYPE", "float16")
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "auto")
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "8"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
TURING_SMEM_BUDGET = int(os.environ.get("TURING_SMEM_BUDGET", "60000"))

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8000")
ENDPOINT = os.environ.get("ENDPOINT", f"http://{HOST}:{PORT}")
VLLM_PIP_SPEC = os.environ.get("VLLM_PIP_SPEC", "vllm")

# THE INTERPRETER IS CONFIGURATION, NOT `python3`, and on this host that is the
# difference between a rig that can serve and one that cannot. `python3` here is
# pyenv 3.12.13 whose site-packages sits on the nearly-full root disk;
# /usr/bin/python3.13 with PYTHONUSERBASE=/opt1/pyuser puts packages on the 250 GB
# volume, and that user base ALREADY holds a vLLM. Every subprocess this module
# runs uses PYTHON_BIN and _child_env() — never a bare "python3".
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")
PYTHONUSERBASE = os.environ.get("PYTHONUSERBASE", "")
PIP_CACHE_DIR = os.environ.get("PIP_CACHE_DIR", "")
TMPDIR = os.environ.get("TMPDIR", "")

RUN_DIR = RIG_DIR / "run"
PID_FILE = RUN_DIR / "vllm.pid"
LOG_FILE = RUN_DIR / "vllm.log"

GIB = 1024**3

# The installed CUDA stack, not a checkpoint. MEASURED ON NO HOST — this is a
# deliberately coarse ARITHMETIC allowance for `pip install vllm`, which pulls
# torch plus the nvidia-cu* wheels. It is here so the disk verdict counts the
# install as well as the weights; a budget that counts only the checkpoint says
# this host is 6 GB short when it is nearer 14.
VLLM_INSTALL_BYTES = int(os.environ.get("VLLM_INSTALL_BYTES", str(7 * 10**9)))

_PATCH_SCRIPT = "patch_triton_turing.py"
# Must equal patch_triton_turing.SENTINEL. Duplicated rather than imported so
# this module stays importable on a host with no vLLM, and pinned by
# `test_the_sentinel_matches_the_patch_script` — if they drift, verification
# looks for a string the patch never writes and blames the wrong thing.
_PATCH_SENTINEL = "Turing shared-memory clamp"
_TRITON_MODULE = "vllm.v1.attention.ops.triton_unified_attention"

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

    PYTHONUSERBASE is the load-bearing one: without it the target interpreter
    cannot see the packages installed for it, and a probe reports "not installed"
    about an install that exists. TMPDIR and PIP_CACHE_DIR matter only for pip,
    and are set here too so that an operator copying a printed command gets a
    complete one — /tmp on this host has 3.88 GB, which a CUDA torch wheel does
    not unpack into.
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
    """Run a snippet under the TARGET interpreter, not under this process.

    This module runs under whatever python started the MCP server; the model runs
    under PYTHON_BIN. Asking the wrong one about torch is how a rig reports a
    confident answer about an interpreter that will never serve.
    """
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
    """Where a `pip install --user` for PYTHON_BIN would land.

    Derived from PYTHONUSERBASE rather than asked of the interpreter, because
    this is called from the capacity arithmetic, which must work when PYTHON_BIN
    does not exist at all.
    """
    if not PYTHONUSERBASE:
        return None
    base = Path(PYTHONUSERBASE)
    candidates = sorted(base.glob("lib/python3.*/site-packages"))
    return candidates[-1] if candidates else base


def _patch_digest() -> str:
    """Short content hash of the patch script beside this file.

    Resolved next to server.py rather than from a fixed path, so the rig root can
    move. Unlike the EC2 siblings there is nothing to stamp it onto — the patch
    is applied in place here — so this exists to tell an operator WHICH revision
    of the clamp a patched site-packages was written by.
    """
    return hashlib.sha256((RIG_DIR / _PATCH_SCRIPT).read_bytes()).hexdigest()[:12]


def _read_pid() -> Optional[int]:
    """The running server's pid, or None.

    Checked against /proc rather than trusted: a stale pid file outlives a
    Ctrl-C, and there is no control plane here to ask for the truth.
    """
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if Path(f"/proc/{pid}").exists() else None


def _meminfo() -> dict:
    """Live host memory, in bytes.

    MemAvailable, not MemFree: MemFree excludes reclaimable page cache and reads
    far too low, while MemAvailable is the kernel's own estimate of what a new
    allocation can actually get.
    """
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            out[key] = int(rest.strip().split()[0]) * 1024
    return out


def _disk_free(path) -> tuple[int, int]:
    """(free, total) bytes on the filesystem holding `path`.

    Walks up to the nearest existing ancestor, so it answers for a directory that
    does not exist yet — a site-packages that pip has not created, for instance.

    ALWAYS CALLED PER TARGET, NEVER ONCE. On this host `/`, `/tmp` and `/opt1` are
    three filesystems with 4.23, 3.88 and 249.20 GB free, and `~/.cache` is a
    symlink into the last of them. One `df` gives a confident answer about the
    wrong volume; that mistake is what the first draft of this rig made.
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
    """Total device memory, from `tpu.env` rather than from a live probe.

    Deliberately not nvidia-smi: this is the go/no-go arithmetic and it must run
    on a host with no driver at all, so the capacity report can explain itself
    when there is no GPU to ask. `verify_gpu_arch` is the tool that asks the
    device, and it is the one that would catch a mismatch.
    """
    return GPU_TOTAL_MIB * 1024 * 1024


def _capacity(model_bytes: Optional[int] = None) -> dict:
    """The whole of this rig's go/no-go arithmetic, in one place.

    FOUR TERMS, ON THREE FILESYSTEMS, AND THEY FAIL DIFFERENTLY — which is the
    entire reason this is a tool rather than a note:

      install disk  — where packages land, i.e. PYTHONUSERBASE's site-packages.
                      Fails with ENOSPC partway through a multi-GB download.
                      Remedy: point PYTHONUSERBASE at the big volume.
      build disk    — TMPDIR, where pip unpacks wheels before moving them. A
                      SEPARATE 3.88 GB filesystem on this host, and the one that
                      fails least visibly. Remedy: TMPDIR=/opt1/tmp.
      weights disk  — HF_HOME. 249 GB here, via a `~/.cache` symlink.
      VRAM          — weights + KV + activations. Fails at engine init. Remedy: a
                      smaller checkpoint, or less context.

    HOST RAM IS DELIBERATELY NOT A HARD TERM. vLLM mmaps the safetensors and
    copies shard by shard, so peak RSS is far below the checkpoint size, and
    asserting `weights <= RAM` would condemn this host on arithmetic nobody has
    measured. It is reported as a risk instead — 7.80 GB and no swap against a
    sibling's 16 GiB.

    VRAM IS EXPECTED TO BE THE TERM THAT PASSES, and saying so is the point: on
    the same silicon `gpu-vllm-g4dn-2b` MEASURED 9.8 GiB of weights inside 15.0
    GiB usable and still got a 329,579-token KV pool. A rig that reports "does not
    fit" without naming the term sends the operator to buy the wrong thing.
    """
    weights = model_bytes if model_bytes is not None else MODEL_SAFETENSORS_BYTES
    mem = _meminfo()
    vram = _gpu_vram_bytes()

    site = _site_packages_dir()
    install_target = site if site is not None else Path(shutil.which(PYTHON_BIN) or "/usr")
    install_free, install_total = _disk_free(install_target)
    build_target = Path(TMPDIR) if TMPDIR else Path(tempfile.gettempdir())
    build_free, build_total = _disk_free(build_target)
    cache = _hf_cache_dir()
    weights_free, weights_total = _disk_free(cache)

    # The install is counted only while it has not happened. Presence of the files
    # is NOT a claim that they work — on this host they do not.
    already = _vllm_files_present()
    install_need = 0 if already else VLLM_INSTALL_BYTES
    # pip needs room to unpack the largest wheel, not the whole install. torch is
    # the big one; a third of the install allowance is a deliberately coarse
    # stand-in, ARITHMETIC and not measured.
    build_need = 0 if already else VLLM_INSTALL_BYTES // 3

    vram_budget = int(vram * GPU_MEMORY_UTILIZATION)
    # Graph capture plus activations. The sibling MEASURED 0.16 GiB for graph
    # capture; 1 GiB is deliberately coarse and deliberately not zero.
    vram_overhead = 1 * GIB
    kv_pool = vram_budget - weights - vram_overhead

    terms = [
        ("install disk", install_need, install_free, str(install_target)),
        ("build disk (TMPDIR)", build_need, build_free, str(build_target)),
        ("weights disk", weights, weights_free, str(cache)),
        ("VRAM", weights + vram_overhead, vram_budget, f"Tesla T4 {GPU_TOTAL_MIB} MiB"),
    ]
    short = [(name, need - free, where) for name, need, free, where in terms if need > free]

    return {
        "weights": weights,
        "terms": terms,
        "short": short,
        "binding": short[0][0] if short else None,
        "fits": not short,
        "already_installed": already,
        "install_need": install_need,
        "install_free": install_free,
        "install_total": install_total,
        "install_target": str(install_target),
        "build_need": build_need,
        "build_free": build_free,
        "build_total": build_total,
        "build_target": str(build_target),
        "weights_free": weights_free,
        "weights_total": weights_total,
        "cache": str(cache),
        "ram_avail": mem.get("MemAvailable", 0),
        "ram_total": mem.get("MemTotal", 0),
        "swap_total": mem.get("SwapTotal", 0),
        "vram": vram,
        "vram_budget": vram_budget,
        "vram_overhead": vram_overhead,
        "kv_pool": kv_pool,
        "vram_fits": kv_pool > 0,
    }


def _vllm_files_present() -> bool:
    """Whether vLLM's files are on disk for the target interpreter.

    FOR THE DISK BUDGET ONLY. This answers "do we still have to spend the install
    bytes", and it is emphatically NOT evidence that vLLM works: on this host the
    files are present and `import vllm` raises `ImportError: libcudart.so.13`,
    because the vLLM is a CUDA build and the torch beside it is `2.11.0+cpu`.
    `verify_gpu_arch` is the tool that asks the interpreter; nothing here reads a
    verdict off a directory listing.

    Deliberately a file check rather than an import: importing vLLM pulls torch,
    which is slow and allocates, and this runs inside the capacity arithmetic.
    """
    site = _site_packages_dir()
    if site is None:
        return False
    return (site / "vllm").is_dir()


def _g(byte_count: int) -> str:
    return f"{byte_count / 1e9:.2f} GB"


def _alt_disk_phrase(reference: dict, alternative: dict) -> str:
    """One sentence on what the w4a16 checkpoint would do to the DISK verdict.

    Whether it changes the verdict is the only thing worth saying, and it is not
    the same question as whether it is smaller — it is smaller by 1.93 GB on every
    host, and on a host short by 13 GB that is irrelevant.
    """
    if alternative["disk_fits"]:
        margin = _g(alternative["disk_free"] - alternative["disk_need"])
        verdict = (
            "enough to change the verdict."
            if not reference["disk_fits"]
            else "and the reference build already fits, so this buys nothing here."
        )
        return f"{margin} spare — {verdict}"
    return f"a {_g(alternative['disk_shortfall'])} shortfall — not enough to change the verdict."


@mcp.tool()
async def check_host_capacity() -> str:
    """Decide whether this host can install and serve the configured checkpoint.

    RUN THIS FIRST. The arithmetic is free; the install is several GB onto a host
    where two of the three candidate filesystems are nearly full.

    It names which term binds, because they have unrelated remedies. On this host
    the interesting answer is that the CHECKPOINT fits comfortably while a default
    `pip install` does not — the reverse of what a single `df` suggests.
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
        + (
            "yes — **and that is not evidence it works**, see `verify_gpu_arch`" if c["already_installed"] else "**no**"
        ),
        "",
        f"**Budget for `{MODEL_NAME}` at {DTYPE}**",
        "",
        "| term | on | needed | available | verdict |",
        "| :--- | :--- | ---: | ---: | :--- |",
    ]
    for name, need, free, where in c["terms"]:
        verdict = "fits" if need <= free else f"**short {_g(need - free)}**"
        if need == 0:
            verdict = "already paid"
        lines.append(f"| {name} | `{where}` | {_g(need)} | {_g(free)} | {verdict} |")
    lines += [
        f"| host RAM (staging) | — | not a fixed figure | {_g(c['ram_avail'])} | UNMEASURED at this shape |",
        "",
    ]

    if c["fits"]:
        lines.append(
            f"✅ **Every term fits.** The KV pool comes out at {_g(c['kv_pool'])} "
            f"({_g(c['weights'])} of weights inside a {_g(c['vram_budget'])} budget). "
            f"Next: `verify_gpu_arch`, then `apply_turing_patch` — neither is optional on "
            f"this part."
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
            "**The install command, with the redirections this host needs**",
            "",
            "Both redirections are load-bearing and a default `pip install` fails without "
            f"them: packages would land on the filesystem holding `{c['install_target']}` "
            f"and unpack in `{c['build_target']}`.",
            "",
            "```bash",
            f"PYTHONUSERBASE={PYTHONUSERBASE or '<big-volume>/pyuser'} \\",
            f"PIP_CACHE_DIR={PIP_CACHE_DIR or '<big-volume>/pipcache'} \\",
            f"TMPDIR={TMPDIR or '<big-volume>/tmp'} \\",
            f"  {PYTHON_BIN} -m pip install --user {VLLM_PIP_SPEC}",
            "```",
            "",
            "**This rig will not run that for you.** It is several GB and irreversible "
            "enough to be the operator's call.",
        ]
    else:
        lines += [
            "",
            "**vLLM's files are here, and that is not the same as a working install**",
            "",
            "As of 2026-09-17 on this host they are present and BROKEN: a CUDA vLLM against "
            "`torch 2.11.0+cpu`, so `torch.version.cuda` is None, the arch list is empty, and "
            "`import vllm` raises `ImportError: libcudart.so.13`. Two packages present, "
            "neither usable. **Run `verify_gpu_arch`** — it asks the interpreter rather than "
            "the filesystem, and it is the only thing here that can tell the difference.",
        ]

    lines += [
        "",
        "**What host RAM does and does not decide**",
        "",
        f"This host has {_g(c['ram_total'])} and **no swap at all**, which INVERTS the "
        f"warning `local-vllm-cpu-2b` carries rather than repeating it. There, exceeding RAM "
        f"is accepted and paid for in 15.4 GB of swap, so a thrashing serve is "
        f"indistinguishable from a loading one. Here an over-budget allocation is an OOM "
        f"kill: loud, immediate, and in `dmesg`.",
        "",
        "It is not a hard term above because vLLM mmaps the safetensors and copies to the "
        "device shard by shard, so peak RSS is well below the checkpoint size. The closest "
        f"measured sibling had 16 GiB against this host's {_g(c['ram_total'])}. **UNMEASURED "
        f"at this shape** — a real risk, not a verdict, and the first run settles it.",
        "",
        "**Do not size KV from 18 KiB/token.** @MODELS.md derives 18,432 B/token from the "
        "geometry and is right about the geometry, but the vLLM CUDA path reports about half "
        "— the T4 sibling's 329,579-token pool works out at ~9,622 B/token, the same figure "
        "the L4 run produced. That is an OPEN DISCREPANCY in @MODELS.md. Size from the "
        "engine's own allocation log.",
    ]
    return "\n".join(lines)


@mcp.tool()
async def verify_gpu_arch() -> str:
    """Ask the TARGET interpreter whether its CUDA kernels cover this GPU.

    THIS IS THE RIG'S CENTRAL CHECK, and on this host it is not hypothetical: the
    vLLM already installed sits on `torch 2.11.0+cpu`, so `import vllm` raises
    `ImportError: libcudart.so.13` and `torch.cuda.get_arch_list()` is `[]`. Files
    on disk are not a working install, and a config flag being accepted proves
    nothing — a kernel either launches or it does not.

    IT ALSO SETTLES THE RIG'S ONE OPEN QUESTION: whether the published wheels
    carry SM 7.5 kernels. For the docker image the answer is known (the amd64
    manifest lists 7.5, the arm64 manifest of the same tag omits it); a wheel is a
    different artifact and nobody in this tree has read its arch list. **A CPU
    torch cannot answer it** — it reports `[]` for everything — so fix the torch
    first and re-run.

    It probes with float16, not bfloat16, which is the opposite of what an Ada rig
    should do: Turing has no bf16 datapath, so a bf16 probe passes by upconversion
    and tells you nothing about what executes.

    PASSING HERE DOES NOT MEAN THE MODEL WILL SERVE. Kernel coverage and the
    Triton shared-memory ceiling are independent problems. Run
    `apply_turing_patch` too.
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
        "print('cuda available:', torch.cuda.is_available());"
        "print('arch list:', torch.cuda.get_arch_list());"
        "print('device:', torch.cuda.get_device_name(0));"
        "print('capability:', torch.cuda.get_device_capability(0));"
        "p = torch.cuda.get_device_properties(0);"
        "print('shared mem per block (static):', p.shared_memory_per_block);"
        "x = torch.randn(256, 256, device='cuda', dtype=torch.float16);"
        "print('fp16 matmul ok:', bool(torch.isfinite((x @ x).sum())))"
    )
    body = (out + err).strip()

    if rc == 127:
        return f"❌ `{PYTHON_BIN}` not found. `PYTHON_BIN` in `tpu.env` names the interpreter."

    notes = []
    if "+cpu" in body or "torch.version.cuda None" in body:
        notes.append(
            "❌ **THIS IS A CPU BUILD OF TORCH.** It cannot use the T4 at all, and its empty "
            "arch list is not an answer about SM 7.5 — a CPU wheel reports `[]` for every "
            "architecture. Replace torch with a CUDA build for this interpreter, keeping the "
            "redirections (`PYTHONUSERBASE`, `PIP_CACHE_DIR`, `TMPDIR`) that this host needs, "
            "then re-run. **Read the installed vLLM's own torch requirement first** rather "
            "than pinning a version here."
        )
    elif "No module named" in body:
        notes.append(
            f"⚠️ Nothing is installed for `{PYTHON_BIN}` yet. `check_host_capacity` prints "
            f"the install command with this host's redirections."
        )
    elif "no kernel image is available" in body:
        notes.append(
            "❌ `no kernel image is available for execution on the device` — the installed "
            "build has no SM 7.5 kernels. **This is the answer to the rig's open question, "
            "and it is the bad one.** The fallback is the from-source build "
            "`gpu-vllm-g5g-2b` documents, on 2 vCPU."
        )
    elif "arch list:" in body and not any(
        token in body.split("arch list:")[1].split("\n")[0] for token in ("sm_75", "7.5")
    ):
        # `sm_75`, not `7.5`: torch prints `['sm_75', 'sm_80', ...]`. Matching on
        # "7.5" alone reported a WORKING Turing build as unsupported, which is the
        # worst direction for this check to fail in — it would send an operator to
        # a 67-minute from-source build they did not need. Caught by
        # `test_a_working_cuda_torch_reports_half_the_question_done`.
        notes.append(
            "❌ **No SM 7.5 in the arch list** — the wheel does not cover this GPU. Check the "
            "probe ran against the intended interpreter before concluding it."
        )
    elif "fp16 matmul ok: True" in body:
        notes.append(
            "✅ SM 7.5 is in the arch list and a real float16 matmul executed on the device. "
            "**Record this in `tpu.env` — it settles the rig's open question.**\n"
            "**It is half the question.** The Triton shared-memory ceiling is independent of "
            "kernel coverage: run `apply_turing_patch`."
        )
    if "shared mem per block (static): 49152" in body:
        notes.append(
            "ℹ️ The 49,152 above is the DEFAULT STATIC limit, not the number the clamp "
            "budgets against. A kernel opting into the dynamic shared-memory attribute "
            "reaches 65,536, which is what Triton measures against. Both are real; "
            "@HARDWARE.md has the distinction, and `verify_gpu_arch` prints it so that a "
            "reader who checks torch does not conclude the docs are wrong."
        )

    return (
        f"### GPU arch probe — `{RIG_NAME}` via `{PYTHON_BIN}`\n\n"
        f"```\n{smi_out}\n\n{body[:1500]}\n```\n\n" + "\n\n".join(notes)
    )


@mcp.tool()
async def apply_turing_patch() -> str:
    """Clamp Triton's attention tiles in the INSTALLED vLLM so they fit 64 KiB.

    Gemma 4's full-attention layers are 512 wide, only FA4 or Triton handle
    heterogeneous head dims, FA4 is unavailable, so vLLM forces TRITON_ATTN — and
    its tile wants 98,304 B against Turing's 65,536 limit. There is no flag: the
    tiles have to come down. Without this, engine start dies with
    `OutOfResources: shared memory, Required: 98304, Hardware limit: 65536`.

    THE TARGET IS THE HOST'S OWN site-packages, WHICH IS THE ONE WAY THIS IS MORE
    DANGEROUS THAN THE EC2 SIBLINGS' ROUTE. They patch a file inside a docker
    image they built and can throw away. This edits the system python's vLLM,
    shared with everything else on the box — and `pip install -U vllm` silently
    reverts it. Re-run after any vLLM upgrade.

    The module path is RESOLVED, never hardcoded: site-packages carries the
    python version, and a hardcoded path would patch a file nothing imports and
    report success.
    """
    rc, out, err = await _probe(f"import {_TRITON_MODULE} as m; print('__TARGET__' + m.__file__)", timeout=300)
    if rc != 0:
        # DISTINGUISH "not installed" FROM "installed and broken", because they
        # have nothing in common as remedies and this host is in the second state.
        # An earlier version guessed "probably not installed" and pointed at a disk
        # constraint that does not exist — both wrong, and wrong confidently.
        blob = (err or out).strip()
        if _vllm_files_present():
            diagnosis = (
                f"**vLLM's files ARE present for `{PYTHON_BIN}`, so this is a broken install "
                f"rather than a missing one.** On this host the cause is that the vLLM is a "
                f"CUDA build while the torch beside it is `2.11.0+cpu`, which surfaces as "
                f"`ImportError: libcudart.so.13`. Run `verify_gpu_arch` — it names the "
                f"mismatch — and fix torch for that interpreter before patching anything. "
                f"There is nothing to patch until the module imports."
            )
        else:
            diagnosis = (
                f"Nothing is installed for `{PYTHON_BIN}`. Run `check_host_capacity`: it "
                f"prints the install command with the redirections this host needs."
            )
        return f"❌ Could not resolve `{_TRITON_MODULE}`.\n\n```\n{blob[:800]}\n```\n\n{diagnosis}"
    target = ""
    for line in out.splitlines():
        if line.startswith("__TARGET__"):
            target = line[len("__TARGET__") :].strip()
    if not target:
        return f"❌ Resolved no path for `{_TRITON_MODULE}`.\n\n```\n{out.strip()[:800]}\n```"

    env_note = f"TURING_SMEM_BUDGET={TURING_SMEM_BUDGET}"
    script = str(RIG_DIR / _PATCH_SCRIPT)
    proc_env = _child_env()
    proc_env["TURING_SMEM_BUDGET"] = str(TURING_SMEM_BUDGET)
    try:
        proc = await asyncio.create_subprocess_exec(
            PYTHON_BIN,
            script,
            target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        rc = proc.returncode or 0
    except (asyncio.TimeoutError, OSError) as exc:
        return f"❌ Could not run the patch: {exc}"

    body = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
    if rc == 0:
        return (
            f"✅ **Patched** `{target}`\n\n`{env_note}` · patch revision `{_patch_digest()}`\n\n"
            f"```\n{body[:1500]}\n```\n\n"
            f"Confirm with `verify_turing_patch`. **A `pip install -U vllm` reverts this.**"
        )
    return (
        f"❌ **The patch REFUSED, and that is the designed behaviour** — it never silently "
        f"no-ops.\n\n```\n{body[:2000]}\n```\n\n"
        f"An unpatched engine behind a patched-looking install fails ~10 minutes later at "
        f"engine start, attributed to the wrong thing. Upstream has probably restructured "
        f"the file; read `docs/turing-on-a-gce-t4.md` before changing an anchor."
    )


@mcp.tool()
async def verify_turing_patch() -> str:
    """Confirm the clamp is present in the module vLLM actually imports.

    Two things can each be true while the deployment still fails: the patch was
    never applied, or it was applied to a DIFFERENT interpreter's site-packages
    than the one that will serve. Both are checked against `PYTHON_BIN` — the same
    interpreter the server is started with — which is the only comparison that
    means anything. On this host that is `/usr/bin/python3.13`, NOT the `python3`
    on PATH, and the two have different site-packages on different filesystems.
    """
    rc, out, err = await _probe(
        f"import {_TRITON_MODULE} as m;"
        "t = open(m.__file__).read();"
        "print('__FILE__' + m.__file__);"
        f"print('CLAMP ' + ('PRESENT' if {_PATCH_SENTINEL!r} in t else 'ABSENT'));"
        "print('OCCURRENCES', t.count('if current_platform.get_device_capability()[0] < 8:'))",
        timeout=300,
    )
    if rc != 0:
        return f"❌ Could not import `{_TRITON_MODULE}`.\n\n```\n{(err or out).strip()[:800]}\n```"
    if "CLAMP ABSENT" in out:
        return (
            f"❌ **UNPATCHED.**\n\n```\n{out.strip()}\n```\n\n"
            f"Engine start will die with `OutOfResources: shared memory, Required: 98304, "
            f"Hardware limit: 65536`. Run `apply_turing_patch`."
        )
    doubled = "OCCURRENCES 2" in out or "OCCURRENCES 3" in out
    if doubled:
        return (
            f"⚠️  **Patched MORE THAN ONCE.**\n\n```\n{out.strip()}\n```\n\n"
            f"Two clamps halve the tiles twice, which serves but throws away throughput for "
            f"no reason. Reinstall vLLM and apply once: `pip install --force-reinstall "
            f"{VLLM_PIP_SPEC}`, then `apply_turing_patch`."
        )
    return (
        f"✅ **Patched**, in the site-packages this host's `python3` imports.\n\n"
        f"```\n{out.strip()}\n```\n\nLocal patch revision `{_patch_digest()}`."
    )


@mcp.tool()
async def start_vllm_server() -> str:
    """Start the vLLM OpenAI-compatible server on the attached T4.

    REFUSES on three conditions, and each refusal is a failure that would
    otherwise be expensive to attribute:

      * over budget — see `check_host_capacity`;
      * vLLM not installed;
      * the Turing clamp absent, which fails ~10 minutes in at engine start with
        an OutOfResources that reads like a configuration problem.

    Startup is not fast and that is expected, not a symptom. The sibling MEASURED
    346 s to healthy, of which 150.90 s was engine init and 82.39 s of that was
    compilation — on 4 vCPU. This host has 2.
    """
    existing = _read_pid()
    if existing is not None:
        return f"✅ Already running (pid {existing}) at {ENDPOINT}. `stop_vllm_server` first."

    if not _vllm_files_present():
        return (
            f"❌ vLLM is not installed for `{PYTHON_BIN}`. Run `check_host_capacity` first — "
            f"it prints the install command with the redirections this host needs, and a "
            f"default `pip install` fails here on `/tmp`."
        )

    c = _capacity()
    if not c["fits"]:
        return (
            f"❌ **Refusing to start: {c['binding']} binds.** Run `check_host_capacity` for "
            f"the breakdown and the remedy — they differ by term, and a bigger machine type "
            f"does not fix a full disk."
        )

    # REFUSE UNLESS THE CLAMP IS POSITIVELY CONFIRMED, rather than refusing on a
    # recognised failure string. An earlier version tested for "UNPATCHED" and so
    # sailed past the case this host is actually in: verification that cannot
    # import vLLM at all returns neither "patched" nor "UNPATCHED", and a
    # whitelist of known-bad answers lets every unknown-bad answer through.
    patch_state = await verify_turing_patch()
    if "✅ **Patched**" not in patch_state:
        return (
            "❌ **Refusing to start: the Turing clamp is not confirmed present.**\n\n"
            "Gemma 4's 512-wide full-attention prefill tile wants 98,304 B of shared memory "
            "and Turing allows 65,536, so an unclamped engine spends ten minutes loading and "
            "then dies with `OutOfResources` — attributed to the wrong thing.\n\n"
            "`verify_turing_patch` said:\n\n" + patch_state
        )

    RUN_DIR.mkdir(exist_ok=True)
    cmd = [
        PYTHON_BIN,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--dtype",
        DTYPE,
        "--kv-cache-dtype",
        KV_CACHE_DTYPE,
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
    ]
    with open(LOG_FILE, "ab") as log:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=log, stderr=log, start_new_session=True)
    PID_FILE.write_text(str(proc.pid))
    return (
        f"📡 Started vLLM (pid {proc.pid}) → {ENDPOINT}\n\n```\n{' '.join(cmd)}\n```\n\n"
        f"**Expect minutes, not seconds.** Poll `server_status`; log at `{LOG_FILE}`.\n\n"
        f"Read the engine's own `GPU KV cache size:` line when it appears — that is the "
        f"authoritative KV figure for this rig, and it is the number @MODELS.md's open "
        f"discrepancy says to trust over either derived one."
    )


@mcp.tool()
async def stop_vllm_server() -> str:
    """Stop the running server.

    Teardown is complete and nothing is billed by it — but note that the VM and
    its attached GPU are billed whether this process runs or not, which is the one
    cost difference from every rig here that provisions its own capacity.
    """
    pid = _read_pid()
    if pid is None:
        PID_FILE.unlink(missing_ok=True)
        return "✅ Not running."
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return f"❌ Could not signal pid {pid}: {exc}"
    PID_FILE.unlink(missing_ok=True)
    return (
        f"✅ Sent SIGTERM to vLLM (pid {pid}). VRAM is released on exit.\n\n"
        f"The T4 stays attached and the VM stays billed — stopping the server is not "
        f"releasing capacity here."
    )


@mcp.tool()
async def server_status() -> str:
    """Check whether the server is up, with live VRAM beside it.

    VRAM IS REPORTED HERE ON PURPOSE. "Still compiling" and "failed at engine
    init" look identical from the outside for the first few minutes, and the tell
    is whether device memory has been claimed.
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
        return f"❌ Not running. Endpoint would be {ENDPOINT}.\n\n{vram}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{ENDPOINT}/health")
        if resp.status_code == 200:
            return f"✅ Serving at {ENDPOINT} (pid {pid}).\n\n{vram}"
        return f"📡 Process up (pid {pid}), `/health` → {resp.status_code}.\n\n{vram}"
    except httpx.HTTPError as exc:
        return (
            f"📡 Process up (pid {pid}) but {ENDPOINT} is not answering yet ({exc}).\n\n{vram}\n\n"
            f"If VRAM used is still near zero after a few minutes, this is compiling rather "
            f"than loading — the sibling MEASURED 82.39 s of compilation on twice this "
            f"host's vCPU. If the process has vanished, check `dmesg` for an OOM kill: this "
            f"host has no swap, so host-memory pressure kills rather than thrashes."
        )


@mcp.tool()
async def query_model(prompt: str, max_tokens: int = 1024) -> str:
    """Send a chat completion to the local endpoint.

    /v1/chat/completions, not /v1/completions — raw completions return an empty
    string on `-it` checkpoints (root CLAUDE.md).

    GEMMA 4 IS A REASONING MODEL, which is a second and unrelated way to get an
    empty reply: the thinking block comes first and a small `max_tokens` truncates
    mid-thought. MEASURED on a sibling: 1274 characters of reasoning before 22
    characters of answer. Hence 1024.
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
                f"{choice.get('finish_reason')}` after {usage.get('completion_tokens', '?')} "
                f"tokens, all of them thinking. Re-run with a larger `max_tokens` "
                f"(currently {max_tokens})."
            )
        return "\n".join(
            [
                "✅ **Reply**",
                "",
                text,
                "",
                "---",
                f"prompt {usage.get('prompt_tokens', '?')} tok · completion {usage.get('completion_tokens', '?')} tok",
            ]
        )
    except httpx.HTTPError as exc:
        return f"❌ Could not reach {ENDPOINT}: {exc}. Is the server running?"
    except (KeyError, IndexError, ValueError) as exc:
        return f"❌ Unexpected response shape from {ENDPOINT}: {exc}"


@mcp.tool()
async def get_help() -> str:
    """List the tools this rig exposes."""
    tools = await mcp.list_tools()
    lines = [
        f"📡 **{MCP_SERVER_NAME}** — vLLM on one attached Tesla T4 (SM {COMPUTE_CAPABILITY}), Gemma 4 E2B.",
        "",
    ]
    for tool in tools:
        lines.append(f"- **{tool.name}** — {(tool.description or '').splitlines()[0]}")
    lines += [
        "",
        "**No provisioning tools, and none is owed.** The GPU is already attached to this "
        "VM: nothing here launches an instance, resolves an image, or releases capacity. "
        "That is not the same claim as `local` — the machine is a Compute Engine VM, so the "
        "platform slot is `gpu` (@NAMING.md).",
        "",
        "**The order matters.** `check_host_capacity` → `verify_gpu_arch` → "
        "`apply_turing_patch` → `verify_turing_patch` → `start_vllm_server`. The first is "
        "free arithmetic and as of 2026-09-17 it REFUSES: 4.23 GB of disk against a "
        "10.25 GB checkpoint plus a multi-GB CUDA install. The GPU is not the problem.",
        "",
        "**The A/B twin is `gpu-vllm-g4dn-2b`** — same TU104 T4, same clocks, same vLLM, "
        "same checkpoint, on EC2 with a control plane. It has MEASURED this configuration; "
        "this rig has measured nothing.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
