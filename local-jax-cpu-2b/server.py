"""Local CPU lifecycle and inference MCP server for Gemma 4 under pure JAX.

There is no control plane here. Every sibling rig in this monorepo spends most of
its server.py asking a cloud for hardware — a Cloud TPU queued resource, a
Compute Engine instance, an EC2 spot launch — and the hardware is the thing that
can fail. This rig runs on the machine the agent is already on, so provisioning,
capacity, quotas, AMIs, spot reclamation, SSM, instance profiles and secrets
managers all collapse to nothing.

What replaces them is smaller but not empty, and it is the same *shape* of
problem: the host is a fixed budget you can exceed, so the tools that matter are
the ones that measure it before you commit (`check_host_capacity`), the one that
turns an install into evidence rather than a flag that was accepted
(`verify_cpu_backend`), and the ones that manage the serving process and read its
own judgement of itself back out (`start_jax_server` … `verify_model_health`).

WHY THIS RIG EXISTS. It is the zero point. Every other rig here measures an
accelerator, and none of them can tell you how much of a result is the chip and
how much is the engine, because there is no run without a chip. This one has no
accelerator at all, so it is the only place a JAX number for this engine can be
attributed entirely to the model port and the XLA CPU backend. It is also the
only rig with no per-hour cost and no capacity risk, which makes it the right
place to reproduce an engine-level bug — `docs/padding-window-eviction.md` was
found and fixed on a GPU and *verified on CPU*, and that verification could have
happened here first.

IT WILL BE SLOW, AND THAT IS NOT THE INTERESTING CONSTRAINT. The binding one is
memory. On an accelerator the weights sit in a device budget you can read off a
spec sheet; here they sit in the same RAM as everything else on the machine, and
when they do not fit the machine does not raise, it swaps — so the failure mode
is a serve that never gets slower than "still loading". `check_host_capacity`
exists to make that arithmetic explicit before a load, not after.

PROVENANCE. This rig was forked from `gpu-jax-g4dn-2b` (itself a `gpu-jax-g5g-2b`
fork) on 2026-08-29. Every performance number in the inherited prose was measured
on Turing GPUs. Nothing in this file quotes one as its own; values measured on
THIS host are marked MEASURED and name the date.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import time

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

# This rig's identity. The directory name is the single identifier everything
# else derives from — MCP server name, log channel, state directory, and the
# `rig` label the serving process stamps on every metric series. A literal rather
# than basename(__file__) because the installed skill copy lives at
# .claude/skills/<skill>/mcp/server.py, where deriving from the path yields "mcp".
RIG_NAME = "local-jax-cpu-2b"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(RIG_NAME)

MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", RIG_NAME)
mcp = FastMCP(MCP_SERVER_NAME)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-E2B-it")

# Bind loopback by DEFAULT, unlike every cloud sibling, which binds 0.0.0.0
# because the whole point there is to reach the box from somewhere else. This
# server has no authentication of any kind and runs on a developer machine that
# is not behind a cloud security group, so binding a public interface would put
# an unauthenticated inference endpoint on the network. Override deliberately.
JAX_HOST = os.getenv("JAX_HOST", "127.0.0.1")
JAX_PORT = int(os.getenv("JAX_PORT", "8000"))

# The interpreter that runs the serving process. The monorepo forbids virtualenvs
# — these rigs deploy as an image with system-wide site-packages — so this is the
# system python3 by default and the deps go into it.
PYTHON = os.getenv("PYTHON", shutil.which("python3") or "python3")

# --- Precision ---------------------------------------------------------------
# DTYPE is an OVERRIDE, not the decision: ports/gemma4/jax_e_model.py reads the
# live device and picks. On CPU there is no compute capability to read, so
# IS_PRE_AMPERE is False and the port resolves bfloat16 — which is the right
# answer here for a reason that has nothing to do with speed.
#
# XLA:CPU has no bf16 datapath either; it upconverts to fp32 in front of every
# use, exactly the emulation the Turing siblings pay. The difference is that on
# those rigs float16 was an available escape and here float32 is NOT: E2B is
# 9.26 GB at 2 bytes per parameter and 18.5 GB at 4, against 14.3 GB of host RAM
# (MEASURED on this host 2026-08-29). So the dtype tax is unavoidable rather than
# a bug to fix, and the arithmetic that makes it unavoidable is memory.
#
# Leave it empty to let the port decide. Set bfloat16/float16 only to override.
DTYPE = os.getenv("DTYPE", "bfloat16")
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto")
QUANT_MODE = os.getenv("QUANT_MODE", "fp16")

# CPU prefill is linear in the PADDED bucket and there is no accelerator to hide
# it. 4096 is what the Turing siblings default to because that is where their
# prefill transient stopped fitting in 14.07 GB of device memory; here the limit
# is patience rather than an allocator, and a bucket you will not wait for is not
# a bucket worth compiling. Lowered to 2048; raise it if you are willing to wait.
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "2048"))

# Not a knob. Gemma4EModelJAX raises NotImplementedError for B > 1 and the decode
# step donates its KV buffers, so concurrency is not an axis on any rig here.
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "1"))

# --- The two quantisation levers, and why they resolve differently here -------
# Both were measured on a T4G 2026-08-26 (gpu-jax-g5g-2b) against the 9.257 GB
# dense baseline:
#
#   ple_bits=4                 5.752 GB  (-3.505)   decode unchanged
#   int8_lm_head               9.660 GB  (+0.403)   decode +2.3%
#   ple_bits=4 + int8_lm_head  6.155 GB  (-3.102)   decode +2.3%
#
# PLE_BITS=4 carries over unconditionally: it is a pure memory win, the table is
# a gather rather than a matmul so decode never streams it, and memory is this
# rig's binding constraint.
#
# INT8_LM_HEAD does NOT carry over, and the sibling default is deliberately
# inverted here. It ADDS 0.403 GB — an int8 copy of the tied embedding placed
# alongside — to buy a throughput win that comes from halving the bytes READ per
# decode step. That is a bandwidth trade, and it is the wrong trade on a host
# where the constraint is resident bytes and the surplus goes to swap. Turn it on
# only after check_host_capacity says there is room to spare.
PLE_BITS = int(os.getenv("PLE_BITS", "4"))
INT8_LM_HEAD = os.getenv("INT8_LM_HEAD", "0").lower() in ("1", "true", "yes")

# Empty = one-shot prefill. Structurally unreachable while window_kv resolves
# True, which it does whenever max_model_len > sliding_window (2048 > 512), so
# setting this raises at startup unless you also set window_kv=off. Recorded here
# because the flag exists and looks like the obvious remedy for a slow prefill.
PREFILL_CHUNK_SIZE = os.getenv("PREFILL_CHUNK_SIZE", "")

# XLA's CPU backend sizes its intra-op thread pool from the host by default.
# Empty means "let it", which is what you want on a dedicated box; set it to pin
# the serve to a subset of cores when you are also using the machine.
XLA_CPU_THREADS = os.getenv("XLA_CPU_THREADS", "")

# Force the CPU backend rather than hoping it is the only one. A machine that
# also has a CUDA plugin installed — plausible on a developer box that works on
# the GPU siblings — would silently grab the GPU and this rig would stop being
# what its name says it is. `jax.devices()` reporting CpuDevice is then a fact,
# not a coincidence, and verify_cpu_backend asserts it.
JAX_PLATFORMS = os.getenv("JAX_PLATFORMS", "cpu")

# The XLA compilation cache. Unlike every cloud sibling this is NOT ephemeral —
# there is no instance to reclaim — so it survives restarts with no S3 sync, no
# timer unit, and no operator-supplied bucket. That whole apparatus is deleted
# rather than ported.
#
# It also fixes, by construction, the bug written up in the inherited CLAUDE.md:
# ports/gemma4/jax_e_model.py sets jax_compilation_cache_dir unconditionally at
# import and wins over jax_openai_server.py's choice, so on the cloud rigs the
# configured directory stayed empty while ~/.cache/jax_compilation_cache filled
# up. Here the default IS that path, so the two agree and the knob is honest.
# expanduser on the RESOLVED value, not just the default: tpu.env spells this
# `~/.cache/...` and dotenv does not expand a tilde the way a shell would, so a
# literal "~" would otherwise reach the serving process and become a directory
# named "~" in the working directory.
JAX_COMPILATION_CACHE_DIR = os.path.expanduser(
    os.getenv("JAX_COMPILATION_CACHE_DIR") or "~/.cache/jax_compilation_cache"
)

# Where the pidfile and the serving log live. Not /var/log and not a systemd
# journal: this runs as the invoking user, with no root and no unit file.
STATE_DIR = os.getenv(
    "STATE_DIR",
    os.path.join(
        os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"),
        RIG_NAME,
    ),
)

# What a serve needs importable. requirements-serving.txt mirrors this list and a
# test asserts they agree, because a drifted pair is invisible until a serve.
#
# jinja2 is NOT optional despite being imported nowhere in this rig: transformers
# renders the chat template through it and every serving path goes through
# apply_chat_template. Without it /health returns 200 and every
# /v1/chat/completions returns 500 — measured on the G5g parent 2026-08-19.
# transformers does not pull it in, and it memoizes the availability check at
# import, so installing it late needs a restart.
_SERVING_REQUIREMENTS = (
    "fastapi", "uvicorn", "pydantic", "transformers",
    "safetensors", "huggingface_hub", "numpy", "jinja2",
)

# Import names, where they differ from the distribution name above. Checking
# importability rather than `pip show` is the point: this rig's failure mode is a
# package installed into an interpreter that is not the one PYTHON resolves to.
_IMPORT_NAMES = {"huggingface_hub": "huggingface_hub", "jinja2": "jinja2"}

# Profiling deps. Their own stage on the cloud rigs so a failure is attributable
# and does not stop the box serving; here they are simply optional and
# check_dependencies reports them separately for the same reason.
_PROFILING_REQUIREMENTS = ("xprof", "tensorboard")

# The serving payload — this rig's own sources. On the cloud siblings these are
# tarred and shipped over SSM, which is where the stale-deploy hazard came from.
# Nothing is shipped here: the process runs these files in place. The digest is
# kept anyway, because it still answers a real question — whether the process you
# are talking to is running the working tree or the skill snapshot.
_PAYLOAD_FILES = (
    "jax_openai_server.py",
    "jax_engine.py",
    "ports/gemma4/jax_e_loader.py",
    "ports/gemma4/jax_e_model.py",
)

# E2B's own numbers, from the root MODELS.md, used by check_host_capacity to do
# the fit arithmetic BEFORE a load rather than discovering it in swap.
_WEIGHT_BYTES_DENSE = 9_257_000_000        # measured on the G5g parent
_PLE_SAVING_BYTES = {0: 0, 8: 2_330_000_000, 4: 3_505_000_000}
_INT8_HEAD_COST_BYTES = 403_000_000
# Prefill transient at a 2048 bucket. FLAT below ~4K on the parent (1.504 GiB at
# both 512 and 1536), so this is the flat term rather than a per-token rate.
_PREFILL_TRANSIENT_BYTES = 1_615_000_000
_CHECKPOINT_DOWNLOAD_BYTES = 10_246_621_918   # google/gemma-4-E2B-it, one shard


# ---------------------------------------------------------------- host facts


def _read_meminfo() -> dict[str, int]:
    """/proc/meminfo in bytes. Empty on a platform that does not have it."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    try:
                        values[key] = int(parts[0]) * 1024
                    except ValueError:
                        continue
    except OSError:
        pass
    return values


def _host_facts() -> dict[str, object]:
    """RAM, swap, cores and free disk. Everything check_host_capacity reasons on.

    MemAvailable, not MemFree: free memory on a warm machine is near zero because
    the page cache holds the rest, and MemAvailable is the kernel's own estimate
    of what a new allocation can actually get. Quoting MemFree is how you conclude
    a 14 GB machine has 500 MB.
    """
    mem = _read_meminfo()
    try:
        cache_dir = os.path.expanduser("~/.cache")
        os.makedirs(cache_dir, exist_ok=True)
        usage = shutil.disk_usage(cache_dir)
        disk_free = usage.free
    except OSError:
        disk_free = 0
    return {
        "cores": os.cpu_count() or 0,
        "ram_total": mem.get("MemTotal", 0),
        "ram_available": mem.get("MemAvailable", 0),
        "swap_total": mem.get("SwapTotal", 0),
        "swap_free": mem.get("SwapFree", 0),
        "disk_free": disk_free,
    }


def _weight_estimate(ple_bits: int = PLE_BITS, int8_lm_head: bool = INT8_LM_HEAD) -> int:
    """Resident parameter bytes for the current levers. Arithmetic, not measured.

    Derived from the G5g parent's measured table; the parameter tree is a
    property of the checkpoint and the levers, not of the device, so it carries.
    """
    total = _WEIGHT_BYTES_DENSE - _PLE_SAVING_BYTES.get(ple_bits, 0)
    if int8_lm_head:
        total += _INT8_HEAD_COST_BYTES
    return total


def _gb(value: float) -> str:
    return f"{value / 1e9:.2f} GB"


# -------------------------------------------------------------- process state


def _state_path(name: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def _pidfile() -> str:
    return _state_path("server.pid")


def _logfile() -> str:
    return _state_path("server.log")


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def _proc_rss(pid: int) -> int:
    """Resident bytes for a pid, or 0. This is the number that matters here.

    On an accelerator rig the equivalent question is answered by the device's own
    allocator (`tpu_jax_hbm_used_bytes`). A CPU JAX device exposes no
    memory_stats, so RSS is the only honest reading of what the serve is costing,
    and it is what the metrics table substitutes.
    """
    try:
        with open(f"/proc/{pid}/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return 0


def _read_pid() -> int | None:
    """The pid of OUR serving process, or None.

    Verifies the command line rather than trusting the pidfile. A stale pidfile
    whose number has been recycled is the classic way a process manager reports a
    dead service as healthy, and on a developer box pid reuse is not rare.
    """
    try:
        with open(_pidfile()) as fh:
            pid = int(fh.read().strip())
    except (OSError, ValueError):
        return None
    if "jax_openai_server.py" not in _cmdline(pid):
        return None
    return pid


def _clear_pidfile() -> None:
    try:
        os.remove(_pidfile())
    except OSError:
        pass


def _payload_root() -> str:
    """Directory holding the serving payload.

    server.py is also installed as a skill snapshot at
    .claude/skills/<skill>/mcp/server.py, so look up from there too. Which one
    wins is worth knowing rather than guessing — start_jax_server prints it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.abspath(os.path.join(here, "..", "..", "..", ".."))):
        if all(os.path.exists(os.path.join(cand, f)) for f in _PAYLOAD_FILES):
            return cand
    raise RuntimeError(
        "serving payload not found next to server.py "
        f"(looked for {', '.join(_PAYLOAD_FILES)}). Start from the rig directory."
    )


def _payload_digest(root: str | None = None) -> str:
    """Content digest of the serving payload, as a short hex build id.

    Computed over the payload FILE CONTENTS. The serving process computes the
    same digest from the files it is executing and reports it on /health, so
    comparing the two answers "is the thing I am talking to running the tree I am
    editing?" — which on this rig is a question about WHICH COPY (working tree vs
    skill snapshot) rather than about a stale upload.
    """
    root = root or _payload_root()
    digest = hashlib.sha256()
    for rel in sorted(_PAYLOAD_FILES):
        digest.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()[:12]


def _serve_argv(model: str = MODEL_NAME) -> list[str]:
    """argv for jax_openai_server.py. A list — never a shell string.

    --quant-mode matches the CHECKPOINT, not the host: a `-w4a16-` export carries
    packed int4 weights and a dense export does not.

    --ple-bits is always emitted, including a 0, so the command records the choice
    rather than deferring to the server's own default.

    There is no tensor-parallel flag and no device selection. The engine is
    single-device (`jax.devices()[0]`) and on CPU that is one CpuDevice however
    many cores the host has; parallelism inside a step comes from XLA's thread
    pool, which is not something this argv can address.
    """
    root = _payload_root()
    argv = [
        PYTHON, os.path.join(root, "jax_openai_server.py"),
        "--model", model,
        "--host", JAX_HOST,
        "--port", str(JAX_PORT),
        "--kv-cache-dtype", KV_CACHE_DTYPE,
        "--quant-mode", QUANT_MODE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--ple-bits", str(PLE_BITS),
    ]
    if INT8_LM_HEAD:
        argv.append("--int8-lm-head")
    if PREFILL_CHUNK_SIZE:
        argv += ["--prefill-chunk-size", PREFILL_CHUNK_SIZE]
    return argv


def _serve_env() -> dict[str, str]:
    """Environment for the serving process.

    RIG_NAME reaches the metrics `rig` label through here, which is what keeps
    two rigs serving the same checkpoint distinguishable: the series names are
    byte-identical across the JAX rigs on purpose, because both benchmark reports
    compare on `tpu_jax_decode_tokens_per_second` BY NAME and renaming the prefix
    would break continuity with them.
    """
    env = dict(os.environ)
    env["RIG_NAME"] = RIG_NAME
    env["JAX_PLATFORMS"] = JAX_PLATFORMS
    env["JAX_COMPILATION_CACHE_DIR"] = JAX_COMPILATION_CACHE_DIR
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_payload_root(), env.get("PYTHONPATH", "")) if p
    )
    if DTYPE:
        env["JAX_E_COMPUTE_DTYPE"] = DTYPE
    if XLA_CPU_THREADS:
        # This caps the OpenMP/BLAS pools that numpy and the linear-algebra
        # backends bring along. It does NOT resize XLA:CPU's own intra-op thread
        # pool, which JAX exposes no documented flag for — the honest lever there
        # is `taskset` or a cgroup on the whole process. Say so rather than
        # implying a knob that does not exist: a flag being accepted is not
        # evidence it did anything, which is this monorepo's standing rule.
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[key] = XLA_CPU_THREADS
    return env


def _endpoint_base() -> str:
    """The OpenAI-compatible base URL. Resolved, never hardcoded.

    Trivial here compared with the cloud siblings — there is no instance to
    describe and no external IP to discover — but it still reads JAX_HOST/JAX_PORT
    rather than assuming, and it normalises a 0.0.0.0 bind to a loopback dial,
    which is the one way a local endpoint actually goes wrong.
    """
    host = JAX_HOST
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{JAX_PORT}/v1"


def _error(exc: Exception) -> str:
    """Render an exception for the MCP client, and put the traceback in the log.

    Tool bodies discard the stack: the client sees one line and stderr had
    nothing at all. logger.exception here covers every call site at once.
    """
    logger.exception("tool call failed: %s", exc)
    return f"❌ {exc}"


async def _run(argv: list[str], timeout: int = 120) -> tuple[int, str]:
    """Run a command and capture combined output. Never shell=True."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"`{argv[0]}` did not finish in {timeout}s") from None
    return proc.returncode or 0, (out or b"").decode(errors="replace")


# --------------------------------------------------------------------- tools


@mcp.tool(title="Save Hugging Face token", annotations=WRITE)
async def save_hf_token(token: str) -> str:
    """Write the token where huggingface_hub reads it, at mode 0600.

    The cloud siblings put this in AWS Secrets Manager and fetch it at boot into
    a root-only EnvironmentFile, because instance metadata is readable by
    anything on the box. There is no metadata service here and no boot; the
    equivalent care is the file mode and NOT echoing it back.

    google/gemma-4-E2B-it is readable anonymously (VERIFIED 2026-08-29), so this
    is only needed for a gated checkpoint.
    """
    try:
        token = token.strip()
        if not token:
            return "❌ Empty token."
        path = os.path.join(
            os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
            "token",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(token)
        return f"✅ Wrote {len(token)} characters to `{path}` (mode 0600)."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check host capacity", annotations=READ_ONLY)
async def check_host_capacity() -> str:
    """Can this host hold the configured model? Arithmetic, before you commit.

    This is the analogue of the cloud siblings' quota check, and it answers the
    same question — will the hardware take what I am about to ask of it — with
    the one difference that matters: exceeding a cloud quota is refused at the
    API, and exceeding host RAM is *accepted* and paid for in swap. A serve that
    is thrashing looks exactly like a serve that is loading, which is why this
    tool exists rather than a comment in the README.

    The verdict counts swap as usable, because it is, and says so separately —
    running the weights out of swap is a working configuration and a very slow
    one, and those are two different findings.
    """
    try:
        facts = _host_facts()
        weights = _weight_estimate()
        need = weights + _PREFILL_TRANSIENT_BYTES
        ram_avail = int(facts["ram_available"])
        swap_free = int(facts["swap_free"])
        headroom = ram_avail + swap_free

        cached = _checkpoint_cache_bytes()
        lines = [
            "### Host capacity",
            "",
            "| | |",
            "| --- | ---: |",
            f"| Cores | {facts['cores']} |",
            f"| RAM total | {_gb(int(facts['ram_total']))} |",
            f"| RAM available | {_gb(ram_avail)} |",
            f"| Swap free | {_gb(swap_free)} |",
            f"| Free disk (HF cache) | {_gb(int(facts['disk_free']))} |",
            "",
            "### What the configured model needs",
            "",
            "| | |",
            "| --- | ---: |",
            f"| Weights (`ple_bits={PLE_BITS}`, `int8_lm_head={int(INT8_LM_HEAD)}`) "
            f"| {_gb(weights)} |",
            f"| Prefill transient at bucket {MAX_MODEL_LEN} | {_gb(_PREFILL_TRANSIENT_BYTES)} |",
            f"| **Total** | **{_gb(need)}** |",
            f"| Checkpoint on disk | {_gb(_CHECKPOINT_DOWNLOAD_BYTES)} "
            f"({'cached' if cached else 'NOT cached — will download'}) |",
            "",
        ]

        if need <= ram_avail:
            lines.append(
                f"✅ Fits in available RAM with {_gb(ram_avail - need)} to spare. "
                "No swap needed."
            )
        elif need <= headroom:
            short = need - ram_avail
            lines.append(
                f"⚠️ Does NOT fit in RAM — {_gb(short)} short — but there is "
                f"{_gb(swap_free)} of swap, so it will load and run **out of swap**. "
                "That is a working configuration, not a healthy one: expect the "
                "load stages and the first request to be dominated by paging "
                "rather than by compute, and do not record a benchmark from it."
            )
        else:
            lines.append(
                f"❌ Does NOT fit: needs {_gb(need)} against {_gb(headroom)} of RAM "
                f"plus swap. Options, cheapest first: set `PLE_BITS=4` if it is not "
                f"already, clear `INT8_LM_HEAD`, lower `MAX_MODEL_LEN` (the "
                f"transient scales with the bucket above ~4K), or add swap."
            )

        if not facts["swap_total"]:
            lines += [
                "",
                "**No swap is configured.** On this host that was not a one-liner: "
                "the root filesystem is btrfs, where a `fallocate`d swapfile is "
                "copy-on-write and `swapon` refuses it with a bare "
                "`Invalid argument` (the real reason, `swapfile must not be "
                "copy-on-write`, appears only in dmesg). MEASURED 2026-08-29. "
                "Use `btrfs filesystem mkswapfile --size 16G /swapfile`, which "
                "sets NOCOW and disables compression, then `swapon`.",
            ]
        if int(facts["disk_free"]) < _CHECKPOINT_DOWNLOAD_BYTES and not cached:
            lines += ["", "❌ Not enough free disk for the checkpoint download."]
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


def _checkpoint_cache_bytes() -> int:
    """Bytes of *.safetensors already in the HF cache for MODEL_NAME, or 0.

    Reads the cache directly rather than asking the Hub, so it works offline and
    costs nothing. A partial download shows as a smaller number, not as absent —
    the `.incomplete` blobs are deliberately not counted.
    """
    root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    repo = "models--" + MODEL_NAME.replace("/", "--")
    base = os.path.join(root, "hub", repo, "snapshots")
    total = 0
    for dirpath, _, filenames in os.walk(base):
        for name in filenames:
            if name.endswith(".safetensors"):
                try:
                    total += os.stat(os.path.join(dirpath, name)).st_size
                except OSError:
                    continue
    return total


@mcp.tool(title="Verify the CPU backend", annotations=READ_ONLY)
async def verify_cpu_backend() -> str:
    """Prove JAX works on this host by running a real matmul, not by reading flags.

    The direct analogue of the GPU siblings' `verify_gpu_arch`, and it exists for
    the same reason: a config flag being accepted is not evidence it did anything.
    This imports jax in a subprocess, reports the resolved device and the dtype
    policy the model port derived from it, and multiplies two matrices in the
    compute dtype so the backend has to actually produce a number.

    It also asserts the device really is a CPU. On a machine that also has a CUDA
    plugin installed — plausible here, since the sibling rigs are GPU rigs — a
    missing JAX_PLATFORMS would hand this rig a GPU and every number it produced
    would be mislabelled.
    """
    script = (
        "import json, os, jax, jax.numpy as jnp\n"
        "d = jax.devices()[0]\n"
        "import sys; sys.path.insert(0, os.environ['PAYLOAD_ROOT'])\n"
        "from ports.gemma4.jax_e_model import COMPUTE_DTYPE, PLATFORM, HARDWARE\n"
        "x = jnp.ones((512, 512), COMPUTE_DTYPE)\n"
        "y = float((x @ x)[0, 0])\n"
        "print(json.dumps({'device': str(d), 'platform': d.platform,\n"
        "  'jax': jax.__version__, 'devices': len(jax.devices()),\n"
        "  'port_platform': PLATFORM, 'compute_dtype': jnp.dtype(COMPUTE_DTYPE).name,\n"
        "  'profile': HARDWARE.name, 'matmul': y}))\n"
    )
    try:
        env = _serve_env()
        env["PAYLOAD_ROOT"] = _payload_root()
        proc = await asyncio.create_subprocess_exec(
            PYTHON, "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode != 0:
            return (
                f"❌ JAX did not come up under `{PYTHON}`.\n\n```\n"
                f"{(err or b'').decode(errors='replace')[-3000:]}\n```"
            )
        info = json.loads((out or b"").decode().strip().splitlines()[-1])

        ok = info["platform"] == "cpu" and info["matmul"] == 512.0
        lines = [
            f"{'✅' if ok else '❌'} JAX {info['jax']} on `{info['device']}` "
            f"({info['devices']} device(s))",
            "",
            "| | |",
            "| --- | --- |",
            f"| Backend platform | `{info['platform']}` |",
            f"| Port's view | `{info['port_platform']}` |",
            f"| Compute dtype resolved | **`{info['compute_dtype']}`** |",
            f"| Hardware profile | `{info['profile']}` |",
            f"| 512x512 matmul | {info['matmul']} (expected 512.0) |",
            "",
        ]
        if info["platform"] != "cpu":
            lines.append(
                f"❌ **This is not a CPU device.** The backend resolved to "
                f"`{info['platform']}`, so `JAX_PLATFORMS={JAX_PLATFORMS}` did not "
                "take effect and anything this rig measures would be mislabelled."
            )
        if info["profile"] != "cpu" and info["platform"] == "cpu":
            lines.append(
                f"⚠️ `detect_hardware_profile()` reports `{info['profile']}`. The "
                "shared port falls back to the TPU profile off-accelerator; it is "
                "a default, not a reading of this host. Do not quote its numbers."
            )
        if info["compute_dtype"] == "bfloat16":
            lines.append(
                "⚠️ Compute dtype is `bfloat16`, and XLA:CPU has no bf16 datapath "
                "— it upconverts to fp32 in front of every use. That tax is "
                "unavoidable here rather than a bug: float32 storage would need "
                f"{_gb(_WEIGHT_BYTES_DENSE * 2)} against "
                f"{_gb(int(_host_facts()['ram_total']))} of host RAM."
            )
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check serving dependencies", annotations=READ_ONLY)
async def check_dependencies() -> str:
    """Report which serving dependencies `PYTHON` can actually import.

    The analogue of `get_install_progress`, and it checks importability under the
    exact interpreter that will run the serve rather than asking pip. This rig's
    characteristic failure is a package installed into a different interpreter —
    which `pip show` reports as present and an import resolves as absent.
    """
    names = [_IMPORT_NAMES.get(n, n) for n in _SERVING_REQUIREMENTS]
    script = (
        "import importlib, json, sys\n"
        f"want = {list(names) + list(_PROFILING_REQUIREMENTS) + ['jax', 'jaxlib']!r}\n"
        "out = {}\n"
        "for name in want:\n"
        "    try:\n"
        "        m = importlib.import_module(name)\n"
        "        out[name] = getattr(m, '__version__', 'present')\n"
        "    except Exception as exc:\n"
        "        out[name] = f'MISSING ({type(exc).__name__})'\n"
        "print(json.dumps({'python': sys.version.split()[0],\n"
        "                  'executable': sys.executable, 'mods': out}))\n"
    )
    try:
        code, out = await _run([PYTHON, "-c", script], timeout=180)
        if code != 0:
            return f"❌ `{PYTHON}` failed:\n\n```\n{out[-2000:]}\n```"
        info = json.loads(out.strip().splitlines()[-1])
        mods = info["mods"]
        missing = [
            n for n in [*names, "jax", "jaxlib"]
            if str(mods.get(n, "")).startswith("MISSING")
        ]
        lines = [
            f"{'❌' if missing else '✅'} Python {info['python']} at "
            f"`{info['executable']}`",
            "",
            "| Package | Version |",
            "| --- | --- |",
        ]
        for name in sorted(mods):
            optional = " *(optional)*" if name in _PROFILING_REQUIREMENTS else ""
            lines.append(f"| `{name}`{optional} | {mods[name]} |")
        if missing:
            lines += [
                "",
                f"❌ Missing on the serving path: {', '.join(f'`{m}`' for m in missing)}",
                "",
                "Install system-wide — the monorepo forbids virtualenvs, because "
                "these rigs run with system-wide site-packages:",
                "",
                "```bash",
                f"{PYTHON} -m pip install -r requirements.txt",
                "```",
            ]
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Fetch the checkpoint", annotations=WRITE)
async def fetch_checkpoint(model_name: str = MODEL_NAME) -> str:
    """Download the checkpoint into the HF cache, separately from a serve.

    Worth its own tool because otherwise the first `start_jax_server` spends ten
    minutes in a stage with no output and no way to tell a slow download from a
    hung one — the same blind spot the parent rig fixed with staged install
    timings. Downloading first makes the load stages mean what they say.
    """
    script = (
        "import json, time\n"
        "from huggingface_hub import snapshot_download\n"
        "t = time.time()\n"
        f"p = snapshot_download({model_name!r},\n"
        "    allow_patterns=['*.safetensors', '*.json'], max_workers=8)\n"
        "print(json.dumps({'path': p, 'seconds': time.time() - t}))\n"
    )
    try:
        code, out = await _run([PYTHON, "-c", script], timeout=7200)
        if code != 0:
            return f"❌ Download failed:\n\n```\n{out[-3000:]}\n```"
        info = json.loads(out.strip().splitlines()[-1])
        size = _checkpoint_cache_bytes()
        return (
            f"✅ `{model_name}` in the cache — {_gb(size)} of safetensors in "
            f"{info['seconds']:.0f}s.\n\n`{info['path']}`"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get the serve command", annotations=READ_ONLY)
async def get_serve_command(model_name: str = MODEL_NAME) -> str:
    """Print the exact command `start_jax_server` runs, and the env it sets.

    The analogue of the cloud siblings' `get_deployment_config`, and it has the
    same rule attached: what it prints must be what the tool does. On the parent
    rig those two drifted — `get_deployment_config` documented a 200 GB root
    volume while the launch tool created 100 — and a copy-pasteable repro that
    provisions something different from the tool it documents is how a manual
    reproduction fails to reproduce. Both render from one place here.
    """
    try:
        argv = _serve_argv(model_name)
        env = _serve_env()
        shown = {
            k: env[k] for k in
            ("RIG_NAME", "JAX_PLATFORMS", "JAX_COMPILATION_CACHE_DIR",
             "JAX_E_COMPUTE_DTYPE", "OMP_NUM_THREADS")
            if k in env
        }
        exports = "\n".join(f"export {k}={v}" for k, v in shown.items())
        return (
            f"### Serve command for `{RIG_NAME}`\n\n"
            f"```bash\n{exports}\n\\\n  " + " \\\n  ".join(argv) + "\n```\n\n"
            f"- Payload root: `{_payload_root()}`\n"
            f"- Build id: `{_payload_digest()}`\n"
            f"- Log: `{_logfile()}`\n"
            f"- Endpoint once up: `{_endpoint_base()}`\n"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Start the JAX server", annotations=WRITE)
async def start_jax_server(model_name: str = MODEL_NAME, restart: bool = False) -> str:
    """Launch jax_openai_server.py as a detached background process.

    Not systemd and not docker: this runs as the invoking user with no root, so
    there is no unit to `journalctl`. stdout and stderr go to one logfile that
    `get_jax_logs` tails, and the pid goes to a pidfile whose command line is
    verified on every read.

    Returns immediately. The load takes minutes — the checkpoint is ~10 GB — so
    poll `get_jax_logs`, which shows the staged load timings, and then
    `verify_model_health`.
    """
    try:
        existing = _read_pid()
        if existing and not restart:
            return (
                f"⚠️ Already running as pid {existing} "
                f"({_gb(_proc_rss(existing))} resident). Pass restart=True to "
                "replace it, or use get_jax_logs to see where it is."
            )
        if existing:
            await stop_jax_server()

        facts = _host_facts()
        need = _weight_estimate() + _PREFILL_TRANSIENT_BYTES
        headroom = int(facts["ram_available"]) + int(facts["swap_free"])
        if need > headroom:
            return (
                f"❌ Refusing to start: the configured model needs ~{_gb(need)} and "
                f"this host has {_gb(headroom)} of RAM plus swap. Run "
                "check_host_capacity for the options. Starting anyway would not "
                "fail cleanly — it would thrash."
            )

        root = _payload_root()
        argv = _serve_argv(model_name)
        log = _logfile()
        with open(log, "ab") as fh:
            fh.write(
                f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} starting "
                f"build_id={_payload_digest(root)} root={root} ===\n".encode()
            )
        # subprocess.Popen, NOT asyncio.create_subprocess_exec, and this is the
        # one deliberate exception to the monorepo's "every subprocess call goes
        # through create_subprocess_exec" rule. The part of that rule that
        # matters -- an argv list, never shell=True -- is kept.
        #
        # MEASURED HERE 2026-08-29, because the first version did use asyncio and
        # the serve died about a second after it started, with the device-policy
        # banner in the log and NOTHING after it: no traceback, no exit message.
        # asyncio's subprocess transport kills its child when the event loop is
        # deallocated, and `start_new_session=True` does not save it -- the kill
        # is a direct kill() on the pid, not a signal to a process group. So a
        # daemon launched through asyncio lives exactly as long as the tool call
        # that started it, which for a model that takes minutes to load is
        # indistinguishable from a crash during loading.
        #
        # Reproduced in isolation with `sh -c 'echo tick; sleep'`: asyncio wrote
        # one tick, Popen wrote all of them.
        with open(log, "ab") as out:
            proc = subprocess.Popen(   # argv list, never shell=True
                argv, cwd=root, env=_serve_env(),
                stdout=out, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        with open(_pidfile(), "w") as fh:
            fh.write(str(proc.pid))

        await asyncio.sleep(2)
        if _read_pid() is None:
            tail = _tail(log, 40)
            return f"❌ The process exited immediately.\n\n```\n{tail}\n```"

        warn = ""
        if need > int(facts["ram_available"]):
            warn = (
                "\n\n⚠️ This will run partly out of swap "
                f"({_gb(need - int(facts['ram_available']))} over available RAM). "
                "It will work and it will be slow; do not benchmark it."
            )
        return (
            f"✅ Started pid {proc.pid}, build id `{_payload_digest(root)}`.\n\n"
            f"- Payload root: `{root}`\n"
            f"- Log: `{log}`\n"
            f"- Endpoint (once loaded): `{_endpoint_base()}`\n\n"
            "The load is staged and timed — `get_jax_logs` shows download, "
            "read_shards, convert_params and device_put separately, so a hang is "
            "attributable to a stage." + warn
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Stop the JAX server", annotations=DESTRUCTIVE)
async def stop_jax_server(timeout: int = 30) -> str:
    """Stop the serving process: SIGTERM, then SIGKILL if it does not go.

    Stopping is genuinely cheap here — nothing was built and nothing is lost but
    the process's warm compilations, and those are on disk in the compilation
    cache, which unlike every cloud sibling's is not ephemeral. Do not import the
    "weigh stop against terminate" reasoning from the EC2 rigs.
    """
    try:
        pid = _read_pid()
        if pid is None:
            return "⚠️ Not running."
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + max(1, timeout)
        while time.time() < deadline:
            if _read_pid() is None:
                _clear_pidfile()
                return f"✅ Stopped pid {pid}."
            await asyncio.sleep(0.5)
        os.kill(pid, signal.SIGKILL)
        await asyncio.sleep(1)
        _clear_pidfile()
        return f"⚠️ pid {pid} ignored SIGTERM for {timeout}s and was killed."
    except ProcessLookupError:
        return "⚠️ Not running (the pid was already gone)."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List the serving process", annotations=READ_ONLY)
async def list_jax_servers() -> str:
    """Report whether a serve is running here, and what it is costing.

    RSS, not device memory: a CPU JAX device exposes no `memory_stats`, so the
    process's resident size is the honest reading of what the serve is using and
    the swap columns are what tell you whether it fits.
    """
    try:
        pid = _read_pid()
        facts = _host_facts()
        lines = [f"### `{RIG_NAME}` on this host", ""]
        if pid is None:
            lines += [
                "**Not running.** Start it with `start_jax_server`.",
                "",
                f"- Log from the last run: `{_logfile()}`",
            ]
        else:
            rss = _proc_rss(pid)
            lines += [
                "| | |",
                "| --- | --- |",
                f"| pid | {pid} |",
                f"| Resident (RSS) | **{_gb(rss)}** |",
                f"| Predicted weights | {_gb(_weight_estimate())} |",
                f"| Endpoint | `{_endpoint_base()}` |",
                f"| Build id | `{_payload_digest()}` |",
                f"| Command | `{_cmdline(pid)}` |",
            ]
        lines += [
            "",
            f"Host: {facts['cores']} cores, {_gb(int(facts['ram_total']))} RAM "
            f"({_gb(int(facts['ram_available']))} available), "
            f"{_gb(int(facts['swap_free']))} swap free.",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


def _tail(path: str, lines: int) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = min(size, max(4096, lines * 400))
            fh.seek(size - block)
            data = fh.read().decode(errors="replace")
        return "\n".join(data.splitlines()[-lines:])
    except OSError as exc:
        return f"(no log: {exc})"


@mcp.tool(title="Get JAX server logs", annotations=READ_ONLY)
async def get_jax_logs(tail: int = 100) -> str:
    """Tail the serving log.

    One file, not a journal — there is no systemd unit here. Everything the
    process writes lands in it, including the device-policy banner the port emits
    at import (platform, compute capability, resolved compute dtype), the staged
    load timings, the one `key=value` line per request, and the READY line
    carrying the whole resolved configuration.
    """
    try:
        tail = max(1, min(tail, 5000))
        return f"```\n{_tail(_logfile(), tail)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get the endpoint", annotations=READ_ONLY)
async def get_endpoint() -> str:
    """Resolve the OpenAI-compatible base URL, and say whether anything is on it."""
    try:
        base = _endpoint_base()
        pid = _read_pid()
        if pid is None:
            return f"📡 `{base}` — but nothing is running. Use `start_jax_server`."
        return f"📡 `{base}` (pid {pid})"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify model health", annotations=READ_ONLY)
async def verify_model_health() -> str:
    """Check /health, the served build id, and whether the reply was degenerate.

    Uses /v1/chat/completions: raw /v1/completions skips the chat template and is
    unreliable on `-it` models, so an empty body there is not evidence of a
    broken serve.

    It deliberately does NOT pass on "the reply was non-empty" — the rule this
    rig's engineering notes call out by name. On the vLLM sibling a broken deploy
    answered `': ok: ok: ok…'`, and KV-ring eviction returned a token loop with
    status="success". Instead this reads `tpu_jax_degenerate_responses_total`
    either side of its own probe, so the verdict is the server's own judgement of
    the full text.

    The build-id comparison is kept from the cloud siblings and means something
    different here: there is no deploy to be stale, so a mismatch says the running
    process is executing a DIFFERENT COPY of the payload — the skill snapshot
    rather than the working tree, or a tree edited since it started.
    """
    try:
        pid = _read_pid()
        if pid is None:
            return "❌ Not running. Start it with `start_jax_server`."
        base = _endpoint_base()
        metrics_url = base.replace("/v1", "/metrics")
        async with httpx.AsyncClient(timeout=300) as client:
            health = await client.get(base.replace("/v1", "/health"))
            before, _, _ = _parse_prom((await client.get(metrics_url)).text)
            chat = await client.post(
                f"{base}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                    "max_tokens": 16,
                },
            )
            after, _, _ = _parse_prom((await client.get(metrics_url)).text)

        body = chat.json()
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = body.get("usage") or {}
        key = "tpu_jax_degenerate_responses_total"
        degenerate = after.get(key, 0.0) > before.get(key, 0.0)

        try:
            health_body = health.json()
        except Exception:
            health_body = {}
        served_build = health_body.get("build_id", "unknown")
        try:
            local_build = _payload_digest()
        except Exception:
            local_build = "unavailable"

        ok = (
            health.status_code == 200
            and chat.status_code == 200
            and not degenerate
            and usage.get("completion_tokens", 0) > 0
        )
        lines = [
            f"{'✅' if ok else '❌'} health={health.status_code} "
            f"tokens={usage.get('completion_tokens', 0)} reply={text!r}",
            "",
            "- Degenerate (server's own verdict on the full text): "
            f"**{'YES — token loop' if degenerate else 'no'}**",
            f"- Build id served: `{served_build}`",
            f"- Resident: {_gb(_proc_rss(pid))}",
        ]
        if local_build not in ("unavailable", served_build) and served_build != "unknown":
            lines.append(
                f"- ⚠️ **DIFFERENT PAYLOAD**: this tree digests to `{local_build}`, "
                f"the process is serving `{served_build}`. It is running another "
                "copy of the sources (the skill snapshot, or the tree as it was "
                "before your edits). Restart it with `start_jax_server(restart=True)`."
            )
        elif local_build == served_build:
            lines.append(f"- Build id matches this tree (`{local_build}`).")
        if usage.get("pad_tokens") is not None:
            lines.append(
                f"- Shape: bucket={usage.get('bucket_size')} "
                f"pad={usage.get('pad_tokens')} cold={usage.get('cold_shape')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Query model", annotations=READ_ONLY)
async def query_model(prompt: str, max_tokens: int = 256, stats: bool = True) -> str:
    """Send a chat completion to the served model.

    With stats=True the reply carries the token counts the server reports in
    `usage`, the wall time, and the tok/s they imply.

    That rate is END-TO-END for one request: it includes prefill and the HTTP
    round trip, so it reads lower than decode throughput and is not the number to
    benchmark on. Prefer `get_metrics`, whose decode gauge times decode alone.

    It is also meaningless on a cold engine, and the gap is larger here than
    anywhere else in this monorepo: `max_new_tokens` is a `static_argnames` entry,
    so `(bucket, max_tokens)` IS the compiled shape, and XLA compiles it on the
    CPU you are also trying to measure. Warm up at the shape you intend to
    measure, not merely "once".
    """
    try:
        if _read_pid() is None:
            return "❌ Not running. Start it with `start_jax_server`."
        base = _endpoint_base()
        async with httpx.AsyncClient(timeout=1800) as client:
            started = time.perf_counter()
            response = await client.post(
                f"{base}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
            )
            wall = time.perf_counter() - started
        body = response.json()
        choice = body["choices"][0]
        text = choice["message"]["content"]
        if not stats:
            return text
        usage = body.get("usage") or {}
        completion = usage.get("completion_tokens") or 0
        rate = f"{completion / wall:.2f} tok/s" if completion and wall > 0 else "n/a"
        note = ""
        if usage.get("cold_shape"):
            note = " ⚠️ COLD — this shape was compiled during the request."
        return (
            f"{text}\n\n---\n"
            f"📡 {completion} completion + {usage.get('prompt_tokens', 0)} prompt tokens "
            f"in {wall:.2f}s — {rate} end-to-end "
            f"(finish: {choice.get('finish_reason')}).{note}"
        )
    except Exception as exc:
        return _error(exc)


def _parse_prom(text: str) -> tuple[dict, dict, str | None]:
    """Split a Prometheus exposition into (samples, precision labels, model).

    Pure and offline so the tests can pin it without a served endpoint — the
    rendering below is the part that needs one, the parsing is not.
    """
    samples, precision, served_model = {}, {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        series, _, raw = line.rpartition(" ")
        try:
            value = float(raw)
        except ValueError:
            continue
        name, _, labels = series.partition("{")
        tags = {}
        for part in labels.rstrip("}").split(","):
            key, sep, val = part.partition("=")
            if sep:
                tags[key] = val.strip('"')
        # The model label rides on every series and is identical across them, so
        # it is noise per row — but it is the only record of WHICH checkpoint
        # produced these numbers, so hoist it out rather than drop it.
        served_model = tags.pop("model", None) or served_model
        if name == "tpu_jax_precision_info":
            precision = tags
            continue
        rendered = ",".join(f'{k}="{v}"' for k, v in sorted(tags.items()))
        samples[f"{name}{{{rendered}}}" if rendered else name] = value
    return samples, precision, served_model


@mcp.tool(title="Get serving metrics", annotations=READ_ONLY)
async def get_metrics() -> str:
    """Read the server's Prometheus metrics, including the decode gauge.

    `tpu_jax_decode_tokens_per_second` is the like-for-like figure every rig here
    compares on, because it times decode alone. The series names carry the
    `tpu_jax_` prefix on a rig with no TPU and no accelerator at all, which is
    deliberate: both existing benchmark reports compare on that series BY NAME,
    and the `rig` label is what separates the rigs. Renaming the prefix would
    break continuity with them for no gain.

    The gauge describes the LAST request only. The counters are cumulative since
    the process started, and `tpu_jax_hbm_used_bytes` is 0 here because a CPU JAX
    device exposes no allocator stats — `list_jax_servers` reports RSS instead,
    which is the real number on this rig.
    """
    try:
        if _read_pid() is None:
            return "❌ Not running. Start it with `start_jax_server`."
        base = _endpoint_base()
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(base.replace("/v1", "/metrics"))
        if response.status_code != 200:
            return f"❌ /metrics returned {response.status_code}."

        samples, precision, served_model = _parse_prom(response.text)
        if not samples:
            return "❌ /metrics returned 200 but exposed no samples."

        lines = [f"### Serving metrics — `{RIG_NAME}`", ""]
        if served_model:
            lines += [f"Served checkpoint: `{served_model}`", ""]
        if precision.get("build_id"):
            lines += [
                f"Build id: `{precision['build_id']}` "
                f"(rig `{precision.get('rig', '?')}`)",
                "",
            ]
        if precision:
            kv, want = precision.get("kv_cache_dtype"), precision.get("kv_cache_requested")
            kv_shown = f"{kv} (requested `{want}`)" if want and want != kv else kv
            lines += [
                "| Precision | Resolved |",
                "| --- | --- |",
                f"| Compute dtype | **{precision.get('compute_dtype')}** |",
                f"| Quant mode | **{precision.get('quant_mode')}** |",
                f"| KV cache dtype | **{kv_shown}** |",
                f"| PLE bits | {precision.get('ple_bits')} |",
                f"| int8 lm_head | {precision.get('int8_lm_head')} |",
                "",
            ]
        lines += ["| Metric | Value |", "| --- | ---: |"]
        for key in sorted(samples):
            value = samples[key]
            shown = f"{value:.2f}" if value % 1 else f"{int(value)}"
            lines.append(f"| `{key}` | {shown} |")

        pid = _read_pid()
        if pid:
            lines += ["", f"Process resident size: **{_gb(_proc_rss(pid))}** "
                          "(there is no device allocator to read here)."]

        completions = samples.get("tpu_jax_completion_tokens_total", 0.0)
        decode_s = samples.get("tpu_jax_decode_seconds_total", 0.0)
        cold = samples.get("tpu_jax_cold_requests_total", 0.0)
        if completions and decode_s:
            lines += [
                "",
                f"Cumulative decode: **{completions / decode_s:.2f} tok/s** over "
                f"{int(completions)} completion tokens in {decode_s:.1f}s of decode. "
                "This excludes prefill and HTTP, so it is comparable to the "
                "`tpu_jax_decode_tokens_per_second` gauge.",
            ]
            if cold:
                lines.append(
                    f"⚠️ {int(cold)} request(s) compiled a new shape and are "
                    "averaged in here. Warm up before quoting this."
                )
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Help and configuration", annotations=READ_ONLY)
async def get_help() -> str:
    """Show this rig's resolved configuration and the constraints that shape it."""
    facts = _host_facts()
    weights = _weight_estimate()
    return f"""### {RIG_NAME}

Serving `{MODEL_NAME}` with **pure JAX on the local CPU** — no accelerator, no
cloud, no provisioning. The engine is this monorepo's own Gemma 4 port
(`ports/gemma4/`) behind an OpenAI-compatible FastAPI server, run as an ordinary
background process owned by the invoking user.

| Setting | Value |
| --- | --- |
| Host | {facts['cores']} cores, {_gb(int(facts['ram_total']))} RAM, {_gb(int(facts['swap_free']))} swap free |
| Interpreter | `{PYTHON}` |
| Backend | `JAX_PLATFORMS={JAX_PLATFORMS}` |
| Compute dtype | `{DTYPE or 'auto (the port decides)'}` |
| KV cache dtype | `{KV_CACHE_DTYPE}` |
| Quant mode | `{QUANT_MODE}` (matches the checkpoint, not the host) |
| PLE bits | `{PLE_BITS}` |
| int8 lm_head | `{int(INT8_LM_HEAD)}` |
| Max model len | `{MAX_MODEL_LEN}` |
| Predicted weights | {_gb(weights)} |
| Endpoint | `{_endpoint_base()}` |
| Log | `{_logfile()}` |
| XLA cache | `{JAX_COMPILATION_CACHE_DIR}` (persistent — nothing to reclaim) |

**Memory is the constraint, not speed.** It will be slow and that is expected;
what will actually stop it is RAM. E2B is {_gb(_WEIGHT_BYTES_DENSE)} of bf16
weights, {_gb(weights)} with the levers above, against
{_gb(int(facts['ram_total']))} of host RAM shared with everything else on the
machine. Exceeding it is not refused the way a cloud quota is — it is accepted
and paid for in swap, and a thrashing serve looks exactly like a loading one.
Run `check_host_capacity` before a load.

**bfloat16 is forced by memory, not chosen for speed.** XLA:CPU has no bf16
datapath and upconverts to fp32 in front of every use — the same tax the Turing
siblings pay. There the escape was float16; here there is none, because float32
storage would need {_gb(_WEIGHT_BYTES_DENSE * 2)}.

**`PLE_BITS=4` is on and `INT8_LM_HEAD` is off**, inverting the GPU siblings'
default. PLE quantisation is a pure memory win ({_gb(_PLE_SAVING_BYTES[4])}
saved, decode unchanged — the table is a gather, never a matmul). int8_lm_head
*adds* {_gb(_INT8_HEAD_COST_BYTES)} to buy a bandwidth win, which is the wrong
trade when the surplus goes to swap.

**Concurrency is not an axis.** `MAX_NUM_SEQS={MAX_NUM_SEQS}`, `Gemma4EModelJAX`
raises `NotImplementedError` for B > 1, and the decode step donates its KV
buffers. That is true of every rig here, not a local limitation.

Order of operations: `check_host_capacity` → `check_dependencies` →
`verify_cpu_backend` → `fetch_checkpoint` → `start_jax_server` → `get_jax_logs`
→ `verify_model_health` → `query_model` / `get_metrics`.
"""


if __name__ == "__main__":
    mcp.run()
