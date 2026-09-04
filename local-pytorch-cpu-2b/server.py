"""Local PyTorch/transformers MCP server — no accelerator, Gemma 4 E2B on CPU.

WHAT THIS RIG IS. `transformers` + `torch` on the host CPU, loading
`google/gemma-4-E2B-it` in bf16 straight from the HF cache. No engine of our
own, no server framework: the model object is held in-process and
`generate()` is called on it. That is deliberate — it is the reference
implementation, and its value here is being the thing other rigs are checked
AGAINST rather than being fast.

THERE IS NO CONTROL PLANE. No EC2 launch, no AMI resolution, no spot handling,
no SSM, no Secrets Manager, no systemd unit, no boto3, no deploy tool.
`tests/test_server.py::TestNoCloudControlPlane` asserts the ABSENCE of that
vocabulary and is the most load-bearing class in the suite: this directory was a
verbatim copy of `gpu-jax-g4dn-2b` until 2026-09-04, and a fork of a cloud rig
keeps passing its own tests while describing hardware that does not exist.

THE BOX HAS A GPU AND THIS RIG DELIBERATELY IGNORES IT. The GTX 1650 Ti has
4096 MiB; these weights are 10.25 GB, and transformers has no analogue of
llama.cpp's lazy PLE gather, so `device_map="auto"` would spill most of the model
across PCIe and measure the link rather than either processor. A GPU PyTorch rig
here would need its own directory with slot 3 = `1650ti`. Do not add a device
flag to this one.

MEMORY IS THE CONSTRAINT, AND SAFETENSORS MAKES IT SUBTLER THAN IT LOOKS.
MEASURED 2026-09-04: the model loads in ~3 s and peak RSS immediately after load
is ~1.2 GB, not 10.25 GB, because safetensors mmaps the file and pages in
lazily. **RSS after load is therefore NOT the footprint** — the pages are touched
as generation walks the weights, and the real high-water mark only appears once
tokens have been produced. `check_host_capacity` sizes against the file, not
against RSS, for exactly this reason.

The failure mode is the one every `local` rig shares and no cloud rig does:
exceeding host RAM is not refused, it is paid for in swap, so an over-budget run
is indistinguishable from a slow one.
"""

import asyncio
import os
import resource
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

RIG_DIR = Path(__file__).resolve().parent
load_dotenv(RIG_DIR / "tpu.env")

RIG_NAME = RIG_DIR.name
MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", RIG_NAME)

MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it")
MODEL_SAFETENSORS_BYTES = int(os.environ.get("MODEL_SAFETENSORS_BYTES", "10246621918"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
DEVICE = os.environ.get("DEVICE", "cpu")
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "6"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

GIB = 1024 ** 3
mcp = FastMCP(MCP_SERVER_NAME)

# The loaded model, held for the life of the process. Loading is ~3 s off a warm
# page cache, so this is a convenience rather than the necessity it is on a rig
# whose weights take minutes to place.
_MODEL = None
_TOK = None


async def run_command(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run a command with no shell. Never shell=True — see CLAUDE.md."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")
    except asyncio.TimeoutError:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"


def _meminfo() -> dict:
    """Live host memory in bytes. MemAvailable, never MemFree: MemFree excludes
    reclaimable page cache and reads catastrophically low on a live desktop."""
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            out[key] = int(rest.strip().split()[0]) * 1024
    return out


def _rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _capacity() -> dict:
    """Go/no-go arithmetic, sized from the FILE rather than from RSS.

    bf16 is not a speed choice here, it is the only one that fits: fp32 would be
    2x the file and does not fit this host at all.
    """
    mem = _meminfo()
    weights = MODEL_SAFETENSORS_BYTES
    overhead = 1 * GIB          # activations, KV, the interpreter itself
    need = weights + overhead
    avail = mem.get("MemAvailable", 0)
    total = mem.get("MemTotal", 0)
    return {
        "available": avail, "total": total, "swap_free": mem.get("SwapFree", 0),
        "weights": weights, "overhead": overhead, "need": need,
        "fits": need < avail, "shortfall": max(0, need - avail),
        "fits_if_freed": need < total,
    }


@mcp.tool()
async def check_host_capacity() -> str:
    """Can this host hold the weights in RAM? Run before loading.

    Sized from the safetensors file, NOT from RSS. Safetensors mmaps, so RSS
    right after a load reads ~1.2 GB for a 10.25 GB checkpoint and would say
    everything is fine when it is not.
    """
    c = _capacity()
    g = lambda b: f"{b / 1e9:.2f} GB"  # noqa: E731
    lines = [
        f"📡 **Host capacity** — `{RIG_NAME}`", "",
        f"- RAM total {g(c['total'])} · **available {g(c['available'])}** · swap free {g(c['swap_free'])}",
        "",
        "| term | |", "| :--- | ---: |",
        f"| weights ({DTYPE}) | {g(c['weights'])} |",
        f"| activations + process | {g(c['overhead'])} |",
        f"| **needed** | **{g(c['need'])}** |",
        f"| available now | {g(c['available'])} |",
        "",
    ]
    if c["fits"]:
        lines.append(f"✅ **Fits now**, {g(c['available'] - c['need'])} spare. This is a live "
                     f"desktop — re-check immediately before loading.")
    elif c["fits_if_freed"]:
        lines.append(f"⚠️  **Short by {g(c['shortfall'])} right now, but it fits this machine** "
                     f"({g(c['total'])} total). Free something and re-check.\n\n"
                     f"Do not load anyway: the kernel accepts the allocation and pays for it out "
                     f"of {g(c['swap_free'])} of swap, and a thrashing run looks exactly like a "
                     f"slow one.")
    else:
        lines.append(f"❌ **Over by {g(c['shortfall'])} against the machine's full {g(c['total'])}.** "
                     f"fp32 would be {g(2 * c['weights'])} and is not an option here.")
    return "\n".join(lines)


def _load():
    """Load the model into this process. Idempotent."""
    global _MODEL, _TOK
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(TORCH_NUM_THREADS)
    _TOK = AutoTokenizer.from_pretrained(MODEL_NAME)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=getattr(torch, DTYPE), device_map=DEVICE
    )
    _MODEL.eval()


@mcp.tool()
async def load_model() -> str:
    """Load the checkpoint into this process, refusing when it will not fit.

    REFUSES rather than swapping. That is the one decision this rig takes away
    from the operator, and it is deliberate.
    """
    c = _capacity()
    if not c["fits"]:
        return (f"❌ **Refusing to load: short by {c['shortfall'] / 1e9:.2f} GB.** "
                f"Run `check_host_capacity`. The kernel would accept this and satisfy it "
                f"from swap, which reads as a slow load rather than a failed one.")
    if _MODEL is not None:
        return f"✅ Already loaded. Peak RSS so far {_rss_bytes() / 1e9:.2f} GB."
    import time
    t0 = time.perf_counter()
    await asyncio.to_thread(_load)
    el = time.perf_counter() - t0
    n = sum(p.numel() for p in _MODEL.parameters())
    return (
        f"✅ Loaded `{MODEL_NAME}` in {el:.1f}s — {n / 1e9:.3f}B params, {DTYPE}, "
        f"{TORCH_NUM_THREADS} threads.\n\n"
        f"Peak RSS {_rss_bytes() / 1e9:.2f} GB — **this is not the footprint.** "
        f"Safetensors mmaps the file and pages in lazily, so RSS only reaches the real "
        f"high-water mark once generation has walked the weights."
    )


@mcp.tool()
async def query_model(prompt: str, max_new_tokens: Optional[int] = None) -> str:
    """Generate a reply, and report decode rate alongside it.

    GEMMA 4 IS A REASONING MODEL. It emits a thinking block first, so a small
    budget measures the thinking phase and returns no answer at all — MEASURED on
    the llama.cpp sibling: 1274 characters of reasoning before 22 of answer.
    Hence the 1024 default.
    """
    import time
    budget = max_new_tokens or MAX_NEW_TOKENS
    if _MODEL is None:
        out = await load_model()
        if out.startswith("❌"):
            return out
    import torch

    enc = _TOK.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    n_in = enc["input_ids"].shape[-1]
    t0 = time.perf_counter()

    def _gen():
        with torch.inference_mode():
            return _MODEL.generate(**enc, max_new_tokens=budget, do_sample=False)

    out_ids = await asyncio.to_thread(_gen)
    el = time.perf_counter() - t0
    new = out_ids.shape[-1] - n_in
    text = _TOK.decode(out_ids[0][n_in:], skip_special_tokens=True)
    body = ["✅ **Reply**", "", text, "", "---",
            f"prompt {n_in} tok · generated {new} tok in {el:.1f}s · "
            f"**{new / el:.3f} tok/s** · peak RSS {_rss_bytes() / 1e9:.2f} GB",
            "",
            "This is a single end-to-end figure covering prefill and decode together — "
            "`generate()` exposes no split. It is NOT comparable to a sibling's decode-only "
            "rate."]
    return "\n".join(body)


@mcp.tool()
async def host_status() -> str:
    """Report the CPU, live memory, and whether the model is resident."""
    mem = _meminfo()
    rc, out, _ = await run_command(["nproc"], timeout=10)
    return (
        f"📡 **Host** — `{RIG_NAME}`\n\n"
        f"- {os.environ.get('HOST_CPU', '?')} · {out.strip() or '?'} threads visible, "
        f"using {TORCH_NUM_THREADS}\n"
        f"- RAM available {mem.get('MemAvailable', 0) / 1e9:.2f} GB of "
        f"{mem.get('MemTotal', 0) / 1e9:.2f} GB · swap free "
        f"{mem.get('SwapFree', 0) / 1e9:.2f} GB\n"
        f"- model loaded: {'yes' if _MODEL is not None else 'no'} · "
        f"peak RSS {_rss_bytes() / 1e9:.2f} GB\n\n"
        f"If available RAM has collapsed while a generation runs, that is swapping, not work."
    )


@mcp.tool()
async def get_help() -> str:
    """List the tools this rig exposes."""
    tools = await mcp.list_tools()
    lines = [f"📡 **{MCP_SERVER_NAME}** — PyTorch/transformers on the local CPU, Gemma 4 E2B.", ""]
    for t in tools:
        lines.append(f"- **{t.name}** — {(t.description or '').splitlines()[0]}")
    lines += [
        "",
        "No provisioning tools, by design: there is no control plane and no chip to find.",
        "",
        "This is the reference implementation, not a fast one. Its job is to be the thing "
        "other rigs are checked against — the runtime control for `local-jax-cpu-2b` on the "
        "same host and checkpoint.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
