"""Which process is answering on the endpoint, and whether it computes on the GPU — read from /proc.

Copied 2026-11-03 from local-llamacpp-3050oem-2b-q4_0/attest.py and changed for vLLM in one place:
the verdict. llama.cpp's verdict needs both a mapped GPU backend and `-ngl > 0`. vLLM has no `-ngl`
and does its GPU work in a child process (`EngineCore`), so here the verdict is GPU libraries mapped
into the process that owns the port or any of its descendants, and CUDA_VISIBLE_DEVICES not empty.
Everything else (port -> inode -> pid, exe hash, argv) is the parent's.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

# This rig is the GPU arm. Not configurable: an arm that can be flipped by an env
# var measures whichever device it happened to find. The CPU twin sets this to
# "cpu". A test asserts it.
EXPECTED_DEVICE = "gpu"

# Substrings that, mapped into a live process, mean work can leave the CPU.
# Matched against /proc/<pid>/maps, so these are runtime facts.
GPU_LIB_MARKERS = (
    "ggml-cuda",
    "ggml-vulkan",
    "ggml-sycl",
    "ggml-hip",
    "ggml-musa",
    "ggml-cann",
    "ggml-opencl",
    "ggml-metal",
    "libcuda",
    "libcudart",
    "libcublas",
    "libvulkan",
    "libamdhip",
)


def listening_inodes(port: int) -> set[str]:
    """Socket inodes in TCP_LISTEN on `port`, from /proc/net/tcp and tcp6."""
    inodes: set[str] = set()
    for table in ("tcp", "tcp6"):
        try:
            rows = Path(f"/proc/net/{table}").read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if len(fields) < 10 or fields[3] != "0A":  # 0A == TCP_LISTEN
                continue
            try:
                if int(fields[1].rsplit(":", 1)[1], 16) != port:
                    continue
            except (IndexError, ValueError):
                continue
            inodes.add(fields[9])
    return inodes


def pid_owning_port(port: int) -> Optional[int]:
    """The pid holding the listening socket on `port`, or None.

    Read out of /proc rather than shelling out to `ss`/`lsof`: no subprocess, no
    dependency, and it answers the exact question every caller has — who owns
    ENDPOINT — rather than "is there a process whose cmdline looks like ours".
    """
    targets = {f"socket:[{inode}]" for inode in listening_inodes(port)}
    if not targets:
        return None
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for fd in (entry / "fd").iterdir():
                try:
                    if os.readlink(fd) in targets:
                        return int(entry.name)
                except OSError:  # fd closed under us, or not ours to read
                    continue
        except OSError:  # process exited between iterdir and open
            continue
    return None


def _proc_text(pid: int, name: str) -> str:
    try:
        return Path(f"/proc/{pid}/{name}").read_text(errors="replace")
    except OSError:
        return ""


def _cmdline(pid: int) -> list[str]:
    raw = _proc_text(pid, "cmdline")
    return [a for a in raw.split("\0") if a]


def _environ(pid: int) -> dict[str, str]:
    env = {}
    for item in _proc_text(pid, "environ").split("\0"):
        key, sep, value = item.partition("=")
        if sep:
            env[key] = value
    return env


def _exe(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""


def _mapped_gpu_libs(pid: int) -> list[str]:
    """GPU backends and driver libraries the process has actually mapped."""
    found: set[str] = set()
    for line in _proc_text(pid, "maps").splitlines():
        # The path is the last field; anonymous mappings have none.
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not parts[1].startswith("/"):
            continue
        name = os.path.basename(parts[1])
        found.update(m for m in GPU_LIB_MARKERS if m in name)
    return sorted(found)


def _children(pid: int) -> list[int]:
    """Every descendant pid of `pid`, from /proc/<pid>/task/*/children."""
    out: list[int] = []
    stack = [pid]
    while stack:
        p = stack.pop()
        try:
            tasks = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for t in tasks:
            try:
                kids = Path(f"/proc/{p}/task/{t}/children").read_text().split()
            except OSError:
                continue
            for k in kids:
                out.append(int(k))
                stack.append(int(k))
    return out


def _flag_value(argv: list[str], *names: str) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg in names and i + 1 < len(argv):
            return argv[i + 1]
        for name in names:
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
    return None


def sha256_of(path: str) -> str:
    """Content hash of the binary that ran. Pairing identity, not provenance."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def attest_port(port: int) -> dict:
    """Everything knowable about the process serving `port`, read from /proc.

    `device` is the verdict a report should be stamped with: `gpu` when a CUDA library is
    mapped into the process that owns the port or any descendant (vLLM's EngineCore) and
    CUDA_VISIBLE_DEVICES does not hide every card; `cpu` otherwise.
    """
    pid = pid_owning_port(port)
    if pid is None:
        return {"serving": False, "device": "none", "port": port}

    argv = _cmdline(pid)
    env = _environ(pid)
    procs = [pid] + _children(pid)
    gpu_libs = sorted({lib for p in procs for lib in _mapped_gpu_libs(p)})
    cuda_visible = env.get("CUDA_VISIBLE_DEVICES")
    # An empty CUDA_VISIBLE_DEVICES hides every card: whatever is mapped, nothing runs there.
    hidden = cuda_visible is not None and cuda_visible.strip() == ""
    device = "gpu" if gpu_libs and not hidden else "cpu"
    ngl = -1  # not a vLLM concept; kept so describe() and old readers still work

    exe = _exe(pid)
    return {
        "serving": True,
        "port": port,
        "pid": pid,
        "exe": exe,
        "exe_sha256": sha256_of(exe) if exe else "",
        "device": device,
        "gpu_libs": gpu_libs,
        "n_gpu_layers": ngl,
        "cuda_visible_devices": cuda_visible,
        "child_pids": procs[1:],
        "max_model_len": _flag_value(argv, "--max-model-len"),
        "max_num_seqs": _flag_value(argv, "--max-num-seqs"),
        "argv": argv,
    }


def describe(att: dict) -> str:
    """One line naming the arm, for a log or a tool response."""
    if not att.get("serving"):
        return f"nothing is listening on port {att.get('port')}"
    bits = [f"device={att['device']}", f"pid={att['pid']}", f"children={len(att.get('child_pids', []))}"]
    if att.get("gpu_libs"):
        bits.append("mapped: " + ", ".join(att["gpu_libs"]))
    if att.get("exe"):
        bits.append(att["exe"])
    return " · ".join(bits)


def mismatch(att: dict, expected: str = EXPECTED_DEVICE) -> Optional[str]:
    """Why `att` is not the arm `expected`, or None if it is."""
    if expected == "any":
        return None
    if not att.get("serving"):
        return f"nothing is serving on port {att.get('port')}"
    if att["device"] != expected:
        return f"the process on port {att['port']} is a **{att['device']}** server, not {expected} ({describe(att)})"
    return None
