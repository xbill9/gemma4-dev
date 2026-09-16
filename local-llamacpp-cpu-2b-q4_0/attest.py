"""Which binary is answering on the endpoint — measured, not asserted.

THIS RIG IS ONE ARM OF A CONTROL. Its twin, `local-llamacpp-1650ti-2b-q4_0`,
serves the same GGUF from the same llama.cpp checkout on the same port, and the
two are run alternately so that the only difference between their numbers is the
device. Sharing the port is the point: the endpoint, the harness and the prompts
stay fixed while the device changes underneath them.

That design has exactly one failure mode, and it is silent. Nothing in an HTTP
response says which binary produced it. `sweep.py` took the rig name from its own
`--rig` argument, and `model_server_status` reported `✅ Serving` for whatever
held port 8080 — so restarting into the other arm and forgetting mislabels a
whole run as the device it is not, with no error anywhere.

So the arm is READ OFF THE RUNNING PROCESS, out of /proc:

* `/proc/<pid>/exe`     — the binary actually executing, not LLAMA_SERVER_BIN.
* `/proc/<pid>/maps`    — the shared objects it HAS LOADED. Strictly better than
                          `ldd` on the binary: llama.cpp dlopen's its ggml
                          backends, so a CUDA backend can be absent from `ldd`
                          and present here. It is also evidence about the process
                          that ran, not about a file on disk that may since have
                          been rebuilt.
* `/proc/<pid>/cmdline` — the real `-ngl`, whatever any env file claims.
* `/proc/<pid>/environ` — CUDA_VISIBLE_DEVICES as the child actually got it.

No subprocess and no shell (CLAUDE.md), and nothing here trusts `tpu.env`: the
point is to catch the case where the configuration and the process disagree.

`sha256` of the exe is recorded because it, not a commit string, is what makes
two arms pairable. A commit is what you meant to build; the hash is what ran.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

# This rig is the CPU arm. Not configurable, for the same reason `-ngl 0` is not:
# an arm that can be flipped by an env var measures whichever device it happened
# to find. The GPU twin sets this to "gpu". A test asserts it.
EXPECTED_DEVICE = "cpu"

# Substrings that, mapped into a live process, mean work can leave the CPU.
# Matched against /proc/<pid>/maps, so these are runtime facts.
GPU_LIB_MARKERS = ("ggml-cuda", "ggml-vulkan", "ggml-sycl", "ggml-hip", "ggml-musa",
                   "ggml-cann", "ggml-opencl", "ggml-metal", "libcuda", "libcudart",
                   "libcublas", "libvulkan", "libamdhip")


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

    `device` is the verdict a report should be stamped with. It is derived from
    two independent signals that must AGREE:

      * `gpu_libs` — a GPU backend is mapped into the process, and
      * `n_gpu_layers` — it was told to put layers there.

    Either one alone is not the question a control asks. A CUDA build with
    `-ngl 0` computes on the CPU but is not a clean CPU arm, because the device
    is initialised and llama.cpp can still move large prefill batches onto it —
    that is why this rig hides the device from the child rather than trusting
    `-ngl 0`. So the two disagreeing is reported as `mixed`, never quietly
    rounded to one or the other.
    """
    pid = pid_owning_port(port)
    if pid is None:
        return {"serving": False, "device": "none", "port": port}

    argv = _cmdline(pid)
    env = _environ(pid)
    gpu_libs = _mapped_gpu_libs(pid)
    ngl_raw = _flag_value(argv, "-ngl", "--n-gpu-layers", "--gpu-layers")
    try:
        ngl = int(ngl_raw) if ngl_raw is not None else 0
    except ValueError:
        ngl = 0
    cuda_visible = env.get("CUDA_VISIBLE_DEVICES")

    if gpu_libs and ngl > 0:
        device = "gpu"
    elif not gpu_libs and ngl == 0:
        device = "cpu"
    else:
        device = "mixed"

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
        "model_arg": _flag_value(argv, "-m", "--model"),
        "threads": _flag_value(argv, "-t", "--threads"),
        "threads_batch": _flag_value(argv, "-tb", "--threads-batch"),
        "ctx_size": _flag_value(argv, "-c", "--ctx-size"),
        "argv": argv,
    }


def describe(att: dict) -> str:
    """One line naming the arm, for a log or a tool response."""
    if not att.get("serving"):
        return f"nothing is listening on port {att.get('port')}"
    bits = [f"device={att['device']}", f"pid={att['pid']}", f"-ngl {att['n_gpu_layers']}"]
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
        return (f"the process on port {att['port']} is a **{att['device']}** server, "
                f"not {expected} ({describe(att)})")
    return None
