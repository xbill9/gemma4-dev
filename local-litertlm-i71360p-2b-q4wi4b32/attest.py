"""Which process is answering on the endpoint, and on what device — measured, not asserted.

Forked in spirit from local-llamacpp-i71360p-2b-q4_0/attest.py; the code is new because
the process is different. That rig's server is a `llama-server` binary whose device is
decided by `-ngl`. Here the server is `litert-lm serve`, a Python process, and the device
is decided by the `backend` of the model's entry in the `--config` JSON it was started
with — and, at run time, by which accelerator libraries it maps.

The failure this catches is silent. Port 8092 answers the OpenAI API whichever
`litert-lm serve` holds it, started from whichever config, and nothing in a response says
which backend produced it. So everything is read off /proc:

* `/proc/<pid>/cmdline` — that it IS `litert-lm serve`, and the `--config` it was given.
* that config file      — the `backend` it assigns this rig's model (re-read, not trusted
                          from tpu.env; a missing entry or file means the CLI default).
* `/proc/<pid>/maps`    — GPU driver libraries actually mapped. liblitert-lm.so is one
                          file that dlopen's OpenCL / Vulkan / EGL / WebGPU for its GPU
                          backend, so a mapped driver means the GPU path was initialised.

No subprocess and no shell (CLAUDE.md).
"""

import json
import os
from pathlib import Path
from typing import Optional

# This rig is the CPU arm. Not configurable: an arm that can be flipped by an env var
# measures whichever device it happened to find. A test asserts it.
EXPECTED_DEVICE = "cpu"

GPU_LIB_MARKERS = (
    "libOpenCL",
    "libvulkan",
    "libEGL",
    "libGLES",
    "libcuda",
    "libwebgpu",
    "libdawn",
    "GpuAccelerator",
    "libnvidia",
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
    """The pid holding the listening socket on `port`, or None."""
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
                except OSError:
                    continue
        except OSError:
            continue
    return None


def _proc_text(pid: int, name: str) -> str:
    try:
        return Path(f"/proc/{pid}/{name}").read_text(errors="replace")
    except OSError:
        return ""


def _cmdline(pid: int) -> list[str]:
    return [a for a in _proc_text(pid, "cmdline").split("\0") if a]


def _cwd(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return ""


def _mapped_gpu_libs(pid: int) -> list[str]:
    found: set[str] = set()
    for line in _proc_text(pid, "maps").splitlines():
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


def is_litertlm_serve(argv: list[str]) -> bool:
    """`litert-lm serve`, however it was launched (console script or python -m)."""
    return "serve" in argv and any("litert-lm" in a or "litert_lm_cli" in a for a in argv)


def config_entry(config_path: Optional[str], model_id: str) -> dict:
    """The settings `litert-lm serve` applies to `model_id`: its `models` entry over `default`."""
    if not config_path:
        return {}
    try:
        cfg = json.loads(Path(config_path).read_text())
    except (OSError, ValueError):
        return {}
    return {**cfg.get("default", {}), **cfg.get("models", {}).get(model_id, {})}


def attest_port(port: int, model_id: str) -> dict:
    """Everything knowable about the process serving `port`, read from /proc.

    `device` must be agreed by two signals: the backend the config assigns `model_id`,
    and the GPU libraries mapped. A config backend of `cpu` with a GPU driver mapped is
    `mixed`, never rounded to one or the other.
    """
    pid = pid_owning_port(port)
    if pid is None:
        return {"serving": False, "device": "none", "port": port}
    argv = _cmdline(pid)
    config_path = _flag_value(argv, "--config")
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.join(_cwd(pid), config_path)
    entry = config_entry(config_path, model_id)
    backend = entry.get("backend")
    gpu_libs = _mapped_gpu_libs(pid)
    if not is_litertlm_serve(argv):
        device = "foreign"
    elif backend == "cpu" and not gpu_libs:
        device = "cpu"
    elif backend in ("gpu", "npu") and gpu_libs:
        device = backend
    else:
        device = "mixed"
    return {
        "serving": True,
        "port": port,
        "pid": pid,
        "device": device,
        "litertlm_serve": is_litertlm_serve(argv),
        "config": config_path,
        "backend": backend,
        "cpu_thread_count": entry.get("cpu_thread_count"),
        "max_num_tokens": entry.get("max_num_tokens"),
        "gpu_libs": gpu_libs,
        "argv": argv,
    }


def describe(att: dict) -> str:
    if not att.get("serving"):
        return f"nothing is listening on port {att.get('port')}"
    bits = [f"device={att['device']}", f"pid={att['pid']}", f"backend={att.get('backend')}"]
    if att.get("gpu_libs"):
        bits.append("mapped: " + ", ".join(att["gpu_libs"]))
    if att.get("config"):
        bits.append(f"config={att['config']}")
    return " · ".join(bits)


def mismatch(att: dict, expected: str = EXPECTED_DEVICE) -> Optional[str]:
    """Why `att` is not the arm `expected`, or None if it is."""
    if not att.get("serving"):
        return f"nothing is serving on port {att.get('port')}"
    if att["device"] == "foreign":
        return f"the process on port {att['port']} is not `litert-lm serve` ({' '.join(att['argv'][:4])})"
    if att["device"] != expected:
        return f"the process on port {att['port']} is a **{att['device']}** server, not {expected} ({describe(att)})"
    return None
