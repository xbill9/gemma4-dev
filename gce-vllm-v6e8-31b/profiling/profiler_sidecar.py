"""In-process xprof trace control for the vLLM TPU engine.

WHY THIS FILE EXISTS
--------------------
vLLM used to expose `/start_profile` and `/stop_profile` when `VLLM_TORCH_PROFILER_DIR`
was set. **That variable does not exist in `vllm/vllm-tpu:nightly`.** Measured on hardware
2026-08-25: setting it logs

    WARNING [envs.py:2208] Unknown vLLM environment variable detected: VLLM_TORCH_PROFILER_DIR

and `POST /start_profile` returns **404** with no profile route in the OpenAPI document. The
only profiler variables the build knows are `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`,
`VLLM_CUSTOM_SCOPES_FOR_PROFILING`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`,
`VLLM_NVTX_SCOPES_FOR_PROFILING` and `VLLM_TRACE_FUNCTION` — none of which start a trace.

So the trigger has to live *inside* the process that owns the TPU. `jax.profiler.start_trace`
can only be called from there, and `tpu_inference` is importable in that process, so JAX is
already loaded. This module re-creates the endpoints vLLM dropped.

HOW IT IS LOADED
----------------
Python imports `sitecustomize` automatically from any `sys.path` entry at interpreter start.
`capture_profile.sh` (or the boot-time `XPROF_GCS_URI` path) bind-mounts this directory into the container and sets `PYTHONPATH`, so a
`sitecustomize.py` next to this file imports it in every Python process the container starts.

That "every process" is the whole difficulty, and the guards below are the substance of this
file rather than defensive noise:

- **vLLM runs the engine as a SEPARATE PROCESS from the API server** (`APIServer pid=1`,
  `EngineCore pid=1615` in the boot log). Only the engine owns the TPU.
- **Deciding which process that is must never touch the device.** Asking JAX directly claims
  the chip; see `_owns_tpu`. This is not hypothetical — it broke a boot on 2026-08-26.
- **Any exception here breaks the container's startup**, because sitecustomize failures
  propagate out of interpreter init. Nothing in this module may raise.
- It stays inert unless `VLLM_XPROF_DIR` is set, so a container that mounts it but does not
  want profiling is unaffected.

The trace is written in TensorBoard's standard layout
(`<logdir>/plugins/profile/<run>/*.xplane.pb`), so both `xprof --logdir` and
`tensorboard --logdir` read it with no conversion.
"""

from __future__ import annotations

import json
import os
import threading

_PORT = int(os.environ.get("VLLM_XPROF_PORT", "9012"))
_LOGDIR = os.environ.get("VLLM_XPROF_DIR", "")
_state = {"tracing": False}


def _owns_tpu() -> bool:
    """True only in the process that has the TPU device files OPEN.

    Two weaker checks were tried on hardware and both were wrong:

    1. `jax.devices()` — not a query. It initialises the backend and CLAIMS THE CHIP. Run from
       sitecustomize in vLLM's API server (pid 1) it took the TPU and EngineCore died with
       "The TPU is already in use by process with pid 1". Cost a boot, 2026-08-26.
    2. `libtpu` in /proc/self/maps — passive, but not selective. The API server imports
       tpu_platform, so libtpu is mapped there too; pid 1 won the race for the control port
       and EngineCore got "Address already in use". Cost a second boot the same night.

    An OPEN FD on the accelerator device is the thing only the owning process has. Reading
    /proc/self/fd initialises nothing and touches no device.
    """
    try:
        fds = os.path.join("/proc", "self", "fd")
        for name in os.listdir(fds):
            try:
                target = os.readlink(os.path.join(fds, name))
            except OSError:
                continue
            if "accel" in target or "vfio" in target:
                return True
        return False
    except Exception:
        return False


def _wait_then_serve():
    """Wait for THIS process to own the TPU, then bind the control port.

    At sitecustomize time no process has initialised the backend yet, so the check above is
    false everywhere. Polling passively lets the right process — and only the right one —
    claim the port once vLLM has brought the engine up on its own terms.
    """
    import time

    deadline = time.time() + float(os.environ.get("VLLM_XPROF_WAIT_S", "1800"))
    while time.time() < deadline:
        if _owns_tpu():
            try:
                from http.server import HTTPServer

                server = HTTPServer(("0.0.0.0", _PORT), _handler_class())
                print(f"[xprof-sidecar] trace control on :{_PORT} -> {_LOGDIR}", flush=True)
                server.serve_forever()
            except Exception as exc:
                # Another process won the race, or the port is taken. Not fatal.
                print(f"[xprof-sidecar] not serving: {type(exc).__name__}: {exc}", flush=True)
            return
        time.sleep(5)


def _handler_class():
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        # Default logging writes every request to stderr, which lands in the middle of the
        # vLLM container log and makes the engine's own output hard to read.
        def log_message(self, *_args):
            pass

        def _reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
            import jax

            try:
                if self.path.rstrip("/") == "/start":
                    if _state["tracing"]:
                        return self._reply(409, {"error": "already tracing"})
                    os.makedirs(_LOGDIR, exist_ok=True)
                    jax.profiler.start_trace(_LOGDIR)
                    _state["tracing"] = True
                    return self._reply(200, {"status": "tracing", "logdir": _LOGDIR})
                if self.path.rstrip("/") == "/stop":
                    if not _state["tracing"]:
                        return self._reply(409, {"error": "not tracing"})
                    jax.profiler.stop_trace()
                    _state["tracing"] = False
                    return self._reply(200, {"status": "stopped", "logdir": _LOGDIR})
                return self._reply(404, {"error": "use POST /start or POST /stop"})
            except Exception as exc:  # never take the engine down over a profiler call
                _state["tracing"] = False
                return self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):  # noqa: N802
            self._reply(200, {"tracing": _state["tracing"], "logdir": _LOGDIR, "port": _PORT})

    return Handler


def install() -> bool:
    """Arm the sidecar. Never raises, and never touches the TPU.

    Returns whether the WAITER started, not whether the control port is up — binding happens
    later, in whichever process turns out to own the chip.
    """
    if not _LOGDIR:
        return False
    try:
        threading.Thread(target=_wait_then_serve, daemon=True).start()
        return True
    except Exception as exc:
        print(f"[xprof-sidecar] not installed: {type(exc).__name__}: {exc}", flush=True)
        return False
