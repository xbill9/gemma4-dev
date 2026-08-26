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
`xprof_capture.sh` bind-mounts this directory into the container and sets `PYTHONPATH`, so a
`sitecustomize.py` next to this file imports it in every Python process the container starts.

That "every process" is the whole difficulty, and the guards below are the substance of this
file rather than defensive noise:

- **vLLM runs the engine as a SEPARATE PROCESS from the API server** (`APIServer pid=1`,
  `EngineCore pid=711` in the boot log). Only the engine owns the TPU. Starting a trace in the
  API server would capture nothing and bind the port the engine wants.
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


def _has_tpu() -> bool:
    """True only in the process that actually owns TPU devices.

    This is the guard that keeps the control server out of the API server process. It also
    imports jax lazily: importing jax in a process that does not need it is slow and, on the
    TPU path, can contend for the device.
    """
    try:
        import jax

        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


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
    """Start the control server if this process owns the TPU. Never raises."""
    if not _LOGDIR:
        return False
    try:
        if not _has_tpu():
            return False
        from http.server import HTTPServer

        server = HTTPServer(("0.0.0.0", _PORT), _handler_class())
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[xprof-sidecar] trace control on :{_PORT} -> {_LOGDIR}", flush=True)
        return True
    except Exception as exc:
        # A bound port or a jax import failure must not stop the engine from serving.
        print(f"[xprof-sidecar] not installed: {type(exc).__name__}: {exc}", flush=True)
        return False
