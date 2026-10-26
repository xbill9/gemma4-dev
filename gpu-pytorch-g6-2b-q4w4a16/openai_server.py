"""OpenAI-compatible chat server over e2b_engine.Engine: one request at a time, greedy, tool calls parsed.

    python3 openai_server.py [--host 0.0.0.0] [--port 8000] [--max-ctx 131072] [--prefill-dtype auto]
    curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'

Endpoints: GET /health, GET /metrics, GET /v1/models, POST /v1/chat/completions (non-streaming). `temperature` is
accepted and ignored: decoding is always greedy. `tools` go through Gemma 4's chat template, and `<|tool_call>` blocks
in the reply come back as OpenAI `tool_calls`. `chat_template_kwargs` is passed to the template (thinking stays off
unless `{"enable_thinking": true}`).

Forked from local-pytorch-1650ti-2b-q4w4a16 for the L4 on EC2 G6. Three additions, all for server.py's tools:
`/health` reports the deployed payload's build id, `/metrics` exposes the `tpu_jax_*` series that `get_metrics` and
`verify_model_health` parse (the prefix is an identifier shared with the sibling rigs), and `--prefill-dtype auto`
times an fp16 and an fp32 GEMM on the device and reads prompts in the faster one (the 1650 Ti has no tensor cores and
picks fp32; a GPU with them should pick fp16).
"""

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from e2b_engine import TEXT, Engine
from gemma4_tools import parse_tool_calls

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--max-ctx", type=int, default=131072)
ap.add_argument("--prefill-dtype", choices=("auto", "float16", "float32"), default="auto")
args = ap.parse_args()

MODEL_ID = TEXT
RIG = os.getenv("RIG_NAME", "gpu-pytorch-g6-2b-q4w4a16")


def _read_build_id():
    """The payload digest deploy_torch_server stamps next to this file; "unknown" when run from a checkout."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PAYLOAD_SHA")) as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


def _gemm_tflops(dtype, m=256, k=1536, n=24576, reps=10):
    x, w = torch.randn(m, k, device="cuda", dtype=dtype), torch.randn(n, k, device="cuda", dtype=dtype)
    torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    return 2 * m * k * n * reps / (time.perf_counter() - t) / 1e12


def _prefill_dtype(choice):
    if choice != "auto":
        return getattr(torch, choice), {}
    rates = {dt: _gemm_tflops(dt) for dt in (torch.float16, torch.float32)}
    torch.cuda.empty_cache()
    return max(rates, key=rates.get), {str(k).replace("torch.", ""): round(v, 2) for k, v in rates.items()}


def _is_degenerate(ids):
    """True when the last 12 tokens are a cycle of period 1-4 repeated at least 4 times (same rule as the siblings)."""
    if len(ids) < 12:
        return False
    tail = ids[-12:]
    for period in (1, 2, 3, 4):
        if all(tail[i] == tail[i % period] for i in range(period * (len(tail) // period))):
            return True
    return False


t0 = time.perf_counter()
PREFILL_DTYPE, GEMM_RATES = _prefill_dtype(args.prefill_dtype)
engine = Engine(max_ctx=args.max_ctx, prefill_dtype=PREFILL_DTYPE)
LOAD_S = time.perf_counter() - t0
lock = threading.Lock()


class Metrics:
    """Prometheus exposition with the series names server.py parses. Only `model` labels the numeric series:
    _parse_prom folds every other label into the sample key, which would hide the series from get_metrics."""

    def __init__(self):
        self.build_id = _read_build_id()
        self.requests = {"success": 0.0, "failed": 0.0}
        self.prompt_tokens = self.completion_tokens = 0.0
        self.prefill_seconds = self.decode_seconds = 0.0
        self.degenerate = 0.0
        self.last_decode_tps = 0.0

    def record(self, n_prompt, n_out, r, degenerate):
        self.requests["success"] += 1
        self.prompt_tokens += n_prompt
        self.completion_tokens += n_out
        self.prefill_seconds += r["prompt_s"]
        self.decode_seconds += r["decode_s"]
        self.degenerate += degenerate
        steps = max(len(r["tokens"]) - 1, 0)
        self.last_decode_tps = steps / r["decode_s"] if r["decode_s"] > 0 else 0.0

    def render(self):
        m = MODEL_ID
        lines = [
            "# HELP tpu_jax_precision_info Dtypes and quantisation resolved on device",
            "# TYPE tpu_jax_precision_info gauge",
            f'tpu_jax_precision_info{{model="{m}",rig="{RIG}",build_id="{self.build_id}",compute_dtype="float16",'
            f'prefill_dtype="{str(PREFILL_DTYPE).replace("torch.", "")}",quant_mode="q4w4a16",'
            f'kv_cache_dtype="float16",ple_bits="4",int8_lm_head="false",max_ctx="{args.max_ctx}"}} 1',
            "# HELP tpu_jax_requests_total Requests by terminal status",
            "# TYPE tpu_jax_requests_total counter",
        ]
        for status, count in self.requests.items():
            lines.append(f'tpu_jax_requests_total{{model="{m}",status="{status}"}} {count}')
        for name, value, kind, help_text in (
            ("tpu_jax_prompt_tokens_total", self.prompt_tokens, "counter", "Prompt tokens"),
            ("tpu_jax_completion_tokens_total", self.completion_tokens, "counter", "Completion tokens"),
            (
                "tpu_jax_prefill_seconds_total",
                self.prefill_seconds,
                "counter",
                "Seconds spent reading prompts",
            ),
            ("tpu_jax_decode_seconds_total", self.decode_seconds, "counter", "Seconds spent in decode alone"),
            (
                "tpu_jax_latency_seconds_sum",
                self.prefill_seconds + self.decode_seconds,
                "counter",
                "Prompt read plus decode seconds",
            ),
            (
                "tpu_jax_cold_requests_total",
                0.0,
                "counter",
                "Always 0: the decode step is captured once at start-up",
            ),
            (
                "tpu_jax_degenerate_responses_total",
                self.degenerate,
                "counter",
                "Replies whose tail was a repeating token cycle",
            ),
            (
                "tpu_jax_decode_tokens_per_second",
                self.last_decode_tps,
                "gauge",
                "Decode throughput of the LAST request, excluding the prompt read and HTTP",
            ),
            (
                "tpu_jax_weight_bytes",
                float(engine.weights_mib * 2**20),
                "gauge",
                "Resident weight bytes on the GPU",
            ),
            ("tpu_jax_kv_cache_bytes", float(engine.kv_mib * 2**20), "gauge", "K/V cache bytes on the GPU"),
            ("tpu_jax_model_load_seconds", LOAD_S, "gauge", "Load plus decode-step capture"),
        ):
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", f'{name}{{model="{m}"}} {value}']
        return "\n".join(lines) + "\n"


metrics = Metrics()


def complete(req):
    ids = engine.render(req["messages"], tools=req.get("tools"), **req.get("chat_template_kwargs", {}))
    with lock:
        r = engine.generate(ids, max_new=int(req.get("max_tokens") or 1024))
    raw = engine.tok.decode(r["tokens"], skip_special_tokens=False)
    content, calls = parse_tool_calls(raw)
    if not calls:
        content = engine.tok.decode(r["tokens"], skip_special_tokens=True)
    message = {"role": "assistant", "content": content.strip() or None}
    if calls:
        message["tool_calls"] = calls
    n_prompt, n_out = ids.shape[1], len(r["tokens"]) + (1 if r["finish"] == "stop" else 0)
    metrics.record(n_prompt, n_out, r, _is_degenerate(r["tokens"]))
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if calls else r["finish"]}
        ],
        "usage": {
            "prompt_tokens": n_prompt,
            "completion_tokens": n_out,
            "total_tokens": n_prompt + n_out,
            # read by sweep.py, as on the bf16 sibling's server
            "decode_tokens_per_second": round(metrics.last_decode_tps, 3),
            "cold_shape": False,
        },
        "timings": {
            "prompt_ms": round(r["prompt_s"] * 1e3, 1),
            "decode_ms": round(r["decode_s"] * 1e3, 1),
            "decode_tok_per_s": round(max(len(r["tokens"]) - 1, 0) / r["decode_s"], 2)
            if r["decode_s"]
            else None,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj, ctype="application/json"):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(
                200,
                {"status": "ok", "model": MODEL_ID, "build_id": metrics.build_id, "max_ctx": args.max_ctx},
            )
        elif self.path == "/metrics":
            self._send(200, metrics.render(), "text/plain; version=0.0.4")
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if req.get("stream"):
                self._send(400, {"error": "streaming is not supported"})
                return
            self._send(200, complete(req))
        except Exception as e:  # report to the client; keep serving
            metrics.requests["failed"] += 1
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *a):
        pass


print(
    json.dumps(
        {
            "serving": f"http://{args.host}:{args.port}",
            "model": MODEL_ID,
            "build_id": metrics.build_id,
            "device": torch.cuda.get_device_name(0),
            "max_ctx": args.max_ctx,
            "prefill_dtype": str(PREFILL_DTYPE).replace("torch.", ""),
            "gemm_tflops": GEMM_RATES,
            "weights_mib": engine.weights_mib,
            "kv_mib": engine.kv_mib,
            "load_s": round(LOAD_S, 1),
        }
    ),
    flush=True,
)
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
