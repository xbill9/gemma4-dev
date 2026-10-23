"""OpenAI-compatible chat server over e2b_engine.Engine: one request at a time, greedy, tool calls parsed.

    python3 openai_server.py [--port 8100] [--max-ctx 131072]
    curl localhost:8100/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (non-streaming). `temperature` is accepted and
ignored: decoding is always greedy. `tools` go through Gemma 4's chat template, and `<|tool_call>` blocks in the
reply come back as OpenAI `tool_calls`. `chat_template_kwargs` is passed to the template (thinking stays off
unless `{"enable_thinking": true}`).
"""

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from e2b_engine import TEXT, Engine
from gemma4_tools import parse_tool_calls

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8100)
ap.add_argument("--max-ctx", type=int, default=131072)
args = ap.parse_args()

engine = Engine(max_ctx=args.max_ctx)
lock = threading.Lock()
MODEL_ID = TEXT


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
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else r["finish"]}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out},
        "timings": {
            "prompt_ms": round(r["prompt_s"] * 1e3, 1),
            "decode_ms": round(r["decode_s"] * 1e3, 1),
            "decode_tok_per_s": round(max(len(r["tokens"]) - 1, 0) / r["decode_s"], 2) if r["decode_s"] else None,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
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
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *a):
        pass


print(
    json.dumps(
        {
            "serving": f"http://{args.host}:{args.port}",
            "model": MODEL_ID,
            "max_ctx": args.max_ctx,
            "weights_mib": engine.weights_mib,
            "kv_mib": engine.kv_mib,
        }
    ),
    flush=True,
)
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
