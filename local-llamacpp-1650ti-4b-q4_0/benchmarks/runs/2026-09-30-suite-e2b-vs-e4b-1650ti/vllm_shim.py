"""Serve vLLM's /v1/completions logprob shape on top of llama-server's native /completion.

    python3 vllm_shim.py [--listen 8090] [--upstream http://127.0.0.1:8080]

The Nimble suite harness (jev-tpu-v5e1/nimble_suite/run_suite.py, JEV_TOPK set) posts
token-id prompts to /v1/completions with max_tokens 1 and logprobs K, and reads
choices[0].logprobs.top_logprobs[0] as {"token_id:N": logprob}. llama-server's
/completion takes the same token-id prompt and returns the top K as
completion_probabilities[0].top_logprobs[{id, logprob}] when n_probs is set. This
translates one to the other and nothing else: no prompt text is touched.
"""
import argparse
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=8090)
ap.add_argument("--upstream", default="http://127.0.0.1:8080")
ARGS = ap.parse_args()


def translate(body):
    assert isinstance(body["prompt"], list), "token-id prompts only"
    up = {
        "prompt": body["prompt"],
        "n_predict": body.get("max_tokens", 1),
        "temperature": body.get("temperature", 0.0),
        "n_probs": int(body.get("logprobs") or 20),
        "post_sampling_probs": False,  # raw softmax logprobs, not the sampler's
        "cache_prompt": True,
    }
    req = urllib.request.Request(ARGS.upstream.rstrip("/") + "/completion", data=json.dumps(up).encode(),
                                 headers={"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    first = d["completion_probabilities"][0]
    top = {f"token_id:{t['id']}": t["logprob"] for t in first["top_logprobs"]}
    return {"choices": [{"text": d.get("content", ""), "logprobs": {"top_logprobs": [top]}}],
            "usage": {"prompt_tokens": d.get("tokens_evaluated"), "cached": d.get("tokens_cached")}}


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        try:
            out, code = translate(body), 200
        except Exception as e:  # surface upstream errors to the harness rather than hang
            out, code = {"error": repr(e)}, 500
        b = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", ARGS.listen), H).serve_forever()
