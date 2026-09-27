"""A stand-in vLLM endpoint for exercising run_eval.py and score.py offline.

    python3 tests/fake_vllm.py --port 18000

Answers /v1/chat/completions (the diffusion read) and /v1/completions (the
autoregressive read) in vLLM's logprobs shapes, with every requested
`logprob_token_ids` present and a random distribution over them. It checks
the fields each arm must send and fails the request if one is missing.
"""

import argparse
import json
import math
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def fake_top(label_ids, rng):
    w = [rng.random() + 0.01 for _ in label_ids] + [rng.random() * 0.2]
    z = sum(w)
    top = {f"token_id:{i}": math.log(x / z) for i, x in zip(label_ids, w)}
    top["token_id:1"] = math.log(w[-1] / z)  # some mass off the labels
    return top


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        rng = random.Random(json.dumps(body, sort_keys=True))
        ids = body["logprob_token_ids"]
        assert body.get("return_tokens_as_token_ids") is True
        if self.path == "/v1/chat/completions":
            x = body["vllm_xargs"]
            assert x["diffusion_read_only"] is True and x["diffusion_max_steps"] == 1
            assert len(x["diffusion_seed_canvas"]) == x["diffusion_canvas_length"]
            content = [
                {"token": "token_id:0", "logprob": 0.0, "top_logprobs": [
                    {"token": k, "logprob": v} for k, v in fake_top(ids, rng).items()
                ]}
                for _ in range(body["max_tokens"])
            ]
            out = {"choices": [{"logprobs": {"content": content}}], "usage": {}}
        elif self.path == "/v1/completions":
            assert body["max_tokens"] == 1 and isinstance(body["prompt"], list)
            out = {"choices": [{"logprobs": {"top_logprobs": [fake_top(ids, rng)]}}], "usage": {}}
        else:
            self.send_error(404)
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18000)
    ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
