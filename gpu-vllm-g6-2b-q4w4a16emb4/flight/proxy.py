"""Rewrite gen_eval.py's requests for `litert-lm serve`, which (0.19.0) ignores
`max_tokens` and answers only to its imported model id.

    python3 proxy.py LISTEN_PORT UPSTREAM_URL MODEL_ID

Every POST body gets `model` = MODEL_ID and `max_tokens` moved to
`max_completion_tokens`; everything else passes through unchanged.
"""

import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT, UPSTREAM, MODEL = int(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]


def rewrite(body: dict) -> dict:
    body = dict(body)
    body["model"] = MODEL
    if "max_tokens" in body:
        body["max_completion_tokens"] = body.pop("max_tokens")
    return body


class Handler(BaseHTTPRequestHandler):
    def _relay(self, data=None):
        req = urllib.request.Request(
            UPSTREAM + self.path, data=data, method=self.command, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                status, payload = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, payload = e.code, e.read()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._relay()

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._relay(json.dumps(rewrite(json.loads(raw))).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
