"""Lambda handler: forward one invocation to the engine's OpenAI-compatible server on 127.0.0.1.

The engine is started by the extension in /opt/extensions (engine_ext.py), so the handler is only a
proxy. Same contract as the handler in AWS's vLLM Lambda image: the payload is the OpenAI request body,
sent as-is; `messages` goes to /v1/chat/completions, anything else to /v1/completions. Standard
library only, so it runs on any of AWS's Lambda base images.
"""

import json
import os
import urllib.error
import urllib.request

BASE = f"http://127.0.0.1:{os.environ.get('ENGINE_PORT', '8000')}"
MODEL = os.environ.get("SERVED_MODEL_NAME", "default")


def handler(event, context):
    if isinstance(event, (bytes, bytearray, str)):
        event = json.loads(event or "{}")
    body = dict(event)
    body.setdefault("model", MODEL)
    body.pop("stream", None)
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    left = context.get_remaining_time_in_millis() / 1000 - 1 if context else 600
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(0.1, left)) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"engine returned {e.code}: {e.read().decode(errors='replace')[:2000]}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"engine at {BASE} did not answer: {e}") from e
