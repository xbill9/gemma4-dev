"""Shared client for AMD Token Lab: config, key, and requests that retry. Standard library only.

Config comes from this rig's tpu.env, the key from .env (TOKENLAB_API_KEY); a real
environment variable wins over both, as in every rig here.

Token Lab answers 429 when the model is at its shared concurrency limit or the key is
over its rate limit, and 502/503/504 when a backend blinks; both carry Retry-After
when they can. Those are retried here, before any byte of the reply has been read.
A stream that breaks after it started cannot be retried by status code; the caller
sees a missing finish_reason / [DONE] and says so.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
RETRY_STATUS = {429, 502, 503, 504}
USER_AGENT = "gemma4-dev-tokenlab/1.0"


def read_env(path: str) -> dict:
    """KEY=VALUE lines; comments and blanks skipped, surrounding quotes stripped."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


def load_config(here: str = HERE) -> dict:
    cfg = read_env(os.path.join(here, "tpu.env"))
    cfg.update(read_env(os.path.join(here, ".env")))
    for k in list(cfg) + ["TOKENLAB_API_KEY", "TOKENLAB_BASE_URL", "MODEL_NAME"]:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    cfg.setdefault("TOKENLAB_BASE_URL", "https://api.tokenlab.amd.com/v1")
    cfg["TOKENLAB_BASE_URL"] = cfg["TOKENLAB_BASE_URL"].rstrip("/")
    cfg.setdefault("MODEL_NAME", "Gemma-4-31B-IT")
    return cfg


def retry_after(headers, default: float) -> float:
    try:
        return max(0.0, float(headers.get("Retry-After")))
    except (TypeError, ValueError):
        return default


def open_api(
    cfg: dict,
    path: str,
    body: Optional[bytes] = None,
    timeout: float = 600,
    on_retry=None,
):
    """urlopen against Token Lab with the key attached, retrying 429/5xx before the body is read.

    `path` is relative to the base URL and starts with '/', e.g. '/chat/completions'.
    Returns an open response (2xx) or the final HTTPError (which is file-like too).
    Raises OSError only when the host cannot be reached at all.
    """
    key = cfg.get("TOKENLAB_API_KEY")
    if not key:
        raise RuntimeError("TOKENLAB_API_KEY is not set (rig .env or environment)")
    tries = int(cfg.get("RETRY_MAX", 30))
    wait = float(cfg.get("RETRY_WAIT_S", 2))
    headers = {"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}
    if body is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(1, tries + 1):
        req = urllib.request.Request(
            cfg["TOKENLAB_BASE_URL"] + path,
            data=body,
            method="POST" if body is not None else "GET",
            headers=headers,
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == tries:
                return e
            delay = retry_after(e.headers, wait)
            if on_retry:
                on_retry(attempt, e.code, delay)
            e.close()
            time.sleep(delay)
    raise AssertionError("unreachable")


def error_message(resp) -> str:
    """The useful half of an error body: Token Lab puts it in error.message."""
    raw = resp.read().decode("utf-8", "replace")
    try:
        return json.loads(raw)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return raw[:300]


def model_entry(cfg: dict) -> Optional[dict]:
    """This rig's model from GET /models, or None."""
    resp = open_api(cfg, "/models", timeout=15)
    if resp.status != 200:
        return None
    with resp:
        data = json.loads(resp.read())
    for m in data.get("data", []):
        if m.get("id") == cfg["MODEL_NAME"]:
            return m
    return None
