#!/usr/bin/env python3
"""Concurrent `lambda invoke` against the published version: output tokens per second at 1..N requests.

Each request is one boto3 Lambda Invoke (synchronous), so the timings include the Lambda invoke path
from this machine to us-east-1. A unique leading nonce per request keeps vLLM's prefix cache from hitting; greedy,
thinking off, fixed max_tokens with ignore_eos so every reply is the same length.

  bench.py OUT_DIR [--levels 1,4,16,48] [--max-tokens 256]
"""

import argparse
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config

import lmi

CLIENT = boto3.client("lambda", region_name=lmi.E["AWS_REGION"],
                      config=Config(max_pool_connections=64, read_timeout=900, retries={"max_attempts": 0}))

PROMPT = "Write a detailed explanation of how a refrigerator moves heat out of its interior."


def one(out_dir, max_tokens, i):
    body = {
        "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] {PROMPT}"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    try:
        r = CLIENT.invoke(FunctionName=f"{lmi.E['FUNCTION_NAME']}:$LATEST.PUBLISHED", Payload=json.dumps(body))
        reply = json.loads(r["Payload"].read())
        toks, err = reply["usage"]["completion_tokens"], r.get("FunctionError")
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        toks, err = 0, repr(e)[:400]
    dt = time.time() - t0
    return {"i": i, "seconds": round(dt, 3), "completion_tokens": toks, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--levels", default="1,4,16,48")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in [int(x) for x in a.levels.split(",")]:
        t0 = time.time()
        with ThreadPoolExecutor(n) as ex:
            res = list(ex.map(lambda i: one(out, a.max_tokens, i), range(n)))
        wall = time.time() - t0
        ok = [r for r in res if not r["error"]]
        toks = sum(r["completion_tokens"] for r in ok)
        row = {
            "concurrency": n,
            "ok": len(ok),
            "errors": n - len(ok),
            "wall_s": round(wall, 3),
            "output_tokens": toks,
            "agg_tok_s": round(toks / wall, 1),
            "median_request_s": round(statistics.median(r["seconds"] for r in ok), 3) if ok else None,
            "requests": res,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
    (out / "bench.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
