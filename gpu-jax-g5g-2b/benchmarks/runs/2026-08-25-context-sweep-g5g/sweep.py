#!/usr/bin/env python3
"""Context x output-length sweep against a live gpu-jax-g5g-2b endpoint.

Concurrency is fixed at 1 and that is a hardware fact, not a shortcut:
MAX_NUM_SEQS=1, Gemma4EModelJAX raises NotImplementedError for B>1, and the
decode step donates its KV buffers -- concurrent requests through one engine
would be a correctness hazard, not a throughput measurement.

Two things this harness does that the rig's earlier manual runs did not:

  * WARM UP AT THE SHAPE IT MEASURES. max_new_tokens is a static_argnames entry,
    so (bucket, max_tokens) is the compiled shape. Warming at a different
    max_tokens than you measure leaves the measured request cold -- previously
    measured here as a 4x error (3.4 vs 13.5 tok/s).
  * RECORD infeasible CELLS rather than dropping them. schema 1.1 has
    status ok|infeasible|failed precisely so a cell that cannot exist on the
    hardware is data instead of a gap.

Results are appended to a JSONL after every single request, so a spot
reclamation loses at most one request.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Roughly one token per repetition for this tokenizer; the ACTUAL prompt length
# is read back from usage.prompt_tokens and that is what the report records.
FILLER = "token "


def post(base: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as res:
        body = json.load(res)
        hdrs = dict(res.headers)
    return body, time.perf_counter() - t0, hdrs


def get_metrics(base: str, timeout: float = 30.0) -> dict:
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=timeout) as res:
            text = res.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, raw = line.rpartition(" ")
        try:
            out[series.partition("{")[0]] = float(raw)
        except ValueError:
            pass
    return out


def one_cell(base: str, in_len: int, out_len: int, repeats: int,
             timeout: float, sink) -> dict:
    """Warm the exact shape, then measure it `repeats` times."""
    prompt = FILLER * in_len
    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": out_len, "temperature": 0.0}

    def record(kind, body, wall, hdrs=None, error=None):
        usage = (body or {}).get("usage") or {}
        row = {
            "kind": kind, "requested_in": in_len, "output_len": out_len,
            "wall_s": round(wall, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prefill_ms": usage.get("prefill_ms"),
            "decode_tok_s": usage.get("decode_tokens_per_second"),
            "bucket_size": usage.get("bucket_size"),
            "pad_tokens": usage.get("pad_tokens"),
            "cold_shape": usage.get("cold_shape"),
            "build_id": usage.get("build_id"),
            "finish": ((body or {}).get("choices") or [{}])[0].get("finish_reason"),
            "request_id": (hdrs or {}).get("X-Request-Id"),
            "error": error,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        return row

    # --- warm-up at this exact shape ---
    try:
        body, wall, hdrs = post(base, payload, timeout)
        warm = record("warmup", body, wall, hdrs)
        print(f"    warmup  {wall:6.2f}s  in={warm['prompt_tokens']} "
              f"bucket={warm['bucket_size']} pad={warm['pad_tokens']} "
              f"cold={warm['cold_shape']}", flush=True)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:300]
        record("warmup", None, 0.0, error=detail)
        status = "infeasible" if _is_oom(detail) else "failed"
        print(f"    warmup FAILED -> {status}: {detail[:140]}", flush=True)
        return {"status": status, "error": detail[:400], "runs": []}
    except Exception as exc:
        record("warmup", None, 0.0, error=repr(exc))
        print(f"    warmup FAILED -> failed: {exc!r}", flush=True)
        return {"status": "failed", "error": repr(exc), "runs": []}

    # --- measured repeats ---
    runs = []
    for i in range(repeats):
        try:
            body, wall, hdrs = post(base, payload, timeout)
            row = record("measure", body, wall, hdrs)
            runs.append(row)
            print(f"    run {i+1}    {wall:6.2f}s  "
                  f"decode={row['decode_tok_s']} tok/s  "
                  f"prefill={row['prefill_ms']} ms  cold={row['cold_shape']}",
                  flush=True)
        except Exception as exc:
            detail = getattr(exc, "read", lambda: b"")().decode()[:300] or repr(exc)
            record("measure", None, 0.0, error=detail)
            print(f"    run {i+1} FAILED: {detail[:140]}", flush=True)
            return {"status": "infeasible" if _is_oom(detail) else "failed",
                    "error": detail[:400], "runs": runs}
    return {"status": "ok", "runs": runs}


def _is_oom(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ("out of memory", "resource_exhausted", "oom",
                                "failed to allocate", "alloc"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="http://IP:8000")
    p.add_argument("--out", required=True, help="results JSON")
    p.add_argument("--jsonl", required=True, help="append-per-request log")
    p.add_argument("--contexts", default="32,128,512,1024,2048,4096")
    p.add_argument("--outputs", default="32,128")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=600.0)
    a = p.parse_args()

    contexts = [int(x) for x in a.contexts.split(",")]
    outputs = [int(x) for x in a.outputs.split(",")]
    cells = []

    with open(a.jsonl, "a") as sink:
        for out_len in outputs:
            for in_len in contexts:
                print(f"\n[cell] input~{in_len} output={out_len}", flush=True)
                before = get_metrics(a.base)
                res = one_cell(a.base, in_len, out_len, a.repeats, a.timeout, sink)
                after = get_metrics(a.base)
                res.update({"requested_in": in_len, "output_len": out_len,
                            "degenerate_delta":
                                after.get("tpu_jax_degenerate_responses_total", 0.0)
                                - before.get("tpu_jax_degenerate_responses_total", 0.0)})
                cells.append(res)
                with open(a.out, "w") as fh:
                    json.dump(cells, fh, indent=2)

    ok = sum(1 for c in cells if c["status"] == "ok")
    print(f"\n=== {ok}/{len(cells)} cells ok ===")
    for c in cells:
        if c["status"] != "ok":
            print(f"  {c['status']:<11} in~{c['requested_in']} out={c['output_len']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
