"""Targeted throughput cells for the bf16-vs-fp8 KV cache comparison.

Runs on the TPU VM host and drives `vllm bench serve` inside the vllm-gemma4 container, so
the load generator sits next to the server and network latency does not pollute TTFT. Run it
once per arm with a different --arm/--output.

Why fp8 can help at all on v5e. The v5e MXU has no native fp8 path (Google publishes bf16 and
int8 peaks only; Ironwood/v7 is the first TPU with fp8 in the MXU). tpu_inference quantizes on
the write path and the kernel widens back to bf16 before the matmul, so fp8 KV changes *bytes
stored and bytes moved*, never FLOPS. Both effects below are bandwidth/capacity effects.

Cell design. bf16 KV capacity is 321,376 tokens (measured); fp8 is predicted to be exactly
642,752. Each cell needs concurrency * (input_len + output_len) tokens of KV. Per-token KV is
15 KiB in bf16 (15 layers x 1 KV head x 256 dim x K+V x 2B), against ~6.7 GiB of weight traffic
per decode step implied by the measured 120 tok/s single-stream. The ratio of those two is what
sorts the cells into three roles, and each role has a *different* prediction:

  - CONTROL: KV is a negligible share of bytes moved (<5%). fp8 should show no gain, and may
    lose a little to convert-on-read. If a control improves, that is warm-up or noise, not fp8.
  - BANDWIDTH: fits in bf16 capacity, but KV is ~35-40% of bytes moved per step. Halving it
    predicts a real ~15-18% gain with no capacity effect whatsoever. These were labelled
    "control" in the first draft, which would have made a genuine bandwidth win read as
    contamination.
  - KEY: exceeds bf16 capacity but fits inside fp8. Capacity *and* bandwidth both pay off here,
    so this is where the largest gain should land.
  - SATURATED: exceeds both. Neither arm can hold the working set; included so a flat result
    there confirms the KEY cells are not just measuring "bigger is better".

A monotone result across control -> bandwidth -> key is much harder to explain away as warm-up
than a single binary control/key split would be.

Note on 16000: max_model_len is 16384 and OUTPUT_LEN is 128, so an input_len of 16384 makes a
request of 16512 and is infeasible-by-config — the same arithmetic that made the 32768 row
infeasible in the 2026-08-06 sweep. 16000 + 128 fits, and at c=32 still needs 516,096 KV tokens:
above bf16's 321,376, below fp8's predicted 642,752, so it remains a valid KEY cell.

Every subprocess call passes list args — never shell=True.
"""

import argparse
import json
import re
import subprocess
from typing import Any, Dict, List, Optional

MODEL = "google/gemma-4-E2B-it"
OUTPUT_LEN = 128

BF16_KV_TOKENS = 321_376
FP8_KV_TOKENS_PREDICTED = 642_752

# (input_len, concurrency, role)
CELLS = [
    (128, 1, "control"),
    (128, 8, "control"),
    (1024, 16, "control"),
    (4096, 64, "bandwidth"),
    (8192, 32, "bandwidth"),
    (8192, 64, "key"),
    (16000, 32, "key"),
    (16000, 64, "saturated"),
]

# Headline metrics from `vllm bench serve` stdout -> our field names.
METRICS = {
    "Successful requests": "successful_requests",
    "Request throughput (req/s)": "request_rate_rps",
    "Output token throughput (tok/s)": "output_tok_per_s",
    "Total Token throughput (tok/s)": "total_tok_per_s",
    "Mean TTFT (ms)": "ttft_ms_mean",
    "Median TTFT (ms)": "ttft_ms_median",
    "P99 TTFT (ms)": "ttft_ms_p99",
    "Mean TPOT (ms)": "tpot_ms_mean",
    "Median TPOT (ms)": "tpot_ms_median",
    "P99 TPOT (ms)": "tpot_ms_p99",
}


def parse_bench_output(text: str) -> Dict[str, float]:
    """Pull the headline numbers out of `vllm bench serve` stdout."""
    found: Dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        for label, field in METRICS.items():
            if stripped.startswith(label):
                m = re.search(r"([-+]?\d+\.?\d*)\s*$", stripped)
                if m:
                    found[field] = float(m.group(1))
    return found


def run_cell(container: str, input_len: int, concurrency: int, num_prompts: int, timeout: float) -> Dict[str, Any]:
    cmd = [
        "sudo",
        "docker",
        "exec",
        container,
        "vllm",
        "bench",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--model",
        MODEL,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(OUTPUT_LEN),
        "--max-concurrency",
        str(concurrency),
        "--seed",
        "0",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": f"timeout after {timeout}s"}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        return {"status": "failed", "error": f"rc={proc.returncode}: " + " | ".join(tail)}
    metrics = parse_bench_output(proc.stdout)
    if not metrics:
        return {"status": "failed", "error": "no metrics parsed from stdout", "raw_stdout": proc.stdout[-2000:]}
    return {"status": "ok", "raw_stdout": proc.stdout[-4000:], **metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="Label for this arm, e.g. bf16 or fp8")
    ap.add_argument("--container", default="vllm-gemma4", help="Docker container to exec the load generator in")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--prompts-per-concurrency", type=int, default=3)
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    for input_len, concurrency, role in CELLS:
        needed = concurrency * (input_len + OUTPUT_LEN)
        num_prompts = max(concurrency * args.prompts_per_concurrency, 16)
        print(
            f"[{args.arm}] input_len={input_len} concurrency={concurrency} ({role}) "
            f"kv_needed={needed:,} prompts={num_prompts}",
            flush=True,
        )
        outcome = run_cell(args.container, input_len, concurrency, num_prompts, args.timeout)
        error: Optional[str] = outcome.get("error")
        if error:
            print(f"  FAILED: {error}", flush=True)
        else:
            print(
                f"  {outcome.get('output_tok_per_s')} out tok/s, median TTFT {outcome.get('ttft_ms_median')} ms",
                flush=True,
            )
        results.append(
            {
                "input_len": input_len,
                "concurrency": concurrency,
                "output_len": OUTPUT_LEN,
                "role": role,
                "num_prompts": num_prompts,
                "kv_tokens_needed": needed,
                "fits_bf16": needed <= BF16_KV_TOKENS,
                "fits_fp8_predicted": needed <= FP8_KV_TOKENS_PREDICTED,
                **outcome,
            }
        )

    with open(args.output, "w") as f:
        json.dump({"arm": args.arm, "model": MODEL, "cells": results}, f, indent=2)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nWrote {args.output} — {ok}/{len(results)} cells ok")


if __name__ == "__main__":
    main()
