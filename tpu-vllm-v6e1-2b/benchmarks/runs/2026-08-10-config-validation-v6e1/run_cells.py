"""Throughput cells that exercise THIS rig's config: v6e-1, E2B, max_model_len 32768.

Runs on the TPU VM host and drives `vllm bench serve` inside the vllm-gemma4 container, so the
load generator sits next to the server and network latency does not pollute TTFT.

What this is for. Two changes need exercising and they interact:

  1. The chip. v6e-1 holds ~1,151,744 KV tokens against v5e-1's 321,376 — 3.6x. `@HARDWARE.md`
     is blunt that v6e is NOT the way to buy decode throughput (2.25x the price for 1.907x the
     bandwidth); it buys context. So the prediction is deliberately asymmetric: the control
     cells should move roughly with bandwidth and no more, while the cells that thrashed on
     v5e should improve a lot because they stop thrashing.
  2. The setting. MAX_MODEL_LEN went 16384 -> 32768. The two 32000-token cells below cannot
     run at all under the old value (32000 + 128 > 16384), and cannot run on v5e at any
     setting, because 514,048 KV tokens exceeds its entire pool.

Cell roles, sized against the v6e pool rather than v5e's:

  * CONTROL     — trivially fits both chips. Isolates raw bandwidth/compute. v6e should gain
                  a little; a large gain here means something other than memory changed.
  * BANDWIDTH   — fits both, but KV is a large share of bytes moved per step.
  * V6E_ONLY    — exceeds v5e's pool, fits v6e's. v5e had to evict and recompute; v6e does not.
                  This is where the generation should pay, and the v5e numbers are recorded
                  inline so the comparison is automatic rather than remembered.
  * LONG_CTX    — needs max_model_len > 16384. Impossible on the pre-retarget config AND on
                  v5e. These two cells are the whole reason for the chip and the setting.

Note on 32000: max_model_len is 32768 and OUTPUT_LEN is 128, so 32000 + 128 = 32128 fits with
room to spare. 32768 itself would not.

v5e reference numbers are from the bf16 arm of `../2026-08-07-kv-quant-v5e1/results/cells_bf16.json`,
measured on a spot v5litepod-1 in us-west4-a. They are a DIFFERENT chip, zone, and engine build —
treat them as context for the shape of the result, not as a controlled A/B.

Every subprocess call passes list args — never shell=True.
"""

import argparse
import json
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

MODEL = "google/gemma-4-E2B-it"
OUTPUT_LEN = 128
MAX_MODEL_LEN = 32_768

# Measured pools. v5e from its own boot log; v6e from the recorded 65,536-context run
# (@HARDWARE.md), which this run re-measures at 32,768 via verify_allocation.py.
V5E_KV_TOKENS = 321_376
V6E_KV_TOKENS = 1_151_744

# (input_len, concurrency, role, v5e_out_tok_per_s or None if never run there)
#
# ORDER IS DELIBERATE AND IS NOT ASCENDING COST. This rig now provisions flex-start, which
# self-terminates at --max-run-duration=4h; boot burns ~15 min of that on compile. If the run
# is guillotined, whatever ran last is lost — so the cells are ordered by *value*, not by size:
#
#   1-3  the three cheap controls (seconds each) — they validate the setup and give the
#        direct v5e comparison, so getting them out of the way costs almost nothing.
#   4-5  the two long_ctx cells — the entire reason for both the chip and the 32768 setting.
#        These run FOURTH, not last, precisely because they are what the run is for.
#   6-10 v6e_only, then bandwidth — real results, but the ones this rig can most afford to lose.
CELLS = [
    (128, 1, "control", 123.26),
    (128, 8, "control", 738.28),
    (1024, 16, "control", 896.11),
    (32000, 16, "long_ctx", None),
    (32000, 32, "long_ctx", None),
    (16000, 64, "v6e_only", 166.69),
    (16000, 32, "v6e_only", 166.76),
    (8192, 64, "v6e_only", 314.44),
    (4096, 64, "bandwidth", 585.92),
    (8192, 32, "bandwidth", 307.76),
]

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


def run_cell(
    container: str, input_len: int, concurrency: int, num_prompts: int, timeout: float, seed: int = 0
) -> Dict[str, Any]:
    """Run one `vllm bench serve` cell.

    NOTE on `seed`: vLLM runs with `enable_prefix_caching=True` by default, and the random
    dataset is deterministic in the seed. Two cells with the same seed and the same
    `input_len` therefore draw overlapping prompts, and the later one is served from the
    prefix cache — its TTFT is not comparable to the earlier one's. That happened in this
    run's main sweep: `16000x32` followed `16000x64` and took ~1.54M cache-hit tokens
    against 1.536M input tokens, i.e. essentially all of it. **Vary the seed across cells at
    the same input_len.** Default stays 0 so the archived main-sweep results remain
    reproducible exactly as they were taken.
    """
    cmd = [
        "sudo", "docker", "exec", container,
        "vllm", "bench", "serve",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--model", MODEL,
        "--dataset-name", "random",
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(input_len),
        "--random-output-len", str(OUTPUT_LEN),
        "--max-concurrency", str(concurrency),
        "--seed", str(seed),
    ]  # fmt: skip
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
    ap.add_argument("--arm", default="v6e1-32768", help="Label for this arm")
    ap.add_argument("--container", default="vllm-gemma4", help="Docker container to exec the load generator in")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--prompts-per-concurrency", type=int, default=3)
    # Flex-start self-terminates at 4h from ACTIVE, and boot spends ~15 min of it compiling.
    # Stop issuing new cells before that lands mid-request, so the JSON is written rather than
    # lost with the node. Cells dropped this way are RECORDED, never silently omitted.
    ap.add_argument(
        "--deadline-seconds",
        type=float,
        default=0.0,
        help="Wall-clock budget for the whole sweep. 0 disables. Cells that do not fit are "
        "recorded with status=skipped_deadline.",
    )
    args = ap.parse_args()

    started = time.monotonic()
    results: List[Dict[str, Any]] = []
    for input_len, concurrency, role, v5e_ref in CELLS:
        needed = concurrency * (input_len + OUTPUT_LEN)
        num_prompts = max(concurrency * args.prompts_per_concurrency, 16)

        elapsed = time.monotonic() - started
        remaining = args.deadline_seconds - elapsed if args.deadline_seconds else float("inf")
        if remaining <= 0:
            print(
                f"[{args.arm}] input_len={input_len} c={concurrency} SKIPPED — "
                f"wall-clock budget exhausted after {elapsed / 60:.1f} min",
                flush=True,
            )
            results.append(
                {
                    "input_len": input_len,
                    "concurrency": concurrency,
                    "output_len": OUTPUT_LEN,
                    "role": role,
                    "status": "skipped_deadline",
                    "error": f"budget {args.deadline_seconds:.0f}s exhausted at {elapsed:.0f}s",
                }
            )
            continue
        cell_timeout = min(args.timeout, remaining) if args.deadline_seconds else args.timeout

        # A cell that cannot exist under this config is recorded, not silently dropped —
        # schema 1.1 has a `status` field precisely so infeasible cells stay visible.
        if input_len + OUTPUT_LEN > MAX_MODEL_LEN:
            print(f"[{args.arm}] input_len={input_len} c={concurrency} SKIPPED — exceeds max_model_len", flush=True)
            results.append(
                {
                    "input_len": input_len,
                    "concurrency": concurrency,
                    "output_len": OUTPUT_LEN,
                    "role": role,
                    "status": "infeasible",
                    "error": f"{input_len}+{OUTPUT_LEN} > max_model_len {MAX_MODEL_LEN}",
                }
            )
            continue

        print(
            f"[{args.arm}] input_len={input_len} concurrency={concurrency} ({role}) "
            f"kv_needed={needed:,} prompts={num_prompts} "
            f"fits_v5e={needed <= V5E_KV_TOKENS} fits_v6e={needed <= V6E_KV_TOKENS}",
            flush=True,
        )
        outcome = run_cell(args.container, input_len, concurrency, num_prompts, cell_timeout)
        error: Optional[str] = outcome.get("error")
        if error:
            print(f"  FAILED: {error}", flush=True)
        else:
            got = outcome.get("output_tok_per_s")
            delta = f"  ({got / v5e_ref:.2f}x v5e's {v5e_ref})" if (v5e_ref and got) else "  (no v5e reference)"
            print(f"  {got} out tok/s, median TTFT {outcome.get('ttft_ms_median')} ms{delta}", flush=True)
        results.append(
            {
                "input_len": input_len,
                "concurrency": concurrency,
                "output_len": OUTPUT_LEN,
                "role": role,
                "num_prompts": num_prompts,
                "kv_tokens_needed": needed,
                "fits_v5e_pool": needed <= V5E_KV_TOKENS,
                "fits_v6e_pool": needed <= V6E_KV_TOKENS,
                "v5e_bf16_out_tok_per_s": v5e_ref,
                **outcome,
            }
        )

    payload = {
        "arm": args.arm,
        "model": MODEL,
        "accelerator": "tpu-v6e-1",
        "zone": "us-east5-b",
        "provisioning_model": "spot",
        "max_model_len": MAX_MODEL_LEN,
        "cells": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nWrote {args.output} — {ok}/{len(results)} cells ok", flush=True)


if __name__ == "__main__":
    main()
