"""Independent re-measurement of the claims published in devto-vllm-gemma4-e2b-v6e1.md.

Every cell here re-derives one specific published number on a freshly provisioned v6e-1. The
point is falsification, not confirmation: each cell carries the value the article asserts, and
the output records the delta so a drift shows up as a number rather than a vibe.

Two things differ from the 2026-08-10 config-validation sweep this checks:

  * **A distinct seed per cell.** That run's `16000x32` cell followed `16000x64` at `--seed 0`
    and drew ~100% prefix-cache hits, which is what made an ordinary queueing curve look like a
    3.4x cliff. Same-`input_len` cells are not independent unless the seed varies.
  * **Preemption counters are read per cell** from /metrics, not once at the end, so the
    "no eviction at 157% of pool" claim is checked against a delta rather than a total.

Usage:
    python3 validate.py --output results/validation.json [--deadline-seconds N]
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

sys.path.insert(0, "../2026-08-10-config-validation-v6e1")
from run_cells import run_cell  # noqa: E402

MODEL = "google/gemma-4-E2B-it"
OUTPUT_LEN = 128
KV_POOL_TOKENS = 1_151_744

# (input_len, concurrency, role, published_tok_s, published_tpot_ms, what it checks)
CELLS = [
    (128, 1, "control", 202.8, 4.72, "single-stream bandwidth floor; 1.70x v5e TPOT"),
    (1024, 16, "control", 1544.3, 9.30, "control regime tracks bandwidth, not price"),
    (4096, 64, "bandwidth", 1643.6, 29.79, "best measured cell; the cost-crossover point"),
    (8192, 32, "bandwidth", 938.6, 30.70, "3.05x — high end of the memory-bound regime"),
    (16000, 64, "v6e_only", 469.2, 67.28, "anchor of the TTFT line, 90% of pool"),
    (16000, 112, "overflow", 473.9, 67.25, "157% of pool: flat throughput, zero preemptions"),
    (32000, 16, "long_ctx", 249.1, 51.78, "context impossible on v5e at any setting"),
]

# TTFT = A + B * concurrency, fitted over 56-157% of pool at input_len 16000.
TTFT_LINE = (-8542.0, 265.4)


def scrape_metric(name: str, port: int = 8000) -> Optional[float]:
    """Read one counter out of the vLLM Prometheus endpoint."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15) as r:
            body = r.read().decode()
    except Exception:
        return None
    total = 0.0
    seen = False
    for line in body.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        m = re.search(r"\s([0-9.eE+-]+)\s*$", line)
        if m:
            total += float(m.group(1))
            seen = True
    return total if seen else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--container", default="vllm-gemma4")
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--prompts-per-concurrency", type=int, default=3)
    ap.add_argument("--deadline-seconds", type=float, default=0.0)
    args = ap.parse_args()

    started = time.monotonic()
    results: List[Dict[str, Any]] = []

    for idx, (input_len, conc, role, pub_tps, pub_tpot, checks) in enumerate(CELLS):
        needed = conc * (input_len + OUTPUT_LEN)
        num_prompts = max(conc * args.prompts_per_concurrency, 16)
        elapsed = time.monotonic() - started
        if args.deadline_seconds and elapsed >= args.deadline_seconds:
            print(f"SKIP {input_len}x{conc} — budget exhausted after {elapsed / 60:.1f} min", flush=True)
            results.append(
                {
                    "input_len": input_len,
                    "concurrency": conc,
                    "role": role,
                    "status": "skipped_deadline",
                    "published_tok_per_s": pub_tps,
                    "checks": checks,
                }
            )
            continue

        pre = scrape_metric("vllm:num_preemptions_total")
        print(
            f"[{idx + 1}/{len(CELLS)}] {input_len}x{conc} ({role}) "
            f"kv={needed:,} ({needed / KV_POOL_TOKENS * 100:.0f}% of pool) seed={idx + 1}",
            flush=True,
        )

        # Distinct seed per cell — same input_len across cells would otherwise share prompts.
        out = run_cell(args.container, input_len, conc, num_prompts, args.timeout, seed=idx + 1)
        post = scrape_metric("vllm:num_preemptions_total")

        rec: Dict[str, Any] = {
            "input_len": input_len,
            "concurrency": conc,
            "role": role,
            "seed": idx + 1,
            "num_prompts": num_prompts,
            "kv_tokens_needed": needed,
            "pct_of_pool": round(needed / KV_POOL_TOKENS * 100, 1),
            "published_tok_per_s": pub_tps,
            "published_tpot_ms": pub_tpot,
            "checks": checks,
            "preemptions_before": pre,
            "preemptions_after": post,
            "preemptions_this_cell": (post - pre) if (pre is not None and post is not None) else None,
            "prefix_hit_tokens": scrape_metric("vllm:prefix_cache_hits_total"),
            "prefix_query_tokens": scrape_metric("vllm:prefix_cache_queries_total"),
            **out,
        }

        if out.get("status") == "ok":
            got = out.get("output_tok_per_s")
            if got:
                rec["tok_per_s_delta_pct"] = round((got / pub_tps - 1) * 100, 1)
            got_tpot = out.get("tpot_ms_median")
            if got_tpot:
                rec["tpot_delta_pct"] = round((got_tpot / pub_tpot - 1) * 100, 1)
            if input_len == 16000:
                pred = TTFT_LINE[0] + TTFT_LINE[1] * conc
                meas = out.get("ttft_ms_median")
                if meas:
                    rec["ttft_line_predicted_ms"] = round(pred, 0)
                    rec["ttft_line_delta_pct"] = round((meas / pred - 1) * 100, 2)
            print(
                f"    -> {got:.1f} tok/s (published {pub_tps}, {rec.get('tok_per_s_delta_pct'):+.1f}%) "
                f"tpot={got_tpot:.2f} preempt={rec['preemptions_this_cell']}",
                flush=True,
            )
        else:
            print(f"    -> FAILED: {out.get('error')}", flush=True)

        results.append(rec)

    payload = {
        "arm": "v6e1-32768-article-validation",
        "model": MODEL,
        "accelerator": "tpu-v6e-1",
        "output_len": OUTPUT_LEN,
        "kv_pool_tokens": KV_POOL_TOKENS,
        "ttft_line": {"intercept_ms": TTFT_LINE[0], "slope_ms_per_client": TTFT_LINE[1]},
        "cells": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
