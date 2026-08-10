"""Locate the capacity knee on v6e-1 by bisecting concurrency at fixed context.

The main sweep (`run_cells.py`) left a gap. Its cells jumped from 46% of the v6e KV pool
(532,480 tokens, median TTFT 1,062 ms) straight to 89% (1,028,096 tokens, 8,713 ms) — a 3.4x
TTFT jump across an interval nothing sampled. This fills it.

Design: **one variable.** Context is pinned at 16,000 and only concurrency moves, so nothing
here is confounded by prefill length, block count, or bucket shape. The two endpoints are
already measured by the main run and are re-listed below as anchors rather than re-run:

    16000 x 32  ->   516,096 tokens (45% of pool),  TTFT 2,461 ms
    16000 x 64  -> 1,032,192 tokens (90% of pool),  TTFT 8,451 ms

What is being tested. `SERVING-PARAMS.md` carried a v5e rule of thumb — keep
`clients x context` under ~250,000 tokens, about **78%** of that chip's pool — and derived,
without measuring, that the v6e equivalent is ~900,000 tokens. The main sweep is consistent
with a knee somewhere in 46-89%, but cannot place it. If the rule is really a *fraction of
pool* rather than an absolute token count, TTFT should turn upward near 78%, i.e. between
concurrency 52 and 56 below.

A knee that lands far from 78% is the more interesting outcome: it would mean the v5e figure
does not transfer as a fraction, and the capacity guidance in `SERVING-PARAMS.md` needs to be
stated per-chip rather than derived.

Throughput is expected to stay roughly flat across these points — the knee showed up in TTFT,
not in tokens/s, which is exactly why a throughput-only sweep would have missed it.
"""

import argparse
import json
import sys
from typing import Any, Dict, List

sys.path.insert(0, "/home/xbill")  # run_cells.py is scp'd to the VM home dir
from run_cells import MODEL, OUTPUT_LEN, run_cell  # noqa: E402

V6E_KV_TOKENS = 1_151_744
INPUT_LEN = 16_000

# Concurrency points bisecting 45% -> 90% of the pool at fixed 16,000 context.
# 52 and 56 straddle the derived 78% (898,360 token) prediction.
CONCURRENCIES = [40, 46, 52, 56, 60]

# Measured in the main sweep; carried for the curve, not re-run.
ANCHORS = [(32, 516_096, 2461.0, 444.0), (64, 1_032_192, 8450.94, 469.2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="vllm-gemma4")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--prompts-per-concurrency", type=int, default=3)
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    for conc in CONCURRENCIES:
        needed = conc * (INPUT_LEN + OUTPUT_LEN)
        pct = 100 * needed / V6E_KV_TOKENS
        num_prompts = max(conc * args.prompts_per_concurrency, 16)
        print(f"[knee] {INPUT_LEN} x c={conc}  kv={needed:,} ({pct:.0f}% of pool) prompts={num_prompts}", flush=True)
        outcome = run_cell(args.container, INPUT_LEN, conc, num_prompts, args.timeout)
        if outcome.get("error"):
            print(f"  FAILED: {outcome['error']}", flush=True)
        else:
            print(
                f"  {outcome.get('output_tok_per_s')} out tok/s, median TTFT {outcome.get('ttft_ms_median')} ms",
                flush=True,
            )
        results.append(
            {
                "input_len": INPUT_LEN,
                "concurrency": conc,
                "output_len": OUTPUT_LEN,
                "kv_tokens_needed": needed,
                "pct_of_v6e_pool": round(pct, 1),
                "num_prompts": num_prompts,
                **outcome,
            }
        )

    payload = {
        "arm": "v6e1-32768-knee",
        "model": MODEL,
        "accelerator": "tpu-v6e-1",
        "zone": "europe-west4-a",
        "provisioning_model": "flex-start",
        "input_len_fixed": INPUT_LEN,
        "v6e_kv_tokens": V6E_KV_TOKENS,
        "anchors_from_main_sweep": [
            {"concurrency": c, "kv_tokens_needed": kv, "ttft_ms_median": t, "output_tok_per_s": o}
            for c, kv, t, o in ANCHORS
        ],
        "cells": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nWrote {args.output} — {ok}/{len(results)} cells ok", flush=True)


if __name__ == "__main__":
    main()
