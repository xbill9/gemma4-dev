"""Push PAST the KV pool and find where preemption actually starts.

The gap this closes. The main sweep and the knee sweep between them covered 0-90% of the
v6e-1 KV pool and found no degradation threshold: TTFT is linear in concurrency
(R^2 = 1.0000, 265 ms per added request) and throughput is flat within 4.2%. The reason turned
out to be mundane — `num_preemptions_total` stayed at **0** for the whole run, so nothing was
ever evicted. Occupancy alone costs nothing; eviction is what would cost something, and no cell
ever triggered any.

**Every cell so far fit.** The largest sat at 90% of pool. So "there is no cliff" was only ever
established below 100%, which is the easy half of the claim. This crosses the line.

Cells run 101% -> 157% of pool at fixed 16,000 context, varying only concurrency.

Two design fixes carried over from mistakes made earlier in this run:

  1. **A distinct seed per cell.** vLLM defaults to `enable_prefix_caching=True` and the random
     dataset is deterministic in the seed, so same-seed cells at the same `input_len` draw
     overlapping prompts and the later one is served from cache. That silently invalidated the
     `16000x32` cell of the main sweep (~1.54M hit tokens against 1.536M input tokens). Every
     cell here gets its own seed, and none reuses seed 0.
  2. **Metrics are sampled before and after each cell**, so preemption is measured as a delta
     per cell rather than inferred from a cumulative total at the end.

What would falsify what. If preemptions stay 0 above 100% of pool, then the scheduler is
admission-controlling rather than evicting — requests queue instead of thrashing, and the
"cliff" simply does not exist for this workload at any occupancy. If preemptions go non-zero,
the TTFT slope should break away from 265 ms/request at that point, and *that* concurrency is
the real capacity limit for the rig.

Note the interaction with the Gemma quirk: sliding-window suppression means all 15 cached
layers hold full-length blocks though 12 of them never read past 512 tokens, so the pool is
charged ~18 KiB/token where ~6.4 KiB would do at this context. Whatever limit shows up here is
therefore ~2.9x closer than the architecture requires. See `@QUANTIZATION.md`.
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/home/xbill")
from run_cells import MODEL, OUTPUT_LEN, run_cell  # noqa: E402

V6E_KV_TOKENS = 1_151_744
INPUT_LEN = 16_000

# (concurrency, seed) — seeds are distinct and none is 0.
CELLS = [(72, 101), (80, 202), (96, 303), (112, 404)]

METRIC_KEYS = (
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
)


def scrape_metrics(container: str) -> Dict[str, float]:
    """Read the counters we care about straight off /metrics."""
    try:
        proc = subprocess.run(
            ["sudo", "docker", "exec", container, "curl", "-s", "http://localhost:8000/metrics"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {}
    out: Dict[str, float] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("#"):
            continue
        for key in METRIC_KEYS:
            if line.startswith(key + "{"):
                try:
                    out[key] = float(line.rsplit("}", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="vllm-gemma4")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--prompts-per-concurrency", type=int, default=2)
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    for conc, seed in CELLS:
        needed = conc * (INPUT_LEN + OUTPUT_LEN)
        pct = 100 * needed / V6E_KV_TOKENS
        num_prompts = max(conc * args.prompts_per_concurrency, 16)

        before = scrape_metrics(args.container)
        print(
            f"[overflow] {INPUT_LEN} x c={conc} seed={seed}  kv={needed:,} ({pct:.0f}% of pool) "
            f"prompts={num_prompts}  preempt_before={before.get('vllm:num_preemptions_total', '?')}",
            flush=True,
        )
        outcome = run_cell(args.container, INPUT_LEN, conc, num_prompts, args.timeout, seed=seed)
        after = scrape_metrics(args.container)

        d_pre: Optional[float] = None
        if "vllm:num_preemptions_total" in before and "vllm:num_preemptions_total" in after:
            d_pre = after["vllm:num_preemptions_total"] - before["vllm:num_preemptions_total"]
        d_hits = after.get("vllm:prefix_cache_hits_total", 0) - before.get("vllm:prefix_cache_hits_total", 0)
        d_q = after.get("vllm:prefix_cache_queries_total", 0) - before.get("vllm:prefix_cache_queries_total", 0)

        if outcome.get("error"):
            print(f"  FAILED: {outcome['error']}", flush=True)
        else:
            print(
                f"  {outcome.get('output_tok_per_s')} out tok/s, median TTFT {outcome.get('ttft_ms_median')} ms"
                f"  | PREEMPTIONS THIS CELL: {d_pre}"
                f"  | prefix hit {100 * d_hits / d_q if d_q else 0:.1f}%",
                flush=True,
            )
        results.append(
            {
                "input_len": INPUT_LEN,
                "concurrency": conc,
                "output_len": OUTPUT_LEN,
                "seed": seed,
                "kv_tokens_needed": needed,
                "pct_of_v6e_pool": round(pct, 1),
                "num_prompts": num_prompts,
                "preemptions_this_cell": d_pre,
                "prefix_hit_tokens_this_cell": d_hits,
                "prefix_query_tokens_this_cell": d_q,
                **outcome,
            }
        )

    with open(args.output, "w") as f:
        json.dump(
            {
                "arm": "v6e1-32768-overflow",
                "model": MODEL,
                "accelerator": "tpu-v6e-1",
                "zone": "europe-west4-a",
                "input_len_fixed": INPUT_LEN,
                "v6e_kv_tokens": V6E_KV_TOKENS,
                "cells": results,
            },
            f,
            indent=2,
        )
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nWrote {args.output} — {ok}/{len(results)} cells ok", flush=True)


if __name__ == "__main__":
    main()
