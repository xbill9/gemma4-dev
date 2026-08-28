"""Corrected benchmark sweep for Gemma 4 E2B QAT JAX on Cloud TPU v6e-1.

Supersedes ``jax_e_benchmark_sweep.py``, which had three methodology defects:

  1. Prefill was timed on an UN-jitted ``model(...)`` call, so it measured
     op-by-op dispatch overhead rather than prefill compute. (Symptom in the
     old data: prefill rose only 1.4x — 544 ms to 779 ms — while context grew
     512x, from 8 to 4096 tokens. Real prefill is at least linear in context.)
  2. "Decode step latency" was ``total_scan_time / 16``, where the scan also
     contained the prefill — so the reported per-step cost silently included
     amortized prefill, which is why it drifted with context length.
  3. ``generate_n_tokens_scan`` runs each step on a single token with no KV
     cache and no causal mask, so it never attends to history. It is a
     hardware throughput probe, not autoregressive decoding.

This version measures the two phases separately, both jitted, using the
cache-correct path verified in ``tests/test_kv_cache_parity.py``:

  * TTFT / prefill: jitted ``prefill_with_kv_cache`` over the padded prompt.
  * Decode: jitted single-token steps against a populated static KV cache,
    timed in steady state (warmup discarded, median of N repeats).

Run on a TPU VM:  python3 ports/gemma4/jax_e_benchmark_sweep_v2.py
"""

import argparse
import json
import statistics
import sys
import os
import time
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from ports.gemma4.jax_e_model import (
    Gemma4EConfig,
    Gemma4EModelJAX,
    make_cached_decode_step,
    prefill_with_kv_cache,
    pad_to_tpu_v6e_bucket,
    dequantize_params_to_dense,
    quantize_lm_head,
    quantize_ple_table,
    set_w4a16_impl,
)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params

# Verified against the shipped checkpoint on 2026-07-28 by reading
# google/gemma-4-E2B-it-qat-w4a16-ct's config.json on a live v6e-1
# (benchmarks/runs/2026-07-28-jax-e2b-v6e1). Three values previously differed from
# the real model and are corrected here:
#   hidden_size            2048 -> 1536
#   num_key_value_heads       4 -> 1   (a single 256-dim KV head; 15.0 KiB/token bf16,
#                                       matching the measured 7,680 B/token under fp8)
#   num_global_key_value_heads 4 -> 1  (config.json reports null)
# The checkpoint also declares sliding_window: 512, which the model does NOT yet
# implement — every layer attends to full context, so long-context results here are
# not faithful to Gemma 4.
# Residual discrepancy: this config yields 18.0 KiB/token of KV, while the measured
# vLLM allocator reports 15.0 KiB (= 15 layers x 1 head x 256 dim x (K+V) x 2 B).
# The gap is the 3 full-attention layers, which we give a 512-dim KV head via
# global_head_dim. The measurement implies KV is uniformly 256-dim and that 512 is
# the *query* head dim for those layers. Unresolved; memory here is ~20% pessimistic.
E2B_CONFIG = dict(
    vocab_size=262144,
    hidden_size=1536,
    intermediate_size=6144,
    num_hidden_layers=35,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
    num_global_key_value_heads=1,
    global_head_dim=512,
    num_kv_shared_layers=20,
    use_double_wide_mlp=True,
    hidden_size_per_layer_input=256,
    vocab_size_per_layer_input=262144,
)


def time_median_ms(fn, repeats: int, warmup: int = 2) -> float:
    """Median wall time in ms, with warmup iterations discarded.

    The warmup covers XLA compilation and first-touch allocation; without it the
    first cell measured in a sweep absorbs one-time costs and reads as anomalously
    slow (the defect that produced the old sweep's B=1 outlier).
    """
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def bench_cell(model, params, B: int, S: int, decode_steps: int, repeats: int, window_kv: bool = True, quant_mode: str = "w4a16") -> Dict[str, Any]:
    """Measure one (batch, context) cell: prefill TTFT and steady-state decode."""
    raw_ids = jnp.ones((B, S), dtype=jnp.int32)
    padded_ids, valid_mask = pad_to_tpu_v6e_bucket(raw_ids)
    bucket_s = padded_ids.shape[1]

    jit_prefill = jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype", "window_kv"),
    )

    def run_prefill():
        return jit_prefill(
            model=model, prompt_ids=padded_ids, prompt_valid=valid_mask,
            params=params, max_new_tokens=decode_steps, quant_mode=quant_mode,
            window_kv=window_kv,
        )

    prefill_ms = time_median_ms(run_prefill, repeats=repeats)

    # Populate the cache once, then time steady-state decode against it.
    last_logits, caches, valid = jax.block_until_ready(run_prefill())
    step = jax.jit(make_cached_decode_step(model, quant_mode=quant_mode, window_kv=window_kv))
    prompt_lens = valid_mask.sum(axis=1).astype(jnp.int32)
    tok = jnp.argmax(last_logits, axis=-1, keepdims=True)

    def run_step():
        # Fixed slot: measures steady-state per-token cost against a full cache
        # without mutating state across repeats.
        return step(params, caches, valid, tok, prompt_lens, jnp.int32(bucket_s))

    step_ms = time_median_ms(run_step, repeats=repeats)

    return {
        "B": B,
        "S": S,
        "bucket_S": bucket_s,
        "prefill_ms": round(prefill_ms, 2),
        "decode_step_ms": round(step_ms, 3),
        "agg_decode_tok_s": round(B * 1000.0 / step_ms, 1),
        "per_user_tok_s": round(1000.0 / step_ms, 1),
        "status": "OK",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--contexts", default="8,64,128,256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--decode-steps", type=int, default=16,
                    help="KV cache headroom reserved beyond the prompt")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--w4a16-impl", default="reference", choices=["auto", "fused", "reference"],
                    help="fused unpacks int4 in-tile (~4x less weight traffic); "
                         "reference dequantizes to BF16 in HBM first")
    ap.add_argument("--w4a16-layout", default="plane", choices=["interleaved", "plane"],
                    help="in-tile nibble layout; both exact, A/B them on TPU")
    ap.add_argument("--no-window-kv", action="store_true",
                    help="disable windowed KV for sliding layers (full-length cache)")
    ap.add_argument("--dequant-at-load", action="store_true",
                    help="materialize W4A16 weights to dense BF16 once at load and run "
                         "the fp16 path; trades storage (which E2B has spare) for "
                         "removing the per-forward dequant")
    ap.add_argument("--int8-ple", action="store_true",
                    help="int8 the per-layer-embedding table (4.70 GB -> 2.35 GB); "
                         "memory headroom, not bandwidth")
    ap.add_argument("--int8-lm-head", action="store_true",
                    help="halve the largest decode read (~0.8%% logit error)")
    ap.add_argument("--tiny", action="store_true",
                    help="Run a small config that fits on CPU — verifies the harness, "
                         "not TPU performance.")
    args = ap.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]

    print("=" * 92)
    print("GEMMA 4 E2B QAT JAX — TPU v6e-1 CORRECTED SWEEP (cache-correct decode)")
    print("=" * 92)
    print(f"JAX devices: {jax.devices()}")

    if args.tiny:
        cfg_kwargs = dict(
            E2B_CONFIG,
            vocab_size=256, hidden_size=128, intermediate_size=192,
            num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_global_key_value_heads=2, global_head_dim=32,
            num_kv_shared_layers=4, hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=256,
        )
        print("*** --tiny: harness verification only, NOT representative performance ***")
    else:
        cfg_kwargs = dict(E2B_CONFIG)

    config = Gemma4EConfig(**cfg_kwargs)
    model = Gemma4EModelJAX(config)
    params = build_benchmark_params(config)

    set_w4a16_impl(args.w4a16_impl, args.w4a16_layout)
    if args.int8_lm_head:
        params = quantize_lm_head(params)
    if args.int8_ple:
        params = quantize_ple_table(params)
    if args.dequant_at_load:
        params = dequantize_params_to_dense(params)
    print(f"W4A16: impl={args.w4a16_impl} layout={args.w4a16_layout} | "
          f"int8_lm_head={args.int8_lm_head} | int8_ple={args.int8_ple}")
    print("NOTE: synthetic architecture-identical weights; throughput is "
          "value-independent under static shapes. Use the real checkpoint via "
          "jax_e_loader.py for quality evaluation.\n")

    results: List[Dict[str, Any]] = []
    for B in batch_sizes:
        for S in contexts:
            label = f"{S // 1024}K" if S >= 1024 else str(S)
            print(f"B={B:4d} | ctx={label:>5s} ...", end="", flush=True)
            try:
                cell = bench_cell(model, params, B, S, args.decode_steps, args.repeats,
                                  window_kv=not args.no_window_kv,
                                  quant_mode="fp16" if args.dequant_at_load else "w4a16")
                print(f" prefill {cell['prefill_ms']:8.1f} ms | step "
                      f"{cell['decode_step_ms']:7.3f} ms | agg "
                      f"{cell['agg_decode_tok_s']:8.1f} tok/s | per-user "
                      f"{cell['per_user_tok_s']:6.1f} tok/s")
            except Exception as exc:  # OOM or shape failure
                msg = str(exc).split("\n")[0][:60]
                print(f" FAILED: {msg}")
                cell = {"B": B, "S": S, "status": "OOM", "error": msg}
            results.append(cell)

    print("\n" + "=" * 92)
    print("| Users (B) | Context (S) | Prefill (TTFT) | Decode Step | Aggregate | Per-User |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        label = f"{r['S'] // 1024}K" if r["S"] >= 1024 else str(r["S"])
        if r["status"] == "OK":
            print(f"| {r['B']} | {label} | {r['prefill_ms']:.1f} ms | "
                  f"{r['decode_step_ms']:.2f} ms | {r['agg_decode_tok_s']:.1f} tok/s | "
                  f"{r['per_user_tok_s']:.1f} tok/s |")
        else:
            print(f"| {r['B']} | {label} | OOM | OOM | OOM | OOM |")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({
                "results": results,
                "devices": [str(d) for d in jax.devices()],
                "config": {
                    "w4a16_impl": args.w4a16_impl,
                    "w4a16_layout": args.w4a16_layout,
                    "int8_lm_head": args.int8_lm_head,
                    "tiny": args.tiny,
                },
            }, fh, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
