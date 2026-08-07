"""Benchmark sweep for Gemma 4 12B QAT JAX on Cloud TPU v6e-1.

Measures TTFT (prefill latency) and steady-state cached decode throughput
across batch sizes (1, 2, 4, 8 users) and context lengths (1K, 2K, 4K, 8K, 16K, 32K, 64K).

Usage:
    python3.13 ports/gemma4/jax_12b_benchmark_sweep.py --batch-sizes 1,2,4,8 --contexts 1024,2048,4096,8192,16384,32768,65536
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from jax_engine import config_from_hf
from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params
from ports.gemma4.jax_e_model import (
    Gemma4EModelJAX,
    make_cached_decode_step,
    pad_to_tpu_v6e_bucket,
    prefill_with_kv_cache,
    set_w4a16_impl,
)

DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"


def time_median_ms(fn, repeats: int = 3, warmup: int = 1) -> float:
    """Measures median wall time in ms, discarding warmup iterations."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def bench_cell(model, params, B: int, S: int, decode_steps: int = 16, repeats: int = 3) -> Dict[str, Any]:
    """Measures prefill TTFT and steady-state decode for one (batch, context) cell."""
    raw_ids = jnp.ones((B, S), dtype=jnp.int32)
    padded_ids, valid_mask = pad_to_tpu_v6e_bucket(raw_ids)
    bucket_s = padded_ids.shape[1]

    jit_prefill = jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype", "window_kv"),
    )

    def run_prefill():
        return jit_prefill(
            model=model,
            prompt_ids=padded_ids,
            prompt_valid=valid_mask,
            params=params,
            max_new_tokens=decode_steps,
            quant_mode="w4a16",
            window_kv=True,
        )

    prefill_ms = time_median_ms(run_prefill, repeats=repeats)

    last_logits, caches, valid = jax.block_until_ready(run_prefill())
    step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16", window_kv=True))
    prompt_lens = valid_mask.sum(axis=1).astype(jnp.int32)
    tok = jnp.argmax(last_logits, axis=-1, keepdims=True)

    def run_step():
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
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--contexts", default="1024,2048,4096,8192,16384,32768,65536")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json-out", default="gemma4_12b_benchmark_results.json")
    args = ap.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]

    print("=" * 92)
    print(f"GEMMA 4 12B QAT JAX — TPU v6e-1 BENCHMARK SWEEP")
    print("=" * 92)
    print(f"JAX devices: {jax.devices()}")

    cfg_path = hf_hub_download(args.model, "config.json")
    with open(cfg_path) as f:
        hf_cfg = json.load(f)
    config = config_from_hf(hf_cfg)
    model = Gemma4EModelJAX(config)

    st_path = hf_hub_download(args.model, "model.safetensors")
    print(f"Loading weights from {st_path}...")
    raw_weights = {}
    with safe_open(st_path, framework="np", device="cpu") as f:
        for k in f.keys():
            raw_weights[k] = f.get_tensor(k)

    params = convert_safetensors_to_jax_params(
        raw_weights,
        num_layers=config.num_hidden_layers,
        first_kv_shared_idx=config.first_kv_shared_layer_idx,
        attention_k_eq_v=config.attention_k_eq_v,
    )
    set_w4a16_impl("reference", "plane")
    print("Model and parameters ready. Starting benchmark sweep...\n")

    results: List[Dict[str, Any]] = []
    for B in batch_sizes:
        for S in contexts:
            label = f"{S // 1024}K" if S >= 1024 else str(S)
            print(f"B={B:2d} users | ctx={label:>5s} ... ", end="", flush=True)
            try:
                cell = bench_cell(model, params, B, S, repeats=args.repeats)
                print(f"prefill {cell['prefill_ms']:8.1f} ms | step {cell['decode_step_ms']:7.3f} ms | "
                      f"agg {cell['agg_decode_tok_s']:8.1f} tok/s | per-user {cell['per_user_tok_s']:6.1f} tok/s")
            except Exception as exc:
                msg = str(exc).split("\n")[0][:60]
                print(f"FAILED: {msg}")
                cell = {"B": B, "S": S, "status": "OOM", "error": msg}
            results.append(cell)

    print("\n" + "=" * 92)
    print("| Users (B) | Context (S) | Prefill (TTFT) | Decode Step | Aggregate Tok/s | Per-User Tok/s |")
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
            json.dump({"results": results, "devices": [str(d) for d in jax.devices()]}, fh, indent=2)
        print(f"\nWrote results to {args.json_out}")


if __name__ == "__main__":
    main()
