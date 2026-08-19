"""Benchmark Sweep for Gemma 4 E2B QAT JAX on Cloud TPU v6e-1.

Full Grid:
  - Users / Batch Sizes (B): [1, 2, 4, 8, 16, 32, 64, 128]
  - Context Lengths (S): [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
"""

import time
import math
import sys
import os
from typing import Dict, Any, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from ports.gemma4.jax_e_model import (
    Gemma4EConfig,
    Gemma4EModelJAX,
    init_kv_cache,
    generate_n_tokens_scan,
    pad_to_tpu_v6e_bucket,
)

def build_benchmark_params(config: Gemma4EConfig):
    key = jax.random.PRNGKey(1337)
    params = {
        "embed_tokens": jax.random.normal(key, (config.vocab_size, config.hidden_size), dtype=jnp.bfloat16) * 0.02,
        "embed_tokens_per_layer": jax.random.normal(key, (config.vocab_size_per_layer_input, config.num_hidden_layers * config.hidden_size_per_layer_input), dtype=jnp.bfloat16) * 0.02,
        "per_layer_model_projection": jax.random.normal(key, (config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input), dtype=jnp.bfloat16) * 0.02,
        "per_layer_projection_norm": jnp.ones((config.hidden_size_per_layer_input,), dtype=jnp.bfloat16),
        "final_norm": jnp.ones((config.hidden_size,), dtype=jnp.bfloat16),
    }

    group_size = 32
    for i in range(config.num_hidden_layers):
        is_sliding = config.layer_types[i] == "sliding_attention"
        h_dim = config.head_dim if is_sliding else config.global_head_dim
        num_kv = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads
        is_shared = i >= config.first_kv_shared_layer_idx
        inter_size = config.intermediate_size * 2 if (is_shared and config.use_double_wide_mlp) else config.intermediate_size

        layer_params = {
            "input_layernorm": jnp.ones((config.hidden_size,), dtype=jnp.bfloat16),
            "post_attention_layernorm": jnp.ones((config.hidden_size,), dtype=jnp.bfloat16),
            "per_layer_input_gate": jax.random.normal(key, (config.hidden_size, config.hidden_size_per_layer_input), dtype=jnp.bfloat16) * 0.02,
            "per_layer_projection": jax.random.normal(key, (config.hidden_size_per_layer_input, config.hidden_size), dtype=jnp.bfloat16) * 0.02,
            "post_per_layer_input_norm": jnp.ones((config.hidden_size,), dtype=jnp.bfloat16),
            "attn": {
                "q_proj_packed": jnp.full((config.num_attention_heads * h_dim, config.hidden_size // 8), 0x22222222, dtype=jnp.int32),
                "q_proj_scale": jnp.ones((config.num_attention_heads * h_dim, config.hidden_size // group_size), dtype=jnp.bfloat16) * 0.01,
                "o_proj_packed": jnp.full((config.hidden_size, (config.num_attention_heads * h_dim) // 8), 0x22222222, dtype=jnp.int32),
                "o_proj_scale": jnp.ones((config.hidden_size, (config.num_attention_heads * h_dim) // group_size), dtype=jnp.bfloat16) * 0.01,
                "q_norm": jnp.ones((h_dim,), dtype=jnp.bfloat16),
            },
            "mlp": {
                "gate_proj_packed": jnp.full((inter_size, config.hidden_size // 8), 0x22222222, dtype=jnp.int32),
                "gate_proj_scale": jnp.ones((inter_size, config.hidden_size // group_size), dtype=jnp.bfloat16) * 0.01,
                "up_proj_packed": jnp.full((inter_size, config.hidden_size // 8), 0x22222222, dtype=jnp.int32),
                "up_proj_scale": jnp.ones((inter_size, config.hidden_size // group_size), dtype=jnp.bfloat16) * 0.01,
                "down_proj_packed": jnp.full((config.hidden_size, inter_size // 8), 0x22222222, dtype=jnp.int32),
                "down_proj_scale": jnp.ones((config.hidden_size, inter_size // group_size), dtype=jnp.bfloat16) * 0.01,
            }
        }

        if not is_shared:
            layer_params["attn"]["k_proj_packed"] = jnp.full((num_kv * h_dim, config.hidden_size // 8), 0x22222222, dtype=jnp.int32)
            layer_params["attn"]["k_proj_scale"] = jnp.ones((num_kv * h_dim, config.hidden_size // group_size), dtype=jnp.bfloat16) * 0.01
            layer_params["attn"]["v_proj_packed"] = jnp.full((num_kv * h_dim, config.hidden_size // 8), 0x22222222, dtype=jnp.int32)
            layer_params["attn"]["v_proj_scale"] = jnp.ones((num_kv * h_dim, config.hidden_size // group_size), dtype=jnp.bfloat16) * 0.01
            layer_params["attn"]["k_norm"] = jnp.ones((h_dim,), dtype=jnp.bfloat16)

        params[f"layer_{i}"] = layer_params
    return params


def run_benchmark_sweep():
    print("=" * 80)
    print("🚀 GEMMA 4 E2B QAT JAX — TPU v6e-1 BENCHMARK SWEEP GRID")
    print("=" * 80)

    devices = jax.devices()
    print(f"JAX Devices: {devices}")

    config = Gemma4EConfig(
        vocab_size=262144,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=35,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        num_global_key_value_heads=4,
        global_head_dim=512,
        num_kv_shared_layers=20,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=256,
        vocab_size_per_layer_input=262144,
    )
    model = Gemma4EModelJAX(config)
    params = build_benchmark_params(config)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    context_lengths = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    results: List[Dict[str, Any]] = []
    jit_scan_gen = jax.jit(generate_n_tokens_scan, static_argnames=("model", "num_steps", "quant_mode"))

    print(f"\nSweeping {len(batch_sizes)} Users x {len(context_lengths)} Contexts...")
    print("-" * 80)

    for B in batch_sizes:
        for S in context_lengths:
            context_str = f"{S//1024}K" if S >= 1024 else f"{S}"
            print(f"Users = {B:3d} | Context = {context_str:>5s} tokens...", end="", flush=True)
            try:
                # Pad sequence length S to static bucket to leverage pre-compiled XLA executables
                raw_ids = jnp.ones((B, S), dtype=jnp.int32)
                padded_ids, _ = pad_to_tpu_v6e_bucket(raw_ids)
                actual_s = padded_ids.shape[1]
                position_ids = jnp.tile(jnp.arange(actual_s, dtype=jnp.int32)[None, :], (B, 1))

                # Prefill measurement (1 run for compilation, 1 for measurement)
                _ = model(padded_ids, params, position_ids, quant_mode="w4a16").block_until_ready()
                start = time.time()
                _ = model(padded_ids, params, position_ids, quant_mode="w4a16").block_until_ready()
                prefill_ms = (time.time() - start) * 1000.0

                # 16-step Scan Generation
                _ = jit_scan_gen(model, padded_ids, params, num_steps=16, quant_mode="w4a16").block_until_ready()
                start = time.time()
                _ = jit_scan_gen(model, padded_ids, params, num_steps=16, quant_mode="w4a16").block_until_ready()
                scan_ms = (time.time() - start) * 1000.0

                step_ms = scan_ms / 16.0
                total_tokens = B * 16
                agg_tok_sec = (total_tokens * 1000.0) / scan_ms
                user_tok_sec = (16 * 1000.0) / scan_ms

                print(f" Prefill: {prefill_ms:7.1f} ms | Step: {step_ms:5.2f} ms | Agg: {agg_tok_sec:6.1f} tok/s | Per-User: {user_tok_sec:5.1f} tok/s")

                results.append({
                    "B": B,
                    "S": S,
                    "prefill_ms": prefill_ms,
                    "step_ms": step_ms,
                    "agg_tok_sec": agg_tok_sec,
                    "user_tok_sec": user_tok_sec,
                    "status": "OK"
                })

            except Exception as e:
                err_str = str(e)
                print(f" OOM / Error: {err_str[:40]}")
                results.append({
                    "B": B,
                    "S": S,
                    "prefill_ms": None,
                    "step_ms": None,
                    "agg_tok_sec": None,
                    "user_tok_sec": None,
                    "status": "OOM"
                })

    print("\n" + "=" * 80)
    print("📊 GEMMA 4 E2B QAT TPU v6e-1 BENCHMARK SWEEP RESULTS")
    print("=" * 80)
    
    print("\n| Users ($B$) | Context ($S$) | Prefill Latency | Decode Step Latency | Aggregate Throughput | Per-User Throughput |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        context_str = f"{r['S']//1024}K" if r['S'] >= 1024 else f"{r['S']}"
        if r["status"] == "OK":
            print(f"| {r['B']:d} | {context_str} | {r['prefill_ms']:.1f} ms | {r['step_ms']:.2f} ms | {r['agg_tok_sec']:.1f} tok/s | {r['user_tok_sec']:.1f} tok/s |")
        else:
            print(f"| {r['B']:d} | {context_str} | OOM | OOM | OOM | OOM |")

if __name__ == "__main__":
    run_benchmark_sweep()
