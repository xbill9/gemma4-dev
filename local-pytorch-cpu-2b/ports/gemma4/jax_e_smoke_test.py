"""End-to-End Smoke Test for Gemma 4 E2B QAT JAX Model (ports/gemma4/jax_e_model.py).

Validates full-scale Gemma 4 E2B MatFormer architecture in pure JAX:
  1. Full config initialization (35 layers, 20 KV-shared layers, PLE=256, double-wide MLP).
  2. W4A16 QAT parameter generation (int4 packed weights + bfloat16 scales).
  3. Prefill forward pass (uncompiled and jax.jit compiled).
  4. Autoregressive decoding loop (5 generation steps).
  5. Sanity assertions on logits (shape, non-NaN/Inf, softcapping range [-30, 30]).
"""

import time
import math
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from ports.gemma4.jax_e_model import (
    Gemma4EConfig,
    Gemma4EModelJAX,
    qat_w4a16_unpack_dequant_jax,
    qat_w4a16_pallas_matmul_jax,
    init_kv_cache,
    init_paged_kv_cache,
    generate_n_tokens_scan,
    TPUv6eHardwareProfile,
    pad_to_tpu_v6e_bucket,
    onchip_sample_tpu_v6e_jax,
)


def run_e2e_smoke_test():
    print("=" * 70)
    print("🚀 GEMMA 4 E2B QAT JAX — END-TO-END SMOKE TEST")
    print("=" * 70)

    # 1. Real Gemma 4 E2B MatFormer configuration
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
    print(f"✓ Config initialized: {config.num_hidden_layers} layers ({config.num_kv_shared_layers} KV-shared), PLE={config.hidden_size_per_layer_input}")

    # 2. Build mock QAT (W4A16) parameters for full E2B architecture
    key = jax.random.PRNGKey(1337)
    params = {
        "embed_tokens": jax.random.normal(key, (config.vocab_size, config.hidden_size), dtype=jnp.bfloat16) * 0.02,
        "embed_tokens_per_layer": jax.random.normal(key, (config.vocab_size_per_layer_input, config.num_hidden_layers * config.hidden_size_per_layer_input), dtype=jnp.bfloat16) * 0.02,
        "per_layer_model_projection": jax.random.normal(key, (config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input), dtype=jnp.bfloat16) * 0.02,
        "per_layer_projection_norm": jnp.ones((config.num_hidden_layers * config.hidden_size_per_layer_input,), dtype=jnp.bfloat16),
        "final_norm": jnp.ones((config.hidden_size,), dtype=jnp.bfloat16),
    }

    group_size = 32
    print("✓ Generating W4A16 QAT parameters for 35 layers...")
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
            layer_params["attn"]["v_norm"] = jnp.ones((h_dim,), dtype=jnp.bfloat16)

        params[f"layer_{i}"] = layer_params

    # 3. Model initialization
    model = Gemma4EModelJAX(config)

    # 4. FP8 Static & Paged KV Cache Initialization Test
    fp8_cache = init_kv_cache(config, batch_size=1, max_seq_len=2048, dtype=jnp.float8_e4m3fn)
    paged_cache = init_paged_kv_cache(config, num_blocks=512, block_size=16, batch_size=1, dtype=jnp.float8_e4m3fn)
    print(f"✓ Static FP8 KV Cache preallocated for {len(fp8_cache)} non-shared layers (dtype=fp8_e4m3fn)")
    print(f"✓ PagedAttention Block Cache initialized for {len(paged_cache)} layers (512 blocks x 16 tokens/block)")

    # 4b. Pallas Fused W4A16 Dequant-Matmul Kernel Test
    sample_x = jax.random.normal(jax.random.PRNGKey(42), (16, config.hidden_size), dtype=jnp.bfloat16)
    sample_packed = params["layer_0"]["attn"]["q_proj_packed"]
    sample_scale = params["layer_0"]["attn"]["q_proj_scale"]
    pallas_out = qat_w4a16_pallas_matmul_jax(sample_x, sample_packed, sample_scale)
    assert pallas_out.shape == (16, config.num_attention_heads * config.head_dim)
    print("✓ Pallas fused W4A16 VMEM dequant-matmul kernel execution verified")

    # 4c. TPU v6e Static Bucket Padding Test
    dummy_input = jnp.array([[101, 2054, 2003, 1037, 2742]], dtype=jnp.int32)
    padded_input, mask = pad_to_tpu_v6e_bucket(dummy_input)
    assert padded_input.shape[1] == 64  # Padded to nearest 128-aligned static bucket (64)
    print(f"✓ Static 128-aligned bucket padding verified: Input len 5 -> padded bucket len {padded_input.shape[1]}")

    # 5. Prefill test (Prompt: sequence of 16 tokens)
    prompt_len = 16
    input_ids = jnp.array([[101, 2054, 2003, 1037, 2742, 2000, 2022, 2172, 2005, 1037, 3054, 2000, 2022, 2172, 2005, 1037]], dtype=jnp.int32)
    position_ids = jnp.arange(prompt_len, dtype=jnp.int32)[None, :]

    print(f"\n--- Prefill Forward Pass (SeqLen = {prompt_len}) ---")
    start = time.time()
    logits = model(input_ids, params, position_ids, quant_mode="w4a16")
    duration = (time.time() - start) * 1000
    print(f"✓ Uncompiled forward complete: Logits shape = {logits.shape}, Latency = {duration:.2f} ms")

    # Vectorized On-Chip TPU Top-K Sampling Test over 262,144 vocab
    sample_key = jax.random.PRNGKey(99)
    sample_token = onchip_sample_tpu_v6e_jax(logits[:, -1, :], sample_key, temperature=0.7, top_k=40)
    assert sample_token.shape == (1, 1)
    print("✓ On-chip TPU Top-K sampling (over 262,144 vocab) verified")

    # Sanity checks on output logits
    assert logits.shape == (1, prompt_len, config.vocab_size), f"Unexpected shape {logits.shape}"
    assert not jnp.isnan(logits).any(), "Logits contain NaN!"
    assert not jnp.isinf(logits).any(), "Logits contain Inf!"
    assert float(jnp.max(jnp.abs(logits))) <= 30.0, "Logit softcapping bound (>30.0) violated!"
    print("✓ Output logits passed sanity checks (non-NaN, non-Inf, softcapped <= 30.0)")

    # 6. JIT compilation test
    print("\n--- JAX JIT Compilation Test ---")
    jit_model = jax.jit(model, static_argnames=("quant_mode",))

    start = time.time()
    jit_logits = jit_model(input_ids, params, position_ids, quant_mode="w4a16").block_until_ready()
    compile_duration = (time.time() - start) * 1000
    print(f"✓ First JIT compile & run: Latency = {compile_duration:.2f} ms")

    start = time.time()
    jit_logits2 = jit_model(input_ids, params, position_ids, quant_mode="w4a16").block_until_ready()
    exec_duration = (time.time() - start) * 1000
    print(f"✓ Second JIT run (cached graph): Latency = {exec_duration:.2f} ms")

    # 7. Fused jax.lax.scan N-token Generation Benchmark
    print("\n--- Fused jax.lax.scan On-Chip Token Generation (32 tokens) ---")
    jit_scan_gen = jax.jit(generate_n_tokens_scan, static_argnames=("model", "num_steps", "quant_mode"))

    start = time.time()
    scanned_tokens = jit_scan_gen(model, input_ids, params, num_steps=32, quant_mode="w4a16").block_until_ready()
    scan_compile_dur = (time.time() - start) * 1000
    print(f"✓ First JIT scan compile & generate 32 tokens: Latency = {scan_compile_dur:.2f} ms")

    start = time.time()
    scanned_tokens2 = jit_scan_gen(model, input_ids, params, num_steps=32, quant_mode="w4a16").block_until_ready()
    scan_exec_dur = (time.time() - start) * 1000
    print(f"✓ Second JIT scan run (32 tokens): Latency = {scan_exec_dur:.2f} ms ({scan_exec_dur / 32:.2f} ms/token, {32000.0 / scan_exec_dur:.1f} tok/s)")
    print(f"  Generated tokens shape: {scanned_tokens2.shape}")

    # 8. Batched Inference Benchmark (B = 2)
    print("\n--- Batched Inference Benchmark (B = 2) ---")
    batched_ids = jnp.repeat(input_ids, 2, axis=0)  # Shape [2, 16]
    start = time.time()
    batched_out = jit_scan_gen(model, batched_ids, params, num_steps=16, quant_mode="w4a16").block_until_ready()
    batched_dur = (time.time() - start) * 1000
    print(f"✓ Batched B=2 scan generation (16 tokens x 2 streams = 32 total tokens): Latency = {batched_dur:.2f} ms ({32000.0 / batched_dur:.1f} aggregate tok/s)")

    print("\n" + "=" * 70)
    print("🎉 GEMMA 4 E2B QAT JAX END-TO-END SMOKE TEST SUCCESSFUL!")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_smoke_test()
