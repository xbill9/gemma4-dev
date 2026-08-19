"""Unit & Parity Tests for Gemma 4 E2B JAX / Flax Model (ports/gemma4/jax_e_model.py)."""

import math
import sys
import unittest
import jax
import jax.numpy as jnp

from ports.gemma4.jax_e_model import (
    Gemma4EConfig,
    Gemma4EModelJAX,
    qat_w4a16_unpack_dequant_jax,
    rms_norm_jax,
)


class TestGemma4EJAX(unittest.TestCase):

    def setUp(self):
        # Small test configuration matching Gemma 4 E2B MatFormer structure
        self.config = Gemma4EConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=5,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_global_key_value_heads=2,
            global_head_dim=64,
            num_kv_shared_layers=2,  # Last 2 layers share KV
            use_double_wide_mlp=True,
            hidden_size_per_layer_input=32,
            vocab_size_per_layer_input=1000,
            layer_types=["sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        )

    def test_kv_share_map(self):
        share_map = self.config.kv_share_map()
        self.assertEqual(len(share_map), 5)
        # Layers 0, 1, 2 are non-shared
        self.assertEqual(share_map[:3], [0, 1, 2])
        # Layer 3 (sliding) maps to last non-shared sliding layer (layer 2)
        self.assertEqual(share_map[3], 2)
        # Layer 4 (full) maps to last non-shared full layer (layer 1)
        self.assertEqual(share_map[4], 1)

    def test_w4a16_unpack_dequant(self):
        out_features, in_features = 8, 32
        group_size = 8
        # Nibbles 0..7 encode q=-8..-1 in every int32 word.
        word = sum(i << (4 * i) for i in range(8))
        packed = jnp.full(
            (out_features, in_features // 8), word, dtype=jnp.int32
        )
        scale = jnp.ones(
            (out_features, in_features // group_size), dtype=jnp.bfloat16
        )

        unpacked = qat_w4a16_unpack_dequant_jax(packed, scale, group_size=group_size)
        self.assertEqual(unpacked.shape, (out_features, in_features))
        self.assertEqual(unpacked.dtype, jnp.bfloat16)
        self.assertAlmostEqual(float(unpacked[0, 0]), -8.0)
        self.assertAlmostEqual(float(unpacked[0, 7]), -1.0)

    def test_model_forward_fp16(self):
        model = Gemma4EModelJAX(self.config)
        
        # Build mock parameters
        key = jax.random.PRNGKey(42)
        params = {
            "embed_tokens": jax.random.normal(key, (self.config.vocab_size, self.config.hidden_size), dtype=jnp.bfloat16),
            "embed_tokens_per_layer": jax.random.normal(key, (self.config.vocab_size_per_layer_input, self.config.num_hidden_layers * self.config.hidden_size_per_layer_input), dtype=jnp.bfloat16),
            "per_layer_model_projection": jax.random.normal(key, (self.config.hidden_size, self.config.num_hidden_layers * self.config.hidden_size_per_layer_input), dtype=jnp.bfloat16),
            "final_norm": jnp.ones((self.config.hidden_size,), dtype=jnp.bfloat16),
        }

        for i in range(self.config.num_hidden_layers):
            is_sliding = self.config.layer_types[i] == "sliding_attention"
            h_dim = self.config.head_dim if is_sliding else self.config.global_head_dim
            num_kv = self.config.num_key_value_heads if is_sliding else self.config.num_global_key_value_heads
            is_shared = i >= self.config.first_kv_shared_layer_idx
            inter_size = self.config.intermediate_size * 2 if (is_shared and self.config.use_double_wide_mlp) else self.config.intermediate_size

            layer_params = {
                "input_layernorm": jnp.ones((self.config.hidden_size,), dtype=jnp.bfloat16),
                "post_attention_layernorm": jnp.ones((self.config.hidden_size,), dtype=jnp.bfloat16),
                "per_layer_input_gate": jax.random.normal(key, (self.config.hidden_size, self.config.hidden_size_per_layer_input), dtype=jnp.bfloat16),
                "per_layer_projection": jax.random.normal(key, (self.config.hidden_size_per_layer_input, self.config.hidden_size), dtype=jnp.bfloat16),
                "attn": {
                    "q_proj": jax.random.normal(key, (self.config.hidden_size, self.config.num_attention_heads * h_dim), dtype=jnp.bfloat16),
                    "o_proj": jax.random.normal(key, (self.config.num_attention_heads * h_dim, self.config.hidden_size), dtype=jnp.bfloat16),
                },
                "mlp": {
                    "gate_proj": jax.random.normal(key, (self.config.hidden_size, inter_size), dtype=jnp.bfloat16),
                    "up_proj": jax.random.normal(key, (self.config.hidden_size, inter_size), dtype=jnp.bfloat16),
                    "down_proj": jax.random.normal(key, (inter_size, self.config.hidden_size), dtype=jnp.bfloat16),
                }
            }

            if not is_shared:
                layer_params["attn"]["k_proj"] = jax.random.normal(key, (self.config.hidden_size, num_kv * h_dim), dtype=jnp.bfloat16)
                layer_params["attn"]["v_proj"] = jax.random.normal(key, (self.config.hidden_size, num_kv * h_dim), dtype=jnp.bfloat16)

            params[f"layer_{i}"] = layer_params

        input_ids = jnp.array([[10, 20, 30]], dtype=jnp.int32)
        position_ids = jnp.array([[0, 1, 2]], dtype=jnp.int32)

        # Test uncompiled execution
        logits = model(input_ids, params, position_ids, quant_mode="fp16")
        self.assertEqual(logits.shape, (1, 3, self.config.vocab_size))

        # Test JIT compilation with static_argnames
        jit_forward = jax.jit(model, static_argnames=("quant_mode",))
        jit_logits = jit_forward(input_ids, params, position_ids, quant_mode="fp16")
        self.assertEqual(jit_logits.shape, (1, 3, self.config.vocab_size))

    def test_loader_mapping(self):
        from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params
        mock_raw = {
            "model.embed_tokens.weight": jnp.ones((100, 32), dtype=jnp.bfloat16),
            "model.norm.weight": jnp.ones((32,), dtype=jnp.bfloat16),
            "model.layers.0.input_layernorm.weight": jnp.ones((32,), dtype=jnp.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": jnp.ones((64, 32), dtype=jnp.bfloat16),
            "model.layers.0.mlp.gate_proj.weight": jnp.ones((64, 32), dtype=jnp.bfloat16),
        }
        params = convert_safetensors_to_jax_params(mock_raw, num_layers=1, first_kv_shared_idx=1)
        self.assertIn("embed_tokens", params)
        self.assertIn("layer_0", params)
        self.assertIn("q_proj", params["layer_0"]["attn"])
        self.assertEqual(params["layer_0"]["attn"]["q_proj"].shape, (32, 64))

    def test_loader_native_compressed_tensors_layout(self):
        from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params
        mock_raw = {
            "model.embed_tokens.weight": jnp.ones((100, 32), dtype=jnp.bfloat16),
            "model.norm.weight": jnp.ones((32,), dtype=jnp.bfloat16),
            "model.layers.0.input_layernorm.weight": jnp.ones((32,), dtype=jnp.bfloat16),
            "model.layers.0.post_attention_layernorm.weight": jnp.ones((32,), dtype=jnp.bfloat16),
            "model.layers.0.self_attn.q_proj.weight_packed": jnp.zeros((64, 4), dtype=jnp.int32),
            "model.layers.0.self_attn.q_proj.weight_scale": jnp.ones((64, 1), dtype=jnp.bfloat16),
        }
        params = convert_safetensors_to_jax_params(
            mock_raw, num_layers=1, first_kv_shared_idx=1
        )
        q = params["layer_0"]["attn"]
        self.assertEqual(q["q_proj_packed"].shape, (64, 4))
        self.assertEqual(q["q_proj_scale"].shape, (64, 1))
        self.assertEqual(q["q_proj_packed"].dtype, jnp.int32)


if __name__ == "__main__":
    unittest.main()
