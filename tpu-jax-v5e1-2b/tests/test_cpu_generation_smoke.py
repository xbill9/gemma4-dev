"""End-to-end generation on CPU, no TPU and no checkpoint download.

The real E2B checkpoint needs ~6.5 GB of parameters (the per-layer-embedding table
alone is 4.7 GB), so it will not fit on a typical dev box. But every *code path*
does: this builds a model with Gemma 4 E2B's real structure at small dimensions —
35-layer sliding/full pattern, KV sharing, per-layer embeddings, sandwich norms,
layer_scalar, W4A16-packed projections, windowed KV — and generates through the
same engine functions the TPU path uses.

Use this to check correctness changes before spending money on a chip.

Run: python3 -m unittest tests.test_cpu_generation_smoke
"""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ports.gemma4.jax_e_model import (  # noqa: E402
    Gemma4EConfig,
    Gemma4EModelJAX,
    generate_with_kv_cache,
    qat_w4a16_reference_linear_jax,
)


def small_e2b_config() -> Gemma4EConfig:
    """E2B's real structure, small dimensions.

    Keeps every structural property that has bitten us: the sliding/full layer
    pattern (full attention every 5th layer), KV sharing over the upper layers,
    per-layer embeddings, double-wide MLP on shared layers, and a sliding window
    shorter than the prompt.
    """
    return Gemma4EConfig(
        vocab_size=256, hidden_size=64, intermediate_size=96,
        num_hidden_layers=10, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16,
        num_global_key_value_heads=1, global_head_dim=32,
        num_kv_shared_layers=4, use_double_wide_mlp=True,
        # ple must be >= the W4A16 group size (32) or the scale tensor is empty
        hidden_size_per_layer_input=32, vocab_size_per_layer_input=256,
        sliding_window=8,
    )


def pack_w4a16(rng, out_f, in_f):
    """Produce compressed-tensors-style int4 weights: [out, in/8] int32 + scales."""
    q = rng.integers(0, 16, size=(out_f, in_f), dtype=np.int64)
    words = np.zeros((out_f, in_f // 8), dtype=np.int64)
    for i in range(8):
        words |= (q[:, i::8] & 0xF) << (4 * i)
    packed = jnp.asarray(words.astype(np.int32))
    scale = jnp.asarray(rng.uniform(0.01, 0.03, size=(out_f, in_f // 32)), dtype=jnp.bfloat16)
    return packed, scale


def build_quantized_params(cfg, seed=0):
    """Params shaped like the real QAT checkpoint: packed projections, real norms."""
    rng = np.random.default_rng(seed)
    H, L, ple = cfg.hidden_size, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input

    def dense(*shape):
        return jnp.asarray(rng.normal(0, 0.02, shape), dtype=jnp.bfloat16)

    def norm(n):
        return jnp.asarray(rng.uniform(0.5, 1.5, (n,)), dtype=jnp.bfloat16)

    params = {
        "embed_tokens": dense(cfg.vocab_size, H),
        "embed_tokens_per_layer": dense(cfg.vocab_size_per_layer_input, L * ple),
        "per_layer_projection_norm": norm(ple),
        "final_norm": norm(H),
    }
    params["per_layer_model_projection_packed"], params["per_layer_model_projection_scale"] = \
        pack_w4a16(rng, L * ple, H)

    for i in range(L):
        sliding = cfg.layer_types[i] == "sliding_attention"
        h_dim = cfg.head_dim if sliding else cfg.global_head_dim
        n_kv = cfg.num_key_value_heads if sliding else cfg.num_global_key_value_heads
        shared = i >= cfg.first_kv_shared_layer_idx
        inter = cfg.intermediate_size * 2 if (shared and cfg.use_double_wide_mlp) else cfg.intermediate_size

        attn = {"q_norm": norm(h_dim)}
        attn["q_proj_packed"], attn["q_proj_scale"] = pack_w4a16(rng, cfg.num_attention_heads * h_dim, H)
        attn["o_proj_packed"], attn["o_proj_scale"] = pack_w4a16(rng, H, cfg.num_attention_heads * h_dim)
        if not shared:
            attn["k_proj_packed"], attn["k_proj_scale"] = pack_w4a16(rng, n_kv * h_dim, H)
            attn["v_proj_packed"], attn["v_proj_scale"] = pack_w4a16(rng, n_kv * h_dim, H)
            attn["k_norm"] = norm(h_dim)

        mlp = {}
        mlp["gate_proj_packed"], mlp["gate_proj_scale"] = pack_w4a16(rng, inter, H)
        mlp["up_proj_packed"], mlp["up_proj_scale"] = pack_w4a16(rng, inter, H)
        mlp["down_proj_packed"], mlp["down_proj_scale"] = pack_w4a16(rng, H, inter)

        layer = {
            "input_layernorm": norm(H),
            "post_attention_layernorm": norm(H),
            "pre_feedforward_layernorm": norm(H),
            "post_feedforward_layernorm": norm(H),
            "post_per_layer_input_norm": norm(H),
            "layer_scalar": jnp.asarray([0.5], dtype=jnp.bfloat16),
            "attn": attn,
            "mlp": mlp,
        }
        layer["per_layer_input_gate_packed"], layer["per_layer_input_gate_scale"] = pack_w4a16(rng, ple, H)
        layer["per_layer_projection_packed"], layer["per_layer_projection_scale"] = pack_w4a16(rng, H, ple)
        params[f"layer_{i}"] = layer
    return params


class CPUGenerationSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_e2b_config()
        cls.model = Gemma4EModelJAX(cls.cfg)
        cls.params = build_quantized_params(cls.cfg)

    def test_runs_on_cpu(self):
        self.assertEqual(jax.devices()[0].platform, "cpu",
                         "this smoke test is meant to prove the CPU path")

    def test_w4a16_projections_are_actually_packed(self):
        """Guards that we exercise the quantized path, not a dense fallback."""
        l0 = self.params["layer_0"]
        self.assertIn("q_proj_packed", l0["attn"])
        self.assertIn("per_layer_input_gate_packed", l0)
        self.assertIn("per_layer_model_projection_packed", self.params)
        self.assertEqual(l0["attn"]["q_proj_packed"].dtype, jnp.int32)

    def test_generates_finite_tokens_in_range(self):
        prompt = jnp.asarray([[5, 9, 12, 3]], dtype=jnp.int32)
        out = generate_with_kv_cache(
            self.model, prompt, jnp.ones(prompt.shape, dtype=jnp.bool_),
            self.params, max_new_tokens=6, quant_mode="w4a16", temperature=0.0,
            window_kv=True,
        )
        self.assertEqual(out.shape, (1, 6))
        self.assertTrue(bool(jnp.all(out >= 0)) and bool(jnp.all(out < self.cfg.vocab_size)))

    def test_greedy_is_reproducible(self):
        prompt = jnp.asarray([[7, 1, 4]], dtype=jnp.int32)
        kw = dict(params=self.params, max_new_tokens=5, quant_mode="w4a16", temperature=0.0)
        a = generate_with_kv_cache(self.model, prompt, jnp.ones(prompt.shape, dtype=jnp.bool_), **kw)
        b = generate_with_kv_cache(self.model, prompt, jnp.ones(prompt.shape, dtype=jnp.bool_), **kw)
        self.assertEqual(a.tolist(), b.tolist())

    def test_windowed_and_full_kv_agree_with_a_long_prompt(self):
        """Prompt longer than sliding_window, so the ring buffer actually wraps."""
        prompt = jnp.asarray([list(range(1, 2 * self.cfg.sliding_window + 3))], dtype=jnp.int32)
        valid = jnp.ones(prompt.shape, dtype=jnp.bool_)
        kw = dict(params=self.params, max_new_tokens=4, quant_mode="w4a16", temperature=0.0)
        full = generate_with_kv_cache(self.model, prompt, valid, window_kv=False, **kw)
        win = generate_with_kv_cache(self.model, prompt, valid, window_kv=True, **kw)
        self.assertEqual(full.tolist(), win.tolist())

    def test_w4a16_reference_matches_dense_equivalent(self):
        """The quantized path must equal an explicit dequantize-then-matmul."""
        from ports.gemma4.jax_e_model import qat_w4a16_unpack_dequant_jax
        pk = self.params["layer_0"]["attn"]["q_proj_packed"]
        sc = self.params["layer_0"]["attn"]["q_proj_scale"]
        x = jnp.asarray(np.random.default_rng(1).normal(0, 1, (3, pk.shape[1] * 8)), dtype=jnp.bfloat16)
        got = qat_w4a16_reference_linear_jax(x, pk, sc)
        want = jnp.matmul(x, qat_w4a16_unpack_dequant_jax(pk, sc).T)
        self.assertEqual(float(jnp.max(jnp.abs(got.astype(jnp.float32) - want.astype(jnp.float32)))), 0.0)


if __name__ == "__main__":
    unittest.main()
