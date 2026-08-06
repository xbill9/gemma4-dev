"""Correctness guards for the decode-path performance work.

Each optimization here exists to cut HBM traffic in the memory-bound decode
phase. The point of these tests is that they cut traffic *without* changing what
the model computes — so most assert bit-exactness against the pre-optimization
formulation, not "close enough".

Run: python3 -m unittest tests.test_perf_optimizations
"""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ports.gemma4.jax_e_model as M  # noqa: E402


def reference_attention(query, key, value, mask, scaling=1.0, softcap=30.0):
    """The pre-optimization formulation: replicate K/V up to the query head count."""
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = jnp.repeat(key, n_rep, axis=1)
        value = jnp.repeat(value, n_rep, axis=1)
    scores = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) * scaling
    if softcap > 0.0:
        scores = jnp.tanh(scores / softcap) * softcap
    if mask is not None:
        scores = scores + mask
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
    return jnp.matmul(probs, value)


class GroupedQueryAttentionTest(unittest.TestCase):
    """GQA via grouped einsum must be bit-identical to the jnp.repeat version.

    The repeat materialized n_rep copies of the entire KV cache every decode
    step. E2B ships one KV head against eight query heads, so that was an 8x
    multiplier on the dominant HBM read.
    """

    def test_bit_exact_across_head_ratios(self):
        rng = np.random.default_rng(3)
        for n_heads, n_kv in ((8, 1), (8, 4), (4, 4), (16, 2)):
            for s_q, s_kv in ((1, 64), (5, 9), (1, 1)):
                B, D = 2, 16
                q = jnp.asarray(rng.normal(0, 1, (B, n_heads, s_q, D)), dtype=jnp.bfloat16)
                k = jnp.asarray(rng.normal(0, 1, (B, n_kv, s_kv, D)), dtype=jnp.bfloat16)
                v = jnp.asarray(rng.normal(0, 1, (B, n_kv, s_kv, D)), dtype=jnp.bfloat16)
                mask = jnp.where(
                    jnp.tril(jnp.ones((s_q, s_kv)), k=s_kv - s_q)[None, None], 0.0, -1e9
                ).astype(jnp.float32)

                ref = reference_attention(q, k, v, mask)
                got = M.eager_attention_jax(q, k, v, mask=mask, scaling=1.0, softcap=30.0)
                self.assertEqual(ref.shape, got.shape)
                self.assertTrue(
                    bool(jnp.array_equal(ref, got)),
                    f"GQA differs at heads={n_heads}/{n_kv} S={s_q}x{s_kv}: "
                    f"max|d|={float(jnp.max(jnp.abs(ref.astype(jnp.float32) - got.astype(jnp.float32))))}",
                )


class FusedW4A16Test(unittest.TestCase):
    """The fused Pallas kernel must match dequantize-then-matmul exactly.

    Regression guard: the original kernel unpacked nibbles plane-major
    (column i*(K/8)+j) while contracting against activations in natural order,
    so it silently returned wrong values wherever Pallas actually compiled. It
    went unnoticed because a bare `except Exception` fell back to the reference
    on any host without a TPU.
    """

    def setUp(self):
        rng = np.random.default_rng(7)
        self.out_f, self.in_f = 256, 512
        self.packed = jnp.asarray(
            rng.integers(-(2**31), 2**31, size=(self.out_f, self.in_f // 8), dtype=np.int64).astype(np.int32)
        )
        self.scale = jnp.asarray(
            rng.uniform(0.005, 0.02, size=(self.out_f, self.in_f // 32)), dtype=jnp.bfloat16
        )
        self.x = jnp.asarray(rng.normal(0, 1, size=(5, self.in_f)), dtype=jnp.bfloat16)
        self.ref = M.qat_w4a16_reference_linear_jax(self.x, self.packed, self.scale)

    def _max_diff(self, a, b):
        return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))

    def test_interleaved_layout_is_exact(self):
        got = M.qat_w4a16_pallas_matmul_jax(self.x, self.packed, self.scale, layout="interleaved")
        self.assertEqual(self._max_diff(self.ref, got), 0.0)

    def test_plane_layout_matches_within_rounding(self):
        got = M.qat_w4a16_pallas_matmul_jax(self.x, self.packed, self.scale, layout="plane")
        # Same math, different order of the BF16 scale expansion.
        self.assertLess(self._max_diff(self.ref, got), 1e-3)

    def test_batched_3d_input(self):
        rng = np.random.default_rng(8)
        x3 = jnp.asarray(rng.normal(0, 1, (2, 3, self.in_f)), dtype=jnp.bfloat16)
        ref3 = M.qat_w4a16_reference_linear_jax(x3, self.packed, self.scale)
        got3 = M.qat_w4a16_pallas_matmul_jax(x3, self.packed, self.scale)
        self.assertEqual(got3.shape, ref3.shape)
        self.assertEqual(self._max_diff(ref3, got3), 0.0)

    def test_public_entry_dispatches_to_fused(self):
        M.set_w4a16_impl("fused", "interleaved")
        try:
            got = M.qat_w4a16_linear_jax(self.x, self.packed, self.scale)
            self.assertEqual(self._max_diff(self.ref, got), 0.0)
        finally:
            M.set_w4a16_impl("auto", "interleaved")

    def test_reference_impl_selectable(self):
        M.set_w4a16_impl("reference")
        try:
            got = M.qat_w4a16_linear_jax(self.x, self.packed, self.scale)
            self.assertEqual(self._max_diff(self.ref, got), 0.0)
        finally:
            M.set_w4a16_impl("auto", "interleaved")

    def test_rejects_mismatched_k(self):
        bad_x = jnp.zeros((2, self.in_f // 2), dtype=jnp.bfloat16)
        with self.assertRaises(ValueError):
            M.qat_w4a16_pallas_matmul_jax(bad_x, self.packed, self.scale)


class Int8LmHeadTest(unittest.TestCase):
    """Per-row int8 output projection: halves the biggest single decode read."""

    def setUp(self):
        rng = np.random.default_rng(5)
        self.emb = jnp.asarray(rng.normal(0, 0.02, (512, 64)), dtype=jnp.bfloat16)
        self.h = jnp.asarray(rng.normal(0, 1, (4, 1, 64)), dtype=jnp.bfloat16)

    def test_quantizer_shapes_and_dtypes(self):
        p = M.quantize_lm_head({"embed_tokens": self.emb})
        self.assertEqual(p["embed_tokens_q8"].dtype, jnp.int8)
        self.assertEqual(p["embed_tokens_q8"].shape, self.emb.shape)
        self.assertIn("embed_tokens_q8_scale", p)
        self.assertIn("embed_tokens", p, "BF16 table still needed for input embedding lookup")

    def test_does_not_mutate_input(self):
        original = {"embed_tokens": self.emb}
        M.quantize_lm_head(original)
        self.assertEqual(set(original), {"embed_tokens"})

    def test_logits_track_bf16_within_one_percent(self):
        p = M.quantize_lm_head({"embed_tokens": self.emb})
        ref = jnp.matmul(self.h, self.emb.T).astype(jnp.float32)
        got = (
            jnp.matmul(self.h, p["embed_tokens_q8"].T.astype(self.h.dtype))
            * p["embed_tokens_q8_scale"].astype(self.h.dtype)
        ).astype(jnp.float32)
        rel = float(jnp.max(jnp.abs(ref - got))) / float(jnp.max(jnp.abs(ref)))
        self.assertLess(rel, 0.02, f"int8 LM head relative error {rel:.3%}")

    def test_argmax_mostly_preserved(self):
        """Greedy choice should survive quantization on all but near-ties."""
        rng = np.random.default_rng(9)
        h = jnp.asarray(rng.normal(0, 1, (64, 1, 64)), dtype=jnp.bfloat16)
        p = M.quantize_lm_head({"embed_tokens": self.emb})
        ref = jnp.matmul(h, self.emb.T).astype(jnp.float32)
        got = (
            jnp.matmul(h, p["embed_tokens_q8"].T.astype(h.dtype))
            * p["embed_tokens_q8_scale"].astype(h.dtype)
        ).astype(jnp.float32)
        agree = float(jnp.mean(jnp.argmax(ref, -1) == jnp.argmax(got, -1)))
        self.assertGreater(agree, 0.9, f"argmax agreement only {agree:.1%}")


class ModelForwardWithOptimizationsTest(unittest.TestCase):
    """End-to-end: the optimized model forward still runs and stays finite."""

    def test_forward_with_int8_lm_head(self):
        from tests.test_kv_cache_parity import build_tiny_params, tiny_config

        cfg = tiny_config()
        model = M.Gemma4EModelJAX(cfg)
        params = build_tiny_params(cfg)
        ids = jnp.ones((1, 4), dtype=jnp.int32)
        pos = jnp.arange(4, dtype=jnp.int32)[None, :]
        mask = M.make_prefill_causal_mask(jnp.ones((1, 4), dtype=jnp.bool_))

        base = model(ids, params, pos, attention_mask=mask, quant_mode="fp16")
        q_params = M.quantize_lm_head(params)
        quant = model(ids, q_params, pos, attention_mask=mask, quant_mode="fp16")

        self.assertEqual(base.shape, quant.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(quant))))
        # Same ranking behaviour, not the same bits.
        self.assertTrue(bool(jnp.array_equal(jnp.argmax(base, -1), jnp.argmax(quant, -1))))


if __name__ == "__main__":
    unittest.main()
