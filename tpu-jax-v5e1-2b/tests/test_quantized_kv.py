"""An 8-bit KV cache must actually be 8 bits, and must actually be correct.

Two bugs motivated these tests, and they fail in opposite directions:

  * fp8 raised `no available implicit dtype promotion path` the moment a cached
    key met a bf16 query. JAX keeps float8 out of its promotion lattice on
    purpose, so the cast has to be written explicitly. Loud failure.
  * int8 did NOT raise. int8 IS in the lattice, so `bf16 x int8` silently
    contracted against raw integers with no scale applied anywhere, and the
    benchmark reported a perfectly good step time for arithmetic that would have
    produced garbage. Silent failure, which is worse.

So the assertions here are about numerics, not just about not throwing.

Run: python3 -m unittest tests.test_quantized_kv
"""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ports.gemma4.jax_e_model import (  # noqa: E402
    Gemma4EConfig,
    Gemma4EModelJAX,
    eager_attention_jax,
    init_kv_cache,
    is_quantized_kv_dtype,
    make_cached_decode_step,
    prefill_with_kv_cache,
    quantize_kv,
)
# Reuse the parity suite's config: 10 layers -> [s,s,s,s,f,s,s,s,s,f] with
# first_kv_shared_layer_idx=6, so the non-shared prefix that actually owns cache
# buffers contains both a sliding and a full-attention layer, and the shared
# suffix exercises inheriting a quantized source layer's scales.
from test_kv_cache_parity import build_tiny_params, tiny_config  # noqa: E402

QUANT_DTYPES = [jnp.int8, jnp.float8_e4m3fn, jnp.float8_e5m2]


class QuantizeKVTest(unittest.TestCase):
    """The scale itself: range, degenerate rows, round-trip error."""

    def setUp(self):
        self.x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 17, 16)) * 4.0

    def test_dtype_classification(self):
        for dt in QUANT_DTYPES:
            self.assertTrue(is_quantized_kv_dtype(dt), f"{dt} needs a scale")
        for dt in (jnp.bfloat16, jnp.float16, jnp.float32):
            self.assertFalse(is_quantized_kv_dtype(dt))

    def test_round_trip_is_close(self):
        # int8 with a per-row scale resolves ~1/127 of each row's max; the fp8
        # formats trade mantissa for exponent, so e5m2 is expected to be coarsest.
        tol = {jnp.int8: 0.02, jnp.float8_e4m3fn: 0.08, jnp.float8_e5m2: 0.30}
        for dt in QUANT_DTYPES:
            q, s = quantize_kv(self.x, dt)
            self.assertEqual(q.dtype, jnp.dtype(dt))
            self.assertEqual(s.shape, self.x.shape[:-1] + (1,))
            back = q.astype(jnp.float32) * s.astype(jnp.float32)
            rel = jnp.max(jnp.abs(back - self.x)) / jnp.max(jnp.abs(self.x))
            self.assertLess(float(rel), tol[dt], f"{dt.__name__} round-trip too lossy")

    def test_zero_rows_do_not_produce_a_zero_or_nan_scale(self):
        # Unwritten cache slots are all-zero. amax/qmax would be a 0 scale.
        zeros = jnp.zeros((1, 1, 4, 16))
        for dt in QUANT_DTYPES:
            q, s = quantize_kv(zeros, dt)
            self.assertTrue(bool(jnp.all(jnp.isfinite(s.astype(jnp.float32)))))
            self.assertTrue(bool(jnp.all(s.astype(jnp.float32) > 0)))
            self.assertTrue(bool(jnp.all(jnp.isfinite(q.astype(jnp.float32)))))


class ScaleFactorizationTest(unittest.TestCase):
    """Applying scales to the contraction result must equal dequantizing first.

    This is the whole reason the cache stays 1 byte wide on the wire: the K scale
    is indexed by key position while the score sums over head_dim, so it factors
    out; the V scale shares an index with the summed axis, so it folds into the
    probabilities. If either rewrite were wrong the cache would be cheap and wrong.
    """

    def _case(self, num_heads, num_kv_heads, dt):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
        B, S, S_kv, D = 2, 3, 11, 16
        q = jax.random.normal(k1, (B, num_heads, S, D), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (B, num_kv_heads, S_kv, D)) * 3.0
        v = jax.random.normal(k3, (B, num_kv_heads, S_kv, D)) * 3.0

        kq, ks = quantize_kv(k, dt)
        vq, vs = quantize_kv(v, dt)

        # Reference: widen the cache, apply scales to K/V themselves.
        k_deq = (kq.astype(jnp.float32) * ks.astype(jnp.float32)).astype(jnp.bfloat16)
        v_deq = (vq.astype(jnp.float32) * vs.astype(jnp.float32)).astype(jnp.bfloat16)
        want = eager_attention_jax(q, k_deq, v_deq, softcap=0.0)

        # Under test: scales applied to the contraction results instead.
        got = eager_attention_jax(q, kq, vq, softcap=0.0, key_scale=ks, value_scale=vs)

        self.assertEqual(got.dtype, want.dtype)
        denom = float(jnp.max(jnp.abs(want.astype(jnp.float32)))) or 1.0
        rel = float(jnp.max(jnp.abs((got - want).astype(jnp.float32)))) / denom
        self.assertLess(rel, 5e-2, f"factorization diverged (heads={num_heads}/"
                                   f"{num_kv_heads}, {dt.__name__}): rel={rel:.4f}")

    def test_multi_query_grouped(self):
        for dt in QUANT_DTYPES:
            self._case(num_heads=8, num_kv_heads=2, dt=dt)   # n_rep = 4

    def test_single_kv_head(self):
        for dt in QUANT_DTYPES:
            self._case(num_heads=8, num_kv_heads=1, dt=dt)   # n_rep = 8, E2B's shape

    def test_no_grouping(self):
        for dt in QUANT_DTYPES:
            self._case(num_heads=4, num_kv_heads=4, dt=dt)   # n_rep = 1


class CacheLayoutTest(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()

    def test_quantized_entries_carry_scales_and_bf16_does_not(self):
        plain = init_kv_cache(self.config, 2, 32, jnp.bfloat16)
        for entry in plain.values():
            self.assertEqual(len(entry), 2, "bf16 entries must stay 2-tuples")

        for dt in QUANT_DTYPES:
            quant = init_kv_cache(self.config, 2, 32, dt)
            for i, entry in quant.items():
                self.assertEqual(len(entry), 4, f"{dt.__name__} layer {i}")
                k, v, ks, vs = entry
                self.assertEqual(k.dtype, jnp.dtype(dt))
                self.assertEqual(ks.shape, k.shape[:-1] + (1,))
                self.assertEqual(vs.shape, v.shape[:-1] + (1,))
                # Zero-initialized scales would poison the first masked softmax.
                self.assertTrue(bool(jnp.all(ks.astype(jnp.float32) > 0)))

    def test_int8_cache_is_about_half_of_bfloat16(self):
        def nbytes(c):
            return sum(a.size * a.dtype.itemsize for entry in c.values() for a in entry)
        # head_dim 16 here, so the per-row scale overhead is an exaggerated 1/16;
        # at E2B's head_dim 256 it is 0.8%.
        ratio = nbytes(init_kv_cache(self.config, 2, 64, jnp.int8)) / \
            nbytes(init_kv_cache(self.config, 2, 64, jnp.bfloat16))
        self.assertLess(ratio, 0.60)
        self.assertGreater(ratio, 0.45)


class QuantizedDecodeTest(unittest.TestCase):
    """End to end: the model must run, and agree with a bf16 cache."""

    def setUp(self):
        self.config = tiny_config()
        self.model = Gemma4EModelJAX(self.config)
        self.params = build_tiny_params(self.config)
        self.prompt = jax.random.randint(jax.random.PRNGKey(7), (2, 6), 1,
                                         self.config.vocab_size)

    def _decode_logits(self, cache_dtype, steps=4, forced=None):
        """Logits from `steps` cached decode steps.

        `forced` teacher-forces the fed tokens. Without it each dtype picks its
        own argmax, and a single flipped token makes every later logit
        incomparable — that measures sampling divergence, not cache accuracy.
        """
        B, S = self.prompt.shape
        valid_prompt = jnp.ones((B, S), dtype=jnp.bool_)
        last, caches, valid = prefill_with_kv_cache(
            self.model, self.prompt, valid_prompt, self.params, steps,
            quant_mode="fp16", cache_dtype=cache_dtype,
        )
        step = jax.jit(make_cached_decode_step(self.model, quant_mode="fp16"))
        lens = jnp.full((B,), S, dtype=jnp.int32)
        tok = jnp.argmax(last, axis=-1, keepdims=True)
        out, fed = [], []
        for t in range(steps - 1):
            if forced is not None:
                tok = forced[t]
            fed.append(tok)
            caches, valid, last = step(self.params, caches, valid, tok,
                                       lens + t, jnp.int32(S + t))
            out.append(last)
            tok = jnp.argmax(last, axis=-1, keepdims=True)
        return out, fed

    def test_fp8_no_longer_raises_on_promotion(self):
        # The original bug, verbatim: a cached fp8 key meeting a bf16 query.
        for dt in (jnp.float8_e4m3fn, jnp.float8_e5m2):
            with self.subTest(dtype=dt.__name__):
                self._decode_logits(dt)  # must not raise

    def test_quantized_decode_tracks_bfloat16(self):
        ref, fed = self._decode_logits(jnp.bfloat16)
        # REGRESSION BOUNDS, NOT QUALITY BOUNDS. This is a random-weight tiny
        # model whose logits span only ~±1.1, so a "relative" error is taken
        # against a nearly uniform distribution and reads far worse than the same
        # cache does on a trained checkpoint. The numbers below are ~2x the
        # observed values (int8 0.086 / e4m3 0.334 / e5m2 0.584 at the first decode
        # step, falling on later steps — the error does not compound). Judge real
        # accuracy on the real checkpoint, not here.
        tol = {jnp.int8: 0.17, jnp.float8_e4m3fn: 0.67, jnp.float8_e5m2: 1.17}
        for dt in QUANT_DTYPES:
            with self.subTest(dtype=dt.__name__):
                got, _ = self._decode_logits(dt, forced=fed)
                scale = float(jnp.max(jnp.abs(ref[0].astype(jnp.float32)))) or 1.0
                worst = max(
                    float(jnp.max(jnp.abs((g - r).astype(jnp.float32)))) / scale
                    for g, r in zip(got, ref))
                self.assertLess(worst, tol[dt], f"{dt.__name__} drifted: {worst:.4f}")

    def test_int8_is_more_accurate_than_the_fp8_formats(self):
        """Ranking, not absolute error — a scale bug would shuffle this order."""
        ref, fed = self._decode_logits(jnp.bfloat16)

        def err(dt):
            got, _ = self._decode_logits(dt, forced=fed)
            return max(float(jnp.max(jnp.abs((g - r).astype(jnp.float32))))
                       for g, r in zip(got, ref))

        self.assertLess(err(jnp.int8), err(jnp.float8_e4m3fn))
        self.assertLess(err(jnp.float8_e4m3fn), err(jnp.float8_e5m2))

    def test_raw_int8_without_scales_would_have_been_wrong(self):
        """Guards the silent-failure mode: no-scale int8 must NOT look correct.

        If this ever starts passing at bf16-like accuracy, the scales have been
        dropped somewhere and the earlier bug is back.
        """
        q = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 2, 16), dtype=jnp.bfloat16)
        k = jax.random.normal(jax.random.PRNGKey(2), (1, 2, 5, 16)) * 4.0
        v = jax.random.normal(jax.random.PRNGKey(3), (1, 2, 5, 16)) * 4.0
        kq, ks = quantize_kv(k, jnp.int8)
        vq, vs = quantize_kv(v, jnp.int8)

        want = eager_attention_jax(q, k.astype(jnp.bfloat16), v.astype(jnp.bfloat16),
                                   softcap=0.0)
        scaled = eager_attention_jax(q, kq, vq, softcap=0.0,
                                     key_scale=ks, value_scale=vs)
        unscaled = eager_attention_jax(q, kq, vq, softcap=0.0)  # the old silent path

        denom = float(jnp.max(jnp.abs(want.astype(jnp.float32))))
        err_scaled = float(jnp.max(jnp.abs((scaled - want).astype(jnp.float32)))) / denom
        err_unscaled = float(jnp.max(jnp.abs((unscaled - want).astype(jnp.float32)))) / denom
        self.assertLess(err_scaled, 0.10)
        self.assertGreater(err_unscaled, err_scaled * 3,
                           "unscaled int8 should be visibly wrong; scales look inert")


if __name__ == "__main__":
    unittest.main()
