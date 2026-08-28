"""Chunked prefill must be a memory optimization, not a behaviour change.

The v6e-1 decode ceiling is a fixed budget of resident KV tokens (measured: a
flat 524,288 across four context lengths, 0.0% spread). The BATCH ceiling is a
different wall entirely — it is set by prefill, whose peak temporaries scale with
prompt length x batch, because attention scores alone are B x H x S x S. Chunking
caps the S in that product.

That only helps if a chunked prefill produces the same cache and the same logits
as the one-shot version, which is what these tests assert.

Run: python3 -m unittest tests.test_chunked_prefill
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
    Gemma4EModelJAX,
    chunked_prefill_with_kv_cache,
    make_cached_decode_step,
    make_chunk_mask,
    prefill_with_kv_cache,
)
from test_kv_cache_parity import DTYPE, build_tiny_params, tiny_config  # noqa: E402


class ChunkMaskTest(unittest.TestCase):
    """The mask is the whole difference between a chunk and a one-shot prefill."""

    def test_chunk_sees_history_and_itself_but_not_the_future(self):
        T, chunk, slot = 12, 4, 4
        valid = jnp.zeros((1, T), dtype=jnp.bool_).at[:, :slot + chunk].set(True)
        m = make_chunk_mask(valid, chunk, jnp.int32(slot))
        self.assertEqual(m.shape, (1, 1, chunk, T))
        allowed = (m[0, 0] == 0.0)
        for s in range(chunk):
            abs_pos = slot + s
            for t in range(T):
                want = t <= abs_pos and t < slot + chunk
                self.assertEqual(bool(allowed[s, t]), want,
                                 f"row s={s} (abs {abs_pos}) key t={t}")

    def test_window_clips_the_left_edge(self):
        T, chunk, slot, window = 16, 4, 8, 3
        valid = jnp.ones((1, T), dtype=jnp.bool_)
        m = make_chunk_mask(valid, chunk, jnp.int32(slot), window=window)
        allowed = (m[0, 0] == 0.0)
        for s in range(chunk):
            abs_pos = slot + s
            for t in range(T):
                self.assertEqual(bool(allowed[s, t]),
                                 (t <= abs_pos) and (t > abs_pos - window),
                                 f"s={s} t={t}")

    def test_padding_is_never_attended(self):
        T, chunk, slot = 10, 5, 0
        valid = jnp.zeros((1, T), dtype=jnp.bool_).at[:, :3].set(True)
        allowed = (make_chunk_mask(valid, chunk, jnp.int32(slot))[0, 0] == 0.0)
        self.assertFalse(bool(allowed[:, 3:].any()), "invalid slots must stay masked")


class ChunkedPrefillParityTest(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.model = Gemma4EModelJAX(self.config)
        self.params = build_tiny_params(self.config)

    def _prompt(self, S, B=2):
        ids = jax.random.randint(jax.random.PRNGKey(11), (B, S), 1, self.config.vocab_size)
        return ids, jnp.ones((B, S), dtype=jnp.bool_)

    def test_matches_one_shot_prefill_logits(self):
        S, new = 12, 4
        ids, valid = self._prompt(S)
        want, _, _ = prefill_with_kv_cache(
            self.model, ids, valid, self.params, new,
            quant_mode="fp16", cache_dtype=DTYPE, window_kv=False)
        for chunk in (2, 3, 4, 6, 12):
            with self.subTest(chunk=chunk):
                got, _, _ = chunked_prefill_with_kv_cache(
                    self.model, ids, valid, self.params, new,
                    chunk_size=chunk, quant_mode="fp16", cache_dtype=DTYPE)
                denom = float(jnp.max(jnp.abs(want.astype(jnp.float32)))) or 1.0
                rel = float(jnp.max(jnp.abs((got - want).astype(jnp.float32)))) / denom
                self.assertLess(rel, 2e-2, f"chunk={chunk} diverged: {rel:.4f}")

    def test_single_chunk_is_the_degenerate_case(self):
        """chunk_size == prompt length must reduce to a plain prefill."""
        S, new = 8, 3
        ids, valid = self._prompt(S)
        want, _, _ = prefill_with_kv_cache(
            self.model, ids, valid, self.params, new,
            quant_mode="fp16", cache_dtype=DTYPE, window_kv=False)
        got, _, _ = chunked_prefill_with_kv_cache(
            self.model, ids, valid, self.params, new,
            chunk_size=S, quant_mode="fp16", cache_dtype=DTYPE)
        denom = float(jnp.max(jnp.abs(want.astype(jnp.float32)))) or 1.0
        self.assertLess(
            float(jnp.max(jnp.abs((got - want).astype(jnp.float32)))) / denom, 2e-2)

    def test_generated_tokens_match(self):
        """The cache, not just the last logits, has to be right."""
        S, new = 12, 5
        ids, valid = self._prompt(S)

        def run(prefill_fn):
            last, caches, v = prefill_fn()
            step = jax.jit(make_cached_decode_step(self.model, quant_mode="fp16"))
            lens = jnp.full((ids.shape[0],), S, dtype=jnp.int32)
            tok = jnp.argmax(last, axis=-1, keepdims=True)
            out = [tok]
            for t in range(new - 1):
                caches, v, last = step(self.params, caches, v, tok, lens + t,
                                       jnp.int32(S + t))
                tok = jnp.argmax(last, axis=-1, keepdims=True)
                out.append(tok)
            return jnp.concatenate(out, axis=1)

        want = run(lambda: prefill_with_kv_cache(
            self.model, ids, valid, self.params, new,
            quant_mode="fp16", cache_dtype=DTYPE, window_kv=False))
        got = run(lambda: chunked_prefill_with_kv_cache(
            self.model, ids, valid, self.params, new,
            chunk_size=4, quant_mode="fp16", cache_dtype=DTYPE))
        self.assertTrue(bool(jnp.array_equal(got, want)),
                        f"tokens differ: chunked {got.tolist()} vs one-shot {want.tolist()}")

    def test_rejects_a_prompt_that_is_not_a_multiple_of_the_chunk(self):
        ids, valid = self._prompt(10)
        with self.assertRaises(ValueError):
            chunked_prefill_with_kv_cache(
                self.model, ids, valid, self.params, 2,
                chunk_size=4, quant_mode="fp16", cache_dtype=DTYPE)

    def test_works_with_a_quantized_cache(self):
        """Chunking and int8 KV are independent; they must compose."""
        S, new = 12, 4
        ids, valid = self._prompt(S)
        want, _, _ = chunked_prefill_with_kv_cache(
            self.model, ids, valid, self.params, new,
            chunk_size=4, quant_mode="fp16", cache_dtype=DTYPE)
        got, caches, _ = chunked_prefill_with_kv_cache(
            self.model, ids, valid, self.params, new,
            chunk_size=4, quant_mode="fp16", cache_dtype=jnp.int8)
        for entry in caches.values():
            self.assertEqual(len(entry), 4, "int8 entries must carry scales")
        denom = float(jnp.max(jnp.abs(want.astype(jnp.float32)))) or 1.0
        rel = float(jnp.max(jnp.abs((got - want).astype(jnp.float32)))) / denom
        self.assertLess(rel, 0.25, f"int8 chunked prefill drifted: {rel:.4f}")


if __name__ == "__main__":
    unittest.main()
