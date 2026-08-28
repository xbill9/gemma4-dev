"""Buffer donation must change scheduling and nothing else.

Without donation, `dynamic_update_slice` writes one token into the KV cache by
producing a whole NEW array, so every decode step reads the cache, writes a full
copy, then reads it again to attend. Measured on a v6e-1 at ctx 8192, B=32, that
copy costs 1.62x on a bf16 cache and 1.22x on int8 — the largest single
inefficiency found in this engine, and larger than every quantization knob
combined.

It is also the trap that hid it: every benchmark in this repo built its own
`jax.jit(make_cached_decode_step(...))` with no donation, so every step-time
ratio measured before 2026-07-29 was taken on the copying path. That is why
"int8 KV is 1.2-1.8x faster" shrank to ~1.18x once donation was enabled — int8
was partly being credited for halving the bytes of a copy that should not have
existed.

The risk donation introduces is aliasing: a donated buffer is INVALIDATED by the
call that consumes it. Code that keeps a reference to a cache it passed in, and
reads it afterwards, is wrong — and wrong silently. These tests pin the contract.

Run: python3 -m unittest tests.test_donation
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
    init_kv_cache,
    make_cached_decode_step,
    prefill_with_kv_cache,
)
from test_kv_cache_parity import DTYPE, build_tiny_params, tiny_config  # noqa: E402


class DonationParityTest(unittest.TestCase):
    """Donated and non-donated decode must produce identical tokens."""

    def setUp(self):
        self.config = tiny_config()
        self.model = Gemma4EModelJAX(self.config)
        self.params = build_tiny_params(self.config)
        self.prompt = jax.random.randint(jax.random.PRNGKey(5), (2, 8), 1,
                                         self.config.vocab_size)

    def _generate(self, donate: bool, cache_dtype=DTYPE, steps=6):
        B, S = self.prompt.shape
        valid_prompt = jnp.ones((B, S), dtype=jnp.bool_)
        last, caches, vmask = prefill_with_kv_cache(
            self.model, self.prompt, valid_prompt, self.params, steps,
            quant_mode="fp16", cache_dtype=cache_dtype, window_kv=False)
        step = jax.jit(
            make_cached_decode_step(self.model, quant_mode="fp16"),
            **({"donate_argnums": (1, 2)} if donate else {}))
        lens = jnp.full((B,), S, dtype=jnp.int32)
        tok = jnp.argmax(last, axis=-1, keepdims=True)
        out = [tok]
        for t in range(steps - 1):
            caches, vmask, last = step(self.params, caches, vmask, tok,
                                       lens + t, jnp.int32(S + t))
            tok = jnp.argmax(last, axis=-1, keepdims=True)
            out.append(tok)
        return jnp.concatenate(out, axis=1)

    def test_tokens_are_identical(self):
        plain = self._generate(donate=False)
        donated = self._generate(donate=True)
        self.assertTrue(bool(jnp.array_equal(plain, donated)),
                        f"donation changed output: {donated.tolist()} vs {plain.tolist()}")

    def test_identical_with_a_quantized_cache(self):
        """int8 carries scale buffers too; donation must not disturb them."""
        plain = self._generate(donate=False, cache_dtype=jnp.int8)
        donated = self._generate(donate=True, cache_dtype=jnp.int8)
        self.assertTrue(bool(jnp.array_equal(plain, donated)))


class DonationContractTest(unittest.TestCase):
    """The aliasing hazard donation introduces, pinned so it cannot regress."""

    def setUp(self):
        self.config = tiny_config()
        self.model = Gemma4EModelJAX(self.config)
        self.params = build_tiny_params(self.config)

    def _args(self, B=2, S=16):
        caches = init_kv_cache(self.config, B, S, DTYPE)
        valid = jnp.zeros((B, S), dtype=jnp.bool_).at[:, :4].set(True)
        tok = jnp.ones((B, 1), dtype=jnp.int32)
        lens = jnp.full((B,), 4, dtype=jnp.int32)
        return caches, valid, tok, lens

    def test_donated_buffers_are_invalidated(self):
        """A caller reusing the cache it donated must FAIL, not read stale data.

        This is the whole risk of the optimization. JAX deletes a donated buffer,
        so touching it afterwards raises — which is the good outcome. If this ever
        starts passing silently, donation has become capable of returning garbage.
        """
        step = jax.jit(make_cached_decode_step(self.model, quant_mode="fp16"),
                       donate_argnums=(1, 2))
        caches, valid, tok, lens = self._args()
        jax.block_until_ready(step(self.params, caches, valid, tok, lens, jnp.int32(4)))
        with self.assertRaises(Exception):
            # caches was donated on the call above; reading it now must not
            # quietly succeed against a recycled buffer.
            jax.block_until_ready(caches[0][0] + 0)

    def test_undonated_buffers_survive(self):
        """Without donation the same reuse is legal — the contract differs."""
        step = jax.jit(make_cached_decode_step(self.model, quant_mode="fp16"))
        caches, valid, tok, lens = self._args()
        jax.block_until_ready(step(self.params, caches, valid, tok, lens, jnp.int32(4)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(
            caches[0][0].astype(jnp.float32)))), "non-donated cache should stay valid")


class EngineDefaultTest(unittest.TestCase):
    def test_donation_is_on_by_default(self):
        """Measured 1.62x on a bf16 cache with token-identical output."""
        from jax_engine import JaxGemmaEngine
        self.assertTrue(JaxGemmaEngine("x").donate_cache)
        self.assertFalse(JaxGemmaEngine("x", donate_cache=False).donate_cache)


if __name__ == "__main__":
    unittest.main()
