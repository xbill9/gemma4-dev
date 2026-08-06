"""Windowed KV cache must be a memory optimization, not a behaviour change.

Sliding-attention layers can never attend outside their window, so allocating
`max_seq_len` slots for them is dead memory. These tests assert the ring-buffer
version produces the SAME tokens as the full-length cache, including when the
prompt is longer than the window (the case the ring exists for).

Run: python3 -m unittest tests.test_windowed_kv
"""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ports.gemma4.jax_e_model import (  # noqa: E402
    Gemma4EConfig,
    Gemma4EModelJAX,
    init_kv_cache,
    make_cached_decode_step,
    make_prefill_causal_mask,
    prefill_with_kv_cache,
)
from test_kv_cache_parity import DTYPE, build_tiny_params  # noqa: E402

WINDOW = 4


def windowed_config() -> Gemma4EConfig:
    """Tiny config with a window small enough that prompts exceed it."""
    return Gemma4EConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96,
        num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_global_key_value_heads=2, global_head_dim=32,
        num_kv_shared_layers=4, use_double_wide_mlp=True,
        hidden_size_per_layer_input=16, vocab_size_per_layer_input=128,
        sliding_window=WINDOW,
    )


def generate(model, params, prompt, n, window_kv, real_len=None):
    """Generate n tokens.

    real_len: if set, the prompt is right-padded and only its first `real_len`
      tokens are real -- which is what the server always does, since it pads to a
      static bucket. Decode then writes at bucket_s + step, so the pad slots
      [real_len, S) fall *inside* [0, slot) and a mask that assumes a contiguous
      fill will attend to them.
    """
    B, S = prompt.shape
    if real_len is None:
        valid_prompt = jnp.ones((B, S), dtype=jnp.bool_)
    else:
        valid_prompt = jnp.arange(S)[None, :] < real_len
    last, caches, valid = prefill_with_kv_cache(
        model, prompt, valid_prompt, params, n,
        quant_mode="fp16", cache_dtype=DTYPE, window_kv=window_kv,
    )
    step = jax.jit(make_cached_decode_step(model, quant_mode="fp16", window_kv=window_kv))
    # Logical position tracks the REAL length; the slot tracks the padded bucket.
    # This mirrors jax_engine.py:434 exactly -- that split is what opens the gap.
    lens = jnp.full((B,), S if real_len is None else real_len, dtype=jnp.int32)
    tok = jnp.argmax(last, axis=-1, keepdims=True)
    out, logits = [tok], [last]
    for t in range(n - 1):
        caches, valid, last = step(params, caches, valid, tok, lens + t, jnp.int32(S + t))
        logits.append(last)
        tok = jnp.argmax(last, axis=-1, keepdims=True)
        out.append(tok)
    return jnp.concatenate(out, axis=1), logits


class WindowedKVTest(unittest.TestCase):
    def setUp(self):
        self.config = windowed_config()
        self.model = Gemma4EModelJAX(self.config)
        self.params = build_tiny_params(self.config)

    def test_sliding_layers_allocate_only_the_window(self):
        full = init_kv_cache(self.config, 1, 64, DTYPE, window_kv=False)
        win = init_kv_cache(self.config, 1, 64, DTYPE, window_kv=True)
        sliding = [i for i in range(self.config.first_kv_shared_layer_idx)
                   if self.config.layer_types[i] == "sliding_attention"]
        full_att = [i for i in range(self.config.first_kv_shared_layer_idx)
                    if self.config.layer_types[i] == "full_attention"]
        self.assertTrue(sliding and full_att, "config must exercise both layer types")
        for i in sliding:
            self.assertEqual(win[i][0].shape[2], WINDOW, f"layer {i} should hold only the window")
            self.assertEqual(full[i][0].shape[2], 64)
        for i in full_att:
            self.assertEqual(win[i][0].shape[2], 64, f"full-attention layer {i} must keep full length")

    def test_saves_memory(self):
        def total(c):
            return sum(k.size * k.dtype.itemsize + v.size * v.dtype.itemsize for k, v in c.values())
        full = total(init_kv_cache(self.config, 1, 64, DTYPE, window_kv=False))
        win = total(init_kv_cache(self.config, 1, 64, DTYPE, window_kv=True))
        self.assertLess(win, full)

    def test_same_tokens_when_prompt_fits_in_window(self):
        prompt = jax.random.randint(jax.random.PRNGKey(1), (1, WINDOW - 1), 1, self.config.vocab_size)
        a, _ = generate(self.model, self.params, prompt, 4, window_kv=False)
        b, _ = generate(self.model, self.params, prompt, 4, window_kv=True)
        self.assertEqual(a.tolist(), b.tolist())

    def test_decode_when_total_cache_is_shorter_than_window(self):
        """The ring mask must match the allocated cache, not the configured window."""
        config = replace(self.config, sliding_window=16)
        model = Gemma4EModelJAX(config)
        params = build_tiny_params(config)
        prompt = jax.random.randint(
            jax.random.PRNGKey(4), (1, 2), 1, config.vocab_size)
        # total cache length is 4, well below sliding_window=16. The old code
        # produced attention scores over 4 keys and a mask over 16 keys.
        a, la = generate(model, params, prompt, 2, window_kv=False)
        b, lb = generate(model, params, prompt, 2, window_kv=True)
        for i, (x, y) in enumerate(zip(la, lb)):
            self.assertLess(float(jnp.max(jnp.abs(x - y))), 1e-4,
                            f"step {i} diverges with a short ring")
        self.assertEqual(a.tolist(), b.tolist())

    def test_same_tokens_when_prompt_exceeds_window(self):
        """The case the ring buffer exists for: prompt longer than the window."""
        prompt = jax.random.randint(jax.random.PRNGKey(2), (2, 3 * WINDOW + 1), 1, self.config.vocab_size)
        a, la = generate(self.model, self.params, prompt, 5, window_kv=False)
        b, lb = generate(self.model, self.params, prompt, 5, window_kv=True)
        for i, (x, y) in enumerate(zip(la, lb)):
            d = float(jnp.max(jnp.abs(x - y)))
            self.assertLess(d, 1e-4, f"step {i} logits diverge by {d:.2e} under windowed KV")
        self.assertEqual(a.tolist(), b.tolist())

    def test_right_padded_prompt_does_not_attend_to_pads(self):
        """Regression: the ring mask must consult `valid`, not assume contiguous fill.

        The server pads every prompt to a static bucket and then decodes at
        bucket_s + step while the logical position stays at the real length
        (jax_engine.py:434). So the pad slots [real_len, bucket) sit INSIDE
        [0, slot). make_ring_decode_mask marked everything under slot+1 as real,
        so windowed layers attended to pad K/V -- silently, with no error.
        make_decode_mask on the non-windowed path takes `valid` and excludes them,
        which is why only window_kv=True corrupted. Every other test here fills
        the prompt completely, so the gap never existed to be caught.
        """
        S, real = 3 * WINDOW, 3 * WINDOW - 3      # 12 slots, 9 real, 3 pad
        prompt = jax.random.randint(jax.random.PRNGKey(5), (1, S), 1, self.config.vocab_size)
        a, la = generate(self.model, self.params, prompt, 4, window_kv=False, real_len=real)
        b, lb = generate(self.model, self.params, prompt, 4, window_kv=True, real_len=real)
        for i, (x, y) in enumerate(zip(la, lb)):
            d = float(jnp.max(jnp.abs(x - y)))
            self.assertLess(d, 1e-4, f"step {i} logits diverge by {d:.2e} — ring mask attended to pads")
        self.assertEqual(a.tolist(), b.tolist())

    def test_decode_wraps_past_the_window(self):
        """Generate more tokens than the window so the ring wraps during decode."""
        prompt = jax.random.randint(jax.random.PRNGKey(3), (1, 2), 1, self.config.vocab_size)
        n = 2 * WINDOW + 2
        a, la = generate(self.model, self.params, prompt, n, window_kv=False)
        b, lb = generate(self.model, self.params, prompt, n, window_kv=True)
        for i, (x, y) in enumerate(zip(la, lb)):
            self.assertLess(float(jnp.max(jnp.abs(x - y))), 1e-4, f"step {i} diverges after wrap")
        self.assertEqual(a.tolist(), b.tolist())


if __name__ == "__main__":
    unittest.main()
