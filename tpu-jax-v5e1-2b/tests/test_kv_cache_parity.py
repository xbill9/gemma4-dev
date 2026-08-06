"""Parity test: KV-cached decoding must match full-sequence re-forward exactly.

Runs a tiny random-weight Gemma4E config on CPU. The reference path re-runs the
full model over the growing sequence each step (no cache); the cached path uses
prefill_with_kv_cache + make_cached_decode_step. Greedy tokens must be identical.

Run: python3 -m unittest tests.test_kv_cache_parity
"""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ports.gemma4.jax_e_model import (  # noqa: E402
    Gemma4EConfig,
    Gemma4EModelJAX,
    generate_n_tokens_scan,
    generate_with_kv_cache,
    make_cached_decode_step,
    make_prefill_causal_mask,
    prefill_with_kv_cache,
)

DTYPE = jnp.float32


def tiny_config() -> Gemma4EConfig:
    # 10 layers -> layer_types [s,s,s,s,f,s,s,s,s,f]; first_kv_shared_layer_idx=6,
    # so the non-shared prefix contains both a sliding and a full-attention layer.
    return Gemma4EConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=10,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_global_key_value_heads=2,
        global_head_dim=32,
        num_kv_shared_layers=4,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )


def build_tiny_params(config: Gemma4EConfig, seed: int = 0):
    key = jax.random.PRNGKey(seed)

    def nxt():
        nonlocal key
        key, sub = jax.random.split(key)
        return sub

    def randn(*shape):
        return jax.random.normal(nxt(), shape, dtype=DTYPE) * 0.05

    L, H, ple = config.num_hidden_layers, config.hidden_size, config.hidden_size_per_layer_input
    params = {
        "embed_tokens": randn(config.vocab_size, H),
        "embed_tokens_per_layer": randn(config.vocab_size_per_layer_input, L * ple),
        "per_layer_model_projection": randn(H, L * ple),
        "per_layer_projection_norm": jnp.ones((ple,), dtype=DTYPE),
        "final_norm": jnp.ones((H,), dtype=DTYPE),
    }
    for i in range(L):
        is_sliding = config.layer_types[i] == "sliding_attention"
        h_dim = config.head_dim if is_sliding else config.global_head_dim
        num_kv = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads
        is_shared = i >= config.first_kv_shared_layer_idx
        inter = config.intermediate_size * 2 if (is_shared and config.use_double_wide_mlp) else config.intermediate_size

        layer = {
            "input_layernorm": jnp.ones((H,), dtype=DTYPE),
            "post_attention_layernorm": jnp.ones((H,), dtype=DTYPE),
            "per_layer_input_gate": randn(H, ple),
            "per_layer_projection": randn(ple, H),
            "post_per_layer_input_norm": jnp.ones((H,), dtype=DTYPE),
            "attn": {
                "q_proj": randn(H, config.num_attention_heads * h_dim),
                "o_proj": randn(config.num_attention_heads * h_dim, H),
                "q_norm": jnp.ones((h_dim,), dtype=DTYPE),
            },
            "mlp": {
                "gate_proj": randn(H, inter),
                "up_proj": randn(H, inter),
                "down_proj": randn(inter, H),
            },
        }
        if not is_shared:
            layer["attn"]["k_proj"] = randn(H, num_kv * h_dim)
            layer["attn"]["v_proj"] = randn(H, num_kv * h_dim)
            layer["attn"]["k_norm"] = jnp.ones((h_dim,), dtype=DTYPE)
        params[f"layer_{i}"] = layer
    return params


def reference_greedy_generate(model, prompt_ids, params, max_new_tokens):
    """No-cache reference: full causal re-forward over the growing sequence.

    Returns (tokens [B, N], per-step last logits list of [B, V]).
    """
    seq = prompt_ids
    out, step_logits = [], []
    for _ in range(max_new_tokens):
        B, S = seq.shape
        pos = jnp.arange(S, dtype=jnp.int32)[None, :].repeat(B, axis=0)
        mask = make_prefill_causal_mask(jnp.ones((B, S), dtype=jnp.bool_))
        logits = model(seq, params, pos, attention_mask=mask, quant_mode="fp16")
        last = logits[:, -1, :]
        step_logits.append(last)
        tok = jnp.argmax(last, axis=-1, keepdims=True)
        out.append(tok)
        seq = jnp.concatenate([seq, tok], axis=1)
    return jnp.concatenate(out, axis=1), step_logits


def cached_generate_with_logits(model, prompt_ids, prompt_valid, params, max_new_tokens):
    """Cached path, exposing per-step logits for tolerance-based comparison."""
    B, S = prompt_ids.shape
    last_logits, caches, valid = prefill_with_kv_cache(
        model, prompt_ids, prompt_valid, params, max_new_tokens,
        quant_mode="fp16", cache_dtype=DTYPE,
    )
    step = jax.jit(make_cached_decode_step(model, quant_mode="fp16"))
    prompt_lens = prompt_valid.sum(axis=1).astype(jnp.int32)

    step_logits = [last_logits]
    tok = jnp.argmax(last_logits, axis=-1, keepdims=True)
    out = [tok]
    for t in range(max_new_tokens - 1):
        caches, valid, last_logits = step(params, caches, valid, tok, prompt_lens + t, jnp.int32(S + t))
        step_logits.append(last_logits)
        tok = jnp.argmax(last_logits, axis=-1, keepdims=True)
        out.append(tok)
    return jnp.concatenate(out, axis=1), step_logits


class KVCacheParityTest(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.model = Gemma4EModelJAX(self.config)
        self.params = build_tiny_params(self.config)

    # float32 roundoff between the two attention orderings; measured max is ~1e-6.
    LOGIT_TOL = 1e-4

    def assert_parity(self, ref_toks, ref_logits, cached_toks, cached_logits, label):
        # 1. Logits must agree to float roundoff at every step.
        for i, (r, c) in enumerate(zip(ref_logits, cached_logits)):
            delta = float(jnp.max(jnp.abs(r - c)))
            self.assertLess(
                delta, self.LOGIT_TOL,
                f"{label}: step {i} logits diverge by {delta:.2e} (> {self.LOGIT_TOL:.0e})",
            )
        # 2. Tokens must agree wherever the decision is not a numerical near-tie.
        #    A top-2 gap under the logit tolerance can be flipped by roundoff alone.
        for i, (r, rt, ct) in enumerate(zip(ref_logits, ref_toks.T, cached_toks.T)):
            srt = jnp.sort(r, axis=-1)
            gaps = (srt[:, -1] - srt[:, -2])
            for b in range(r.shape[0]):
                if float(gaps[b]) > self.LOGIT_TOL:
                    self.assertEqual(
                        int(rt[b]), int(ct[b]),
                        f"{label}: step {i} row {b} token differs with a decisive "
                        f"top-2 gap of {float(gaps[b]):.6f}",
                    )

    def test_cached_decode_matches_full_reforward(self):
        prompt = jax.random.randint(jax.random.PRNGKey(42), (2, 7), 1, self.config.vocab_size)
        ref, ref_lg = reference_greedy_generate(self.model, prompt, self.params, max_new_tokens=8)
        cached, cac_lg = cached_generate_with_logits(
            self.model, prompt, jnp.ones(prompt.shape, dtype=jnp.bool_), self.params, 8
        )
        self.assert_parity(ref, ref_lg, cached, cac_lg, "unpadded")

    def test_cached_decode_with_right_padded_prompt(self):
        """A prompt right-padded to a bucket must behave like the unpadded prompt."""
        real_len, bucket = 5, 8
        prompt = jax.random.randint(jax.random.PRNGKey(7), (1, real_len), 1, self.config.vocab_size)
        ref, ref_lg = reference_greedy_generate(self.model, prompt, self.params, max_new_tokens=6)

        padded = jnp.pad(prompt, ((0, 0), (0, bucket - real_len)), constant_values=0)
        valid = jnp.concatenate(
            [jnp.ones((1, real_len), dtype=jnp.bool_), jnp.zeros((1, bucket - real_len), dtype=jnp.bool_)],
            axis=1,
        )
        cached, cac_lg = cached_generate_with_logits(self.model, padded, valid, self.params, 6)
        self.assert_parity(ref, ref_lg, cached, cac_lg, "right-padded")

    def test_scan_probe_is_not_a_correct_decoder(self):
        """Guard the documented caveat: generate_n_tokens_scan has no KV cache.

        It is a throughput probe only. If someone later makes it cache-correct,
        this test should be deleted along with the warning in its docstring.
        """
        prompt = jax.random.randint(jax.random.PRNGKey(3), (1, 6), 1, self.config.vocab_size)
        ref, _ = reference_greedy_generate(self.model, prompt, self.params, max_new_tokens=6)
        probe = generate_n_tokens_scan(
            self.model, prompt, self.params, num_steps=6, quant_mode="fp16"
        )
        self.assertFalse(
            bool(jnp.array_equal(ref, probe)),
            "generate_n_tokens_scan now matches the reference decoder — update its "
            "docstring warning and the benchmark methodology note.",
        )


if __name__ == "__main__":
    unittest.main()
