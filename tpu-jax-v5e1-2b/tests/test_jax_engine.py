"""End-to-end test for JaxGemmaEngine against a synthetic safetensors checkpoint.

Writes a tiny Gemma4E-shaped checkpoint to a temp dir, loads it through the real
loader path (safetensors -> convert_safetensors_to_jax_params -> device_put),
and exercises streaming generation, EOS handling, and length capping.

No network, no PyTorch, no TPU required.

Run: python3 -m unittest tests.test_jax_engine
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jax_engine import GenerationStats, JaxGemmaEngine, config_from_hf  # noqa: E402
from ports.gemma4.jax_e_model import Gemma4EConfig  # noqa: E402

TINY_HF_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 10,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_global_key_value_heads": 2,
    "global_head_dim": 32,
    "num_kv_shared_layers": 4,
    "hidden_size_per_layer_input": 16,
    "vocab_size_per_layer_input": 256,
    "rms_norm_eps": 1e-6,
    "final_logit_softcapping": 30.0,
}


def write_tiny_checkpoint(path: Path, seed: int = 0) -> Gemma4EConfig:
    """Write config.json + model.safetensors in Hugging Face key/layout convention."""
    from safetensors.flax import save_file

    cfg = config_from_hf(TINY_HF_CONFIG)
    rng = np.random.default_rng(seed)

    def w(*shape):
        # HF stores dense Linear weights as [out, in]; the loader transposes.
        return jnp.asarray(rng.normal(0, 0.05, size=shape), dtype=jnp.bfloat16)

    def ones(n):
        return jnp.ones((n,), dtype=jnp.bfloat16)

    L, H, ple = cfg.num_hidden_layers, cfg.hidden_size, cfg.hidden_size_per_layer_input
    tensors = {
        "model.embed_tokens.weight": w(cfg.vocab_size, H),
        "model.norm.weight": ones(H),
        "model.embed_tokens_per_layer.weight": w(cfg.vocab_size_per_layer_input, L * ple),
        "model.per_layer_model_projection.weight": w(L * ple, H),
        "model.per_layer_projection_norm.weight": ones(ple),
    }
    for i in range(L):
        is_sliding = cfg.layer_types[i] == "sliding_attention"
        h_dim = cfg.head_dim if is_sliding else cfg.global_head_dim
        num_kv = cfg.num_key_value_heads if is_sliding else cfg.num_global_key_value_heads
        is_shared = i >= cfg.first_kv_shared_layer_idx
        inter = cfg.intermediate_size * 2 if (is_shared and cfg.use_double_wide_mlp) else cfg.intermediate_size
        p = f"model.layers.{i}"

        tensors[f"{p}.input_layernorm.weight"] = ones(H)
        tensors[f"{p}.post_attention_layernorm.weight"] = ones(H)
        tensors[f"{p}.per_layer_input_gate.weight"] = w(ple, H)
        tensors[f"{p}.per_layer_projection.weight"] = w(H, ple)
        tensors[f"{p}.post_per_layer_input_norm.weight"] = ones(H)

        tensors[f"{p}.self_attn.q_proj.weight"] = w(cfg.num_attention_heads * h_dim, H)
        tensors[f"{p}.self_attn.o_proj.weight"] = w(H, cfg.num_attention_heads * h_dim)
        tensors[f"{p}.self_attn.q_norm.weight"] = ones(h_dim)
        if not is_shared:
            # KV-shared layers legitimately omit k/v projections and k_norm —
            # the omission that upstream's loader mishandles (tpu-inference #3225).
            tensors[f"{p}.self_attn.k_proj.weight"] = w(num_kv * h_dim, H)
            tensors[f"{p}.self_attn.v_proj.weight"] = w(num_kv * h_dim, H)
            tensors[f"{p}.self_attn.k_norm.weight"] = ones(h_dim)

        tensors[f"{p}.mlp.gate_proj.weight"] = w(inter, H)
        tensors[f"{p}.mlp.up_proj.weight"] = w(inter, H)
        tensors[f"{p}.mlp.down_proj.weight"] = w(H, inter)

    path.mkdir(parents=True, exist_ok=True)
    with open(path / "config.json", "w") as fh:
        json.dump(TINY_HF_CONFIG, fh)
    save_file(tensors, str(path / "model.safetensors"))
    return cfg


class JaxEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = Path(cls._tmp.name) / "tiny-gemma4e"
        cls.cfg = write_tiny_checkpoint(cls.ckpt)

        cls.engine = JaxGemmaEngine(
            model_id="synthetic/tiny-gemma4e",
            kv_cache_dtype="bf16",
            quant_mode="fp16",   # dense weights in this fixture
            max_model_len=64,
        )
        cls.engine.load(local_dir=str(cls.ckpt))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_loads_without_torch(self):
        self.assertTrue(self.engine.is_ready)
        self.assertNotIn("torch", sys.modules, "engine load pulled in PyTorch")

    def test_kv_shared_layers_have_no_k_norm(self):
        """The #3225 shape: layers >= first_kv_shared_layer_idx carry no k/v params."""
        first = self.cfg.first_kv_shared_layer_idx
        for i in range(self.cfg.num_hidden_layers):
            attn = self.engine.params[f"layer_{i}"]["attn"]
            if i >= first:
                self.assertNotIn("k_norm", attn, f"layer {i} should not have k_norm")
                self.assertNotIn("k_proj", attn, f"layer {i} should not have k_proj")
            else:
                self.assertIn("k_norm", attn, f"layer {i} is missing k_norm")
                self.assertIn("k_proj", attn, f"layer {i} is missing k_proj")

    def test_streaming_yields_tokens_then_stats(self):
        out = list(self.engine.generate_stream([5, 9, 12], max_new_tokens=6, temperature=0.0))
        stats = out[-1]
        tokens = out[:-1]
        self.assertIsInstance(stats, GenerationStats)
        self.assertEqual(len(tokens), 6)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        self.assertEqual(stats.completion_tokens, 6)
        self.assertEqual(stats.prompt_tokens, 3)
        self.assertGreater(stats.prefill_ms, 0.0)
        self.assertEqual(stats.finish_reason, "length")

    def test_greedy_is_deterministic(self):
        a, _ = self.engine.generate([7, 3, 1], max_new_tokens=5, temperature=0.0)
        b, _ = self.engine.generate([7, 3, 1], max_new_tokens=5, temperature=0.0)
        self.assertEqual(a, b)

    def test_eos_stops_generation(self):
        """Feeding the first greedy token back as EOS must halt immediately."""
        tokens, _ = self.engine.generate([4, 4, 4], max_new_tokens=8, temperature=0.0)
        first = tokens[0]
        stopped, stats = self.engine.generate(
            [4, 4, 4], max_new_tokens=8, temperature=0.0, eos_token_ids=[first]
        )
        self.assertEqual(stopped, [])
        self.assertEqual(stats.finish_reason, "stop")
        self.assertEqual(stats.completion_tokens, 0)

    def test_max_model_len_is_enforced(self):
        long_prompt = list(range(1, 40))
        tokens, _ = self.engine.generate(long_prompt, max_new_tokens=1000, temperature=0.0)
        self.assertLessEqual(len(long_prompt) + len(tokens), self.engine.max_model_len)

    def test_rejects_prompt_longer_than_window(self):
        with self.assertRaises(ValueError):
            self.engine.generate(list(range(1, 200)), max_new_tokens=4)

    def test_sampling_respects_temperature(self):
        """Non-zero temperature with distinct seeds should not be locked to one output."""
        outs = {
            tuple(self.engine.generate(
                [2, 8, 6], max_new_tokens=6, temperature=1.5, top_k=50, seed=s
            )[0])
            for s in range(6)
        }
        self.assertGreater(len(outs), 1, "temperature sampling produced identical outputs")


if __name__ == "__main__":
    unittest.main()
