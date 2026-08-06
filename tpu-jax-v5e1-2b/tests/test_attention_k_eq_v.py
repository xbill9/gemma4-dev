"""`attention_k_eq_v`: some checkpoints ship no v_proj, because V *is* K.

Discovered on google/gemma-4-31B-it-qat-w4a16-ct, whose config sets the flag and
whose ten full-attention layers (i % 6 == 5) carry k_proj and k_norm but no
v_proj at all. Loading it without handling this produces ten silently missing
tensors — the exact failure the loader's strict validation exists to catch.

E2B sets the flag False and ships v_proj on every layer, so the default path
must stay byte-for-byte unchanged.

Run: python3 -m unittest tests.test_attention_k_eq_v
"""

import sys
import unittest
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jax_engine import config_from_hf  # noqa: E402
from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params  # noqa: E402


def packed_pair(out_dim: int, in_groups: int = 8, seed: int = 0):
    """A minimal compressed-tensors W4A16 pair the loader will accept."""
    packed = jnp.full((out_dim, in_groups * 4), seed + 1, dtype=jnp.int32)
    scale = jnp.ones((out_dim, in_groups), dtype=jnp.bfloat16)
    return packed, scale


def build_raw(n_layers: int, v_proj_on: set[int]) -> dict:
    """Checkpoint-shaped dict; only layers in `v_proj_on` get a v_proj."""
    raw: dict = {
        "model.embed_tokens.weight": jnp.ones((16, 8), dtype=jnp.bfloat16),
        "model.norm.weight": jnp.ones((8,), dtype=jnp.bfloat16),
    }
    for i in range(n_layers):
        lp = f"model.layers.{i}"
        for name, dim in (("q_proj", 8), ("o_proj", 8), ("k_proj", 4)):
            p, s = packed_pair(dim, seed=i)
            raw[f"{lp}.self_attn.{name}.weight_packed"] = p
            raw[f"{lp}.self_attn.{name}.weight_scale"] = s
        if i in v_proj_on:
            p, s = packed_pair(4, seed=i)
            raw[f"{lp}.self_attn.v_proj.weight_packed"] = p
            raw[f"{lp}.self_attn.v_proj.weight_scale"] = s
        for name in ("gate_proj", "up_proj", "down_proj"):
            p, s = packed_pair(8, seed=i)
            raw[f"{lp}.mlp.{name}.weight_packed"] = p
            raw[f"{lp}.mlp.{name}.weight_scale"] = s
        for nm in ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            raw[f"{lp}.{nm}.weight"] = jnp.ones((8,), dtype=jnp.bfloat16)
        raw[f"{lp}.self_attn.k_norm.weight"] = jnp.ones((4,), dtype=jnp.bfloat16)
    return raw


class ConfigPlumbingTest(unittest.TestCase):
    def test_flag_reaches_the_config(self):
        self.assertFalse(config_from_hf({}).attention_k_eq_v, "must default off")
        self.assertTrue(
            config_from_hf({"text_config": {"attention_k_eq_v": True}}).attention_k_eq_v)
        self.assertFalse(
            config_from_hf({"text_config": {"attention_k_eq_v": False}}).attention_k_eq_v)


class LoaderAliasTest(unittest.TestCase):
    """The 31B layout: full-attention layers omit v_proj."""

    N = 12
    FULL = {i for i in range(12) if i % 6 == 5}      # {5, 11}

    def test_loads_when_v_proj_is_absent(self):
        raw = build_raw(self.N, v_proj_on=set(range(self.N)) - self.FULL)
        params = convert_safetensors_to_jax_params(
            raw, num_layers=self.N, first_kv_shared_idx=self.N,
            prefix="model.", attention_k_eq_v=True)
        for i in range(self.N):
            attn = params[f"layer_{i}"]["attn"]
            self.assertIn("v_proj_packed", attn, f"layer {i} has no V")
            self.assertIn("v_proj_scale", attn, f"layer {i} has no V scale")

    def test_v_aliases_k_exactly_on_the_affected_layers(self):
        raw = build_raw(self.N, v_proj_on=set(range(self.N)) - self.FULL)
        params = convert_safetensors_to_jax_params(
            raw, num_layers=self.N, first_kv_shared_idx=self.N,
            prefix="model.", attention_k_eq_v=True)
        for i in self.FULL:
            attn = params[f"layer_{i}"]["attn"]
            self.assertIs(attn["v_proj_packed"], attn["k_proj_packed"],
                          f"layer {i}: V should BE K, not a copy")
            self.assertIs(attn["v_proj_scale"], attn["k_proj_scale"])

    def test_does_not_overwrite_a_v_proj_that_exists(self):
        """A layer shipping its own v_proj must keep it, flag or no flag."""
        raw = build_raw(self.N, v_proj_on=set(range(self.N)))
        params = convert_safetensors_to_jax_params(
            raw, num_layers=self.N, first_kv_shared_idx=self.N,
            prefix="model.", attention_k_eq_v=True)
        for i in range(self.N):
            attn = params[f"layer_{i}"]["attn"]
            self.assertIsNot(attn["v_proj_packed"], attn["k_proj_packed"],
                             f"layer {i}: real v_proj was clobbered by the alias")

    def test_flag_off_still_reports_a_missing_v_proj(self):
        """Without the flag, an absent v_proj must remain a loud failure."""
        raw = build_raw(self.N, v_proj_on=set(range(self.N)) - self.FULL)
        params = convert_safetensors_to_jax_params(
            raw, num_layers=self.N, first_kv_shared_idx=self.N,
            prefix="model.", attention_k_eq_v=False)
        for i in self.FULL:
            attn = params[f"layer_{i}"]["attn"]
            self.assertNotIn("v_proj_packed", attn,
                             f"layer {i}: V invented with the flag off")

    def test_e2b_shape_is_untouched(self):
        """Every layer ships v_proj and the flag is off: nothing may change."""
        raw = build_raw(self.N, v_proj_on=set(range(self.N)))
        params = convert_safetensors_to_jax_params(
            raw, num_layers=self.N, first_kv_shared_idx=self.N,
            prefix="model.", attention_k_eq_v=False)
        for i in range(self.N):
            attn = params[f"layer_{i}"]["attn"]
            self.assertIn("v_proj_packed", attn)
            self.assertIsNot(attn["v_proj_packed"], attn["k_proj_packed"])


if __name__ == "__main__":
    unittest.main()
