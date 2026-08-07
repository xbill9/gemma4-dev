"""E-series (MatFormer) parity test: ports/gemma4/e_model.py vs HF.

MINI E config — the real gemma-4-E2B text config shrunk the way
parity_test.py's builder does, with ALL E-series features on:

  * num_kv_shared_layers=3 of 6 layers: shared layers 3 (sliding), 4
    (sliding), 5 (full) — KV sharing over both layer types; sources are
    layer 1 (last non-shared sliding) and layer 2 (last non-shared full).
  * hidden_size_per_layer_input=16 per-layer embeddings (PLE), packed
    vocab_size_per_layer_input x (6 * 16) table.
  * use_double_wide_mlp=True: layers 3..5 use intermediate_size * 2.
  * attention_k_eq_v=False, matching the real E2B (full layers have v_proj).
  * both attention geometries (head_dim 16 sliding / global_head_dim 32 full
    with proportional partial RoPE), softcapping, tied embeddings.

Checks:
  (a) full-model parity vs HF `Gemma4ForCausalLM` (transformers 5.12.1,
      eager, float32, fuzzed weights) < 1e-4, with per-layer localization —
      no-padding and right-padding cases;
  (b) cached decode vs the no-cache reference (like decode_test.py):
      identical greedy tokens and per-step logits < 1e-4, for batch 1 and 2,
      SDPA and eager prefill, two MAX_SEQ paddings — this exercises
      KV-sharing under the static cache;
  (c) torch.compile(decode_step, backend="aot_eager", fullgraph=True) traces
      without graph breaks and matches the reference.

Run:  cd ports/gemma4 && python3 e_parity_test.py
"""

import sys

import torch

import e_model
from decode_test import compare as decode_compare
from decode_test import reference_generate
from parity_test import TOLERANCE, compare_case, fuzz_weights

PROMPT_LEN = 12
NEW_TOKENS = 24
MAX_SEQ = 64  # > PROMPT_LEN + NEW_TOKENS - 1 = 35 used positions
ALT_MAX_SEQ = 48  # different padding amount must not change anything

MINI_E_TEXT_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_global_key_value_heads": 1,
    "global_head_dim": 32,
    "attention_k_eq_v": False,  # as on the real E2B
    "attention_bias": False,
    "sliding_window": 16,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": True,
    "pad_token_id": 0,
    # HF forces the last layer to full_attention; sources for the shared
    # layers are 1 (sliding) and 2 (full).
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    # --- E-series features, ALL on ---
    "num_kv_shared_layers": 3,
    "use_double_wide_mlp": True,
    "hidden_size_per_layer_input": 16,
    "vocab_size_per_layer_input": 512,
}

EXPECTED_SHARE_MAP = [0, 1, 2, 1, 1, 2]


def build_hf_model():
  import transformers  # deferred so the port itself never needs it

  config = transformers.Gemma4TextConfig(
      **MINI_E_TEXT_CONFIG,
      attention_dropout=0.0,
      enable_moe_block=False,
      use_bidirectional_attention="vision",  # text path stays causal
      use_cache=False,
  )
  config._attn_implementation = "eager"
  hf = transformers.Gemma4ForCausalLM(config).eval()
  return hf


def build_port_model(hf):
  cfg = e_model.Gemma4EConfig.from_text_config(dict(MINI_E_TEXT_CONFIG))
  assert cfg.layer_types == list(hf.config.layer_types), (
      cfg.layer_types,
      hf.config.layer_types,
  )
  assert cfg.kv_share_map() == EXPECTED_SHARE_MAP, cfg.kv_share_map()
  port = e_model.Gemma4EForCausalLM(cfg).eval()

  # Structural checks against HF before loading.
  for i, layer in enumerate(port.model.layers):
    hf_attn = hf.model.layers[i].self_attn
    assert layer.is_kv_shared_layer == hf_attn.is_kv_shared_layer, i
    assert layer.store_full_length_kv == hf_attn.store_full_length_kv, i
    if layer.is_kv_shared_layer:
      assert isinstance(layer.self_attn, e_model.Gemma4ESharedAttention), i
      assert not hasattr(hf_attn, "k_proj"), i  # HF ships no k/v modules
    assert (
        layer.mlp.gate_proj.out_features
        == hf.model.layers[i].mlp.intermediate_size
    ), i
  print(f"structure match (share map {EXPECTED_SHARE_MAP}, "
        f"double-wide on shared layers)")

  result = e_model.load_hf_state_dict(port, hf.state_dict(), strict=True)
  print(f"state dict load: {result}")

  hf_numel = sum(p.numel() for p in hf.parameters())
  port_numel = sum(p.numel() for p in port.parameters())
  assert hf_numel == port_numel, (hf_numel, port_numel)
  print(f"parameter count match: {hf_numel}")
  return port


def compile_decode_check(model, input_ids):
  """decode_step must trace fullgraph under aot_eager (static-shape proof)."""
  batch, prompt_len = input_ids.shape
  try:
    compiled = torch.compile(
        model.decode_step, backend="aot_eager", fullgraph=True
    )
    with torch.no_grad():
      cache = model.new_kv_cache(batch, MAX_SEQ)
      logits = model.prefill(input_ids, cache)
      step_logits = [logits]
      next_token = logits.argmax(dim=-1, keepdim=True)
      tokens = torch.cat([input_ids, next_token], dim=1)
      for step in range(NEW_TOKENS - 1):
        pos = torch.tensor([prompt_len + step], dtype=torch.long)
        logits = compiled(next_token, pos, cache)
        step_logits.append(logits)
        next_token = logits.argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
  except Exception as exc:  # graph break / trace failure
    print(f"[compile fullgraph] FAIL: {type(exc).__name__}: {exc}")
    return False, None
  print("[compile fullgraph] decode_step traced with fullgraph=True "
        "(backend=aot_eager): PASS")
  return True, (tokens, step_logits)


def check_e2b_config_accepted():
  """The real E2B text_config must parse (all its features implemented)."""
  import json
  import os

  path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "e2b_text_config.json")
  e2b_text = None
  if os.path.exists(path):
    e2b_text = json.load(open(path))
  else:  # inline copy of the fields from the fetched E2B config.json
    e2b_text = {
        "attention_bias": False, "attention_k_eq_v": False,
        "enable_moe_block": False, "final_logit_softcapping": 30.0,
        "global_head_dim": 512, "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 1536,
        "hidden_size_per_layer_input": 256, "intermediate_size": 6144,
        "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 7,
        "max_position_embeddings": 131072, "num_attention_heads": 8,
        "num_global_key_value_heads": None, "num_hidden_layers": 35,
        "num_key_value_heads": 1, "num_kv_shared_layers": 20,
        "pad_token_id": 0, "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10000.0, "rope_type": "default",
            },
        },
        "sliding_window": 512, "tie_word_embeddings": True,
        "use_double_wide_mlp": True, "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144,
    }
  cfg = e_model.Gemma4EConfig.from_text_config(e2b_text)
  share_map = cfg.kv_share_map()
  assert cfg.first_kv_shared_layer_idx == 15
  # E2B pattern is 4:1 -> last non-shared sliding is 13, last full is 14.
  assert all(
      share_map[i] == (14 if cfg.layer_types[i] == "full_attention" else 13)
      for i in range(15, 35)
  ), share_map
  print("[e2b config] real E2B text_config accepted "
        f"(first shared layer 15, sources sliding=13 full=14): PASS")
  return True


def main():
  torch.manual_seed(0)

  hf = build_hf_model()
  fuzz_weights(hf)
  port = build_port_model(hf)

  results = []

  # (pre) the real E2B config must be accepted by from_text_config.
  results.append((check_e2b_config_accepted(), 0.0))

  # (a) full-model HF parity, fp32, with per-layer localization on failure.
  batch, seq_len = 2, 48  # > sliding_window=16, > 2 * PLE dims
  input_ids = torch.randint(
      0, MINI_E_TEXT_CONFIG["vocab_size"], (batch, seq_len)
  )
  results.append(compare_case("no padding", hf, port, input_ids))

  attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
  attention_mask[1, 44:] = 0  # right padding
  results.append(
      compare_case("right padding", hf, port, input_ids, attention_mask)
  )

  # (b) cached decode vs no-cache reference (KV-sharing under the cache).
  gen = torch.Generator().manual_seed(42)
  ref_by_batch = {}
  for b in (1, 2):
    prompt = torch.randint(
        1, MINI_E_TEXT_CONFIG["vocab_size"], (b, PROMPT_LEN), generator=gen
    )
    ref = reference_generate(port, prompt, NEW_TOKENS)
    ref_by_batch[b] = (prompt, ref)

    cached_sdpa = port.generate_cached(
        prompt, NEW_TOKENS, MAX_SEQ, use_sdpa_causal=True
    )
    cached_eager = port.generate_cached(
        prompt, NEW_TOKENS, MAX_SEQ, use_sdpa_causal=False
    )
    cached_alt = port.generate_cached(
        prompt, NEW_TOKENS, ALT_MAX_SEQ, use_sdpa_causal=True
    )
    results.append(
        decode_compare(f"batch={b} cached sdpa MAX_SEQ={MAX_SEQ}", ref,
                       cached_sdpa)
    )
    results.append(
        decode_compare(f"batch={b} cached eager MAX_SEQ={MAX_SEQ}", ref,
                       cached_eager)
    )
    results.append(
        decode_compare(f"batch={b} cached sdpa MAX_SEQ={ALT_MAX_SEQ}", ref,
                       cached_alt)
    )

  # (c) fullgraph compile of the decode step, and it must match the reference.
  prompt, ref = ref_by_batch[2]
  compile_ok, compiled_run = compile_decode_check(port, prompt)
  if compile_ok:
    results.append(
        decode_compare("batch=2 compiled decode_step", ref, compiled_run)
    )
  else:
    results.append((False, float("inf")))

  worst = max(diff for _, diff in results)
  if all(ok for ok, _ in results):
    print(f"E PARITY PASS (worst-case max abs diff {worst:.3e}, "
          f"tolerance {TOLERANCE:g})")
    return 0
  print(f"E PARITY FAIL (worst-case max abs diff {worst:.3e})")
  return 1


if __name__ == "__main__":
  sys.exit(main())
