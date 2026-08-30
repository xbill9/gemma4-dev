"""MoE parity test: ports/gemma4/moe_model.py vs HF transformers (5.12.1).

Three checks on a MINI MoE Gemma4 text config (every layer MoE, both
attention geometries, truncated 5:1 sliding/full pattern), float32 on CPU:

  a. Full-model parity: HF `Gemma4ForCausalLM` (enable_moe_block=True,
     random weights, eager attention) vs the port with the REFERENCE experts
     backend — final logits, tolerance 1e-4, with per-layer hidden-state
     localization on failure.
  b. Quantized consistency: port with W4A16Experts (quantized from the same
     reference weights) vs port with reference experts holding the
     DEQUANTIZED weights — expected to match to ~1e-5 (same dispatch).
  c. torch.compile(backend="aot_eager", fullgraph=True) of the full MoE model
     forward — any graph break fails the test.

Run:  cd ports/gemma4 && python3 moe_parity_test.py

Note: HF's Gemma4TextConfig force-coerces the LAST layer to "full_attention",
so the truncated pattern ends in a full layer (same as parity_test.py).
"""

import copy
import sys

import torch

import moe_model

TOLERANCE_HF = 1e-4
TOLERANCE_QUANT = 1e-5

MINI_MOE_TEXT_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 96,  # dense MLP branch
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "num_global_key_value_heads": 1,
    "global_head_dim": 64,
    "attention_k_eq_v": True,
    "attention_bias": False,
    "sliding_window": 16,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": True,
    "pad_token_id": 0,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
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
    # MoE (real 26B-A4B: 128 experts, top_k 8, moe_intermediate 704).
    # hidden_size and moe_intermediate_size must stay multiples of 32 for the
    # w4a16 group quantization.
    "enable_moe_block": True,
    "num_experts": 8,
    "top_k_experts": 2,
    "moe_intermediate_size": 64,
}


def build_hf_model():
  import transformers  # deferred so the port itself never needs it

  config = transformers.Gemma4TextConfig(
      **MINI_MOE_TEXT_CONFIG,
      attention_dropout=0.0,
      hidden_size_per_layer_input=0,  # no per-layer embeddings
      vocab_size_per_layer_input=MINI_MOE_TEXT_CONFIG["vocab_size"],
      num_kv_shared_layers=0,
      use_bidirectional_attention="vision",  # text path stays causal
      use_cache=False,
  )
  config._attn_implementation = "eager"
  hf = transformers.Gemma4ForCausalLM(config).eval()
  return hf


def fuzz_weights(hf):
  """Randomizes every parameter (norm/scale weights fuzzed around 1.0)."""
  gen = torch.Generator().manual_seed(1234)
  with torch.no_grad():
    for name, param in hf.named_parameters():
      noise = torch.randn(param.shape, generator=gen, dtype=torch.float32)
      if "norm" in name or name.endswith("scale"):
        param.copy_(1.0 + 0.1 * noise)
      else:
        param.copy_(0.05 * noise)


def build_port_model(hf):
  cfg = moe_model.Gemma4MoEConfig.from_text_config(
      dict(MINI_MOE_TEXT_CONFIG)
  )
  assert cfg.layer_types == list(hf.config.layer_types), (
      cfg.layer_types,
      hf.config.layer_types,
  )
  port = moe_model.Gemma4MoEForCausalLM(cfg).eval()
  result = moe_model.load_hf_state_dict(port, hf.state_dict(), strict=True)
  print(f"state dict load: {result}")

  hf_numel = sum(p.numel() for p in hf.parameters())
  port_numel = sum(p.numel() for p in port.parameters())
  assert hf_numel == port_numel, (hf_numel, port_numel)
  print(f"parameter count match: {hf_numel}")
  return port


def compare_hf_case(name, hf, port, input_ids, attention_mask=None):
  with torch.no_grad():
    hf_out = hf(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    port_logits, port_hidden = port(
        input_ids, attention_mask=attention_mask, output_hidden_states=True
    )

  max_diff = (hf_out.logits - port_logits).abs().max().item()
  ok = max_diff < TOLERANCE_HF
  print(f"[hf parity: {name}] final logits max abs diff: {max_diff:.3e} "
        f"({'PASS' if ok else 'FAIL'}, tolerance {TOLERANCE_HF:g})")

  if not ok:
    hf_hidden = hf_out.hidden_states
    print(f"  localizing: hf reports {len(hf_hidden)} hidden states, "
          f"port has {len(port_hidden)} (embeddings + per-layer)")
    offset = len(port_hidden) - len(hf_hidden)
    for i, hf_h in enumerate(hf_hidden):
      port_h = port_hidden[i + offset]
      d = (hf_h - port_h).abs().max().item()
      print(f"  hidden_states[{i}] (port idx {i + offset}): "
            f"max abs diff {d:.3e}")
  return ok, max_diff


def check_quantized_consistency(port, input_ids):
  """Port w/ W4A16Experts vs port w/ reference experts on DEQUANT weights."""
  quant = moe_model.quantize_experts_(copy.deepcopy(port)).eval()
  ref_dq = moe_model.dequantize_experts_(copy.deepcopy(quant)).eval()

  with torch.no_grad():
    quant_logits = quant(input_ids)
    ref_dq_logits = ref_dq(input_ids)

  max_diff = (quant_logits - ref_dq_logits).abs().max().item()
  ok = max_diff < TOLERANCE_QUANT
  print(f"[quantized consistency] final logits max abs diff: {max_diff:.3e} "
        f"({'PASS' if ok else 'FAIL'}, tolerance {TOLERANCE_QUANT:g})")

  # Sanity: the quantized model should still be CLOSE to the unquantized
  # reference (int4 round-trip error only), not wildly off. Informational.
  with torch.no_grad():
    ref_logits = port(input_ids)
  drift = (quant_logits - ref_logits).abs().max().item()
  print(f"  (informational: quantized vs unquantized reference logits "
        f"max abs diff {drift:.3e})")
  return ok, max_diff


def check_aot_eager_fullgraph(port, input_ids):
  """Full-model forward must compile with fullgraph=True (no graph breaks)."""
  torch._dynamo.reset()
  compiled = torch.compile(port, backend="aot_eager", fullgraph=True)
  try:
    with torch.no_grad():
      compiled_logits = compiled(input_ids)
      eager_logits = port(input_ids)
  except Exception as e:  # graph break with fullgraph=True raises
    print(f"[aot_eager fullgraph] FAIL: {type(e).__name__}: {e}")
    return False, float("inf")
  max_diff = (compiled_logits - eager_logits).abs().max().item()
  ok = max_diff < TOLERANCE_HF
  print(f"[aot_eager fullgraph] compiled, no graph breaks; compiled-vs-eager "
        f"logits max abs diff: {max_diff:.3e} ({'PASS' if ok else 'FAIL'})")
  return ok, max_diff


def main():
  torch.manual_seed(0)

  hf = build_hf_model()
  fuzz_weights(hf)
  port = build_port_model(hf)

  batch, seq_len = 2, 48  # seq_len > sliding_window=16 to exercise the window
  input_ids = torch.randint(
      0, MINI_MOE_TEXT_CONFIG["vocab_size"], (batch, seq_len)
  )

  results = []
  results.append(compare_hf_case("no padding", hf, port, input_ids))

  attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
  attention_mask[1, 44:] = 0  # right padding, shorter than the sliding window
  results.append(
      compare_hf_case("right padding", hf, port, input_ids, attention_mask)
  )

  results.append(check_quantized_consistency(port, input_ids))
  results.append(check_aot_eager_fullgraph(port, input_ids))

  if all(ok for ok, _ in results):
    print("MOE PARITY PASS")
    return 0
  print("MOE PARITY FAIL")
  return 1


if __name__ == "__main__":
  sys.exit(main())
