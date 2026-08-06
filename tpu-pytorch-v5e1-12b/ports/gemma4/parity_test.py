"""Layer-by-layer parity test: ports/gemma4/model.py vs HF transformers.

Builds a MINI Gemma4 dense text config (both attention geometries, truncated
5:1 sliding/full layer pattern), instantiates the HF `Gemma4ForCausalLM`
(transformers 5.12.1, eager attention) and this port in float32 on CPU with
identical (fuzzed) weights, and compares final logits — with per-layer hidden
state localization on failure.

Run:  cd /home/xbill/tpu-pytorch-v5e1-12b/ports/gemma4 && python3 parity_test.py

Note: HF's Gemma4TextConfig force-coerces the LAST layer to "full_attention"
(a hard __post_init__ rule), so the truncated pattern used here ends in a full
layer: [sliding, sliding, full, sliding, full]. This still covers every
transition: sliding runs, sliding->full, and full->sliding.
"""

import sys

import torch

import model as port_model

TOLERANCE = 1e-4

MINI_TEXT_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 256,
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
}


def build_hf_model():
  import transformers  # deferred so the port itself never needs it

  config = transformers.Gemma4TextConfig(
      **MINI_TEXT_CONFIG,
      attention_dropout=0.0,
      hidden_size_per_layer_input=0,  # no per-layer embeddings
      vocab_size_per_layer_input=MINI_TEXT_CONFIG["vocab_size"],
      num_kv_shared_layers=0,
      enable_moe_block=False,
      use_bidirectional_attention="vision",  # text path stays causal
      use_cache=False,
  )
  config._attn_implementation = "eager"
  hf = transformers.Gemma4ForCausalLM(config).eval()
  return hf


def fuzz_weights(hf):
  """Randomizes every parameter (norm weights fuzzed around 1.0)."""
  gen = torch.Generator().manual_seed(1234)
  with torch.no_grad():
    for name, param in hf.named_parameters():
      noise = torch.randn(param.shape, generator=gen, dtype=torch.float32)
      if "norm" in name:
        param.copy_(1.0 + 0.1 * noise)
      else:
        param.copy_(0.05 * noise)


def build_port_model(hf):
  cfg = port_model.Gemma4DenseConfig.from_text_config(dict(MINI_TEXT_CONFIG))
  assert cfg.layer_types == list(hf.config.layer_types), (
      cfg.layer_types,
      hf.config.layer_types,
  )
  port = port_model.Gemma4DenseForCausalLM(cfg).eval()
  result = port_model.load_hf_state_dict(port, hf.state_dict(), strict=True)
  print(f"state dict load: {result}")

  hf_numel = sum(p.numel() for p in hf.parameters())
  port_numel = sum(p.numel() for p in port.parameters())
  assert hf_numel == port_numel, (hf_numel, port_numel)
  print(f"parameter count match: {hf_numel}")
  return port


def compare_case(name, hf, port, input_ids, attention_mask=None):
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
  ok = max_diff < TOLERANCE
  print(f"[{name}] final logits max abs diff: {max_diff:.3e} "
        f"({'PASS' if ok else 'FAIL'}, tolerance {TOLERANCE:g})")

  if not ok:
    hf_hidden = hf_out.hidden_states
    print(f"  localizing: hf reports {len(hf_hidden)} hidden states, "
          f"port has {len(port_hidden)} (embeddings + per-layer)")
    # HF may or may not include the embedding output as element 0.
    offset = len(port_hidden) - len(hf_hidden)
    for i, hf_h in enumerate(hf_hidden):
      port_h = port_hidden[i + offset]
      d = (hf_h - port_h).abs().max().item()
      print(f"  hidden_states[{i}] (port idx {i + offset}): "
            f"max abs diff {d:.3e}")
  return ok, max_diff


def main():
  torch.manual_seed(0)

  hf = build_hf_model()
  fuzz_weights(hf)
  port = build_port_model(hf)

  batch, seq_len = 2, 48  # seq_len > sliding_window=16 to exercise the window
  input_ids = torch.randint(0, MINI_TEXT_CONFIG["vocab_size"],
                            (batch, seq_len))

  results = []
  results.append(compare_case("no padding", hf, port, input_ids))

  attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
  attention_mask[1, 44:] = 0  # right padding, shorter than the sliding window
  results.append(
      compare_case("right padding", hf, port, input_ids, attention_mask)
  )

  worst = max(diff for _, diff in results)
  if all(ok for ok, _ in results):
    print(f"PARITY PASS (worst-case final logits max abs diff {worst:.3e})")
    return 0
  print(f"PARITY FAIL (worst-case final logits max abs diff {worst:.3e})")
  return 1


if __name__ == "__main__":
  sys.exit(main())
