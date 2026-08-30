"""Gemma 4 MoE text model (gemma-4-26B-A4B architecture) — port extension.

Extends the parity-proven dense port (`ports/gemma4/model.py`) with the MoE
decoder variant, composing the dense building blocks by import (attention,
MLP, norms, rotary, masks, config) — nothing is copied.

HF ground truth (transformers 5.12.1 `modeling_gemma4.py`):

  * `Gemma4TextRouter`: on the flattened [T, H] input,
        h = RMSNorm_no_scale(h)                 (eps = rms_norm_eps)
        h = h * scale * hidden_size**-0.5       (`scale` is a ones-init [H] param)
        probs = softmax(proj(h), dim=-1)        (`proj`: Linear H -> E, no bias)
        top_k_weights, top_k_index = topk(probs, k)
        top_k_weights = top_k_weights / top_k_weights.sum(-1, keepdim=True)
        top_k_weights = top_k_weights * per_expert_scale[top_k_index]
    i.e. the per-expert scale is applied AFTER re-normalizing the top-k
    softmax probabilities to sum to 1.

  * `Gemma4TextDecoderLayer` MoE wiring (when `enable_moe_block`, which in HF
    turns the MoE block on for EVERY decoder layer): the MoE branch runs from
    the PRE-MLP residual, in parallel with the dense MLP:
        residual = h                                  # post-attention stream
        mlp_out  = mlp(pre_feedforward_layernorm(residual))
        h1 = post_feedforward_layernorm_1(mlp_out)
        top_k = router(residual.flat)                 # router sees the RAW residual
        h2 = experts(pre_feedforward_layernorm_2(residual.flat), top_k)
        h2 = post_feedforward_layernorm_2(h2)
        h  = post_feedforward_layernorm(h1 + h2)      # shared post-FFW norm on the SUM
        h  = residual + h
        h *= layer_scalar

Two experts backends, selectable per model (`experts_backend`):
  * "reference": `Gemma4RefExperts` — HF-semantics eager math on the bf16/fp32
    3D weights (`gate_up_proj` [E, 2I, H], `down_proj` [E, H, I]), but with
    the same STATIC-SHAPE gather + bmm dispatch as `W4A16Experts` (no
    data-dependent loop over hit experts as in the HF eager reference; the
    result only differs by matmul reduction order).
  * "w4a16": `W4A16Experts` from `ports/moe_experts/quant_experts.py` —
    int4-sym group-32 quantized experts; swap in with `quantize_experts_()`.

Static-shape discipline throughout: topk with a static k, gather via
index_select, no `.item()`, no data-dependent Python control flow.

Parameter names match HF (`model.layers.{i}.router.proj.weight`,
`.router.scale`, `.router.per_expert_scale`, `.experts.gate_up_proj`,
`.experts.down_proj`, `.post_feedforward_layernorm_1/_2.weight`,
`.pre_feedforward_layernorm_2.weight`) so `model.load_hf_state_dict` works
unchanged with the reference backend. (The router's `norm` is scale-less on
both sides, so it contributes no state-dict key.)

Real 26B-A4B values for reference: 128 experts, top_k 8,
moe_intermediate_size 704, hidden 2816, dense intermediate 2112, 30 layers.

Plain torch only — no transformers import, no backend package import.
"""

import dataclasses
import os
import sys
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOE_EXPERTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "moe_experts")
for _p in (_THIS_DIR, _MOE_EXPERTS_DIR):
  if _p not in sys.path:
    sys.path.insert(0, _p)

import model as port_model  # the dense port (read-only import)
from quant_experts import W4A16Experts  # ports/moe_experts (read-only import)


@dataclasses.dataclass
class Gemma4MoEConfig(port_model.Gemma4DenseConfig):
  """Dense config plus the MoE fields of Gemma4TextConfig.

  As in HF, `enable_moe_block=True` enables the MoE block on EVERY decoder
  layer (there is no per-layer MoE mask in transformers 5.12.1).
  """

  enable_moe_block: bool = True
  num_experts: int = 128
  top_k_experts: int = 8
  moe_intermediate_size: int = 704

  def __post_init__(self):
    super().__post_init__()
    if self.enable_moe_block:
      for field in ("num_experts", "top_k_experts", "moe_intermediate_size"):
        if not getattr(self, field):
          raise ValueError(f"enable_moe_block requires {field}")

  @classmethod
  def from_text_config(cls, text_config: dict) -> "Gemma4MoEConfig":
    """Builds a config from an HF `text_config` dict with MoE enabled."""
    if not text_config.get("enable_moe_block"):
      raise ValueError(
          "enable_moe_block is false; use Gemma4DenseConfig for dense models"
      )
    dense_dict = dict(text_config)
    dense_dict["enable_moe_block"] = False  # reuse the dense parser
    dense = port_model.Gemma4DenseConfig.from_text_config(dense_dict)
    return cls(
        **dataclasses.asdict(dense),
        enable_moe_block=True,
        num_experts=text_config["num_experts"],
        top_k_experts=text_config["top_k_experts"],
        moe_intermediate_size=text_config["moe_intermediate_size"],
    )


class RouterProjLinear(nn.Linear):
  """Named Linear subclass (quantization swap hook), like the dense port's."""


class Gemma4Router(nn.Module):
  """Gemma4TextRouter with HF-exact math and parameter names.

  Returns only (top_k_weights, top_k_index); the full router probability
  tensor HF also returns is unused by the decoder layer.
  """

  def __init__(self, config: Gemma4MoEConfig):
    super().__init__()
    self.top_k = config.top_k_experts
    self.scalar_root_size = config.hidden_size**-0.5
    self.norm = port_model.Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, with_scale=False
    )
    self.proj = RouterProjLinear(
        config.hidden_size, config.num_experts, bias=False
    )
    self.scale = nn.Parameter(torch.ones(config.hidden_size))
    self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

  def forward(self, hidden_states: torch.Tensor) -> tuple:
    """[T, H] -> (top_k_weights [T, K], top_k_index [T, K]). Static k."""
    hidden_states = self.norm(hidden_states)
    hidden_states = hidden_states * self.scale * self.scalar_root_size
    router_probabilities = F.softmax(self.proj(hidden_states), dim=-1)
    top_k_weights, top_k_index = torch.topk(
        router_probabilities, k=self.top_k, dim=-1
    )
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
    return top_k_weights, top_k_index


class Gemma4RefExperts(nn.Module):
  """Reference (unquantized) experts: HF Gemma4TextExperts semantics.

  Same 3D weights and math as the HF eager module —
      gate, up = linear(x, gate_up_proj[e]).chunk(2, -1)
      y = linear(act(gate) * up, down_proj[e]) * top_k_weights[t, k]
  summed over the K selected experts — but dispatched with the SAME
  static-shape gather + bmm pattern as `W4A16Experts` (so a W4A16Experts
  holding a quantization of these weights, or this module holding the
  dequantized weights, computes bit-identically apart from the weights
  themselves). Handles the HF padding convention (expert index ==
  num_experts) by clamping the gather index and zeroing the routing weight.
  """

  def __init__(
      self, num_experts: int, hidden_dim: int, intermediate_dim: int
  ):
    super().__init__()
    self.num_experts = num_experts
    self.hidden_dim = hidden_dim
    self.intermediate_dim = intermediate_dim
    self.gate_up_proj = nn.Parameter(
        torch.zeros(num_experts, 2 * intermediate_dim, hidden_dim)
    )
    self.down_proj = nn.Parameter(
        torch.zeros(num_experts, hidden_dim, intermediate_dim)
    )
    # Matches ACT2FN["gelu_pytorch_tanh"]; module form so W4A16Experts.from_hf
    # can reuse it directly.
    self.act_fn = nn.GELU(approximate="tanh")

  def forward(
      self,
      hidden_states: torch.Tensor,
      top_k_index: torch.Tensor,
      top_k_weights: torch.Tensor,
  ) -> torch.Tensor:
    """([T, H], [T, K], [T, K]) -> [T, H]. Static shapes in (T, K)."""
    T, H = hidden_states.shape
    K = top_k_index.shape[-1]

    flat_idx = top_k_index.reshape(-1)  # [T*K]
    valid = flat_idx < self.num_experts
    safe_idx = torch.where(valid, flat_idx, torch.zeros_like(flat_idx))
    w = top_k_weights.reshape(-1) * valid.to(top_k_weights.dtype)

    gu = self.gate_up_proj.index_select(0, safe_idx)  # [T*K, 2I, H]
    x = hidden_states.unsqueeze(1).expand(T, K, H).reshape(T * K, H, 1)
    gate_up = torch.bmm(gu, x).squeeze(-1)  # [T*K, 2I] == linear(x, W_e)
    gate, up = gate_up.chunk(2, dim=-1)  # HF order: [gate ; up]
    inter = self.act_fn(gate) * up  # [T*K, I]

    dn = self.down_proj.index_select(0, safe_idx)  # [T*K, H, I]
    out = torch.bmm(dn, inter.unsqueeze(-1)).squeeze(-1)  # [T*K, H]

    out = out * w.unsqueeze(-1)
    return out.to(hidden_states.dtype).reshape(T, K, H).sum(dim=1)


class Gemma4MoEDecoderLayer(port_model.Gemma4DenseDecoderLayer):
  """Dense decoder layer plus the parallel router/experts branch (HF wiring)."""

  def __init__(self, config: Gemma4MoEConfig, layer_idx: int):
    super().__init__(config, layer_idx)  # attn, dense mlp, 4 norms, scalar
    self.router = Gemma4Router(config)
    self.experts = Gemma4RefExperts(
        config.num_experts, config.hidden_size, config.moe_intermediate_size
    )
    self.post_feedforward_layernorm_1 = port_model.Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.post_feedforward_layernorm_2 = port_model.Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.pre_feedforward_layernorm_2 = port_model.Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )

  def _feedforward(self, residual: torch.Tensor) -> torch.Tensor:
    """The parallel dense-MLP + MoE feed-forward block.

    Input is the post-attention residual stream (PRE any FFW norm); returns
    the tensor that goes through the shared `post_feedforward_layernorm`
    before the residual add — exactly HF's wiring.
    """
    mlp_out = self.mlp(self.pre_feedforward_layernorm(residual))
    hidden_states_1 = self.post_feedforward_layernorm_1(mlp_out)

    # MoE branch: router sees the RAW residual; experts see it normed.
    flat = residual.reshape(-1, residual.shape[-1])
    top_k_weights, top_k_index = self.router(flat)
    hidden_states_2 = self.pre_feedforward_layernorm_2(flat)
    hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
    hidden_states_2 = hidden_states_2.reshape(residual.shape)
    hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

    return hidden_states_1 + hidden_states_2

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple,
      attention_mask: torch.Tensor,
  ) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self._feedforward(residual)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    hidden_states = hidden_states * self.layer_scalar
    return hidden_states

  def _forward_cached(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple,
      attention_mask: Optional[torch.Tensor],
      k_cache: torch.Tensor,
      v_cache: torch.Tensor,
      cache_positions: torch.Tensor,
      is_decode: bool,
      use_sdpa_causal: bool = True,
  ) -> torch.Tensor:
    """Cached-attention variant with the same MoE feed-forward block."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    if is_decode:
      hidden_states = self.self_attn.decode_attention(
          hidden_states,
          position_embeddings,
          attention_mask,
          k_cache,
          v_cache,
          cache_positions,
      )
    else:
      hidden_states = self.self_attn.prefill_attention(
          hidden_states,
          position_embeddings,
          attention_mask,
          k_cache,
          v_cache,
          cache_positions,
          use_sdpa_causal=use_sdpa_causal,
      )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self._feedforward(residual)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    hidden_states = hidden_states * self.layer_scalar
    return hidden_states


class Gemma4MoETextModel(port_model.Gemma4DenseTextModel):
  """Dense text model with the decoder layers swapped for MoE layers."""

  def __init__(self, config: Gemma4MoEConfig):
    super().__init__(config)
    if config.enable_moe_block:
      self.layers = nn.ModuleList([
          Gemma4MoEDecoderLayer(config, layer_idx)
          for layer_idx in range(config.num_hidden_layers)
      ])


class Gemma4MoEForCausalLM(port_model.Gemma4DenseForCausalLM):
  """Gemma4 MoE text model with tied LM head and softcapped logits.

  `forward` / `prefill` / `decode_step` / `generate_cached` are inherited
  from the dense port; only the decoder stack differs.
  """

  def __init__(self, config: Gemma4MoEConfig):
    super().__init__(config)
    self.model = Gemma4MoETextModel(config)
    if config.tie_word_embeddings:
      self.lm_head.weight = self.model.embed_tokens.weight


def quantize_experts_(model: Gemma4MoEForCausalLM) -> Gemma4MoEForCausalLM:
  """Swaps every layer's reference experts for W4A16Experts, in place.

  `W4A16Experts.from_hf` quantizes the reference module's 3D weights
  (int4 symmetric, group 32 along the contraction dim) and reuses its
  activation object, so the quantized model differs from the reference one
  only by the int4 round-trip of the expert weights.
  """
  for layer in model.model.layers:
    if isinstance(layer, Gemma4MoEDecoderLayer):
      layer.experts = W4A16Experts.from_hf(layer.experts)
  return model


def dequantize_experts_(model: Gemma4MoEForCausalLM) -> Gemma4MoEForCausalLM:
  """Swaps W4A16Experts back to Gemma4RefExperts holding the DEQUANTIZED
  weights, in place. With the identical dispatch in both modules, this model
  matches the quantized one to float-roundoff (the parity-test "quantized
  consistency" reference)."""
  for layer in model.model.layers:
    if isinstance(layer, Gemma4MoEDecoderLayer) and isinstance(
        layer.experts, W4A16Experts
    ):
      w4 = layer.experts
      gate_up, down = w4.dequantized_weights()
      ref = Gemma4RefExperts(w4.num_experts, w4.hidden_dim, w4.intermediate_dim)
      with torch.no_grad():
        ref.gate_up_proj.copy_(gate_up)
        ref.down_proj.copy_(down)
      ref.act_fn = w4.act_fn
      layer.experts = ref
  return model


# Re-export: HF-name-compatible state dict loading works unchanged for the
# reference backend (all MoE parameter names match HF's).
load_hf_state_dict = port_model.load_hf_state_dict
