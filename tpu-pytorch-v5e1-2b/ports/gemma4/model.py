"""Gemma 4 DENSE text model (gemma-4-31b architecture) — single-file PyTorch port.

Clean-room reimplementation of the text decoder of HF
`transformers.models.gemma4.modeling_gemma4` (validated against transformers
5.12.1), in the style of the upstream backend examples/gemma3 single-file
model:

  * plain `torch` only — no transformers import, no backend package import;
  * every projection is a named `nn.Linear` subclass (QProjLinear, ...) so
    quantized replacements can be swapped in later by class;
  * hand-rolled eager attention with explicit causal / sliding-window masks;
  * the parity-proven no-cache full-sequence `forward()` PLUS a static-shape
    KV-cache path (`Gemma4StaticKVCache`, `prefill()`, `decode_step()`,
    `generate_cached()`) designed so torch.compile(backend="tpu",
    dynamic=False) compiles exactly one prefill graph and one decode graph:
    preallocated [B, n_kv, MAX_SEQ, D] tensors, writes via `index_copy_` with
    a tensor index, masks built from position tensors, no data-dependent
    Python control flow.

KV-cache contents (documented choice): the cache stores K AFTER k_norm and
AFTER RoPE, and V AFTER v_norm — i.e. exactly the per-position tensors the
attention consumes. RoPE here depends only on a token's own absolute position
(no relative rescaling), so post-RoPE K never needs recomputation, and caching
post-norm makes each decode step a single-token q/k/v compute plus one
attention over the cache. For `attention_k_eq_v` (full-attention) layers,
which have no v_proj, the cached V is `v_norm(raw k_proj output)` — the
pre-k_norm, pre-RoPE K passed through the scale-less v_norm, matching the
no-cache forward bit-for-bit.

Parameter names are identical to the HF `Gemma4ForCausalLM` text model
("model.embed_tokens.weight", "model.layers.{i}.self_attn.q_proj.weight", ...,
"lm_head.weight") so HF weights load with `load_state_dict` after the simple
key remap implemented in `load_hf_state_dict()`.

Gemma4 text-architecture quirks captured here (vs. Gemma3):

  * RMSNorm scales by `weight` (init ones), NOT `(1 + weight)`; normalization
    uses `x * pow(mean(x^2) + eps, -0.5)` computed in float32.
  * Two attention geometries: sliding_attention layers use `head_dim` (256) and
    `num_key_value_heads` (16); full_attention layers use `global_head_dim`
    (512) and `num_global_key_value_heads` (4).
  * `attention_k_eq_v`: on full_attention layers there is NO v_proj; V is the
    RAW k_proj output (before k_norm / RoPE), passed through v_norm only.
  * q_norm / k_norm are per-head-dim RMSNorms applied BEFORE RoPE; v_norm is a
    scale-less RMSNorm (no weight).
  * Attention scaling is 1.0 (not 1/sqrt(head_dim)) — the qk norms make the
    logits scale-free.
  * RoPE differs per layer type: sliding layers use default RoPE with theta
    10_000 over the full head_dim; full layers use "proportional" RoPE with
    theta 1_000_000 and partial_rotary_factor 0.25 over global_head_dim — only
    the first quarter of the rotary angles get nonzero frequencies (frequencies
    are exponents over the FULL head_dim), the rest of inv_freq is zero-padded
    so cos=1 / sin=0 leaves those dims unrotated.
  * RoPE is applied in (B, S, H, D) layout (unsqueeze_dim=2), before the
    transpose to (B, H, S, D).
  * Each decoder layer ends with a persistent ones(1) buffer `layer_scalar`
    multiplying the output (identity for released checkpoints, but present in
    state dicts).
  * Sliding-window mask is EXCLUSIVE: a query attends to keys with
    0 <= q_idx - kv_idx < sliding_window (window includes self).
  * Embeddings are scaled by sqrt(hidden_size); lm_head is tied to
    embed_tokens; final logits are soft-capped at 30.0 via tanh.

Ignored (not part of the 31b dense text model): vision/audio towers, MoE
blocks (enable_moe_block=false), per-layer embeddings
(hidden_size_per_layer_input=0), and KV-shared layers (num_kv_shared_layers=0).
"""

import dataclasses
import json
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


# The text_config of the real gemma-4-31b checkpoint (fields this port uses).
# The full layer_types list is the 5:1 sliding/full pattern over 60 layers.
GEMMA4_31B_TEXT = json.dumps({
    "vocab_size": 262144,
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "num_hidden_layers": 60,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "head_dim": 256,
    "num_global_key_value_heads": 4,
    "global_head_dim": 512,
    "attention_k_eq_v": True,
    "attention_bias": False,
    "sliding_window": 1024,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": True,
    "pad_token_id": 0,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
})


@dataclasses.dataclass
class Gemma4DenseConfig:
  """Mirror of the Gemma4TextConfig fields the dense text model needs."""

  vocab_size: int = 262_144
  hidden_size: int = 5376
  intermediate_size: int = 21504
  num_hidden_layers: int = 60
  num_attention_heads: int = 32
  num_key_value_heads: int = 16
  head_dim: int = 256
  num_global_key_value_heads: int = 4
  global_head_dim: int = 512
  attention_k_eq_v: bool = True
  attention_bias: bool = False
  sliding_window: int = 1024
  max_position_embeddings: int = 262_144
  rms_norm_eps: float = 1e-6
  final_logit_softcapping: Optional[float] = 30.0
  tie_word_embeddings: bool = True
  pad_token_id: int = 0
  # RoPE, per layer type.
  rope_theta_full: float = 1_000_000.0
  partial_rotary_factor_full: float = 0.25
  rope_theta_sliding: float = 10_000.0
  # None -> the default 5:1 pattern ([5x sliding, full] repeated).
  layer_types: Optional[list] = None

  def __post_init__(self):
    if self.layer_types is None:
      self.layer_types = [
          "sliding_attention" if bool((i + 1) % 6) else "full_attention"
          for i in range(self.num_hidden_layers)
      ]
    if len(self.layer_types) != self.num_hidden_layers:
      raise ValueError(
          f"layer_types has {len(self.layer_types)} entries, expected"
          f" {self.num_hidden_layers}"
      )

  @classmethod
  def from_text_config(cls, text_config: dict) -> "Gemma4DenseConfig":
    """Builds a config from an HF `text_config` dict (e.g. config.json)."""
    if text_config.get("enable_moe_block"):
      raise ValueError("MoE text models are not supported by this port")
    if text_config.get("hidden_size_per_layer_input"):
      raise ValueError("Per-layer-input models are not supported by this port")
    if text_config.get("num_kv_shared_layers"):
      raise ValueError("KV-shared layers are not supported by this port")
    rope = text_config.get("rope_parameters") or {}
    rope_full = rope.get("full_attention") or {}
    rope_sliding = rope.get("sliding_attention") or {}
    return cls(
        vocab_size=text_config["vocab_size"],
        hidden_size=text_config["hidden_size"],
        intermediate_size=text_config["intermediate_size"],
        num_hidden_layers=text_config["num_hidden_layers"],
        num_attention_heads=text_config["num_attention_heads"],
        num_key_value_heads=text_config["num_key_value_heads"],
        head_dim=text_config["head_dim"],
        num_global_key_value_heads=text_config["num_global_key_value_heads"],
        global_head_dim=text_config["global_head_dim"],
        attention_k_eq_v=text_config.get("attention_k_eq_v", False),
        attention_bias=text_config.get("attention_bias", False),
        sliding_window=text_config["sliding_window"],
        max_position_embeddings=text_config["max_position_embeddings"],
        rms_norm_eps=text_config.get("rms_norm_eps", 1e-6),
        final_logit_softcapping=text_config.get("final_logit_softcapping"),
        tie_word_embeddings=text_config.get("tie_word_embeddings", True),
        pad_token_id=text_config.get("pad_token_id", 0),
        rope_theta_full=rope_full.get("rope_theta", 1_000_000.0),
        partial_rotary_factor_full=rope_full.get("partial_rotary_factor", 0.25),
        rope_theta_sliding=rope_sliding.get("rope_theta", 10_000.0),
        layer_types=text_config.get("layer_types"),
    )


# ---------------------------------------------------------------------------
# Named Linear subclasses: swap hooks for quantized replacements later.
# Plain nn.Linear behavior, no changes.
# ---------------------------------------------------------------------------


class QProjLinear(nn.Linear):
  pass


class KProjLinear(nn.Linear):
  pass


class VProjLinear(nn.Linear):
  pass


class OProjLinear(nn.Linear):
  pass


class GateProjLinear(nn.Linear):
  pass


class UpProjLinear(nn.Linear):
  pass


class DownProjLinear(nn.Linear):
  pass


class LMHeadLinear(nn.Linear):
  pass


class Gemma4RMSNorm(nn.Module):
  """Gemma4 RMSNorm.

  Unlike Gemma3 (zero-init weight, `(1 + weight)` scaling), Gemma4 initializes
  the weight to ones and scales by `weight` directly. `with_scale=False`
  (used for v_norm and nothing else in the dense text model) drops the weight
  entirely. Matches HF: norm in float32 via `x * pow(mean(x^2)+eps, -0.5)`,
  scale in float32, then cast back.
  """

  def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
    super().__init__()
    self.eps = eps
    self.with_scale = with_scale
    if self.with_scale:
      self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    hidden = x.float()
    mean_squared = hidden.pow(2).mean(-1, keepdim=True) + self.eps
    # torch.pow(mean, -0.5) to match HF bit-for-bit (they avoid rsqrt).
    normed = hidden * torch.pow(mean_squared, -0.5)
    if self.with_scale:
      normed = normed * self.weight.float()
    return normed.type_as(x)


class Gemma4ScaledWordEmbedding(nn.Embedding):
  """nn.Embedding scaled by sqrt(hidden_size)."""

  def __init__(
      self,
      num_embeddings: int,
      embedding_dim: int,
      padding_idx: int,
      embed_scale: float = 1.0,
  ):
    super().__init__(num_embeddings, embedding_dim, padding_idx)
    self.register_buffer(
        "embed_scale", torch.tensor(embed_scale), persistent=False
    )

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma4DenseMLP(nn.Module):
  """gate/up/down MLP with gelu_pytorch_tanh activation."""

  def __init__(self, config: Gemma4DenseConfig):
    super().__init__()
    self.gate_proj = GateProjLinear(
        config.hidden_size, config.intermediate_size, bias=False
    )
    self.up_proj = UpProjLinear(
        config.hidden_size, config.intermediate_size, bias=False
    )
    self.down_proj = DownProjLinear(
        config.intermediate_size, config.hidden_size, bias=False
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # ACT2FN["gelu_pytorch_tanh"] == F.gelu(..., approximate="tanh").
    return self.down_proj(
        F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
    )


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def _default_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
  """Standard RoPE inverse frequencies over the full head_dim."""
  return 1.0 / (
      theta
      ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
  )


def _proportional_inv_freq(
    head_dim: int, theta: float, partial_rotary_factor: float
) -> torch.Tensor:
  """"proportional" RoPE inverse frequencies (HF modeling_rope_utils).

  Only the first `int(partial * head_dim // 2)` angles rotate; their
  frequencies use the FULL head_dim in the exponent (i.e. they are the first
  quarter of a full-head_dim RoPE spectrum, not a compressed one). The
  remaining angles get inv_freq 0 => cos=1, sin=0 => identity rotation, so the
  returned inv_freq always has head_dim // 2 entries.
  """
  rope_angles = int(partial_rotary_factor * head_dim // 2)
  inv_freq = 1.0 / (
      theta
      ** (
          torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float()
          / head_dim
      )
  )
  nope_angles = head_dim // 2 - rope_angles
  if nope_angles > 0:
    inv_freq = torch.cat(
        (inv_freq, torch.zeros(nope_angles, dtype=torch.float32)), dim=0
    )
  return inv_freq


class Gemma4RotaryEmbedding(nn.Module):
  """Per-layer-type rotary embedding.

  sliding_attention: default RoPE, theta 10_000, over head_dim.
  full_attention:    proportional RoPE, theta 1_000_000, partial factor 0.25,
                     over global_head_dim.
  """

  def __init__(self, config: Gemma4DenseConfig):
    super().__init__()
    self.register_buffer(
        "sliding_attention_inv_freq",
        _default_inv_freq(config.head_dim, config.rope_theta_sliding),
        persistent=False,
    )
    self.register_buffer(
        "full_attention_inv_freq",
        _proportional_inv_freq(
            config.global_head_dim,
            config.rope_theta_full,
            config.partial_rotary_factor_full,
        ),
        persistent=False,
    )

  @torch.no_grad()
  def forward(
      self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str
  ) -> tuple:
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
    inv_freq_expanded = (
        inv_freq[None, :, None]
        .float()
        .expand(position_ids.shape[0], -1, 1)
        .to(x.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
  """Applies RoPE to a single (B, S, H, D) tensor (unsqueeze_dim=2)."""
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """(B, num_kv_heads, S, D) -> (B, num_kv_heads * n_rep, S, D)."""
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(
      batch, num_key_value_heads, n_rep, slen, head_dim
  )
  return hidden_states.reshape(
      batch, num_key_value_heads * n_rep, slen, head_dim
  )


def eager_attention_forward(
    num_key_value_groups: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
  """Plain eager attention.

  Args:
    num_key_value_groups: Q heads per KV head.
    query: (B, num_heads, S, D)
    key/value: (B, num_kv_heads, S, D)
    attention_mask: additive float mask broadcastable to (B, 1, S, S).
    scaling: attention logit scale (1.0 for Gemma4 text).

  Returns:
    (B, S, num_heads, D) attention output.
  """
  key_states = repeat_kv(key, num_key_value_groups)
  value_states = repeat_kv(value, num_key_value_groups)

  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
  attn_weights = attn_weights + attention_mask
  attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
      query.dtype
  )
  attn_output = torch.matmul(attn_weights, value_states)
  return attn_output.transpose(1, 2).contiguous()


class Gemma4DenseAttention(nn.Module):
  """Gemma4 text attention (dense; no KV cache, no KV sharing).

  Layer geometry depends on the layer type:
    * sliding_attention: head_dim, num_key_value_heads, separate v_proj.
    * full_attention (with attention_k_eq_v): global_head_dim,
      num_global_key_value_heads, and NO v_proj — V reuses the raw k_proj
      output (pre-norm, pre-RoPE), then goes through the scale-less v_norm.
  """

  def __init__(self, config: Gemma4DenseConfig, layer_idx: int):
    super().__init__()
    self.layer_type = config.layer_types[layer_idx]
    self.is_sliding = self.layer_type == "sliding_attention"
    self.head_dim = (
        config.head_dim if self.is_sliding else config.global_head_dim
    )
    self.use_alternative_attention = (
        config.attention_k_eq_v and not self.is_sliding
    )
    num_key_value_heads = (
        config.num_global_key_value_heads
        if self.use_alternative_attention
        else config.num_key_value_heads
    )
    self.num_key_value_heads = num_key_value_heads
    self.num_key_value_groups = (
        config.num_attention_heads // num_key_value_heads
    )
    self.scaling = 1.0  # qk-norm makes logits scale-free; NOT 1/sqrt(head_dim)

    self.q_proj = QProjLinear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = KProjLinear(
        config.hidden_size,
        num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    if not self.use_alternative_attention:
      self.v_proj = VProjLinear(
          config.hidden_size,
          num_key_value_heads * self.head_dim,
          bias=config.attention_bias,
      )
    else:
      self.v_proj = None
    self.o_proj = OProjLinear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )

    self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
    self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
    self.v_norm = Gemma4RMSNorm(
        dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple,
      attention_mask: torch.Tensor,
  ) -> torch.Tensor:
    input_shape = hidden_states.shape[:-1]  # (B, S)
    hidden_shape = (*input_shape, -1, self.head_dim)  # (B, S, H, D)
    cos, sin = position_embeddings

    # Q: proj -> per-head RMSNorm -> RoPE (in B,S,H,D layout) -> (B,H,S,D).
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin)
    query_states = query_states.transpose(1, 2)

    # K/V: V branches off the RAW k_proj output when v_proj is absent.
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if self.v_proj is not None:
      value_states = self.v_proj(hidden_states).view(hidden_shape)
    else:
      value_states = key_states  # attention_k_eq_v: pre-norm, pre-RoPE K

    key_states = self.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states, cos, sin)
    key_states = key_states.transpose(1, 2)

    value_states = self.v_norm(value_states)
    value_states = value_states.transpose(1, 2)

    attn_output = eager_attention_forward(
        self.num_key_value_groups,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output)

  # ----- static-KV-cache path (forward() above stays the parity reference) --

  def _compute_qkv(
      self, hidden_states: torch.Tensor, position_embeddings: tuple
  ) -> tuple:
    """Q/K/V exactly as in forward(): returns

      q: (B, H, S, D)      post q_norm, post RoPE
      k: (B, n_kv, S, D)   post k_norm, post RoPE   <- what gets cached
      v: (B, n_kv, S, D)   post v_norm              <- what gets cached
                           (k_eq_v layers: v_norm of the RAW k_proj output)
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin)
    query_states = query_states.transpose(1, 2)

    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if self.v_proj is not None:
      value_states = self.v_proj(hidden_states).view(hidden_shape)
    else:
      value_states = key_states  # attention_k_eq_v: pre-norm, pre-RoPE K

    key_states = self.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states, cos, sin)
    key_states = key_states.transpose(1, 2)

    value_states = self.v_norm(value_states)
    value_states = value_states.transpose(1, 2)
    return query_states, key_states, value_states

  def prefill_attention(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple,
      attention_mask: Optional[torch.Tensor],
      k_cache: torch.Tensor,
      v_cache: torch.Tensor,
      cache_positions: torch.Tensor,
      use_sdpa_causal: bool = True,
  ) -> torch.Tensor:
    """Prefill: fills cache rows `cache_positions`, attends over the prompt.

    Full-attention layers (no padding, prompt at positions 0..S-1) use SDPA
    with attn_mask=None / is_causal=True when `use_sdpa_causal` — on TPU that
    unlocks the fused kernel. Sliding layers always use the exact exclusive
    window mask (`attention_mask`), via eager attention.
    """
    input_shape = hidden_states.shape[:-1]
    query_states, key_states, value_states = self._compute_qkv(
        hidden_states, position_embeddings
    )
    k_cache.index_copy_(2, cache_positions, key_states)
    v_cache.index_copy_(2, cache_positions, value_states)

    if use_sdpa_causal and not self.is_sliding:
      attn_output = F.scaled_dot_product_attention(
          query_states,
          repeat_kv(key_states, self.num_key_value_groups),
          repeat_kv(value_states, self.num_key_value_groups),
          attn_mask=None,
          is_causal=True,
          scale=self.scaling,
      )
      attn_output = attn_output.transpose(1, 2).contiguous()
    else:
      attn_output = eager_attention_forward(
          self.num_key_value_groups,
          query_states,
          key_states,
          value_states,
          attention_mask,
          scaling=self.scaling,
      )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output)

  def decode_attention(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple,
      attention_mask: torch.Tensor,
      k_cache: torch.Tensor,
      v_cache: torch.Tensor,
      cache_positions: torch.Tensor,
  ) -> torch.Tensor:
    """Decode step: writes the new token's K/V at `cache_positions` (a (1,)
    tensor) and attends q (S=1) over the FULL static cache; `attention_mask`
    ((1, 1, 1, MAX_SEQ) additive) masks future and out-of-window rows, so the
    unused cache padding contributes exactly zero after softmax.
    """
    input_shape = hidden_states.shape[:-1]
    query_states, key_states, value_states = self._compute_qkv(
        hidden_states, position_embeddings
    )
    k_cache.index_copy_(2, cache_positions, key_states)
    v_cache.index_copy_(2, cache_positions, value_states)

    attn_output = eager_attention_forward(
        self.num_key_value_groups,
        query_states,
        k_cache,
        v_cache,
        attention_mask,
        scaling=self.scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output)


class Gemma4DenseDecoderLayer(nn.Module):
  """Gemma-style sandwich-norm decoder layer with trailing layer_scalar."""

  def __init__(self, config: Gemma4DenseConfig, layer_idx: int):
    super().__init__()
    self.attention_type = config.layer_types[layer_idx]
    self.self_attn = Gemma4DenseAttention(config, layer_idx)
    self.mlp = Gemma4DenseMLP(config)
    self.input_layernorm = Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.post_attention_layernorm = Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.pre_feedforward_layernorm = Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.post_feedforward_layernorm = Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    # Persistent buffer, present in HF state dicts (ones => identity).
    self.register_buffer("layer_scalar", torch.ones(1))

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
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
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
    """forward() with the attention swapped for the cached prefill/decode."""
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
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    hidden_states = hidden_states * self.layer_scalar
    return hidden_states


# ---------------------------------------------------------------------------
# Static KV cache
# ---------------------------------------------------------------------------


class Gemma4StaticKVCache:
  """Preallocated fixed-shape KV cache for the prefill/decode path.

  One (k, v) tensor pair per decoder layer, each of shape
  [batch, n_kv_heads, max_seq_len, head_dim] with that LAYER's geometry
  (sliding layers: num_key_value_heads x head_dim; full/k_eq_v layers:
  num_global_key_value_heads x global_head_dim). Contents: K post-k_norm and
  post-RoPE, V post-v_norm (see the module docstring). All writes go through
  `index_copy_` with a position TENSOR so every shape in the graph is static.
  """

  def __init__(
      self,
      text_model: "Gemma4DenseTextModel",
      batch_size: int,
      max_seq_len: int,
      dtype: Optional[torch.dtype] = None,
      device: Optional[torch.device] = None,
  ):
    ref = text_model.embed_tokens.weight
    dtype = ref.dtype if dtype is None else dtype
    device = ref.device if device is None else device
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.k = []
    self.v = []
    for layer in text_model.layers:
      attn = layer.self_attn
      n_kv = attn.num_key_value_heads
      shape = (batch_size, n_kv, max_seq_len, attn.head_dim)
      self.k.append(torch.zeros(shape, dtype=dtype, device=device))
      self.v.append(torch.zeros(shape, dtype=dtype, device=device))

  def reset(self):
    for k in self.k:
      k.zero_()
    for v in self.v:
      v.zero_()


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def _create_causal_mask(
    seq_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
  """(1, 1, S, S) additive causal mask (min where kv_idx > q_idx)."""
  mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
  mask.masked_fill_(
      torch.triu(
          torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
          diagonal=1,
      ),
      torch.finfo(dtype).min,
  )
  return mask[None, None, :, :]


def _create_sliding_window_causal_mask(
    seq_len: int, sliding_window: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
  """(1, 1, S, S) additive sliding-window causal mask.

  Gemma4/HF semantics are EXCLUSIVE: allowed iff 0 <= q_idx - kv_idx <
  sliding_window (i.e. self plus the previous sliding_window - 1 tokens).
  Note this differs from the Gemma3 example port, which allowed
  q_idx - kv_idx == sliding_window as well.
  """
  mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
  future_mask = torch.triu(
      torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
  )
  # Masked when kv_idx <= q_idx - sliding_window.
  past_mask = torch.tril(
      torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
      diagonal=-sliding_window,
  )
  mask.masked_fill_(future_mask | past_mask, torch.finfo(dtype).min)
  return mask[None, None, :, :]


def _create_decode_masks(
    pos: torch.Tensor,
    max_seq_len: int,
    sliding_window: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
  """(1, 1, 1, MAX_SEQ) additive masks for a single decode step at `pos`.

  `pos` is a (1,) long TENSOR (never a Python int) so the graph stays static.
  Built purely from a positions tensor — no data-dependent control flow:

    full_attention:    keep iff kv_idx <= pos                       (causal)
    sliding_attention: keep iff kv_idx <= pos AND pos - kv_idx < window
                       (the EXCLUSIVE Gemma4 window, self included)

  Everything else — future rows AND the never-written cache padding beyond
  the used length — gets finfo.min, which softmax turns into exactly 0.
  """
  kv_idx = torch.arange(max_seq_len, dtype=torch.long, device=device)
  keep_causal = kv_idx <= pos  # (MAX_SEQ,)
  keep_sliding = keep_causal & ((pos - kv_idx) < sliding_window)
  zero = torch.zeros((), dtype=dtype, device=device)
  neg = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
  return {
      "full_attention": torch.where(keep_causal, zero, neg)[
          None, None, None, :
      ],
      "sliding_attention": torch.where(keep_sliding, zero, neg)[
          None, None, None, :
      ],
  }


def _apply_padding_mask(
    mask: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
  """Merges a 2D (B, S) padding mask (1 = keep) into an additive 4D mask."""
  dtype = mask.dtype
  padding = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(
      dtype
  ).min
  return mask + padding


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Gemma4DenseTextModel(nn.Module):
  """Gemma4 dense text decoder stack (embeddings -> layers -> final norm)."""

  def __init__(self, config: Gemma4DenseConfig):
    super().__init__()
    self.config = config
    self.embed_tokens = Gemma4ScaledWordEmbedding(
        config.vocab_size,
        config.hidden_size,
        config.pad_token_id,
        embed_scale=config.hidden_size**0.5,
    )
    self.layers = nn.ModuleList([
        Gemma4DenseDecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
    ])
    self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.rotary_emb = Gemma4RotaryEmbedding(config)

  def forward(
      self,
      input_ids: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      output_hidden_states: bool = False,
  ):
    """Full-sequence forward.

    Args:
      input_ids: (B, S) token ids.
      attention_mask: optional (B, S) padding mask (1 = attend, 0 = pad).
      output_hidden_states: also return the per-layer hidden states
        (embeddings output + each decoder layer output).

    Returns:
      final hidden states (B, S, hidden), or a
      (hidden_states, all_hidden_states) tuple if output_hidden_states.
    """
    _, seq_len = input_ids.shape
    device = input_ids.device
    dtype = self.embed_tokens.weight.dtype

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)

    causal_mask_mapping = {
        "full_attention": _create_causal_mask(seq_len, dtype, device),
        "sliding_attention": _create_sliding_window_causal_mask(
            seq_len, self.config.sliding_window, dtype, device
        ),
    }
    if attention_mask is not None:
      causal_mask_mapping = {
          k: _apply_padding_mask(v, attention_mask)
          for k, v in causal_mask_mapping.items()
      }

    hidden_states = self.embed_tokens(input_ids)

    position_embeddings = {
        layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
        for layer_type in set(self.config.layer_types)
    }

    all_hidden_states = [hidden_states] if output_hidden_states else None
    for decoder_layer in self.layers:
      hidden_states = decoder_layer(
          hidden_states,
          position_embeddings=position_embeddings[
              decoder_layer.attention_type
          ],
          attention_mask=causal_mask_mapping[decoder_layer.attention_type],
      )
      if output_hidden_states:
        all_hidden_states.append(hidden_states)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
      return hidden_states, all_hidden_states
    return hidden_states

  # ----- static-KV-cache path ---------------------------------------------

  def _position_embeddings(
      self, hidden_states: torch.Tensor, position_ids: torch.Tensor
  ) -> dict:
    """cos/sin per layer type; both types computed unconditionally (static)."""
    return {
        "sliding_attention": self.rotary_emb(
            hidden_states, position_ids, "sliding_attention"
        ),
        "full_attention": self.rotary_emb(
            hidden_states, position_ids, "full_attention"
        ),
    }

  def prefill(
      self,
      input_ids: torch.Tensor,
      cache: Gemma4StaticKVCache,
      use_sdpa_causal: bool = True,
  ) -> torch.Tensor:
    """Prompt forward: fills cache positions 0..S-1, returns (B, S, hidden).

    Assumes an unpadded prompt starting at position 0 (that is what makes
    full-attention layers eligible for SDPA with is_causal=True and no mask).
    """
    _, seq_len = input_ids.shape
    device = input_ids.device
    dtype = self.embed_tokens.weight.dtype

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    cache_positions = position_ids  # (S,) tensor index for index_copy_
    position_ids = position_ids.unsqueeze(0)

    masks = {
        "sliding_attention": _create_sliding_window_causal_mask(
            seq_len, self.config.sliding_window, dtype, device
        ),
        "full_attention": (
            None
            if use_sdpa_causal
            else _create_causal_mask(seq_len, dtype, device)
        ),
    }

    hidden_states = self.embed_tokens(input_ids)
    position_embeddings = self._position_embeddings(
        hidden_states, position_ids
    )

    for layer_idx, decoder_layer in enumerate(self.layers):
      hidden_states = decoder_layer._forward_cached(
          hidden_states,
          position_embeddings[decoder_layer.attention_type],
          masks[decoder_layer.attention_type],
          cache.k[layer_idx],
          cache.v[layer_idx],
          cache_positions,
          is_decode=False,
          use_sdpa_causal=use_sdpa_causal,
      )
    return self.norm(hidden_states)

  def decode_step(
      self,
      input_ids: torch.Tensor,
      pos: torch.Tensor,
      cache: Gemma4StaticKVCache,
  ) -> torch.Tensor:
    """One decode step: `input_ids` (B, 1), `pos` a (1,) long tensor giving
    the token's absolute position. Writes cache row `pos`, attends over the
    full static cache with masks built from a positions tensor, returns
    (B, 1, hidden). All shapes depend only on B, 1 and MAX_SEQ — one graph.
    """
    device = input_ids.device
    dtype = self.embed_tokens.weight.dtype

    position_ids = pos.view(1, 1)
    masks = _create_decode_masks(
        pos,
        cache.max_seq_len,
        self.config.sliding_window,
        dtype,
        device,
    )

    hidden_states = self.embed_tokens(input_ids)
    position_embeddings = self._position_embeddings(
        hidden_states, position_ids
    )

    for layer_idx, decoder_layer in enumerate(self.layers):
      hidden_states = decoder_layer._forward_cached(
          hidden_states,
          position_embeddings[decoder_layer.attention_type],
          masks[decoder_layer.attention_type],
          cache.k[layer_idx],
          cache.v[layer_idx],
          pos,
          is_decode=True,
      )
    return self.norm(hidden_states)


class Gemma4DenseForCausalLM(nn.Module):
  """Gemma4 dense text model with tied LM head and softcapped logits."""

  def __init__(self, config: Gemma4DenseConfig):
    super().__init__()
    self.config = config
    self.model = Gemma4DenseTextModel(config)
    self.lm_head = LMHeadLinear(
        config.hidden_size, config.vocab_size, bias=False
    )
    if config.tie_word_embeddings:
      self.lm_head.weight = self.model.embed_tokens.weight

  def forward(
      self,
      input_ids: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      output_hidden_states: bool = False,
  ):
    """Returns (B, S, vocab) logits; optionally also per-layer hiddens."""
    outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
    )
    if output_hidden_states:
      hidden_states, all_hidden_states = outputs
    else:
      hidden_states, all_hidden_states = outputs, None

    logits = self.lm_head(hidden_states)
    if self.config.final_logit_softcapping is not None:
      logits = logits / self.config.final_logit_softcapping
      logits = torch.tanh(logits)
      logits = logits * self.config.final_logit_softcapping

    if output_hidden_states:
      return logits, all_hidden_states
    return logits

  # ----- static-KV-cache path ---------------------------------------------

  def _cap_logits(self, logits: torch.Tensor) -> torch.Tensor:
    if self.config.final_logit_softcapping is not None:
      logits = logits / self.config.final_logit_softcapping
      logits = torch.tanh(logits)
      logits = logits * self.config.final_logit_softcapping
    return logits

  def new_kv_cache(
      self,
      batch_size: int,
      max_seq_len: int,
      dtype: Optional[torch.dtype] = None,
      device: Optional[torch.device] = None,
  ) -> Gemma4StaticKVCache:
    return Gemma4StaticKVCache(
        self.model, batch_size, max_seq_len, dtype=dtype, device=device
    )

  def prefill(
      self,
      input_ids: torch.Tensor,
      cache: Gemma4StaticKVCache,
      use_sdpa_causal: bool = True,
  ) -> torch.Tensor:
    """Prompt forward filling the cache; returns LAST-position logits (B, V)."""
    hidden_states = self.model.prefill(
        input_ids, cache, use_sdpa_causal=use_sdpa_causal
    )
    return self._cap_logits(self.lm_head(hidden_states[:, -1, :]))

  def decode_step(
      self,
      input_ids: torch.Tensor,
      pos: torch.Tensor,
      cache: Gemma4StaticKVCache,
  ) -> torch.Tensor:
    """One cached decode step -> next-token logits (B, V).

    `input_ids` is (B, 1) — the token at absolute position `pos`, a (1,) long
    TENSOR. Static shapes throughout; safe for
    torch.compile(fullgraph=True, dynamic=False).
    """
    hidden_states = self.model.decode_step(input_ids, pos, cache)
    return self._cap_logits(self.lm_head(hidden_states[:, -1, :]))

  @torch.no_grad()
  def generate_cached(
      self,
      input_ids: torch.Tensor,
      max_new_tokens: int,
      max_seq_len: int,
      use_sdpa_causal: bool = True,
      decode_fn=None,
  ) -> tuple:
    """Greedy generation via prefill + decode_step.

    This driver loop lives OUTSIDE the compiled forward paths (argmax /
    concatenation here are host-side); `decode_fn` lets callers substitute a
    torch.compile'd decode_step. Returns (tokens (B, S+N), [per-step logits]).
    """
    batch, prompt_len = input_ids.shape
    device = input_ids.device
    if decode_fn is None:
      decode_fn = self.decode_step

    cache = self.new_kv_cache(batch, max_seq_len, device=device)
    logits = self.prefill(input_ids, cache, use_sdpa_causal=use_sdpa_causal)
    step_logits = [logits]
    next_token = logits.argmax(dim=-1, keepdim=True)
    tokens = torch.cat([input_ids, next_token], dim=1)

    for step in range(max_new_tokens - 1):
      pos = torch.tensor(
          [prompt_len + step], dtype=torch.long, device=device
      )
      logits = decode_fn(next_token, pos, cache)
      step_logits.append(logits)
      next_token = logits.argmax(dim=-1, keepdim=True)
      tokens = torch.cat([tokens, next_token], dim=1)
    return tokens, step_logits


def load_hf_state_dict(
    model: Gemma4DenseForCausalLM,
    hf_state_dict: dict,
    strict: bool = True,
):
  """Loads an HF Gemma4 state dict into the port.

  Key remap:
    * `Gemma4ForConditionalGeneration` checkpoints (the released 31b): text
      weights live under "model.language_model.*" -> renamed to "model.*";
      vision/audio/projector keys ("model.vision_tower.*", "model.audio_*",
      "model.embed_vision.*", "model.embed_audio.*",
      "model.multi_modal_projector.*") are dropped.
    * `Gemma4ForCausalLM` / `Gemma4TextModel` text-only state dicts
      ("model.*" / "lm_head.*") pass through unchanged.
    * A missing "lm_head.weight" is fine when embeddings are tied.

  Everything else (including the per-layer "layer_scalar" buffers) maps 1:1
  because this port uses the HF parameter names.
  """
  drop_prefixes = (
      "model.vision_tower.",
      "model.audio_tower.",
      "model.embed_vision.",
      "model.embed_audio.",
      "model.multi_modal_projector.",
      "vision_tower.",
      "audio_tower.",
  )
  remapped = {}
  for key, value in hf_state_dict.items():
    if key.startswith(drop_prefixes):
      continue
    if key.startswith("model.language_model."):
      key = "model." + key[len("model.language_model.") :]
    elif key.startswith("language_model."):
      key = "model." + key[len("language_model.") :]
    remapped[key] = value

  if (
      "lm_head.weight" not in remapped
      and model.config.tie_word_embeddings
      and "model.embed_tokens.weight" in remapped
  ):
    remapped["lm_head.weight"] = remapped["model.embed_tokens.weight"]

  return model.load_state_dict(remapped, strict=strict)
