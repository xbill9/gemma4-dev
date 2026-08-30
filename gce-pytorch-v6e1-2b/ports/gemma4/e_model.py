"""Gemma 4 E-series (MatFormer) text model — port extension for gemma-4-E2B.

Extends the parity-proven dense port (`ports/gemma4/model.py`) with the
E-series decoder variant, composing the dense building blocks by import
(attention, MLP, norms, rotary, masks, config, cache, loader) — the same
pattern as `moe_model.py`. Nothing in `model.py` is modified.

E-series features (ground truth: the real google/gemma-4-E2B config.json +
transformers 5.12.1 `modeling_gemma4.py`; the E2B config has NO altup, laurel,
or activation-sparsity fields — those are NOT part of Gemma 4 E):

  * KV-shared layers (`num_kv_shared_layers` = 20 of 35 on E2B): the LAST
    `num_kv_shared_layers` decoder layers compute NO K/V of their own. Layer i
    (i >= first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers)
    REUSES the full-length key/value states of the last NON-shared layer with
    the same layer_type ("sliding_attention" layers reuse the last non-shared
    sliding layer; "full_attention" layers the last non-shared full layer).
    Shared layers keep their own q_proj / q_norm / o_proj (queries are theirs)
    but have NO k_proj, v_proj, k_norm or v_norm modules AT ALL — HF does not
    instantiate them (`if not self.is_kv_shared_layer` in Gemma4TextAttention)
    and additionally lists them in `_keys_to_ignore_on_load_unexpected`. The
    E2B qat-w4a16-ct checkpoint accordingly ships only
    `self_attn.{q_proj,q_norm,o_proj}` for layers 15..34 (verified from the
    safetensors header: no k_norm on ANY shared layer — this is by
    construction, not an export accident).

  * Per-Layer Embeddings, PLE (`hidden_size_per_layer_input` = 256,
    `vocab_size_per_layer_input` = 262144 on E2B): a second, packed token
    embedding `embed_tokens_per_layer` [V_ple, L * D_ple] (scaled by
    sqrt(D_ple)) plus a context projection of the scaled input embeddings
    (`per_layer_model_projection` [H -> L * D_ple], scaled by 1/sqrt(H), then
    RMSNorm over D_ple). Combined as (context + token) / sqrt(2) into a
    [B, S, L, D_ple] tensor; layer i consumes slice [:, :, i, :] AFTER its
    feed-forward residual add:
        r = h
        x = gelu_tanh(per_layer_input_gate(h))      # H -> D_ple
        x = x * per_layer_input[i]
        x = per_layer_projection(x)                  # D_ple -> H
        h = r + post_per_layer_input_norm(x)
    and only then the trailing `layer_scalar` multiply.

  * Double-wide MLP (`use_double_wide_mlp`): KV-shared layers use
    intermediate_size * 2 (E2B: 12288 vs 6144); non-shared layers are normal.

  * Everything else (RMSNorm flavor, two attention geometries with
    global_head_dim on full layers, proportional partial RoPE, exclusive
    sliding window, logit softcapping 30.0, tied embeddings, layer_scalar) is
    identical to the dense port and reused by import. On E2B
    `attention_k_eq_v` is FALSE, so full-attention layers have a real v_proj
    (the dense attention already handles both settings).

Static KV cache with KV sharing (`Gemma4EStaticKVCache`):

  Shared layers allocate NO cache slots. The cache keeps one (k, v) tensor
  pair per SOURCE layer; entry i of `cache.k` / `cache.v` is an ALIAS
  (same tensor object) of entry `share_map[i]`, where
      share_map[i] = i                                   for non-shared i
      share_map[i] = last non-shared j with              for shared i
                     layer_types[j] == layer_types[i]
  (E2B: layers 15..34 -> j = 13 for sliding, j = 14 for full; every shared
  sliding layer reads layer 13's slots, every shared full layer layer 14's.)
  Only non-shared layers write (index_copy_ at the position tensor); shared
  layers read the alias — during prefill the [:, :, :S, :] slice (the source
  layer earlier in the stack has already written rows 0..S-1 this call),
  during decode the full [MAX_SEQ] cache under the same additive decode mask
  as the dense port (rows > pos and unwritten padding get -inf). Because the
  cache is full-length there is none of the sliding-window truncation HF's
  `shared_kv_states` dict exists to work around — reading the source slots is
  exactly the shared full-length state. Same static-shape discipline as the
  dense port: no data-dependent control flow, writes via index_copy_ with a
  tensor index, decode masks built from position tensors.

Parameter names match HF (`model.embed_tokens_per_layer.weight`,
`model.per_layer_model_projection.weight`, `model.per_layer_projection_norm.
weight`, `model.layers.{i}.per_layer_input_gate.weight`,
`.per_layer_projection.weight`, `.post_per_layer_input_norm.weight`) so
`model.load_hf_state_dict` works unchanged (re-exported below).

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
if _THIS_DIR not in sys.path:
  sys.path.insert(0, _THIS_DIR)

import model as port_model  # the dense port (read-only import)


@dataclasses.dataclass
class Gemma4EConfig(port_model.Gemma4DenseConfig):
  """Dense config plus the E-series (MatFormer) fields of Gemma4TextConfig."""

  num_kv_shared_layers: int = 0
  use_double_wide_mlp: bool = False
  hidden_size_per_layer_input: int = 0
  vocab_size_per_layer_input: int = 0

  def __post_init__(self):
    super().__post_init__()
    if not 0 <= self.num_kv_shared_layers < self.num_hidden_layers:
      raise ValueError(
          f"num_kv_shared_layers={self.num_kv_shared_layers} must be in"
          f" [0, num_hidden_layers={self.num_hidden_layers})"
      )
    first = self.first_kv_shared_layer_idx
    for i in range(first, self.num_hidden_layers):
      if self.layer_types[i] not in self.layer_types[:first]:
        raise ValueError(
            f"KV-shared layer {i} has type {self.layer_types[i]!r} but no"
            " non-shared layer of that type exists to share from"
        )
    if self.hidden_size_per_layer_input and not self.vocab_size_per_layer_input:
      raise ValueError(
          "hidden_size_per_layer_input requires vocab_size_per_layer_input"
      )

  @property
  def first_kv_shared_layer_idx(self) -> int:
    return self.num_hidden_layers - self.num_kv_shared_layers

  def kv_share_map(self) -> list:
    """share_map[i]: the layer whose K/V layer i consumes (i itself if not
    shared). Shared layers map to the LAST non-shared layer of their type —
    HF's `store_full_length_kv` rule."""
    first = self.first_kv_shared_layer_idx
    last_of_type = {}
    for i in range(first):
      last_of_type[self.layer_types[i]] = i
    return [
        i if i < first else last_of_type[self.layer_types[i]]
        for i in range(self.num_hidden_layers)
    ]

  @classmethod
  def from_text_config(cls, text_config: dict) -> "Gemma4EConfig":
    """Builds a config from an HF `text_config` dict (e.g. the E2B one).

    Raises on any feature this port does not implement.
    """
    if text_config.get("enable_moe_block"):
      raise ValueError(
          "enable_moe_block with the E-series port is not supported (use"
          " moe_model.Gemma4MoEConfig for pure-MoE models)"
      )
    if text_config.get("use_bidirectional_attention") == "all":
      raise ValueError(
          'use_bidirectional_attention="all" is not supported by this port'
      )
    activation = text_config.get("hidden_activation", "gelu_pytorch_tanh")
    if activation != "gelu_pytorch_tanh":
      raise ValueError(f"hidden_activation {activation!r} is not supported")

    # Reuse the dense parser for all shared fields.
    dense_dict = dict(text_config)
    dense_dict["hidden_size_per_layer_input"] = 0
    dense_dict["num_kv_shared_layers"] = 0
    dense_dict["enable_moe_block"] = False
    dense = port_model.Gemma4DenseConfig.from_text_config(dense_dict)
    return cls(
        **dataclasses.asdict(dense),
        num_kv_shared_layers=text_config.get("num_kv_shared_layers") or 0,
        use_double_wide_mlp=bool(text_config.get("use_double_wide_mlp")),
        hidden_size_per_layer_input=(
            text_config.get("hidden_size_per_layer_input") or 0
        ),
        vocab_size_per_layer_input=(
            text_config.get("vocab_size_per_layer_input") or 0
        ),
    )


# Named Linear subclasses (quantization swap hooks), like the dense port's.


class PerLayerInputGateLinear(nn.Linear):
  pass


class PerLayerProjectionLinear(nn.Linear):
  pass


class PerLayerModelProjectionLinear(nn.Linear):
  pass


def _dense_attn_forward_capture(
    attn: port_model.Gemma4DenseAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: torch.Tensor,
) -> tuple:
  """Dense attention forward that ALSO returns the (k, v) it computed.

  Bit-identical to `Gemma4DenseAttention.forward` (same `_compute_qkv` +
  `eager_attention_forward` + o_proj); used for the no-cache path of source
  layers so their full-length K/V can be published to the shared layers.
  """
  input_shape = hidden_states.shape[:-1]
  query_states, key_states, value_states = attn._compute_qkv(
      hidden_states, position_embeddings
  )
  attn_output = port_model.eager_attention_forward(
      attn.num_key_value_groups,
      query_states,
      key_states,
      value_states,
      attention_mask,
      scaling=attn.scaling,
  )
  attn_output = attn_output.reshape(*input_shape, -1).contiguous()
  return attn.o_proj(attn_output), key_states, value_states


class Gemma4ESharedAttention(nn.Module):
  """Attention for a KV-shared layer: q_proj / q_norm / o_proj ONLY.

  K/V are supplied by the caller (the source layer's states — either the
  current full-length tensors in the no-cache path, a [:, :, :S, :] cache
  slice during prefill, or the full static cache during decode). Matches the
  HF `Gemma4TextAttention` shared branch: no k_proj / v_proj / k_norm /
  v_norm modules exist, so the state dict holds exactly what the E2B ct
  checkpoint ships for layers 15..34.
  """

  def __init__(self, config: Gemma4EConfig, layer_idx: int):
    super().__init__()
    self.layer_type = config.layer_types[layer_idx]
    self.is_sliding = self.layer_type == "sliding_attention"
    self.head_dim = (
        config.head_dim if self.is_sliding else config.global_head_dim
    )
    use_alternative_attention = (
        config.attention_k_eq_v and not self.is_sliding
    )
    # Same geometry as the SOURCE layer (same layer_type => same kv heads).
    self.num_key_value_heads = (
        config.num_global_key_value_heads
        if use_alternative_attention
        else config.num_key_value_heads
    )
    self.num_key_value_groups = (
        config.num_attention_heads // self.num_key_value_heads
    )
    self.scaling = 1.0

    self.q_proj = port_model.QProjLinear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.q_norm = port_model.Gemma4RMSNorm(
        dim=self.head_dim, eps=config.rms_norm_eps
    )
    self.o_proj = port_model.OProjLinear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple,
      attention_mask: Optional[torch.Tensor],
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      use_sdpa_causal: bool = False,
  ) -> torch.Tensor:
    """Attention with externally supplied K/V (already normed + roped).

    `use_sdpa_causal` mirrors the dense prefill fast path: only taken for
    full-attention layers with `attention_mask is None` and q/kv of equal
    length (the unpadded prompt at positions 0..S-1).
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = port_model.apply_rotary_pos_emb(query_states, cos, sin)
    query_states = query_states.transpose(1, 2)

    if use_sdpa_causal and not self.is_sliding and attention_mask is None:
      attn_output = F.scaled_dot_product_attention(
          query_states,
          port_model.repeat_kv(key_states, self.num_key_value_groups),
          port_model.repeat_kv(value_states, self.num_key_value_groups),
          attn_mask=None,
          is_causal=True,
          scale=self.scaling,
      )
      attn_output = attn_output.transpose(1, 2).contiguous()
    else:
      attn_output = port_model.eager_attention_forward(
          self.num_key_value_groups,
          query_states,
          key_states,
          value_states,
          attention_mask,
          scaling=self.scaling,
      )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output)


class Gemma4EDecoderLayer(port_model.Gemma4DenseDecoderLayer):
  """Dense decoder layer plus KV sharing, double-wide MLP and the PLE block."""

  def __init__(self, config: Gemma4EConfig, layer_idx: int):
    super().__init__(config, layer_idx)  # attn, mlp, 4 norms, layer_scalar
    self.layer_idx = layer_idx
    first = config.first_kv_shared_layer_idx
    self.is_kv_shared_layer = layer_idx >= first
    share_map = config.kv_share_map()
    # HF's store_full_length_kv: the last non-shared layer of each type.
    self.store_full_length_kv = (
        not self.is_kv_shared_layer
        and layer_idx == max(
            i for i in range(first)
            if config.layer_types[i] == self.attention_type
        )
    ) if first > 0 else False
    self.kv_source_layer_idx = share_map[layer_idx]

    if self.is_kv_shared_layer:
      # Replace the dense attention (which owns k/v projections) with the
      # shared-KV attention (q_proj / q_norm / o_proj only).
      self.self_attn = Gemma4ESharedAttention(config, layer_idx)
      if config.use_double_wide_mlp and first > 0:  # HF's `> 0` guard
        self.mlp = port_model.Gemma4DenseMLP(
            dataclasses.replace(
                config, intermediate_size=2 * config.intermediate_size
            )
        )

    self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
    if self.hidden_size_per_layer_input:
      self.per_layer_input_gate = PerLayerInputGateLinear(
          config.hidden_size, config.hidden_size_per_layer_input, bias=False
      )
      self.per_layer_projection = PerLayerProjectionLinear(
          config.hidden_size_per_layer_input, config.hidden_size, bias=False
      )
      self.post_per_layer_input_norm = port_model.Gemma4RMSNorm(
          config.hidden_size, eps=config.rms_norm_eps
      )

  def _apply_per_layer_input(
      self, hidden_states: torch.Tensor, per_layer_input: torch.Tensor
  ) -> torch.Tensor:
    """The PLE residual block (runs after the FFW residual add, HF order)."""
    residual = hidden_states
    hidden_states = self.per_layer_input_gate(hidden_states)
    hidden_states = F.gelu(hidden_states, approximate="tanh")
    hidden_states = hidden_states * per_layer_input
    hidden_states = self.per_layer_projection(hidden_states)
    hidden_states = self.post_per_layer_input_norm(hidden_states)
    return residual + hidden_states

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple,
      attention_mask: torch.Tensor,
      per_layer_input: Optional[torch.Tensor] = None,
      shared_kv_states: Optional[dict] = None,
  ) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    if self.is_kv_shared_layer:
      key_states, value_states = shared_kv_states[self.attention_type]
      hidden_states = self.self_attn(
          hidden_states,
          position_embeddings,
          attention_mask,
          key_states,
          value_states,
      )
    else:
      hidden_states, key_states, value_states = _dense_attn_forward_capture(
          self.self_attn, hidden_states, position_embeddings, attention_mask
      )
      if self.store_full_length_kv and shared_kv_states is not None:
        shared_kv_states[self.attention_type] = (key_states, value_states)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    if self.hidden_size_per_layer_input:
      hidden_states = self._apply_per_layer_input(
          hidden_states, per_layer_input
      )

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
      per_layer_input: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Cached-attention variant.

    Non-shared layers write their own cache entries (via the dense
    prefill/decode attention). Shared layers NEVER write: `k_cache` /
    `v_cache` are aliases of the source layer's tensors, already filled by
    the source layer earlier in this same call — prefill reads the
    [:, :, :S, :] slice, decode the full cache under the decode mask.
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    if self.is_kv_shared_layer:
      if is_decode:
        key_states, value_states = k_cache, v_cache
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            key_states,
            value_states,
        )
      else:
        seq_len = hidden_states.shape[1]
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            k_cache[:, :, :seq_len, :],
            v_cache[:, :, :seq_len, :],
            use_sdpa_causal=use_sdpa_causal,
        )
    elif is_decode:
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

    if self.hidden_size_per_layer_input:
      hidden_states = self._apply_per_layer_input(
          hidden_states, per_layer_input
      )

    hidden_states = hidden_states * self.layer_scalar
    return hidden_states


class Gemma4EStaticKVCache(port_model.Gemma4StaticKVCache):
  """Static KV cache with KV sharing: shared layers allocate NO slots.

  `self.k[i]` / `self.v[i]` for a shared layer i are ALIASES of the source
  layer's tensors (`share_map[i]`), so the model's per-layer loops stay
  uniform; only non-shared layers ever write. See the module docstring for
  the slot mapping.
  """

  def __init__(
      self,
      text_model: "Gemma4ETextModel",
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
    self.share_map = text_model.config.kv_share_map()
    num_layers = len(text_model.layers)
    self.k = [None] * num_layers
    self.v = [None] * num_layers
    for i, layer in enumerate(text_model.layers):
      if self.share_map[i] == i:  # source (non-shared) layer: allocate
        attn = layer.self_attn
        shape = (batch_size, attn.num_key_value_heads, max_seq_len,
                 attn.head_dim)
        self.k[i] = torch.zeros(shape, dtype=dtype, device=device)
        self.v[i] = torch.zeros(shape, dtype=dtype, device=device)
    for i in range(num_layers):  # shared layer: alias the source's tensors
      if self.share_map[i] != i:
        self.k[i] = self.k[self.share_map[i]]
        self.v[i] = self.v[self.share_map[i]]

  def reset(self):
    for i, src in enumerate(self.share_map):
      if src == i:
        self.k[i].zero_()
        self.v[i].zero_()


class Gemma4ETextModel(port_model.Gemma4DenseTextModel):
  """Dense text model with E-series layers, PLE embeddings and KV sharing."""

  def __init__(self, config: Gemma4EConfig):
    super().__init__(config)
    self.layers = nn.ModuleList([
        Gemma4EDecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
    ])
    self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
    if self.hidden_size_per_layer_input:
      self.embed_tokens_per_layer = port_model.Gemma4ScaledWordEmbedding(
          config.vocab_size_per_layer_input,
          config.num_hidden_layers * config.hidden_size_per_layer_input,
          config.pad_token_id,
          embed_scale=config.hidden_size_per_layer_input**0.5,
      )
      self.per_layer_input_scale = 2.0**-0.5
      self.per_layer_model_projection = PerLayerModelProjectionLinear(
          config.hidden_size,
          config.num_hidden_layers * config.hidden_size_per_layer_input,
          bias=False,
      )
      self.per_layer_model_projection_scale = config.hidden_size**-0.5
      self.per_layer_projection_norm = port_model.Gemma4RMSNorm(
          config.hidden_size_per_layer_input, eps=config.rms_norm_eps
      )

  def _per_layer_inputs(
      self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
  ) -> Optional[torch.Tensor]:
    """[B, S] ids + [B, S, H] scaled embeds -> [B, S, L, D_ple] PLE tensor.

    HF pipeline: token component = embed_tokens_per_layer(ids) (scaled by
    sqrt(D_ple)), context component = RMSNorm(proj(embeds) / sqrt(H)); the
    combination is (context + token) / sqrt(2).
    """
    if not self.hidden_size_per_layer_input:
      return None
    ple_shape = (
        *input_ids.shape,
        self.config.num_hidden_layers,
        self.hidden_size_per_layer_input,
    )
    token_component = self.embed_tokens_per_layer(input_ids).reshape(ple_shape)
    context = (
        self.per_layer_model_projection(inputs_embeds)
        * self.per_layer_model_projection_scale
    )
    context = self.per_layer_projection_norm(context.reshape(ple_shape))
    return (context + token_component) * self.per_layer_input_scale

  def forward(
      self,
      input_ids: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      output_hidden_states: bool = False,
  ):
    """Full-sequence no-cache forward (the parity reference). Same contract
    as the dense port's forward, plus PLE and KV sharing."""
    _, seq_len = input_ids.shape
    device = input_ids.device
    dtype = self.embed_tokens.weight.dtype

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)

    causal_mask_mapping = {
        "full_attention": port_model._create_causal_mask(
            seq_len, dtype, device
        ),
        "sliding_attention": port_model._create_sliding_window_causal_mask(
            seq_len, self.config.sliding_window, dtype, device
        ),
    }
    if attention_mask is not None:
      causal_mask_mapping = {
          k: port_model._apply_padding_mask(v, attention_mask)
          for k, v in causal_mask_mapping.items()
      }

    hidden_states = self.embed_tokens(input_ids)
    per_layer_inputs = self._per_layer_inputs(input_ids, hidden_states)

    position_embeddings = {
        layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
        for layer_type in set(self.config.layer_types)
    }

    shared_kv_states = {}
    all_hidden_states = [hidden_states] if output_hidden_states else None
    for layer_idx, decoder_layer in enumerate(self.layers):
      per_layer_input = (
          per_layer_inputs[:, :, layer_idx, :]
          if per_layer_inputs is not None
          else None
      )
      hidden_states = decoder_layer(
          hidden_states,
          position_embeddings=position_embeddings[
              decoder_layer.attention_type
          ],
          attention_mask=causal_mask_mapping[decoder_layer.attention_type],
          per_layer_input=per_layer_input,
          shared_kv_states=shared_kv_states,
      )
      if output_hidden_states:
        all_hidden_states.append(hidden_states)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
      return hidden_states, all_hidden_states
    return hidden_states

  # ----- static-KV-cache path ---------------------------------------------

  def prefill(
      self,
      input_ids: torch.Tensor,
      cache: Gemma4EStaticKVCache,
      use_sdpa_causal: bool = True,
  ) -> torch.Tensor:
    """Prompt forward: source layers fill cache rows 0..S-1; shared layers
    read the freshly written source slices. Returns (B, S, hidden)."""
    _, seq_len = input_ids.shape
    device = input_ids.device
    dtype = self.embed_tokens.weight.dtype

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    cache_positions = position_ids  # (S,) tensor index for index_copy_
    position_ids = position_ids.unsqueeze(0)

    masks = {
        "sliding_attention": port_model._create_sliding_window_causal_mask(
            seq_len, self.config.sliding_window, dtype, device
        ),
        "full_attention": (
            None
            if use_sdpa_causal
            else port_model._create_causal_mask(seq_len, dtype, device)
        ),
    }

    hidden_states = self.embed_tokens(input_ids)
    per_layer_inputs = self._per_layer_inputs(input_ids, hidden_states)
    position_embeddings = self._position_embeddings(
        hidden_states, position_ids
    )

    for layer_idx, decoder_layer in enumerate(self.layers):
      per_layer_input = (
          per_layer_inputs[:, :, layer_idx, :]
          if per_layer_inputs is not None
          else None
      )
      hidden_states = decoder_layer._forward_cached(
          hidden_states,
          position_embeddings[decoder_layer.attention_type],
          masks[decoder_layer.attention_type],
          cache.k[layer_idx],
          cache.v[layer_idx],
          cache_positions,
          is_decode=False,
          use_sdpa_causal=use_sdpa_causal,
          per_layer_input=per_layer_input,
      )
    return self.norm(hidden_states)

  def decode_step(
      self,
      input_ids: torch.Tensor,
      pos: torch.Tensor,
      cache: Gemma4EStaticKVCache,
  ) -> torch.Tensor:
    """One decode step ((B, 1) ids at position tensor `pos`). Source layers
    write row `pos` before any of their shared consumers read the (aliased)
    cache — the source always precedes its consumers in the stack. Static
    shapes throughout."""
    device = input_ids.device
    dtype = self.embed_tokens.weight.dtype

    position_ids = pos.view(1, 1)
    masks = port_model._create_decode_masks(
        pos,
        cache.max_seq_len,
        self.config.sliding_window,
        dtype,
        device,
    )

    hidden_states = self.embed_tokens(input_ids)
    per_layer_inputs = self._per_layer_inputs(input_ids, hidden_states)
    position_embeddings = self._position_embeddings(
        hidden_states, position_ids
    )

    for layer_idx, decoder_layer in enumerate(self.layers):
      per_layer_input = (
          per_layer_inputs[:, :, layer_idx, :]
          if per_layer_inputs is not None
          else None
      )
      hidden_states = decoder_layer._forward_cached(
          hidden_states,
          position_embeddings[decoder_layer.attention_type],
          masks[decoder_layer.attention_type],
          cache.k[layer_idx],
          cache.v[layer_idx],
          pos,
          is_decode=True,
          per_layer_input=per_layer_input,
      )
    return self.norm(hidden_states)


class Gemma4EForCausalLM(port_model.Gemma4DenseForCausalLM):
  """Gemma4 E-series text model with tied LM head and softcapped logits.

  `forward` / `prefill` / `decode_step` / `generate_cached` / `_cap_logits`
  are inherited from the dense port; the decoder stack and the KV cache
  factory differ.
  """

  def __init__(self, config: Gemma4EConfig):
    super().__init__(config)
    self.model = Gemma4ETextModel(config)
    if config.tie_word_embeddings:
      self.lm_head.weight = self.model.embed_tokens.weight

  def new_kv_cache(
      self,
      batch_size: int,
      max_seq_len: int,
      dtype: Optional[torch.dtype] = None,
      device: Optional[torch.device] = None,
  ) -> Gemma4EStaticKVCache:
    return Gemma4EStaticKVCache(
        self.model, batch_size, max_seq_len, dtype=dtype, device=device
    )


# HF-name-compatible state dict loading works unchanged (all E-series
# parameter names match HF's).
load_hf_state_dict = port_model.load_hf_state_dict
