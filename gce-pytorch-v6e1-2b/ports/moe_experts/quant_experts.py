"""TPU-compile-friendly w4a16 replacement for HF transformers' Gemma4TextExperts.

Quantization scheme (DIY, no published w4a16 checkpoint for gemma-4-26B-A4B):
  - symmetric int4, group_size=32 along the LAST dim of each 3D expert tensor,
    i.e. the contraction/input dim of each per-expert matmul
    (gate_up_proj: hidden, down_proj: intermediate)
  - compressed-tensors "pack-quantized" layout per expert slice:
      stored value = q + 8 in [0, 15]
      nibble i of int32 word j holds column 8*j + i
      packed: [E, out, in/8]  int32
      scale:  [E, out, in/32] float32
        (fp32 chosen so dequant is bit-exact against the fp32 reference
         dequantized weights in the parity test; cast to bf16 for the real
         model if the ~2.9 GB fp32 scale footprint matters — see README)

Dispatch (static shapes, TPU-friendly):
  no data-dependent Python loops over hit experts; the top-k experts' packed
  slices are gathered with index_select, dequantized in-graph with
  shift/mask/scale, and applied with torch.bmm per (token, k) pair. All shapes
  are static functions of (tokens, k).

Plain torch only — no backend package import, no PyTorch/XLA.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers.activations import ACT2FN
except ImportError:  # keep the module importable without transformers
    ACT2FN = {"gelu_pytorch_tanh": nn.GELU(approximate="tanh")}

GROUP_SIZE = 32
NIBBLES_PER_INT32 = 8


def quantize_expert_tensor(
    w3d: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D expert weight tensor [E, out, in] to packed int4.

    Symmetric int4 (q in [-8, 7], scale = group_absmax / 7), group_size along
    the last (contraction) dim. Returns:
      packed [E, out, in/8]  int32 (compressed-tensors nibble order:
                                    nibble i of word j = column 8*j + i,
                                    stored value q + 8 in [0, 15])
      scale  [E, out, in/32] float32
    """
    if w3d.dim() != 3:
        raise ValueError(f"expected 3D [E, out, in], got shape {tuple(w3d.shape)}")
    E, out_dim, in_dim = w3d.shape
    if in_dim % group_size != 0 or in_dim % NIBBLES_PER_INT32 != 0:
        raise ValueError(f"in_dim={in_dim} must be divisible by {group_size} and 8")

    wf = w3d.detach().float()
    grouped = wf.reshape(E, out_dim, in_dim // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp(min=torch.finfo(torch.float32).tiny)  # all-zero groups
    q = torch.clamp(torch.round(grouped / scale), -8, 7).to(torch.int64)
    stored = (q + 8).reshape(E, out_dim, in_dim)  # [0, 15]

    nib = stored.reshape(E, out_dim, in_dim // NIBBLES_PER_INT32, NIBBLES_PER_INT32)
    shifts = torch.arange(NIBBLES_PER_INT32, dtype=torch.int64, device=w3d.device) * 4
    packed = (nib << shifts).sum(dim=-1).to(torch.int32)  # disjoint bits: sum == OR
    return packed, scale.reshape(E, out_dim, in_dim // group_size)


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """[..., in/8] int32 -> [..., in] integer values in [-8, 7]. Traceable."""
    shifts = torch.arange(
        NIBBLES_PER_INT32, dtype=torch.int32, device=packed.device
    ) * 4
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF  # arithmetic shift + mask is safe
    return nib.reshape(*packed.shape[:-1], packed.shape[-1] * NIBBLES_PER_INT32) - 8


def dequantize_expert_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype | None = None,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Inverse of quantize_expert_tensor: [.., out, in/8] + [.., out, in/32] -> [.., out, in].

    Works on full [E, out, in/8] tensors and on gathered [N, out, in/8] slices.
    All ops are traceable (shift/mask/mul) so this can run in-graph.
    """
    dtype = dtype or scale.dtype
    q = _unpack_int4(packed).to(dtype)
    in_dim = q.shape[-1]
    grouped = q.reshape(*q.shape[:-1], in_dim // group_size, group_size)
    w = grouped * scale.to(dtype).unsqueeze(-1)
    return w.reshape(*q.shape[:-1], in_dim)


class W4A16Experts(nn.Module):
    """Drop-in w4a16 replacement for Gemma4TextExperts (transformers 5.12.1).

    Same forward signature as the HF eager reference:
        forward(hidden_states [T, H], top_k_index [T, K], top_k_weights [T, K])
    and the same math: per selected expert,
        gate, up = linear(x, gate_up_proj[e]).chunk(2, -1)
        y = linear(act_fn(gate) * up, down_proj[e]) * top_k_weights[t, k]
    summed over the K selected experts and cast to hidden_states.dtype.

    STATIC-SHAPE dispatch: instead of the HF loop over hit experts, gather the
    top-k experts' packed weights per (token, k) pair, dequantize the gathered
    slices in-graph, and batch the two matmuls with torch.bmm. Every shape is a
    static function of (T, K); no .item()/int() on tensors, no data-dependent
    control flow.

    The HF reference skips a padding index equal to num_experts; that is
    reproduced by clamping the gather index and zeroing the routing weight.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        hidden_activation: str = "gelu_pytorch_tanh",
        scale_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = ACT2FN[hidden_activation]

        E, H, I = num_experts, hidden_dim, intermediate_dim
        self.register_buffer(
            "gate_up_packed", torch.zeros(E, 2 * I, H // 8, dtype=torch.int32)
        )
        self.register_buffer(
            "gate_up_scale", torch.zeros(E, 2 * I, H // GROUP_SIZE, dtype=scale_dtype)
        )
        self.register_buffer(
            "down_packed", torch.zeros(E, H, I // 8, dtype=torch.int32)
        )
        self.register_buffer(
            "down_scale", torch.zeros(E, H, I // GROUP_SIZE, dtype=scale_dtype)
        )

    @classmethod
    def from_hf(cls, experts_module: nn.Module) -> "W4A16Experts":
        """Quantize an existing (HF) Gemma4TextExperts instance."""
        gate_up = experts_module.gate_up_proj  # [E, 2I, H]
        down = experts_module.down_proj        # [E, H, I]
        E, H, I = down.shape
        mod = cls(num_experts=E, hidden_dim=H, intermediate_dim=I)
        mod.act_fn = experts_module.act_fn  # exact same activation object/impl
        gu_packed, gu_scale = quantize_expert_tensor(gate_up.data)
        dn_packed, dn_scale = quantize_expert_tensor(down.data)
        mod.gate_up_packed.copy_(gu_packed)
        mod.gate_up_scale.copy_(gu_scale)
        mod.down_packed.copy_(dn_packed)
        mod.down_scale.copy_(dn_scale)
        return mod

    def dequantized_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(gate_up_proj [E, 2I, H], down_proj [E, H, I]) as float tensors."""
        return (
            dequantize_expert_tensor(self.gate_up_packed, self.gate_up_scale),
            dequantize_expert_tensor(self.down_packed, self.down_scale),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        T, H = hidden_states.shape
        K = top_k_index.shape[-1]
        compute_dtype = hidden_states.dtype

        flat_idx = top_k_index.reshape(-1)  # [T*K], row-major: token t, slot k
        # HF eager skips expert_idx == num_experts (routing padding): clamp the
        # gather index and zero that slot's weight instead of branching.
        valid = flat_idx < self.num_experts
        safe_idx = torch.where(valid, flat_idx, torch.zeros_like(flat_idx))
        w = top_k_weights.reshape(-1) * valid.to(top_k_weights.dtype)  # [T*K]

        # ---- gate_up: gather packed slices, dequantize in-graph, bmm ----
        gu = dequantize_expert_tensor(
            self.gate_up_packed.index_select(0, safe_idx),  # [T*K, 2I, H/8]
            self.gate_up_scale.index_select(0, safe_idx),   # [T*K, 2I, H/32]
            dtype=compute_dtype,
        )  # [T*K, 2I, H]
        x = hidden_states.unsqueeze(1).expand(T, K, H).reshape(T * K, H, 1)
        gate_up = torch.bmm(gu, x).squeeze(-1)  # [T*K, 2I] == linear(x, W_e)
        gate, up = gate_up.chunk(2, dim=-1)     # HF order: [gate ; up]
        inter = self.act_fn(gate) * up          # [T*K, I]

        # ---- down: gather, dequantize, bmm ----
        dn = dequantize_expert_tensor(
            self.down_packed.index_select(0, safe_idx),  # [T*K, H, I/8]
            self.down_scale.index_select(0, safe_idx),   # [T*K, H, I/32]
            dtype=compute_dtype,
        )  # [T*K, H, I]
        out = torch.bmm(dn, inter.unsqueeze(-1)).squeeze(-1)  # [T*K, H]

        # HF: weight by top_k_weights[token, k] then cast to output dtype.
        out = out * w.unsqueeze(-1)
        out = out.to(hidden_states.dtype).reshape(T, K, H).sum(dim=1)
        return out
