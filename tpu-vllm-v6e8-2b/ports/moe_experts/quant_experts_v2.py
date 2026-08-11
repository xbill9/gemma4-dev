"""v2 MoE experts: the fused dequant-inside-matmul Pallas kernel behind the
v1 module API.

`W4A16ExpertsV2` is a drop-in subclass of `W4A16Experts` (same buffers, same
quantized layout, same `from_hf`); only `forward`'s dispatch differs:

  * decode-sized calls (T*K <= FUSED_MAX_SLICES): the fused Pallas kernel
    (`fused_experts_kernel.fused_experts_forward` via the backend jax_op
    bridge). The expert-id gather of PACKED int32 slices + compact scales
    happens inside the same compiled jax graph (jnp.take — 0.5 B/weight,
    ~285 MB transient per layer at 26B decode dims) and dequantization
    happens INSIDE the matmul kernel, so no [T*K, out, in] bf16 weight
    temporaries ever exist — the fix for the 42.13 GB HLO-temp OOM.
  * prefill-sized calls: the v1 in-graph dequant + bmm path
    (`super().forward`). At prefill scale (T*K = batch*prefill_len*top_k,
    thousands of slices) the per-slice kernel's gathered transients would be
    multi-GB and its per-block overhead dominates; prefill stays bounded by
    prefill_len exactly as in the v1 bench (use 64 at batch=8).

The branch is on static shapes only, so under torch.compile(dynamic=False)
the prefill graph traces the v1 path and the decode graph traces the fused
kernel — no data-dependent control flow.

jax_op bridge notes (learned on real Mosaic): the wrapped fn must have type
annotations on EVERY parameter and the return, and `jax` must be imported.
The op is registered once at module import. On machines without the backend
(local CPU dev) the module still imports and falls back to the v1 path.
"""

from __future__ import annotations

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import jax  # noqa: E402  (jax_op requires jax.Array annotations below)

import fused_experts_kernel as fek  # noqa: E402
from quant_experts import W4A16Experts  # noqa: E402

try:
    import importlib as _il
    import os as _os
    # Backend package name is operator-supplied (not committed); see TPU_BACKEND_MODULE.
    _tt_pallas = _il.import_module(f"{_os.environ['TPU_BACKEND_MODULE']}._internal.pallas")
except ImportError:  # local CPU development / tests: v1 fallback only
    _tt_pallas = None

# Fused-kernel dispatch ceiling: decode is T*K = batch*top_k (64 at b8);
# prefill is batch*prefill_len*top_k (>= 2048 in any real config). Calls
# above the ceiling take the v1 in-graph path.
FUSED_MAX_SLICES = 256


def _moe_experts_v2_jax(
    hidden: jax.Array,
    top_k_index: jax.Array,
    top_k_weights: jax.Array,
    gu_packed: jax.Array,
    gu_scale: jax.Array,
    dn_packed: jax.Array,
    dn_scale: jax.Array,
) -> jax.Array:
    return fek.fused_experts_forward(
        hidden, top_k_index, top_k_weights,
        gu_packed, gu_scale, dn_packed, dn_scale,
    )


_moe_fused = (
    _tt_pallas.jax_op("w4a16::moe_experts_v2", _moe_experts_v2_jax)
    if _tt_pallas is not None
    else None
)


def fused_op_available() -> bool:
    """True when the backend Pallas bridge registered the fused op."""
    return _moe_fused is not None


class W4A16ExpertsV2(W4A16Experts):
    """W4A16Experts with the fused Pallas kernel on decode-sized dispatch.

    Identical buffers, quantization (`from_hf`) and semantics as v1,
    including the padding-expert clamp (index == num_experts -> weight 0).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        n_slices = hidden_states.shape[0] * top_k_index.shape[-1]
        if _moe_fused is None or n_slices > FUSED_MAX_SLICES:
            return super().forward(hidden_states, top_k_index, top_k_weights)
        return _moe_fused(
            hidden_states,
            top_k_index.to(torch.int32),
            top_k_weights,
            self.gate_up_packed,
            self.gate_up_scale,
            self.down_packed,
            self.down_scale,
        )
