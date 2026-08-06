"""v2 MoE expert path: fused w4a16 dequant + expert matmul as a Pallas kernel.

Why v2 exists
-------------
The v1 `W4A16Experts` (quant_experts.py) gathers packed int4 expert slices per
(token, k) pair and dequantizes them IN-GRAPH to the compute dtype before
`torch.bmm`. XLA therefore materializes `[T*K, out, in]` bf16 weight
temporaries in HBM; at real 26B-A4B dims with batch=8 decode the HLO
temporaries reached 42.13 GB > 31 GB HBM. This module fuses the dequant INTO
the matmul kernel: int4 nibbles are unpacked and scaled one VMEM block at a
time, so no dequantized bf16 weight tensor ever exists in HBM.

Design chosen: PRE-GATHERED slices (not scalar-prefetch)
--------------------------------------------------------
The kernel takes already-gathered packed slices `[N, out, in/8]` (N = T*K
selected experts). Gathering packed int32 is cheap — 0.5 B/weight, ~190 MB
transient per layer at 26B dims vs the 42 GB of bf16 temps — it was only the
DEQUANTIZED temporaries that blew up HBM.

`pltpu.PrefetchScalarGridSpec` (scalar-prefetch, index_map reading block
indices from a prefetched expert-id array) was probed and works in interpret
mode on jax 0.11.0, so a gather-free variant is expressible. It was rejected
for v2 because it would need the repeat_interleave(4) scale table resident
per expert (fp32 [E, out, in/8] = +8.6 GB, bf16 +4.3 GB over the non-rep
layout at 26B dims — a real bite out of the ~10 GB HBM headroom), or an
in-kernel lane-repeat that is not part of the proven-on-Mosaic technique
set. The pre-gathered design reuses the exact structure of the proven dense
fused kernel (w4a16_fused_model_bench.py) and is fully sufficient to fix the
batch=8 OOM.

Quantized layout (identical to v1 / compressed-tensors "pack-quantized")
------------------------------------------------------------------------
    packed [E, out, in/8]  int32, stored value q+8 in [0, 15],
                           nibble i of int32 word j = column 8*j + i
    scale  [E, out, in/32] float32, symmetric int4 group_size=32 along `in`

Nibble-plane technique (mandatory on real Mosaic)
-------------------------------------------------
A naive [K/8, 8] -> [K] unpack is a cross-lane interleave and fails to
compile on real TPU even though interpret mode accepts it. Instead, per
contraction chunk of `ck` columns (`ck8 = ck//8` packed words):
  * 8 elementwise shift/mask planes `((p >> 4i) & 0xF) - 8`, each [blk, ck8],
    are concatenated along lanes -> [blk, ck],
  * the activation columns are permuted OUTSIDE the kernel so column
    `c*ck + 8*m + i` of the original space sits at chunk-local position
    `i*ck8 + m` (matching the plane concat),
  * scales are expanded with repeat_interleave(4) along the group axis so a
    plain `[:, ci*ck8:(ci+1)*ck8]` slice of the rep4 tensor is correct for
    every plane (valid because ck % 32 == 0 => ck8 % 4 == 0, and
    8*(m % 4) + i < 32 for all i in [0, 8)).
Here the rep4 expansion is applied AFTER the gather, on the [N, out, in/32]
slices only, so the resident scale table stays in the compact v1 layout.

TPU constraints respected: all blocks are 3D with leading dim 1; the last two
block dims are (multiple-of-8/128, multiple-of-128) or full-size; per-step
VMEM (blocks x2 for double buffering + kernel scratch) is far below the
~64 MB scoped VMEM budget — see `block_plan()` and test_fused_kernel.py.

The gelu-tanh gating (act(gate) * up) is kept OUTSIDE the kernel: it acts on
tiny [N, I] activations (90 KB at 26B dims), contributes nothing to the HBM
problem, and keeping it out lets one kernel serve both projections.

Down-proj lane-width note: at 26B dims down has k = I = 704 -> single chunk
ck = 704, ck8 = 88 (full-size last block dim, allowed; plane width 88 is not
a multiple of 128). If real Mosaic rejects the 88-wide lane concat, pass
`pad_k_to=1024` for the down matmul: contraction is zero-padded (scales are
padded with 0.0, which zeroes the padded weights regardless of nibble bits)
giving fully aligned ck = 1024, ck8 = 128 at a 1.45x weight-read cost.

This file is plain jax + numpy (no torch, no backend import); the TPU-side
torch bridge lives in fused_experts_bench_tpu.py via
the backend's `_internal.pallas.jax_op`, mirroring the proven dense bench.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

GROUP = 32           # int4 quantization group size along the contraction dim
NIB = 8              # nibbles (int4 values) per int32 word
SCALE_REP = GROUP // NIB  # repeat_interleave factor: 1 scale entry per 4 words
VMEM_BUDGET = 64 * 1024 * 1024


def _ck_for(k: int) -> int | None:
    """Contraction chunk size: largest multiple of 128 <= 1024 dividing k,
    else k itself if k <= 1024 (single full-size chunk). Same policy as the
    proven dense kernel."""
    for cand in range(1024, 0, -128):
        if k % cand == 0:
            return cand
    return k if k <= 1024 else None


def _blk_for(out_f: int) -> int:
    """Output-block size along `out`: 256/128 if they tile, else full-size."""
    return 256 if out_f % 256 == 0 else (128 if out_f % 128 == 0 else out_f)


def _perm_for(k: int, ck: int) -> np.ndarray:
    """Static activation-column permutation matching the nibble-plane concat.

    perm[j] = original column that must sit at permuted position j, i.e.
    x_perm[:, j] = x[:, perm[j]]. Within chunk c, permuted position
    i*ck8 + m holds original column c*ck + 8*m + i.
    """
    ck8 = ck // NIB
    j = np.arange(k)
    c, r = j // ck, j % ck
    return c * ck + NIB * (r % ck8) + r // ck8


def _fused_slice_kernel(x_ref, packed_ref, scale_ref, out_ref, *, ck: int):
    """One grid step: (slice n, output block ob) -> out[n, ob*blk:(ob+1)*blk].

    x_ref:      (1, 1, k)      activations, columns pre-permuted (see _perm_for)
    packed_ref: (1, blk, k/8)  int32 packed nibbles for slice n's expert
    scale_ref:  (1, blk, k/8)  group scales, repeat_interleave(4) along lanes
    out_ref:    (1, 1, blk)
    """
    x = x_ref[0]        # [1, k]
    p = packed_ref[0]   # [blk, k/8]
    s4 = scale_ref[0]   # [blk, k/8]
    k = x.shape[-1]
    blk = p.shape[0]
    ck8 = ck // NIB

    acc = jnp.zeros((1, blk), jnp.float32)
    for ci in range(k // ck):
        pc = p[:, ci * ck8:(ci + 1) * ck8]
        # 8 elementwise nibble planes, concatenated along lanes (NOT a
        # [k/8, 8] -> [k] reshape, which is a cross-lane interleave and does
        # not lower on real Mosaic).
        planes = [((pc >> (4 * i)) & 0xF) - 8 for i in range(NIB)]
        w = jnp.concatenate(planes, axis=1).astype(s4.dtype)
        sr = s4[:, ci * ck8:(ci + 1) * ck8]           # plane-independent slice
        w = (w * jnp.concatenate([sr] * NIB, axis=1)).astype(x.dtype)
        acc += jax.lax.dot_general(
            x[:, ci * ck:(ci + 1) * ck], w,
            (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
        )
    out_ref[0] = acc.astype(out_ref.dtype)


def fused_slice_matmul(
    x: jax.Array,
    packed: jax.Array,
    scale_rep4: jax.Array,
    *,
    pad_k_to: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Per-slice fused dequant+matmul: out[n] = x[n] @ dequant(packed[n]).T.

    x          [N, k]         activations (one row per selected expert slice)
    packed     [N, out, k/8]  int32, compressed-tensors pack-quantized
    scale_rep4 [N, out, k/8]  group scales after repeat_interleave(4, axis=-1)
    returns    [N, out]       in x.dtype

    No [N, out, k] dequantized tensor is ever materialized: dequant happens
    inside the kernel on one (blk, ck) tile at a time.
    """
    n, k = x.shape
    out_f = packed.shape[1]
    if packed.shape != (n, out_f, k // NIB):
        raise ValueError(f"packed shape {packed.shape} != {(n, out_f, k // NIB)}")
    if scale_rep4.shape != packed.shape:
        raise ValueError(f"scale_rep4 shape {scale_rep4.shape} != {packed.shape}")
    if k % GROUP:
        raise ValueError(f"k={k} must be divisible by group size {GROUP}")

    if pad_k_to is not None and pad_k_to > k:
        if pad_k_to % GROUP:
            raise ValueError(f"pad_k_to={pad_k_to} must be divisible by {GROUP}")
        dk, dk8 = pad_k_to - k, (pad_k_to - k) // NIB
        x = jnp.pad(x, ((0, 0), (0, dk)))
        packed = jnp.pad(packed, ((0, 0), (0, 0), (0, dk8)))
        # scale 0.0 in the pad region zeroes the dequantized weights there,
        # whatever the (zero-filled) packed nibbles decode to.
        scale_rep4 = jnp.pad(scale_rep4, ((0, 0), (0, 0), (0, dk8)))
        k = pad_k_to

    ck = _ck_for(k)
    if ck is None:
        raise ValueError(f"no contraction chunking for k={k}; use pad_k_to")
    blk = _blk_for(out_f)

    xp = x[:, _perm_for(k, ck)][:, None, :]  # [N, 1, k], static permutation

    out = pl.pallas_call(
        functools.partial(_fused_slice_kernel, ck=ck),
        grid=(n, out_f // blk),
        in_specs=[
            pl.BlockSpec((1, 1, k), lambda nb, ob: (nb, 0, 0)),
            pl.BlockSpec((1, blk, k // NIB), lambda nb, ob: (nb, ob, 0)),
            pl.BlockSpec((1, blk, k // NIB), lambda nb, ob: (nb, ob, 0)),
        ],
        out_specs=pl.BlockSpec((1, 1, blk), lambda nb, ob: (nb, 0, ob)),
        out_shape=jax.ShapeDtypeStruct((n, 1, out_f), x.dtype),
        interpret=interpret,
    )(xp, packed, scale_rep4)
    return out[:, 0, :]


def fused_experts_forward(
    hidden: jax.Array,          # [T, H]
    top_k_index: jax.Array,     # [T, K] int; == num_experts marks padding
    top_k_weights: jax.Array,   # [T, K]
    gu_packed: jax.Array,       # [E, 2I, H/8]  int32
    gu_scale: jax.Array,        # [E, 2I, H/32] (v1 compact layout)
    dn_packed: jax.Array,       # [E, H, I/8]   int32
    dn_scale: jax.Array,        # [E, H, I/32]
    *,
    down_pad_k: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """jax mirror of W4A16Experts.forward with the fused v2 kernel.

    Same math and (token, k) row-major slice order as v1: per selected expert
        gate, up = split(x @ dequant(gate_up[e]).T)
        y = (gelu_tanh(gate) * up) @ dequant(down[e]).T * top_k_weights[t, k]
    summed over K, cast to hidden.dtype. The v1 padding-expert semantics are
    kept: index == num_experts is clamped to 0 and its routing weight zeroed.
    """
    t, h = hidden.shape
    kk = top_k_index.shape[-1]
    num_experts = gu_packed.shape[0]
    two_i = gu_packed.shape[1]

    flat = top_k_index.reshape(-1).astype(jnp.int32)          # [N], N = T*K
    valid = flat < num_experts
    safe = jnp.where(valid, flat, 0)
    w = top_k_weights.reshape(-1) * valid.astype(top_k_weights.dtype)

    # Row-major (token, k) expansion — matches v1's expand/reshape order.
    x = jnp.repeat(hidden, kk, axis=0)                        # [N, H]

    # Gather PACKED slices (0.5 B/weight) + compact scales, then rep4 the
    # gathered scales only — the resident tables stay in the v1 layout.
    gu_p = jnp.take(gu_packed, safe, axis=0)                  # [N, 2I, H/8]
    gu_s = jnp.repeat(jnp.take(gu_scale, safe, axis=0), SCALE_REP, axis=-1)
    gate_up = fused_slice_matmul(x, gu_p, gu_s, interpret=interpret)  # [N, 2I]

    gate, up = gate_up[:, : two_i // 2], gate_up[:, two_i // 2:]
    inter = jax.nn.gelu(gate, approximate=True) * up          # [N, I], tiny

    dn_p = jnp.take(dn_packed, safe, axis=0)                  # [N, H, I/8]
    dn_s = jnp.repeat(jnp.take(dn_scale, safe, axis=0), SCALE_REP, axis=-1)
    out = fused_slice_matmul(
        inter, dn_p, dn_s, pad_k_to=down_pad_k, interpret=interpret
    )                                                          # [N, H]

    out = out * w[:, None].astype(out.dtype)
    return out.astype(hidden.dtype).reshape(t, kk, h).sum(axis=1)


def block_plan(
    out_f: int,
    k: int,
    *,
    x_bytes: int = 2,
    scale_bytes: int = 4,
    pad_k_to: int | None = None,
) -> dict:
    """Static per-grid-step block/VMEM byte math for one projection.

    Returns block shapes plus a conservative resident-VMEM estimate:
    2x every in/out block (double-buffered pipeline) + kernel scratch
    (8 int32 planes, the concatenated fp32-ish weight tile, the scale
    concat, and the fp32 accumulator).
    """
    k_eff = pad_k_to if (pad_k_to and pad_k_to > k) else k
    ck = _ck_for(k_eff)
    if ck is None:
        raise ValueError(f"k={k_eff} needs pad_k_to (no chunking)")
    blk = _blk_for(out_f)
    ck8 = ck // NIB
    x_blk = k_eff * x_bytes
    p_blk = blk * (k_eff // NIB) * 4
    s_blk = blk * (k_eff // NIB) * scale_bytes
    o_blk = blk * x_bytes
    scratch = (
        NIB * blk * ck8 * 4    # 8 int32 nibble planes
        + blk * ck * 4         # concatenated dequant tile (fp32 upper bound)
        + blk * ck * scale_bytes  # concatenated scale tile
        + blk * 4              # fp32 accumulator row
    )
    total = 2 * (x_blk + p_blk + s_blk + o_blk) + scratch
    return {
        "out_f": out_f, "k": k, "k_eff": k_eff, "blk": blk, "ck": ck,
        "ck8": ck8, "grid_out_blocks": out_f // blk,
        "x_block_bytes": x_blk, "packed_block_bytes": p_blk,
        "scale_block_bytes": s_blk, "out_block_bytes": o_blk,
        "kernel_scratch_bytes": scratch, "vmem_resident_bytes": total,
        "fits_64mb": total <= VMEM_BUDGET,
    }
