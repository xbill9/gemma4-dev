"""Gemma 4 E-series (MatFormer) text model — Pure JAX / Flax implementation for gemma-4-E2B.

Clean-room JAX implementation of Gemma 4 E2B MatFormer architecture for Google Cloud TPUs:
  * Pure JAX / Flax / Pallas — ZERO dependence on PyTorch, transformers, or proprietary/confidential repos.
  * Full MatFormer support for Gemma 4 E2B:
    - Per-Layer Embeddings (PLE): Dual token embedding + context projection RMSNorm over D_ple.
    - KV-Sharing across layers: Last 20 of 35 layers reuse KV states from source layers.
    - Double-Wide MLP for shared layers.
    - Dual Attention Geometries (sliding window & global head dim with partial RoPE).
    - Logit softcapping (30.0), scale-less RMSNorms, tied embeddings.
  * QAT (Quantization-Aware Training) Support:
    - Int8 symmetric quantization (matrix multiply + scaling).
    - W4A16 packed int4 quantization with grouped scaling for TPU MXUs via jax.lax / Pallas.
  * Static-shape KV-cache for jax.jit compilation on TPU v4/v5e/v5p/v6e.
"""

import dataclasses
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import jax
import jax.numpy as jnp
from jax import lax

logger = logging.getLogger(__name__)

# Pallas has no TPU backend on a CPU host; interpret mode lets the fused W4A16
# kernel run (slowly) so its numerics can be tested off-TPU. Auto-enabled when no
# TPU is present; override with JAX_E_PALLAS_INTERPRET=1/0.
_PALLAS_INTERPRET = os.environ.get(
    "JAX_E_PALLAS_INTERPRET",
    "1" if not any(d.platform == "tpu" for d in jax.devices()) else "0",
) == "1"

# Enable native TPU MXU bfloat16 matmul precision by default
jax.config.update("jax_default_matmul_precision", "bfloat16")

# Persistent JAX XLA Compilation Disk Cache (skips ~17s compilation on restart)
_cache_dir = os.path.expanduser("~/.cache/jax_compilation_cache")
os.makedirs(_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@dataclasses.dataclass
class Gemma4EConfig:
    """Configuration for Gemma 4 E2B MatFormer JAX Model."""
    vocab_size: int = 262144
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    num_global_key_value_heads: int = 4
    global_head_dim: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    global_rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.25
    logit_softcapping: float = 30.0        # final logits only
    # Gemma 3+ dropped softcapping of ATTENTION scores; the E2B checkpoint declares
    # final_logit_softcapping but no attn_logit_softcapping. Softcapping the scores
    # saturates tanh and destroys the attention distribution.
    attn_logit_softcapping: float = 0.0
    # attention_k_eq_v: V is K — one projection feeds both, and the checkpoint
    # ships no v_proj on the affected layers. True on gemma-4-31B (its 10
    # full-attention layers have k_proj/k_norm and no v_proj at all); False on
    # E2B, which ships v_proj everywhere. The loader aliases V to K when set;
    # the cache still stores both, which is redundant but correct.
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 20
    sliding_window: Optional[int] = None
    use_double_wide_mlp: bool = True
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262144
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.layer_types is None:
            # Default Gemma 4 E2B pattern: interleaved sliding and full attention
            self.layer_types = [
                "sliding_attention" if (i % 5 != 4) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def kv_share_map(self) -> List[int]:
        """Maps each layer index to the source layer index for KV state sharing."""
        first = self.first_kv_shared_layer_idx
        last_of_type = {}
        for i in range(first):
            last_of_type[self.layer_types[i]] = i
        return [
            i if i < first else last_of_type[self.layer_types[i]]
            for i in range(self.num_hidden_layers)
        ]


# ==============================================================================
# JAX Primitives & QAT Ops (W4A16 & Int8)
# ==============================================================================

def rms_norm_jax(x: jax.Array, weight: Optional[jax.Array] = None, eps: float = 1e-6) -> jax.Array:
    """RMSNorm in JAX matching Gemma 4 spec (computed in float32)."""
    dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = (x_f32 * lax.rsqrt(var + eps)).astype(dtype)
    if weight is not None:
        normed = normed * weight
    return normed


def qat_int8_matmul_jax(x: jax.Array, weight_int8: jax.Array, scale: jax.Array) -> jax.Array:
    """Int8 Symmetric QAT Matrix Multiplication for TPU in JAX."""
    # x: [..., K], weight_int8: [K, N], scale: [N]
    w_fp = weight_int8.astype(x.dtype) * scale
    return jnp.matmul(x, w_fp)


def qat_w4a16_unpack_dequant_jax(
    packed_int4: jax.Array,
    scale: jax.Array,
    group_size: int = 32,
) -> jax.Array:
    """Decode compressed-tensors ``pack-quantized`` W4A16 weights.

    The Gemma 4 checkpoints store a linear weight in its native HF orientation:
    ``packed_int4[out, in/8]`` as int32 and ``scale[out, in/32]`` as BF16.
    Nibble ``i`` of word ``j`` is input column ``8*j+i`` and stores ``q + 8``.
    The returned array is BF16 ``[out, in]``.

    This reference implementation intentionally materializes the dequantized
    layer weight. It is correctness-first; replace it with a fused Pallas
    dequant-matmul before calling W4A16 performance-optimal.
    """
    if packed_int4.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"W4A16 expects rank-2 packed/scale arrays, got "
            f"{packed_int4.shape} and {scale.shape}"
        )
    if packed_int4.dtype != jnp.int32:
        raise TypeError(
            f"W4A16 packed weights must be int32, got {packed_int4.dtype}"
        )

    out_features, packed_k = packed_int4.shape
    in_features = packed_k * 8
    expected_scale_shape = (out_features, in_features // group_size)
    if in_features % group_size or scale.shape != expected_scale_shape:
        raise ValueError(
            f"W4A16 scale shape {scale.shape} does not match packed shape "
            f"{packed_int4.shape}; expected {expected_scale_shape}"
        )

    shifts = (jnp.arange(8, dtype=jnp.int32) * 4)[None, None, :]
    words = packed_int4[:, :, None]
    q = ((words >> shifts) & jnp.int32(0xF)).reshape(
        out_features, in_features
    )
    q = q.astype(jnp.bfloat16) - jnp.bfloat16(8)
    # Apply the group scale by BROADCASTING over the group axis, not by
    # materializing it to full weight width. `jnp.repeat(scale, 32, axis=1)`
    # expands [out, in/32] -> [out, in] (a 32x blow-up, e.g. 1.18 MB -> 37.7 MB per
    # projection) and a decode step does ~245 of them; the profile attributed 18.6%
    # of the step to broadcast ops. Reshaping to expose the group axis lets the
    # multiply broadcast instead.
    grouped = q.reshape(out_features, in_features // group_size, group_size)
    scaled = grouped * scale.astype(jnp.bfloat16)[:, :, None]
    return scaled.reshape(out_features, in_features)


# W4A16 execution strategy.
#
# MEASURED on TPU v6e-1 (2026-07-28, benchmarks/runs/2026-07-28-jax-e2b-v6e1):
# the fused Pallas kernel is a REGRESSION, not a win. Decode step vs the
# reference path: 0.59x at B=1 (34.9ms vs 20.6ms) and 0.21x at B=2, and it
# CompileTimeScopedVmemOom'd on 5 of 8 cells because the kernel loads all of x
# as one VMEM block instead of tiling the sequence axis.
#
# The theory said 4-bit weights should cut decode traffic ~4x. It didn't help
# because decode here is not bandwidth-bound: the pure-bandwidth floor is
# ~3.6ms/token at B=1 but the measured step is 20.6ms, ~6x above it. Cutting
# bytes read cannot speed up a step that is not waiting on bytes.
#
# So "reference" is the default until the kernel tiles properly and the real
# bottleneck is identified. Keep the fused path selectable for that work.
#   "reference" — dequantize-then-matmul (DEFAULT; fastest measured)
#   "fused"     — require the fused Pallas kernel; raise if unavailable
#   "auto"      — try fused, fall back to reference with a warning
_W4A16_IMPL = "reference"
_W4A16_LAYOUT = "plane"
_W4A16_FUSED_OK: Optional[bool] = None
_W4A16_WARNED = False


def set_w4a16_impl(impl: str = "auto", layout: str = "plane") -> None:
    """Select the W4A16 matmul strategy.

    layout picks how the kernel materializes the int4 nibbles in-tile:
      "plane"       — nibble i of word j -> column i*(K/8)+j, with the matching
                      per-chunk activation permutation applied first. THE DEFAULT:
                      it is the only variant that fits TPU VMEM at real sizes.
      "interleaved" — nibble i of word j -> column 8j+i (checkpoint order). Exact,
                      and fine under Pallas interpret mode, but the in-tile
                      stack/reshape blows the 32 MB scoped-VMEM limit on v6e
                      (measured 35.43 MB for a 1024x2048 projection). Kept for
                      reference/testing; do not use it on TPU.
    """
    global _W4A16_IMPL, _W4A16_LAYOUT, _W4A16_FUSED_OK
    if impl not in ("auto", "fused", "reference"):
        raise ValueError(f"impl must be auto|fused|reference, got {impl!r}")
    if layout not in ("interleaved", "plane"):
        raise ValueError(f"layout must be interleaved|plane, got {layout!r}")
    _W4A16_IMPL, _W4A16_LAYOUT = impl, layout
    _W4A16_FUSED_OK = None


def qat_w4a16_linear_jax(x: jax.Array, packed_int4: jax.Array, scale: jax.Array, group_size: int = 32) -> jax.Array:
    """W4A16 QAT linear. Uses the fused Pallas kernel unless told otherwise."""
    global _W4A16_FUSED_OK, _W4A16_WARNED
    if _W4A16_IMPL != "reference" and group_size == 32 and _W4A16_FUSED_OK is not False:
        try:
            out = qat_w4a16_pallas_matmul_jax(x, packed_int4, scale, layout=_W4A16_LAYOUT)
            _W4A16_FUSED_OK = True
            return out
        except Exception as exc:
            if _W4A16_IMPL == "fused":
                raise
            _W4A16_FUSED_OK = False
            if not _W4A16_WARNED:
                _W4A16_WARNED = True
                # Loud on purpose: the previous silent fallback hid a kernel that
                # returned wrong values wherever Pallas actually compiled.
                logger.warning(
                    "W4A16 fused Pallas kernel unavailable (%s); falling back to "
                    "dequantize-then-matmul. Decode will read BF16 weights.", exc,
                )
    return qat_w4a16_reference_linear_jax(x, packed_int4, scale, group_size=group_size)


def qat_w4a16_reference_linear_jax(x: jax.Array, packed_int4: jax.Array, scale: jax.Array, group_size: int = 32) -> jax.Array:
    """Correctness reference: materialize the BF16 weight, then matmul."""
    w_dequant = qat_w4a16_unpack_dequant_jax(packed_int4, scale, group_size=group_size)
    return jnp.matmul(x, w_dequant.T)


def _permute_activations_for_plane_layout(x: jax.Array, chunk: int) -> jax.Array:
    """Reorder x columns to match the kernel's per-chunk plane-major weight layout.

    Within each `chunk`-wide slice, plane-major column i*(chunk/8)+j holds the
    weight for true column 8j+i, so the activations must be reordered the same
    way. Cheap: a reshape/transpose on [seq, K], and at decode seq is 1.
    """
    seq, k = x.shape
    return x.reshape(seq, k // chunk, chunk // 8, 8).transpose(0, 1, 3, 2).reshape(seq, k)


def qat_w4a16_pallas_matmul_jax(
    x: jax.Array,
    packed_int4: jax.Array,
    scale: jax.Array,
    layout: str = "plane",
) -> jax.Array:
    """Fused W4A16 dequantization + matmul: int4 is unpacked inside the VMEM tile.

    Reads packed int32 weights (8 nibbles each) and per-32-column BF16 scales
    straight from HBM, so weight traffic is ~4x lower than the reference path.
    """
    if x.ndim == 3:
        B, S, K = x.shape
        out_2d = qat_w4a16_pallas_matmul_jax(x.reshape(B * S, K), packed_int4, scale, layout=layout)
        return out_2d.reshape(B, S, packed_int4.shape[0])

    seq, k = x.shape
    out_f = packed_int4.shape[0]
    if packed_int4.shape[1] * 8 != k:
        raise ValueError(f"packed weight expects K={packed_int4.shape[1] * 8}, got x with K={k}")
    blk = 256 if out_f % 256 == 0 else (128 if out_f % 128 == 0 else out_f)
    ck = 256 if k % 256 == 0 else (128 if k % 128 == 0 else k)
    if ck % 32 or k % ck:
        raise ValueError(f"K={k} does not tile into 32-aligned chunks for the fused kernel")
    ck8, ck32 = ck // 8, ck // 32

    from jax.experimental import pallas as pl

    plane = layout == "plane"
    x_in = _permute_activations_for_plane_layout(x, ck) if plane else x

    def kernel(x_ref, packed_ref, scale_ref, out_ref):
        x_all, p, sc = x_ref[...], packed_ref[...], scale_ref[...]
        acc = jnp.zeros((seq, blk), jnp.float32)
        for ci in range(k // ck):
            pc = p[:, ci * ck8 : (ci + 1) * ck8]
            sc_c = sc[:, ci * ck32 : (ci + 1) * ck32]
            planes = [((pc >> (4 * i)) & 0xF) - 8 for i in range(8)]
            if plane:
                # column i*(ck/8)+j  <- nibble i of word j; that column's group is j//4
                w = jnp.concatenate(planes, axis=1)
                s_exp = jnp.concatenate([jnp.repeat(sc_c, 4, axis=1)] * 8, axis=1)
            else:
                # column 8j+i <- nibble i of word j; that column's group is (8j+i)//32
                w = jnp.stack(planes, axis=-1).reshape(blk, ck)
                s_exp = jnp.repeat(sc_c, 32, axis=1)
            w = w.astype(sc.dtype) * s_exp
            acc += jax.lax.dot_general(
                x_all[:, ci * ck : (ci + 1) * ck],
                w.T,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        out_ref[...] = acc.astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        grid=(out_f // blk,),
        in_specs=[
            pl.BlockSpec((seq, k), lambda i: (0, 0)),
            pl.BlockSpec((blk, k // 8), lambda i: (i, 0)),
            pl.BlockSpec((blk, k // 32), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((seq, blk), lambda i: (0, i)),
        out_shape=jax.ShapeDtypeStruct((seq, out_f), jnp.bfloat16),
        interpret=_PALLAS_INTERPRET,
    )(x_in, packed_int4, scale.astype(jnp.bfloat16))


# ==============================================================================
# Rotary Position Embedding (RoPE)
# ==============================================================================

def rotate_half_jax(x: jax.Array) -> jax.Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope_jax(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    partial_factor: float = 1.0,
) -> jax.Array:
    """Applies RoPE to x. If partial_factor < 1.0, only the first fraction is rotated."""
    if partial_factor < 1.0:
        # Reference "proportional" RoPE keeps the full head_dim and simply zeroes
        # the inverse frequencies past `partial_factor`, so unrotated channels pass
        # through as cos=1, sin=0. Reproduce that by masking the frequency table
        # rather than slicing channels, which keeps the cat(freqs, freqs) pairing
        # intact.
        d = x.shape[-1]
        half = d // 2
        n_rot = int(partial_factor * d // 2)          # rotated frequency pairs
        keep = (jnp.arange(half) < n_rot).astype(cos.dtype)
        keep_full = jnp.concatenate([keep, keep], axis=-1)
        cos_p = cos * keep_full + (1.0 - keep_full)   # cos -> 1 where unrotated
        sin_p = sin * keep_full                       # sin -> 0 where unrotated
        return (x * cos_p) + (rotate_half_jax(x) * sin_p)
    else:
        return (x * cos) + (rotate_half_jax(x) * sin)


# ==============================================================================
# Attention & Decoder Layer Primitives
# ==============================================================================

_MASK_MIN = -1e9


def make_ring_decode_mask(valid_mask: jax.Array, window: int, slot: jax.Array) -> jax.Array:
    """Additive mask over a windowed sliding layer's ring buffer at decode.

    Ring slot r holds the most recent absolute position p <= slot for which
    p % window == r. Read `valid` at that position.

    The obvious shortcut — "slots [0, slot] are filled, everything else is not" —
    is wrong, and wrong silently. The cache is NOT filled contiguously: the
    server right-pads every prompt to a static bucket and then decodes at
    bucket + step while the logical position tracks the real length
    (jax_engine.py:434), so the pad slots [real_len, bucket) sit *inside*
    [0, slot). Assuming a contiguous fill attends to pad K/V and corrupts the
    output with no error at all. The non-windowed path has always taken `valid`
    (see make_decode_mask); this has to as well.

    Returns [B, 1, 1, window].
    """
    B, T = valid_mask.shape
    s = jnp.asarray(slot, jnp.int32)
    r = jnp.arange(window, dtype=jnp.int32)[None, :]
    # Most recent absolute position living in each ring slot; negative before the
    # ring has reached that slot at all.
    pos = s - jnp.mod(s - r, jnp.int32(window))
    gather = jnp.broadcast_to(jnp.clip(pos, 0, T - 1), (B, window))
    real = jnp.take_along_axis(valid_mask, gather, axis=1) & (pos >= 0)
    return jnp.where(real[:, None, None, :], 0.0, _MASK_MIN).astype(jnp.float32)


def make_prefill_causal_mask(valid_mask: jax.Array, window: Optional[int] = None) -> jax.Array:
    """Additive causal attention mask for prefill.

    valid_mask: [B, S] bool — True for real (non-pad) tokens.
    window: if set, a sliding-attention layer may only see the last `window`
      positions (Gemma 4 E2B declares sliding_window: 512 for 28 of its 35 layers).
    Returns [B, 1, S, S] float mask: 0 where query i may attend key j, -1e9 otherwise.
    """
    B, S = valid_mask.shape
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    if window is not None:
        # keep j in (i - window, i]
        causal = causal & (~jnp.tril(jnp.ones((S, S), dtype=jnp.bool_), -int(window)))
    allowed = causal[None, None, :, :] & valid_mask[:, None, None, :]
    return jnp.where(allowed, 0.0, _MASK_MIN).astype(jnp.float32)


def make_decode_mask(valid_mask: jax.Array, window: Optional[int] = None,
                     slot: Optional[jax.Array] = None) -> jax.Array:
    """Additive attention mask for a single cached decode step.

    valid_mask: [B, S_total] bool — True for cache slots holding real tokens.
    window/slot: if set, restrict to cache positions in (slot - window, slot].
    Returns [B, 1, 1, S_total] float mask.
    """
    allowed = valid_mask
    if window is not None and slot is not None:
        idx = jnp.arange(valid_mask.shape[1])[None, :]
        allowed = allowed & (idx > jnp.asarray(slot, jnp.int32) - int(window))
    return jnp.where(allowed[:, None, None, :], 0.0, _MASK_MIN).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Quantized KV cache
# ---------------------------------------------------------------------------
# Storing the cache in 8 bits needs two things that a plain `dtype=` argument
# does not provide, and the absence of either fails in a different way:
#
#   1. An explicit read-side cast. JAX deliberately excludes float8 from its
#      type-promotion lattice — there are several mutually incompatible fp8
#      layouts (e4m3/e5m2, differing NaN conventions), so no implicit rule is
#      defensible — and an fp8 buffer meeting a bf16 query in an einsum raises
#      rather than promoting. int8 fails worse: it IS in the lattice, so
#      bf16 x int8 silently succeeds and contracts against raw integers.
#   2. A scale. 8 bits cannot hold attention activations directly.
#
# Scales are symmetric and per (batch, head, position) over the head_dim axis.
# That is the finest granularity which still factors out of both attention
# contractions (see `eager_attention_jax`), so no dequantized copy of K or V is
# ever materialized and the bandwidth saving survives. Cost is one bf16 scalar
# per head_dim row: 0.8% on top of an int8 cache at D=256.
_KV_SCALE_DTYPE = jnp.bfloat16


def is_quantized_kv_dtype(dtype) -> bool:
    """True for cache dtypes that cannot hold activations without a scale."""
    return jnp.dtype(dtype).itemsize == 1


def _kv_quant_max(dtype) -> float:
    dt = jnp.dtype(dtype)
    if jnp.issubdtype(dt, jnp.integer):
        return float(jnp.iinfo(dt).max)
    return float(jnp.finfo(dt).max)


def quantize_kv(x: jax.Array, dtype) -> Tuple[jax.Array, jax.Array]:
    """Symmetric per-(batch, head, position) quantization over head_dim.

    Returns (quantized [B, H, S, D], scale [B, H, S, 1]).
    """
    qmax = _kv_quant_max(dtype)
    x_f32 = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x_f32), axis=-1, keepdims=True)
    # An all-zero row is a real case (an unwritten cache slot). Its amax is 0,
    # which would give a 0 scale and NaN on the reciprocal; pin those to 1.0.
    scale = jnp.where(amax > 0, amax / qmax, 1.0)
    q = x_f32 / scale
    if jnp.issubdtype(jnp.dtype(dtype), jnp.integer):
        q = jnp.clip(jnp.round(q), -qmax, qmax)
    return q.astype(dtype), scale.astype(_KV_SCALE_DTYPE)


def _unpack_kv(entry):
    """Normalize a cache entry to (k, v, k_scale, v_scale).

    Unquantized entries stay 2-tuples so existing callers that write
    ``for k, v in cache.values()`` keep working; quantized ones carry scales.
    """
    if entry is None:
        return None, None, None, None
    if len(entry) == 4:
        return entry
    k, v = entry
    return k, v, None, None


def _pack_kv(k, v, k_scale, v_scale):
    return (k, v) if k_scale is None else (k, v, k_scale, v_scale)


def make_chunk_mask(valid_mask: jax.Array, chunk_len: int, slot,
                    window: Optional[int] = None) -> jax.Array:
    """Mask for a prefill CHUNK attending over the whole cache.

    One-shot prefill attends over the S freshly computed keys, so its mask is
    S x S. A chunk must instead see every earlier chunk already in the cache, so
    the mask is chunk_len x cache_len: row s (absolute position slot + s) may
    attend to cache slot t when t holds a real token and t <= slot + s.

    Returns [B, 1, chunk_len, cache_len].
    """
    T = valid_mask.shape[1]
    t_idx = jnp.arange(T)[None, None, :]                                    # [1, 1, T]
    s_abs = jnp.asarray(slot, jnp.int32) + jnp.arange(chunk_len)[None, :, None]
    allowed = valid_mask[:, None, :] & (t_idx <= s_abs)                     # [B, S, T]
    if window is not None:
        allowed = allowed & (t_idx > s_abs - int(window))
    return jnp.where(allowed[:, None, :, :], 0.0, _MASK_MIN).astype(jnp.float32)


def eager_attention_jax(
    query: jax.Array,   # [B, H, S, D]
    key: jax.Array,     # [B, H_kv, S_kv, D]
    value: jax.Array,   # [B, H_kv, S_kv, D]
    mask: Optional[jax.Array] = None,
    scaling: float = 1.0,
    softcap: float = 30.0,
    key_scale: Optional[jax.Array] = None,     # [B, H_kv, S_kv, 1]
    value_scale: Optional[jax.Array] = None,   # [B, H_kv, S_kv, 1]
) -> jax.Array:
    """Eager Multi-Head Attention with grouped-query broadcast and logit softcapping.

    GQA is expressed by reshaping the query into its KV groups and contracting
    against the un-replicated K/V, rather than `jnp.repeat`-ing K/V up to the
    query head count. The repeat materialized n_rep copies of the whole KV cache
    on every decode step — pure HBM traffic in the phase that is bandwidth-bound
    (E2B ships a single KV head against 8 query heads, so n_rep = 8).

    When key/value come from a quantized cache their scales are applied to the
    contraction RESULT rather than to K/V themselves. Both are exact rewrites:
    the K scale is indexed by the key position t while the score sums over the
    head dim d, so it factors straight out; the V scale is also indexed by t,
    which is the axis the output sums over, so it folds into the probabilities
    instead. Neither ever materializes a widened copy of the cache.
    """
    B, num_heads, S, D = query.shape
    num_kv_heads = key.shape[1]
    n_rep = num_heads // num_kv_heads

    # The cast JAX refuses to do implicitly. Elementwise and fusible into the
    # dot's operand read, so the 1-byte HBM traffic is what actually happens.
    if key.dtype != query.dtype:
        key = key.astype(query.dtype)
    if value.dtype != query.dtype:
        value = value.astype(query.dtype)

    if n_rep > 1:
        q = query.reshape(B, num_kv_heads, n_rep, S, D)
        # [B, Hkv, n_rep, S, D] x [B, Hkv, S_kv, D] -> [B, Hkv, n_rep, S, S_kv]
        scores = jnp.einsum("bgnsd,bgtd->bgnst", q, key)
        if key_scale is not None:
            scores = scores * key_scale[..., 0][:, :, None, None, :].astype(scores.dtype)
        scores = scores.reshape(B, num_heads, S, -1) * scaling
    else:
        scores = jnp.matmul(query, jnp.swapaxes(key, -1, -2))
        if key_scale is not None:
            scores = scores * key_scale[..., 0][:, :, None, :].astype(scores.dtype)
        scores = scores * scaling

    if softcap > 0.0:
        scores = jnp.tanh(scores / softcap) * softcap

    if mask is not None:
        scores = scores + mask

    # The V scale is applied in f32, before the downcast, so folding it in costs
    # no precision relative to scaling a dequantized V.
    probs_f32 = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
    if value_scale is not None:
        vs = value_scale[..., 0].astype(jnp.float32)          # [B, Hkv, S_kv]
        if n_rep > 1:
            # Regroup into KV groups to broadcast one scale across the n_rep query
            # heads that share a KV head. A reshape, not a repeat — the scale array
            # is never widened to the query head count.
            p5 = probs_f32.reshape(B, num_kv_heads, n_rep, S, -1)
            probs_f32 = (p5 * vs[:, :, None, None, :]).reshape(B, num_heads, S, -1)
        else:
            probs_f32 = probs_f32 * vs[:, :, None, :]
    attn_probs = probs_f32.astype(query.dtype)

    if n_rep > 1:
        probs = attn_probs.reshape(B, num_kv_heads, n_rep, S, -1)
        out = jnp.einsum("bgnst,bgtd->bgnsd", probs, value)
        return out.reshape(B, num_heads, S, D).astype(query.dtype)
    return jnp.matmul(attn_probs, value)


def _ring_store(k_buf: jax.Array, v_buf: jax.Array, k: jax.Array, v: jax.Array):
    """Store a prefill's K/V into ring buffers that may be shorter than the prompt.

    Position p lives at slot ``p % buf_len``. When the prompt is longer than the
    buffer (a windowed sliding layer), only the last ``buf_len`` positions are kept,
    written as at most two contiguous slices — the split point is static, so this
    stays jit-friendly.
    """
    return _ring_store_one(k_buf, k), _ring_store_one(v_buf, v)


def _ring_store_one(buf: jax.Array, val: jax.Array) -> jax.Array:
    """Ring-store a single [B, H, S, X] tensor. See `_ring_store`.

    Split out so the per-token scale buffers of a quantized cache — same leading
    axes, X=1 instead of head_dim — reuse the identical slot arithmetic.
    """
    buf_len = buf.shape[2]
    S = val.shape[2]
    if S <= buf_len:
        return jax.lax.dynamic_update_slice(buf, val.astype(buf.dtype), (0, 0, 0, 0))

    start = S - buf_len                 # first position we keep
    off = start % buf_len               # its ring slot
    head = buf_len - off                # positions written at slots off..buf_len-1
    tail = val[:, :, start:, :]
    out = jax.lax.dynamic_update_slice(buf, tail[:, :, :head, :].astype(buf.dtype), (0, 0, off, 0))
    if off:
        out = jax.lax.dynamic_update_slice(out, tail[:, :, head:, :].astype(buf.dtype), (0, 0, 0, 0))
    return out


class Gemma4EAttentionJAX:
    """Gemma 4 E2B Multi-Head Attention layer in pure JAX."""

    def __init__(self, config: Gemma4EConfig, layer_idx: int):
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.is_shared = layer_idx >= config.first_kv_shared_layer_idx

        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim
        self.num_kv_heads = config.num_global_key_value_heads if (not self.is_sliding) else config.num_key_value_heads
        self.softcap = config.attn_logit_softcapping
        # Reference Gemma4 attention uses scaling = 1.0, NOT head_dim ** -0.5:
        # q_norm/k_norm already normalize the query and key before the dot product,
        # so the usual 1/sqrt(d) is not applied on top.
        self.scaling = 1.0

    def __call__(
        self,
        hidden_states: jax.Array,
        params: Dict[str, jax.Array],
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        kv_shared_states: Optional[Tuple[jax.Array, jax.Array]] = None,
        quant_mode: str = "fp16",
        kv_cache: Optional[Tuple[jax.Array, jax.Array]] = None,
        cache_slot: Optional[jax.Array] = None,
        chunked: bool = False,
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        B, S, _ = hidden_states.shape
        kv_out_override = None
        # Scales for the K/V actually handed to attention. Stay None for a bf16
        # cache, so the unquantized path is byte-for-byte what it was.
        key_scale = value_scale = None

        # Query projection
        if quant_mode == "w4a16":
            q = qat_w4a16_linear_jax(hidden_states, params["q_proj_packed"], params["q_proj_scale"])
        else:
            q = jnp.matmul(hidden_states, params["q_proj"])

        q = q.reshape(B, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        q = rms_norm_jax(q, params.get("q_norm"))

        # Key / Value projections
        if self.is_shared:
            assert kv_shared_states is not None, "Shared attention layers require source KV states."
            # A KV-shared layer inherits the source layer's quantization state too:
            # during decode those are the source's cache buffers, scales and all.
            k, v, key_scale, value_scale = _unpack_kv(kv_shared_states)
        else:
            if quant_mode == "w4a16":
                k = qat_w4a16_linear_jax(hidden_states, params["k_proj_packed"], params["k_proj_scale"])
                v = qat_w4a16_linear_jax(hidden_states, params["v_proj_packed"], params["v_proj_scale"])
            else:
                k = jnp.matmul(hidden_states, params["k_proj"])
                v = jnp.matmul(hidden_states, params["v_proj"])

            k = k.reshape(B, S, self.num_kv_heads, self.head_dim).swapaxes(1, 2)
            v = v.reshape(B, S, self.num_kv_heads, self.head_dim).swapaxes(1, 2)

            k = rms_norm_jax(k, params.get("k_norm"))
            v = rms_norm_jax(v, params.get("v_norm"))

        # Apply RoPE to Query and Key
        partial_factor = 0.25 if not self.is_sliding else 1.0
        q = apply_rope_jax(q, cos, sin, partial_factor=partial_factor)
        if not self.is_shared:
            k = apply_rope_jax(k, cos, sin, partial_factor=partial_factor)

            # Static KV cache. Storing and attending are separate concerns:
            #
            #   decode (S == 1): write the new K/V into the ring, then attend over
            #     the whole buffer, which holds the history.
            #   prefill (S > 1): attend over the freshly computed K/V — all S
            #     positions, which is what the causal mask is sized for — and only
            #     *store* what the buffer can hold. Attending over the padded buffer
            #     instead costs S x (S + max_new_tokens) scores to compute a
            #     S x S result, and it is what forces the cache to be as long as the
            #     full context even for windowed layers.
            if kv_cache is not None:
                k_buf, v_buf, k_scale_buf, v_scale_buf = _unpack_kv(kv_cache)
                quantized = k_scale_buf is not None
                buf_len = k_buf.shape[2]
                slot = jnp.asarray(cache_slot, dtype=jnp.int32)
                if quantized:
                    k_q, k_s = quantize_kv(k, k_buf.dtype)
                    v_q, v_s = quantize_kv(v, v_buf.dtype)
                else:
                    k_q, v_q, k_s, v_s = k, v, None, None

                if S == 1 or chunked:
                    # Same shape of operation for both: write this step's K/V into
                    # the buffer at `slot`, then attend over the WHOLE buffer so the
                    # history is visible. A prefill chunk is just a decode step that
                    # advances the slot by more than one.
                    pos = (jax.lax.rem(slot, jnp.int32(buf_len)) if S == 1
                           else jnp.asarray(slot, jnp.int32))
                    k = jax.lax.dynamic_update_slice(k_buf, k_q.astype(k_buf.dtype), (0, 0, pos, 0))
                    v = jax.lax.dynamic_update_slice(v_buf, v_q.astype(v_buf.dtype), (0, 0, pos, 0))
                    if quantized:
                        key_scale = jax.lax.dynamic_update_slice(
                            k_scale_buf, k_s.astype(k_scale_buf.dtype), (0, 0, pos, 0))
                        value_scale = jax.lax.dynamic_update_slice(
                            v_scale_buf, v_s.astype(v_scale_buf.dtype), (0, 0, pos, 0))
                else:
                    # Prefill attends over the freshly computed bf16 K/V, so the
                    # quantized copy goes to the cache only — no scales are applied
                    # to this pass's attention, and no round-trip error enters it.
                    new_k, new_v = _ring_store(k_buf, v_buf, k_q, v_q)
                    if quantized:
                        kv_out_override = (new_k, new_v,
                                           _ring_store_one(k_scale_buf, k_s),
                                           _ring_store_one(v_scale_buf, v_s))
                    else:
                        kv_out_override = (new_k, new_v)

        # Compute Attention
        attn_out = eager_attention_jax(q, k, v, mask=mask, scaling=self.scaling,
                                       softcap=self.softcap,
                                       key_scale=key_scale, value_scale=value_scale)
        attn_out = attn_out.swapaxes(1, 2).reshape(B, S, -1)

        # Output projection
        if quant_mode == "w4a16":
            out = qat_w4a16_linear_jax(attn_out, params["o_proj_packed"], params["o_proj_scale"])
        else:
            out = jnp.matmul(attn_out, params["o_proj"])

        if self.is_shared:
            return out, None, None
        # Two different things, deliberately: later layers SHARE the K/V this pass
        # computed (length S during prefill), while the cache STORES whatever the
        # ring buffer holds (possibly shorter, for a windowed layer).
        kv_share = _pack_kv(k, v, key_scale, value_scale)
        kv_store = (kv_out_override if kv_out_override is not None
                    else _pack_kv(k, v, key_scale, value_scale))
        return out, kv_share, kv_store


class Gemma4EMLPJAX:
    """Gemma 4 E2B MLP with MatFormer double-wide intermediate support."""

    def __init__(self, config: Gemma4EConfig, is_shared_layer: bool):
        self.is_shared_layer = is_shared_layer
        self.intermediate_size = (
            config.intermediate_size * 2 if (is_shared_layer and config.use_double_wide_mlp)
            else config.intermediate_size
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        params: Dict[str, jax.Array],
        quant_mode: str = "fp16",
    ) -> jax.Array:
        if quant_mode == "w4a16":
            gate = qat_w4a16_linear_jax(hidden_states, params["gate_proj_packed"], params["gate_proj_scale"])
            up = qat_w4a16_linear_jax(hidden_states, params["up_proj_packed"], params["up_proj_scale"])
        else:
            gate = jnp.matmul(hidden_states, params["gate_proj"])
            up = jnp.matmul(hidden_states, params["up_proj"])

        # GeLU Tanh activation matching Gemma spec
        act = jax.nn.gelu(gate, approximate=True) * up

        if quant_mode == "w4a16":
            down = qat_w4a16_linear_jax(act, params["down_proj_packed"], params["down_proj_scale"])
        else:
            down = jnp.matmul(act, params["down_proj"])

        return down


# ==============================================================================
# Full Gemma 4 E2B Model in JAX
# ==============================================================================

def maybe_quantized_linear(x: jax.Array, params: Dict[str, jax.Array], name: str,
                           quant_mode: str = "w4a16") -> jax.Array:
    """Apply a linear that may be stored dense ([in, out]) or W4A16-packed.

    QAT checkpoints ship the PLE projections packed; older/plain exports ship them
    dense. Pick per tensor rather than per model.
    """
    packed = params.get(f"{name}_packed")
    if packed is not None:
        return qat_w4a16_linear_jax(x, packed, params[f"{name}_scale"])
    return jnp.matmul(x, params[name])


class Gemma4EModelJAX:
    """Complete Gemma 4 E2B MatFormer text decoder in pure JAX."""

    def __init__(self, config: Gemma4EConfig):
        self.config = config
        self.share_map = config.kv_share_map()
        self.layers = [
            (
                Gemma4EAttentionJAX(config, i),
                Gemma4EMLPJAX(config, i >= config.first_kv_shared_layer_idx),
            )
            for i in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        input_ids: jax.Array,
        params: Dict[str, jax.Array],
        position_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        quant_mode: str = "fp16",
        kv_caches: Optional[Dict[int, Tuple[jax.Array, jax.Array]]] = None,
        cache_slot: Optional[jax.Array] = None,
        sliding_attention_mask: Optional[jax.Array] = None,
        chunked_prefill: bool = False,
    ):
        B, S = input_ids.shape

        # Primary token embeddings
        h = params["embed_tokens"][input_ids] * math.sqrt(self.config.hidden_size)

        # Per-Layer Embeddings (PLE) for MatFormer context injection
        if self.config.hidden_size_per_layer_input > 0:
            L, D_ple = self.config.num_hidden_layers, self.config.hidden_size_per_layer_input
            # The projection norm is per-layer-slice ([D_ple]) in the shipped
            # checkpoint, not over the flattened [L*D_ple] projection — so reshape
            # to [B, S, L, D_ple] before normalizing.
            ple_embed = (gather_ple(params, input_ids)
                         * math.sqrt(D_ple)).reshape(B, S, L, D_ple)
            ple_proj = maybe_quantized_linear(
                h * (1.0 / math.sqrt(self.config.hidden_size)), params,
                "per_layer_model_projection", quant_mode).reshape(B, S, L, D_ple)
            ple_proj = rms_norm_jax(ple_proj, params.get("per_layer_projection_norm"),
                                    eps=self.config.rms_norm_eps)
            ple_context = (ple_proj + ple_embed) / math.sqrt(2.0)
        else:
            ple_context = None

        # Precompute RoPE cos/sin for sequence length
        inv_freq_sliding = 1.0 / (self.config.rope_theta ** (jnp.arange(0, self.config.head_dim, 2).astype(jnp.float32) / self.config.head_dim))
        inv_freq_global = 1.0 / (self.config.global_rope_theta ** (jnp.arange(0, self.config.global_head_dim, 2).astype(jnp.float32) / self.config.global_head_dim))

        pos_f32 = position_ids.astype(jnp.float32)[:, :, None]
        freqs_sliding = pos_f32 * inv_freq_sliding[None, None, :]
        freqs_global = pos_f32 * inv_freq_global[None, None, :]

        # rotate_half pairs channel i with i + d/2, so the frequency table must be
        # concatenated (cat(freqs, freqs)), not repeat-interleaved. Interleaving
        # pairs every channel against the wrong frequency.
        cos_sliding = jnp.concatenate([jnp.cos(freqs_sliding)] * 2, axis=-1)[:, None, :, :]
        sin_sliding = jnp.concatenate([jnp.sin(freqs_sliding)] * 2, axis=-1)[:, None, :, :]
        cos_global = jnp.concatenate([jnp.cos(freqs_global)] * 2, axis=-1)[:, None, :, :]
        sin_global = jnp.concatenate([jnp.sin(freqs_global)] * 2, axis=-1)[:, None, :, :]

        # Gemma interleaves sliding and full attention; a sliding layer must not
        # see beyond its window. `attention_mask` is the full-attention mask, and
        # `sliding_attention_mask` (if the caller supplied one) is used for the rest.
        masks = {"full_attention": attention_mask,
                 "sliding_attention": sliding_attention_mask
                 if sliding_attention_mask is not None else attention_mask}

        kv_cache_dict = {}    # what later KV-shared layers reuse this pass
        kv_store_dict = {}    # what gets written back to the persistent cache

        # Layer execution loop
        for i, (attn_layer, mlp_layer) in enumerate(self.layers):
            layer_params = params[f"layer_{i}"]
            is_sliding = self.config.layer_types[i] == "sliding_attention"
            cos = cos_sliding if is_sliding else cos_global
            sin = sin_sliding if is_sliding else sin_global

            # Determine KV states (shared vs non-shared)
            source_layer_idx = self.share_map[i]
            kv_shared_states = kv_cache_dict.get(source_layer_idx) if i >= self.config.first_kv_shared_layer_idx else None

            # Attention block
            norm_h = rms_norm_jax(h, layer_params.get("input_layernorm"), eps=self.config.rms_norm_eps)
            attn_out, kv_share, kv_store = attn_layer(
                norm_h,
                layer_params["attn"],
                cos,
                sin,
                mask=masks[self.config.layer_types[i]],
                kv_shared_states=kv_shared_states,
                quant_mode=quant_mode,
                kv_cache=kv_caches.get(i) if kv_caches is not None else None,
                cache_slot=cache_slot,
                chunked=chunked_prefill,
            )
            # Gemma applies post_attention_layernorm to the attention OUTPUT before
            # the residual add, not as a pre-norm for the MLP. Getting this wrong
            # still runs and still produces tokens — they are just the wrong tokens.
            h = h + rms_norm_jax(
                attn_out, layer_params.get("post_attention_layernorm"), eps=self.config.rms_norm_eps)

            if kv_share is not None:
                kv_cache_dict[i] = kv_share
                kv_store_dict[i] = kv_store

            # Feed-forward, sandwiched: pre-norm in, post-norm on the way out.
            ffn_in = rms_norm_jax(
                h, layer_params.get("pre_feedforward_layernorm"), eps=self.config.rms_norm_eps)
            mlp_out = mlp_layer(ffn_in, layer_params["mlp"], quant_mode=quant_mode)
            h = h + rms_norm_jax(
                mlp_out, layer_params.get("post_feedforward_layernorm"), eps=self.config.rms_norm_eps)

            # Per-Layer Embedding (PLE) injection
            if ple_context is not None:
                ple_slice = ple_context[:, :, i, :]  # [B, S, D_ple]
                gate_out = jax.nn.gelu(
                    maybe_quantized_linear(h, layer_params, "per_layer_input_gate", quant_mode),
                    approximate=True)
                ple_fused = gate_out * ple_slice
                ple_proj_back = maybe_quantized_linear(
                    ple_fused, layer_params, "per_layer_projection", quant_mode)
                ple_normed = rms_norm_jax(ple_proj_back, layer_params.get("post_per_layer_input_norm"), eps=self.config.rms_norm_eps)
                h = h + ple_normed

            # The reference decoder layer ends with `hidden_states *= layer_scalar`
            # — the WHOLE residual stream, after every residual add, not the layer
            # delta. It is the counterweight to this checkpoint's large RMSNorm
            # weights (final_norm mean ~14); without it the stream grows layer over
            # layer and the output logits saturate against final_logit_softcapping.
            layer_scalar = layer_params.get("layer_scalar")
            if layer_scalar is not None:
                h = h * layer_scalar.astype(h.dtype)

        # Final RMSNorm
        h = rms_norm_jax(h, params.get("final_norm"), eps=self.config.rms_norm_eps)

        # Output LM Head (tied embeddings scaled by 1/sqrt(hidden_size)).
        # At decode this streams the entire [vocab, hidden] table from HBM for a
        # single token — 1.07 GB at E2B's 262144x2048 in BF16, the largest single
        # read in the step. An int8-quantized copy (see quantize_lm_head) halves it.
        if "embed_tokens_q8" in params:
            logits = jnp.matmul(
                h, params["embed_tokens_q8"].T.astype(h.dtype)
            ) * params["embed_tokens_q8_scale"].astype(h.dtype)
        else:
            logits = jnp.matmul(h, params["embed_tokens"].T)
        if self.config.logit_softcapping > 0.0:
            logits = jnp.tanh(logits / self.config.logit_softcapping) * self.config.logit_softcapping

        if kv_caches is not None:
            new_caches = {i: kv_store_dict[i] for i in range(self.config.first_kv_shared_layer_idx)}
            return logits, new_caches
        return logits


# ==============================================================================
# Performance Utilities: Static KV Cache & Fused jax.lax.scan Generation
# ==============================================================================

def init_kv_cache(
    config: Gemma4EConfig,
    batch_size: int = 1,
    max_seq_len: int = 2048,
    dtype: jnp.dtype = jnp.bfloat16,
    window_kv: bool = False,
) -> Dict[int, Tuple[jax.Array, jax.Array]]:
    """Initialize static preallocated KV cache buffers.

    dtype: bfloat16/float16 store activations directly. One-byte dtypes (int8,
      float8_e4m3fn, float8_e5m2) cannot, so each entry additionally carries
      per-token scale buffers and becomes a 4-tuple (k, v, k_scale, v_scale);
      unquantized entries stay 2-tuples. See `quantize_kv`.

    window_kv: allocate only `sliding_window` slots for sliding-attention layers
      instead of `max_seq_len`. Those layers can never attend outside their window,
      so the extra slots are dead memory. On E2B that is 12 of the 15 KV-holding
      layers; at 8K context it is ~2.7x less KV per sequence. The buffers are ring
      buffers indexed by `position % buf_len`.
    """
    quantized = is_quantized_kv_dtype(dtype)
    cache = {}
    for i in range(config.first_kv_shared_layer_idx):
        is_sliding = config.layer_types[i] == "sliding_attention"
        layer_len = max_seq_len
        if window_kv and is_sliding and config.sliding_window:
            layer_len = min(max_seq_len, int(config.sliding_window))
        h_dim = config.head_dim if is_sliding else config.global_head_dim
        num_kv = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads
        k_shape = (batch_size, num_kv, layer_len, h_dim)
        v_shape = (batch_size, num_kv, layer_len, h_dim)
        entry = (jnp.zeros(k_shape, dtype=dtype), jnp.zeros(v_shape, dtype=dtype))
        if quantized:
            # Ones, not zeros: an unwritten slot is masked out of the softmax, but
            # a zero scale would still feed 0 * inf into the score before masking.
            scale_shape = (batch_size, num_kv, layer_len, 1)
            entry += (jnp.ones(scale_shape, dtype=_KV_SCALE_DTYPE),
                      jnp.ones(scale_shape, dtype=_KV_SCALE_DTYPE))
        cache[i] = entry
    return cache


def generate_n_tokens_scan(
    model: Gemma4EModelJAX,
    prompt_ids: jax.Array,  # [B, S]
    params: Dict[str, jax.Array],
    num_steps: int = 32,
    quant_mode: str = "w4a16",
) -> jax.Array:
    """Hardware throughput probe: N argmax steps on-chip via jax.lax.scan.

    NOTE: this is a benchmark harness, not a correct decoder — the prefill runs
    without a causal mask and the scan steps run the model on a single token
    with no KV cache, so generated tokens do not attend to history. Step cost
    matches real cached decode only where attention is a small fraction of the
    per-step FLOPs (short contexts). For correct generation use
    generate_with_kv_cache / make_cached_decode_step.
    """
    B, prompt_len = prompt_ids.shape
    position_ids = jnp.arange(prompt_len, dtype=jnp.int32)[None, :].repeat(B, axis=0)

    # 1. Prefill pass
    logits = model(prompt_ids, params, position_ids, quant_mode=quant_mode)
    first_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # [B, 1]

    # 2. Fused scan step for autoregressive token generation
    def scan_step(state, _):
        curr_ids, pos = state
        curr_pos_ids = pos[:, None]
        step_logits = model(curr_ids, params, curr_pos_ids, quant_mode=quant_mode)
        tok = jnp.argmax(step_logits[:, -1, :], axis=-1, keepdims=True)
        return (tok, pos + 1), tok

    init_state = (first_token, jnp.full((B,), prompt_len, dtype=jnp.int32))
    (final_tok, _), gen_tokens = jax.lax.scan(
        scan_step, init_state, None, length=num_steps - 1
    )

    # Combine first token + scanned tokens into [B, num_steps]
    scanned_ids = gen_tokens.squeeze(-1).swapaxes(0, 1)
    all_generated = jnp.concatenate([first_token, scanned_ids], axis=1)
    return all_generated


def dequantize_params_to_dense(params: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize every W4A16 weight to dense BF16 once, at load.

    W4A16 is a memory-for-compute trade: 4x smaller storage, paid for with a
    dequantize on every forward pass. That trade is only worth making when storage
    actually binds. E2B is 3.7 GiB dense against 31.24 GiB of HBM — it fits eight
    times over, so the runtime dequant buys nothing.

    Profiled on v6e-1, the dequant is a large share of the decode step
    (`multiply_reduce_fusion` ~25%, plus the group-scale broadcast). Dequantizing
    once at load keeps the QAT weights' *quality* while removing that per-step cost
    entirely, at the price of storing BF16.

    Use this whenever the dense model fits. Keep the packed path for models that
    only fit quantized (31B, 26B MoE), where the trade runs the other way.

    Returns a new tree with `<name>_packed`/`<name>_scale` replaced by a dense
    `<name>` in [in, out] orientation (what the fp16 path expects).
    """
    def convert(node):
        if not isinstance(node, dict):
            return node
        out = {}
        packed_names = {k[: -len("_packed")] for k in node if k.endswith("_packed")}
        for k, v in node.items():
            if isinstance(v, dict):
                out[k] = convert(v)
            elif k.endswith("_packed") or k.endswith("_scale"):
                continue
            else:
                out[k] = v
        for name in packed_names:
            scale = node.get(f"{name}_scale")
            if scale is None:
                raise ValueError(f"{name}_packed has no matching {name}_scale")
            # unpack -> [out, in]; the dense path contracts x[..., in] @ w[in, out]
            out[name] = qat_w4a16_unpack_dequant_jax(node[f"{name}_packed"], scale).T
        return out

    return convert(params)


def quantize_ple_table(params: Dict[str, jax.Array], bits: int = 8,
                       group_size: int = 0) -> Dict[str, jax.Array]:
    """Replace the per-layer-embedding table with an int8 or int4 copy.

    `embed_tokens_per_layer` is [vocab, layers*D_ple] — 4.70 GB in BF16 on E2B,
    the single largest tensor in the model and **72% of resident weights**. The
    shipped QAT checkpoint leaves it unquantized: W4A16 compressed the 1.06 GB of
    transformer weights and none of the 5.50 GB of lookup tables.

    It is used exactly ONCE, as a gather (`table[input_ids]`), and never in a
    matmul, so quantization error cannot compound the way it does for
    `embed_tokens` (which is also the tied LM head). That makes it the
    lowest-risk quantization target in the model, and the largest.

    This is a MEMORY optimization, not a bandwidth one: a gather reads only the
    rows the prompt touches, so decode never streams the table. The win is HBM
    headroom -> more resident KV tokens -> more concurrent sequences.

    bits=8:  per-row scale,   4.70 GB -> 2.35 GB
    bits=4:  grouped scale,   4.70 GB -> 1.17 GB (two nibbles per byte)

    group_size: 0 means one scale per row. For int4 that is far too coarse — 16
      levels across all 8960 elements — so pass the per-layer slice width
      (`hidden_size_per_layer_input`, 256 on E2B), which is also the semantically
      natural grouping: each layer's slice gets its own scale. Overhead is 35
      scales x 2 B against 4480 packed bytes, i.e. 1.6%.

    Returns a new dict; the original is not mutated. The BF16 table is dropped,
    since nothing else uses it.
    """
    if bits not in (4, 8):
        raise ValueError(f"PLE quantization supports 4 or 8 bits, got {bits}")
    tbl = params["embed_tokens_per_layer"]
    V, LD = tbl.shape
    g = int(group_size) or LD
    if LD % g:
        raise ValueError(f"group_size {g} does not divide row width {LD}")
    qmax = 127.0 if bits == 8 else 7.0

    # Quantize on the HOST, in row chunks. This is a load-time operation and has
    # no business competing for accelerator memory: upcasting E2B's 4.70 GB table
    # to float32 in one shot needs 8.75 GB, which OOMs a v6e-1 that already holds
    # the rest of the parameters. Chunking bounds the working set to a few hundred
    # MB, and doing it off-device leaves HBM entirely alone.
    cpu = jax.devices("cpu")[0]
    # Where the table came from is where the quantized copy must end up. Leaving
    # it on the host is catastrophic and silent in the wrong direction: capacity
    # measurements look BETTER (the table no longer occupies HBM at all) while
    # every gather crosses the host interconnect. Measured at 18.5 s per decode
    # step against 60 ms resident — the only symptom, since nothing errors.
    src_devices = getattr(tbl, "devices", lambda: set())()
    home = next(iter(src_devices), None)
    rows_per_chunk = max(1, (1 << 26) // (LD * 4))       # ~256 MB of float32
    q_chunks, s_chunks = [], []
    for start in range(0, V, rows_per_chunk):
        blk = jax.device_put(tbl[start:start + rows_per_chunk], cpu)
        blk = blk.astype(jnp.float32).reshape(-1, LD // g, g)
        amax = jnp.max(jnp.abs(blk), axis=-1, keepdims=True)
        scale = jnp.maximum(amax, 1e-8) / qmax
        q = jnp.clip(jnp.round(blk / scale), -qmax, qmax).reshape(-1, LD)
        if bits == 8:
            q_chunks.append(q.astype(jnp.int8))
        else:
            # Two signed nibbles per byte. Bias by +8 into [0, 15] so the shift
            # and mask are unsigned; `gather_ple` subtracts it back.
            u = (q + 8.0).astype(jnp.uint8)
            q_chunks.append((u[:, 0::2] | (u[:, 1::2] << 4)).astype(jnp.uint8))
        s_chunks.append(scale.astype(jnp.bfloat16))

    out = dict(params)
    out.pop("embed_tokens_per_layer")
    key = "embed_tokens_per_layer_q8" if bits == 8 else "embed_tokens_per_layer_q4"
    q_all = jnp.concatenate(q_chunks, axis=0)
    s_all = jnp.concatenate(s_chunks, axis=0)
    if home is not None:
        q_all = jax.device_put(q_all, home)
        s_all = jax.device_put(s_all, home)
    out[key] = q_all
    out["embed_tokens_per_layer_scale"] = s_all
    return out


def gather_ple(params: Dict[str, jax.Array], input_ids: jax.Array) -> jax.Array:
    """Gather per-layer embeddings, from either the BF16 or int8 table."""
    q8 = params.get("embed_tokens_per_layer_q8")
    q4 = params.get("embed_tokens_per_layer_q4")
    if q8 is None and q4 is None:
        return params["embed_tokens_per_layer"][input_ids]

    if q8 is not None:
        rows = q8[input_ids].astype(jnp.bfloat16)                    # [B, S, LD]
    else:
        # Gather the packed bytes, then split nibbles — half the HBM traffic of
        # gathering an unpacked table, and the unpack touches only the rows the
        # prompt actually references.
        packed = q4[input_ids]                                        # [B, S, LD/2]
        lo = (packed & 0x0F).astype(jnp.int32) - 8
        hi = (packed >> 4).astype(jnp.int32) - 8
        rows = jnp.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], -1)
        rows = rows.astype(jnp.bfloat16)                              # [B, S, LD]

    scale = params["embed_tokens_per_layer_scale"][input_ids].astype(jnp.bfloat16)
    # scale is [B, S, LD/g, 1]. Derive the group count from its SHAPE rather than
    # from a stored integer: a Python int inside the params pytree becomes a
    # traced array under jit, and `int()` on a tracer raises.
    n_groups = scale.shape[-2]
    grouped = rows.reshape(*rows.shape[:-1], n_groups, rows.shape[-1] // n_groups)
    # Regroup rather than repeat, so the scale is never widened to the full row.
    return (grouped * scale).reshape(*rows.shape)


def quantize_lm_head(params: Dict[str, jax.Array], keep_bf16: bool = False) -> Dict[str, jax.Array]:
    """Add a per-row int8 copy of the tied embedding for the output projection.

    The LM head is the largest single HBM read in a decode step (the full
    [vocab, hidden] table for one token). Symmetric per-row (per-vocab-entry)
    int8 halves that traffic. The BF16 table is still needed for the *input*
    embedding lookup, so by default it is kept and this trades memory for
    bandwidth; pass keep_bf16=False semantics are handled by the caller.

    Returns a new params dict; the original is not mutated.
    """
    emb = params["embed_tokens"]
    amax = jnp.max(jnp.abs(emb.astype(jnp.float32)), axis=-1, keepdims=True)
    scale = jnp.maximum(amax, 1e-8) / 127.0
    q8 = jnp.clip(jnp.round(emb.astype(jnp.float32) / scale), -127, 127).astype(jnp.int8)
    out = dict(params)
    out["embed_tokens_q8"] = q8
    out["embed_tokens_q8_scale"] = scale.astype(jnp.bfloat16).reshape(1, 1, -1)
    if not keep_bf16:
        pass  # input embedding lookup still needs the BF16 table
    return out


# ==============================================================================
# Correct KV-Cached Autoregressive Generation
# ==============================================================================

def prefill_with_kv_cache(
    model: Gemma4EModelJAX,
    prompt_ids: jax.Array,       # [B, S] (right-padded to a static bucket)
    prompt_valid: jax.Array,     # [B, S] bool — True for real tokens
    params: Dict[str, jax.Array],
    max_new_tokens: int,
    quant_mode: str = "w4a16",
    cache_dtype: jnp.dtype = jnp.bfloat16,
    window_kv: Optional[bool] = None,
):
    """Run the prefill pass, writing K/V into buffers sized S + max_new_tokens.

    window_kv=None (default) decides per call: windowing sliding layers costs ~3%
    on the decode step (measured, ring indexing + per-type mask) and only saves
    memory once the context exceeds the window. So enable it exactly when it pays.

    Returns (last_logits [B, V], kv_caches, valid_mask [B, S+max_new_tokens]).
    """
    B, S = prompt_ids.shape
    total_len = S + max_new_tokens
    if window_kv is None:
        win = model.config.sliding_window
        window_kv = bool(win) and total_len > int(win)
    caches = init_kv_cache(model.config, batch_size=B, max_seq_len=total_len,
                           dtype=cache_dtype, window_kv=window_kv)

    position_ids = jnp.arange(S, dtype=jnp.int32)[None, :].repeat(B, axis=0)
    # Prefill attends over the freshly computed K/V (S keys), not the padded cache,
    # so the mask is S x S — no padding out to S + max_new_tokens.
    window = model.config.sliding_window
    mask = make_prefill_causal_mask(prompt_valid)
    sliding_mask = (make_prefill_causal_mask(prompt_valid, window=window)
                    if window is not None else None)

    logits, caches = model(
        prompt_ids, params, position_ids,
        attention_mask=mask, quant_mode=quant_mode,
        kv_caches=caches, cache_slot=jnp.int32(0),
        sliding_attention_mask=sliding_mask,
    )

    # Logits at the last REAL token of each row
    prompt_lens = prompt_valid.sum(axis=1).astype(jnp.int32)          # [B]
    last_logits = jnp.take_along_axis(
        logits, (prompt_lens - 1)[:, None, None], axis=1
    )[:, 0, :]                                                        # [B, V]

    valid = jnp.concatenate(
        [prompt_valid, jnp.zeros((B, max_new_tokens), dtype=jnp.bool_)], axis=1
    )
    return last_logits, caches, valid


def make_chunked_prefill_step(model: Gemma4EModelJAX, chunk_size: int,
                              quant_mode: str = "w4a16"):
    """Build a jittable prefill step over ONE chunk of `chunk_size` tokens.

    step(params, caches, valid, ids [B, chunk], slot) -> (caches, valid, logits)

    Why chunk at all: on a v6e-1 the decode ceiling is a fixed budget of resident
    KV tokens (measured: 524,288 at bf16 and 1,048,576 at int8, both flat to 0.0%
    across contexts from 512 to 32768). The BATCH ceiling is a separate and much
    tighter wall set by prefill, whose peak temporaries are linear in the total
    prompt tokens of the pass at a flat ~2.13 MB/token — roughly 11,900 tokens
    once the weights are resident. Chunking bounds B * chunk_size against that
    budget, so a prompt too long to admit in one pass becomes a scheduling
    problem instead of an OOM.

    The cache must be full-length (window_kv=False): a chunk writes `chunk_size`
    contiguous slots at an arbitrary offset, which a shorter ring buffer would
    wrap. Sliding layers still get their window through the mask.
    """
    window = model.config.sliding_window

    def step(params, caches, valid, ids, slot):
        slot_i = jnp.asarray(slot, jnp.int32)
        # Mark this chunk's slots valid BEFORE building the mask: a chunk attends
        # to itself as well as to history.
        idx = slot_i + jnp.arange(chunk_size)
        valid = valid.at[:, idx].set(True)
        mask = make_chunk_mask(valid, chunk_size, slot_i)
        sliding_mask = (make_chunk_mask(valid, chunk_size, slot_i, window=window)
                        if window is not None else None)
        position_ids = (slot_i + jnp.arange(chunk_size, dtype=jnp.int32))[None, :]
        position_ids = jnp.broadcast_to(position_ids, ids.shape)
        logits, caches = model(
            ids, params, position_ids,
            attention_mask=mask, quant_mode=quant_mode,
            kv_caches=caches, cache_slot=slot_i,
            sliding_attention_mask=sliding_mask,
            chunked_prefill=True,
        )
        return caches, valid, logits
    return step


def chunked_prefill_with_kv_cache(
    model: Gemma4EModelJAX,
    prompt_ids: jax.Array,       # [B, S], S a multiple of chunk_size
    prompt_valid: jax.Array,     # [B, S] bool
    params: Dict[str, jax.Array],
    max_new_tokens: int,
    chunk_size: int = 256,
    quant_mode: str = "w4a16",
    cache_dtype: jnp.dtype = jnp.bfloat16,
):
    """Prefill in fixed-size chunks. Same result as `prefill_with_kv_cache`,
    with peak temporaries bounded by `chunk_size` instead of the prompt length.

    Returns (last_logits [B, V], kv_caches, valid_mask [B, S + max_new_tokens]).
    """
    B, S = prompt_ids.shape
    if S % chunk_size:
        raise ValueError(f"prompt length {S} must be a multiple of chunk_size {chunk_size}")
    total_len = S + max_new_tokens
    caches = init_kv_cache(model.config, batch_size=B, max_seq_len=total_len,
                           dtype=cache_dtype, window_kv=False)
    valid = jnp.zeros((B, total_len), dtype=jnp.bool_)

    step = jax.jit(make_chunked_prefill_step(model, chunk_size, quant_mode=quant_mode))
    logits = None
    for start in range(0, S, chunk_size):
        caches, valid, logits = step(
            params, caches, valid, prompt_ids[:, start:start + chunk_size], jnp.int32(start))

    # Padding slots were marked valid chunk-by-chunk; restore the real mask.
    valid = jnp.concatenate(
        [prompt_valid, jnp.zeros((B, max_new_tokens), dtype=jnp.bool_)], axis=1)

    # Logits at the last REAL token. Only the final chunk is still in hand, so
    # index within it — every row's last real token lives there when prompts are
    # right-padded to the same bucket.
    prompt_lens = prompt_valid.sum(axis=1).astype(jnp.int32)
    within = prompt_lens - 1 - (S - chunk_size)
    within = jnp.clip(within, 0, chunk_size - 1)
    last_logits = jnp.take_along_axis(logits, within[:, None, None], axis=1)[:, 0, :]
    return last_logits, caches, valid


def make_cached_decode_step(model: Gemma4EModelJAX, quant_mode: str = "w4a16",
                            window_kv: bool = False):
    """Build a jittable single-token decode step over a static KV cache.

    step(params, caches, valid, tok [B,1], logical_pos [B], slot scalar)
      -> (caches, valid, last_logits [B, V])
    """
    window = model.config.sliding_window

    def step(params, caches, valid, tok, logical_pos, slot):
        valid = valid.at[:, slot].set(True)
        mask = make_decode_mask(valid)
        if window is None:
            sliding_mask = None
        elif window_kv:
            # Sliding layers attend over their ring buffer, not the full cache.
            # A short request allocates fewer slots than the configured window:
            # init_kv_cache uses min(total_len, window). Match that physical
            # length here instead of constructing (for E4B) an unconditional
            # 512-key mask for, e.g., a 72-key cache.
            ring_len = min(int(valid.shape[1]), int(window))
            sliding_mask = make_ring_decode_mask(
                valid, ring_len, jnp.asarray(slot, jnp.int32))
        else:
            sliding_mask = make_decode_mask(valid, window=window, slot=slot)
        logits, caches = model(
            tok, params, logical_pos[:, None],
            attention_mask=mask, quant_mode=quant_mode,
            kv_caches=caches, cache_slot=slot,
            sliding_attention_mask=sliding_mask,
        )
        return caches, valid, logits[:, -1, :]
    return step


def generate_with_kv_cache(
    model: Gemma4EModelJAX,
    prompt_ids: jax.Array,       # [B, S] right-padded
    prompt_valid: jax.Array,     # [B, S] bool
    params: Dict[str, jax.Array],
    max_new_tokens: int,
    quant_mode: str = "w4a16",
    temperature: float = 0.0,
    top_k: int = 40,
    prng_key: Optional[jax.Array] = None,
    cache_dtype: jnp.dtype = jnp.bfloat16,
    window_kv: bool = False,
) -> jax.Array:
    """Correct autoregressive generation with a static KV cache.

    Every generated token attends to the full (unpadded) history. Greedy when
    temperature <= 0, on-chip top-k sampling otherwise. Returns [B, max_new_tokens].
    """
    B, S = prompt_ids.shape
    if prng_key is None:
        prng_key = jax.random.PRNGKey(0)

    last_logits, caches, valid = prefill_with_kv_cache(
        model, prompt_ids, prompt_valid, params, max_new_tokens,
        quant_mode=quant_mode, cache_dtype=cache_dtype, window_kv=window_kv,
    )
    step = jax.jit(make_cached_decode_step(model, quant_mode=quant_mode, window_kv=window_kv))

    prompt_lens = prompt_valid.sum(axis=1).astype(jnp.int32)
    tokens = []
    tok = onchip_sample_tpu_v6e_jax(last_logits, prng_key, temperature=temperature, top_k=top_k)
    tokens.append(tok)

    for t in range(max_new_tokens - 1):
        prng_key, sample_key = jax.random.split(prng_key)
        caches, valid, last_logits = step(
            params, caches, valid, tok, prompt_lens + t, jnp.int32(S + t)
        )
        tok = onchip_sample_tpu_v6e_jax(last_logits, sample_key, temperature=temperature, top_k=top_k)
        tokens.append(tok)

    return jnp.concatenate(tokens, axis=1)


# ==============================================================================
# PagedAttention Manager in JAX (Zero Fragmentation)
# ==============================================================================

@dataclasses.dataclass
class PagedKVCache:
    """Paged Key-Value cache manager in JAX (vLLM-style zero fragmentation)."""
    k_pages: jax.Array        # [num_blocks, num_kv_heads, block_size, head_dim]
    v_pages: jax.Array        # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables: jax.Array   # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array   # [batch_size]
    block_size: int = 16


def init_paged_kv_cache(
    config: Gemma4EConfig,
    num_blocks: int = 512,
    block_size: int = 16,
    batch_size: int = 1,
    max_blocks_per_seq: int = 128,
    dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> Dict[int, PagedKVCache]:
    """Initialize paged block KV cache pools for non-shared attention layers."""
    paged_caches = {}
    for i in range(config.first_kv_shared_layer_idx):
        is_sliding = config.layer_types[i] == "sliding_attention"
        h_dim = config.head_dim if is_sliding else config.global_head_dim
        num_kv = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads
        k_pages = jnp.zeros((num_blocks, num_kv, block_size, h_dim), dtype=dtype)
        v_pages = jnp.zeros((num_blocks, num_kv, block_size, h_dim), dtype=dtype)
        block_tables = jnp.zeros((batch_size, max_blocks_per_seq), dtype=jnp.int32)
        context_lens = jnp.zeros((batch_size,), dtype=jnp.int32)
        paged_caches[i] = PagedKVCache(
            k_pages=k_pages,
            v_pages=v_pages,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=block_size,
        )
    return paged_caches


# ==============================================================================
# TPU v6e Single-Chip Hardware Profile & Vectorized On-Chip Sampling
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class TPUv6eHardwareProfile:
    """Hardware specifications and MXU alignment rules for Cloud TPU v6e (Trillium)."""
    hbm_capacity_bytes: int = 33_546_042_880   # 32 GB HBM3
    hbm_bandwidth_gbps: int = 1638             # 1,638 GB/s HBM3 bandwidth
    vmem_capacity_bytes: int = 16 * 1024 * 1024  # 16 MB VMEM per core
    mxu_tile_dim: int = 128                    # 128x128 systolic matrix array
    optimal_k_tile: int = 256
    optimal_n_tile: int = 256
    static_sequence_buckets: Tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096, 8192)

    @classmethod
    def get_nearest_bucket(cls, seq_len: int) -> int:
        """Find the nearest 128-aligned static bucket size to prevent XLA retrace."""
        for b in cls.static_sequence_buckets:
            if b >= seq_len:
                return b
        return (seq_len + 127) // 128 * 128


def pad_to_tpu_v6e_bucket(input_ids: jax.Array, pad_token_id: int = 0) -> Tuple[jax.Array, jax.Array]:
    """Pads input sequence IDs to nearest 128-aligned TPU v6e static sequence bucket."""
    B, S = input_ids.shape
    bucket_s = TPUv6eHardwareProfile.get_nearest_bucket(S)
    if bucket_s == S:
        return input_ids, jnp.ones((B, S), dtype=jnp.bool_)

    pad_len = bucket_s - S
    padded_ids = jnp.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
    mask = jnp.concatenate([jnp.ones((B, S), dtype=jnp.bool_), jnp.zeros((B, pad_len), dtype=jnp.bool_)], axis=1)
    return padded_ids, mask


def onchip_sample_tpu_v6e_jax(
    logits: jax.Array,           # [B, V] where V = 262,144 (2,048 x 128 tile-aligned)
    prng_key: jax.Array,
    temperature: float = 0.7,
    top_k: int = 40,
) -> jax.Array:
    """Vectorized on-chip Top-K sampling executed 100% on TPU core (zero host latency)."""
    B, V = logits.shape

    if temperature <= 0.0:
        return jnp.argmax(logits, axis=-1, keepdims=True)

    scaled_logits = logits / max(temperature, 1e-5)

    if top_k > 0 and top_k < V:
        top_k_val, top_k_idx = jax.lax.top_k(scaled_logits, top_k)
        mask_val = jnp.full_like(scaled_logits, -1e9)
        scaled_logits = mask_val.at[jnp.arange(B)[:, None], top_k_idx].set(top_k_val)

    sampled_idx = jax.random.categorical(prng_key, scaled_logits, axis=-1)
    return sampled_idx[:, None]
