"""Single-query decode attention for Gemma 4 on a static KV cache, as a transformers attention function.

One Triton program per query head streams the fp16 K/V cache once, applies the boolean mask transformers builds
(True = attend), and keeps an online softmax in fp32. Prompt reads (more than one query token) go to SDPA.
Register with `register("w4decode")` and set `config._attn_implementation = "w4decode"`.
"""

import torch
import triton
import triton.language as tl
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn}, num_warps=w) for bn in (16, 32, 64) for w in (2, 4, 8)],
    key=["L", "D", "GROUP"],
)
@triton.jit
def _decode_attn(
    q_ptr,
    k_ptr,
    v_ptr,
    m_ptr,
    o_ptr,
    L,
    scale,
    sq_h,
    sk_h,
    sk_n,
    sv_h,
    sv_n,
    so_h,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    h = tl.program_id(0)
    kvh = h // GROUP
    scale = scale.to(tl.float32)  # torch.compile can pass a Python float as fp64
    d = tl.arange(0, D)
    q = tl.load(q_ptr + h * sq_h + d).to(tl.float32)
    m_i = tl.full((), -1e30, tl.float32)
    l_i = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    for n0 in range(0, L, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < L
        keep = (tl.load(m_ptr + n, mask=nm, other=0) != 0) & nm
        k = tl.load(k_ptr + kvh * sk_h + n[:, None] * sk_n + d[None, :], mask=keep[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * scale
        s = tl.where(keep, s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(keep, tl.exp(s - m_new), 0.0)
        v = tl.load(v_ptr + kvh * sv_h + n[:, None] * sv_n + d[None, :], mask=keep[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    tl.store(o_ptr + h * so_h + d, (acc / l_i).to(o_ptr.dtype.element_ty))


def w4_decode_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """query [1, H, 1, D], key/value [1, KVH, L, D], attention_mask bool [1, 1, 1, L]. Returns ([1, 1, H, D], None)."""
    b, H, q_len, D = query.shape
    if b != 1 or q_len != 1 or attention_mask is None or attention_mask.dtype != torch.bool:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )
    KVH, L = key.shape[1], key.shape[2]
    assert key.stride(-1) == 1 and value.stride(-1) == 1 and query.stride(-1) == 1
    out = torch.empty(1, 1, H, D, device=query.device, dtype=query.dtype)
    mask = attention_mask.reshape(-1)[-L:].to(torch.uint8)
    scale = 1.0 if scaling is None else float(scaling)
    _decode_attn[(H,)](
        query,
        key,
        value,
        mask,
        out,
        L,
        scale,
        query.stride(1),
        key.stride(1),
        key.stride(2),
        value.stride(1),
        value.stride(2),
        out.stride(2),
        D=D,
        GROUP=H // KVH,
    )
    return out, None


def register(name="w4decode"):
    AttentionInterface.register(name, w4_decode_attention)
    AttentionMaskInterface.register(name, sdpa_mask)
