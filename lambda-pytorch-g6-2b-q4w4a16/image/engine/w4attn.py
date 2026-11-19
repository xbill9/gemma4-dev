"""Long-context attention for Gemma 4 E2B on one KV head: split decode kernel plus SDPA for chunked prompt reads.

Decode (one query token): `decode_attention` splits the keys into S slices; each Triton program reads its slice of
K and V once for all query heads (E2B has one KV head), keeps an online softmax, and a second kernel combines the
slices. Full-attention layers read slots 0..pos (slot index = position); sliding layers read a ring buffer and mask
by the position stored with each slot. Everything it needs (pos) is on the device, so it can sit in a CUDA graph.

Prompt chunks: `prefill_attention` uses PyTorch's memory-efficient SDPA with K/V expanded across heads (a view,
no copy). Full layers use a bottom-right causal mask; sliding layers an explicit [n, keys] window mask.
"""

import torch
import triton
import triton.language as tl
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

# Defaults from dec_sweep3.jsonl (GTX 1650 Ti, D=512): one head at a time over 2-D [keys, D] tiles (mode 3), 64 slices
# of 8-key blocks, 2 warps: 4.11 ms at 131,072 tokens held (65 GB/s). fp16 tl.dot ran at 24 GB/s on this card, which
# has no tensor cores (dec_sweep.jsonl).
SPLITS = 64
BN, WARPS, MODE, HP = 8, 2, 3, 8


@triton.jit
def _dec_split(
    q_ptr,
    k_ptr,
    v_ptr,
    kpos_ptr,
    pos_ptr,
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    L,
    window,
    scale,
    H: tl.constexpr,
    HP: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    S: tl.constexpr,
    SLIDING: tl.constexpr,
    MODE: tl.constexpr,
):
    sid = tl.program_id(0)
    scale = scale.to(tl.float32)
    pos = tl.load(pos_ptr).to(tl.int32)
    if SLIDING:
        hi = L  # every ring slot; the stored positions decide what counts
    else:
        hi = pos + 1  # slot index == position
    per = tl.cdiv(tl.cdiv(hi, S), BN) * BN
    start = sid * per
    end = tl.minimum(start + per, hi)
    hh = tl.arange(0, HP)
    hm = hh < H
    d = tl.arange(0, D)
    q = tl.load(q_ptr + hh[:, None] * D + d[None, :], mask=hm[:, None], other=0.0)
    m_i = tl.full((HP,), -1e30, tl.float32)
    l_i = tl.zeros((HP,), tl.float32)
    acc = tl.zeros((HP, D), tl.float32)
    for n0 in range(start, end, BN):
        n = n0 + tl.arange(0, BN)
        nm = n < end
        if SLIDING:
            kp = tl.load(kpos_ptr + n, mask=nm, other=-1)
            keep = nm & (kp >= 0) & (kp <= pos) & (kp > pos - window)
        else:
            keep = nm
        k = tl.load(k_ptr + n[:, None] * D + d[None, :], mask=keep[:, None], other=0.0)
        if MODE == 0:
            s = tl.dot(q, tl.trans(k)) * scale
        elif MODE == 1:
            s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="ieee") * scale
        elif MODE == 2:
            s = tl.sum(q.to(tl.float32)[:, None, :] * k.to(tl.float32)[None, :, :], axis=2) * scale
        else:
            # One head at a time over a 2-D [keys, D] tile, so the reduction runs along D inside a warp; the eight
            # rows of s are assembled with selects. q rows are re-read from L1 each block.
            kf = k.to(tl.float32)
            s = tl.zeros((HP, BN), tl.float32)
            for h in tl.static_range(H):
                qh = tl.load(q_ptr + h * D + d).to(tl.float32)
                sh = tl.sum(kf * qh[None, :], axis=1)
                s = tl.where(hh[:, None] == h, sh[None, :], s)
            s = s * scale
        s = tl.where(keep[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(keep[None, :], tl.exp(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + n[:, None] * D + d[None, :], mask=keep[:, None], other=0.0)
        if MODE == 0:
            pv = tl.dot(p.to(tl.float16), v)
        elif MODE == 1:
            pv = tl.dot(p, v.to(tl.float32), input_precision="ieee")
        elif MODE == 2:
            pv = tl.sum(p[:, :, None] * v.to(tl.float32)[None, :, :], axis=1)
        else:
            vf = v.to(tl.float32)
            pv = tl.zeros((HP, D), tl.float32)
            for h in tl.static_range(H):
                ph = tl.sum(tl.where(hh[:, None] == h, p, 0.0), axis=0)
                pvh = tl.sum(ph[:, None] * vf, axis=0)
                pv = tl.where(hh[:, None] == h, pvh[None, :], pv)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
    tl.store(pm_ptr + sid * H + hh, m_i, mask=hm)
    tl.store(pl_ptr + sid * H + hh, l_i, mask=hm)
    tl.store(pacc_ptr + (sid * H + hh[:, None]) * D + d[None, :], acc, mask=hm[:, None])


@triton.jit
def _dec_combine(pm_ptr, pl_ptr, pacc_ptr, o_ptr, H: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
    h = tl.program_id(0)
    s = tl.arange(0, S)
    d = tl.arange(0, D)
    m = tl.load(pm_ptr + s * H + h)
    l = tl.load(pl_ptr + s * H + h)
    w = tl.exp(m - tl.max(m, axis=0))
    acc = tl.load(pacc_ptr + (s[:, None] * H + h) * D + d[None, :])
    out = tl.sum(w[:, None] * acc, axis=0) / tl.sum(w * l, axis=0)
    tl.store(o_ptr + h * D + d, out.to(o_ptr.dtype.element_ty))


_scratch = {}


def decode_attention(
    q, k_buf, v_buf, kpos, pos, sliding, window, scale, splits=None, bn=None, warps=None, mode=None, hp=None
):
    """q [1, 1, H, D]; k_buf/v_buf [L, D]; kpos [L] int32 (ring slots) or None; pos [1] int64 on device."""
    H, D = q.shape[2], q.shape[3]
    L = k_buf.shape[0]
    S = splits or SPLITS
    key = (H, D, S)
    if key not in _scratch:
        _scratch[key] = (
            torch.empty(S, H, device=q.device, dtype=torch.float32),
            torch.empty(S, H, device=q.device, dtype=torch.float32),
            torch.empty(S, H, D, device=q.device, dtype=torch.float32),
        )
    pm, pl, pacc = _scratch[key]
    out = torch.empty(1, 1, H, D, device=q.device, dtype=q.dtype)
    qc = q.reshape(H, D).contiguous()
    _dec_split[(S,)](
        qc,
        k_buf,
        v_buf,
        kpos if kpos is not None else pos,
        pos,
        pm,
        pl,
        pacc,
        L,
        window if window else 0,
        float(scale),
        H=H,
        HP=hp or HP,
        D=D,
        BN=bn or BN,
        S=S,
        SLIDING=bool(sliding),
        MODE=MODE if mode is None else mode,
        num_warps=warps or WARPS,
    )
    _dec_combine[(H,)](pm, pl, pacc, out, H=H, D=D, S=S, num_warps=4)
    return out


def prefill_attention(q, keys, values, kpos, qpos0, sliding, window, scale):
    """q [1, n, H, D]; keys/values [Lk, D] in temporal order, the last n being this chunk; kpos [Lk] int (positions).

    Returns [1, n, H, D].
    """
    n, H, D = q.shape[1], q.shape[2], q.shape[3]
    Lk = keys.shape[0]
    qt = q.transpose(1, 2)
    k = keys.view(1, 1, Lk, D).expand(1, H, Lk, D)
    v = values.view(1, 1, Lk, D).expand(1, H, Lk, D)
    if sliding:
        # Real copies, not head-broadcast views: memory-efficient SDPA with an explicit mask and stride-0 K/V gets the
        # last query row wrong when n % 32 == 1 (torch 2.13, sm_75; benchmarks/runs/2026-10-23-accuracy-1650ti/sdpa_repro.txt). At most 768 keys here.
        k, v = k.contiguous(), v.contiguous()
        qpos = qpos0 + torch.arange(n, device=q.device)
        kp = kpos.to(q.device)
        mask = (kp[None, :] <= qpos[:, None]) & (kp[None, :] > qpos[:, None] - window) & (kp[None, :] >= 0)
        mask = mask.view(1, 1, n, Lk)
    else:
        mask = causal_lower_right(n, Lk)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(qt, k, v, attn_mask=mask, scale=scale)
    return o.transpose(1, 2)
