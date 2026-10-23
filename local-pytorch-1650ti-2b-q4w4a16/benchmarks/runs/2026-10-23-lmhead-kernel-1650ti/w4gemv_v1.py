"""Int4 W4A16 matrix-vector product that reads the packed weights directly (no fp16 weight copy).

Layout is compressed-tensors pack-quantized, symmetric, group 32: packed[o, j] holds input columns
8j..8j+7, low nibble first, stored with offset 8; scale[o, g] covers columns 32g..32g+31.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_O": bo, "BLOCK_KP": bk}, num_warps=w)
        for bo in (16, 32, 64)
        for bk in (32, 64, 128)
        for w in (4, 8)
    ],
    key=["O", "KP"],
)
@triton.jit
def _w4_gemv(x_ptr, p_ptr, s_ptr, y_ptr, O, KP, G, BLOCK_O: tl.constexpr, BLOCK_KP: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_O + tl.arange(0, BLOCK_O)
    rmask = rows < O
    shifts = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
    for k0 in range(0, KP, BLOCK_KP):
        kp = k0 + tl.arange(0, BLOCK_KP)
        kmask = kp < KP
        m2 = rmask[:, None] & kmask[None, :]
        p = tl.load(p_ptr + rows[:, None] * KP + kp[None, :], mask=m2, other=0)
        q = ((p[:, :, None] >> shifts[None, None, :]) & 15).to(tl.float32) - 8.0
        x = tl.load(x_ptr + kp[:, None] * 8 + tl.arange(0, 8)[None, :], mask=kmask[:, None], other=0.0).to(tl.float32)
        part = tl.sum(q * x[None, :, :], axis=2)
        s = tl.load(s_ptr + rows[:, None] * G + (kp // 4)[None, :], mask=m2, other=0.0).to(tl.float32)
        acc += tl.sum(part * s, axis=1)
    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=rmask)


def w4_gemv_v1(x, packed, scale):
    """x: (..., I) with exactly one row; packed: (O, I/8) int32; scale: (O, I/32). Returns (..., O)."""
    O, KP = packed.shape
    xv = x.reshape(-1).contiguous()
    y = torch.empty(O, device=x.device, dtype=x.dtype)
    _w4_gemv[lambda m: (triton.cdiv(O, m["BLOCK_O"]),)](xv, packed, scale, y, O, KP, scale.shape[1])
    return y.view(*x.shape[:-1], O)
