"""Int4 W4A16 matrix-vector product, v3: v2 with the int-to-float conversion done by the 2^23 bit trick.

Same layout as v1 (compressed-tensors pack-quantized, symmetric, group 32): packed[o, j] holds input columns
8j..8j+7, low nibble first, offset 8; scale[o, g] covers columns 32g..32g+31.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_O": bo, "BLOCK_G": bg}, num_warps=w, num_stages=s)
        for bo in (8, 16, 32, 64)
        for bg in (4, 8, 16, 32)
        for w in (2, 4, 8)
        for s in (1, 2, 3)
    ],
    key=["O", "KP"],
)
@triton.jit
def _w4_gemv3(x_ptr, p_ptr, s_ptr, y_ptr, O, KP, G, BLOCK_O: tl.constexpr, BLOCK_G: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_O + tl.arange(0, BLOCK_O)
    rmask = rows < O
    acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
    for g0 in range(0, G, BLOCK_G):
        g = g0 + tl.arange(0, BLOCK_G)
        gmask = g < G
        kp = g0 * 4 + tl.arange(0, BLOCK_G * 4)  # packed int32 columns covered by these groups
        m2 = rmask[:, None] & (kp < KP)[None, :]
        p = tl.load(p_ptr + rows[:, None] * KP + kp[None, :], mask=m2, other=0)
        part = tl.zeros((BLOCK_O, BLOCK_G * 4), dtype=tl.float32)
        for j in tl.static_range(8):
            # (nibble | bits of 2^23) read as fp32 is exactly 2^23 + nibble; subtracting 2^23 + 8 leaves nibble - 8.
            # Turing converts int to float at a quarter of its FMA rate; this costs one OR and one FADD.
            qj = (((p >> (4 * j)) & 15) | 0x4B000000).to(tl.float32, bitcast=True) - 8388616.0
            xj = tl.load(x_ptr + kp * 8 + j, mask=kp < KP, other=0.0).to(tl.float32)
            part += qj * xj[None, :]
        grp = tl.sum(tl.reshape(part, (BLOCK_O, BLOCK_G, 4)), axis=2)
        s = tl.load(s_ptr + rows[:, None] * G + g[None, :], mask=rmask[:, None] & gmask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(grp * s, axis=1)
    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=rmask)


def w4_gemv(x, packed, scale):
    """x: (..., I) with exactly one row; packed: (O, I/8) int32; scale: (O, I/32). Returns (..., O)."""
    O, KP = packed.shape
    xv = x.reshape(-1).contiguous()
    y = torch.empty(O, device=x.device, dtype=x.dtype)
    _w4_gemv3[lambda m: (triton.cdiv(O, m["BLOCK_O"]),)](xv, packed, scale, y, O, KP, scale.shape[1])
    return y.view(*x.shape[:-1], O)
