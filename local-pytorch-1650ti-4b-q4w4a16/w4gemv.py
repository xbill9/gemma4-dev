"""Int4 W4A16 matrix-vector product, v5: the v4 kernel with each shape's autotuned configuration fixed.

No autotuning at start-up, and the same tiling (so the same summation order) on every run.

Same layout as v1 (compressed-tensors pack-quantized, symmetric, group 32): packed[o, j] holds input columns
8j..8j+7, low nibble first, offset 8; scale[o, g] covers columns 32g..32g+31.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _w4_gemv(x_ptr, p_ptr, s_ptr, y_ptr, O, KP, G, BLOCK_O: tl.constexpr, BLOCK_G: tl.constexpr):
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


# (O, KP) -> (BLOCK_O, BLOCK_G, num_warps, num_stages), best of a grid on E4B's shapes on the GTX 1650 Ti
# (benchmarks/runs/2026-10-31-tune-1650ti/tune.jsonl). A merged q+k+v is 3072 or 6144 rows, gate+up 20480.
CONFIGS = {
    (2048, 320): (8, 16, 1, 4),
    (2560, 256): (16, 32, 4, 3),
    (2560, 512): (16, 32, 4, 1),
    (2560, 1280): (16, 32, 4, 1),
    (3072, 320): (8, 16, 1, 2),
    (4096, 320): (8, 16, 1, 2),
    (6144, 320): (8, 16, 1, 1),
    (20480, 320): (8, 16, 1, 1),
    (262144, 320): (8, 16, 1, 4),
}
DEFAULT = (8, 16, 1, 1)  # for shapes not in the table


def w4_gemv(x, packed, scale):
    """x: (..., I) with exactly one row; packed: (O, I/8) int32; scale: (O, I/32). Returns (..., O)."""
    O, KP = packed.shape
    xv = x.reshape(-1).contiguous()
    y = torch.empty(O, device=x.device, dtype=x.dtype)
    bo, bg, w, st = CONFIGS.get((O, KP), DEFAULT)
    _w4_gemv[(triton.cdiv(O, bo),)](xv, packed, scale, y, O, KP, scale.shape[1], BLOCK_O=bo, BLOCK_G=bg,
                                     num_warps=w, num_stages=st)
    return y.view(*x.shape[:-1], O)
