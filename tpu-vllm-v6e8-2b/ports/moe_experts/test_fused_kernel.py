#!/usr/bin/env python3
"""Local (CPU) validation of the v2 fused MoE expert kernel.

1. Numeric parity, Pallas INTERPRET mode (pl.pallas_call(..., interpret=True)
   on CPU jax): fused_experts_forward vs the v1 W4A16Experts torch reference
   (in-graph dequant + bmm) on mini shapes, fp32, including a padding expert
   index (== num_experts) and the down_pad_k zero-padding path.
   Target: max rel err < 1e-4 (accumulation-order differences only).
2. Shape/VMEM test at real 26B-A4B per-slice dims: prints the per-grid-step
   block byte math and asserts it fits the ~64 MB scoped VMEM budget.
   No TPU needed; no large arrays allocated.

Run: python3 test_fused_kernel.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
import torch

import fused_experts_kernel as fek
from quant_experts import W4A16Experts, quantize_expert_tensor

REL_TOL = 1e-4


def build_v1(E, H, I, seed):
    torch.manual_seed(seed)
    gate_up = torch.randn(E, 2 * I, H)
    down = torch.randn(E, H, I)
    mod = W4A16Experts(num_experts=E, hidden_dim=H, intermediate_dim=I)
    gu_p, gu_s = quantize_expert_tensor(gate_up)
    dn_p, dn_s = quantize_expert_tensor(down)
    mod.gate_up_packed.copy_(gu_p)
    mod.gate_up_scale.copy_(gu_s)
    mod.down_packed.copy_(dn_p)
    mod.down_scale.copy_(dn_s)
    return mod


def parity_case(name, E, H, I, T, K, seed, down_pad_k=None):
    mod = build_v1(E, H, I, seed)
    torch.manual_seed(seed + 1)
    hidden = torch.randn(T, H, dtype=torch.float32)
    idx = torch.randint(0, E, (T, K), dtype=torch.int64)
    idx[0, -1] = E  # padding expert: must be clamped + weight-zeroed
    wts = torch.softmax(torch.randn(T, K), dim=-1)

    ref = mod(hidden, idx, wts).detach().numpy()

    j = lambda t: jnp.asarray(t.detach().numpy())
    got = fek.fused_experts_forward(
        j(hidden), j(idx), j(wts),
        j(mod.gate_up_packed), j(mod.gate_up_scale),
        j(mod.down_packed), j(mod.down_scale),
        down_pad_k=down_pad_k, interpret=True,
    )
    got = np.asarray(got)

    assert got.shape == ref.shape, (got.shape, ref.shape)
    abs_err = np.abs(got - ref)
    denom = max(np.abs(ref).max(), 1e-30)
    rel = abs_err.max() / denom
    status = "PASS" if rel < REL_TOL else "FAIL"
    print(f"[{status}] {name}: E={E} H={H} I={I} T={T} K={K} "
          f"down_pad_k={down_pad_k}  max_abs={abs_err.max():.3e}  "
          f"max_rel={rel:.3e}  (tol {REL_TOL:.0e})")
    assert rel < REL_TOL, f"{name}: rel err {rel:.3e} >= {REL_TOL}"


def fmt_plan(label, plan):
    mb = plan["vmem_resident_bytes"] / (1 << 20)
    print(f"  {label}: out={plan['out_f']} k={plan['k']} (k_eff={plan['k_eff']}) "
          f"blk={plan['blk']} ck={plan['ck']} ck8={plan['ck8']} "
          f"out_blocks={plan['grid_out_blocks']}")
    print(f"    blocks: x={plan['x_block_bytes']:,} B  "
          f"packed={plan['packed_block_bytes']:,} B  "
          f"scale={plan['scale_block_bytes']:,} B  "
          f"out={plan['out_block_bytes']:,} B  "
          f"scratch={plan['kernel_scratch_bytes']:,} B")
    print(f"    resident (2x io + scratch): {plan['vmem_resident_bytes']:,} B "
          f"= {mb:.2f} MB  fits_64mb={plan['fits_64mb']}")
    assert plan["fits_64mb"], f"{label} exceeds 64 MB VMEM budget"


def shape_test_26b():
    """Real gemma-4-26B-A4B dims: E=128, H=2816, I=704, batch=8 decode top-8
    => N = T*K = 64 selected slices. bf16 activations, fp32 scales."""
    E, H, I, T, K = 128, 2816, 704, 8, 8
    N = T * K
    print(f"[SHAPE] 26B-A4B per-slice block/VMEM math (N={N} slices):")
    gu = fek.block_plan(out_f=2 * I, k=H, x_bytes=2, scale_bytes=4)
    dn = fek.block_plan(out_f=H, k=I, x_bytes=2, scale_bytes=4)
    dn_pad = fek.block_plan(out_f=H, k=I, x_bytes=2, scale_bytes=4, pad_k_to=1024)
    fmt_plan("gate_up  [2816x1408 @ k=2816]", gu)
    fmt_plan("down     [2816x 704 @ k= 704]", dn)
    fmt_plan("down+pad [k padded 704->1024]", dn_pad)

    # HBM transients for the gathered slices (v2) vs v1's dequantized temps.
    gu_gather = N * 2 * I * (H // 8) * 4 * 2      # packed int32 + fp32 rep4 scale
    dn_gather = N * H * (I // 8) * 4 * 2
    v1_gu = N * 2 * I * H * 2                     # bf16 dequantized weights
    v1_dn = N * H * I * 2
    print(f"  HBM transients/layer: v2 gathered packed+scales "
          f"{(gu_gather + dn_gather) / (1 << 20):.1f} MB  vs  "
          f"v1 dequantized bf16 weights {(v1_gu + v1_dn) / (1 << 20):.1f} MB "
          f"(the 42.13 GB b8 OOM came from these, live across layers)")

    # Tiling sanity at real dims.
    assert gu["blk"] == 128 and gu["ck"] == 256
    assert dn["blk"] == 256 and dn["ck"] == 704   # single full-size chunk
    assert dn_pad["ck"] == 1024 and dn_pad["ck8"] == 128
    print("[PASS] shape test: all blocks fit the 64 MB scoped VMEM budget")


def main():
    print(f"jax {jax.__version__} backend={jax.default_backend()} "
          f"torch {torch.__version__} (Pallas interpret mode)")
    # Mini config: single out block, single contraction chunk, both matmuls.
    parity_case("mini", E=8, H=64, I=32, T=5, K=2, seed=0)
    # down_pad_k exercises the zero-padded contraction path (k 32 -> 64).
    parity_case("mini+pad", E=8, H=64, I=32, T=5, K=2, seed=1, down_pad_k=64)
    # Multi-block, multi-chunk: gate_up out=384 (3 blocks of 128), k=1152
    # (3 chunks of 384); down out=1152 (9 blocks of 128), k=192.
    parity_case("multiblock", E=8, H=1152, I=192, T=3, K=4, seed=2)
    shape_test_26b()
    print("ALL PASS")


if __name__ == "__main__":
    main()
