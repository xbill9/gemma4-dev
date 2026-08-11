#!/usr/bin/env python3.12
"""Single-layer MoE expert bench ON THE TPU VM: v1 in-graph dequant
(quant_experts.W4A16Experts, the path whose bf16 HLO temporaries hit
42.13 GB at 26B batch=8) vs v2 fused Pallas dequant-in-matmul kernel
(fused_experts_kernel.py), at real 26B-A4B dims: E=128, H=2816, I=704,
T*K = 8*8 = 64 selected slices per layer.

DO NOT run on the workstation — TPU only. To run later (from the repo root):

    gcloud compute tpus tpu-vm scp \
        ports/moe_experts/quant_experts.py \
        ports/moe_experts/fused_experts_kernel.py \
        ports/moe_experts/fused_experts_bench_tpu.py \
        <VM_NAME>:~/ --zone=<ZONE>
    gcloud compute tpus tpu-vm ssh <VM_NAME> --zone=<ZONE> \
        --command='python3.12 fused_experts_bench_tpu.py'

Optional args: python3.12 fused_experts_bench_tpu.py [T] [K]
If Mosaic rejects the down-proj ck8=88 lane concat (k=704 is a single
full-size chunk, plane width 88), set DOWN_PAD_K = 1024 below — validated
numerically in interpret mode by test_fused_kernel.py (mini+pad case).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import numpy as np
import torch

import importlib
import os

# The PyTorch TPU backend package is not publicly distributable, so its name is
# not committed here: set TPU_BACKEND_MODULE (e.g. in /etc/environment on the VM)
# to the installed backend package. Imported dynamically for the Pallas bridge.
_BACKEND = os.getenv("TPU_BACKEND_MODULE")
if not _BACKEND:
    raise RuntimeError(
        "TPU_BACKEND_MODULE is not set — export it to the installed PyTorch TPU "
        "backend package name so the Pallas bridge can be imported."
    )
pallas = importlib.import_module(f"{_BACKEND}._internal.pallas")  # registers the TPU backend

import fused_experts_kernel as fek
from quant_experts import GROUP_SIZE, W4A16Experts

E, H, I = 128, 2816, 704
T = int(sys.argv[1]) if len(sys.argv) > 1 else 8
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
WARMUP, STEPS = 3, 32
DOWN_PAD_K = None  # set to 1024 if Mosaic rejects the 88-wide lane concat


def _moe_fused_jax(hidden: jax.Array, idx: jax.Array, wts: jax.Array,
                   gu_p: jax.Array, gu_s: jax.Array,
                   dn_p: jax.Array, dn_s: jax.Array) -> jax.Array:
    return fek.fused_experts_forward(
        hidden, idx, wts, gu_p, gu_s, dn_p, dn_s, down_pad_k=DOWN_PAD_K
    )


moe_fused = pallas.jax_op("w4a16::moe_experts", _moe_fused_jax)


class FusedExperts(torch.nn.Module):
    """v2: same buffers/layout as W4A16Experts, forward via the fused kernel."""

    def __init__(self, v1: W4A16Experts):
        super().__init__()
        self.register_buffer("gu_p", v1.gate_up_packed.clone())
        self.register_buffer("gu_s", v1.gate_up_scale.clone())
        self.register_buffer("dn_p", v1.down_packed.clone())
        self.register_buffer("dn_s", v1.down_scale.clone())

    def forward(self, hidden, top_k_index, top_k_weights):
        return moe_fused(
            hidden, top_k_index, top_k_weights,
            self.gu_p, self.gu_s, self.dn_p, self.dn_s,
        )


def random_buffers(mod: W4A16Experts, seed: int = 0) -> None:
    """Random packed nibbles + small positive scales (perf bench: values need
    not come from a real quantization, only the layout/dtypes must match)."""
    rng = np.random.default_rng(seed)

    def rand_packed(shape):
        bits = rng.integers(0, 1 << 32, size=shape, dtype=np.uint32)
        return torch.from_numpy(bits.view(np.int32))

    mod.gate_up_packed.copy_(rand_packed((E, 2 * I, H // 8)))
    mod.down_packed.copy_(rand_packed((E, H, I // 8)))
    torch.manual_seed(seed)
    mod.gate_up_scale.copy_(torch.rand(E, 2 * I, H // GROUP_SIZE) * 0.02 + 1e-3)
    mod.down_scale.copy_(torch.rand(E, H, I // GROUP_SIZE) * 0.02 + 1e-3)


def bench(name, fn, hidden, idx, wts):
    with torch.no_grad():
        for _ in range(WARMUP):
            out = fn(hidden, idx, wts)
        out[0, :1].cpu()  # force materialization (includes compile)
        t0 = time.monotonic()
        for _ in range(STEPS):
            out = fn(hidden, idx, wts)
        out[0, :1].cpu()
        elapsed = time.monotonic() - t0
    ms = 1000 * elapsed / STEPS
    print(f"MARKER {name}: {STEPS} steps in {elapsed:.3f}s = {ms:.2f} ms/step")
    return ms


def main() -> int:
    device = torch.device("tpu")
    print(f"MARKER bench config: E={E} H={H} I={I} T={T} K={K} "
          f"(N={T * K} slices) DOWN_PAD_K={DOWN_PAD_K}")

    v1 = W4A16Experts(num_experts=E, hidden_dim=H, intermediate_dim=I)
    random_buffers(v1)
    v2 = FusedExperts(v1)
    v1 = v1.to(device).eval()
    v2 = v2.to(device).eval()

    torch.manual_seed(1)
    idx = torch.randint(0, E, (T, K), dtype=torch.int32).to(device)
    wts32 = torch.softmax(torch.randn(T, K), dim=-1)
    h32 = torch.randn(T, H)

    # ---- numeric cross-check (fp32, one shot) ----
    hidden32, w32 = h32.to(device), wts32.to(device)
    with torch.no_grad():
        ref = v1(hidden32, idx, w32).cpu()
        got = v2(hidden32, idx, w32).cpu()
    rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
    print(f"MARKER parity v1 vs v2 (fp32): max_rel={rel:.3e} "
          f"({'OK' if rel < 1e-3 else 'CHECK'})")

    # ---- timing (bf16 activations, compiled) ----
    hidden = h32.to(torch.bfloat16).to(device)
    wts = wts32.to(torch.bfloat16).to(device)
    v1_c = torch.compile(lambda h, i, w: v1(h, i, w), backend="tpu", dynamic=False)
    v2_c = torch.compile(lambda h, i, w: v2(h, i, w), backend="tpu", dynamic=False)

    ms1 = bench("v1 in-graph dequant + bmm ", v1_c, hidden, idx, wts)
    ms2 = bench("v2 fused Pallas kernel    ", v2_c, hidden, idx, wts)
    print(f"MARKER speedup v2/v1: {ms1 / ms2:.2f}x "
          f"({ms1:.2f} -> {ms2:.2f} ms/layer-step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
