"""CPU parity tests for W4A16Experts vs HF transformers' Gemma4TextExperts.

Mini setting: 8 experts, hidden 64, intermediate 32, top_k 2, 16 tokens, fp32.

Checks:
  a. QUANT-NEUTRALITY parity: W4A16Experts output == HF eager experts module
     loaded with the dequantized weights (same math, ~1e-5).
  b. Round-trip error of int4 g32 quantization (expect a few percent).
  c. STATIC check: torch.compile(..., backend="aot_eager", fullgraph=True)
     traces the forward without graph breaks and matches eager.

Run: python3 test_experts.py
"""

import copy
import sys

import torch

import os

# quant_experts sits next to this test. This was an absolute path to a rig checkout —
# in two of the three copies, to a rig that does not exist under this monorepo at all.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_experts import W4A16Experts, dequantize_expert_tensor, quantize_expert_tensor

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

NUM_EXPERTS = 8
HIDDEN = 64
INTERMEDIATE = 32
TOP_K = 2
TOKENS = 16


def build_hf_experts(seed: int = 0) -> Gemma4TextExperts:
    torch.manual_seed(seed)
    config = Gemma4TextConfig(
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_experts=NUM_EXPERTS,
        moe_intermediate_size=INTERMEDIATE,
        top_k_experts=TOP_K,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    experts = Gemma4TextExperts(config)
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.05)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.05)
    return experts.float().eval()


def build_routing(seed: int = 1):
    """Synthesize (hidden_states, top_k_index, top_k_weights) the way the HF
    MoE block produces them: flat [T, H] hidden states, and per-token top-k
    softmax router weights (topk of a softmax, renormalized to sum to 1 —
    the Gemma4TextRouter semantics, minus its learned per-expert scale)."""
    torch.manual_seed(seed)
    hidden_states = torch.randn(TOKENS, HIDDEN)
    router_logits = torch.randn(TOKENS, NUM_EXPERTS)
    probs = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float32)
    top_k_weights, top_k_index = torch.topk(probs, k=TOP_K, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    return hidden_states, top_k_index, top_k_weights


def check_a_parity(hf_experts, quant_experts, inputs) -> tuple[bool, str]:
    """W4A16Experts vs an HF module carrying the dequantized weights."""
    hf_dq = copy.deepcopy(hf_experts)
    gu_dq, dn_dq = quant_experts.dequantized_weights()
    with torch.no_grad():
        hf_dq.gate_up_proj.copy_(gu_dq)
        hf_dq.down_proj.copy_(dn_dq)

    with torch.no_grad():
        ref = hf_dq(*inputs)
        got = quant_experts(*inputs)

    max_diff = (ref - got).abs().max().item()
    ok = max_diff < 1e-5
    return ok, (
        f"max |W4A16Experts - dequantized-HF| = {max_diff:.3e} "
        f"(tol 1e-5, output scale max |ref| = {ref.abs().max().item():.3e})"
    )


def check_b_roundtrip(hf_experts) -> tuple[bool, str]:
    lines = []
    ok = True
    for name, w in (
        ("gate_up_proj", hf_experts.gate_up_proj.data),
        ("down_proj", hf_experts.down_proj.data),
    ):
        packed, scale = quantize_expert_tensor(w)
        dq = dequantize_expert_tensor(packed, scale)
        err = (w.float() - dq).abs()
        max_abs = err.max().item()
        rel = max_abs / w.float().abs().max().item()
        mean_rel = (err.mean() / w.float().abs().mean()).item()
        # int4 g32: max error per group is scale/2 = absmax/14 ~= 7% of absmax
        ok = ok and rel < 0.10
        lines.append(
            f"{name}: max abs {max_abs:.4e}, max/absmax rel {rel:.2%}, "
            f"mean/absmean rel {mean_rel:.2%}"
        )
    return ok, "; ".join(lines)


def check_c_static(quant_experts, inputs) -> tuple[bool, str]:
    compiled = torch.compile(quant_experts, backend="aot_eager", fullgraph=True)
    with torch.no_grad():
        eager_out = quant_experts(*inputs)
        compiled_out = compiled(*inputs)  # fullgraph=True: any break raises
    max_diff = (eager_out - compiled_out).abs().max().item()
    ok = max_diff < 1e-6
    return ok, (
        f"fullgraph aot_eager trace OK, |compiled - eager| max = {max_diff:.3e}"
    )


def main() -> int:
    hf_experts = build_hf_experts()
    inputs = build_routing()
    quant = W4A16Experts.from_hf(hf_experts).eval()

    results = [
        ("a. quant-neutrality parity", *check_a_parity(hf_experts, quant, inputs)),
        ("b. int4 g32 round-trip error", *check_b_roundtrip(hf_experts)),
        ("c. static compile (aot_eager fullgraph)", *check_c_static(quant, inputs)),
    ]

    all_ok = True
    for name, ok, detail in results:
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
