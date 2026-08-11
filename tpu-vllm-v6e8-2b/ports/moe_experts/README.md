# w4a16 Gemma 4 MoE experts (`W4A16Experts`)

TPU-compile-friendly int4-quantized replacement for HF transformers 5.12.1's
`Gemma4TextExperts` (the routed-experts module of `google/gemma-4-26B-A4B`).
Plain torch only — no backend package import, no PyTorch/XLA.

## What it replaces

`transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts` stores fused 3D
parameters per layer — `gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]` — and
dispatches with a **data-dependent Python loop** over hit experts
(`one_hot -> nonzero -> for expert_idx in expert_hit`), which cannot compile
into a single static TPU graph. `W4A16Experts` keeps the exact eager math
(`gate, up = linear(x, gate_up[e]).chunk(2); down(act(gate) * up) * w[t, k]`,
summed over the top-k) and the exact forward signature
`forward(hidden_states [T, H], top_k_index [T, K], top_k_weights [T, K])`,
but replaces the loop with a static gather + `torch.bmm` formulation:

1. `index_select` the K selected experts' **packed int4** slices per token
   (`[T*K, 2I, H/8]` / `[T*K, H, I/8]`),
2. dequantize the gathered slices in-graph (shift/mask/scale — all traceable),
3. two `torch.bmm`s against per-(token, k) hidden states, gelu-tanh gating,
   weight by `top_k_weights`, sum over K.

All shapes are static functions of `(T, K)`; verified with
`torch.compile(backend="aot_eager", fullgraph=True)` (check c in
`test_experts.py`).

Quantization: DIY symmetric int4 (there is no published w4a16 checkpoint for
26B-A4B), group_size 32 along the contraction dim, compressed-tensors
pack-quantized nibble layout (stored `q+8` in `[0,15]`, nibble *i* of int32
word *j* = column `8j+i`), fp32 group scales (cast to bf16 for the real model
if the scale footprint matters).

## Memory math — real 26B-A4B on v6e-1

Config: 128 experts, top-8, hidden 2816, `moe_intermediate_size` 704, 30 layers.

| tensor | per layer | 30 layers |
|---|---|---|
| `gate_up_proj` 128 x 1408 x 2816 | 507.5 M params | 15.23 B |
| `down_proj` 128 x 2816 x 704 | 253.8 M params | 7.61 B |
| **experts total** | 761.3 M params | **22.84 B** |

- int4 packed weights: 22.84 B x 0.5 B = **~11.4 GB**
- group scales (1 per 32 weights = 713.7 M): **~1.4 GB** bf16 (2.9 GB fp32)
- everything else (embeddings, attention, dense/shared MLP, routers — the
  ~3.2 B non-expert params) in bf16: ~6.4 GB

Total ~19-21 GB, comfortably inside a v6e-1's 32 GB HBM — impossible with
bf16 experts alone (45.7 GB).

## Wiring plan

Per decoder layer with `enable_moe_block`, swap the module instance only:

```python
layer.experts = W4A16Experts.from_hf(layer.experts)   # quantizes in place
```

The call site (`Gemma4TextDecoderLayer.forward`) is unchanged — it already
calls `self.experts(hidden_states_flat, top_k_index, top_k_weights)`. The
`Gemma4TextRouter` and the dense shared MLP (`Gemma4TextMLP`) stay bf16 and
are Linear-swapped separately (they are ordinary `nn.Linear`s, unlike the 3D
expert tensors handled here).

## Files

- `quant_experts.py` — `quantize_expert_tensor` / `dequantize_expert_tensor`
  (pack-quantized int4 g32 for 3D `[E, out, in]` tensors) and `W4A16Experts`
  with `from_hf(experts_module)`.
- `test_experts.py` — mini-setting (8 experts, H=64, I=32, k=2, 16 tokens,
  fp32) CPU checks: (a) parity vs an HF `Gemma4TextExperts` loaded with the
  dequantized weights (~1e-8), (b) int4 g32 round-trip error (~7% of absmax),
  (c) `fullgraph=True` compile with no graph breaks.
  Run: `python3 test_experts.py`
