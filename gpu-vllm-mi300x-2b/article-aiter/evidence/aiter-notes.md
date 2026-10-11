# VLLM_ROCM_USE_AITER=1, 12B fp8, 2026-10-09

Against `../2026-10-09-stock-mi300x` (same droplet, day and image digest `ec62abec…`): 0.974x to
0.994x output tok/s over the nine cells, every cell slower, repeat spread under 1%.

What the flag changed, from the two boot logs:

- Attention: unchanged, `TRITON_ATTN` in both. vLLM falls back to it for Gemma 4 when no
  `--attention-backend` is given ("heterogeneous head dimensions {'sliding_attention': 256,
  'full_attention': 512}. FA4 not available"). The driver gave none. The next line says "selected via
  --attention-backend" because the fallback writes the same setting the flag does.
  `ROCM_AITER_UNIFIED_ATTN` exists at this vLLM commit (43b4aaea3) and accepts head size >= 32; not run.
- fp8 linear layers: still `RowWiseTorchFP8ScaledMMLinearKernel`, now calling AITER's a8w8 GEMM, which
  has no tuned config for any 12B shape: 924 lines of `not found tuned config … will use default config!`.
- RMSNorm: `rms_norm=['aiter', 'native']`.

So this measures AITER's untuned fp8 GEMM against the stock path, with Gemma 4's attention left on
vLLM's Triton fallback. The stock run reproduces the 2026-10-07 sweep's 12B fp8 to within ±2.1% per cell on a
different droplet.
