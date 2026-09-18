# End-to-end fp8 vs bf16 — MI300X, Gemma 4 E2B, 2026-09-18

**fp8 serving is slower than bf16 on this card: 0.53x–0.80x output throughput.** The GEMM-level
measurement (`../2026-09-18-gemm-decode-shapes-mi300x/`, fp8 1.50x bf16 summed per token) does not
carry through vLLM's fp8 path.

| concurrency | input | bf16 out tok/s | fp8 out tok/s | fp8/bf16 | bf16 TPOT ms | fp8 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 276.9 | 202.5 | 0.73 | 3.59 | 4.91 |
| 1 | 1024 | 262.6 | 194.6 | 0.74 | 3.76 | 5.08 |
| 8 | 128 | 1665.4 | 1304.2 | 0.78 | 4.73 | 6.07 |
| 8 | 1024 | 1557.6 | 1247.3 | 0.80 | 4.91 | 6.24 |
| 64 | 128 | 10009.4 | 6579.3 | 0.66 | 6.08 | 9.50 |
| 64 | 1024 | 7918.8 | 4229.6 | 0.53 | 7.20 | 14.33 |

Output 512 tokens, 3 repeats per cell (median-throughput run reported), same prompt seeds
(`--seed-base 5000`) for both, each on a freshly started container. Ratios computed from
`../../reports/2026-09-18-vllm-sweep-mi300x-v0191-{fp8,bf16}.json`.

**Against the GEMM prediction.** The GEMM sweep says fp8 should save 0.66 ms per token at M=1 and
0.78 ms at M=64. Measured, fp8 *adds* 1.32 ms at concurrency 1, 1.33 ms at 8, and 3.42–7.13 ms at 64.
The cost grows with batch and context, so it lives in something that scales with tokens in flight —
the per-call dynamic activation quantisation is the first suspect; not isolated here.

## The engine really served fp8

| | bf16 | fp8 |
| --- | --- | --- |
| engine log | `quantization=None` | `quantization=fp8` |
| linear kernel | — | `ROCmFP8ScaledMMLinearKernel for Fp8OnlineLinearMethod` |
| model load | — | 7.68 GiB |
| KV cache | 5,499,584 tokens | 5,570,336 tokens |
| sanity prompt | — | "17 times 23 is 391, and the capital of Australia is Canberra." |

Attention backend `TRITON_ATTN` in both.

## Why this ran on vLLM 0.19.1 and not the rig's nightly

`--quantization fp8` **cannot start** on `vllm/vllm-openai-rocm:nightly-rocm100` as pulled
2026-09-18 (vLLM 0.3.1.dev85+gdee37d891). Both quantisation paths fail on `e4m3fnuz`:

- **torch.compile path:** Inductor emits `tl.full(..., float("-inf"), <fp8 dtype>)` and the MLIR
  lowering rejects it — `'llvm.fptrunc' op result #0 must be floating point ... but got 'i8'`.
  `e4m3fnuz` has no infinity to fill with. The engine dies during encoder profiling and
  `--restart unless-stopped` loops it (15 restarts observed).
- **custom-op path** (`--compilation-config '{"custom_ops":["+quant_fp8"]}'`):
  `"scaled_fp8_quant_kernel_scalar_type" not implemented for 'Float8_e4m3fnuz'` from the image's
  compiled HIP kernel.

So both arms of the A/B use `rocm/vllm:rocm7.13.0_gfx94X-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`
(vLLM 0.19.1+rocm7.13.0rc2). A bf16 sweep on the nightly with the same grid and seeds is kept as
`2026-09-18-vllm-sweep-mi300x-bf16ctl`; the nightly is 1.14x–1.24x faster than 0.19.1 in bf16, so
**never difference an fp8 figure from one image against a bf16 figure from the other.**

## Noise

Two cells are noisy in both arms: c8-in128 (cv 22.6% bf16, 18.1% fp8) and c64-in128 fp8 (10.8%).
Every other cell is under 7%. No conclusion above rests on those cells alone — the ratio is below
1 in all six.
