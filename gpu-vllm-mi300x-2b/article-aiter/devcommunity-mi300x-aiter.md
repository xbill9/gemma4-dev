A follow-up to [the Gemma 4 size sweep on one MI300X](https://devcommunity.amd.com/t/gemma-4-from-e2b-to-31b-on-one-mi300x-fp8-overtakes-bf16-from-12b-up/1174). I ran Gemma 4 12B in fp8 twice on the same Developer Cloud droplet, the same day and the same image (`vllm/vllm-openai-rocm` nightly, vLLM `0.31.1rc1.dev23`): once stock and once with `VLLM_ROCM_USE_AITER=1`. Same grid as before, 1, 8 and 64 requests by 128, 1,024 and 8,192-token prompts, three repeats per cell.

| Requests | Prompt tokens | AITER tok/s | Stock tok/s | AITER / stock |
|---:|---:|---:|---:|---:|
| 1 | 128 | 148.3 | 152.2 | 0.974 |
| 8 | 1,024 | 896.5 | 917.1 | 0.978 |
| 64 | 1,024 | 2,694.7 | 2,731.0 | 0.987 |
| 64 | 8,192 | 849.4 | 857.7 | 0.990 |

All nine cells are slower with AITER, 0.974x to 0.994x, with repeats varying by 0.78% or less. The boot logs show what the flag changed:

- **Attention stays on Triton.** vLLM forces it for Gemma 4: `Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN backend.`
- **The fp8 GEMMs move to AITER, untuned.** The log carries 924 lines like `[aiter] shape is M:1, N:8192, K:3840, q_dtype_w:torch.float8_e4m3fnuz, not found tuned config in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv, will use default config!` That is all six of 12B's weight shapes (N x K of 8192x3840, 9216x3840, 30720x3840, 3840x4096, 3840x8192 and 3840x15360), each at 77 token counts, in both the `a8w8` and `a8w8_bpreshuffle` tables.
- **RMSNorm moves to AITER** (`rms_norm=['aiter', 'native']`).

Two things would let AITER help Gemma 4 here: tuned `a8w8` configs for those six shapes on gfx942, and an attention path that serves 512-wide heads alongside 256-wide ones. If anyone has tuned a8w8 entries for these shapes, or knows whether AITER's attention is planned to cover head size 512, I'd like to rerun with them.

The stock run reproduced the earlier sweep's 12B fp8 numbers within 2.1% on a different droplet a day later, so the platform is steady enough to read a 1% to 3% difference.

The full write-up: https://dev.to/gde/vllmrocmuseaiter1-slows-gemma-4-on-an-amd-mi300x-what-the-flag-changes-and-why-69f

Code, logs and evidence: https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b
