A follow-up to [the E2B weight-format sweep](https://devcommunity.amd.com/t/ten-weight-formats-of-gemma-4-e2b-on-one-mi300x-fp8-is-the-only-one-at-bf16s-pace/1171). E2B is small enough that each token reads few weight bytes, so I ran the rest of the Gemma 4 family on the same MI300X Developer Cloud droplet and the same vLLM image (`vllm/vllm-openai-rocm` nightly, vLLM `0.31.1rc1.dev23`): E4B, 12B, 26B-A4B and 31B, each in bf16, fp8, int8 W8A8 and int4 W4A16, timed at 1, 8 and 64 parallel requests with 128, 1,024 and 8,192-token prompts.

fp8 and int4 against bf16, output tokens per second:

| Size | bf16, 1 request | fp8, 1 request | fp8 vs bf16, 9 cells | int4 vs bf16, 9 cells |
|---|---:|---:|---|---|
| E2B | 343 | 259 | 0.75x – 1.08x | 0.14x – 0.63x |
| E4B | 239 | 212 | 0.89x – 1.19x | 0.14x – 0.54x |
| 12B | 134 | 152 | 1.12x – 1.41x | 0.16x – 0.57x |
| 26B-A4B | 224 | 222 | 0.99x – 1.23x | 0.27x – 0.69x |
| 31B | 58 | 71 | 1.20x – 1.45x | 0.19x – 0.59x |

What the larger sizes add:

- **fp8 overtakes bf16 between E4B and 12B.** At 12B and 31B it is faster than bf16 in every cell, with the largest margin at 8 requests: 1.41x and 1.45x. At 31B it loads in 30.63 GiB against 58.99.
- **26B-A4B behaves like a small model.** Each token goes through a few of its 128 experts, so bf16 runs at 224 tok/s for one request, next to E4B's 239, and fp8 sits at 0.99x there.
- **All three 26B-A4B formats ran on vLLM's default expert-kernel settings.** The image has no tuned config for this shape on the MI300X: `Config file not found at .../fused_moe/configs/E=128,N=704,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json`. A tuned config for `E=128,N=704` could move the 26B numbers, and I'd be interested if anyone has generated one.
- **int4 W4A16 stays at 0.14x to 0.69x at every size**, through `TritonW4A16LinearKernel` for the dense models and the Triton WNA16 MoE backend for 26B-A4B. The gap has the same shape at every size, which points at the kernel.
- **The E4B and 12B int8 builds with int4 embedding tables stop at load** on this image with `AttributeError: 'VocabParallelEmbedding' object has no attribute 'weight'`.

On this card: fp8 for 12B and up, bf16 for a single user on the smaller sizes, fp8 for many.

The full write-up, with every table, the commands and the run logs: https://dev.to/gde/gemma-4-from-e2b-to-31b-on-an-amd-mi300x-fp8-overtakes-bf16-from-12b-up-h4e

Code and evidence: https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b
