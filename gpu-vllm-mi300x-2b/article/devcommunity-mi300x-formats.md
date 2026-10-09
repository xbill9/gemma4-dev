A follow-up to [my earlier post on the numeric formats an MI300X executes](https://devcommunity.amd.com/t/which-numeric-formats-an-mi300x-actually-executes-measured-on-a-developer-cloud-droplet/1043). That post timed raw matrix multiplies. This one serves a real model: ten weight formats of Gemma 4 E2B, one after another on one MI300X Developer Cloud droplet, all on the same vLLM image (`vllm/vllm-openai-rocm` nightly, vLLM `0.31.1rc1.dev23`), timed at 1, 8 and 64 parallel requests with 128, 1,024 and 8,192-token prompts.

Output tokens per second with 1,024-token prompts, and the range against bf16 across all nine cells:

| Build | Weights | 1 / 8 / 64 requests | vs bf16 |
|---|---:|---|---|
| bf16 | 9.42 GiB | 321 / 1,795 / 8,951 | 1.00x |
| fp8 (E4M3) | 7.07 GiB | 242 / 1,827 / 9,700 | 0.75x – 1.08x |
| fp8 (E4M3FNUZ) | 7.07 GiB | 245 / 1,819 / 9,739 | 0.75x – 1.09x |
| int8 W8A8 | 7.07 GiB | 98 / 757 / 4,717 | 0.29x – 0.87x |
| int4 W4A16 | 6.32 GiB | 47 / 367 / 3,288 | 0.14x – 0.63x |

What carried over from the matrix-multiply numbers, and what changed:

- **fp8 is still the only format at bf16's pace.** The 1.5x it showed on raw multiplies shrinks to 0.75x for a single request and up to 1.09x with 8 or 64 in flight.
- **An E4M3 checkpoint from the Hub serves fine.** A raw E4M3 multiply still raises `HIPBLAS_STATUS_NOT_SUPPORTED`, but vLLM's compressed-tensors loader converts the weights to E4M3FNUZ at load and picks `RowWiseTorchFP8ScaledMMLinearKernel`. A checkpoint built natively in E4M3FNUZ loads into the same kernel and runs at 0.995x to 1.014x of it. So the advice in the first post to quantize online is no longer needed with this image.
- **int8 stays slow** through `TritonInt8ScaledMMLinearKernel`, below fp8 in eight of nine cells.
- **int4 W4A16 runs at 20.91 ms per output token against bf16's 2.89 ms** for one request, through `TritonW4A16LinearKernel`, which unpacks the weights to bf16 before each multiply. A tuned int4 path for gfx942 would change this more than anything else here.
- **int4 embedding tables cost 1% to 5% of speed** and halve the weights, which matters little with 192 GB.

For a single user on this card, bf16 is the fastest build. For many concurrent users, fp8 in either flavour.

The full write-up, with every build linked on Hugging Face, the build commands and the run logs: https://dev.to/gde/gemma-4-e2b-on-an-amd-mi300x-which-weight-format-should-you-serve-1a01

Code and evidence: https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b
