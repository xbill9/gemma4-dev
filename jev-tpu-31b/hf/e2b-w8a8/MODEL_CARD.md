---
library_name: vllm
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model:
- google/gemma-4-E2B-it-qat-q4_0-unquantized
base_model_relation: quantized
tags:
- gemma4
- compressed-tensors
- w8a8
- int8
- qat
- vllm
- tpu
- text-only
---

# Gemma 4 E2B-it QAT, compressed-tensors int8 W8A8, text only (unofficial)

**This is an unofficial build, made and published independently of Google.** It is Google's quantization-aware-trained (QAT) Gemma 4 E2B-it, from [`google/gemma-4-E2B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized), stored as compressed-tensors int8 W8A8 and loadable as a text-only `Gemma4ForCausalLM`. Google's model card is kept unchanged in `ORIGINAL_README.md`.

## What it is

- **Scheme:** int8 weights with one bf16 scale per output channel (`weight_scale` `[out, 1]`); activations quantized to int8 per token at run time (`format: int-quantized`). The same scheme as an llm-compressor W8A8 export.
- **Source:** the QAT weights, which already sit on the Q4_0 grid, rounded to int8 per channel. The int8 values are within 0.6–1.7% relative error of the QAT values (mean about 0.8%).
- **Quantized:** every attention and MLP linear of the language model and the per-layer-embedding projections (`per_layer_input_gate`, `per_layer_projection`, `per_layer_model_projection`).
- **Kept bf16:** the token embedding (tied to the LM head), the per-layer embedding table, norms.
- **Dropped:** the vision and audio towers.
- **Size:** 6.88 GiB.

Made with [`w8a8_from_qat.py`](https://github.com/xbill9/gemma4-dev/blob/main/jev-tpu-31b/w8a8_from_qat.py).

## Measured on one TPU v5e chip

One `v5litepod-1`, vLLM TPU `vllm/vllm-tpu@sha256:19a1a052…` with the patches below, all builds with the same flags. Suite: a 3,880-record public benchmark read by label probability, paired record for record.

| E2B, one v5e chip | Suite | Output tok/s at 1 / 4 / 16 requests | First-token latency |
|---|---:|---|---:|
| `google/gemma-4-E2B-it` (bf16) | 68.3% | 144 / 560 / 2,008 | 12.6 ms |
| `google/gemma-4-E2B-it-qat-q4_0-unquantized`, repacked as W4A16 | 67.8% | 136 / 532 / 1,906 | 16.2 ms |
| `glenic/gemma-4-E2B-it-W8A8-INT8` (W8A8 rounded from bf16) | 67.0% | 220 / 842 / 2,876 | 10.7 ms |
| **This checkpoint** | **68.6%** | **220 / 841 / 2,872** | **10.4 ms** |

Against bf16 it is +0.3 points (95% range −0.7 to +1.3), and against the W8A8 build rounded from the original bf16 weights +1.5 (+0.5 to +2.6). On v5e int8 x int8 runs natively in the matrix units at twice the bf16 rate, which is where the 1.5x throughput comes from. Logs and per-record outputs: https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-v5e1

## Serving it

**Google Cloud TPU.** vLLM's TPU backend has no int8 W8A8 method on its JAX path (only fp8 W8A8) and reads `hf_config.text_config`; both are added by the patches in [`tpu-vllm-v5e1-2b-w8a8`](https://github.com/xbill9/gemma4-dev/tree/main/tpu-vllm-v5e1-2b-w8a8), which also serves it. Measured with `--max-model-len 2048 --gpu-memory-utilization 0.80 --max-num-batched-tokens 512` and `MIN_TOKEN_BUCKET=64`.

**NVIDIA GPUs.** Untested. vLLM's CUDA path supports this compressed-tensors W8A8 scheme.

## Limitations

- Text only: image and audio inputs are not supported.
- Measured on TPU v5e only.
- Unofficial. Report problems here, not to Google.

## License and attribution

Gemma 4 is released by Google DeepMind under the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository redistributes Google's weights in a changed format, under the same license. The weights, training and the original model card are Google DeepMind's; this build and card are not affiliated with or endorsed by Google.
