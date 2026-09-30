---
library_name: vllm
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model:
- google/gemma-4-12B-it-qat-q4_0-unquantized
base_model_relation: quantized
tags:
- gemma4
- compressed-tensors
- w8a8
- int8
- int4
- qat
- vllm
- tpu
- text-only
---

# Gemma 4 12B-it QAT, int8 W8A8 with int4 embeddings, text only (unofficial)

**This is an unofficial build, made and published independently of Google.** It is Google's quantization-aware-trained (QAT) Gemma 4 12B-it, from [`google/gemma-4-12B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized), with int8 W8A8 linear layers and int4 token embeddings and output head, loadable as a text-only `Gemma4ForCausalLM`. Google's model card is kept unchanged in `ORIGINAL_README.md`.

It is built to fit one 16 GB TPU v5e chip, where it serves at bf16 accuracy.

## What it is

- **Linear layers:** compressed-tensors int8 W8A8: int8 weights with one bf16 scale per output channel, activations quantized to int8 per token at run time. Rounded from the QAT weights, which already sit on the Q4_0 grid: 328 tensors within 0.78–1.59% relative error of the QAT values (mean 0.91%).
- **`embed_tokens` and `lm_head`:** pack-quantized symmetric int4, group 32, fp16 scales. The QAT weights put the embedding on the same 4-bit grid as the linears, so each group's step is recovered, not chosen. `lm_head` is an untied copy of the same values (`tie_word_embeddings: false`), so the output layer runs as an int4 linear.
- **Kept bf16:** norms and scalars.
- **Dropped:** the vision tower.
- **Size:** 11.21 GiB, against 12.03 GiB for the same W8A8 build with bf16 embeddings.

Made with [`w8a8_from_qat.py`](https://github.com/xbill9/gemma4-dev/blob/main/jev-tpu-31b/w8a8_from_qat.py) and [`w8a8_emb4.py`](https://github.com/xbill9/gemma4-dev/blob/main/jev-tpu-31b/w8a8_emb4.py).

## Measured on one TPU v5e chip

One `v5litepod-1` (15.75 GiB of HBM), vLLM TPU `vllm/vllm-tpu@sha256:19a1a052…` with the patches below. Suite: a 3,880-record public benchmark read by label probability, paired record for record against `google/gemma-4-12B-it` in bf16 (on a larger chip, since bf16 12B does not fit v5e).

| 12B on one v5e chip | Weights on chip | Suite | Output tok/s at 1 / 4 / 16 requests | KV tokens |
|---|---:|---:|---|---:|
| bf16 (measured on v6e) | — | 76.0% | — | — |
| W4A16 QAT repack | 7.59 GiB | 75.8% | — | 5,120 |
| W4A16 with int4 embeddings | 6.86 GiB | 76.2% | 35 / 105 / 433 | 13,824 |
| **This checkpoint** | **11.31 GiB** | **76.1%** | **57 / 217 / 723** | **9,728** |

Against bf16 it is +0.1 points (95% range −0.6 to +0.8). On v5e int8 x int8 runs natively in the matrix units at twice the bf16 rate; int4 is storage only. Logs and per-record outputs: https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-v5e1

## Serving it

**Google Cloud TPU v5e.** vLLM's TPU backend has no int8 W8A8 method on its JAX path, no int4 embedding table and no quantized output head, and reads `hf_config.text_config`; the patches in [`tpu-vllm-v5e1-12b-w8a8emb4`](https://github.com/xbill9/gemma4-dev/tree/main/tpu-vllm-v5e1-12b-w8a8emb4) add them, and that rig serves it. Measured with `--gpu-memory-utilization 0.92 --max-model-len 8192 --max-num-batched-tokens 512` and `MIN_TOKEN_BUCKET=64`. Memory is the limit on this chip: a larger KV cache, a block override or an fp8 KV cache each failed at start-up.

**NVIDIA GPUs.** Untested. vLLM's CUDA path supports compressed-tensors W8A8 linears and int4 embedding groups separately.

## Limitations

- Text only: image inputs are not supported.
- Measured on TPU v5e only, at 8,192 tokens of context.
- Unofficial. Report problems here, not to Google.

## License and attribution

Gemma 4 is released by Google DeepMind under the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository redistributes Google's weights in a changed format, under the same license. The weights, training and the original model card are Google DeepMind's; this build and card are not affiliated with or endorsed by Google.
