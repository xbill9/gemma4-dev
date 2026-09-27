---
library_name: vllm
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model:
- google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
base_model_relation: quantized
tags:
- gemma4
- compressed-tensors
- w4a16
- int4
- qat
- vllm
- tpu
---

# Gemma 4 26B-A4B-it QAT, compressed-tensors W4A16 (unofficial repack)

**This is an unofficial repack, made and published independently of Google.** It holds Google's quantization-aware-trained (QAT) weights for Gemma 4 26B-A4B-it, from [`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized), repacked into the compressed-tensors W4A16 format that vLLM loads. Google's model card is kept unchanged in `ORIGINAL_README.md`; the model, its intended uses and its limitations are as described there.

Google publishes compressed-tensors W4A16 QAT checkpoints for Gemma 4 E2B, E4B, 12B and 31B. Its QAT model card lists none for 26B-A4B, which ships as Q4_0 GGUF and as a 48 GiB bf16 export. This repository fills that gap.

## What it is

- **Format:** compressed-tensors `pack-quantized`, symmetric int4, group size 32, bf16 scales, activations unquantized (W4A16).
- **Quantized:** attention, the dense MLP, and all 3,840 experts (30 layers × 128), one module per expert: `experts.{i}.{gate,up,down}_proj`.
- **Kept bf16, copied byte for byte:** embeddings, router, norms, and the vision tower.
- **Size:** 15.29 GiB, against 48.07 GiB for the bf16 source.

## How it was made

The `-qat-q4_0-unquantized` export stores bf16 values that already lie on a 4-bit grid: within every group of 32 weights along the input dimension, each weight is `step × level` with level from −8 to 7. The repack recovers each group's step and writes the levels as packed int4. It does no new quantization.

The textbook Q4_0 step, `max|w| / 8`, is wrong for any group whose largest weight sits below level 8, and would re-round those groups onto a different grid. [`repack_q4_0.py`](https://github.com/xbill9/gemma4-dev/blob/main/jev-tpu-31b/repack_q4_0.py) instead finds, for each group, the step `max|w| / m` (m from 1 to 8) that reproduces all 32 values. It then refines that step by least squares and stores it at bf16.

## Verification

`repack_q4_0.py verify` rereads both checkpoints and checks every group:

| Tensors | Groups of 32 | Levels off the source grid | Values bit-identical |
|---|---:|---:|---:|
| Experts, gate and up | 475,791,360 | 0 | 92.6% |
| Experts, down | 237,895,680 | 0 | 92.6% |
| Dense MLP | 16,727,040 | 0 | 92.0–92.5% |
| Attention | 34,693,120 | 0 | 89.7–90.4% |

All 748 unquantized tensors are byte-identical to the source. The values that are not bit-identical differ through the bf16 scale (Q4_0 carries a 16-bit float step), by at most 1.1e-2 relative. `repack_report.json` and `verify_report.json` in this repository are that run's outputs.

## Serving it

**NVIDIA GPUs, stock vLLM.** Tested with vLLM 0.30.0 on one NVIDIA L4 (24 GB), no patches. vLLM uses its Marlin int4 kernels for both the experts and the linear layers.

```bash
vllm serve <this-repo> --max-model-len 2048 \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'
```

**Google Cloud TPU.** This needs two changes to vLLM's TPU backend that are not yet merged: [tpu-inference #3653](https://github.com/vllm-project/tpu-inference/pull/3653) (W4A16 linear layers) and [tpu-inference #3660](https://github.com/vllm-project/tpu-inference/pull/3660) (W4A16 experts). With both, it serves on one v6e chip.

## Measured

On one TPU v6e chip (`vllm/vllm-tpu@sha256:19a1a052…` with #3653 and #3660), against `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`, the only other 26B-A4B that fits one chip:

| One v6e chip | HBM used | KV cache | Output tok/s |
|---|---:|---:|---:|
| RedHat FP8 | 27.99 GiB | 3,456 tokens | 668 |
| This checkpoint | 17.43 GiB | 53,888 tokens | 1,283 |

On one NVIDIA L4 with stock vLLM 0.30.0: 14.8 GiB model load, 17,990 KV tokens, 394 output tok/s.

Throughput is 16 concurrent requests of exactly 256 output tokens, median of three passes, with `--max-model-len 2048 --max-num-seqs 16`. Accuracy is measured on Bespoke Labs' 3,880-record public suite, reading each answer by label probability:

| Build | Suite | Difference from bf16 (95% range) |
|---|---:|---|
| bf16 `google/gemma-4-26B-A4B-it`, v6e-4 at tensor parallelism 4 | 76.4% | |
| RedHat FP8, one v6e chip | 76.0% | −0.4 points (−0.9 to +0.2) |
| **This checkpoint, NVIDIA L4, stock vLLM** | 76.0% | −0.4 points (−1.2 to +0.3) |
| **This checkpoint, one v6e chip, #3653 + #3660** | 75.3% | −1.1 points (−1.8 to −0.3) |

On four 300-example classification tasks every build is within noise of bf16. Logs, per-record outputs and scripts are at https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-31b.

## Limitations

- Tested on one NVIDIA L4 at tensor parallelism 1, and on TPU v6e at tensor parallelism 1 and 4 (on a v6e-4 with #3660: 21.75 GiB across four chips, 407,168 KV tokens, 2,073 output tok/s, the same suite accuracy as on one chip). Other GPUs and other TPU generations are untested.
- Text only was tested; the vision tower is present, but image input was not exercised.
- Unofficial. Report problems here, not to Google.

## License and attribution

Gemma 4 is released by Google DeepMind under the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository redistributes Google's weights in a changed container format, under the same license. The weights, training and the original model card are Google DeepMind's; the repack and this card are not affiliated with or endorsed by Google.
