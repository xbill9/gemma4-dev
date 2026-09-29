---
library_name: vllm
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model:
- google/gemma-4-E2B-it-qat-q4_0-unquantized
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

# Gemma 4 E2B-it QAT, compressed-tensors W4A16 (unofficial repack)

**This is an unofficial repack, made and published independently of Google.** It holds Google's quantization-aware-trained (QAT) weights for Gemma 4 E2B-it, from [`google/gemma-4-E2B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized), repacked into the compressed-tensors W4A16 format that vLLM loads. Google's model card is kept unchanged in `ORIGINAL_README.md`; the model, its intended uses and its limitations are as described there.

## How it differs from `google/gemma-4-E2B-it-qat-w4a16-ct`

Google publishes its own W4A16 QAT checkpoint for E2B. Two differences:

- **Values.** This repack stores the Q4_0 grid values of the `-q4_0-unquantized` export exactly. Google's W4A16 export records llm-compressor's `memoryless_minmax` observer in its `quantization_config`: the QAT weights were re-quantized by round-to-nearest with each group's scale at max|w| / 7.5, which moves every weight to a shifted grid (measured at 6.67% relative error on the 12B and 31B exports; not measured tensor by tensor on E2B).
- **Size.** Google's E2B export also stores `lm_head.weight` in bf16 (0.75 GiB), a byte-for-byte copy of the token embedding although the config ties them (`tie_word_embeddings: true`). This repack does not, so it is 0.75 GiB smaller on disk and on the chip.

## Measured

On a 3,880-record public suite read by label probability, paired record for record:

| E2B build | Where | Suite | This repack, difference (95% range) |
|---|---|---:|---|
| `google/gemma-4-E2B-it` bf16 | the same v5e chip | 68.5% | −0.6 points (−1.6 to +0.3) |
| `google/gemma-4-E2B-it-qat-w4a16-ct` | the same v5e chip, same flags | 65.5% | **+2.4 points (+1.4 to +3.4)** |
| **This repack** | one v5e chip | **67.8%** | |

On one TPU v5e chip (`v5litepod-1`, 15.75 GiB HBM), vLLM TPU `vllm/vllm-tpu@sha256:19a1a052…` with the patches below:

| | This repack | `google/gemma-4-E2B-it-qat-w4a16-ct` |
|---|---:|---:|
| Weights resident | 6.42 GiB | 7.17 GiB |
| First-token latency, one request, median | 16.2 ms | 16.0 ms |
| Output tokens/s at 1 / 4 / 16 concurrent requests (256 tokens each) | 136.3 / 531.6 / 1,906.2 | not measured |

Logs, per-record outputs and scripts: https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-v5e1

## What it is

- **Format:** compressed-tensors `pack-quantized`, symmetric int4, group size 32, bf16 scales, activations unquantized (W4A16).
- **Quantized:** every attention and MLP linear of the language model (35 layers), and the 71 per-layer-embedding (PLE) projections (`per_layer_input_gate`, `per_layer_projection`, `per_layer_model_projection`), which QAT also puts on the grid. This is the same set Google's export quantizes.
- **Kept bf16, copied byte for byte:** the token embedding (tied to the LM head), the per-layer embedding table (4.375 GiB, the largest part of the file), norms, and the vision and audio towers.
- **Size:** 7.00 GiB, against 9.51 GiB for the bf16 source.

## How it was made

The `-qat-q4_0-unquantized` export stores bf16 values that already lie on a 4-bit grid: within every group of 32 weights along the input dimension, each weight is `step × level` with level from −8 to 7. The repack recovers each group's step and writes the levels as packed int4. It does no new quantization. [`repack_q4_0.py`](https://github.com/xbill9/gemma4-dev/blob/main/jev-tpu-31b/repack_q4_0.py) finds, for each group, the step `max|w| / m` (m from 1 to 8) that reproduces all 32 values, refines it by least squares and stores it at bf16.

## Verification

`repack_q4_0.py verify` rereads both checkpoints and checks every group: **58,650,624 groups of 32, none off the source grid, 90.0% of values bit-identical**; the rest differ through the bf16 scale, by at most 1.09e-2 relative. All 1,675 unquantized tensors are byte-identical to the source, and every linear left bf16 is on the `ignore` list. `repack_report.json` and `verify_report.json` in this repository are that run's outputs.

## Serving it

**Google Cloud TPU.** vLLM's TPU backend needs [tpu-inference #3653](https://github.com/vllm-project/tpu-inference/pull/3653) (W4A16 linear layers) and [#3299](https://github.com/vllm-project/tpu-inference/pull/3299) (the KV-shared layers of the E models own no K/V parameters). On one v5e chip, leave HBM for the compiled model:

```bash
vllm serve <this-repo> --max-model-len 2048 --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'
```

**NVIDIA GPUs.** Untested with this checkpoint.

## Limitations

- Measured on TPU v5e at tensor parallelism 1; other hardware untested.
- Text only was tested; the vision and audio towers are present but not exercised.
- Unofficial. Report problems here, not to Google.

## License and attribution

Gemma 4 is released by Google DeepMind under the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository redistributes Google's weights in a changed container format, under the same license. The weights, training and the original model card are Google DeepMind's; the repack and this card are not affiliated with or endorsed by Google.
