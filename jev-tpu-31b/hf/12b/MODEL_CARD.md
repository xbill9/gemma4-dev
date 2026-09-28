---
library_name: vllm
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model:
- google/gemma-4-12B-it-qat-q4_0-unquantized
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

# Gemma 4 12B-it QAT, compressed-tensors W4A16 (unofficial repack)

**This is an unofficial repack, made and published independently of Google.** It holds Google's quantization-aware-trained (QAT) weights for Gemma 4 12B-it, from [`google/gemma-4-12B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized), repacked into the compressed-tensors W4A16 format that vLLM loads. Google's model card is kept unchanged in `ORIGINAL_README.md`; the model, its intended uses and its limitations are as described there.

## How it differs from `google/gemma-4-12B-it-qat-w4a16-ct`

Google publishes its own W4A16 QAT checkpoint for the 12B. Its 4-bit values are not the grid values of the `-q4_0-unquantized` export: on the 31B, dequantizing Google's `-w4a16-ct` against its `-q4_0-unquantized` gives 6.67% relative error on every projection, while every unquantized tensor is bit-identical. This repack stores the `-q4_0-unquantized` grid values themselves.

On a 3,880-record public suite read by label probability, on one TPU v6e chip, the two 12B sources scored:

| 12B build | Suite |
|---|---:|
| `google/gemma-4-12B-it` bf16 | 76.0% |
| `google/gemma-4-12B-it-qat-q4_0-unquantized` (the values this repack stores, served in bf16) | 75.7% |
| `google/gemma-4-12B-it-qat-w4a16-ct` | 75.1% |

The difference between the last two is −0.7 points (95% range −1.3 to 0.0). This repack itself has not yet been scored.

## What it is

- **Format:** compressed-tensors `pack-quantized`, symmetric int4, group size 32, bf16 scales, activations unquantized (W4A16).
- **Quantized:** every attention and MLP linear layer of the language model (48 layers).
- **Kept bf16, copied byte for byte:** embeddings, norms, and the vision tower.
- **Size:** 7.68 GiB, against 22.28 GiB for the bf16 source.

## How it was made

The `-qat-q4_0-unquantized` export stores bf16 values that already lie on a 4-bit grid: within every group of 32 weights along the input dimension, each weight is `step × level` with level from −8 to 7. The repack recovers each group's step and writes the levels as packed int4. It does no new quantization.

The textbook Q4_0 step, `max|w| / 8`, is wrong for any group whose largest weight sits below level 8, and would re-round those groups onto a different grid. [`repack_q4_0.py`](https://github.com/xbill9/gemma4-dev/blob/main/jev-tpu-31b/repack_q4_0.py) instead finds, for each group, the step `max|w| / m` (m from 1 to 8) that reproduces all 32 values. It then refines that step by least squares and stores it at bf16.

## Verification

`repack_q4_0.py verify` rereads both checkpoints and checks every group:

| Tensors | Groups of 32 | Levels off the source grid | Values bit-identical |
|---|---:|---:|---:|
| MLP gate, up, down | 144 tensors | 0 | 89.5–89.7% |
| Attention q, k, v, o | 184 tensors | 0 | 89.6–89.9% |
| **All** | **340,623,360** | **0** | **89.6%** |

All 349 unquantized tensors are byte-identical to the source. The values that are not bit-identical differ through the bf16 scale (Q4_0 carries a 16-bit float step), by at most 1.09e-2 relative. `repack_report.json` and `verify_report.json` in this repository are that run's outputs.

## Serving it

**Google Cloud TPU.** vLLM's TPU backend loads W4A16 linear layers with [tpu-inference #3653](https://github.com/vllm-project/tpu-inference/pull/3653). The 12B loads on the JAX path as a text-only model with:

```bash
vllm serve <this-repo> --max-model-len 2048 \
  --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}'
```

**NVIDIA GPUs.** Untested with this checkpoint. The 26B repack made the same way loads unpatched in vLLM 0.30.0 with its Marlin int4 kernels.

## Limitations

- Not yet served or scored; the measurements above are of the source checkpoints.
- Text only is the intended path on TPU; the vision tower is present but untested.
- Unofficial. Report problems here, not to Google.

## License and attribution

Gemma 4 is released by Google DeepMind under the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository redistributes Google's weights in a changed container format, under the same license. The weights, training and the original model card are Google DeepMind's; the repack and this card are not affiliated with or endorsed by Google.
