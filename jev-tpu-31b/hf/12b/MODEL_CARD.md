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

Google publishes its own W4A16 QAT checkpoint for the 12B. Its 4-bit values are not the grid values of the `-q4_0-unquantized` export. Its `recipe.yaml` shows why: it was made by re-quantizing the QAT weights with llm-compressor's round-to-nearest `QuantizationModifier` and a `memoryless_minmax` observer, which sets every group's scale to max|w| / 7.5. The QAT grid's step is max|w| / m for the level m that group's largest weight sits on, so the two steps never coincide, and every weight is rounded a second time onto a shifted grid. Measured on this 12B: Google's scales are exactly max|w| / 7.5 in every group, equal to the QAT step in none, and its values differ from the QAT values by 6.67% relative error (the same as on the 31B). This repack recovers each group's QAT step and stores the `-q4_0-unquantized` grid values themselves.

On a 3,880-record public suite read by label probability, paired record for record:

| 12B build | Where | Suite | This repack on one v5e chip, difference (95% range) |
|---|---|---:|---|
| `google/gemma-4-12B-it` bf16 | one v6e chip | 76.0% | −0.1 points (−0.7 to +0.5) |
| `google/gemma-4-12B-it-qat-q4_0-unquantized` (this repack's source, served in bf16) | one v6e chip | 75.7% | +0.1 (−0.2 to +0.4) |
| `google/gemma-4-12B-it-qat-w4a16-ct` | one v6e chip | 75.1% | +0.7 (+0.1 to +1.3) |
| `google/gemma-4-12B-it-qat-w4a16-ct` | **the same v5e chip, same flags** | 75.2% | **+0.6 (+0.0 to +1.2)** |
| **This repack** | one v5e chip | **75.8%** | |

It scores level with bf16 and its bf16 source, and above Google's W4A16 export on the same chip.

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

## Measured on one TPU v5e chip

One `v5litepod-1` (15.75 GiB HBM), vLLM TPU `vllm/vllm-tpu@sha256:19a1a052…` with the W4A16 patches below:

| | This repack | `google/gemma-4-12B-it-qat-w4a16-ct` |
|---|---:|---:|
| Weights resident | 7.59 GiB | 9.46 GiB |
| First-token latency, one request, median | 62.9 ms | 63.1 ms |
| Output tokens/s at 1 / 4 / 16 concurrent requests (256 tokens each) | 33.0 / 127.4 / 387.9 | not measured |

On v5e it needs gmm_v2 tiles chosen with VMEM headroom, fewer compile buckets and a capped KV cache; the settings, logs and per-record outputs are at https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-v5e1 (section "12B QAT W4A16 repack on one v5e chip").

## Serving it

**Google Cloud TPU.** vLLM's TPU backend loads W4A16 linear layers with [tpu-inference #3653](https://github.com/vllm-project/tpu-inference/pull/3653). The 12B loads on the JAX path as a text-only model with:

```bash
vllm serve <this-repo> --max-model-len 2048 \
  --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}'
```

**NVIDIA GPUs.** Untested with this checkpoint. The 26B repack made the same way loads unpatched in vLLM 0.30.0 with its Marlin int4 kernels.

## Limitations

- Measured on TPU v5e and v6e only, at tensor parallelism 1; NVIDIA GPUs untested with this checkpoint.
- Text only is the intended path on TPU; the vision tower is present but untested.
- Unofficial. Report problems here, not to Google.

## License and attribution

Gemma 4 is released by Google DeepMind under the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository redistributes Google's weights in a changed container format, under the same license. The weights, training and the original model card are Google DeepMind's; the repack and this card are not affiliated with or endorsed by Google.
