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
- w4a16
- int4
- qat
- vllm
- text-only
---

# Gemma 4 12B-it QAT, compressed-tensors W4A16, text only (unofficial repack)

**This is an unofficial repack, made and published independently of Google.** It is the language model of the repack of [`google/gemma-4-12B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized), [`xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct), with the vision and audio towers removed, loadable as a text-only `Gemma4ForCausalLM`. Google's model card is kept unchanged in `ORIGINAL_README.md`.

It stores Google's quantization-aware-trained weights exactly as QAT left them on the Q4_0 grid. Google's own `-qat-w4a16-ct` export instead re-quantizes them with a min-max round-to-nearest quantizer (scale max|w| / 7.5), which moves every weight onto a shifted grid, and it also stores a bf16 `lm_head` byte-identical to the tied token embedding.

## What it is

- **Dropped:** every tensor under `model.vision_embedder.`, `model.embed_vision.` and `model.embed_audio.` (11 tensors, 0.09 GiB).
- **Kept:** 1,322 language-model tensors, names unchanged (vLLM's `Gemma4ForCausalLM` maps `model.language_model.*` itself).
- **Verified** with [`repack_q4_0.py`](https://github.com/xbill9/gemma4-dev/blob/main/jev-tpu-31b/repack_q4_0.py) `verify` against Google's source: 340,623,360 groups of 32, none off the source grid; every unquantized tensor byte-identical.
- **Config:** the source `text_config` with `architectures: ["Gemma4ForCausalLM"]` and the `quantization_config`, the tower modules removed from its `ignore` list. `processor_config.json` is not included. Made with [`text_only.py`](https://github.com/xbill9/gemma4-dev/blob/main/jev-tpu-31b/text_only.py).
- **Size:** 7.59 GiB, against 7.68 GiB with the towers.

## Measured

Measured on one TPU v5e chip through the same text-only class (`Gemma4ForCausalLM`): suite 75.8% against 75.2% for `google/gemma-4-12B-it-qat-w4a16-ct` on the same chip (+0.6 points, 95% range +0.0 to +1.2) and 76.0% for bf16; 7.59 GiB resident, 33 output tok/s for one request, 388 at 16. Logs and per-record outputs: https://github.com/xbill9/gemma4-dev (`jev-tpu-v5e1/`, `jev-tpu-31b/`).

## Serving it

No `--limit-mm-per-prompt` or `--hf-overrides` flag is needed; there is no multimodal path.

**Google Cloud TPU.** vLLM's TPU backend reads `hf_config.text_config`, which a text-only config does not have; it needs the one-line fallback in [`patches/textonly.diff` and `patches/lowmem.diff`](https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-v5e1/patches), plus tpu-inference #3653 for W4A16. This text-only config has not yet been served on TPU.

**NVIDIA GPUs.** Untested with this checkpoint. The E2B build made the same way (`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`) serves on one T4 with vLLM 0.29.0.

## Limitations

- Text only: image and audio inputs are not supported.
- Unofficial. Report problems here, not to Google.

## License and attribution

Gemma 4 is released by Google DeepMind under the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository redistributes Google's weights in a changed container format, under the same license. The weights, training and the original model card are Google DeepMind's; the repack and this card are not affiliated with or endorsed by Google.
