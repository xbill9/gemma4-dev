---
title: "Gemma 4 From E2B to 31B on an AMD MI300X: fp8 Overtakes bf16 From 12B Up"
published: false
description: "Every Gemma 4 size from E2B to 31B served on one AMD Instinct MI300X in bf16, fp8, int8 W8A8 and int4 W4A16, on the same vLLM image. fp8 goes from 0.75x bf16 for one request at E2B to 1.23x at 31B and is faster than bf16 in every cell at 12B and 31B. The mixture-of-experts 26B sits at bf16's pace, and the 4-bit builds run at 0.14x to 0.69x at every size."
tags: gemma, amd, machinelearning, llm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-mi300x-2b/article-sizes/devto-mi300x-sizes-cover.2c57f68d.jpg
---

This article provides a step by step guide to serving every Gemma 4 size, E2B, E4B, 12B, 26B-A4B and 31B, on one AMD Instinct MI300X through vLLM in four weight formats, with each build timed across the same grid of request counts and prompt lengths on the same image. Every log, report and script is committed.

The answer to "which format should I serve?" changes with the size of the model. At E2B, bf16 is fastest for a single user and fp8 only catches it with 8 or more requests in flight. From 12B up, fp8 is faster than bf16 in every cell: 1.12x to 1.41x at 12B and 1.20x to 1.45x at 31B. The 4-bit builds stay at 0.14x to 0.69x of bf16 at every size.

https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b

---

#### Why Sweep Every Size?

The E2B article in this series found fp8 at 0.75x bf16 for one request on this card, and concluded that a small model gains nothing from smaller weights when the card has 192 GB. A small model also reads few bytes per output token. The larger sizes read many more, so they test whether a format that halves the bytes pays off once reading weights becomes the bigger cost.

Memory still decides nothing here. 31B at bf16 loads in 58.99 GiB and leaves a cache of 121,290 tokens.

---

#### The Sizes and the Builds

| Size | bf16 checkpoint | bf16 weights | Formats served |
|---|---|---:|---|
| E2B | `google/gemma-4-E2B-it` | 9.42 GiB | bf16, fp8, int8, int4, and int4-table variants |
| E4B | `google/gemma-4-E4B-it` | 14.79 GiB | bf16, fp8, int8, int4, and int4-table variants |
| 12B | `google/gemma-4-12B-it` | 22.83 GiB | bf16, fp8, int8, int4, and int4-table variants |
| 26B-A4B | `google/gemma-4-26B-A4B-it` | 48.54 GiB | bf16, fp8, int4 |
| 31B | `google/gemma-4-31B-it` | 58.99 GiB | bf16, fp8, int4 |

Every quantized build starts from Google's quantization-aware-trained (QAT) `-qat-q4_0-unquantized` release for its size, text only, and is on Hugging Face under `xbill9/gemma-4-<size>-it-qat-*`:

| Format | Linear layers | Built with |
|---|---|---|
| `fp8` | FP8 E4M3 weights per channel, FP8 activations per token | `fp8_text.py` |
| `w8a8` | int8 weights per channel, int8 activations per token | `w8a8_from_qat.py` |
| `q4w4a16` | int4 holding the QAT grid exactly, bf16 activations | `repack_q4_0.py` |
| `emb4` variants | as above, with int4 vocabulary tables | `fp8_text.py build-on`, `embed_int4.py`, `w8a8_emb4.py` |

---

#### At This Point You Should Have…

- An AMD Developer Cloud account with an MI300X droplet (`gpu-mi300x1-192gb`), and its SSH key
- A Hugging Face token
- A clone: `git clone https://github.com/xbill9/gemma4-dev`
- The droplet prepared with `make scaffold` from [amd-gputools](https://github.com/xbill9/amd-gputools), which installs Docker, adds the GPU groups and pulls the vLLM image

---

#### Step 1 — Build the Larger Repacks

The E2B article has the build commands. Two things change at 26B-A4B, a mixture-of-experts model with 128 experts per layer. Google's QAT release stores each layer's experts as two fused banks, `experts.gate_up_proj` and `experts.down_proj`, and the fp8 and int8 builders split them into one module per expert, `experts.{i}.{gate,up,down}_proj`, the layout vLLM's compressed-tensors MoE methods load. The router, `router.proj`, stays bf16.

```bash
python3 repack/fp8_text.py build src-26b ct-text-26b/config.json \
  ct-text-26b/model.safetensors.index.json gemma-4-26B-A4B-it-qat-q4_0-fp8-text
python3 repack/fp8_text.py verify src-26b gemma-4-26B-A4B-it-qat-q4_0-fp8-text
```

---

#### Step 2 — Serve Each Size on One Pinned Image

`dtype_sweep.py --size` serves each size's builds one after another through the same server code, on the image digest the E2B sweep used, with the same settings: `--max-model-len 32768`, `--gpu-memory-utilization 0.90`, prefix caching on and no `--quantization` flag. 26B-A4B and 31B ran three formats each:

```bash
python3 -u dtype_sweep.py --droplet $DROPLET --image $IMG --run-id $RID --size 31b --only bf16 fp8 q4w4a16
```

Each build is checked at boot for the kernel vLLM chose and the memory it loaded:

```text
=== [1/7] bf16 (gpu-vllm-mi300x-31b)
    ok: loading 58.99 GiB, KV 121290, kernels []
=== [2/7] q4w4a16 (gpu-vllm-mi300x-31b-q4w4a16)
    ok: loading 18.7 GiB, KV 170112, kernels ['TritonW4A16LinearKernel']
=== [4/7] fp8 (gpu-vllm-mi300x-31b-fp8)
    ok: loading 30.63 GiB, KV 155730, kernels ['RowWiseTorchFP8ScaledMMLinearKernel']
```

---

#### Step 3 — Sweep Request Count and Prompt Length

Every build runs 1, 8 and 64 requests at once against prompts of 128, 1,024 and 8,192 tokens, 512 output tokens each, three repeats per cell, with a fresh random seed for every run so no prompt is served from the prefix cache.

---

#### Does fp8 Pay Off as the Model Grows?

fp8 against bf16 on the same card, for one request with a 128-token prompt and across all nine cells:

| Size | One request | Range, 9 cells | Cells where fp8 is faster |
|---|---:|---|---:|
| E2B | 0.75x | 0.75x – 1.08x | 5 of 9 |
| E4B | 0.89x | 0.89x – 1.19x | 6 of 9 |
| 12B | 1.13x | 1.12x – 1.41x | 9 of 9 |
| 26B-A4B | 0.99x | 0.99x – 1.23x | 7 of 9 |
| 31B | 1.23x | 1.20x – 1.45x | 9 of 9 |

Among the dense models the single-request figure rises with size, from 0.75x at E2B to 1.23x at 31B, and crosses bf16 between E4B and 12B. Above that crossing fp8 wins everywhere, with the largest margin at 8 requests: 1.41x at 12B and 1.45x at 31B.

Per GiB of weights loaded, bf16 at 31B produces 1.0 output token per second for one request and fp8 2.3, so halving the bytes more than doubles what each byte yields. At E2B the two are level, 36.5 against 36.6.

---

#### Why the Mixture-of-Experts Model Sits Lower

26B-A4B routes each token through a few of its 128 experts, so it reads far fewer weights per token than its 48.54 GiB suggests. Its bf16 speed for one request, 224 output tokens per second, sits next to E4B's 239 and far above 12B's 134, and fp8 lands where it does for E4B: 0.99x for one request, ahead from 8 requests up.

All three 26B-A4B builds ran on vLLM's default expert-kernel settings, because the image ships no tuned configuration for this expert shape on the MI300X:

```text
[fused_moe.py:1163] Using default MoE config. Performance might be sub-optimal! Config file not found at .../fused_moe/configs/E=128,N=704,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
```

So the 26B-A4B comparison is between three untuned paths, and a tuned configuration could move any of them.

---

#### How Fast Is Each Size?

Output tokens per second at 1 request (128-token prompt), and at 8 and 64 requests (1,024-token prompts):

| Size | bf16 | fp8 | q4w4a16 |
|---|---|---|---|
| E2B | 343 / 1,795 / 8,951 | 259 / 1,827 / 9,700 | 48 / 367 / 3,288 |
| E4B | 239 / 1,284 / 5,778 | 212 / 1,456 / 6,662 | 33 / 254 / 2,183 |
| 12B | 134 / 677 / 2,248 | 152 / 911 / 2,767 | 22 / 159 / 1,109 |
| 26B-A4B | 224 / 979 / 2,585 | 222 / 1,132 / 3,162 | 60 / 382 / 1,772 |
| 31B | 58 / 306 / 950 | 71 / 432 / 1,219 | 11 / 80 / 512 |

---

#### The 4-Bit Builds Stay Slow at Every Size

The 4-bit builds keep the QAT grid exactly and load in about a third of bf16's memory, 18.70 GiB at 31B against 58.99, which grows the KV cache by 1.40x there. They run at 0.14x to 0.63x of bf16 at the four dense sizes, and 0.27x to 0.69x at 26B-A4B. The dense sizes use `TritonW4A16LinearKernel`, which unpacks the weights to bf16 before each multiply, and 26B-A4B's experts use vLLM's Triton 4-bit expert path:

```text
Using CompressedTensorsWNA16MoEMethod
Using 'TRITON' WNA16 MoE backend.
```

The gap closes with more requests and with longer prompts at every size. Averaged over prompt lengths, 31B's 4-bit build runs at 0.20x bf16 for one request and 0.53x for 64.

int8 W8A8 sits between the two at every size it ran: 0.29x to 0.87x at E2B, 0.31x to 0.82x at E4B and 0.39x to 0.93x at 12B, below fp8 in every cell but one, E2B's noisiest.

---

#### 🔎 Tip: int4 Embedding Tables Cost More at Larger Sizes

The `emb4` builds pack the vocabulary tables as int4. Against the same format with bf16 tables, median over nine cells:

| Pair | E2B | E4B | 12B |
|---|---:|---:|---:|
| `fp8emb4` / `fp8` | 0.968x | 0.935x | 0.948x |
| `q4w4a16emb4` / `q4w4a16` | 0.996x | 0.986x | 0.987x |

They save 4.31 GiB at E4B and 0.82 GiB at 12B for a few percent of speed. On this card that memory is not needed.

---

#### 🔎 Tip: Two int8 Builds Do Not Load on This Image

The E4B and 12B `w8a8emb4` builds, made by `w8a8_emb4.py` for TPU serving, stop at model load on the stock ROCm image:

```text
AttributeError: 'VocabParallelEmbedding' object has no attribute 'weight'
```

The E2B `w8a8emb4` build, made by a different script for the NVIDIA T4, loads and serves. The `w8a8_emb4.py` builds, as published, do not serve on this image.

---

#### Compare and Contrast

| Size | 🥇 Fastest for one user | 🥇 Fastest for many | Smallest that serves |
|---|---|---|---|
| E2B | bf16, 343 tok/s | `fp8`, up to 1.08x | `q4w4a16emb4`, 2.85 GiB |
| E4B | bf16, 239 tok/s | `fp8`, up to 1.19x | `q4w4a16emb4`, 4.51 GiB |
| 12B | `fp8`, 152 tok/s | `fp8`, up to 1.41x | `q4w4a16emb4`, 7.36 GiB |
| 26B-A4B | bf16 224 / `fp8` 222 tok/s | `fp8`, up to 1.23x | `q4w4a16`, 14.80 GiB |
| 31B | `fp8`, 71 tok/s | `fp8`, up to 1.45x | `q4w4a16`, 18.70 GiB |

---

#### So, Which One?

For 12B and 31B, serve fp8: it is faster than bf16 in every cell and halves the weights. For E2B and E4B, bf16 for a single user and fp8 when many requests share the card. For 26B-A4B, fp8, which matches bf16 for one request and leads from 8 up. The 4-bit builds belong on cards where memory runs out first; on an MI300X they cost a factor of two to seven in speed at every size.

---

#### Teardown

Powering a droplet off does not stop billing. Destroy it in the AMD Developer Cloud console when the sweep finishes; the MI300X droplet bills $1.99 an hour until then.

---

#### Summary

The goal of this article was to find how the best weight format for Gemma 4 on an AMD MI300X changes with model size. The key to the solution was serving every size and format through one server and one pinned vLLM image, checking the kernel and loaded memory at boot, and timing every build on the same grid. The results were:

- 🟢 fp8 overtakes bf16 between E4B and 12B, and is faster in every cell at 12B (1.12x to 1.41x) and 31B (1.20x to 1.45x)
- 🟢 At 31B fp8 halves the weights to 30.63 GiB and serves 71 output tokens per second for one request against bf16's 58
- 🟢 26B-A4B behaves like a small model: fp8 at 0.99x for one request, up to 1.23x with more
- ⚠️ int4 tables cost a median of 0.4% to 6.5% of speed and save memory this card does not need
- ⚠️ 26B-A4B ran on vLLM's untuned default expert-kernel settings in every format
- ❌ The 4-bit builds run at 0.14x to 0.69x of bf16 at every size
- ❌ The E4B and 12B `w8a8emb4` builds do not load on the stock ROCm image

Scope: one MI300X droplet in AMD Developer Cloud's atl1 region, vLLM `0.31.1rc1.dev23+g43b4aaea3` at `vllm/vllm-openai-rocm@sha256:ec62abec…`, served between 2026-10-07 and 2026-10-09 with three repeats per cell. 26B-A4B and 31B ran bf16, fp8 and int4 only. Each bf16 build is Google's original release and each quantized build is made from Google's QAT release, so bf16 differs in its weights as well as its format. Repeat spread stayed at 5.7% or less in every cell above E2B; at E2B one cell with 64 requests and 8,192-token prompts varied by up to 15.7%. No accuracy benchmark was run on this card. The repacked checkpoints are unofficial and derived from Google's release under Apache 2.0. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for choosing a weight format for every Gemma 4 size on an AMD MI300X was validated with an incremental step by step approach.

---

#### References

- Code, logs, reports and this article's evidence: https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b
- The E2B article, with every E2B build and the build commands: https://dev.to/gde/gemma-4-e2b-on-an-amd-mi300x-which-weight-format-should-you-serve-1a01
- 31B fp8 build: https://huggingface.co/xbill9/gemma-4-31B-it-qat-q4_0-fp8-text
- 12B fp8 build: https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-fp8-text
- 26B-A4B fp8 build: https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-fp8-text
- Google's QAT source checkpoints: https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized
- AMD Instinct MI300X: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- AMD Developer Cloud: https://devcloud.amd.com
