---
title: "Gemma 4 E2B on an AMD MI300X: Which Weight Format Should You Serve?"
published: false
description: "Ten builds of Gemma 4 E2B (bf16, fp8 in both E4M3 flavours, int8 W8A8 and int4 W4A16, each with and without int4 embedding tables) served one after another on one AMD Instinct MI300X with the same vLLM image. fp8 is the only format that keeps pace with bf16, the 4-bit builds run at 0.14x to 0.63x, and int4 embedding tables cost almost nothing."
tags: gemma, amd, machinelearning, llm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-mi300x-2b/article/devto-mi300x-formats-cover.355c09b0.jpg
---

This article provides a step by step guide to serving ten weight formats of Gemma 4 E2B on one AMD Instinct MI300X through vLLM, with every build timed across a grid of request counts and prompt lengths on the same card, image and day. Every log, report and script is committed.

On the MI300X, fp8 is the only format that keeps pace with bf16: 0.75x for a single request and up to 1.09x with 8 or 64 at once. int8 W8A8 runs at 0.29x to 0.87x and the 4-bit W4A16 builds at 0.14x to 0.63x. The format that serves fastest also rounds the weights furthest from what Google trained, so the choice is a trade between speed and fidelity.

https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b

---

#### Why Compare Formats on a 192 GB Card?

Gemma 4 E2B loads in 9.42 GiB at bf16. One MI300X has 192 GB of HBM, so memory decides nothing here: bf16 leaves room for a KV cache of 9,045,060 tokens, and the smallest build only stretches that to 9,475,223.

What is left to choose on is speed and precision. The card multiplies some number formats natively and emulates others, and every format below stores Google's quantization-aware-trained (QAT) weights with a different amount of rounding.

---

#### The Ten Builds

Every quantized build starts from Google's `gemma-4-E2B-it-qat-q4_0-unquantized` release, whose weights already sit on a 4-bit grid with one scale per group of 32 values:

| Build | Linear layers | Vocabulary tables |
|---|---|---|
| bf16 | bf16 (`google/gemma-4-E2B-it`) | bf16 |
| `fp8` | FP8 E4M3 weights, FP8 activations per token | bf16 |
| `fp8fnuz` | FP8 E4M3FNUZ, the MI300X's own FP8 | bf16 |
| `w8a8` | int8 weights per channel, int8 activations per token | bf16 |
| `q4w4a16` | int4 holding the QAT grid exactly, bf16 activations | bf16 |
| `*emb4` | as above | int4: `embed_tokens`, an untied `lm_head`, per-layer embeddings |
| `q4w4a16ple4` | as `q4w4a16` | int4 per-layer embeddings only |

All of them are on Hugging Face under `xbill9/gemma-4-E2B-it-qat-*`, text only.

---

#### At This Point You Should Have…

- An AMD Developer Cloud account with an MI300X droplet (`gpu-mi300x1-192gb`), and its SSH key
- A Hugging Face token
- A clone: `git clone https://github.com/xbill9/gemma4-dev`
- The droplet prepared with `make scaffold` from [amd-gputools](https://github.com/xbill9/amd-gputools), which installs Docker, adds the GPU groups and pulls the vLLM image

---

#### Step 1 — Time the Card's Matrix Multiplies

Before serving anything, `gemm_decode_shapes.py` times every matrix multiply E2B runs per token, at 1, 8 and 64 rows, from a HIP graph so launch overhead stays out of the number:

```text
| M | dtype | us/token | x bf16 |
| 1 | bf16 | 1966.2 | 1.00 |
| 1 | fp8 | 1304.2 | 1.51 |
| 64 | bf16 | 2383.6 | 1.00 |
| 64 | fp8 | 1589.4 | 1.50 |
| 64 | int8 | 12146.4 | 0.20 |
```

fp8 does the same work in two thirds of bf16's time. int8 is five times slower than bf16 at 64 rows, and below 17 rows PyTorch's int8 multiply refuses to run at all:

```text
control_8192 M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
```

The MI300X has no int4 multiply, so a 4-bit build unpacks its weights to bf16 inside the kernel before each multiply.

---

#### Step 2 — Serve Every Build on One Pinned Image

`dtype_sweep.py` serves each build in turn from its own rig directory, on one image digest, with identical serving settings: `--max-model-len 32768`, `--gpu-memory-utilization 0.90`, prefix caching on, and no `--quantization` flag, so vLLM reads the format from each checkpoint:

```bash
python3 dtype_sweep.py --droplet debian-gpu-mi300x1-192gb-devcloud-atl1 \
  --image vllm/vllm-openai-rocm@sha256:ec62abecc13923172cf1225c0270b8cea7d84b652eacd3c8dffe2dd35dd73e37 \
  --run-id 2026-10-07-dtype-sweep-mi300x
```

Each build is checked at boot for the kernel vLLM chose and the memory it loaded:

```text
=== [3/8] w8a8 (gpu-vllm-mi300x-2b-w8a8)
    ok: loading 7.07 GiB, KV 9025606, kernels ['TritonInt8ScaledMMLinearKernel']

=== [4/8] fp8 (gpu-vllm-mi300x-2b-fp8)
    ok: loading 7.07 GiB, KV 9017531, kernels ['RowWiseTorchFP8ScaledMMLinearKernel']
```

A quantized build that loaded near bf16's 9.42 GiB would have been unpacked to bf16 at load. None was.

---

#### Step 3 — Check Each Build Answers

`verify_capabilities` sends a text question, a thinking prompt and a tool call to each build, with known answers:

```text
✅ 3/3 capabilities verified on `debian-gpu-mi300x1-192gb-devcloud-atl1`.

| text | ✅ | The AMD MI300X GPU is based on the AMD CDNA 3 architecture |
| thinking | ✅ | 1704 chars, 587 reasoning tokens |
| tool calling | ✅ | tool_calls → get_weather{"city": "Reykjavik"} |
```

All nine quantized builds pass all three. bf16 passes the same three and refuses the image check, because it was served text only like the others.

---

#### Step 4 — Sweep Request Count and Prompt Length

Each build runs 1, 8 and 64 requests at once against prompts of 128, 1,024 and 8,192 tokens, 512 output tokens each, three repeats per cell, with a fresh random seed for every run so no prompt is served from the prefix cache.

---

#### How Fast Is Each Format?

Output tokens per second at 1, 8 and 64 parallel requests with 1,024-token prompts, and the range against bf16 across all nine cells:

| Build | Weights | 1 / 8 / 64 requests | Range vs bf16 |
|---|---:|---|---|
| bf16 | 9.42 GiB | 321 / 1,795 / 8,951 | 1.00x |
| `fp8` | 7.07 GiB | 242 / 1,827 / 9,700 | 0.75x – 1.08x |
| `fp8fnuz` | 7.07 GiB | 245 / 1,819 / 9,739 | 0.75x – 1.09x |
| `fp8emb4` | 3.61 GiB | 234 / 1,742 / 9,450 | 0.72x – 1.08x |
| `w8a8` | 7.07 GiB | 98 / 757 / 4,717 | 0.29x – 0.87x |
| `w8a8emb4` | 3.60 GiB | 97 / 746 / 4,680 | 0.29x – 0.79x |
| `q4w4a16` | 6.32 GiB | 47 / 367 / 3,288 | 0.14x – 0.63x |
| `q4w4a16emb4` | 2.85 GiB | 47 / 365 / 3,285 | 0.14x – 0.63x |

fp8 trails bf16 by a quarter on a single request and runs at 1.00x to 1.08x with 8 or 64 requests, outside the noisiest cell. The int8 and 4-bit builds are slower than bf16 in every cell.

For one request the time per output token is 2.89 ms at bf16, 3.83 ms at fp8, 9.94 ms at int8 and 20.91 ms at 4 bits. The 4-bit build loads two thirds of bf16's bytes and takes seven times as long per token, so the unpacking kernel sets its pace. The 4-bit and int8 builds close the gap as prompts grow, which fits more of each step going to the prompt and to attention, where the weight format does not change the work: with 64 requests and 8,192-token prompts `q4w4a16` reaches 0.63x.

---

#### How Close Are They to the Trained Weights?

Each build was checked offline against the QAT values it was made from:

| Build | Error against the QAT weights |
|---|---|
| `q4w4a16` | none on the grid: 58,650,624 groups, 0 off the grid, 90.0% of values bit-identical, the rest within 1.1% through the scale's rounding |
| `w8a8` | 0.59% to 1.71% relative error per tensor, mean 0.90% |
| `fp8` | 2.64% relative RMS error, worst value 3.57% of its row's largest |
| `fp8fnuz` | 2.64% relative RMS error, worst value 3.33% of its row's largest |

The order is the reverse of the speed table. The 4-bit build keeps the trained grid exactly, int8 rounds each value to one of 255 levels per row, and FP8's three mantissa bits round furthest. Accuracy of these same checkpoint files on a classification suite, GSM8K and tool calls is in the TPU v5e article linked below; this sweep measured speed only.

---

#### 🔎 Tip: E4M3 or E4M3FNUZ, Either Serves

The MI300X's FP8 is E4M3FNUZ: the same 3-bit mantissa as the E4M3 most published FP8 checkpoints use, a largest value of 240 against 448, and no negative zero. A raw E4M3 matrix multiply raises `HIPBLAS_STATUS_NOT_SUPPORTED` on this card, yet the E4M3 checkpoint serves, because vLLM's compressed-tensors loader converts it at load:

```text
Selected RowWiseTorchFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8
```

Built natively in E4M3FNUZ, the same weights load into the same kernel, the same 7.07 GiB and the same 9,017,531-token cache, and run at 0.995x to 1.014x of the E4M3 build in eight of nine cells. Either file works on this card.

---

#### 🔎 Tip: int4 Embedding Tables Cost Almost Nothing

The `emb4` builds pack the vocabulary tables, which are a large share of E2B's weights, as int4. Against the same format with bf16 tables, in eight of nine cells:

| Pair | Weights | Output tok/s |
|---|---|---|
| `fp8emb4` / `fp8` | 3.61 / 7.07 GiB | 0.954x – 0.974x |
| `w8a8emb4` / `w8a8` | 3.60 / 7.07 GiB | 0.984x – 0.992x |
| `q4w4a16emb4` / `q4w4a16` | 2.85 / 6.32 GiB | 0.990x – 0.999x |

Half the weight memory for 1% to 5% of speed, and the KV cache grows by about 4%. On this card the memory is not needed; on a smaller one this is the build to reach for.

---

#### Compare and Contrast

| Build | Weights | 1 request | 64 requests, 1,024 tokens | Fidelity to QAT |
|---|---:|---:|---:|---|
| 🥇 `fp8` / `fp8fnuz` | 7.07 GiB | 259 | 9,700 / 9,739 | 2.64% RMS |
| 🥈 `fp8emb4` | 3.61 GiB | 247 | 9,450 | 2.64% RMS |
| 🥉 bf16 | 9.42 GiB | 343 | 8,951 | Google's bf16 release |
| `w8a8` | 7.07 GiB | 100 | 4,717 | 0.59% – 1.71% |
| `q4w4a16` | 6.32 GiB | 48 | 3,288 | exact grid |
| `q4w4a16emb4` | 2.85 GiB | 47 | 3,285 | exact grid |

Single-request figures use 128-token prompts; medals rank output at 64 requests.

---

#### So, Which One?

For a single user, bf16: it is the fastest build at one request, 343 output tokens per second, and memory costs nothing on this card. For many users, `fp8` or `fp8fnuz`, up to 1.09x bf16 with 8 to 64 requests in flight, with `fp8emb4` at half the weights if the card is shared. For the weights closest to what Google trained, `q4w4a16`, at about a seventh of bf16's single-request speed on this card. int8 sits between them on fidelity and below fp8 on speed in every cell but the noisiest, so it is never the best pick here.

---

#### Teardown

Powering a droplet off does not stop billing. Destroy it in the AMD Developer Cloud console when the sweep finishes; the MI300X droplet bills $1.99 an hour until then.

---

#### Summary

The goal of this article was to find the best weight format for Gemma 4 E2B on one AMD MI300X. The key to the solution was serving every build on one pinned vLLM image with identical settings, checking the chosen kernel and loaded memory at boot, and timing all of them on the same grid. The results were:

- 🟢 fp8 is the only format at bf16's pace: 0.75x for one request, up to 1.09x with 8 or 64
- 🟢 E4M3 and E4M3FNUZ checkpoints serve identically, through vLLM's conversion at load
- 🟢 int4 vocabulary tables halve the weights for 1% to 5% of speed
- 🟢 Every quantized build answers text, thinking and tool-call checks
- ⚠️ The 4-bit builds keep the QAT grid exactly but run at 0.14x to 0.63x bf16 through a kernel that unpacks to bf16
- ⚠️ fp8 rounds furthest from the trained weights: 2.64% relative RMS error
- ❌ int8 W8A8 runs at 0.29x to 0.87x bf16, below fp8 in eight of nine cells

Scope: one MI300X droplet in AMD Developer Cloud's atl1 region, vLLM `0.31.1rc1.dev23+g43b4aaea3` at `vllm/vllm-openai-rocm@sha256:ec62abec…`, all ten builds served on 2026-10-07 and 2026-10-08, three repeats per cell. The bf16 build is Google's `gemma-4-E2B-it`, and the nine quantized builds are made from Google's QAT release, so bf16 differs in its weights as well as its format. The cell with 64 requests and 8,192-token prompts varied by up to 15.7% between repeats and is left out of the pairwise ranges; every other cell varied by 4.1% or less, except one `fp8emb4` cell at 10.7%. No accuracy benchmark was run on this card. The repacked checkpoints are unofficial and derived from Google's release under Apache 2.0. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for choosing a weight format for Gemma 4 on an AMD MI300X was validated with an incremental step by step approach.

---

#### References

- Code, logs, reports and this article's evidence: https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b
- The same checkpoints on TPU v5e, with accuracy: https://dev.to/gde/gemma-4-qat-on-one-tpu-v5e-what-runs-and-what-doesnt-2gii
- FP8 build, E4M3: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-fp8-text
- FP8 build, E4M3FNUZ: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-fp8fnuz-text
- 4-bit repack: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text
- Google's QAT source checkpoint: https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized
- AMD Instinct MI300X: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- AMD Developer Cloud: https://devcloud.amd.com
- vLLM on ROCm: https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
