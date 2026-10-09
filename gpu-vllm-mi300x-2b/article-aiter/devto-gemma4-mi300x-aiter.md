---
title: "VLLM_ROCM_USE_AITER=1 Slows Gemma 4 on an AMD MI300X: What the Flag Changes and Why"
published: false
description: "AMD's AITER kernel library is the usual first switch for vLLM speed on an Instinct MI300X. On Gemma 4 12B fp8 it made every one of nine cells 0.6% to 2.6% slower. The boot logs show why: Gemma 4's 512-wide attention heads keep attention on Triton, and AITER's fp8 matrix multiply has no tuned settings for any of the model's six weight shapes."
tags: gemma, amd, machinelearning, llm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-mi300x-2b/article-aiter/devto-mi300x-aiter-cover.923b77d8.jpg
---

This article provides a step by step guide to turning on AMD's AITER kernels for vLLM on one AMD Instinct MI300X, serving Gemma 4 12B in fp8, and reading from the server's own boot log what the switch replaced. Every log, report and script is committed.

`VLLM_ROCM_USE_AITER=1` made Gemma 4 12B slower in every cell measured, by 0.6% to 2.6%, against the same build served the same day on the same droplet and image. The boot log explains it. Attention, where AITER usually helps most, never leaves Triton, because Gemma 4's full-attention layers use 512-wide heads. The fp8 matrix multiplies do move to AITER, which has no tuned settings for any of the model's six weight shapes and runs them on defaults.

https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b

---

#### What Is AITER?

AITER is AMD's library of tuned GPU kernels for Instinct cards, and vLLM on ROCm uses it when the server starts with `VLLM_ROCM_USE_AITER=1`. It covers attention, matrix multiplies and normalization, and it picks its matrix-multiply settings from tables tuned per matrix shape.

The earlier articles in this series served every Gemma 4 size on one MI300X and found fp8 the fastest format from 12B up. This one asks whether AITER adds to that.

---

#### At This Point You Should Have…

- An AMD Developer Cloud account with an MI300X droplet (`gpu-mi300x1-192gb`), and its SSH key
- A clone: `git clone https://github.com/xbill9/gemma4-dev`
- The droplet prepared with `make scaffold` from [amd-gputools](https://github.com/xbill9/amd-gputools), which installs Docker, adds the GPU groups and pulls the vLLM image

---

#### Step 1 — Pass the Flag Into the Server Container

AITER is switched on by an environment variable inside the vLLM container, so the server needs a way to pass one through. The rig's server takes `VLLM_DOCKER_ENV`, a comma-separated list it turns into `docker run -e` arguments, and the sweep driver takes the same list as `--engine-env`:

```bash
python3 -u dtype_sweep.py --droplet $DROPLET --image $IMG --run-id 2026-10-09-aiter-mi300x \
  --size 12b --only fp8 --engine-env VLLM_ROCM_USE_AITER=1
```

---

#### Step 2 — Run AITER and Stock Back to Back

Both runs serve `xbill9/gemma-4-12B-it-qat-q4_0-fp8-text` on the same droplet, one after the other, on the image digest the earlier sweep used, with the same settings: `--max-model-len 32768`, `--gpu-memory-utilization 0.90`, prefix caching on. Each times 1, 8 and 64 requests against 128, 1,024 and 8,192-token prompts with 512 output tokens, three repeats per cell, with a fresh seed for every run so no prompt comes from the prefix cache.

```text
2026-10-09T14:51:53Z start aiter fp8
2026-10-09T15:13:26Z done aiter fp8 exit 0; start stock fp8
2026-10-09T15:34:11Z done stock fp8 exit 0
```

---

#### Step 3 — Read What the Flag Changed

Whether a flag changed the work shows in the boot log. The two, side by side:

| Part of the model | Stock | `VLLM_ROCM_USE_AITER=1` |
|---|---|---|
| Attention | `TRITON_ATTN` | `TRITON_ATTN` |
| fp8 linear layers | `RowWiseTorchFP8ScaledMMLinearKernel` | the same, calling AITER's fp8 GEMM on default settings |
| RMSNorm | native | AITER |
| Weights loaded | 12.65 GiB | 12.65 GiB |

```text
Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['aiter', 'native'], ...)
Selected RowWiseTorchFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8
```

---

#### How Fast Is It?

Output tokens per second, same droplet, same day, same image digest:

| Requests | Prompt tokens | AITER | Stock | AITER / stock |
|---:|---:|---:|---:|---:|
| 1 | 128 | 148.3 | 152.2 | 0.974 |
| 1 | 8,192 | 100.6 | 102.7 | 0.980 |
| 8 | 1,024 | 896.5 | 917.1 | 0.978 |
| 64 | 128 | 4,780.2 | 4,874.2 | 0.981 |
| 64 | 1,024 | 2,694.7 | 2,731.0 | 0.987 |
| 64 | 8,192 | 849.4 | 857.7 | 0.990 |

All nine cells are slower with AITER, from 0.974x to 0.994x, median 0.981x. Repeats varied by 0.78% or less, so the gap is bigger than the run-to-run noise in every cell. With 64 requests and 1,024-token prompts the time per output token rises from 20.24 ms to 21.21 ms.

---

#### Why Attention Stays on Triton

vLLM picks the attention backend from the model's head sizes, and for Gemma 4 it has one choice:

```text
Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN backend.
```

Gemma 4 alternates sliding-window layers with 256-wide heads and full-attention layers with 512-wide heads. One backend has to serve both, and Triton is the one that does, so attention runs the same code with the flag on or off.

---

#### Why the fp8 Multiplies Run Slower

With the flag on, vLLM's fp8 linear layer hands its matrix multiply to AITER, which looks up tuned settings for each weight shape and token count. It found none for Gemma 4 12B:

```text
[aiter] shape is M:1, N:8192, K:3840, q_dtype_w:torch.float8_e4m3fnuz, not found tuned config in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv, will use default config!
```

The log carries 924 of these: the model's six distinct weight shapes, each at 77 token counts from 1 to 262,144, in each of AITER's two fp8 tuning tables. Every fp8 multiply in the model ran on AITER's default settings, and the end-to-end numbers fit those defaults being slightly slower on these shapes than PyTorch's own fp8 path.

| Weight shape (N x K) | Untuned lookups |
|---|---:|
| 8192 x 3840 | 154 |
| 9216 x 3840 | 154 |
| 30720 x 3840 | 154 |
| 3840 x 4096 | 154 |
| 3840 x 8192 | 154 |
| 3840 x 15360 | 154 |

---

#### 🔎 Tip: The Stock Run Reproduces Across Droplets

The stock run here lands within 2.1% of the same build's numbers from the earlier sweep, two days before on a different droplet, in every cell (0.979x to 1.007x). Results on this platform hold from one droplet to the next on a pinned image digest, which is what makes a 1% to 3% difference readable at all.

---

#### 🔎 Tip: Check What a Flag Changed Before Timing It

`VLLM_ROCM_USE_AITER=1` was accepted without a warning, and the server reported the same linear kernel name with it as without it. The difference shows only in the lines around it: the operator priority list, the attention backend decision, and AITER's own tuning lookups. Diff the two boot logs before reading any throughput number.

---

#### Compare and Contrast

| | Stock | `VLLM_ROCM_USE_AITER=1` |
|---|---|---|
| Output, 1 request | 🥇 152.2 tok/s | 148.3 tok/s |
| Output, 64 requests, 1,024 tokens | 🥇 2,731.0 tok/s | 2,694.7 tok/s |
| Attention | Triton | Triton |
| fp8 matrix multiply | PyTorch row-wise fp8 | AITER, default settings |
| Weights, KV cache | 12.65 GiB, 470,708 tokens | 12.65 GiB, 471,183 tokens |

---

#### So, Which One?

For Gemma 4 on this image, leave `VLLM_ROCM_USE_AITER` off. AITER can only help here once two things change: tuned fp8 settings for Gemma 4's six weight shapes, generated with the tuning scripts AITER ships, and an attention path that serves 512-wide heads. Until then the flag swaps a tuned multiply for an untuned one and leaves the rest alone.

---

#### Teardown

Powering a droplet off does not stop billing. Destroy it in the AMD Developer Cloud console; the MI300X droplet bills $1.99 an hour until then.

---

#### Summary

The goal of this article was to find whether AMD's AITER kernels speed up Gemma 4 on an AMD MI300X. The key to the solution was running the same fp8 build with and without `VLLM_ROCM_USE_AITER=1` back to back on one droplet and one image digest, and diffing the two boot logs to see what the flag replaced. The results were:

- ❌ AITER made Gemma 4 12B fp8 slower in every one of nine cells, by 0.6% to 2.6%
- ⚠️ Attention stays on Triton either way: Gemma 4's 512-wide full-attention heads force it
- ⚠️ The fp8 multiplies move to AITER, which has no tuned settings for any of the model's six weight shapes
- 🟢 RMSNorm moves to AITER
- 🟢 The stock numbers reproduce within 2.1% across droplets two days apart

Scope: one MI300X droplet in AMD Developer Cloud's atl1 region, vLLM `0.31.1rc1.dev23+g43b4aaea3` at `vllm/vllm-openai-rocm@sha256:ec62abec…`, Gemma 4 12B in fp8 only, both runs on 2026-10-09 with three repeats per cell. bf16, the other sizes and a tuned AITER configuration were not run. The repacked checkpoint is unofficial and derived from Google's release under Apache 2.0. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for testing AITER with Gemma 4 on an AMD MI300X was validated with an incremental step by step approach.

---

#### References

- Code, logs, reports and this article's evidence: https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b
- Gemma 4 from E2B to 31B on an AMD MI300X: https://dev.to/gde/gemma-4-from-e2b-to-31b-on-an-amd-mi300x-fp8-overtakes-bf16-from-12b-up-h4e
- 12B fp8 build: https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-fp8-text
- AITER: https://github.com/ROCm/aiter
- AMD Instinct MI300X: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
