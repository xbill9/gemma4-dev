---
title: "Six Serving Engines on Four Accelerators: Gemma 4 E2B on an RTX 3050, a T4, an L4 and a TPU v5e"
published: false
description: "The same Gemma 4 E2B, prompts, load sweep and GSM8K and BFCL questions through the same serving engines on an 8 GB RTX 3050, a Tesla T4, an NVIDIA L4 and one TPU v5e chip. Answers hold on every chip; vLLM with its drafter leads on all three GPUs, and SGLang-JAX leads on the TPU."
tags: gemma, vllm, nvidia, tpu
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/serving-engines-rollup/devto-cover.083b3fac.jpg
---

This article provides a step by step comparison of the same inference serving engines on four accelerators: an 8 GB RTX 3050 in a desktop, a Tesla T4 on Compute Engine, an NVIDIA L4 in an EC2 G6, and one Cloud TPU v5e chip. Each one served Gemma 4 E2B on Google's QAT 4-bit weights, and each has its own article; this one puts the four side by side.

https://github.com/xbill9/gemma4-dev

The question is whether the engine to run changes with the chip. Each platform got the same prompts, the same 1-to-64-request load sweep and the same 1,319 GSM8K and 400 BFCL questions, with Google's E2B drafter wherever the engine supports one. Every cross-platform figure below is read from the four ladders' own result files and computed by `rollup.py`.

---

#### The Results in Brief

Answers do not depend on the chip. Five CUDA engines on three GPUs give 15 GSM8K scores between 0.8954 and 0.9030 and BFCL scores between 0.9150 and 0.9250, a spread of 4 questions. On the TPU the int4 builds score 0.8969 to 0.9007 and the bf16 engines 0.9113 to 0.9143. LiteRT-LM on Google's int2/4/8 file is the lowest on every GPU.

For one request without a drafter, the chip sets the speed and the engines move together: on each GPU the five CUDA engines are within 1.07 to 1.16x of each other. The drafter separates them, and vLLM gains the most on every GPU: 1.90x on the 3050, 1.78x on the T4, 1.96x on the L4. vLLM with its drafter is the fastest engine for one request on all three GPUs. On the TPU the drafter does not fit beside vLLM, and SGLang-JAX is fastest at 159.9 tokens per second.

At 64 requests vLLM leads on the T4 and the L4 at both shapes, and its drafter still pays there. On the 3050 SGLang leads vLLM by 3% on long replies and vLLM leads on long prompts. On the TPU, SGLang-JAX with an E2B patch reaches 2.52x vLLM on long replies.

---

#### What Changes and What Stays

| Platform | Chip | Memory | Host | Engines |
| :--- | :--- | :--- | :--- | :--- |
| 3050 | RTX 3050 OEM, Ampere | 8,192 MiB | i7-5820K desktop | llama.cpp, vLLM, SGLang, mistral.rs, PyTorch, LiteRT-LM |
| T4 | Tesla T4, Turing | 15,360 MiB | GCE `n1-standard-2` | the same six |
| L4 | NVIDIA L4, Ada | 23,034 MiB | EC2 g6.4xlarge | the same six |
| v5e-1 | TPU v5e, one chip | 15.75 GiB | flex-start `v5litepod-1` | vLLM, SGLang-JAX, a JAX engine, a Rust model over PJRT, TorchTPU |

The same on every platform: Gemma 4 E2B, greedy decoding, thinking off for speed and accuracy, one engine at a time on the chip, and one Python interpreter with no virtual environment. On the three GPUs every engine version and checkpoint revision is the same: llama.cpp `7182ef71e`, vLLM 0.33.0, SGLang 0.5.23, mistral.rs v0.9.5, LiteRT-LM 0.19.0, and the exact Q4_0 GGUF or the W4A16 checkpoint carrying the same 4-bit values. The TPU runs vLLM from its pinned TPU image on the same W4A16 checkpoint, and the JAX engines and SGLang-JAX on bf16 `google/gemma-4-E2B-it`.

llama.cpp and mistral.rs have no TPU backend.

---

#### At This Point You Should Have…

- The four ladder articles, one per platform (References)
- A clone: `git clone https://github.com/xbill9/gemma4-dev`
- `docs/serving-engines-rollup/rollup.py`, which reads each ladder's result files from `evidence/` and prints every table below

---

#### Step 1 — One Method on Four Platforms

Each ladder ran the same three jobs per engine against one server start:

- **One request:** six chat prompts of 256 tokens, three repeats, the geometric mean of each prompt's median, end to end.
- **Load:** `sweep.py` at 1 to 64 requests, two shapes (512 tokens in / 128 out and 128 in / 512 out), five repeats, a unique prompt per request so no prefix cache can hit.
- **Accuracy:** `gen_eval.py` with all 1,319 GSM8K test questions and the 400 BFCL simple function-calling questions, one request at a time.

```text
cd docs/serving-engines-rollup && python3 rollup.py
ONE REQUEST, tok/s end to end (six prompts, 256 tokens, geometric mean of medians)
engine                            3050                    T4                    L4                 v5e-1
vLLM                 121.1 / 230.5 (4)     108.4 / 193.0 (3)     140.4 / 275.3 (4)                 143.9
```

Each engine got the memory settings its chip allows, so the setting differs where the card does: mistral.rs holds 4 requests on the 3050 and T4 rigs and 32 on the L4.

---

#### Step 2 — Answer Quality on Every Chip

GSM8K / BFCL, one request at a time, no drafter:

| Engine | 3050 | T4 | L4 |
| :--- | :--- | :--- | :--- |
| vLLM | 0.9014 / 0.9250 | 0.8999 / 0.9225 | 0.8969 / 0.9225 |
| SGLang | 0.8992 / 0.9150 | 0.8954 / 0.9175 | 0.8954 / 0.9150 |
| llama.cpp | 0.9014 / 0.9225 | 0.9030 / 0.9250 | 0.8984 / 0.9250 |
| mistral.rs | 0.9030 / 0.9150 | 0.8999 / 0.9175 | 0.9014 / 0.9200 |
| PyTorch | 0.8992 / 0.9225 | 0.8961 / 0.9225 | 0.8961 / 0.9225 |
| LiteRT-LM, Google's file | 0.8795 | 0.8734 | 0.8908 |

The 15 CUDA GSM8K scores span 0.0076, and each ladder found its engines level with each other question by question. The drafters change no score on any GPU.

On the TPU:

| Engine | Weights | GSM8K |
| :--- | :--- | :--- |
| SGLang-JAX | bf16 | 🥇 0.9143 |
| JAX engine | bf16 | 0.9128 |
| Rust model over PJRT | bf16 | 0.9113 |
| vLLM | W4A16, int4 tables | 0.9007 |
| vLLM | W4A16, bf16 tables | 0.8992 |
| JAX engine | W4A16 | 0.8969 |

The int4 builds score with the GPUs. The bf16 engines run about one point higher, and in the v5e ladder only SGLang-JAX's lead over vLLM had a paired range above zero. vLLM is the only TPU engine that returns tool calls (BFCL 0.9200).

🔎 **Tip: LiteRT-LM's own file costs answers.** Google's `gemma-4-E2B-it.litertlm` mixes int2, int4 and int8 weights, and it scores 0.0046 to 0.0220 below the lowest CUDA engine on the same card. LiteRT-LM serving the exact Q4_0 build scores 0.8999 on the L4, level with the others.

---

#### Step 3 — One Request Without a Drafter

Tokens per second, end to end:

| Engine | 3050 | T4 | L4 | v5e-1 |
| :--- | :--- | :--- | :--- | :--- |
| vLLM | 🥇 121.1 | 108.4 | 140.4 | 143.9 |
| SGLang | 119.9 | 97.8 | 136.5 | 🥇 159.9 (SGLang-JAX) |
| llama.cpp | 108.8 | 104.8 | 135.7 | — |
| mistral.rs | 118.9 | 🥇 113.5 | 🥇 144.6 | — |
| PyTorch | 118.9 | 109.8 | 134.7 | 6.6 (TorchTPU) |
| LiteRT-LM | 62.0 | 44.6 | 98.4 | — |
| JAX engine, bf16 | — | — | — | 134.3 |
| Rust model over PJRT | — | — | — | 145.2 |

On each GPU the five CUDA engines sit close together: fastest over slowest is 1.113 on the 3050, 1.161 on the T4 and 1.073 on the L4. The chip moves them all at once. Against the L4, the same engine runs at 0.80 to 0.88x on the 3050 and 0.72 to 0.82x on the T4. vLLM on one v5e chip runs at 1.025x vLLM on the L4.

LiteRT-LM runs at 0.426 to 0.725x llama.cpp on the same card.

---

#### Step 4 — The Drafter for One Request

Best drafter against no drafter, same engine and card:

| Engine | 3050 | T4 | L4 |
| :--- | :--- | :--- | :--- |
| vLLM | 🥇 1.90x (4 tokens) | 🥇 1.78x (3 tokens) | 🥇 1.96x (4 tokens) |
| llama.cpp | 1.35x (2 tokens) | 1.18x (1 token) | 1.53x (2 tokens) |
| SGLang | 1.33x (4 tokens) | 0.72x (4 tokens) | 1.39x (4 tokens) |
| LiteRT-LM | 1.10x (its own) | 0.98x (its own) | 1.15x (its own) |
| mistral.rs | 0.72x (1 token) | fails | 0.95x (1 token) |

So the fastest engine for one request is vLLM with its drafter on every GPU: 230.5 tokens per second on the 3050, 193.0 on the T4 and 275.3 on the L4. Every gain is smallest on the T4, the oldest card. On the TPU vLLM with Google's drafter runs out of memory during warmup:

```text
Error allocating device buffer: Attempting to allocate 512.00M. That was not possible. There are 450.17M free.; (0x0x0_HBM0)
```

🔎 **Tip: mistral.rs's drafter fails on a T4.** The T4 runs the target model in float16 and the drafter loads as bf16, and every drafted request fails with `dtype mismatch in matmul, lhs: F16, rhs: BF16`. On the 3050 and L4 it runs and is slower than no drafter.

---

#### Step 5 — 64 Requests at Once

Aggregate output tokens per second at 64 requests, each engine at its best setting, 128 in / 512 out:

| Engine | 3050 | T4 | L4 | v5e-1 |
| :--- | :--- | :--- | :--- | :--- |
| vLLM | 1,658 (1 drafted) | 🥇 1,165.5 (1 drafted) | 🥇 4,516.3 (4 drafted) | 1,689.1 (none) |
| SGLang | 🥇 1,711 (none) | 908.6 (none) | 2,967.1 (none) | 🥇 4,249.4 (SGLang-JAX) |
| llama.cpp | 633 (none) | 462.5 (4 drafted) | 1,170.3 (2 drafted) | — |
| mistral.rs | 308 (none) | 237.3 (none) | 1,825.2 (none) | — |

512 in / 128 out:

| Engine | 3050 | T4 | L4 | v5e-1 |
| :--- | :--- | :--- | :--- | :--- |
| vLLM | 🥇 603 (none) | 🥇 328.9 (1 drafted) | 🥇 1,710.0 (1 drafted) | 1,288.4 (none) |
| SGLang | 544 (none) | 175.0 (none) | 1,373.0 (none) | 🥇 2,758.9 (SGLang-JAX) |
| llama.cpp | 368 (none) | 281.9 (none) | 630.8 (none) | — |
| mistral.rs | 259 (none) | 196.7 (none) | 1,132.0 (none) | — |

vLLM leads on the T4 and the L4 at both shapes, by 1.283x and 1.522x over SGLang on long replies. On the 3050 SGLang leads vLLM by 1.032x on long replies and vLLM leads on long prompts. On the TPU SGLang-JAX leads vLLM by 2.516x on long replies and 2.141x on long prompts, with the bf16 model against vLLM's int4 build, so engine and weights differ together there. PyTorch, LiteRT-LM, the JAX engine and the Rust model serve one request at a time.

---

#### Step 6 — The Drafter Under Load

vLLM's best drafted setting against no drafter, at 64 requests:

| Card | 128 in / 512 out | 512 in / 128 out |
| :--- | :--- | :--- |
| 3050 | 1.004x (1 drafted) | level (none) |
| T4 | 🥇 1.337x (1 drafted) | 1.114x (1 drafted) |
| L4 | 1.215x (4 drafted) | 1.073x (1 drafted) |

On every card the best draft length falls as requests rise, from 3 or 4 tokens for one request to 1 token or none at 64, except the L4 on long replies, where 4 tokens stay best. The drafter still pays at 64 requests on the T4 and the L4, and on the 3050 it is level from 32. Each ladder found SGLang's drafter losing under load: 0.30 to 0.49x of running without it at 64 on the 3050 and 0.69 to 0.73x on the L4.

---

#### Step 7 — The Same Engine on Another Chip

vLLM without a drafter, against vLLM on the L4:

| Card | 1 request | 64 requests, 128/512 | 64 requests, 512/128 |
| :--- | :--- | :--- | :--- |
| 3050 | 0.863 | 0.444 | 0.378 |
| T4 | 0.772 | 0.235 | 0.185 |
| v5e-1 | 1.025 | 0.455 | 0.808 |

One request depends little on the card; many requests depend a lot. vLLM's output at 64 requests is 26.77x its one-request rate on the L4, 13.76x on the 3050, 11.69x on the v5e chip and 8.59x on the T4 (128 in / 512 out). So the v5e chip matches an L4 for one user and falls to 0.455x at 64 long replies.

---

#### Step 8 — What Each Platform Asks of Each Engine

| Engine | 3050 | T4 | L4 | v5e-1 |
| :--- | :--- | :--- | :--- | :--- |
| vLLM | 🟢 `--no-deps` over the system torch | ⚠️ one patch, Triton tile clamp | 🟢 its own torch | 🟢 pinned TPU image, int4 table patches |
| SGLang | ❌ two patches | ❌ four patches, Marlin from vLLM | ❌ two patches | ❌ SGLang-JAX E2B patch |
| llama.cpp | 🟢 as installed | 🟢 as installed | 🟢 as installed | ❌ no TPU backend |
| mistral.rs | ⚠️ fixed cache | ⚠️ no flash-attn | ⚠️ cuRAND, fixed cache | ❌ no TPU backend |
| PyTorch | 🟢 this repo's engine | 🟢 this repo's engine | 🟢 this repo's engine | ❌ TorchTPU at 6.6 tok/s |
| LiteRT-LM | ⚠️ Vulkan | ⚠️ Vulkan | ⚠️ Vulkan | — |

SGLang needs patches on every platform for this checkpoint. On the T4 its own 4-bit kernels are built only for sm_80 and newer, and the patch sends them to vLLM's Turing build.

---

#### Compare and Contrast

| | 3050 | T4 | L4 | v5e-1 |
| :--- | :--- | :--- | :--- | :--- |
| Fastest, 1 request | vLLM + drafter, 230.5 | vLLM + drafter, 193.0 | vLLM + drafter, 275.3 | SGLang-JAX, 159.9 |
| Fastest, 64 long replies | SGLang, 1,711 | vLLM + drafter, 1,165.5 | vLLM + drafter, 4,516.3 | SGLang-JAX, 4,249.4 |
| Fastest, 64 long prompts | vLLM, 603 | vLLM + drafter, 328.9 | vLLM + drafter, 1,710.0 | SGLang-JAX, 2,758.9 |
| vLLM drafter at 64 | level | 🟢 1.34x | 🟢 1.22x | ❌ does not fit |
| GSM8K range | 0.8992–0.9030 | 0.8954–0.9030 | 0.8954–0.9014 | 0.8969–0.9143 |
| Simplest engine | llama.cpp | llama.cpp | llama.cpp | vLLM |

---

#### So, Which One?

On any of the three GPUs, run vLLM with Google's W4A16 drafter. It is the fastest engine for one request on all three, and at 64 requests it leads on the T4 and the L4 and is within 3% of SGLang on the 3050; drop the draft length toward 1 token as load rises. llama.cpp with its drafter is the engine that runs as installed everywhere, and the right one for one user in under 3 GB. On one TPU v5e chip, vLLM from its pinned image is the engine that serves E2B with tool calls as shipped; SGLang-JAX with the E2B patch is 2.14 to 2.52x faster at 64 requests. The chip takes vLLM at 64 long replies from 871.8 tokens per second on the T4 to 3,715.7 on the L4, and leaves the answers where they were.

---

#### Summary

The goal of this article was to find whether the serving engine to run for Gemma 4 E2B changes with the accelerator. The key was running the same model, prompts, load sweep and question sets through the same engines on four chips, and computing every comparison from the ladders' own result files. The results were:

- 🟢 Five CUDA engines on three GPUs give 15 GSM8K scores between 0.8954 and 0.9030 and BFCL between 0.9150 and 0.9250; drafters change no score.
- 🟢 vLLM with its drafter is the fastest engine for one request on every GPU: 230.5, 193.0 and 275.3 tokens per second on the 3050, T4 and L4.
- 🟢 vLLM leads at 64 requests on the T4 and L4 at both shapes, and its drafter still pays there (1.34x and 1.22x on long replies).
- 🟢 On one TPU v5e chip SGLang-JAX with an E2B patch is the fastest engine, 2.52x vLLM at 64 long replies.
- ⚠️ For one request without a drafter the engines on each GPU are within 1.07 to 1.16x of each other; the chip sets the speed.
- ⚠️ The drafter's gain depends on the engine and the card: 1.78 to 1.96x for vLLM, 0.72 to 1.39x for SGLang, smallest on the T4.
- ❌ vLLM with Google's drafter runs out of memory on one v5e chip, and mistral.rs's drafter fails on a T4.
- ❌ SGLang needs patches for this checkpoint on all four platforms, and LiteRT-LM's own int2/4/8 file scores lowest on every GPU.

Scope: one RTX 3050 OEM in a desktop, one Tesla T4 on a GCE `n1-standard-2`, one NVIDIA L4 on an EC2 g6.4xlarge and one flex-start TPU v5e chip, Gemma 4 E2B, greedy decoding, one engine at a time. The ladders ran on different days between 2026-10-14 and 2026-11-11, and the TPU ran different engines and bf16 weights for SGLang-JAX and the JAX engines, so the TPU rows compare engine and weights together. One-request speed is six prompts of 256 tokens, three repeats; load is two synthetic shapes at 1 to 64 requests, five repeats each; accuracy is one pass of GSM8K (1,319) and BFCL simple (400). Cost and power were not compared.

The strategy for comparing inference serving engines across GPUs and TPUs was validated with an incremental step by step approach.

---

#### References

- [Six Serving Engines, One 8 GB GPU: Gemma 4 E2B on an RTX 3050](https://dev.to/gde/six-serving-engines-one-8-gb-gpu-gemma-4-e2b-on-an-rtx-3050-ad7)
- [Six Serving Engines on One Tesla T4: Gemma 4 E2B on a Turing GPU](https://dev.to/gde/six-serving-engines-on-one-tesla-t4-gemma-4-e2b-on-a-turing-gpu-3a6i)
- [Six Serving Engines on One EC2 G6: Gemma 4 E2B on an NVIDIA L4](https://dev.to/aws-builders/six-serving-engines-on-one-ec2-g6-gemma-4-e2b-on-an-nvidia-l4-3ff3)
- [Six Serving Engines on One Cloud TPU v5e Chip: Gemma 4 E2B](https://dev.to/gde/six-serving-engines-on-one-cloud-tpu-v5e-chip-gemma-4-e2b-22hn)
- [Porting Gemma 4 E2B to SGLang-JAX on One Cloud TPU v5e](https://dev.to/gde/porting-gemma-4-e2b-to-sglang-jax-on-one-cloud-tpu-v5e-5e9l)
- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [Roll-up script and evidence](https://github.com/xbill9/gemma4-dev/tree/main/docs/serving-engines-rollup)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [SGLang-JAX](https://github.com/sgl-project/sglang-jax)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
- [Gemma 4 E2B QAT W4A16 checkpoint](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4)
- [GSM8K](https://github.com/openai/grade-school-math)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

Hope you enjoyed it!
