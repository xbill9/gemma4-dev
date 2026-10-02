---
title: "Gemma 4 at Over 70 Tokens/s on a 2021 Laptop's 4 GB GPU: The Live Demo, Step by Step"
published: false
description: "The live demo, step by step: Gemma 4 E2B answering at 76 tok/s from a GTX 1650 Ti with 4 GB of memory, and the exact re-pack of Google's QAT weights on Hugging Face that makes it fit, run faster and stay close to bf16."
tags: gemma, llamacpp, cuda, huggingface
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/local-llamacpp-1650ti-2b-q4_0/docs/demo-1650ti/devto-cover.f8e939f0.jpg
---

This article provides a step by step guide to the live demo of Gemma 4 E2B on a local, laptop hosted GPU enabled system: a Lenovo Yoga 9 from 2021 with a 4 GB GTX 1650 Ti. One person asks questions, and the answers stream back at over 70 tokens per second with no cloud involved. The model file that makes it work is a re-pack of Google's quantization-aware-trained (QAT) weights, published on Hugging Face.

https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0

**Gemma 4 E2B in bf16 is 9.5 GiB of weights and this card has 4 GiB. The re-packed QAT file serves it in 1488 MiB and decodes at 76 tok/s, 1.12x faster than Google's own GGUF on the same card and 32x closer to bf16.**

---

#### What is this project trying to Do?

Every other rig in this repository serves Gemma 4 from rented hardware. This one serves it from the laptop under the desk, for one person having a conversation. The demo is about responsiveness: the first words of a short answer on screen in under half a second, and a long answer streaming faster than anyone reads it.

Two things get it there. QAT, which trains the model to live on a 4-bit grid, makes it small enough. Packing those trained 4-bit values into the file exactly, with nothing re-rounded, makes it faster and more accurate than the stock conversion. The demo steps come first; the re-pack is explained after them.

---

#### At This Point You Should Have…

- An Nvidia GPU with 2 GB or more of free memory. This one is a GTX 1650 Ti Max-Q: compute capability 7.5, 4096 MiB, a 40 W power cap and no tensor cores.
- llama.cpp built with CUDA for your card's architecture (`make build` in the rig builds it for `sm_75`).
- The re-packed GGUF from Hugging Face: `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`.
- About 3 GB of disk.

---

#### The Hardware

```text
product_version: Yoga 9 15IMH5
Model name:                              Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
Mem:            15Gi       7.8Gi       1.0Gi       890Mi       7.9Gi       7.4Gi
NVIDIA GeForce GTX 1650 Ti with Max-Q Design, 4096 MiB, 7.5
            Current Power Limit                        : 40.00 W
```

The earliest published reviews of this laptop are from January 2021. The GTX 16 series shares compute capability 7.5 with the T4 datacenter card, but Nvidia left the tensor cores off this chip, so every matrix multiply runs on ordinary CUDA cores.

---

#### Why Does It Fit at All?

| Gemma 4 E2B | Weights | Fits in 4 GiB? |
| :--- | ---: | :--- |
| bf16 | 9.5 GiB | ❌ |
| int8 | ~4.8 GB | ❌ |
| Google QAT GGUF, resident on the card | 1.31 GiB | 🟢 |
| Re-packed QAT GGUF, resident on the card | **1.20 GiB** | 🟢 🥇 |

Two properties of the model close the gap.

**QAT.** Google trained a release of Gemma 4 knowing it would be stored at 4 bits, so the 4-bit copy keeps close to bf16 quality. Most of the shrink from 9.5 GiB comes from here.

**Per-layer embeddings stay in host memory.** The `E` in E2B is a large per-layer embedding table, `per_layer_token_embd`, which llama.cpp loads lazily: it stays memory-mapped in system RAM and each token reads one row. That table is half of the re-packed file and none of it occupies the card.

---

#### Step 1 — Download the Re-Packed GGUF

```shell
hf download xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf gemma-4-E2B-it-q4_0-exact.gguf \
  --local-dir ~/models/gemma-4-E2B-it-qat-q4_0-exact-v2
sha256sum ~/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
```

```text
419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8  gemma-4-E2B-it-q4_0-exact.gguf
```

The repository also holds `gguf_exact.py`, the script that builds the file from Google's two Hugging Face downloads, so anyone can rebuild it and compare the hash.

---

#### Step 2 — Configure the Server

`tpu.env` is the rig's configuration. These are the values the demo serves with:

```shell
MODEL_PATH=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
MODEL_SHA256=419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8
N_GPU_LAYERS=99
CONTEXT_SIZE=8192
KV_CACHE_TYPE=f16
FLASH_ATTENTION=1
PARALLEL_SLOTS=1
REASONING=off
```

| Setting | Why |
| :--- | :--- |
| `-ngl 99` | every transformer layer on the card |
| `-fa 1` | flash attention, +4.8% decode on this card |
| `-ctk f16 -ctv f16` | KV cache at full precision; `q8_0` costs 12% of decode here |
| `--parallel 1` | one user gets the whole card |
| `--reasoning off` | the answer starts at once; one request can still ask for thinking |

`make serve` runs `llama-server` with exactly these flags.

---

#### Step 3 — Start It

The demo uses two small shell wrappers: `gpu` starts and stops the server and checks which device it is running on, and `ask` streams a chat from the terminal. Both read every value from `tpu.env`.

```shell
gpu start
ask -q "hi"
```

```text
gpu: starting /home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf on 127.0.0.1:8080 (-ngl 99)
gpu: pid 132745, log /home/xbill/gemma4-dev/local-llamacpp-1650ti-2b-q4_0/run/llama-server.log
gpu: waiting up to 180s for http://127.0.0.1:8080/health

gpu: healthy after 4s -- http://127.0.0.1:8080
gpu: device=gpu · pid=132745 · -ngl 99 · mapped: ggml-cuda, libcublas, libcuda, libcudart · /home/xbill/llama.cpp/build/bin/llama-server
Hi! How can I help you today? 😊
```

Healthy in 4 seconds, because llama.cpp memory-maps the file. The `hi` is a warm-up so the first question in front of an audience does not wait on the disk. Keep a second terminal open on `nvtop` for Demo Step 3.

---

#### Demo Step 1 — It Answers Instantly

```shell
ask What is the capital of Australia? One sentence.
```

```text
The capital of Australia is Canberra.
[wall 0.38 s]
```

0.38 seconds end to end, from a 4 GB laptop GPU.

---

#### Demo Step 2 — Pipe It Real Work

`ask` reads standard input as material and its arguments as the instruction, so it slots into a shell pipeline:

```shell
git -C ~/gemma4-dev log --oneline -12 | ask summarize what this project has been working on, in 3 bullets
```

```text
Here is a 3-bullet summary of what this project has been working on:

* **Article Refinement and Content Updates:** Significant effort has been dedicated to rewriting, retitling, and restructuring articles related to v5e QAT builds, ...
* **Benchmark and Index Maintenance:** The project has been actively working on updating and regenerating various benchmarks, indices, and documentation files ...
* **Resource and Environment Configuration:** The work also involves managing and referencing specific hardware/software configurations, ...
[wall 4.26 s]
```

---

#### Demo Step 3 — Show the Footprint

```shell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=name,memory.used,memory.total,power.draw,temperature.gpu --format=csv
gpu arm
```

```text
pid, process_name, used_gpu_memory [MiB]
7978, /home/xbill/llama.cpp/build/bin/llama-server, 1488 MiB
name, memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu
NVIDIA GeForce GTX 1650 Ti with Max-Q Design, 1493 MiB, 4096 MiB, 40.43 W, 47
device=gpu · pid=7978 · -ngl 99 · mapped: ggml-cuda, libcublas, libcuda, libcudart · /home/xbill/llama.cpp/build/bin/llama-server
```

The server holds 1488 MiB, 36% of the card, drawing 40 W. `gpu arm` reads the device from the running process: the loaded CUDA libraries and the real `-ngl`. This rig has a CPU-only twin that serves the same file on the same port, so a healthy endpoint alone does not say which device is answering.

---

#### Demo Step 4 — Hold a Conversation

`ask` with no arguments opens a session that keeps the conversation in context:

- "Explain what quantizing a model means, in three sentences, for an engineer."
- "Give me an everyday analogy for it."
- "Now as a haiku."

The second answer, from the same session:

```text
Imagine you have a super detailed blueprint for a house (the high-precision model), where every single measurement is recorded with extreme accuracy (32-bit precision). Quantizing is like printing that blueprint on a smaller, simpler sketch (like an 8-bit drawing), where you only record the most essential measurements, sacrificing a tiny bit of microscopic detail to make the drawing much easier and faster to read and reproduce.
```

`!` clears the conversation and Ctrl-D leaves. Keep questions on general topics: a small model asked about its own training will invent an answer with confidence.

---

#### Demo Step 5 — Turn Thinking On

```shell
ask -T Which is larger, 9.11 or 9.9? Explain.
```

```text
**9.9 is larger than 9.11.**

Here is the explanation based on comparing the place values:

1. **Whole Numbers:** Both numbers have a whole number part of **9**.
2. **Tenths Place:**
   * In 9.**1**1, the digit in the tenths place is **1**.
   * In 9.**9**, the digit in the tenths place is **9**.
3. **Comparison:** Since 9 is greater than 1, the number 9.9 is larger than 9.11.
[wall 9.36 s]
```

In a terminal the model's reasoning streams first, dimmed, then the answer. That reasoning is why thinking is off by default: this question took 9.36 seconds with it on, against well under a second for a short answer with it off.

---

#### How Fast Is It, Measured?

llama-server reports its own timings with every response. Five requests for a 250-word explanation, thinking off:

```text
{"run":1,"finish":"stop","completion_tokens":302,"predicted_per_second":76.1367071253587}
{"run":2,"finish":"length","completion_tokens":400,"predicted_per_second":76.3564025991796}
{"run":3,"finish":"stop","completion_tokens":366,"predicted_per_second":76.20224702680754}
{"run":4,"finish":"stop","completion_tokens":239,"predicted_per_second":76.87420440850953}
{"run":5,"finish":"stop","completion_tokens":285,"predicted_per_second":77.1396194517872}
```

**76.14 to 77.14 tokens per second** while serving. `llama-bench`, which measures generation with no HTTP in the way, puts the same file at 81.75 tok/s and Google's GGUF at 73.26 on this card.

---

#### How the Re-Pack Works

**What QAT stores.** Google publishes the QAT weights twice: as a ready GGUF, and unpacked as bf16 safetensors (`google/gemma-4-E2B-it-qat-q4_0-unquantized`). In the unpacked copy every block of 32 values along a row holds at most 16 distinct values: an integer level from -8 to 7 times one step size per block. That is Q4_0's own layout, 4-bit levels plus one fp16 scale per 32 values. A QAT weight matrix therefore fits into Q4_0 with nothing lost except rounding the step to fp16.

**What Google's GGUF does to it.** Its transformer layers are Q4_0, but quantized with llama.cpp's default step, the block's largest value divided by 8. Wherever a block's largest value sits below level 8, that step differs from the trained one and values move off the grid. Its two embedding tables are stored as Q6_K, a different format altogether, and they are most of the file:

```text
gemma-4-E2B_q4_0-it.gguf
  by tensor type (the slot-5 token is q4_0; the file mostly is not):
    Q6_K      2.257 GB  (67.7%)
    Q4_0      1.048 GB  (31.4%)
    F16       0.028 GB  ( 0.8%)
    F32       0.001 GB  ( 0.0%)
```

**What the re-pack does.** `gguf_exact.py` copies Google's GGUF header, tokenizer, chat template and settings byte for byte. It rewrites all 275 layer matrices, both embedding tables and `per_layer_model_proj` as Q4_0. For each block of 32 values it finds the step the block was trained on, the first `amax/m` for `m` from 1 to 8 that puts every value on an integer level, and refines it by least squares. A block with no such step stops the build.

```text
gemma-4-E2B-it-q4_0-exact.gguf
  lazy (never on GPU):        1.321 GB  (51% of file)
  must be resident:           1.283 GB = 1.20 GiB

  by tensor type (the slot-5 token is q4_0; the file mostly is not):
    Q4_0      2.603 GB  (100.0%)
    F32       0.001 GB  ( 0.0%)
```

Of 4.63 billion rebuilt values, 96.84% decode back to the exact source value; the rest are within the fp16 rounding of the scale. The file shrank from 3,349,516,256 bytes to 2,620,370,912, 21.8% smaller.

---

#### Why the Re-Pack Is Faster

Generating one token reads every resident weight once, so on this card decode speed follows the number of bytes read per token. `token_embd` doubles as the output layer, so the whole table is read for every generated token. As Q6_K it is 330.3 MB; as Q4_0 it is 226.5 MB, 31% smaller.

`llama-bench`, full offload, four passes alternating the order of the files, with the card cooled to 50 °C before each pass:

| tok/s, mean of 4 passes | Google GGUF | Re-packed |
| :--- | ---: | ---: |
| Generation (tg128) | 73.26 | **81.75** 🥇 |
| Prefill (pp512) | 338.26 | **344.64** 🥇 |

Generation is 1.12x faster and held between 1.111x and 1.121x in every pass. The re-packed file also leaves the card 134 MiB lighter: 1480 MiB against 1614 MiB for the server process.

---

#### Why the Re-Pack Is More Accurate

The quality check compares each file's next-token predictions against the bf16 model over wikitext-2, 16 chunks of 512 tokens, scored on the card. KL divergence measures how far the two probability distributions are apart; zero means identical.

```text
Mean PPL(Q)/PPL(base)         :   1.021563 ±   0.005079
Mean    KLD:   0.001680 ±   0.000072
99.0%   KLD:   0.016059
Same top p: 98.260 ± 0.205 %
```

| against bf16 | Google GGUF | Re-packed |
| :--- | ---: | ---: |
| Mean KL divergence | 0.054264 | **0.001680** 🥇 |
| Same most-likely token | 87.18% | **98.26%** 🥇 |
| Perplexity / bf16 perplexity | 1.092 | **1.022** 🥇 |

Google's file picks a different most-likely next token from bf16 about one time in eight. The re-packed file does so under one time in fifty. The weights QAT trained are the weights the card serves.

---

#### 🔎 Tip: Settings That Look Like Speedups on This Card

Measured on this card at first light, and all three stay off:

- **Quantizing the KV cache to `q8_0`** costs 12% of decode and 20–40% of prefill. With no tensor cores the dequantization is not hidden, and E2B's KV cache is small at 8192 tokens.
- **`GGML_CUDA_FORCE_MMQ`**, which llama.cpp itself suggests for cards without tensor cores, measured marginally slower.
- **Lowering `-ngl` to save memory** moves real matrix multiplies to the CPU and buys memory the card was never short of.

---

#### 🔎 Tip: Keep Demo Inputs Short

The context is 8192 tokens and prefill runs at about 345 tokens per second, so a 4,000-token paste is over 11 seconds of silence before the first word. Pipe a commit summary into `ask`, never a large diff.

---

#### Compare and Contrast

| on the GTX 1650 Ti | Google QAT GGUF | Re-packed QAT GGUF |
| :--- | ---: | ---: |
| File size | 3.35 GB | **2.62 GB** 🥇 |
| Share of weights stored as Q4_0 | 31.4% | **100%** 🥇 |
| Server memory on the card | 1614 MiB | **1480 MiB** 🥇 |
| Generation, `llama-bench` | 73.26 tok/s | **81.75 tok/s** 🥇 |
| Mean KL divergence from bf16 | 0.054 | **0.0017** 🥇 |

---

#### So, Which One?

For this laptop, the re-packed file. It loads in the same llama.cpp build with the same flags; the only change is the path in `tpu.env`. It is smaller, faster and closer to bf16, on a card with little room to spare.

The re-pack depends on Google's unpacked QAT checkpoint staying published, and it works only where the source was trained onto a 4-bit grid. `gguf_exact.py` refuses any tensor with blocks off it.

---

#### Summary

The goal of this article was to run Gemma 4 E2B as a responsive, single-user assistant on a 2021 laptop with a 4 GB GPU, and to show the demo step by step. The key to the solution was the QAT release on Hugging Face, re-packed so every trained 4-bit value lands in the file exactly. The measured results were:

- 🟢 **76.14 to 77.14 tok/s while serving**, measured by llama-server over five requests; 81.75 tok/s in `llama-bench`.
- 🟢 **0.38 s for a one-sentence answer**, end to end, and a 4-second cold start.
- 🟢 **1488 MiB of a 4096 MiB card**, drawing 40 W.
- 🟢 **1.12x faster generation than Google's GGUF**, from a 31% smaller output table.
- 🟢 **32x closer to bf16** by mean KL divergence; same top token 98.26% of the time against 87.18%.
- ⚠️ **Thinking costs seconds**: 9.36 s for a short reasoning question, so it stays off by default.
- ❌ **KV cache quantization and `FORCE_MMQ` are slower** on this card.

Scope: one Lenovo Yoga 9 15IMH5 (Core i7-10750H, GTX 1650 Ti Max-Q, 4096 MiB, 40 W cap), llama.cpp `f95b0d9` built for sm_75 with CUDA 13.4, one user, thinking off unless stated. The demo steps were re-run on 2026-10-02 for the output shown here. `llama-bench` and KL divergence figures come from the 2026-09-29 runs of four alternating passes and 16 × 512 tokens of wikitext-2. Task accuracy beyond KL divergence was not measured.

The strategy for running Gemma 4 on a 4 GB laptop GPU with an exactly re-packed QAT GGUF was validated with an incremental step by step approach.

---

#### References

* [local-llamacpp-1650ti-2b-q4_0 | GitHub](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0)
* [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
* [google/gemma-4-E2B-it-qat-q4_0-gguf | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
* [google/gemma-4-E2B-it-qat-q4_0-unquantized | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized)
* [Gemma 4 on an Old 4 GB Laptop GPU: QAT Takes It From 9.5 GiB to 1.6 | dev.to](https://dev.to/gde/gemma-4-on-an-old-4-gb-laptop-gpu-qat-takes-it-from-95-gib-to-16-b5l)
* [Quantization-Aware Training for Gemma 4 | Google](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
