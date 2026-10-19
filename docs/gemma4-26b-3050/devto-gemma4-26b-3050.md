---
title: "Can a 26B MoE Run on an 8 GB GPU? Gemma 4 26B A4B on an RTX 3050"
published: false
description: "Gemma 4 26B A4B, a 14.4 GB mixture-of-experts model, serves from an 8 GB RTX 3050 with most of its experts in system RAM: 131,072 tokens of context, 28 to 34 tok/s empty, and GSM8K and BFCL scores level with the 12B."
tags: gemma, llamacpp, cuda, benchmarking
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-26b-3050/devto-cover.00669e4e.jpg
---

This article provides a step by step guide to serving Gemma 4 26B A4B, Google's quantization-aware trained (QAT) mixture-of-experts model, on an 8 GB NVIDIA RTX 3050 with llama.cpp. The model file is 14.4 GB, so it cannot fit on the card. llama.cpp can keep the experts in system RAM and run them on the CPU, while the rest of the model runs on the GPU. The measures are how fast it generates, how much context fits, what the drafter adds, and whether the answers match the 12B that fits the card whole. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

This is the fifth part of a short series. [Part 1, Gemma 4's MTP Drafter, Explained](https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0), covers how the drafter works. [Part 2, Do Drafters Work on Small Models?](https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g), measures E2B and E4B on this card. [Part 3, Can a QAT 12B Fit an 8 GB GPU?](https://dev.to/gde/can-a-qat-12b-fit-an-8-gb-gpu-gemma-4-12b-on-an-rtx-3050-1267), fits the 12B and its drafter on the card. [Part 4, How Much Context Fits Beside a 12B on an 8 GB GPU?](https://dev.to/gde/how-much-context-fits-beside-a-12b-on-an-8-gb-gpu-gemma-4-12bs-context-cache-on-an-rtx-3050-20im), gives the 12B a 4-bit context cache for long conversations. This part runs a model twice the card's size.

https://github.com/xbill9/gemma4-dev

**Can a 26B run on an 8 GB card? Yes, with 21 of its 30 expert layers in system RAM**, and its answers match the 12B's:

| RTX 3050 OEM, 8 GB, 4-bit cache, drafter | Gemma 4 12B | Gemma 4 26B A4B |
| :--- | ---: | ---: |
| Model file | 6.7 GB, all on the card | 14.4 GB, experts mostly in RAM |
| Most context that loads | 81,920 tokens | **131,072 tokens** |
| Generation, nothing held | 55.1 tok/s | 33.7 tok/s |
| Generation, 65,536 tokens held | 30.8 tok/s | 22.8 tok/s |
| GSM8K / BFCL simple | 0.961 / 0.950 | **0.965 / 0.945** |
| GSM8K run, 1,319 questions | 1 h 57 min | 3 h 26 min |

The 26B holds 1.6x the 12B's context and scores the same, at 0.61x to 0.74x the 12B's generation speed.

---

#### What Is Being Measured?

Gemma 4 26B A4B is a mixture-of-experts model. Each of its 30 layers holds 128 experts, small feed-forward networks, and a router picks 8 of them for each token. The file is 14.4 GB: 12,251 MiB of expert weights (408 MiB per layer) and 1,505 MiB of everything else (attention, norms, routers and the token embedding). So 89% of the file is experts, and each token reads only a sixteenth of them.

llama.cpp's `-ncmoe N` keeps the experts of the first N layers in system RAM, where the CPU runs them, and puts everything else on the GPU. Every token still crosses the CPU, so the CPU and its memory are part of the serving path, unlike on the 12B.

The host is an i7-5820K desktop (6 cores, 12 threads, quad-channel DDR4, 46 GiB) with the RTX 3050 OEM. Every setting is measured on one llama.cpp build, `7182ef71e`, with Google's `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` unmodified, and one request at a time (`--parallel 1`).

---

#### Step 1 — Serve the Rig

The rig is `local-llamacpp-3050oem-26b-q4_0`. Its `tpu.env` holds every serving setting, and `make serve` starts `llama-server` from it:

```
git clone https://github.com/xbill9/gemma4-dev
cd gemma4-dev/local-llamacpp-3050oem-26b-q4_0
make download draft
make serve
```

The setting this article arrives at:

```
llama-server -m gemma-4-26B_q4_0-it.gguf -ngl 99 -ncmoe 21 -c 131072 -ctk q4_0 -ctv q4_0 -fa 1 \
  -t 8 -tb 12 --parallel 1 --reasoning off --metrics \
  -md gemma-4-26B-A4B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

`-ngl 99` puts every layer on the GPU; `-ncmoe 21` then takes the experts of the first 21 layers back to system RAM.

---

#### Step 2 — Find How Many Expert Layers the Card Can Hold

`llama-bench` measures generation (128 tokens) and prompt reading (512 tokens) with the experts of 30 down to 16 layers in RAM:

| experts in RAM, of 30 layers | 30 (all) | 24 | 20 | 18 | 17 | 16 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generation, tok/s | 24.35 | 27.66 | 30.65 | 32.27 | **33.31** | does not load |
| Prompt reading, tok/s | 264 | 313 | 365 | 394 | 408 | |

**Each expert layer moved to the card adds about half a token per second.** With every expert in RAM the 26B already generates at 24 tok/s; 17 layers in RAM is the fastest setting that loads, 1.37x faster. At 16 the working buffer no longer fits.

---

#### 🔎 Tip: Threads Matter Here

Here the CPU runs most of the experts, so the thread count sets part of the speed. At 17 layers in RAM, 4 threads generate at 28.49 tok/s, 6 at 33.03 and 8 at 33.97, with 12 level with 8 inside the run's spread. The rig serves `-t 8`, and anything else busy on the CPU or the memory bus slows the model.

---

#### Step 3 — Trade Expert Layers for Context

Every expert layer on the card costs about 410 MiB that the context cache could use. Each row below is the most context that loads with that many layers in RAM, one request at a time:

| experts in RAM | 16-bit cache | 8-bit cache | 4-bit cache | 4-bit cache + drafter |
| :--- | ---: | ---: | ---: | ---: |
| 18 | does not load | | 98,304 | |
| 19 | 32,768 | 65,536 | 131,072 | 65,536 |
| 20 | 32,768 | 65,536 | 131,072 | 98,304 |
| **21** | 32,768 | 131,072 | 131,072 | **131,072**, 7,836 MiB |
| 22 | | 131,072 | **262,144**, 7,874 MiB | 131,072 |

Blank cells were not tried. **The 26B holds more context than the 12B on the same card**: 131,072 tokens beside the drafter against the 12B's 81,920. Its weights mostly live in RAM, so the card's memory goes to the cache. Without the drafter, 22 layers in RAM hold the model's full 262,144 tokens. The rig serves 21 with the drafter, the most context that fits with it.

---

#### Step 4 — Measure the Drafter

Google ships a QAT drafter for the 26B, a small model that proposes the next tokens for the 26B to check. It is stored as exact 4-bit (Q4_0) on its trained grid, 252 MB against 462 MB for the converter's 8-bit file, and published as `xbill9/gemma-4-26B-A4B-it-qat-assistant-q4_0-exact-gguf`. `spec_ab.py` runs six prompts, greedy three times and sampled once, 256 tokens each, with 20 expert layers in RAM:

| drafter setting | greedy tok/s | x no drafter | drafted tokens kept | sampled x no drafter |
| :--- | ---: | ---: | ---: | ---: |
| none | 29.6 | 1.00x | — | 1.00x |
| exact Q4_0, 1 token | 37.0 | 1.25x | 0.76 | 1.25x |
| **exact Q4_0, 2 tokens** | **36.8** | **1.24x** | **0.65** | **1.22x** |
| exact Q4_0, 3 tokens | 30.1 | 1.01x | 0.58 | 0.61x |
| 8-bit, 2 tokens | 35.6 | 1.20x | 0.66 | 1.17x |

**The drafter gives 1.24x here against 2.01x on the 12B.** It keeps as many drafted tokens as the 12B's does, so the gain is lost in the checking: each drafted token the 26B checks routes to its own 8 experts, and the CPU reads them (inferred, not measured). By prompt it ranges from 1.51x on code to 1.12x on a story. 2 tokens are served: level with 1 overall and ahead on code. 3 tokens lose on sampled text.

---

#### Step 5 — Measure Speed With a Long Conversation Held

The 12B's context test at the serving setting: 0 to 98,304 tokens of a long document (the NVIDIA driver README) ahead of three questions (a story, a Python function, a summary of the document), 256 greedy tokens each, with the prompt cache on:

| tokens already in context | 0 | 16,384 | 32,768 | 65,536 | 98,304 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **26B, drafter** | **33.7 tok/s** | **30.0** | **28.0** | **22.8** | **19.3** |
| 26B, no drafter | 28.2 | 24.9 | 22.3 | 17.5 | 14.2 |
| 12B, drafter | 55.1 | 42.4 | 37.8 | 30.8 | does not fit |
| 12B, no drafter | 27.5 | 24.1 | 22.2 | 19.2 | 16.8 |

Each figure is the geometric mean of the questions (story and code only with nothing held). **Without drafters the 26B generates as fast as the 12B** up to 32,768 tokens held (1.00x to 1.03x), and falls to 0.85x at 98,304. The 12B's drafter is what puts it ahead.

**The 26B's drafter gains with depth**, from 1.20x with nothing held to 1.36x at 98,304 tokens, while the 12B's falls from 2.00x to 1.60x at 65,536. On code alone the 26B's drafter reaches 1.67x at 98,304 tokens (23.6 against 14.1 tok/s). Reading new text falls from 302 tok/s at 8,192 tokens to 227 at 98,304; the 12B reads 828 tok/s at 8,192.

---

#### Step 6 — Check the Answers

`gen_eval.py` runs GSM8K (1,319 grade-school math questions, up to 2,048 tokens per reply) and BFCL simple (400 single function calls), one request at a time, at the serving setting:

```
JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 google/gemma-4-26B-A4B-it gsm8k gsm8k-q4x.jsonl --concurrency 1
```

| one request at a time, 4-bit cache | GSM8K | BFCL simple |
| :--- | ---: | ---: |
| **26B, drafter** | **0.965** | **0.945** |
| 26B, no drafter | 0.962 | 0.945 |
| 12B, drafter | 0.961 | 0.950 |
| 12B, no drafter | 0.962 | 0.950 |

**The 26B scores level with the 12B, and the drafter changes neither score.** `compare.py` pairs the runs question by question. 26B against 12B with drafters: +0.38 points on GSM8K (95% range −0.45 to +1.29) and −0.50 on BFCL (−2.50 to +1.50). 26B with the drafter against without: +0.30 on GSM8K (−0.23 to +0.83), with 12 of 1,319 answers different, and zero on BFCL, with 392 of 400 replies word for word the same.

The time is where the two models part. GSM8K took the 26B 3 h 26 min with the drafter and 4 h 38 min without (1.35x), against 1 h 57 min for the 12B with its drafter.

---

#### 🔎 Tip: An MoE Model Keeps Its Speed With Its Experts Off the Card

A dense model with two thirds of its layers on the CPU reads all of them for every token. On this card the dense 31B generates at 2.5 tok/s with 20 of 60 layers on the GPU. The 26B reads 8 of 128 experts per layer, so its CPU share per token is small, and it generates at 24 tok/s with every expert in RAM. For a model larger than the card, the active parameter count decides the speed.

---

#### Compare and Contrast

| RTX 3050 OEM, 8 GB | 🥇 12B, 4-bit cache + drafter | 26B, 4-bit cache + drafter | 26B, 4-bit cache, no drafter |
| :--- | ---: | ---: | ---: |
| Expert layers in RAM | — | 21 of 30 | 22 of 30 |
| Most context | 81,920 tokens | 131,072 tokens | 262,144 tokens |
| Generation, nothing held | 55.1 tok/s | 33.7 tok/s | 28.2 tok/s |
| Generation, 65,536 held | 30.8 tok/s | 22.8 tok/s | 17.5 tok/s |
| GSM8K | 0.961 | 0.965 | 0.962 |
| BFCL simple | 0.950 | 0.945 | 0.945 |
| GSM8K run | 1 h 57 min | 3 h 26 min | 4 h 38 min |

---

#### So, Which One?

For one long conversation up to 81,920 tokens, the 12B with its drafter. It gives the same GSM8K and BFCL answers as the 26B and generates 1.35x to 1.64x faster.

For a conversation between 81,920 and 131,072 tokens, the 26B with its drafter at 21 expert layers in RAM. It holds the context with the drafter on, where the 12B has to drop its drafter to get there.

For the model's full 262,144 tokens, the 26B without the drafter at 22 expert layers in RAM, at 28 tok/s with nothing held.

---

#### What Comes Next

The answer tests here use short questions. The next measurements are the dense 31B on the same card, now running GSM8K, function calling with a long conversation held, and the 26B's 4-bit cache against the 16-bit cache for answers.

---

#### Summary

The goal of this article was to find whether Gemma 4 26B A4B, a model twice the card's size, can serve from an 8 GB RTX 3050, and how it compares with the 12B that fits. The key to the solution was keeping the experts in system RAM with `-ncmoe` and spending the freed card memory on context. The results were:

- 🟢 The 14.4 GB 26B serves from an 8 GB card with 21 of its 30 expert layers in system RAM
- 🟢 It holds 131,072 tokens of context with its drafter, 1.6x the 12B, and the full 262,144 without it
- 🟢 GSM8K 0.965 and BFCL 0.945, level with the 12B, with or without the drafter
- 🟢 Without drafters it generates as fast as the 12B up to 32,768 tokens held
- ⚠️ Its drafter gives 1.24x against the 12B's 2.01x, so with drafters it runs at 0.61x to 0.74x the 12B
- ⚠️ The CPU is in the serving path: thread count and other load on the host change its speed
- ❌ A GSM8K run takes 1.76x the 12B's time for the same score

Scope: Gemma 4 26B A4B-it QAT only, Google's Q4_0 GGUF unmodified with the exact Q4_0 26B drafter, text only, one RTX 3050 OEM (8 GB) in an i7-5820K desktop, llama.cpp `7182ef71e`, 16-bit, 8-bit and 4-bit context caches, 0 to 98,304 tokens of one document held for speed, greedy decoding, thinking off, GSM8K and BFCL simple with short prompts, one request at a time throughout.

The strategy for using MCP for local accelerator comparison was validated with an incremental step by step approach.

---

#### References

- Part 1, Gemma 4's MTP Drafter, Explained: https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0
- Part 2, Do Drafters Work on Small Models?: https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g
- Part 3, Can a QAT 12B Fit an 8 GB GPU?: https://dev.to/gde/can-a-qat-12b-fit-an-8-gb-gpu-gemma-4-12b-on-an-rtx-3050-1267
- Repository: https://github.com/xbill9/gemma4-dev
- 26B rig: https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-26b-q4_0
- First light run: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-26b-q4_0/benchmarks/runs/2026-10-18-first-light-3050oem/REPORT.md
- Answers run: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-26b-q4_0/benchmarks/runs/2026-10-18-accuracy-26b-3050oem/REPORT.md
- 12B context cache run: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem/REPORT.md
- 31B first light run: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-31b-q4_0/benchmarks/runs/2026-10-18-first-light-3050oem/REPORT.md
- 26B GGUF: https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf
- 26B drafter: https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-assistant-q4_0-exact-gguf
- llama.cpp: https://github.com/ggml-org/llama.cpp
