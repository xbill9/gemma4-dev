---
title: "Can a 12B Run on a 4 GB Laptop GPU? Gemma 4 12B on a GTX 1650 Ti"
published: false
description: "Gemma 4 12B serves from a 4 GB GTX 1650 Ti laptop GPU with 28 of its 48 layers on the CPU: the 12B's answers, at 7 to 10 tok/s with the drafter and about 60 tok/s of prompt reading. Too slow for chat."
tags: gemma, llamacpp, cuda, benchmarking
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-12b-1650ti/devto-cover.c9abd9a1.jpg
---

This article provides a step by step guide to serving Gemma 4 12B, Google's quantization-aware trained (QAT) model, on a 4 GB NVIDIA GTX 1650 Ti laptop GPU with llama.cpp. The model file is 6.7 GB, so it cannot fit on the card. llama.cpp puts as many layers as will fit on the GPU and runs the rest on the CPU. The measures are how fast it generates, how fast it reads a prompt, what the drafter adds, how a long conversation changes both, and whether the answers match the same model on an 8 GB card. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

This is the seventh part of a short series. [Part 1, Gemma 4's MTP Drafter, Explained](https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0), covers how the drafter works. [Part 2, Do Drafters Work on Small Models?](https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g), measures E2B and E4B on this laptop's card and the RTX 3050. [Part 3, Can a QAT 12B Fit an 8 GB GPU?](https://dev.to/gde/can-a-qat-12b-fit-an-8-gb-gpu-gemma-4-12b-on-an-rtx-3050-1267), fits the 12B and its drafter on the 3050. Parts 4 to 6 give the 12B a 4-bit context cache and run the 26B and 31B on the 3050 with part of each model on the CPU. This part takes the 12B down to the 4 GB laptop card.

https://github.com/xbill9/gemma4-dev

**The 12B runs on the 4 GB card, too slowly for a conversation.** Twenty of its 48 layers fit on the card and the CPU runs the other 28:

| Gemma 4 12B, exact 4-bit file, drafter | GTX 1650 Ti, 4 GB laptop | RTX 3050 OEM, 8 GB desktop |
| :--- | ---: | ---: |
| Layers on the card | 20 of 48 | 48 of 48 |
| Context as served | 32,768 tokens | 81,920 tokens |
| Generation, nothing held | **8.5 tok/s** | 56.4 tok/s |
| Prompt reading | **about 60 tok/s** | 987 tok/s |
| Reading a 4,096-token document | **100 s** | |

A short question gets a 128-token reply in 15 to 19 seconds, and pasting in a 16,384-token document takes 5 minutes before the first word. On this card E4B with its drafter generates at 55.1 tok/s, and the 12B belongs on the 8 GB card.

---

#### What Is Being Measured?

Gemma 4 12B is a dense model: 48 layers of about 133 MiB, 6.24 GiB in all as the exact 4-bit (Q4_0) rebuild from Part 3. Every token reads every layer, so the layers that run on the CPU set the generation speed.

llama.cpp's `-ngl N` puts the first N of the 48 layers on the GPU and runs the rest on the CPU. The card holds 20 at the serving setting, so 28 run on the CPU for every token.

The host is a Lenovo Yoga 9 15IMH5 laptop: an i7-10750H (6 cores, 12 threads), 15 GiB of RAM, and a GTX 1650 Ti with Max-Q Design (Turing, 4 GB, no tensor cores). Every setting is measured on one llama.cpp build, `7182ef71e`, the same build as the 3050 articles, with one request at a time (`--parallel 1`). The 12B file is `xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf` and the drafter `xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf`, the same files the 3050 serves.

---

#### Step 1 — Serve the Rig

The rig is `local-llamacpp-1650ti-12b-q4_0`. Its `tpu.env` holds every serving setting, and `make serve` starts `llama-server` from it:

```
git clone https://github.com/xbill9/gemma4-dev
cd gemma4-dev/local-llamacpp-1650ti-12b-q4_0
make download draft
make serve
```

The setting this article arrives at:

```
llama-server -m gemma-4-12B-it-q4_0-exact.gguf -ngl 20 -c 32768 -ctk q4_0 -ctv q4_0 -fa 1 \
  -t 6 -tb 12 --parallel 1 --reasoning off --metrics \
  -md gemma-4-12B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 3
```

`-ngl 20` puts 20 of the 48 layers on the card; the drafter proposes 3 tokens at a time. It loads in 3,625 MiB of the card's 4,096.

---

#### Step 2 — Find How Many Layers the Card Can Hold

`llama-bench` measures generation (64 tokens) and prompt reading (512 tokens) with 0 to 25 layers on the card, `-t 6`:

| layers on the card, of 48 | 0 | 12 | 16 | 20 | 21 | 24 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generation, tok/s | 3.63 | 4.72 | 5.06 | 5.43 | 5.64 | 5.98 |
| Prompt reading, tok/s | 58.1 | 59.6 | 60.1 | 60.7 | 60.8 | 52.2 |

**Each layer moved to the card adds about 0.1 tok/s.** With all 48 on the CPU it generates 3.63 tok/s; 21 layers give 1.55x that. 24 layers load only with a smaller prompt batch (`-ub 128`), which reads prompts 15% slower, and 25 do not load at all.

**Prompt reading stays near 60 tok/s whatever is on the card.** A 2,048-token prompt read at 56.6, 59.4 and 60.5 tok/s with prompt batches of 512, 1,024 and 2,048, so a larger batch buys nothing here and only costs layers.

Google's own 12B file runs at the same speed with the same layers on the card (5.06 against 5.09 tok/s at 16). The exact rebuild is 0.24 GiB smaller, which buys one or two more layers.

Threads, 21 layers on the card, two passes each: `-t 4` 4.90 / 4.88, `-t 6` 5.03 / 4.66, `-t 8` 4.91 / 4.97 and `-t 12` 4.05 / 3.94 tok/s. 4 to 8 are level and 12 is slower; the rig serves `-t 6`.

---

#### 🔎 Tip: On a Laptop, Compare Settings in Rotated Rounds

A laptop's CPU and GPU share one power and cooling budget. During these runs the GPU clock read anywhere from 1,035 to 1,875 MHz under load, and generation at one setting ran 11 to 17% slower while a browser tab kept a CPU core busy. In a single pass through the settings in a fixed order, the same no-drafter setting read 4.6 tok/s at the start and 5.3 at the end. The drafter results below come from three rounds with the order rotated each round, so every setting meets the same conditions, and each pass starts with the CPU package below 55 C.

---

#### Step 3 — Trade Layers for Context

Every layer on the card costs memory the context cache could use. Each cell is the most layers that loaded at that context, one request at a time:

| context | 4-bit cache + drafter | 4-bit cache | 16-bit cache + drafter | 16-bit cache |
| :--- | ---: | ---: | ---: | ---: |
| 8,192 | 22 | 24 | 20 | 23 |
| 32,768 | **20** | 23 | 19 | 21 |
| 131,072 | 14 | 19 | | |

The cache for the layers on the CPU stays in system RAM, so a longer context costs only a few layers, and at 0.1 tok/s a layer, little speed. The full 131,072 tokens load with the drafter at 14 layers. The rig serves 32,768, because at 60 tok/s of prompt reading a document that fills it already takes minutes to read.

---

#### Step 4 — Measure the Drafter

Google's QAT drafter for the 12B is a small model that proposes the next tokens for the 12B to check; it is stored as exact 4-bit, 253.7 MB. `spec_ab_rounds.py` runs six prompts, 128 tokens each, three rotated rounds, 20 layers on the card and 32,768 tokens of 4-bit cache:

| drafted tokens | none | 1 | 2 | 3 | none, 23 layers |
| :--- | ---: | ---: | ---: | ---: | ---: |
| tok/s | 5.4 | 8.1 | 6.2 | **8.5** | 5.9 |
| x none | 1.00x | 1.50x | 1.15x | **1.56x** | 1.09x |
| drafted tokens kept | — | 0.76 | 0.66 | 0.52 | — |

**3 drafted tokens give 1.56x, and lead 1 token in all three rounds.** Code gains most (1.97x) and story least (1.35x). The drafter costs three layers on the card, worth only 1.09x, so it pays for itself many times over. 2 drafted tokens fall below both 1 and 3 on every prompt, as they did for the 31B on the 3050.

---

#### Step 5 — Hold a Long Conversation

`depth.py` puts part of a long document in front of three questions, with the prompt cache on, so the first question reads the document and the next two reuse it:

| tokens held | drafter, story / code | no drafter, 23 layers | reading the document |
| :--- | ---: | ---: | ---: |
| 0 | 7.4 / 10.1 tok/s | 5.6 tok/s | |
| 4,096 | 5.8 / 9.3 | 4.9 | 100 s |
| 16,384 | 4.1 / 6.5 | 4.3 | 297 s |
| 30,000 | 3.2 / 5.0 | 3.6 | 397 s |

**A long conversation slows everything, and the drafter's lead shrinks with it.** From 16,384 tokens the drafter is behind on story, level on the summary and ahead only on code. Prompt reading falls from 60 to 38 tok/s as the document grows, so a full 30,000-token document takes six and a half minutes before the first word.

---

#### Step 6 — Check the Answers

ANSWERS-PENDING

---

#### Compare and Contrast

| Gemma on a 4 GB GTX 1650 Ti laptop | 🥇 E4B speed | 🥈 12B answers |
| :--- | ---: | ---: |
| Layers on the CPU | none | 28 of 48 |
| Generation with drafter, short prompts | 55.1 tok/s | 8.5 tok/s |
| GSM8K / BFCL simple, on the 3050 | 12B leads by 3.0 and 3.75 points | |
| A 128-token reply | a few seconds | 15 to 19 s |

---

#### So, Which One?

For a conversation on this laptop, E4B with its drafter, at 55.1 tok/s. It answers about 3 points behind the 12B on GSM8K and BFCL (measured on the 3050 in Part 3) and generates 6.5x faster here.

For the 12B's answers on this laptop, run it as a batch: a queue of short questions left running, at 7 to 10 tok/s with the drafter at 3 tokens. Keep prompts short; at about 60 tok/s every 1,000 tokens of prompt takes about 17 seconds to read.

For the 12B in a conversation, the 8 GB RTX 3050 from Parts 3 and 4, at 56.4 tok/s with all 48 layers on the card.

---

#### Summary

The goal of this article was to find whether Gemma 4 12B, a model larger than the card, can serve from a 4 GB GTX 1650 Ti laptop GPU, and how usable it is. The key to the solution was putting 20 layers on the card, running the other 28 on the CPU, and adding the drafter at 3 tokens. The results were:

- 🟢 The 6.7 GB 12B serves from a 4 GB laptop card with 28 of its 48 layers on the CPU, in 3,625 MiB
- 🟢 The drafter gives 1.56x at 3 tokens, 8.5 tok/s against 5.4, far more than the three layers it costs
- 🟢 Context costs only a few layers: 131,072 tokens load with the drafter at 14 layers
- ANSWERS-SUMMARY
- ⚠️ By 30,000 tokens held generation slows by a third without the drafter and by half with it, and from 16,384 the drafter leads only on code
- ❌ Prompt reading stays near 60 tok/s on every setting: 100 s for a 4,096-token document, 5 minutes for 16,384
- ❌ At 7 to 10 tok/s it is too slow for a conversation; E4B on the same card generates 6.5x faster

Scope: Gemma 4 12B-it QAT only, the exact Q4_0 GGUF and drafter, text only, one GTX 1650 Ti Max-Q (4 GB) in an i7-10750H laptop, llama.cpp `7182ef71e`, 4-bit context cache, greedy decoding, thinking off, one request at a time. The 3050 figures are from Parts 3 and 4 (same files and build, a different host); the E4B figure is from Part 2's runs on this card on llama.cpp `77c4540ce` with 256-token replies. The laptop's clocks move with its load, so absolute speeds vary by about 10%; settings were compared in rotated rounds under the same conditions.

The strategy for using MCP for local accelerator comparison was validated with an incremental step by step approach.

---

#### References

- Part 1, Gemma 4's MTP Drafter, Explained: https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0
- Part 2, Do Drafters Work on Small Models?: https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g
- Part 3, Can a QAT 12B Fit an 8 GB GPU?: https://dev.to/gde/can-a-qat-12b-fit-an-8-gb-gpu-gemma-4-12b-on-an-rtx-3050-1267
- Part 4, How Much Context Fits Beside a 12B on an 8 GB GPU?: https://dev.to/gde/how-much-context-fits-beside-a-12b-on-an-8-gb-gpu-gemma-4-12bs-context-cache-on-an-rtx-3050-20im
- Part 5, Can a 26B MoE Run on an 8 GB GPU?: https://dev.to/gde/can-a-26b-moe-run-on-an-8-gb-gpu-gemma-4-26b-a4b-on-an-rtx-3050-2efn
- Part 6, A Dense 31B on an 8 GB GPU: https://dev.to/gde/a-dense-31b-on-an-8-gb-gpu-gemma-4-31b-on-an-rtx-3050-26p9
- Repository: https://github.com/xbill9/gemma4-dev
- 12B laptop rig: https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-12b-q4_0
- First light run: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-1650ti-12b-q4_0/benchmarks/runs/2026-10-21-first-light-1650ti/REPORT.md
- Answers run: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-1650ti-12b-q4_0/benchmarks/runs/2026-10-21-accuracy-12b-1650ti/REPORT.md
- 12B GGUF: https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf
- 12B drafter: https://huggingface.co/xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf
- llama.cpp: https://github.com/ggml-org/llama.cpp
