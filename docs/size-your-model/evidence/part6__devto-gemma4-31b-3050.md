---
title: "A Dense 31B on an 8 GB GPU: Gemma 4 31B on an RTX 3050"
published: false
description: "Gemma 4 31B, a dense 17.7 GB model, serves from an 8 GB RTX 3050 with 40 of its 60 layers in system RAM: the highest GSM8K and BFCL scores of the family, at about 4 tok/s and 11x the 12B's GSM8K time."
tags: gemma, llamacpp, cuda, benchmarking
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-31b-3050/devto-cover.b440d4e2.jpg
---

This article provides a step by step guide to serving Gemma 4 31B, Google's quantization-aware trained (QAT) dense model, on an 8 GB NVIDIA RTX 3050 with llama.cpp. The model file is 17.7 GB, so it cannot fit on the card. llama.cpp puts as many layers as will fit on the GPU and runs the rest on the CPU. The measures are how fast it generates, how much context fits, what the drafter adds, and whether the answers beat the 26B and 12B sisters that came before it. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

This is the sixth part of a short series. [Part 1, Gemma 4's MTP Drafter, Explained](https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0), covers how the drafter works. [Part 2, Do Drafters Work on Small Models?](https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g), measures E2B and E4B on this card. [Part 3, Can a QAT 12B Fit an 8 GB GPU?](https://dev.to/gde/can-a-qat-12b-fit-an-8-gb-gpu-gemma-4-12b-on-an-rtx-3050-1267), fits the 12B and its drafter on the card. [Part 4, How Much Context Fits Beside a 12B on an 8 GB GPU?](https://dev.to/gde/how-much-context-fits-beside-a-12b-on-an-8-gb-gpu-gemma-4-12bs-context-cache-on-an-rtx-3050-20im), gives the 12B a 4-bit context cache. [Part 5, Can a 26B MoE Run on an 8 GB GPU?](https://dev.to/gde/can-a-26b-moe-run-on-an-8-gb-gpu-gemma-4-26b-a4b-on-an-rtx-3050-2efn), runs the 26B mixture-of-experts with its experts in system RAM. This part runs the dense 31B, the largest and most accurate of the family.

https://github.com/xbill9/gemma4-dev

**The 31B is the most accurate Gemma 4 that serves on this card, and the slowest.** It scores highest on both tests and takes 11x the 12B's GSM8K time:

| RTX 3050 OEM, 8 GB, 4-bit cache, drafter | Gemma 4 12B | Gemma 4 26B A4B | Gemma 4 31B |
| :--- | ---: | ---: | ---: |
| Model file | 6.7 GB, all on the card | 14.4 GB, experts mostly in RAM | 17.7 GB, 40 of 60 layers in RAM |
| Context as served | 81,920 tokens | 131,072 tokens | 32,768 tokens |
| Generation, nothing held | 55.1 tok/s | 33.7 tok/s | **4.14 tok/s** |
| GSM8K / BFCL simple | 0.961 / 0.950 | 0.965 / 0.945 | **0.974 / 0.960** |
| GSM8K run, 1,319 questions | 1 h 57 min | 3 h 26 min | **21 h 57 min** |

The 31B leads the 12B on GSM8K by 1.2 points and the 26B by 0.8, both beyond the run-to-run range, while the 12B generates 13x faster. At about 4 tok/s it is a batch model on this card: a 200-token reply takes most of a minute, so it earns its place on an offline run of questions and the 12B serves a conversation.

---

#### What Is Being Measured?

Gemma 4 31B is a dense model: 60 layers of 257 to 287 MiB, 15,716 MiB in all, plus a 1,102 MiB token embedding. Every token reads every layer, so the layers that run on the CPU set the speed. The 26B of Part 5 is a mixture-of-experts model that reads only 8 of its 128 experts per layer, so moving its experts to the CPU costs little; the dense 31B has no such shortcut.

llama.cpp's `-ngl N` puts the first N of the 60 layers on the GPU and runs the rest on the CPU. The card holds 20 at the serving setting, so 40 run on the CPU, and every token crosses all 40.

The host is an i7-5820K desktop (6 cores, 12 threads, quad-channel DDR4, 46 GiB) with the RTX 3050 OEM (8 GB). Every setting is measured on one llama.cpp build, `7182ef71e`, with Google's `google/gemma-4-31B-it-qat-q4_0-gguf` unmodified, and one request at a time (`--parallel 1`).

---

#### Step 1 — Serve the Rig

The rig is `local-llamacpp-3050oem-31b-q4_0`. Its `tpu.env` holds every serving setting, and `make serve` starts `llama-server` from it:

```
git clone https://github.com/xbill9/gemma4-dev
cd gemma4-dev/local-llamacpp-3050oem-31b-q4_0
make download draft
make serve
```

The setting this article arrives at:

```
llama-server -m gemma-4-31B_q4_0-it.gguf -ngl 20 -c 32768 -ctk q4_0 -ctv q4_0 -fa 1 \
  -t 8 -tb 12 --parallel 1 --reasoning off --metrics \
  -md gemma-4-31B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 1
```

`-ngl 20` puts 20 of the 60 layers on the card; the drafter runs at 1 token. The check reply through `make serve` generated at 4.14 tok/s in 7,734 MiB.

---

#### Step 2 — Find How Many Layers the Card Can Hold

`llama-bench` measures generation (64 tokens) and prompt reading (512 tokens) with 0 to 24 layers on the card, `-t 6`:

| layers on the card, of 60 | 0 | 12 | 16 | 18 | 20 | 22 | 24 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Generation, tok/s | 1.68 | 2.13 | 2.30 | 2.39 | 2.52 | 2.62 | does not load |
| Prompt reading, tok/s | 116 | 133 | 141 | 145 | 149 | 154 | |

**Each layer moved to the card adds about 0.05 tok/s**, a tenth of what an expert layer adds on the 26B, because the dense model reads every layer for every token. With all 60 on the CPU it generates 1.68 tok/s, rising to 2.52 at 20 layers and 2.62 at 22; at 24 the working buffer no longer fits. The rig serves 20 layers, which leaves room for 32,768 tokens of 4-bit cache with the drafter (Step 3). Threads at 22 layers: 4 / 6 / 8 / 12 give 2.09 / 2.62 / 2.66 / 2.63 tok/s, so the rig serves `-t 8`.

---

#### 🔎 Tip: The Active Parameter Count Sets the Speed

The 26B A4B holds 26.5B parameters against the 31B's 31B, but reads only about 4B of them per token, so with its experts in system RAM it generates at 24 tok/s against the dense 31B's 1.68 with every layer in RAM. Moving work to the CPU is cheap for a sparse model and expensive for a dense one. For a model larger than the card, the count that decides the speed is how many parameters each token reads, not how many the file holds.

---

#### Step 3 — Trade Layers for Context

Every layer on the card costs memory the context cache could use. Each row is the largest context that loaded with that many layers on the card, one request at a time, 4-bit cache:

| layers on the card | with drafter | no drafter |
| :--- | ---: | ---: |
| 20 | 32,768 (7,734 MiB) | 65,536 (7,752 MiB) |
| 16 | 65,536 (7,448 MiB) | 131,072 (7,682 MiB) |
| 12 | 131,072 (7,810 MiB) | |
| 10 | | 262,144 (7,802 MiB) |

The dense weights take most of the card, so context and speed trade against each other. The rig serves 20 layers at 32,768 tokens for the generation speed; dropping to 12 holds 131,072 tokens with the drafter, matching the 26B; at 12 layers llama-bench generates 2.13 tok/s without the drafter, and generation with it is not measured. Without the drafter, 10 layers on the card hold the model's full 262,144 tokens.

---

#### Step 4 — Measure the Drafter

Google ships a QAT drafter for the 31B, a small model that proposes the next tokens for the 31B to check. It is stored as exact 4-bit (Q4_0) on its trained grid, 280 MB against 515 MB for the converter's 8-bit file, and published as `xbill9/gemma-4-31B-it-qat-assistant-q4_0-exact-gguf`. `spec_ab.py` runs six prompts, 128 tokens each, one run per setting, with 20 layers on the card:

| drafted tokens | none | 1 | 2 | 3 | 4 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| tok/s | 2.5 | **4.0** | 3.1 | 3.5 | 2.7 |
| x none | 1.00x | **1.59x** | 1.21x | 1.37x | 1.09x |
| drafted tokens kept | — | 0.76 | 0.66 | 0.56 | 0.49 |

**1 drafted token gives 1.59x, and longer drafts give less.** The 31B keeps three quarters of a single drafted token, and 1.47x to 1.71x on every prompt, facts and code most and the summary least. Checking several drafted tokens at once reads the 40 CPU-side layers for each one, which costs more than it saves, so 2, 3 and 4 tokens all gain less than 1 token does (1.09x to 1.37x). The rig serves 1 token.

---

#### Step 5 — Check the Answers

`gen_eval.py` runs GSM8K (1,319 grade-school math questions, up to 2,048 tokens per reply) and BFCL simple (400 single function calls), one request at a time, at the serving setting:

```
JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 google/gemma-4-31B-it gsm8k gsm8k-q4x.jsonl --concurrency 1
```

Each sister's score below is on the same questions the 31B answered, so the rows compare like for like:

| as served, 4-bit cache, drafter | GSM8K | BFCL simple |
| :--- | ---: | ---: |
| **31B** | **0.974** | **0.960** |
| 26B A4B | 0.965 | 0.945 |
| 12B | 0.961 | 0.950 |

**The 31B is the family's most accurate on both tests.** `compare.py` pairs the runs question by question with a 95% bootstrap range. On GSM8K the 31B leads the 26B by 0.83 points (range +0.08 to +1.67) and the 12B by 1.21 (+0.45 to +2.05), both ranges above zero. On BFCL simple it leads the 26B by 1.50 points (−0.50 to +3.50) and the 12B by 1.00 (−0.75 to +2.75), both ranges spanning zero, so the function-calling lead is within the run-to-run range.

The drafter changes one answer in 663, for the better. On GSM8K the drafter and no-drafter arms agree on every one of 263 paired questions, with 96 of them word for word the same and the rest reworded to the same answer. On BFCL the two arms differ in answer on 1 of 400 replies, a wrong answer the drafter gets right, with 397 word for word the same.

The time is where the dense model pays. GSM8K took the 31B 79,000 s with the drafter, against 12,346 s for the 26B and 7,032 s for the 12B, both with their own drafters.

---

#### Compare and Contrast

| RTX 3050 OEM, 8 GB, 4-bit cache, drafter | 🥇 31B accuracy | 🥈 26B context | 🥉 12B speed |
| :--- | ---: | ---: | ---: |
| Layers / experts on the CPU | 40 of 60 layers | 21 of 30 expert layers | — |
| Context as served | 32,768 tokens | 131,072 tokens | 81,920 tokens |
| Generation, nothing held | 4.14 tok/s | 33.7 tok/s | 55.1 tok/s |
| GSM8K | 0.974 | 0.965 | 0.961 |
| BFCL simple | 0.960 | 0.945 | 0.950 |
| GSM8K run | 21 h 57 min | 3 h 26 min | 1 h 57 min |

---

#### So, Which One?

For the best answers this card can give, run as an offline batch, the 31B, at 32,768 tokens and about 4 tok/s with the drafter. It wins GSM8K by a margin beyond the run-to-run range and leads function calling within it, at a speed that suits a queue of questions left to run overnight.

For a long conversation at a usable speed, the 12B with its drafter, at 55.1 tok/s and 81,920 tokens. It answers 1.2 points behind the 31B on GSM8K and 1.0 on BFCL, and generates 13x faster.

For context between the 12B's 81,920 tokens and the full 262,144, the 26B with its experts in RAM, which holds 131,072 tokens with its drafter and generates at 33.7 tok/s.

---

#### What Comes Next

The answer tests here use short questions and one request at a time. The next measurements are function calling with a long conversation held across the three models, and the 26B's 4-bit cache against the 16-bit cache for answers.

---

#### Summary

The goal of this article was to find whether Gemma 4 31B, a dense model more than twice the card's size, can serve from an 8 GB RTX 3050, and how its answers compare with the 26B and 12B sisters. The key to the solution was putting 20 layers on the card and running the other 40 on the CPU. The results were:

- 🟢 The 17.7 GB dense 31B serves from an 8 GB card with 40 of its 60 layers in system RAM
- 🟢 GSM8K 0.974, the highest of the family, ahead of the 26B by 0.83 points and the 12B by 1.21, both beyond the run-to-run range
- 🟢 BFCL simple 0.960, the highest of the family, with a lead within the run-to-run range
- 🟢 The drafter changes one answer in 663, a wrong BFCL reply it gets right; all 263 paired GSM8K questions agree
- 🟢 It holds 131,072 tokens with the drafter at 12 layers on the card, and the full 262,144 without it at 10
- ⚠️ 1 drafted token gives 1.59x; longer drafts read the CPU-side layers too many times and give less
- ❌ At about 4 tok/s the 12B generates 13x faster, and a GSM8K run takes 11x the 12B's time

Scope: Gemma 4 31B-it QAT only, Google's Q4_0 GGUF unmodified with the exact Q4_0 31B drafter, text only, one RTX 3050 OEM (8 GB) in an i7-5820K desktop, llama.cpp `7182ef71e`, 4-bit context cache, greedy decoding, thinking off, GSM8K and BFCL simple with short prompts, one request at a time throughout. The sister figures are from Parts 4 and 5 on the same card and build; the GSM8K-without-drafter arm is a 263-question check.

The strategy for using MCP for local accelerator comparison was validated with an incremental step by step approach.

---

#### References

- Part 1, Gemma 4's MTP Drafter, Explained: https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0
- Part 2, Do Drafters Work on Small Models?: https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g
- Part 3, Can a QAT 12B Fit an 8 GB GPU?: https://dev.to/gde/can-a-qat-12b-fit-an-8-gb-gpu-gemma-4-12b-on-an-rtx-3050-1267
- Part 4, How Much Context Fits Beside a 12B on an 8 GB GPU?: https://dev.to/gde/how-much-context-fits-beside-a-12b-on-an-8-gb-gpu-gemma-4-12bs-context-cache-on-an-rtx-3050-20im
- Part 5, Can a 26B MoE Run on an 8 GB GPU?: https://dev.to/gde/can-a-26b-moe-run-on-an-8-gb-gpu-gemma-4-26b-a4b-on-an-rtx-3050-2efn
- Repository: https://github.com/xbill9/gemma4-dev
- 31B rig: https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-31b-q4_0
- First light run: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-31b-q4_0/benchmarks/runs/2026-10-18-first-light-3050oem/REPORT.md
- Answers run: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-31b-q4_0/benchmarks/runs/2026-10-19-accuracy-31b-3050oem/REPORT.md
- 31B GGUF: https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf
- 31B drafter: https://huggingface.co/xbill9/gemma-4-31B-it-qat-assistant-q4_0-exact-gguf
- llama.cpp: https://github.com/ggml-org/llama.cpp
