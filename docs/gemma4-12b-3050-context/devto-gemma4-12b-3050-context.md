---
title: "How Much Context Fits Beside a 12B on an 8 GB GPU? Gemma 4 12B's Context Cache on an RTX 3050"
published: false
description: "A 4-bit context cache lets Gemma 4 12B hold 81,920 tokens beside its drafter on an 8 GB RTX 3050, 3.3x the 16-bit cache, with the same GSM8K, BFCL and long-document recall scores, at 55 tok/s empty and 31 tok/s with 65,536 tokens held."
tags: gemma, llamacpp, cuda, benchmarking
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-12b-3050-context/devto-cover.abc59f31.jpg
---

This article provides a step by step guide to choosing the context cache, context size and drafter settings for Gemma 4 12B, Google's quantization-aware trained (QAT) build, on an 8 GB NVIDIA RTX 3050 with llama.cpp. The target workload is one long agentic session, a coding assistant whose conversation grows turn by turn, so the measure is how much context fits, how fast the model generates with that context held, and whether the answers change. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

This is the fourth part of a short series. [Part 1, Gemma 4's MTP Drafter, Explained](https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0), covers how the drafter works. [Part 2, Do Drafters Work on Small Models?](https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g), measures E2B and E4B on this card. Part 3, Can a QAT 12B Fit an 8 GB GPU?, fits the 12B and its drafter on the card with a 16-bit cache. This part makes room for a long conversation.

https://github.com/xbill9/gemma4-dev

**How much context fits beside the 12B and its drafter? 81,920 tokens, with a 4-bit context cache**, and the answers stay the same:

| RTX 3050 OEM, 8 GB, 12B + drafter | 16-bit cache | 4-bit cache |
| :--- | ---: | ---: |
| Most context that loads | 24,576 tokens | **81,920 tokens** |
| Generation, nothing held | 57.5 tok/s | **55.1 tok/s** |
| Generation, 16,384 tokens held | 49.8 tok/s | **42.4 tok/s** |
| Generation, 65,536 tokens held | does not fit | **30.8 tok/s** |
| GSM8K / BFCL simple | 0.960 / 0.950 | **0.961 / 0.950** |

The 4-bit cache holds 3.3x the tokens for 4% of the generation speed with nothing held and 15% at 16,384 tokens. Without the drafter the same cache holds the model's full 131,072 tokens at about half the speed.

---

#### What Is Being Measured?

Gemma 4 12B's weights take 6,390 MiB of the card's 8,192 MiB in the exact Q4_0 rebuild from Part 3. Everything else shares what is left: the context cache (the keys and values of every token already read), the drafter (Google's 12B multi-token prediction assistant, 227 MiB of weights), and working buffers.

The context cache is the part that grows with the conversation. llama.cpp stores it at 16 bits by default and can store it at 8 bits (`q8_0`) or 4 bits (`q4_0`), which halves and quarters what each token costs. The questions are what a smaller cache costs in speed and in answers, and whether the drafter is still worth its memory once the context is long.

Every setting is measured on one llama.cpp build, `7182ef71e`, with the same model and drafter files and one request at a time (`--parallel 1`), because the workload is one conversation.

---

#### Step 1 — Serve the Rig

The rig is `local-llamacpp-3050oem-12b-q4_0`. Its `tpu.env` holds every serving setting, and `make serve` starts `llama-server` from it:

```
git clone https://github.com/xbill9/gemma4-dev
cd gemma4-dev/local-llamacpp-3050oem-12b-q4_0
make download draft
make serve
```

The setting this article arrives at:

```
llama-server -m gemma-4-12B-it-q4_0-exact.gguf -ngl 99 -c 81920 -ctk q4_0 -ctv q4_0 -fa 1 \
  -t 6 -tb 12 --parallel 1 --reasoning off --metrics \
  -md gemma-4-12B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

---

#### Step 2 — Find the Most Context That Fits

`kv_sweep.py` starts `llama-server` with each cache type, with and without the drafter, at falling context sizes, and keeps the largest that loads:

```
python3 kv_sweep.py ../../.. .
```

| RTX 3050, 8,192 MiB | with the drafter | without |
| :--- | ---: | ---: |
| 16-bit cache | 24,576 tokens | 40,960 tokens |
| 8-bit cache | 49,152 tokens | 65,536 tokens |
| 4-bit cache | **81,920 tokens**, 7,853 MiB | **131,072 tokens** |

With the drafter, 86,016 tokens fail to load. 131,072 is the model's own limit, so without the drafter the 4-bit cache holds all of it.

---

#### Step 3 — Measure Speed With a Long Conversation Held

The same script puts 0 to 98,304 tokens of a long document (the NVIDIA driver README) ahead of three questions (a story, a Python function, a summary) and measures 256 greedy tokens of generation, with the prompt cache on as a conversation would use it:

| tokens already in context | 0 | 16,384 | 32,768 | 65,536 | 98,304 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 16-bit cache, drafter | 57.5 tok/s | 49.8 | — | — | — |
| 8-bit cache, drafter | 55.1 | 43.3 | 39.3 | — | — |
| **4-bit cache, drafter** | **55.1** | **42.4** | **37.8** | **30.8** | — |
| 16-bit cache, no drafter | 28.1 | 25.7 | 24.8 | — | — |
| 4-bit cache, no drafter | 27.5 | 24.1 | 22.2 | 19.2 | 16.8 |

Each figure is the geometric mean of the three questions. Reading new text is level across cache types: about 840 tok/s for 8,192 tokens, 720 at 16,384, 600 at 32,768 and 408 at 65,536.

**The drafter keeps paying with a long conversation held**: with the 4-bit cache it is 2.00x faster with nothing held, 1.70x at 32,768 tokens and 1.60x at 65,536. It keeps 61% to 71% of drafted tokens at every depth and cache type.

---

#### 🔎 Tip: A Smaller Cache Costs the Drafter More

Without the drafter, the 4-bit cache generates at 0.94x the 16-bit cache with 16,384 tokens held. With the drafter, 0.85x. The drafter checks two drafted tokens in one pass over the whole cache, so a cache that is slower to read costs it more. The drafter still wins at every depth: the slowest drafter setting at 16,384 tokens (42.4 tok/s) is 1.65x the fastest setting without one (25.7, arithmetic).

---

#### Step 4 — Check the Answers

`gen_eval.py` runs GSM8K (1,319 grade-school math questions, up to 2,048 tokens per reply) and BFCL simple (400 single function calls), one request at a time, with each cache at its largest context:

```
JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 google/gemma-4-12B-it gsm8k gsm8k-q4x.jsonl --concurrency 1
```

| one request at a time | GSM8K | BFCL simple |
| :--- | ---: | ---: |
| 16-bit cache, drafter | 0.960 | 0.950 |
| 16-bit cache, no drafter | 0.963 | 0.948 |
| 8-bit cache, drafter | 0.960 | 0.948 |
| 8-bit cache, no drafter | 0.959 | 0.948 |
| **4-bit cache, drafter** | **0.961** | **0.950** |
| 4-bit cache, no drafter | 0.962 | 0.950 |

**The cache type changes no score.** `compare.py` pairs each run question by question against the 16-bit cache with the same drafter setting. For the 4-bit cache with the drafter the difference is +0.15 points on GSM8K (95% range −0.45 to +0.76) and zero on BFCL (−1.0 to +1.0), and every other range also includes zero. The time a run takes is set by the drafter: GSM8K took 1 h 57 min with the drafter and 4 h 20 min without, on the 4-bit cache.

---

#### Step 5 — Check Recall From a Long Document

GSM8K and BFCL questions are short. `recall.py` tests the long context itself: it plants eight lines of the form "The access code for project Heron is 601531." through the document, asks for every code as JSON, and scores the codes returned, ten trials per depth with the same codes for every setting:

| codes right, of 80 | 16,384 | 32,768 | 64,000 | 98,304 | 126,976 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 16-bit cache, no drafter | 69 | 55 | — | — | — |
| 8-bit cache, no drafter | 68 | 58 | 57 | — | — |
| 4-bit cache, no drafter | 67 | 58 | 59 | 48 | 56 |
| **4-bit cache, drafter** | — | **58** | **57** | — | — |

**The cache type and the drafter leave recall level** at every depth they share. Recall falls with depth on every cache, from about 85% of codes at 16,384 tokens to about 70% from 32,768 on.

---

#### Step 6 — Tune the Drafter and Prompt Batch

`tune.py` tries the remaining settings on the 4-bit cache at 81,920 tokens:

| 4-bit cache, drafter | 0 tokens held | 32,768 | 65,536 | drafted tokens kept |
| :--- | ---: | ---: | ---: | ---: |
| 1 drafted token | 42.9 tok/s | 30.9 | 27.3 | 0.72–0.77 |
| **2 drafted tokens** | **55.1** | **37.8** | **30.8** | 0.64–0.71 |
| 3 drafted tokens | 56.0 | 39.3 | 32.1 | 0.55–0.59 |

3 drafted tokens ran within 4% of 2, inside this card's spread from one server start to the next: two starts of the same setting read a 32,768-token prompt at 589 and 673 tok/s. 2 stays. A larger prompt batch (`-ub 1024` or `2048`) does not load beside the drafter, even at 65,536 tokens.

---

#### 🔎 Tip: Keep the Conversation in the Prompt Cache

Reading 65,536 tokens from scratch at about 406 tok/s takes about 160 seconds (arithmetic). In one conversation llama.cpp's prompt cache keeps what it has already read, so each turn reads only the new text. One request at a time (`--parallel 1`) keeps the whole 81,920 tokens for that one conversation, because llama.cpp splits the context across parallel slots.

---

#### Compare and Contrast

| RTX 3050 OEM, 8 GB, Gemma 4 12B | 16-bit cache + drafter | 🥇 4-bit cache + drafter | 4-bit cache, no drafter |
| :--- | ---: | ---: | ---: |
| Most context | 24,576 tokens | 81,920 tokens | 131,072 tokens |
| Generation, nothing held | 57.5 tok/s | 55.1 tok/s | 27.5 tok/s |
| Generation, 16,384 held | 49.8 tok/s | 42.4 tok/s | 24.1 tok/s |
| Generation, 65,536 held | — | 30.8 tok/s | 19.2 tok/s |
| GSM8K | 0.960 | 0.961 | 0.962 |
| BFCL simple | 0.950 | 0.950 | 0.950 |
| Codes recalled at 32,768 | — | 58 of 80 | 58 of 80 |

---

#### So, Which One?

For one long conversation, the 4-bit cache with the drafter at 81,920 tokens. It holds 3.3x the 16-bit cache's context with the same answers, and generates at 55 tok/s with nothing held and 31 tok/s at 65,536 tokens.

For a conversation that must go past 81,920 tokens, drop the drafter: the 4-bit cache then holds the full 131,072 tokens at about half the speed.

For short conversations only, the 16-bit cache with the drafter is the fastest setting, up to 24,576 tokens.

---

#### What Comes Next

The answer tests here use short questions, and the recall test is the only one with a long context held. The next measurements are function calling with 32,768 to 81,920 tokens of conversation ahead of each call, a replayed multi-turn coding session that measures each turn's speed and how much of the prompt cache it reuses, and repeated server starts to resolve the 2-against-3 drafted-token gap.

---

#### Summary

The goal of this article was to find the context cache, context size and drafter settings that serve Gemma 4 12B best for one long conversation on an 8 GB RTX 3050. The key to the solution was measuring speed with a long conversation already held and checking answers question by question for each cache type. The results were:

- 🟢 A 4-bit context cache holds 81,920 tokens beside the 12B and its drafter, 3.3x the 16-bit cache's 24,576
- 🟢 GSM8K, BFCL and long-document recall are level across the 16-bit, 8-bit and 4-bit caches, with and without the drafter
- 🟢 The drafter is still 1.60x faster with 65,536 tokens held
- ⚠️ With the drafter, the 4-bit cache generates at 0.85x the 16-bit cache once 16,384 tokens are held
- ⚠️ The full 131,072-token context fits only without the drafter, at about half the speed
- ❌ Recall of eight facts planted in a long document falls from about 85% at 16,384 tokens to about 70% from 32,768 on, on every cache

Scope: Gemma 4 12B-it QAT only, exact Q4_0 rebuild and exact Q4_0 12B drafter, text only, one RTX 3050 OEM (8 GB) in an i7-5820K desktop, llama.cpp `7182ef71e`, 16-bit, 8-bit and 4-bit context caches, 0 to 98,304 tokens of one document held for speed and up to 126,976 for recall, greedy decoding, thinking off, GSM8K and BFCL simple with short prompts, one request at a time throughout.

The strategy for using MCP for local accelerator comparison was validated with an incremental step by step approach.

---

#### References

- Part 1, Gemma 4's MTP Drafter, Explained: https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0
- Part 2, Do Drafters Work on Small Models?: https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g
- Repository: https://github.com/xbill9/gemma4-dev
- 12B rig: https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-12b-q4_0
- The chosen settings, trade-offs and open questions: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/SERVING-CONFIG.md
- Context cache run: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem/REPORT.md
- 16-bit cache answers: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-accuracy-12b-3050oem/REPORT.md
- 12B GGUF: https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf
- 12B drafter: https://huggingface.co/xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf
- llama.cpp: https://github.com/ggml-org/llama.cpp
