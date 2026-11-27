---
title: "Thinking on Constrained Deployments: Gemma E4B on 4GB 1650"
published: false
description: "Gemma 4 E4B fits a 4 GB laptop card and answers like an 8 GB desktop card in 15 of 16 settings, at 1.59 times the time. The drafter nearly doubles its speed; thinking doubles its time."
tags: gemma, llm, benchmark, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-e4b-thinking-1650ti/devto-cover.6acaec69.jpg
---

Gemma 4 E4B fits a 4 GB laptop card and answers like an 8 GB desktop card in 15 of 16 settings, at 1.59 times the time. The drafter nearly doubles its speed; thinking doubles its time.

The same 16 settings that ran on an 8 GB RTX 3050 ran on a GTX 1650 Ti Max-Q, through the same llama.cpp build and the same three files by SHA-256. GSM8K scored 0.9265 to 0.9416 here against 0.9272 to 0.9439 on the 3050, and AIME 11 to 13 of 30 against 9 to 13. In 15 settings the two cards differ by −11 to +9 GSM8K problems, none beyond chance (smallest p = 0.0708). One setting, Google's file with the drafter and thinking on, scored 18 problems below the 3050 (p = 0.0029).

The drafter lifted generation 1.88 to 1.90 times on GSM8K for 102 MiB of card memory. Thinking on made GSM8K take 2.13 to 2.24 times as long and moved no score beyond chance once the 24 tests on this card are counted; its largest change, +20 problems, is p = 0.0105 alone and 0.24 after correction. The exact Q4_0 repack scored level with Google's file, in 204 MiB less card memory and 4.2 to 5.7 percent faster.

For a 4 GB card: serve E4B with the repack weights and the drafter, leave thinking off, and expect about two thirds of the 3050's speed without the drafter and just over half with it. Each setting ran once, at temperature 0.

This article provides a step by step guide to fitting Gemma 4 E4B on a 4 GB laptop GPU and checking its answers against a bigger card: thinking off and on, two 4-bit builds of Google's QAT weights, each with and without a drafter, on every GSM8K test problem and all 30 AIME 2025 problems.

https://github.com/xbill9/gemma4-dev/tree/main/docs/gemma4-e4b-thinking-1650ti

---

#### The Findings

**Same answers as the 8 GB card, in 15 of 16 settings.** Setting by setting, the two cards differ by −11 to +9 GSM8K problems and −1 to +3 AIME problems in those 15, with no paired test below p = 0.0708.

**One setting stands apart.** Google's file with the drafter and thinking on scored 1,227 of 1,319 here against 1,245 on the 3050: 8 problems right only here, 26 only there, p = 0.0029. On this card the same setting without the drafter scored 1,242 (p = 0.0059). Across 16 cross-card tests a p this low clears a per-test bar of 0.05 / 16 = 0.0031; one run of each setting cannot say whether it repeats.

**Different text, same verdicts.** 318 to 353 of the 1,319 GSM8K replies with thinking off came out byte-identical on the two cards, and 27 to 40 with thinking on. The right-or-wrong verdict matched on 1,285 to 1,299.

**What thinking cost.** GSM8K took 2.13 to 2.24 times as long with thinking on, and AIME 1.23 to 1.46 times. GSM8K moved by +1, +4, +11 and +20 problems across the four setups; the +20, Google's file without the drafter, is p = 0.0105 on its own and 0.24 after a Holm correction over the 24 tests on this card, and the others are p = 0.2000 or above.

**What the drafter bought.** Generation rose 1.88 to 1.90 times on GSM8K, from 36.1–38.1 to 67.9–72.3 tokens per second, against 2.23 to 2.29 times on the RTX 3050 at the same acceptance. In one setting it scored lower: Google's file with thinking on fell from 1,242 to 1,227, p = 0.0059 alone and 0.14 after correction over the 24 tests on this card. That is the same setting that scored low against the 3050.

**The two weight builds.** The exact Q4_0 repack and Google's QAT Q4_0 file scored level in every paired test. The repack used 204 MiB less card memory and generated 4.2 to 5.7 percent faster.

**What to do with it.**

- 🟢 A 4 GB laptop card answers like the 8 GB desktop card for E4B. Pick the card by time.
- 🟢 Serve E4B with the drafter at two drafted tokens; it adds 102 MiB and 1.88 to 1.90 times the speed.
- ⚠️ Leave thinking off for grade-school math; it doubles the time on this card too.

**Caveats.** One run of each setting, temperature 0. AIME's 30 problems can only show large effects.

---

#### What Was Measured

The earlier articles, [Thinking On or Off? Gemma 4 E4B on an 8 GB RTX 3050](https://dev.to/gde/thinking-on-or-off-gemma-4-e4b-on-an-8-gb-rtx-3050-5d5) and [Thinking on Constrained Deployments: Gemma E2B on 4GB 1650](https://dev.to/gde/thinking-on-constrained-deployments-gemma-e2b-on-4gb-1650-487c), crossed thinking, weight build and drafter on the desktop card, then repeated it for E2B on the laptop. This run does the same for E4B, the larger model, on half the card memory, and asks: does the smaller card change any answer, and what does it cost in time?

| Factor | Settings |
| :--- | :--- |
| Thinking | off, on (`enable_thinking` in the chat template) |
| Weights | exact Q4_0 repack of Google's QAT weights, Google's QAT Q4_0 GGUF |
| Drafter | none, Google's E4B QAT assistant as exact Q4_0, 2 drafted tokens |
| Problems | GSM8K test, 1,319; AIME 2025 I and II, 30 |

Two settings are this card's own. The context is 36,864 tokens against the 3050's 131,072, the largest that loads with Google's file and the drafter; the longest prompt in the study is 323 tokens, so every reply keeps its full 32,768-token limit. And E4B's per-layer embedding table stays in system memory, this rig's serving setting.

That is 16 runs, 42.63 hours, every reply saved.

---

#### At This Point You Should Have…

- An NVIDIA card with 4 GB, here a GTX 1650 Ti with Max-Q Design on driver 620.70.10
- llama.cpp built with CUDA, here commit `7182ef7`, the build the 3050 run used
- Python 3 with `huggingface_hub`, `tokenizers` and `scipy`
- A clone of [gemma4-dev](https://github.com/xbill9/gemma4-dev)

---

#### Step 1 — Fit E4B in 4 GB

All three files are the ones the 3050 served, checked by SHA-256 in `weights.sha256`:

| File | Bytes | Card memory when serving |
| :--- | ---: | ---: |
| Repack, `gemma-4-E4B-it-q4_0-exact.gguf` | 4,215,695,712 | 🥇 3,355 MiB |
| Google, `gemma-4-E4B_q4_0-it.gguf` | 5,154,941,280 | 3,559 MiB |
| Drafter, `gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf` | 59,678,368 | +102 MiB |

A 4.2 GB file serves in 3,355 MiB because the 1.59 GB per-layer embedding table is read from system memory a row at a time. The largest arm, Google's file with the drafter, held 3,661 MiB of 4,096 at 36,864 tokens of context; at 40,960 it ran out of card memory at load.

---

#### Step 2 — Run All Sixteen on a Laptop

```bash
cd local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-11-25-thinking-e4b-1650ti
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why="E4B thinking run" \
  ./run.sh >> nohup.out 2>&1 < /dev/null &
```

`systemd-inhibit` keeps the laptop awake with the lid closed for the 43 hours. `run.sh` starts the server once per weights and drafter pair and runs four cells on it: GSM8K thinking off, AIME off, AIME on, GSM8K on. It saves each reply as it finishes, skips any cell already complete when restarted, and commits and pushes each pair as it ends.

The run's own log, `driver.log`, shows one line per cell, with the card's temperature and clock:

```
2026-11-25 15:05:29 64.56, 1860 MHz END gsm8k repack-none thinking=off rc=0 {"task": "gsm8k", "max_tokens": 32768, "n": 1319, "right": 1224, "accuracy": 0.928, "errors": 0, "truncated": 0, ...
```

---

#### Step 3 — The Scores, Beside the 3050

`analyze.py` builds every table from the saved replies.

**GSM8K, right of 1,319: 1650 Ti / RTX 3050**

| Weights | Drafter | Thinking off | Thinking on |
| :--- | :--- | ---: | ---: |
| Repack | none | 1,224 / 1,226 | 1,235 / 1,230 |
| Repack | yes | 1,232 / 1,223 | 1,236 / 1,236 |
| Google | none | 1,222 / 1,233 | 1,242 / 1,235 |
| Google | yes | 1,226 / 1,229 | ⚠️ 1,227 / 1,245 |

**AIME 2025, right of 30: 1650 Ti / RTX 3050**

| Weights | Drafter | Thinking off | Thinking on |
| :--- | :--- | ---: | ---: |
| Repack | none | 13 / 12 | 11 / 10 |
| Repack | yes | 13 / 10 | 11 / 12 |
| Google | none | 11 / 12 | 12 / 12 |
| Google | yes | 12 / 13 | 12 / 9 |

No reply on either card ran into the 32,768-token limit, and no request failed.

---

#### Step 4 — Same Problems, Two Cards

`agreement.py` lines up each problem's reply from the two cards in the same setting.

| GSM8K | Thinking off | Thinking on |
| :--- | ---: | ---: |
| Same right-or-wrong verdict | 1,288 to 1,299 | 1,285 to 1,299 |
| Byte-identical reply | 318 to 353 | 27 to 40 |
| Paired test, smallest p | 0.0708 | 0.0029 |

Same files, same llama.cpp commit, temperature 0, and most replies still differ in their wording. The likely cause is that the two GPUs add up floating-point numbers in a different order, so the greedy pick flips wherever two candidate tokens are nearly tied, and a longer reply has more such points. No AIME reply matched byte for byte.

The verdicts flip on 20 to 34 GSM8K problems per setting. In 15 settings they split close to evenly. In Google's file with the drafter and thinking on they split 8 against 26, and that same setting on the 3050 was its best GSM8K score of the eight.

---

#### Step 5 — Where the Tokens Go

`tokens.py` counts thinking and answer tokens in each saved reply with Gemma 4 E4B's tokenizer.

| | GSM8K off | GSM8K on | AIME off | AIME on |
| :--- | ---: | ---: | ---: | ---: |
| Thinking tokens, mean | 0 | 503 to 516 | 0 | 3,838 to 4,351 |
| Answer tokens, mean | 342 to 346 | 262 to 265 | 3,494 to 4,201 | 1,178 to 1,270 |

The pattern matches the 3050 and E2B. Thinking moves the working out of the answer and adds to it: about 500 thinking tokens in front of a GSM8K answer a quarter shorter. Thinking on used more tokens on 1,307 to 1,313 of the 1,319 GSM8K problems and on 21 to 26 of 30 AIME problems.

---

#### Step 6 — Time and the Drafter

| Weights | Drafter | GSM8K off | GSM8K on | AIME off | AIME on |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Repack | none | 3.54 h | 7.91 h | 0.81 h | 1.19 h |
| Repack | yes | 🥇 1.98 h | 4.25 h | 🥇 0.47 h | 0.66 h |
| Google | none | 3.69 h | 8.07 h | 0.93 h | 1.35 h |
| Google | yes | 2.09 h | 4.43 h | 0.57 h | 0.70 h |

With the drafter, GSM8K finished 1.77 to 1.86 times sooner. The main model accepted 0.828 to 0.835 of the drafted tokens on GSM8K, the same as the 3050's 0.829 to 0.835, yet the speed-up is smaller: 1.88 to 1.90 times against 2.23 to 2.29. An earlier measurement on these rigs found a drafting step costs the 1650 Ti a larger share of a plain step than faster cards pay, which fits a smaller gain here.

The drafter pays more for E4B than for E2B on this card: 1.88 to 1.90 times at two drafted tokens against 1.37 to 1.39 at one, each model's best length here, for 102 MiB against 192.

Without the drafter this card generates at 0.63 to 0.66 times the 3050; with it, at 0.52 to 0.56 times.

---

#### 🔎 Tip: Re-run a Setting Before Trusting One Odd Result

Sixteen settings compared across two cards give sixteen chances for one to land low by chance. Here one did, at p = 0.0029, in a setting that scored best of eight GSM8K settings on the other card and below its own no-drafter twin on this one. Before changing how you serve on the strength of one cell, run that cell again; the saved replies and `run.sh` make a single cell a 4.4-hour job on this card.

---

#### Compare and Contrast

| | GTX 1650 Ti, 4 GB | RTX 3050, 8 GB |
| :--- | :--- | :--- |
| GSM8K score range | 0.9265 to 0.9416 | 0.9272 to 0.9439 |
| AIME right, of 30 | 11 to 13 | 9 to 13 |
| All 16 settings | 42.63 h | 🥇 26.83 h |
| Generation, no drafter | 36.1 to 38.1 tok/s | 🥇 54.5 to 58.4 tok/s |
| Generation, drafter | 67.9 to 72.3 tok/s | 🥇 121.6 to 132.3 tok/s |
| Drafter speed-up, GSM8K | 1.88 to 1.90x | 🥇 2.23 to 2.29x |
| Context | 36,864 | 🥇 131,072 |
| Card memory, repack, no drafter | 3,355 MiB, per-layer table in system memory | 6,430 MiB, per-layer table on the card |

| | Thinking off | Thinking on |
| :--- | :--- | :--- |
| GSM8K time, repack with drafter | 🥇 1.98 h | 4.25 h |
| Thinking tokens per GSM8K problem | 0 | 503 to 516 |
| Largest GSM8K change | — | +20, p = 0.0105 alone, 0.24 corrected |

---

#### So, Which One?

If a 4 GB laptop card is what you have, E4B fits it and gives the same answers as an 8 GB desktop card in 15 of 16 settings. What you pay is time, 1.59 times as much across the whole study, and context, 36,864 tokens against 131,072.

Turn the drafter on. It costs 102 MiB and lifts E4B's generation 1.88 to 1.90 times on this card, more than it does for E2B.

Leave thinking off for grade-school math. On GSM8K it doubled the time and moved no score beyond chance after correction; on hard math, measure on your own problems first.

Between the two 4-bit builds, take the repack: same answers, 204 MiB less card memory, faster generation.

---

#### Summary

The goal of this article was to find out whether a 4 GB laptop card answers like an 8 GB desktop card for Gemma 4 E4B with thinking off and on, and what it costs in time. The key to the solution was fitting E4B in 4 GB with the per-layer table in system memory, then running the same 16 settings with the same files and llama.cpp build on both cards and pairing every problem's verdict between them. The results were:

- 🟢 15 of 16 settings scored within chance of the RTX 3050: −11 to +9 GSM8K problems, −1 to +3 AIME problems.
- ⚠️ Google's file with the drafter and thinking on scored 18 GSM8K problems below the 3050 (p = 0.0029), in one run.
- ⚠️ Only 27 to 353 of 1,319 GSM8K replies came out word-for-word the same on the two cards; the verdicts agreed on 1,285 to 1,299.
- ⚠️ The study took 42.63 hours against 26.83, 1.59 times as long.
- ❌ Thinking doubled the GSM8K time (2.13 to 2.24 times) and moved no score beyond chance after correction for 24 tests.
- 🟢 The drafter lifted generation 1.88 to 1.90 times on GSM8K for 102 MiB; its one low score, p = 0.0059, does not survive correction.
- 🟢 The repack and Google's QAT file scored level; the repack used 204 MiB less and generated 4.2 to 5.7 percent faster.

Scope: one GTX 1650 Ti Max-Q with 4 GB in one laptop on mains power, llama.cpp `7182ef7`, Gemma 4 E4B only, one run of each of 16 settings from 2026-11-25 to 27, temperature 0, one request at a time, 36,864 tokens of context here and 131,072 on the 3050, a 32,768-token reply limit, 2 drafted tokens on both. The per-layer table stayed in system memory here and sat on the card on the 3050; both are each card's own setting, so the time comparisons are between the two setups as served.

The strategy for measuring thinking and a drafter for Gemma 4 E4B on a 4 GB laptop GPU was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [This run: replies, logs, analysis](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-11-25-thinking-e4b-1650ti)
- [The RTX 3050 run](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem)
- [Thinking On or Off? Gemma 4 E4B on an 8 GB RTX 3050](https://dev.to/gde/thinking-on-or-off-gemma-4-e4b-on-an-8-gb-rtx-3050-5d5)
- [Thinking on Constrained Deployments: Gemma E2B on 4GB 1650](https://dev.to/gde/thinking-on-constrained-deployments-gemma-e2b-on-4gb-1650-487c)
- [The scoring script](https://github.com/xbill9/gemma4-dev/tree/main/benchmarks/gen-eval)
- [Exact Q4_0 repack of Gemma 4 E4B QAT](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf)
- [Google's Gemma 4 E4B QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)
- [Exact Q4_0 E4B QAT drafter](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Hope you enjoyed it!
