---
title: "E2B, E4B or 12B? Gemma 4 on an 8 GB RTX 3050"
published: false
description: "Three Gemma 4 models on one 8 GB RTX 3050, on all 1,319 GSM8K problems and the 30 of AIME 2025, thinking off and on. Each step up in size scored higher with thinking off and took longer. Thinking on changed nothing beyond chance for E2B and E4B; at temperature 0 the 12B found the answer and then never stopped, scoring 87 of 157 against 152."
tags: gemma, llm, benchmark, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-e2b-e4b-12b-3050/devto-cover.f74e04e1.jpg
---

This article provides a step by step comparison of three Gemma 4 models on one local GPU: E2B, E4B and 12B, each as a 4-bit QAT build on an 8 GB RTX 3050 through llama.cpp. It rolls up the [E2B](https://dev.to/gde/thinking-on-or-off-gemma-4-e2b-on-an-8-gb-rtx-3050-30e0) and [E4B](https://dev.to/gde/thinking-on-or-off-gemma-4-e4b-on-an-8-gb-rtx-3050-5d5) articles and adds the 12B, on the same problems and settings, then says which one to serve.

https://github.com/xbill9/gemma4-dev/tree/main/docs/gemma4-e2b-e4b-12b-3050

The short version: with thinking off, every step up in size answers more and takes longer. Serve E4B with the drafter as the default. Take the 12B for the last three points on math, with thinking off. Do not turn thinking on for the 12B at temperature 0: it works out the answer and then argues with the question until it runs out of tokens.

---

#### The Findings

**Thinking off, bigger scores higher.** GSM8K 0.9014 for E2B, 0.9295 for E4B and 0.9621 for the 12B; each step is beyond chance on a paired test (p = 0.00039 and p = 8.9e-7). AIME 2025: 7, 12 and 15 of 30.

**Bigger is slower.** A full GSM8K pass took 1.36, 2.38 and 6.10 hours. The 12B took 2.56 times as long as E4B for 43 more problems.

**Thinking on, E2B and E4B: no change.** GSM8K moved by −11 and +4 problems, both within chance, for 2.31 and 2.29 times the time.

**Thinking on, 12B: it falls apart.** On the first 157 GSM8K problems it got 87 right against 152 with thinking off, and 70 replies ran to the 32,768-token limit. In 61 of those 70 the thinking had already written the right answer, around token 333.

**Why.** At temperature 0 the model always takes the most likely next token. The 12B second-guesses the wording of the question and repeats the same block of doubt until the limit; an f16 cache loops the same way. With Google's recommended sampling settings, all 8 of the looping problems tested finished and 7 were right.

**What to do with it.**

- 🟢 E4B with the drafter is the default: 0.9295 on GSM8K, thinking safe, the full 131,072-token context.
- 🟢 The 12B with thinking off is the most accurate model that fits the card.
- ❌ The 12B with thinking on at temperature 0 is unusable.

---

#### What Was Measured

The same run three times, once per model, on one card:

| | E2B | E4B | 12B |
| :--- | :--- | :--- | :--- |
| Weights | exact Q4_0 repack of Google's QAT | same | same |
| File | 2.62 GB | 4.22 GB | 6.72 GB |
| Card memory when serving | 🥇 3,610 MiB | 6,430 MiB | 7,294 MiB |
| Context, cache | 131,072, f16 | 131,072, f16 | 81,920, q4_0 |
| Generation, one request | 🥇 109.2 tok/s | 58.4 tok/s | 26.7 tok/s |

Each model answered all 1,319 GSM8K test problems and the 30 problems of AIME 2025, with thinking off and on, one request at a time, at temperature 0, with a 32,768-token limit on each reply, thinking included. E2B and E4B also ran Google's own QAT file and a drafter; those results are in their own articles. The 12B ran the repack without a drafter, and its GSM8K thinking-on run was stopped at 157 problems.

---

#### At This Point You Should Have…

- An NVIDIA card with 8 GB, here an RTX 3050 OEM on driver 615.78.08
- llama.cpp built with CUDA, here commit `7182ef71e`
- Python 3
- A clone of [gemma4-dev](https://github.com/xbill9/gemma4-dev)

---

#### Step 1 — Fitting Three Sizes on 8 GB

E2B and E4B fit with room to spare: the full 131,072-token context at f16 takes 3,610 and 6,430 MiB. The 12B's weights alone are 6.72 GB, so its context cache has to shrink to 4 bits (q4_0) to hold 81,920 tokens in 7,294 MiB. A 4-bit cache scored level with f16 on GSM8K in an earlier 12B run, and the loop test in Step 5 ran both caches.

---

#### Step 2 — Thinking Off: Scores and Time

**GSM8K, 1,319 problems, thinking off**

| | E2B | E4B | 12B |
| :--- | ---: | ---: | ---: |
| Right | 1,189 | 1,226 | 🥇 1,269 |
| Score | 0.9014 | 0.9295 | 🥇 0.9621 |
| 95% range | 0.884 to 0.916 | 0.914 to 0.942 | 0.950 to 0.971 |
| Hours | 🥇 1.36 | 2.38 | 6.10 |
| Seconds per problem | 🥇 3.72 | 6.50 | 16.64 |

**AIME 2025, 30 problems, thinking off**

| | E2B | E4B | 12B |
| :--- | ---: | ---: | ---: |
| Right | 7 | 12 | 🥇 15 |
| Replies at the token limit | 3 | 🥇 0 | 9 |
| Hours | 0.53 | 🥇 0.52 | 4.20 |

The 12B reached 15 of 30 on AIME with 9 replies cut off at the limit, so even with thinking off it loops on hard problems.

---

#### Step 3 — Is Each Step Up Bigger Than Chance?

The same problems ran for every model, so each comparison is an exact paired test on the problems one model got right and the other got wrong.

| Comparison, thinking off | Right only smaller | Right only bigger | p |
| :--- | ---: | ---: | ---: |
| GSM8K, E2B to E4B | 34 | 71 | 0.00039 |
| GSM8K, E4B to 12B | 17 | 60 | 8.9e-7 |
| AIME, E2B to E4B | 0 | 5 | 0.0625 |
| AIME, E4B to 12B | 2 | 5 | 0.453 |
| AIME, E2B to 12B | 0 | 8 | 0.0078 |

On GSM8K both steps hold. On AIME only the jump from E2B to the 12B does; 30 problems are too few to separate neighbours.

---

#### Step 4 — Thinking On

| | E2B | E4B | 12B |
| :--- | ---: | ---: | ---: |
| GSM8K, off to on | 1,189 to 1,178 | 1,226 to 1,230 | 152 to 87, first 157 |
| GSM8K, p | 0.294 | 0.683 | 5.4e-20 |
| GSM8K time with thinking | 2.31x | 2.29x | 22.2x |
| AIME, off to on | 7 to 10 | 12 to 10 | 15 to 10 |
| AIME, p | 0.375 | 0.688 | 0.227 |
| AIME replies at the limit, on | 2 | 🥇 0 | 19 |

For E2B and E4B thinking doubled the time and moved no score beyond chance. For the 12B it lost 65 GSM8K problems and gained none. Every one of its 87 replies that stopped on its own was right; the other 70 hit the limit and never wrote an answer.

On the same first 157 problems with thinking on, E4B got 148 and E2B 142.

---

#### Step 5 — Why the 12B Falls Apart

`why.py` reads the 70 GSM8K replies that hit the limit:

- In 61 of 70 the thinking writes `= <the right answer>`, at about token 333 (median).
- From about token 1,985 a block of about 1,094 tokens repeats until the limit (medians).
- The last 3,000 characters of 55 of the 70 re-read the wording: "Could it mean…", "Let's think about the wording … again".
- None of the 70 writes its final `Answer:` line.

The same 70 problems for the smaller models, thinking on:

| | Finished | Right | Longest reply |
| :--- | ---: | ---: | ---: |
| E2B | 70 | 58 | 6,027 tokens |
| E4B | 70 | 🥇 63 | 🥇 2,131 tokens |
| 12B | 0 | 0 | 32,768 tokens |

Then the first 8 of those problems again, one setting changed at a time:

| Setting | Cache | Finished | Right |
| :--- | :--- | ---: | ---: |
| Temperature 0, as in the run | q4_0 | 0 | 0 |
| Temperature 0 | f16 | 0 | 0 |
| Google's settings: temperature 1.0, top-k 64, top-p 0.95 | q4_0 | 🥇 8 | 🥇 7 |

The cache makes no difference; temperature 0 is what keeps the 12B in the loop. The likely mechanism: once a block of doubt has repeated, the same block again is the most likely next text, and temperature 0 always takes it, while sampling can pick a different token and leave. The sampled replies took 1,596 to 11,894 tokens.

---

#### Step 6 — The Drafter

Each model has a small QAT drafter from Google that guesses the next two tokens for the main model to check in one step. From the earlier runs on this card:

| | E2B | E4B | 12B |
| :--- | ---: | ---: | ---: |
| GSM8K time, drafter on | 1.75 to 1.87x sooner | 🥇 1.99 to 2.39x sooner | 2.34x sooner |
| Guesses accepted | 0.80 | 🥇 0.83 to 0.84 | 0.66 |
| Score change | none beyond chance | none beyond chance | none beyond chance |

The 12B figures come from its October run: GSM8K with thinking off, an f16 cache at 24,576 tokens, 0.9598 with the drafter and 0.9629 without, and 0.658 acceptance on six chat prompts. The drafter pays on every model and costs nothing in answers.

---

#### 🔎 Tip: Copy the Model's Sampling Settings Before You Turn Thinking On

Benchmarks usually run at temperature 0 so the results repeat. For the 12B with thinking on, that setting is what breaks it. Google's `generation_config.json` for the 12B sets temperature 1.0, top-k 64 and top-p 0.95; with those, all 8 looping problems tested finished. Read the model's settings before you serve it with thinking on, and check how many replies hit your token limit: a looping model shows up there before it shows up anywhere else.

---

#### Compare and Contrast

| | E2B | E4B | 12B |
| :--- | :--- | :--- | :--- |
| GSM8K, thinking off | 0.9014 | 0.9295 | 🥇 0.9621 |
| AIME, thinking off | 7 of 30 | 12 of 30 | 🥇 15 of 30 |
| GSM8K hours, no drafter | 🥇 1.36 | 2.38 | 6.10 |
| Card memory | 🥇 3,610 MiB | 6,430 MiB | 7,294 MiB |
| Context on 8 GB | 🥇 131,072 | 🥇 131,072 | 81,920, 4-bit cache |
| Thinking on at temperature 0 | safe, no gain | 🥇 safe, no gain | loops |
| Replies at the limit, all runs | 24 of 10,792 | 🥇 0 of 10,792 | 101 of 1,536 |

**Pros and cons**

- **E2B.** 🟢 Fastest, smallest, the full context in 3.6 GB. ❌ Lowest scores: 37 fewer GSM8K problems than E4B and 80 fewer than the 12B.
- **E4B.** 🟢 Never hit the token limit in 10,792 replies. 🟢 The best drafter gain, up to 2.39x. 🟢 The full context. ⚠️ 1.75 times E2B's time.
- **12B.** 🟢 The most right answers with thinking off, beyond chance over E4B. ❌ 2.56 times E4B's time and 7.3 GB of the card. ❌ Loops at temperature 0 with thinking on, and on some hard problems with it off.

---

#### So, Which One?

**Default: E4B with the drafter.** It is about three points behind the 12B on GSM8K and finishes a pass in 2.38 hours against 6.10 without a drafter. It loops on nothing, and it keeps the full 131,072-token context on the card.

**Most accurate: the 12B with thinking off.** It answered 60 GSM8K problems E4B missed and missed 17 E4B answered. Serve it with its drafter, and if you need thinking, use Google's sampling settings.

**Smallest and fastest: E2B.** It uses 3.6 GB and answers in 3.72 seconds per GSM8K problem.

**Thinking: leave it off on all three.** It doubled the time on E2B and E4B for no change beyond chance, and at temperature 0 it broke the 12B.

---

#### Summary

The goal of this article was to compare Gemma 4 E2B, E4B and 12B on one 8 GB card and say which to serve. The key to the solution was running all three on the same 1,349 problems with the same settings, so every comparison is paired problem by problem. The results were:

- 🟢 Thinking off, every step up in size scored higher on GSM8K beyond chance: 0.9014, 0.9295, 0.9621.
- ⚠️ Every step up in size took longer: 1.36, 2.38 and 6.10 hours for GSM8K.
- ⚠️ Thinking doubled the time on E2B and E4B and changed no score beyond chance.
- ❌ At temperature 0, the 12B with thinking on loops: 70 of 157 GSM8K replies hit the limit, 61 of them after writing the right answer.
- 🟢 Google's sampling settings stopped the loop on all 8 problems tested; the cache type made no difference.
- 🟢 The drafter made every model 1.75 to 2.39 times faster on GSM8K and changed no score.

Scope: one RTX 3050 OEM with 8 GB in one desktop, llama.cpp `7182ef71e`, one run of each setting, exact Q4_0 repacks of Google's QAT weights, temperature 0 unless stated, one request at a time, a 32,768-token reply limit, 2026-11-16 to 21. GSM8K is all 1,319 test problems and AIME 2025 is papers I and II. The 12B's GSM8K thinking-on run covers the first 157 problems, the sampling test 8 problems, and its drafter figures come from an October run with a different cache and context. Times move with clock and temperature.

The strategy for choosing a Gemma 4 size for a small GPU was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [This article's evidence: scripts and outputs](https://github.com/xbill9/gemma4-dev/tree/main/docs/gemma4-e2b-e4b-12b-3050/evidence)
- [The 12B run: replies and logs](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-17-thinking-12b-3050oem)
- [The 12B loop test](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-20-thinking-loops-12b-3050oem)
- [Thinking On or Off? Gemma 4 E2B on an 8 GB RTX 3050](https://dev.to/gde/thinking-on-or-off-gemma-4-e2b-on-an-8-gb-rtx-3050-30e0)
- [Thinking On or Off? Gemma 4 E4B on an 8 GB RTX 3050](https://dev.to/gde/thinking-on-or-off-gemma-4-e4b-on-an-8-gb-rtx-3050-5d5)
- [The scoring script](https://github.com/xbill9/gemma4-dev/tree/main/benchmarks/gen-eval)
- [Exact Q4_0 repack of Gemma 4 12B QAT](https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf)
- [Exact Q4_0 12B QAT drafter](https://huggingface.co/xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Hope you enjoyed it!
