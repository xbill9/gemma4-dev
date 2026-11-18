---
title: "Thinking On or Off? Gemma 4 E4B on an 8 GB RTX 3050"
published: false
description: "Gemma 4 E4B on a local RTX 3050 with thinking off and on, two QAT 4-bit builds, with and without a drafter, on all 1,319 GSM8K problems and the 30 problems of AIME 2025. E4B scored 0.927 to 0.944 on GSM8K, beyond chance above E2B in all eight setups; thinking took 2.24 to 2.47 times as long for at most 16 more problems; the drafter cut the time by 1.38 to 2.39 times and changed no score."
tags: gemma, llm, benchmark, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-e4b-thinking-3050/devto-cover.690126f5.jpg
---

This article provides a step by step guide to measuring what thinking buys Gemma 4 E4B on a local GPU, and what it costs in time. E4B runs on an 8 GB RTX 3050 through llama.cpp and answers every GSM8K test problem and all 30 AIME 2025 problems with thinking off and with thinking on, on two 4-bit builds of the same QAT weights, each with and without a drafter. It is the second model in the series, after [Gemma 4 E2B](https://dev.to/gde/thinking-on-or-off-gemma-4-e2b-on-an-8-gb-rtx-3050-30e0), on the same card, the same problems and the same settings.

https://github.com/xbill9/gemma4-dev/tree/main/docs/gemma4-e4b-thinking-3050

The short version: E4B gets 93 to 94 percent of GSM8K right, clearly above E2B. Thinking more than doubled the time and lifted the score by at most 16 problems. The drafter made every run up to 2.39 times faster and changed no score.

---

#### The Findings

**How E4B scores.** GSM8K 0.9272 to 0.9439 across the eight setups, and 9 to 13 of 30 on AIME 2025. Against E2B on the same problems with the same settings, E4B got 35 to 58 more GSM8K problems right, beyond chance in all eight setups (p = 0.00082 or smaller).

**What thinking cost.** On GSM8K, thinking on produced 2.24 to 2.30 times as many tokens per problem and took 2.24 to 2.47 times as long.

**What thinking bought.** A little, at most. GSM8K rose in all four setups, by +2, +4, +13 and +16 problems. The +16, Google's file with the drafter, passes a paired test at p = 0.0440; the other three do not (p = 0.1175 to 0.8919). With four tests, one passing at 0.05 happens by chance alone 18.5 percent of the time. AIME moved by −4 to +2 problems, within chance.

**What the drafter bought.** Every GSM8K run finished 1.99 to 2.39 times sooner with Google's QAT drafter, at 130.8 to 132.3 against 57.0 to 58.4 generated tokens per second with the repack. It changed no score beyond chance in any of eight paired tests.

**The two weight builds.** The exact Q4_0 repack and Google's QAT Q4_0 file scored level on every paired test. The repack is 4.22 GB against 5.15 GB, and serves in 6,430 MiB against 7,326 MiB.

**What to do with it.**

- 🟢 Serve E4B with the drafter. It is the one change here that pays every time.
- 🟢 On a card that holds it, pick E4B over E2B for math: 35 to 58 more GSM8K problems right at 1.29 to 1.75 times the time.
- ⚠️ Thinking on adds about a point on GSM8K at best, for more than twice the time.

**Caveats.** One run of each setting, temperature 0. AIME's 30 problems can only show large effects.

---

#### What Was Measured

The [E2B article](https://dev.to/gde/thinking-on-or-off-gemma-4-e2b-on-an-8-gb-rtx-3050-30e0) set up the question: on a local GPU the bill for thinking is time, and Gemma 4's chat template has two thinking settings, off and on. This run repeats it with the next model up.

One model, Gemma 4 E4B, on one card, with three factors crossed:

| Factor | Settings |
| :--- | :--- |
| Thinking | off, on (`enable_thinking` in the chat template) |
| Weights | exact Q4_0 repack of Google's QAT weights, Google's QAT Q4_0 GGUF |
| Drafter | none, Google's E4B QAT assistant as exact Q4_0, 2 drafted tokens |
| Problems | GSM8K test, 1,319; AIME 2025 I and II, 30 |

That is 16 runs, 26.83 hours of GPU time, every reply saved.

---

#### At This Point You Should Have…

- An NVIDIA card with 8 GB, here an RTX 3050 OEM on driver 615.78.08
- llama.cpp built with CUDA, here commit `7182ef71e`
- Python 3 with `huggingface_hub`, `tokenizers` and `scipy`
- A clone of [gemma4-dev](https://github.com/xbill9/gemma4-dev)

---

#### Step 1 — Two Builds of the Same Weights, and a Drafter

Google trained Gemma 4 E4B to run at 4 bits (quantization-aware training, QAT) and ships it as a Q4_0 GGUF. The repack, `xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`, stores the same QAT 4-bit values in a smaller file.

| File | Bytes | Card memory when serving |
| :--- | ---: | ---: |
| Repack, `gemma-4-E4B-it-q4_0-exact.gguf` | 4,215,695,712 | 6,430 MiB |
| Google, `gemma-4-E4B_q4_0-it.gguf` | 5,154,941,280 | 7,326 MiB |
| Drafter, `gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf` | 59,678,368 | +194 MiB |

The drafter is Google's small E4B QAT assistant model, stored as exact Q4_0. It guesses the next two tokens and the main model checks them in one step, so accepted guesses come out for the price of one step.

Every run serves the same way as E2B's: the full 131,072-token context with an f16 cache, the per-layer embedding table on the card, one request at a time, temperature 0, and a 32,768-token limit on each reply, thinking included. Google's file with the drafter uses 7,520 of the card's 8,192 MiB.

---

#### Step 2 — Run All Sixteen

```bash
cd local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem
setsid nohup ./chain.sh > nohup-chain.out 2>&1 < /dev/null &
```

`run.sh` reads the model's files from `wave.env`, starts the server once per weights and drafter pair, and runs four cells on it: GSM8K thinking off, AIME off, AIME on, GSM8K on. The scoring script asks for the final answer on its own last line as `Answer: <number>` and marks a reply right when that number matches. Each reply is saved as it finishes, so a stopped run picks up where it left off.

The run's own log, `driver.log`, shows one line per cell:

```
2026-11-17 12:58:18 59, 1897 MHz END gsm8k repack-none thinking=off rc=0 {"task": "gsm8k", "max_tokens": 32768, "n": 1319, "right": 1226, "accuracy": 0.9295, "errors": 0, "truncated": 0, ...
```

---

#### Step 3 — The Scores

`analyze.py` builds every table from the saved replies.

**GSM8K, 1,319 problems**

| Weights | Drafter | Thinking off | Thinking on |
| :--- | :--- | ---: | ---: |
| Repack | none | 0.9295 | 0.9325 |
| Repack | yes | 0.9272 | 0.9371 |
| Google | none | 0.9348 | 0.9363 |
| Google | yes | 0.9318 | 🥇 0.9439 |

All eight sit between 0.9272 and 0.9439, and every 95% range overlaps every other.

**AIME 2025, 30 problems, right answers**

| Weights | Drafter | Thinking off | Thinking on |
| :--- | :--- | ---: | ---: |
| Repack | none | 12 | 10 |
| Repack | yes | 10 | 12 |
| Google | none | 12 | 12 |
| Google | yes | 🥇 13 | 9 |

No reply in any of the 16 runs reached the 32,768-token limit.

---

#### Step 4 — E4B Against E2B

The E2B run used the same problems, scoring script, card, llama.cpp commit and settings, so each E4B cell pairs with its E2B twin problem by problem.

| Weights | Drafter | Thinking | GSM8K right, E2B → E4B | p | Time, E4B / E2B |
| :--- | :--- | :--- | ---: | ---: | ---: |
| Repack | none | off | 1,189 → 1,226 | 0.00039 | 1.75 |
| Repack | yes | off | 1,188 → 1,223 | 0.00082 | 1.29 |
| Google | none | on | 1,194 → 1,235 | 0.000038 | 1.58 |
| Google | yes | on | 1,187 → 1,245 | 0.0000000019 | 1.39 |

All eight GSM8K pairs favour E4B beyond chance; the four above span the range, and the full list is in `derived.txt`. On AIME with thinking off, E4B got 10 to 13 right against E2B's 5 to 7, beyond chance in two of four setups; with thinking on the two models are level (9 to 12 against 6 to 10).

E4B generates at 0.52 to 0.68 times E2B's speed. Its GSM8K replies are 0.84 to 0.89 times as long, which brings the GSM8K time to 1.29 to 1.75 times E2B's. On AIME its replies are half to two thirds as long, so AIME runs took 0.74 to 1.47 times E2B's time.

---

#### Step 5 — Where the Tokens Go

`tokens.py` counts thinking and answer tokens in each saved reply with Gemma 4 E4B's tokenizer. Its count matches the server's on 5,393 of 5,396 replies at each thinking setting, up to the fixed end-of-turn and thinking markers, and is one token off on the other 3.

| | GSM8K off | GSM8K on | AIME off | AIME on |
| :--- | ---: | ---: | ---: | ---: |
| Thinking tokens, mean | 0 | 505 to 514 | 0 | 3,760 to 4,288 |
| Answer tokens, mean | 340 to 344 | 263 to 264 | 3,582 to 4,174 | 1,138 to 1,336 |

The pattern matches E2B's. On GSM8K the answer shrinks by about a quarter and about 510 thinking tokens are added in front of it. With thinking off on AIME, E4B works the problem out in the answer itself; with thinking on, that working moves into the thinking text. E4B thinks less than E2B on both: 505 to 514 tokens against 601 to 615 on GSM8K, and 3,760 to 4,288 against 7,668 to 10,240 on AIME.

Thinking on used more tokens on 1,309 to 1,313 of the 1,319 GSM8K problems and on 23 to 26 of 30 AIME problems in every setup, far beyond chance (signed-rank p below 0.006 in all eight).

---

#### Step 6 — Time and the Drafter

| Weights | Drafter | GSM8K off | GSM8K on | AIME off | AIME on |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Repack | none | 2.38 h | 5.45 h | 0.52 h | 0.76 h |
| Repack | yes | 🥇 1.00 h | 2.46 h | 0.38 h | 🥇 0.34 h |
| Google | none | 2.34 h | 5.24 h | 0.62 h | 0.80 h |
| Google | yes | 1.16 h | 2.64 h | 0.33 h | 0.40 h |

The drafter made every GSM8K run 1.99 to 2.39 times faster and every AIME run 1.38 to 2.23 times faster. The main model accepted 0.83 to 0.84 of the drafted tokens on GSM8K and 0.80 to 0.82 on AIME, and generation rose from 54.5–58.4 tokens per second to 121.6–132.3. That is a larger gain than on E2B, where the same drafter setup gave 1.55 to 2.04 times.

Thinking took 1.19 to 1.46 times as long on AIME in three setups. In the fourth, the repack with the drafter, the thinking-off run spent 395 seconds outside generation on a few replies, so its clock time reads 0.90 times; by generation time alone thinking took 1.21 times as long.

Per problem, GSM8K took 2.72 seconds with the repack, the drafter and thinking off, and 14.87 seconds with the repack, no drafter and thinking on.

---

#### Step 7 — Is Any Difference Bigger Than Chance?

The same problems ran in every cell, so each comparison is an exact paired test on the problems one setting got right and the other got wrong.

| Comparison, GSM8K | Right only before | Right only after | p |
| :--- | ---: | ---: | ---: |
| Thinking on, repack, no drafter | 25 | 29 | 0.6835 |
| Thinking on, repack, drafter | 23 | 36 | 0.1175 |
| Thinking on, Google, no drafter | 26 | 28 | 0.8919 |
| Thinking on, Google, drafter | 20 | 36 | 0.0440 |
| Drafter, largest change | 10 | 20 | 0.0987 |
| Google instead of repack, largest change | 15 | 24 | 0.1996 |

Thinking changed the answer on 54 to 59 GSM8K problems in each setup, and the gains outnumber the losses in all four, by 2 to 16. On E2B thinking changed 79 to 106 answers and the direction split both ways. On AIME no comparison does better than p = 0.2891. The full table of 24 tests is in `RESULTS.md`.

---

#### 🔎 Tip: Turn the Drafter On Before You Turn Thinking On

Both change how long an answer takes. On E4B the drafter bought 1.38 to 2.39 times the speed and changed no answer beyond chance; thinking cost 2.24 to 2.47 times the time on GSM8K for 2 to 16 more problems. With both on, a GSM8K problem took 6.72 to 7.20 seconds, against 14.30 to 14.87 for thinking on without the drafter.

---

#### Compare and Contrast

| | Thinking off | Thinking on |
| :--- | :--- | :--- |
| GSM8K score range | 0.9272 to 0.9348 | 0.9325 to 🥇 0.9439 |
| AIME right, of 30 | 10 to 13 | 9 to 12 |
| GSM8K time, with drafter | 🥇 1.00 to 1.16 h | 2.46 to 2.64 h |
| Thinking tokens per GSM8K problem | 🥇 0 | 505 to 514 |
| Score change beyond chance | — | one of eight tests |

| | E2B | E4B |
| :--- | :--- | :--- |
| GSM8K score range | 0.8908 to 0.9052 | 🥇 0.9272 to 0.9439 |
| AIME right, thinking off | 5 to 7 | 🥇 10 to 13 |
| GSM8K time, drafter, thinking off | 🥇 0.77 to 0.84 h | 1.00 to 1.16 h |
| Drafter speed-up, GSM8K | 1.75 to 1.87 | 🥇 1.99 to 2.39 |
| Card memory, repack, no drafter | 🥇 3,610 MiB | 6,430 MiB |

---

#### So, Which One?

For math on an 8 GB card, serve E4B with the drafter. It answers 35 to 58 more GSM8K problems than E2B, and with the drafter a full GSM8K pass takes 1.00 hour against E2B's 0.77.

Leave thinking off for grade-school math unless the last point matters. It raised the score in all four setups, by 2 to 16 problems, one of them beyond chance, and it more than doubled the time.

On hard math, thinking on gave E4B no extra AIME answers: 9 to 12 right against 10 to 13 off.

Between the two 4-bit builds, take the repack: same answers, 0.94 GB smaller and 896 MiB less card memory.

---

#### Summary

The goal of this article was to measure what thinking buys Gemma 4 E4B on a local GPU and what it costs in time. The key to the solution was holding the model and card fixed and crossing thinking with the weight build and the drafter, on the same 1,349 problems in every cell, with E2B's run as the twin. The results were:

- 🟢 E4B scored 0.9272 to 0.9439 on GSM8K, beyond chance above E2B in all eight setups.
- ⚠️ Thinking took 2.24 to 2.47 times as long on GSM8K for 2 to 16 more problems, one of four gains beyond chance.
- ❌ Thinking gave no AIME gain: −4 to +2 problems.
- 🟢 The drafter made every GSM8K run 1.99 to 2.39 times faster and changed no score beyond chance.
- 🟢 The exact Q4_0 repack and Google's QAT file scored level; the repack is 0.94 GB smaller.
- 🟢 No reply hit the 32,768-token limit.

Scope: one RTX 3050 OEM with 8 GB in one desktop, llama.cpp `7182ef71e`, Gemma 4 E4B, one run of each of 16 settings on 2026-11-17 and 18, temperature 0, one request at a time, the full 131,072-token context, a 32,768-token reply limit. GSM8K is all 1,319 test problems and AIME 2025 is papers I and II. The E2B comparison uses that model's run of 2026-11-16 and 17 on the same card and commit. Times are for this card and move with clock and temperature; the 12B is measured separately.

The strategy for measuring thinking against a drafter on a local GPU was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [This run: replies, logs, analysis](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem)
- [The scoring script](https://github.com/xbill9/gemma4-dev/tree/main/benchmarks/gen-eval)
- [Thinking On or Off? Gemma 4 E2B on an 8 GB RTX 3050](https://dev.to/gde/thinking-on-or-off-gemma-4-e2b-on-an-8-gb-rtx-3050-30e0)
- [High or Low Thinking? Gemini 4 and Gemma 4 on Hard Math, Cost First](https://dev.to/gde/high-or-low-thinking-gemini-4-and-gemma-4-on-hard-math-cost-first-3091)
- [Exact Q4_0 repack of Gemma 4 E4B QAT](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf)
- [Google's Gemma 4 E4B QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)
- [Exact Q4_0 E4B QAT drafter](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Hope you enjoyed it!
