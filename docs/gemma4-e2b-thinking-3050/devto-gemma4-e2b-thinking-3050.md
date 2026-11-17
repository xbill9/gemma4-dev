---
title: "Thinking On or Off? Gemma 4 E2B on an 8 GB RTX 3050"
published: false
description: "Gemma 4 E2B on a local RTX 3050 with thinking off and on, two QAT 4-bit builds, with and without a drafter, on all 1,319 GSM8K problems and the 30 problems of AIME 2025. Thinking doubled the time and moved no score beyond chance; the drafter cut the time by 1.55 to 2.04 times and changed no score."
tags: gemma, llm, benchmark, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-e2b-thinking-3050/devto-cover.6b0d9ada.jpg
---

This article provides a step by step guide to measuring what thinking buys a small model on a local GPU, and what it costs in time. Gemma 4 E2B runs on an 8 GB RTX 3050 through llama.cpp and answers every GSM8K test problem and all 30 AIME 2025 problems with thinking off and with thinking on, on two 4-bit builds of the same QAT weights, each with and without a drafter.

https://github.com/xbill9/gemma4-dev/tree/main/docs/gemma4-e2b-thinking-3050

The short version: thinking doubled the time on GSM8K and changed no score by more than chance, while the drafter nearly halved the time and changed no score at all.

---

#### The Findings

**What thinking cost.** On GSM8K, thinking on produced 2.15 to 2.22 times as many tokens per problem and took 2.18 to 2.31 times as long. On AIME it took 1.44 to 1.71 times as long.

**What thinking bought.** Nothing measurable. Across the four setups, GSM8K moved by −11, −1, +12 and +18 problems, and none passes a paired test (p = 0.0982 to 1.0). AIME gained zero to four problems, again within chance.

**What the drafter bought.** Every cell finished 1.55 to 2.04 times sooner with Google's QAT drafter, at 195.1 against 109.2 generated tokens per second on GSM8K with the repack. It changed no score beyond chance in any of eight paired tests.

**The two weight builds.** The exact Q4_0 repack and Google's QAT Q4_0 file scored level on every paired test. The repack is 2.62 GB against 3.35 GB, loads in 3,610 MiB against 4,306 MiB, and generates 6.5 to 9.7 percent faster.

**What to do with it.**

- 🟢 Serve E2B with the drafter. It is the one change here that pays every time.
- ⚠️ Leave thinking off for grade-school math on E2B; on hard math, measure it on your own problems first.
- 🟢 Either 4-bit QAT build gives the same answers; the repack is smaller and faster.

**Caveats.** One run of each setting, temperature 0. AIME's 30 problems can only show large effects.

---

#### What Was Measured

The earlier article, [High or Low Thinking?](https://dev.to/gde/high-or-low-thinking-gemini-4-and-gemma-4-on-hard-math-cost-first-3091), set hosted Gemini and Gemma models at their lowest and highest thinking settings. On a local GPU the bill is time instead of money, and Gemma 4's chat template has only two thinking settings: off and on.

Everything else stays fixed. One model, Gemma 4 E2B, on one card, with three factors crossed:

| Factor | Settings |
| :--- | :--- |
| Thinking | off, on (`enable_thinking` in the chat template) |
| Weights | exact Q4_0 repack of Google's QAT weights, Google's QAT Q4_0 GGUF |
| Drafter | none, Google's E2B QAT assistant as exact Q4_0, 2 drafted tokens |
| Problems | GSM8K test, 1,319; AIME 2025 I and II, 30 |

That is 16 runs, 18.81 hours of GPU time, every reply saved.

---

#### At This Point You Should Have…

- An NVIDIA card with 8 GB, here an RTX 3050 OEM on driver 615.78.08
- llama.cpp built with CUDA, here commit `7182ef71e`
- Python 3 with `huggingface_hub`, `tokenizers` and `scipy`
- A clone of [gemma4-dev](https://github.com/xbill9/gemma4-dev)

---

#### Step 1 — Two Builds of the Same Weights, and a Drafter

Google trained Gemma 4 E2B to run at 4 bits (quantization-aware training, QAT) and ships it as a Q4_0 GGUF. The repack, `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`, stores the same QAT 4-bit values in a smaller file.

| File | Bytes | Card memory when serving |
| :--- | ---: | ---: |
| Repack, `gemma-4-E2B-it-q4_0-exact.gguf` | 2,620,370,912 | 3,610 MiB |
| Google, `gemma-4-E2B_q4_0-it.gguf` | 3,349,516,256 | 4,306 MiB |
| Drafter, `gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf` | 59,236,000 | +192 MiB |

The drafter is Google's small E2B QAT assistant model, stored as exact Q4_0. It guesses the next two tokens and the main model checks them in one step, so accepted guesses come out for the price of one step.

Every run serves the same way: the full 131,072-token context, one request at a time, temperature 0, and a 32,768-token limit on each reply, thinking included.

---

#### Step 2 — Run All Sixteen

```bash
cd local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem
setsid nohup ./run.sh > nohup.out 2>&1 < /dev/null &
```

`run.sh` starts the server once per weights and drafter pair and runs four cells on it: GSM8K thinking off, AIME off, AIME on, GSM8K on. The scoring script asks for the final answer on its own last line as `Answer: <number>` and marks a reply right when that number matches. Each reply is saved as it finishes, so `resume.sh` picks a stopped run up where it left off.

The run's own log, `driver.log`, shows one line per cell:

```
2026-11-16 17:04:42 64, 1890 MHz END gsm8k repack-none thinking=off rc=0 {"task": "gsm8k", "max_tokens": 32768, "n": 1319, "right": 1189, "accuracy": 0.9014, "errors": 0, "truncated": 0, ...
```

That first cell, 0.9014, matches what the same build with thinking off scored on this card in October, there with a 2,048-token reply limit.

---

#### Step 3 — The Scores

`analyze.py` builds every table from the saved replies.

**GSM8K, 1,319 problems**

| Weights | Drafter | Thinking off | Thinking on |
| :--- | :--- | ---: | ---: |
| Repack | none | 0.9014 | 0.8931 |
| Repack | yes | 0.9007 | 0.8999 |
| Google | none | 0.8916 | 🥇 0.9052 |
| Google | yes | 0.8908 | 0.8999 |

All eight sit between 0.8908 and 0.9052, and every 95% range overlaps every other.

**AIME 2025, 30 problems, right answers**

| Weights | Drafter | Thinking off | Thinking on |
| :--- | :--- | ---: | ---: |
| Repack | none | 7 | 🥇 10 |
| Repack | yes | 7 | 🥇 10 |
| Google | none | 6 | 6 |
| Google | yes | 5 | 9 |

Two to five AIME replies per cell ran into the 32,768-token limit and count as wrong. With thinking off they ran out while writing the answer; with thinking on they ran out while still thinking. One GSM8K reply did the same, with thinking on.

---

#### Step 4 — Where the Tokens Go

`tokens.py` counts thinking and answer tokens in each saved reply with Gemma 4 E2B's tokenizer. Its count matches the server's on every reply, up to the fixed end-of-turn and thinking markers.

| | GSM8K off | GSM8K on | AIME off | AIME on |
| :--- | ---: | ---: | ---: | ---: |
| Thinking tokens, mean | 0 | 601 to 615 | 0 | 7,668 to 10,240 |
| Answer tokens, mean | 399 to 408 | 270 to 280 | 5,812 to 6,548 | 846 to 1,105 |

Thinking moves the working from the answer into the thinking text. On GSM8K the answer shrinks by about a third and about 600 thinking tokens are added in front of it. With thinking off on AIME, E2B works the problem out in the answer itself, at 5,812 to 6,548 tokens per reply; with thinking on, that working moves into the thinking text and grows.

Thinking on used more tokens on 1,301 to 1,307 of the 1,319 GSM8K problems and on 26 of 30 AIME problems in every setup, which is far beyond chance (signed-rank p below 0.0012 in all eight).

---

#### Step 5 — Time and the Drafter

| Weights | Drafter | GSM8K off | GSM8K on | AIME off | AIME on |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Repack | none | 1.36 h | 3.15 h | 0.53 h | 0.80 h |
| Repack | yes | 🥇 0.77 h | 1.69 h | 🥇 0.26 h | 0.40 h |
| Google | none | 1.52 h | 3.32 h | 0.58 h | 0.83 h |
| Google | yes | 0.84 h | 1.89 h | 0.31 h | 0.54 h |

The drafter made every cell 1.55 to 2.04 times faster. The main model accepted 0.80 of the drafted tokens on GSM8K and 0.83 to 0.86 on AIME, and GSM8K generation rose from 99.1–109.2 tokens per second to 178.4–195.1.

Per problem, GSM8K took 2.11 seconds with the repack, the drafter and thinking off, and 9.05 seconds with Google's file, no drafter and thinking on.

---

#### Step 6 — Is Any Difference Bigger Than Chance?

The same problems ran in every cell, so each comparison is an exact paired test on the problems one setting got right and the other got wrong.

| Comparison, GSM8K | Right only before | Right only after | p |
| :--- | ---: | ---: | ---: |
| Thinking on, repack, no drafter | 51 | 40 | 0.2945 |
| Thinking on, repack, drafter | 40 | 39 | 1.0000 |
| Thinking on, Google, no drafter | 44 | 62 | 0.0982 |
| Thinking on, Google, drafter | 42 | 54 | 0.2615 |
| Drafter, largest change | 34 | 27 | 0.4426 |
| Google instead of repack, largest change | 29 | 45 | 0.0805 |

Thinking changed the answer on 79 to 106 GSM8K problems in each setup, in both directions, and the wins and losses come out close to even. On AIME no comparison does better than p = 0.125. The full table of 24 tests is in `RESULTS.md`.

---

#### 🔎 Tip: Turn the Drafter On Before You Turn Thinking On

Both change how long an answer takes. The drafter bought 1.55 to 2.04 times the speed and changed no answer beyond chance; thinking cost 1.44 to 2.31 times the time and bought no answer beyond chance. If a small local model is to think, serve it with a drafter so the extra tokens come out at nearly twice the speed.

---

#### Compare and Contrast

| | Thinking off | Thinking on |
| :--- | :--- | :--- |
| GSM8K score range | 0.8908 to 0.9014 | 0.8931 to 0.9052 |
| AIME right, of 30 | 5 to 7 | 6 to 10 |
| GSM8K time, with drafter | 🥇 0.77 to 0.84 h | 1.69 to 1.89 h |
| Thinking tokens per GSM8K problem | 🥇 0 | 601 to 615 |
| Score change beyond chance | — | none |

| | Repack | Google file |
| :--- | :--- | :--- |
| File size | 🥇 2.62 GB | 3.35 GB |
| Card memory, no drafter | 🥇 3,610 MiB | 4,306 MiB |
| Generation speed | 🥇 6.5 to 9.7 percent faster | — |
| Scores | level | level |

---

#### So, Which One?

For Gemma 4 E2B on a small GPU, turn the drafter on. It cut every run's time by 1.55 to 2.04 times on both builds and both tasks, cost 192 MiB of card memory, and changed no score.

Leave thinking off for grade-school math. On GSM8K it doubled the time and the score moved by no more than chance, down on one build and up on the other.

On hard math, thinking on gave zero to four more right AIME answers. That could be chance on 30 problems; measure on your own hard problems before paying 1.44 to 1.71 times the time.

Between the two 4-bit builds, take the repack: same answers, smaller file, less card memory, faster generation.

---

#### Summary

The goal of this article was to measure what thinking buys Gemma 4 E2B on a local GPU and what it costs in time. The key to the solution was holding the model and card fixed and crossing thinking with the weight build and the drafter, on the same 1,349 problems in every cell. The results were:

- ⚠️ Thinking took 2.18 to 2.31 times as long on GSM8K and 1.44 to 1.71 times on AIME.
- ❌ No thinking change in score was beyond chance: −11 to +18 GSM8K problems, zero to four AIME problems.
- 🟢 The drafter made every run 1.55 to 2.04 times faster and changed no score beyond chance.
- 🟢 The exact Q4_0 repack and Google's QAT file scored level; the repack is 0.73 GB smaller and 6.5 to 9.7 percent faster.
- ⚠️ With thinking off, E2B writes its working into the answer: 5,812 to 6,548 tokens per AIME reply.
- ⚠️ Two to five AIME replies per run hit the 32,768-token limit.

Scope: one RTX 3050 OEM with 8 GB in one desktop, llama.cpp `7182ef71e`, Gemma 4 E2B only, one run of each of 16 settings on 2026-11-16 and 17, temperature 0, one request at a time, the full 131,072-token context, a 32,768-token reply limit. GSM8K is all 1,319 test problems and AIME 2025 is papers I and II. Times are for this card and move with clock and temperature; the larger models, E4B and 12B, are measured separately.

The strategy for measuring thinking against a drafter on a local GPU was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [This run: replies, logs, analysis](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem)
- [The scoring script](https://github.com/xbill9/gemma4-dev/tree/main/benchmarks/gen-eval)
- [High or Low Thinking? Gemini 4 and Gemma 4 on Hard Math, Cost First](https://dev.to/gde/high-or-low-thinking-gemini-4-and-gemma-4-on-hard-math-cost-first-3091)
- [Exact Q4_0 repack of Gemma 4 E2B QAT](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
- [Google's Gemma 4 E2B QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [Exact Q4_0 E2B QAT drafter](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Hope you enjoyed it!
