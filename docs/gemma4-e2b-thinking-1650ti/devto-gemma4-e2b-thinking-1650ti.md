---
title: "Thinking on Constrained Deployments: Gemma E2B on 4GB 1650"
published: false
description: "The RTX 3050 thinking study, repeated on a 4 GB GTX 1650 Ti laptop with the same files and llama.cpp build: thinking off and on, two QAT 4-bit builds, with and without a drafter, on all 1,319 GSM8K problems and AIME 2025. Every score matched the 8 GB card within chance; the run took 1.63 times as long, and the drafter paid 1.37 to 1.39 times."
tags: gemma, llm, benchmark, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-e2b-thinking-1650ti/devto-cover.bf86abbb.jpg
---

A 4 GB laptop card gives the same answers as an 8 GB desktop card, at 1.63 times the time. On grade-school math, thinking doubles the time and leaves the score where it was.

The same 16 settings that ran on an 8 GB RTX 3050 ran on a GTX 1650 Ti Max-Q, through the same llama.cpp build and the same weight files. Every one scored within chance of the 3050: −11 to +12 GSM8K problems (smallest p = 0.1882) and −2 to +3 AIME problems (smallest p = 0.25). The reply text differs between the cards; the right-or-wrong verdicts match.

Thinking on made GSM8K take 2.04 to 2.21 times as long and moved no score beyond chance. The drafter lifted generation 1.37 to 1.39 times for 192 MiB of card memory and changed no score. The exact Q4_0 repack scored level with Google's file, in 118 MiB less card memory and 8.2 to 9.8 percent faster.

For a constrained card: serve E2B with the repack weights and the drafter at one drafted token, leave thinking off, and pick the card by how long you can wait. Each setting ran once, at temperature 0.

This article provides a step by step guide to running that study on a 4 GB laptop GPU and checking its answers against a bigger card: thinking off and on, two 4-bit builds of Google's QAT weights, each with and without a drafter, on every GSM8K test problem and all 30 AIME 2025 problems.

https://github.com/xbill9/gemma4-dev/tree/main/docs/gemma4-e2b-thinking-1650ti

---

#### The Findings

**Same answers as the 8 GB card.** GSM8K scored 0.8863 to 0.9030 here against 0.8908 to 0.9052 on the RTX 3050. Setting by setting, the two cards differ by −11 to +12 GSM8K problems and −2 to +3 AIME problems, and no paired test between them gets below p = 0.1882 on GSM8K or 0.25 on AIME.

**Different text, same verdicts.** Only 151 to 190 of the 1,319 GSM8K replies with thinking off came out byte-identical on the two cards, and 5 to 13 with thinking on. The right-or-wrong verdict matched on 1,249 to 1,268 of them.

**What thinking cost.** GSM8K took 2.04 to 2.21 times as long with thinking on, and AIME 1.43 to 1.70 times. GSM8K moved by 0, +8, +13 and +17 problems across the four setups, none beyond chance (p = 0.1145 to 1.0).

**What the drafter bought.** Generation rose 1.37 to 1.39 times on GSM8K, from 69.6–76.7 to 96.1–106.1 tokens per second, against 1.79 to 1.82 times on the RTX 3050. It changed no score beyond chance.

**The two weight builds.** The exact Q4_0 repack and Google's QAT Q4_0 file scored level in every paired test. The repack used 118 MiB less card memory and generated 8.2 to 9.8 percent faster on GSM8K.

**What to do with it.**

- 🟢 A 4 GB laptop card answers like the 8 GB desktop card. Pick the card by time, not by score.
- 🟢 Serve E2B with the drafter; at one drafted token it adds 192 MiB and 1.37 to 1.39 times the speed.
- ⚠️ Leave thinking off for grade-school math; it doubles the time on this card too.

**Caveats.** One run of each setting, temperature 0. AIME's 30 problems can only show large effects.

---

#### What Was Measured

The earlier article, [Thinking On or Off? Gemma 4 E2B on an 8 GB RTX 3050](https://dev.to/gde/thinking-on-or-off-gemma-4-e2b-on-an-8-gb-rtx-3050-30e0), crossed thinking, weight build and drafter on a desktop card. This run repeats it on a laptop with half the card memory and asks two questions: does the smaller card change any answer, and what does it cost in time?

| Factor | Settings |
| :--- | :--- |
| Thinking | off, on (`enable_thinking` in the chat template) |
| Weights | exact Q4_0 repack of Google's QAT weights, Google's QAT Q4_0 GGUF |
| Drafter | none, Google's E2B QAT assistant as exact Q4_0, 1 drafted token |
| Problems | GSM8K test, 1,319; AIME 2025 I and II, 30 |

Two settings are this card's own. The drafter guesses one token at a time, the best length measured earlier on the 1650 Ti, where the 3050 used two. And E2B's per-layer embedding table stays in system memory, this rig's serving setting.

That is 16 runs, 30.63 hours, every reply saved.

---

#### At This Point You Should Have…

- An NVIDIA card with 4 GB, here a GTX 1650 Ti with Max-Q Design on driver 620.70.06
- llama.cpp built with CUDA, here commit `7182ef7`, the build the 3050 run used
- Python 3 with `huggingface_hub`, `tokenizers` and `scipy`
- A clone of [gemma4-dev](https://github.com/xbill9/gemma4-dev)

---

#### Step 1 — Same Files, Smaller Card

Both model files are the ones the 3050 served, checked by SHA-256 in `weights.sha256`:

| File | Bytes | Card memory when serving |
| :--- | ---: | ---: |
| Repack, `gemma-4-E2B-it-q4_0-exact.gguf` | 2,620,370,912 | 🥇 2,331 MiB |
| Google, `gemma-4-E2B_q4_0-it.gguf` | 3,349,516,256 | 2,449 MiB |
| Drafter, `gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf` | 59,236,000 | +192 MiB |

Every arm fits the 4,096 MiB card with the full 131,072-token context, one request at a time; the largest, Google's file with the drafter, used 2,641 MiB. Each reply may run to 32,768 tokens, thinking included.

---

#### Step 2 — Run All Sixteen on a Laptop

```bash
cd local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-11-23-thinking-e2b-1650ti
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why="E2B thinking run" \
  ./run.sh >> nohup.out 2>&1 < /dev/null &
```

`systemd-inhibit` keeps the laptop awake with the lid closed for the 30 hours. `run.sh` starts the server once per weights and drafter pair and runs four cells on it: GSM8K thinking off, AIME off, AIME on, GSM8K on. It saves each reply as it finishes, skips any cell already complete when restarted, and commits and pushes each pair as it ends.

The run's own log, `driver.log`, shows one line per cell, with the card's temperature and clock:

```
2026-11-23 15:02:24 64.31, 1845 MHz END gsm8k repack-none thinking=off rc=0 {"task": "gsm8k", "max_tokens": 32768, "n": 1319, "right": 1182, "accuracy": 0.8961, "errors": 0, "truncated": 1, ...
```

---

#### Step 3 — The Scores, Beside the 3050

`analyze.py` builds every table from the saved replies.

**GSM8K, 1,319 problems: 1650 Ti / RTX 3050**

| Weights | Drafter | Thinking off | Thinking on |
| :--- | :--- | ---: | ---: |
| Repack | none | 0.8961 / 0.9014 | 0.9022 / 0.8931 |
| Repack | yes | 0.8931 / 0.9007 | 🥇 0.9030 / 0.8999 |
| Google | none | 0.8969 / 0.8916 | 0.8969 / 0.9052 |
| Google | yes | 0.8863 / 0.8908 | 0.8992 / 0.8999 |

**AIME 2025, 30 problems, right answers: 1650 Ti / RTX 3050**

| Weights | Drafter | Thinking off | Thinking on |
| :--- | :--- | ---: | ---: |
| Repack | none | 10 / 7 | 8 / 10 |
| Repack | yes | 7 / 7 | 8 / 10 |
| Google | none | 5 / 6 | 9 / 6 |
| Google | yes | 6 / 5 | 7 / 9 |

One to five AIME replies per cell ran into the 32,768-token limit and count as wrong; on GSM8K, zero or one per cell.

The 10 against 7 is three problems right only on the 1650 Ti and none the other way; on two of them the 3050 reply ran into the length limit. That split is p = 0.25.

---

#### Step 4 — Same Problems, Two Cards

`agreement.py` lines up each problem's reply from the two cards in the same setting.

| GSM8K | Thinking off | Thinking on |
| :--- | ---: | ---: |
| Same right-or-wrong verdict | 1,259 to 1,268 | 1,249 to 1,262 |
| Byte-identical reply | 151 to 190 | 5 to 13 |
| Paired test, smallest p | 0.2203 | 0.1882 |

Same files, same llama.cpp commit, temperature 0, and still most replies differ in their wording. The likely cause is that the two GPUs add up floating-point numbers in a different order, so the greedy pick flips wherever two candidate tokens are nearly tied, and a longer reply has more such points. No AIME reply matched byte for byte. The verdicts land the same way within chance: right on one card and wrong on the other 51 to 70 times on GSM8K, split close to evenly.

---

#### Step 5 — Where the Tokens Go

`tokens.py` counts thinking and answer tokens in each saved reply with Gemma 4 E2B's tokenizer. Its count matches the server's on every reply, up to the fixed end-of-turn and thinking markers.

| | GSM8K off | GSM8K on | AIME off | AIME on |
| :--- | ---: | ---: | ---: | ---: |
| Thinking tokens, mean | 0 | 601 to 633 | 0 | 7,075 to 9,113 |
| Answer tokens, mean | 404 to 427 | 272 to 280 | 5,425 to 6,497 | 919 to 1,735 |

The pattern matches the 3050. Thinking moves the working out of the answer and adds to it: about 600 thinking tokens in front of a GSM8K answer a third shorter. Thinking on used more tokens on 1,299 to 1,303 of the 1,319 GSM8K problems and on 26 to 28 of 30 AIME problems in every setup.

---

#### Step 6 — Time and the Drafter

| Weights | Drafter | GSM8K off | GSM8K on | AIME off | AIME on |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Repack | none | 2.15 h | 4.39 h | 0.67 h | 0.96 h |
| Repack | yes | 🥇 1.52 h | 3.25 h | 0.59 h | 0.99 h |
| Google | none | 2.23 h | 4.93 h | 0.85 h | 1.31 h |
| Google | yes | 1.74 h | 3.61 h | 🥇 0.53 h | 0.91 h |

With the drafter, GSM8K finished 1.28 to 1.41 times sooner. The main model accepted 0.863 to 0.871 of the drafted tokens on GSM8K, more than the 3050's 0.796 to 0.806 at two tokens, yet the speed-up is smaller: 1.37 to 1.39 times against 1.79 to 1.82. An earlier measurement on these rigs found a drafting step costs the 1650 Ti a larger share of a plain step than faster cards pay, which fits a smaller gain here.

On AIME, generation rose 1.29 to 1.39 times, but the time follows the length of the replies as much as the speed. The repack with thinking on took 0.99 hours with the drafter and 0.96 without, because its replies with the drafter ran 10,789 tokens on average against 8,096.

Without the drafter this card generates at 0.68 to 0.71 times the 3050; with it, at 0.49 to 0.54 times, since the 3050 drafts two tokens and gains more from them.

---

#### 🔎 Tip: Compare Verdicts Across Cards, Not Text

Two GPUs with the same model file, the same engine build and temperature 0 gave word-for-word the same GSM8K reply on at most 190 of 1,319 problems. A check that diffs reply text between machines will report almost everything as changed. Score each reply, then pair the verdicts problem by problem: that is the comparison that held steady here.

---

#### Compare and Contrast

| | GTX 1650 Ti, 4 GB | RTX 3050, 8 GB |
| :--- | :--- | :--- |
| GSM8K score range | 0.8863 to 0.9030 | 0.8908 to 0.9052 |
| AIME right, of 30 | 5 to 10 | 5 to 10 |
| All 16 settings | 30.63 h | 🥇 18.81 h |
| Generation, no drafter | 69.6 to 76.7 tok/s | 🥇 99.1 to 109.2 tok/s |
| Drafter speed-up, GSM8K | 1.37 to 1.39x (1 token) | 🥇 1.79 to 1.82x (2 tokens) |
| Card memory, repack, no drafter | 🥇 2,331 MiB | 3,610 MiB |

| | Thinking off | Thinking on |
| :--- | :--- | :--- |
| GSM8K time, repack with drafter | 🥇 1.52 h | 3.25 h |
| Thinking tokens per GSM8K problem | 🥇 0 | 601 to 633 |
| Score change beyond chance | — | none |

---

#### So, Which One?

If a 4 GB laptop card is what you have, it gives the same answers as an 8 GB desktop card. Every setting scored within chance of the 3050; what you pay is time, 1.63 times as much across the whole study.

Turn the drafter on at one drafted token. It costs 192 MiB and lifts generation 1.37 to 1.39 times on this card.

Leave thinking off for grade-school math. On GSM8K it doubled the time here as on the 3050, with no change in score beyond chance; on hard math, measure on your own problems first.

Between the two 4-bit builds, take the repack: same answers, less card memory, faster generation.

---

#### Summary

The goal of this article was to find out whether a 4 GB laptop card answers like an 8 GB desktop card for Gemma 4 E2B with thinking off and on, and what it costs in time. The key to the solution was running the same 16 settings with the same files and llama.cpp build on both cards and pairing every problem's verdict between them. The results were:

- 🟢 Every setting scored within chance of the RTX 3050: −11 to +12 GSM8K problems, −2 to +3 AIME problems.
- ⚠️ Only 5 to 190 of 1,319 GSM8K replies came out word-for-word the same on the two cards; the verdicts agreed on 1,249 to 1,268.
- ⚠️ The study took 30.63 hours against 18.81, 1.63 times as long.
- ❌ Thinking doubled the GSM8K time (2.04 to 2.21 times) and moved no score beyond chance.
- 🟢 The drafter lifted generation 1.37 to 1.39 times on GSM8K and changed no score beyond chance.
- 🟢 The repack and Google's QAT file scored level; the repack used 118 MiB less and generated 8.2 to 9.8 percent faster.

Scope: one GTX 1650 Ti Max-Q with 4 GB in one laptop on mains power, llama.cpp `7182ef7`, Gemma 4 E2B only, one run of each of 16 settings on 2026-11-23 and 24, temperature 0, one request at a time, the full 131,072-token context, a 32,768-token reply limit. The drafter guessed one token here and two on the 3050, and the per-layer table stayed in system memory here and sat on the card on the 3050; both are each card's own setting, so the time comparisons are between the two setups as served.

The strategy for measuring thinking and a drafter on a 4 GB laptop GPU was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [This run: replies, logs, analysis](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-11-23-thinking-e2b-1650ti)
- [The RTX 3050 run](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem)
- [Thinking On or Off? Gemma 4 E2B on an 8 GB RTX 3050](https://dev.to/gde/thinking-on-or-off-gemma-4-e2b-on-an-8-gb-rtx-3050-30e0)
- [The scoring script](https://github.com/xbill9/gemma4-dev/tree/main/benchmarks/gen-eval)
- [Exact Q4_0 repack of Gemma 4 E2B QAT](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
- [Google's Gemma 4 E2B QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [Exact Q4_0 E2B QAT drafter](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Hope you enjoyed it!
