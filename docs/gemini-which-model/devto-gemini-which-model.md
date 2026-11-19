---
title: "Which Gemini Model Should You Use? Neon, Argon, Hydrogen, Carbon, Flash or Gemma. It Depends."
published: false
description: "Gemini 4 Neon joins seven other models on GSM8K, BFCL and 30 AIME 2025 problems through the Gemini API, with cost per right answer, thinking and speed for each. Every Gemini 4 model solved all 30 hard problems at its lowest setting, so price and speed decide among them; Flash-Lite needs high thinking; Neon is the cheaper, faster choice against Carbon at high thinking."
tags: gemini, gemma, benchmark, ai
---

This article provides a step by step guide to choosing a Gemini API model by measured cost, thinking and speed. Gemini 4 Neon, announced on 2026-11-18, joins seven models measured here before: Gemma 4 26B and 31B, Gemini 3.5 Flash-Lite and 3.8 Flash, and Gemini 4 Hydrogen, Carbon and Argon. All eight answer the same questions through the same endpoint with the same scoring.

https://github.com/xbill9/gemma4-dev/tree/main/docs/gemini-which-model

The short version: on these tests the four Gemini 4 models are level on accuracy, so pick among them by cost and speed, and pick between tiers by how hard the work is.

---

#### The Findings

**Accuracy.** Hydrogen, Carbon, Argon and Neon solved all 30 AIME 2025 problems at their lowest thinking setting and again at their high setting. On these problems no Gemini 4 model can be ranked above another. The smaller models can: Flash-Lite solved 18 of 30 at its lowest setting, and Argon solved the 12 it missed and lost none (exact test, p < 0.001).

**Cost.** At the lowest setting a hard problem cost $0.018 on Argon, $0.026 on Hydrogen, $0.032 on Carbon and $0.049 on Neon. On short problems (GSM8K) the order is the same: $2.39 per 1,000 on Argon and $6.76 on Neon.

**Thinking.** Raising the setting bought nothing on Gemini 4 (30 of 30 either way) for 1.9 to 4.6 times the cost. It paid on the smallest paid model: Flash-Lite went from 18 to 24 of 30 (+6 / −0, p = 0.031) for 2.6 times the cost.

**Speed.** With every model running at the same time on AIME 2025 II, Hydrogen was the fastest Gemini 4 model at its lowest setting (11.3 s median per problem) and Neon the slowest (40.8 s).

**Neon against Carbon.** At high thinking, the setting Google compares them at, Neon cost 0.73 times Carbon per problem and took 0.68 times as long, with the same 15 of 15 on AIME 2025 II. Google states 32% lower cost per task than Carbon on the Artificial Analysis Intelligence Index.

**What to do with it.**

- 🟢 Default to Gemini 4 Argon at its lowest setting: every hard problem solved, the lowest cost of the Gemini 4 models.
- 🟢 Use Hydrogen when response time matters most, and from 2026-12-01, when Argon's price doubles, for cost too.
- 🟢 Use Neon in place of Carbon for work that needs high thinking.
- ⚠️ Use Flash-Lite only on easy work, or at high thinking.
- ⚠️ Use Gemma 4 when the model must be free or run on your own hardware.

**Caveats.** AIME 2025 has been public since early 2025, so some models may have seen these problems in training; that would raise every Gemini 4 score alike and is untested here. Google positions Neon and Carbon for long agent and coding work, which none of these tests measures.

---

#### What Was Measured

| Model | Price in / out, per 1M tokens | Lowest setting | High setting |
| :--- | ---: | :--- | :--- |
| Gemma 4 31B, 26B | free tier only | `minimal` | `high` |
| Gemini 3.5 Flash-Lite | $0.30 / $2.50 | `minimal` | `high` |
| Gemini 3.8 Flash | $0.75 / $3.75 | `low` | `high` |
| Gemini 4 Hydrogen | $1.50 / $7.50 | `low` | `xhigh` |
| Gemini 4 Carbon | $2.00 / $10.00 | `low` | `high` |
| Gemini 4 Argon | $2.00 / $10.00 | `low` | `high` |
| Gemini 4 Neon | $3.00 / $15.00 | `low` | `high` |

Output prices include thinking tokens. Carbon and Argon go to $4 / $20 on 2026-12-01. Neon is paid tier only. 3.8 Flash and the Gemini 4 models accept `none` and run it as `low`; Neon was sent `low` on AIME 2025 I and `none` elsewhere and thought on every problem either way. Hydrogen's high setting is its top, `xhigh`; Neon's is `high`, Google's default for it.

Three tests, each scored in code, temperature 0:

- **GSM8K**, 1,319 short math word problems, and **BFCL simple**, 400 function calls, at the lowest setting. Seven models ran together on 2026-11-14; Neon ran on 2026-11-19 with the same settings, alone.
- **AIME 2025 I**, 15 competition problems, 32,768-token limit, both settings, 2026-11-15 (Neon 2026-11-19).
- **AIME 2025 II**, the other 15, same limit, both settings, all models started at the same time on 2026-11-19. Its times are the ones compared below.

Costs are the tokens each reply reported times the list price.

---

#### Step 1 — Short Problems: Everyone Scores, Price Differs

| Model | GSM8K | BFCL | GSM8K, per 1,000 | BFCL, per 1,000 | Thinks on, GSM8K |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 31B | 0.9735 | 0.9675 | 🥇 free | 🥇 free | 0% |
| Gemma 4 26B | 0.9742 | 0.9575 | 🥇 free | 🥇 free | 0% |
| 3.5 Flash-Lite | 0.9629 | 0.9350 | $0.66 | $0.13 | 0% |
| 3.8 Flash | 0.9788 | 0.9400 | $1.26 | $0.36 | 98% |
| Gemini 4 Hydrogen | 0.9803 | 0.9550 | $3.42 | $1.26 | 100% |
| Gemini 4 Carbon | 0.9803 | 0.9600 | $2.81 | $0.78 | 20% |
| Gemini 4 Argon | 🥇 0.9841 | 0.9675 | $2.39 | $0.78 | 5% |
| Gemini 4 Neon | 0.9818 | 🥇 0.9750 | $6.76 | $2.87 | 100% |

Neon's GSM8K score is level with 3.8 Flash and every Gemini 4 model and ahead of both Gemma 4 models and Flash-Lite. Its BFCL score is ahead of Hydrogen, Carbon, 3.8 Flash and Flash-Lite and level with Argon and Gemma 4 31B. Each comparison is paired by question with a 95% range of the difference.

Hydrogen and Neon think on every request even at their lowest setting, and that sets the bill: Neon cost 2.8 times Argon per GSM8K problem.

---

#### Step 2 — Hard Problems: Gemini 4 Solves Them All

AIME 2025 I and II together, 30 problems:

| Model | Lowest setting | High setting |
| :--- | ---: | ---: |
| Gemma 4 31B | 12 of 15 (AIME I) | 13 of 15 (AIME I) |
| Gemma 4 26B | 9 of 15 (AIME I) | 11 of 15 (AIME I) |
| 3.5 Flash-Lite | 18 | 24 |
| 3.8 Flash | 27 | 29 |
| Gemini 4 Hydrogen | 🥇 30 | 🥇 30 |
| Gemini 4 Carbon | 🥇 30 | 🥇 30 |
| Gemini 4 Argon | 🥇 30 | 🥇 30 |
| Gemini 4 Neon | 🥇 30 | 🥇 30 |

Gemma 4 on the free tier is shown for AIME I only.

With 30 of 30 the 95% range on each Gemini 4 score is 0.89 to 1.00, so these problems show that all four solve them and cannot say which solves them best. 3.8 Flash at its lowest setting missed 3 that Argon solved; on 30 problems that could be chance (p = 0.25).

---

#### Step 3 — Thinking: Pay for It Only on the Small Model

| Model | High against lowest, 30 problems | Cost, high ÷ lowest |
| :--- | :--- | ---: |
| 3.5 Flash-Lite | +6 / −0, p = 0.031 | 2.63x |
| 3.8 Flash | +2 / −0, p = 0.50 | 3.45x |
| Gemini 4 Hydrogen | +0 / −0 | 4.61x |
| Gemini 4 Carbon | +0 / −0 | 3.76x |
| Gemini 4 Argon | +0 / −0 | 2.78x |
| Gemini 4 Neon | +0 / −0 | 1.86x |

Mean thinking per hard problem at the lowest setting was 118 tokens on Argon, 542 on Carbon, 2,314 on Neon and 2,406 on Hydrogen. Argon and Carbon decide on their own how much to think, and on most problems they barely do.

---

#### Step 4 — Speed: Hydrogen Answers First

Median seconds per AIME 2025 II problem, every model running at the same time:

| Model | Lowest setting | High setting |
| :--- | ---: | ---: |
| 3.5 Flash-Lite | 🥇 7.5 | 27.9 |
| 3.8 Flash | 18.3 | 89.0 |
| Gemini 4 Hydrogen | 11.3 | 61.2 |
| Gemini 4 Argon | 16.8 | 41.3 |
| Gemini 4 Carbon | 20.2 | 109.1 |
| Gemini 4 Neon | 40.8 | 73.7 |

On short problems, in the 2026-11-14 run of the first seven models together: Flash-Lite 1.10 s per GSM8K problem, Hydrogen 2.84 s, 3.8 Flash 3.79 s, Carbon 7.34 s and Argon 7.63 s.

---

#### Step 5 — Neon Against Carbon

| AIME 2025 II, high setting | Neon | Carbon | Neon ÷ Carbon |
| :--- | ---: | ---: | ---: |
| Right | 15 of 15 | 15 of 15 | |
| Cost per problem | $0.080 | $0.110 | 0.73x |
| From 2026-12-01 | $0.080 | $0.220 | 0.36x |
| Median time | 73.7 s | 109.1 s | 0.68x |
| Thinking per problem | 4,547 | 9,718 | 0.47x |

Neon reaches the same answers as Carbon with half the thinking. Argon at the same setting cost $0.048 a problem and took 41.3 s for the same 15.

---

#### 🔎 Tip: Compare Models by Cost per Right Answer

List price per token misleads. Hydrogen has the lowest Gemini 4 price and Argon cost less per hard problem at the lowest setting ($0.018 against $0.026), because Argon thought on few problems and Hydrogen on all of them. Run 20 to 50 of your own requests on each candidate and divide the bill by the number right.

---

#### Compare and Contrast

| | Flash-Lite | 3.8 Flash | Hydrogen | Argon | Carbon | Neon | Gemma 4 31B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Hard problems, lowest | 18 / 30 | 27 / 30 | 30 / 30 | 30 / 30 | 30 / 30 | 30 / 30 | 12 / 15 |
| Cost per hard problem, lowest | 🥇 $0.013 | $0.015 | $0.026 | $0.018 | $0.032 | $0.049 | 🥇 free |
| From 2026-12-01 | $0.013 | $0.015 | $0.026 | $0.036 | $0.064 | $0.049 | free |
| Time per hard problem, lowest | 🥇 7.5 s | 18.3 s | 11.3 s | 16.8 s | 20.2 s | 40.8 s | AIME I only |
| Thinks on every request | no | most | yes | rarely | sometimes | yes | no |
| Best function-calling score | 0.9350 | 0.9400 | 0.9550 | 0.9675 | 0.9600 | 🥇 0.9750 | 0.9675 |

---

#### So, Which One?

| Work | Use | Why, measured |
| :--- | :--- | :--- |
| Hard reasoning, cost first | Gemini 4 Argon, lowest | 30 of 30, $0.018 a problem |
| Hard reasoning, speed first | Gemini 4 Hydrogen, lowest | 30 of 30, 11.3 s a problem |
| Hard reasoning from 2026-12-01 | Gemini 4 Hydrogen, lowest | 30 of 30, $0.026 against Argon's $0.036 |
| Work that needs high thinking | Gemini 4 Neon, high | Carbon's score at 0.73x the cost, 0.68x the time |
| Function calling, best score | Neon, or Argon for less | 0.9750 at $2.87 per 1,000; Argon 0.9675 at $0.78 |
| Short math, lowest paid cost | 3.8 Flash | GSM8K 0.9788 at $1.26 per 1,000 |
| Fastest reply on easy work | 3.5 Flash-Lite | 1.10 s per GSM8K problem, $0.66 per 1,000 |
| Free, private or offline | Gemma 4 31B | GSM8K 0.9735, BFCL 0.9675, free tier, or on your own GPU |

---

#### Summary

The goal of this article was to place every Gemini API model, Neon included, by what it gets right, what it costs and how fast it answers. The key to the solution was one scoring script, one endpoint and the same questions for all eight models. The results were:

- 🟢 Hydrogen, Carbon, Argon and Neon solved all 30 AIME 2025 problems at their lowest setting.
- 🟢 Argon cost the least of them per problem, $0.018; Hydrogen answered fastest, 11.3 s.
- 🟢 Neon matched Carbon at high thinking for 0.73 times the cost and 0.68 times the time.
- 🟢 Neon has the best function-calling score, 0.9750, level with Argon and Gemma 4 31B.
- 🟢 High thinking lifted Flash-Lite from 18 to 24 of 30.
- ⚠️ High thinking added nothing on Gemini 4 for 1.9 to 4.6 times the cost.
- ⚠️ Neon thinks on every request and cost 2.7 to 2.8 times Argon at the lowest setting.
- ❌ These problems cannot rank the Gemini 4 models on accuracy.

Scope: one run per model and setting, temperature 0, through the Gemini API's OpenAI-compatible endpoint, between 2026-11-14 and 2026-11-19, at list prices on those dates. GSM8K and BFCL simple at the lowest setting only; 30 AIME 2025 problems at two settings, with Gemma 4 on AIME 2025 I only. Times compare only within one run. Long agent work, coding, and tests harder than AIME 2025 were not measured.

The strategy for choosing a Gemini model by measured cost, thinking and speed was validated with an incremental step by step approach.

---

#### References

- [Evidence for this article](https://github.com/xbill9/gemma4-dev/tree/main/docs/gemini-which-model/evidence)
- [Introducing Gemini 4 Neon](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-neon/)
- [Gemini API pricing](https://aistudio.google.com/docs/pricing)
- [Gemini 4 or Gemma 4? It Depends.](https://dev.to/gde/gemini-4-or-gemma-4-it-depends-2o18)
- [Which Model Should You Use? Frontier, Hosted or Local? It Depends.](https://dev.to/gde/which-model-should-you-use-frontier-hosted-or-local-it-depends-2n45)
- [AIME 2025 problems (opencompass/AIME2025)](https://huggingface.co/datasets/opencompass/AIME2025)
- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)

Hope you enjoyed it!
