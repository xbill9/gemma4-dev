---
title: "High or Low Thinking? Gemini 4 and Gemma 4 on Hard Math, Cost First"
published: false
description: "Seven models on the 15 problems of AIME 2025 I at their lowest and highest thinking settings, under a hard budget. The top setting cost 2.6 to 4.8 times as much on every paid model; Hydrogen, Carbon and Argon solved all 15 at either setting, and no score change was bigger than the run-to-run noise."
tags: gemini, gemma, benchmark, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemini-high-vs-low/devto-cover.25535354.jpg
---

This article provides a step by step guide to measuring what extra thinking buys on hard problems, and what it costs. Seven models, Gemma 4 26B and 31B, Gemini 3.5 Flash-Lite and 3.8 Flash, and Gemini 4 Hydrogen, Carbon and Argon, answer the same 15 competition math problems twice through the Gemini API: once at the lowest thinking setting each accepts and once at the highest.

https://github.com/xbill9/gemma4-dev/tree/main/docs/gemini-high-vs-low

The short version: the highest thinking setting cost about four times as much as the lowest and bought almost nothing on these problems.

---

#### The Findings

**Cost.** Both settings together cost $7.47: $1.53 at the lowest setting and $5.95 at the highest, 3.9 times as much. The run was budgeted for a worst case of $33.23.

| Model | Highest ÷ lowest |
| :--- | ---: |
| Gemini 3.5 Flash-Lite | 2.61x |
| Gemini 4 Argon | 3.24x |
| Gemini 4 Carbon | 3.93x |
| Gemini 3.8 Flash | 4.20x |
| Gemini 4 Hydrogen | 4.75x |

The extra cost is extra thinking: mean thinking tokens per problem went from 215 to 4,129 on Argon and from 2,468 to 15,062 on Hydrogen. The highest setting also took 3.2 to 5.9 times as long per problem on the paid models.

**What it bought on Gemini 4.** Nothing. Hydrogen, Carbon and Argon solved all 15 problems at both settings, so the extra cost bought the same 15 answers.

**What it bought on the smaller models.** Flash-Lite, 3.8 Flash and both Gemma 4 models solved one or two more problems each. On 15 problems a gain has to be about 35 points to stand out from run-to-run noise; the largest gain was 13 points, two problems, so these gains could be chance.

**Cheapest right answer.** Argon at its lowest setting, $0.0162 per problem. Hydrogen and Carbon at their highest settings cost $0.1232 and $0.1297 per right answer, 7.6 and 8.0 times as much, for the same score.

**What to do with it.**

- 🟢 For hard math on Gemini 4, start at the lowest setting.
- ⚠️ For the smaller models, test the highest setting on your own problems before paying 2.6 to 4.2 times as much.
- ⚠️ Carbon's and Argon's prices double on 2026-12-01, which widens the gap.

**Caveats.** One run of 15 problems shows what extra thinking costs; it cannot rank models that finish within two problems of each other. Google positions Carbon's highest setting for long multi-step work such as debugging and architecture, which this run did not test.

---

#### What Was Measured

The earlier comparison, [Gemini 4 or Gemma 4? It Depends.](https://dev.to/gde/gemini-4-or-gemma-4-it-depends-2o18), ran 1,319 GSM8K math problems and 400 BFCL function calls at the lowest thinking setting, where Argon scored 0.9841 and Gemma 4 31B 0.9735. Those problems are easy enough that more thinking has little room to help. This run uses harder problems and both ends of the thinking range.

| Model | Lowest setting | Highest setting | Output price per 1M tokens |
| :--- | :--- | :--- | ---: |
| Gemma 4 31B, 26B | `minimal` | `high` | free tier |
| Gemini 3.5 Flash-Lite | `minimal` | `high` | $2.50 |
| Gemini 3.8 Flash | `none` | `high` | $3.75 |
| Gemini 4 Hydrogen | `none` | `xhigh` | $7.50 |
| Gemini 4 Carbon | `none` | `high` | $10.00 |
| Gemini 4 Argon | `none` | `high` | $10.00 |

Output prices include thinking tokens. Carbon and Argon go to $20 per million output tokens on 2026-12-01.

---

#### At This Point You Should Have…

- A Gemini API key in `~/gemini.key`, on a paid tier for the Gemini 4 Carbon and Argon models
- Python 3 with `huggingface_hub` and `scipy`
- A clone of [gemma4-dev](https://github.com/xbill9/gemma4-dev)

---

#### Step 1 — Fifteen Hard Problems

The problems are AIME 2025 I, all 15, from the `opencompass/AIME2025` dataset at a pinned revision. Each answer is a whole number from 0 to 999. One answer is published as `336^\circ` and is read as 336.

```bash
cd benchmarks/gen-eval
python3 build_hard_set.py aime
```

```
wrote /home/xbill/gemma4-dev/benchmarks/gen-eval/data/aime2025.jsonl: 30 records
```

The file holds AIME 2025 I and II with a SHA-256 in `data/hard_manifest.json`; this run uses the first 15, paper I. The scoring script, [`benchmarks/gen-eval/gen_eval.py`](https://github.com/xbill9/gemma4-dev/tree/main/benchmarks/gen-eval), asks for the final answer on a line of its own as `Answer: <number>` and marks a reply right when that number matches.

---

#### Step 2 — The Top Thinking Level for Each Model

Google's thinking page lists `xhigh` for Hydrogen only. One small request per model and level confirms what the OpenAI-compatible endpoint accepts:

```bash
docs/gemini-high-vs-low/evidence/top-levels.sh
```

```
gemma-4-31b-it         high   ok  thinking 225 answer 27 finish length
gemma-4-26b-a4b-it     high   ok  thinking 253 answer 0 finish length
gemini-3.5-flash-lite  high   ok  thinking 243 answer 9 finish length
gemini-3.8-flash       high   ok  thinking 118 answer 66 finish stop
gemini-4-hydrogen      high   ok  thinking 80 answer 67 finish stop
gemini-4-hydrogen      xhigh  ok  thinking 150 answer 78 finish stop
gemini-4-carbon        high   ok  thinking 253 answer 3 finish length
gemini-4-carbon        xhigh  rejected: Thinking level XHIGH is not supported for this model. Please retry with other thinking level.
gemini-4-argon         high   ok  thinking 241 answer 15 finish length
gemini-4-argon         xhigh  rejected: Thinking level XHIGH is not supported for this model. Please retry with other thinking level.
```

The lowest settings are the ones the earlier run used: `minimal` for Gemma and Flash-Lite, `none` for the rest.

---

#### Step 3 — A Budget the Run Cannot Exceed

Thinking has no limit of its own. The only cap on what one request can cost is its output limit, because thinking and answer share it. So the most a run can cost is known before it starts: every reply using its whole limit, times the output price, plus the input.

Every model gets the same limit, 32,768 tokens, which is the most Carbon and Argon produce in one non-streaming request. `budget_run.py` prints that worst case, refuses a run whose worst case is over the budget, and sends the problems in rounds of three to every model at once, checking spend plus the next round's worst case before each round.

```bash
python3 budget_run.py models-high.json aime2025 aime-high --budget 20 --limit 15 --dry-run
```

```
15 questions x 7 models, output limit 32,768: worst case $16.62, budget $20.00
```

Two settings at $16.62 each is $33.23 in the worst case, under a combined $40.

---

#### Step 4 — Run Both Settings

```bash
docs/gemini-high-vs-low/evidence/run.sh high low
```

The last line of each arm's log, `aime-high.log` and `aime-low.log`:

```
{"task": "aime2025", "spent_usd": 5.9455, "budget_usd": 20.0, "worst_case_usd": 16.62, "wall_s": 9749.6}
{"task": "aime2025", "spent_usd": 1.5255, "budget_usd": 20.0, "worst_case_usd": 16.62, "wall_s": 12953.1}
```

Every request has a 20-minute limit and up to three retries. Two Gemma 4 26B requests on the free tier timed out on all four tries, problem 2 at `minimal` and problem 13 at `high`; both count as wrong. Every paid model answered every problem at both settings.

---

#### Step 5 — The Scores

`derived.py` builds every table from the saved replies.

| Model | Right, lowest | Right, highest | Gained / lost at highest |
| :--- | ---: | ---: | :--- |
| Gemma 4 31B | 12 | 13 | +2 / −1 |
| Gemma 4 26B | 9 | 11 | +4 / −2 |
| Gemini 3.5 Flash-Lite | 9 | 11 | +2 / −0 |
| Gemini 3.8 Flash | 12 | 14 | +2 / −0 |
| Gemini 4 Hydrogen | 🥇 15 | 🥇 15 | none |
| Gemini 4 Carbon | 🥇 15 | 🥇 15 | none |
| Gemini 4 Argon | 🥇 15 | 🥇 15 | none |

Problem 14 is the one that separates the generations: Hydrogen, Carbon and Argon solved it at both settings, and the other four models missed it at both.

Replies that ran into the 32,768-token limit were almost all Gemma's: Gemma 4 26B four times at `minimal` and three at `high`, Gemma 4 31B once and twice, and 3.8 Flash once at `high`. At `minimal` the Gemma replies hit the limit while writing the answer; at `high` they hit it while thinking.

---

#### Step 6 — What the Extra Thinking Cost

| Model | Cost, lowest | Cost, highest | Highest ÷ lowest | Mean thinking tokens, lowest → highest |
| :--- | ---: | ---: | ---: | :--- |
| Gemini 3.5 Flash-Lite | $0.1937 | $0.5049 | 🥇 2.61x | 0 → 12,624 |
| Gemini 3.8 Flash | $0.2053 | $0.8616 | 4.20x | 2,801 → 14,210 |
| Gemini 4 Hydrogen | $0.3893 | $1.8485 | 4.75x | 2,468 → 15,062 |
| Gemini 4 Carbon | $0.4947 | $1.9459 | 3.93x | 886 → 11,613 |
| Gemini 4 Argon | $0.2425 | $0.7847 | 3.24x | 215 → 4,129 |

Costs are for all 15 problems, from the token counts the API returned times list prices.

Cost per right answer moves the same way. At the lowest setting it was $0.0162 on Argon, $0.0260 on Hydrogen and $0.0330 on Carbon; at the highest, $0.0523, $0.1232 and $0.1297, for the same 15 right answers.

Hydrogen at `none` still thought on these problems, 2,468 tokens per problem on average, as it did on GSM8K. Argon thought least at both settings, and at its top setting it thought less than the others did at theirs: 4,129 tokens per problem against 11,613 to 15,742.

---

#### Step 7 — Is the Difference Bigger Than the Noise?

The audit follows the `ablation-stat-auditor` skill from research-kit, a Claude Code plugin for checking research statistics: an exact paired test on the same problems for the scores, a signed-rank test on each problem's output tokens, and 95% Wilson intervals, all computed in `audit_stats.py`.

| Model | Right, highest [95% range] | Right, lowest [95% range] | Paired test on scores | Output tokens, highest / lowest |
| :--- | :--- | :--- | :--- | :--- |
| Gemma 4 31B | 13 [62.1, 96.3] | 12 [54.8, 93.0] | p = 1 | 182,736 / 79,971, p = 6.1e-05 |
| Gemma 4 26B | 11 [48.0, 89.1] | 9 [35.8, 80.2] | p = 0.688 | 246,381 / 211,934, p = 0.389 |
| Gemini 3.5 Flash-Lite | 11 [48.0, 89.1] | 9 [35.8, 80.2] | p = 0.5 | 201,684 / 77,200, p = 0.000183 |
| Gemini 3.8 Flash | 14 [70.2, 98.8] | 12 [54.8, 93.0] | p = 0.5 | 229,292 / 54,290, p = 6.1e-05 |
| Gemini 4 Hydrogen | 15 [79.6, 100.0] | 15 [79.6, 100.0] | p = 1 | 245,907 / 51,350, p = 6.1e-05 |
| Gemini 4 Carbon | 15 [79.6, 100.0] | 15 [79.6, 100.0] | p = 1 | 194,037 / 48,915, p = 6.1e-05 |
| Gemini 4 Argon | 15 [79.6, 100.0] | 15 [79.6, 100.0] | p = 1 | 77,914 / 23,694, p = 6.1e-05 |

Ranges are percentages. With 15 problems, a gain has to be about 25 points against a fixed score, or 35 points between two runs, to stand out from the noise. The largest gain here, two problems, is 13 points. So the score changes are within the noise for every model, and the extra tokens are bigger than the noise for every model except Gemma 4 26B, which used more tokens at the top setting on 11 of 15 problems because four of its lowest-setting answers ran to the limit too.

---

#### Step 8 — Time

| Model | Median seconds per problem, lowest | Highest |
| :--- | ---: | ---: |
| Gemini 3.5 Flash-Lite | 9.8 | 32.8 |
| Gemini 4 Hydrogen | 13.5 | 73.2 |
| Gemini 4 Argon | 18.0 | 57.4 |
| Gemini 4 Carbon | 22.7 | 134.3 |
| Gemini 3.8 Flash | 22.9 | 96.2 |
| Gemma 4 31B | 62.5 | 291.6 |
| Gemma 4 26B | 476.2 | 358.8 |

The top setting took 3.2 to 5.9 times as long on the paid models. The free-tier Gemma models were the slowest at both settings, and the run's length was set by them.

---

#### 🔎 Tip: Budget for the Worst Case, Then Measure the Real One

Before the run, the only number that can be known is the worst case: problems times models times output limit times price. It came to $33.23 here, and the run spent $7.47 of it. Set the output limit from the budget, and send problems in rounds so a stop leaves every model on the same problems.

---

#### Compare and Contrast

| | Lowest setting | Highest setting |
| :--- | :--- | :--- |
| Both arms, 7 models | 🥇 $1.53 | $5.95 |
| Gemini 4 scores | 🥇 15, 15, 15 | 🥇 15, 15, 15 |
| Smaller models' scores | 9 to 12 | 11 to 14 |
| Extra tokens beyond the noise | — | every paid model |
| Median time, paid models | 🥇 9.8 to 22.9 s | 32.8 to 134.3 s |

---

#### What 1,000 Problems Like These Would Cost

Arithmetic from the measured cost per problem, at list prices:

| Model | Lowest | Highest |
| :--- | ---: | ---: |
| Gemini 3.5 Flash-Lite | $12.91 | $33.66 |
| Gemini 3.8 Flash | $13.69 | $57.44 |
| Gemini 4 Argon | $16.16 | $52.31 |
| Gemini 4 Hydrogen | $25.95 | $123.23 |
| Gemini 4 Carbon | $32.98 | $129.73 |

At the 2026-12-01 prices, Carbon becomes $65.96 and $259.45 per 1,000, and Argon $32.33 and $104.62. Gemma 4 is $0 on the free tier, at the free tier's speed and rate limits.

---

#### So, Which One?

For hard math on the Gemini 4 models, start at the lowest setting. Hydrogen, Carbon and Argon solved all 15 problems there, and the top setting added 3.24 to 4.75 times the cost and 3.2 to 5.9 times the time for the same answers.

For the smaller models, the top setting gained one or two problems each, within the noise on 15 problems, for 2.61 times the cost on Flash-Lite and 4.20 times on 3.8 Flash. Measure on your own problems before paying that.

For the lowest cost per right answer on this set, Argon at its lowest setting: $0.0162. It also thought least at its top setting.

Google positions Carbon's top setting for hard, multi-step work such as architecture, debugging and security audits. Fifteen math problems that every Gemini 4 model solved at the lowest setting leave that case untested.

---

#### Summary

The goal of this article was to measure what the highest thinking setting buys on hard problems and what it costs. The key to the solution was running all seven models at both ends of their thinking range on the same 15 AIME 2025 problems, under a budget whose worst case was known before the first request. The results were:

- 🟢 Hydrogen, Carbon and Argon solved all 15 problems at both their lowest and highest settings.
- ⚠️ The highest setting cost 2.61 to 4.75 times the lowest on every paid model, $5.95 against $1.53 for all seven.
- ⚠️ No model's score change was bigger than the run-to-run noise; the largest gain was two problems.
- 🟢 Argon had the lowest cost per right answer, $0.0162 at its lowest setting.
- 🟢 The whole pilot cost $7.47 against a worst case of $33.23.
- ⚠️ The highest setting took 3.2 to 5.9 times as long per problem on the paid models.
- ❌ Gemma 4 26B on the free tier timed out twice and ran to the output limit seven times.
- ⚠️ Carbon's and Argon's costs double on 2026-12-01.

Scope: one laptop in the US calling the Gemini API on 2026-11-15, one run per model and setting, temperature 0, the same 32,768-token output limit for every model, the 15 problems of AIME 2025 I. Fifteen problems can show what extra thinking costs but cannot rank models that score within two problems of each other. Costs are token counts times list prices. Times include waiting on the service and will move with its load. Google does not say what hardware or engine serves any of these models.

The strategy for measuring high against low thinking on a fixed budget was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [This run's evidence: replies, logs, analysis and audit](https://github.com/xbill9/gemma4-dev/tree/main/docs/gemini-high-vs-low/evidence)
- [The scoring script and budget driver](https://github.com/xbill9/gemma4-dev/tree/main/benchmarks/gen-eval)
- [Gemini 4 or Gemma 4? It Depends.](https://dev.to/gde/gemini-4-or-gemma-4-it-depends-2o18)
- [AIME 2025 on Hugging Face](https://huggingface.co/datasets/opencompass/AIME2025)
- [Gemini API pricing](https://aistudio.google.com/docs/pricing)

Hope you enjoyed it!
