---
title: "You Cannot Turn Off Thinking on Gemini 4 Hydrogen: Seven Models Against Gemma 4, Same Questions"
published: false
description: "Gemini 4 Hydrogen and 3.8 Flash think on every request whatever you set, and the thinking is billed as output. Seven models through one Gemini API endpoint, started together with identical settings, scored on GSM8K and BFCL: scores, cost and speed."
tags: gemini, gemma, benchmark, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemini-vs-gemma/devto-cover-baseline.f29f1bde.jpg
---

This article provides a step by step guide to comparing the new Gemini 4 models with Gemma 4 on equal terms. Seven models, Gemma 4 26B and 31B, Gemini 3.5 Flash-Lite and 3.8 Flash, and Gemini 4 Hydrogen, Carbon and Argon, run through the same Gemini API endpoint with the same questions, settings and scoring, started at the same moment.

https://github.com/xbill9/gemma4-dev/tree/main/gemini-api-google-hydrogen4

The surprise: on Gemini 4 Hydrogen and Gemini 3.8 Flash, thinking cannot be turned off, and it eats the token budget. Every off setting the API offers still left Hydrogen thinking on all 1,319 GSM8K problems; thinking was 55% of its bill. Hydrogen has the lowest Gemini 4 price and cost the most.

The scores: on 1,319 GSM8K math problems Argon scored 0.9841, Hydrogen and Carbon 0.9803, 3.8 Flash 0.9788, Gemma 4 26B 0.9742, Gemma 4 31B 0.9735 and 3.5 Flash-Lite 0.9629. On 400 BFCL function calls Argon and Gemma 4 31B tied at 0.9675. Gemma 4 ran free; the Gemini runs cost $0.92 to $5.02. Hydrogen answered fastest of the Gemini 4 models, 2.84 s per GSM8K problem against 7.63 s for Argon.

---

#### What Are the New Models?

Gemini 4 Hydrogen, Carbon and Argon are on the Gemini API next to the 3.x Flash models and Gemma 4. Google's pricing page, saved on 2026-11-14 (paid tier, USD per million tokens; output includes thinking tokens):

| Model | Input | Output | Free tier |
| :--- | ---: | ---: | :--- |
| `gemma-4-31b-it`, `gemma-4-26b-a4b-it` | free tier only | free tier only | yes |
| `gemini-3.5-flash-lite` | $0.30 | $2.50 | yes |
| `gemini-3.8-flash` | $0.75 | $3.75 | yes |
| `gemini-4-hydrogen` | $1.50 | $7.50 | yes |
| `gemini-4-carbon` | $2.00 | $10.00 | no |
| `gemini-4-argon` | $2.00 | $10.00 | no |

Carbon and Argon go to $4 / $20 on 2026-12-01, and 3.8 Flash to $1.50 / $7.50 on 2027-01-01.

---

#### What Is the Same for Every Model

Every model got:

- the same endpoint, `/v1beta/openai/chat/completions` on the Gemini API
- the same 1,319 GSM8K and 400 BFCL simple questions, prompts and scoring code
- temperature 0, up to 2,048 tokens per GSM8K answer and 512 per BFCL answer
- eight requests at a time, a 180 s limit per request, up to eight retries
- the same start time: all seven began at 09:11:16 on 2026-11-14
- thinking set as low as each model accepts

Step 2 shows that setting for each model, and the two models that keep thinking at it.

---

#### At This Point You Should Have…

- A Gemini API key from Google AI Studio, saved in `~/gemini.key`
- Python 3 (the scoring script uses only the standard library)
- A clone of `gemma4-dev`

---

#### Step 1 — The Endpoint

The Gemini API answers OpenAI chat requests with the key as a Bearer token. Gemma 4 and Gemini take the same request body; only `model` changes:

```bash
curl -sS https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
  -H "Authorization: Bearer $(cat ~/gemini.key)" -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-31b-it","temperature":0,"max_tokens":2048,"reasoning_effort":"minimal",
       "messages":[{"role":"user","content":"Janet has 16 eggs, eats 3, bakes with 4, sells the rest at $2. How much does she make? End with Answer: <number>"}]}'
```

The reply ends `Answer: 18`.

---

#### Step 2 — Thinking Cannot Be Turned Off on Hydrogen or 3.8 Flash

The OpenAI endpoint sets thinking with `reasoning_effort`. `levels.sh` sends the question to every model with each value:

```
gemma-4-31b-it         minimal  thinking 0  finish stop   Answer: 18
gemma-4-31b-it         none     rejected
gemma-4-31b-it         low      rejected
gemma-4-26b-a4b-it     minimal  thinking 0  finish stop   Answer: 18
gemma-4-26b-a4b-it     none     rejected
gemini-3.5-flash-lite  minimal  thinking 0  finish stop   Answer: 18
gemini-3.5-flash-lite  none     rejected
gemini-3.8-flash       minimal  rejected
gemini-3.8-flash       none     thinking 70  finish stop   Answer: 18
gemini-4-hydrogen      minimal  rejected
gemini-4-hydrogen      none     thinking 81  finish stop   Answer: 18
gemini-4-carbon        minimal  rejected
gemini-4-carbon        none     thinking 0  finish stop   Answer: 18
gemini-4-argon         minimal  rejected
gemini-4-argon         none     thinking 0  finish stop   Answer: 18
```

Gemma 4 and Flash-Lite go down to `minimal`, and there they think zero tokens. Hydrogen, 3.8 Flash, Carbon and Argon go down to `none`; Carbon and Argon think zero tokens there, and Hydrogen and 3.8 Flash keep thinking.

The native API has more controls, and none of them stops it. Google's thinking page lists Hydrogen's levels as `low`, `medium`, `high` and `xhigh`, and says `minimal` is "Not supported on Gemini 4 models or `gemini-3.8-flash`". `thinking-controls.sh` tries each control on Hydrogen and reads the API's own counter, `thoughtsTokenCount`:

```
gemini-4-hydrogen  {"thinkingBudget":0}                          answer 88 thoughts 134 total 292 thought-parts 0
gemini-4-hydrogen  {"thinkingLevel":"NONE"}                      rejected: Invalid value at 'generation_config.thinking_config.thinking_level' (type.googleapis.com/g
gemini-4-hydrogen  {"thinkingLevel":"MINIMAL"}                   rejected: Thinking level MINIMAL is not supported for this model. Please retry with other thinking l
gemini-4-hydrogen  {"thinkingLevel":"LOW"}                       answer 70 thoughts 106 total 246 thought-parts 0
gemini-4-hydrogen  {"thinkingBudget":0,"includeThoughts":true}   answer 73 thoughts 79 total 222 thought-parts 1
```

With `thinkingBudget: 0` and `includeThoughts: true`, the reply carries the thought itself, beginning "Okay, here's my breakdown of this little egg scenario". 3.8 Flash reports 74 thinking tokens under every control.

Across the full run, Hydrogen thought on 1,319 of 1,319 GSM8K problems, 3.8 Flash on 1,293, Carbon on 268, Argon on 66, and Gemma 4 and Flash-Lite on none. Every one of those tokens is billed as output.

---

#### 🔎 Tip: Thinking Tokens Are Not in `completion_tokens`

On this endpoint `completion_tokens` counts only the visible answer. Thinking is `total_tokens - prompt_tokens - completion_tokens`, and Google bills it as output. The scoring script records it as `reasoning_tokens` on every row.

---

#### Step 3 — Seven Sibling Rigs

gemma4-dev names a rig `<platform>-<runtime>-<hardware>-<model>`. On the Gemini API the platform is `gemini`, the runtime `api` and the hardware `google`, since Google publishes neither; the model slot is the Gemma size or the Gemini family:

```
gemini-api-google-31b
gemini-api-google-26b
gemini-api-google-flashlite35
gemini-api-google-flash38
gemini-api-google-hydrogen4
gemini-api-google-carbon4
gemini-api-google-argon4
```

Each `tpu.env` names the model and the lowest thinking value it accepts, and the seven `run.sh` files are identical:

```
MODEL_NAME=gemini-4-hydrogen
LOWEST_REASONING_EFFORT=none
```

---

#### Step 4 — One Scoring Script

`jev-tpu-v5e1/gen_eval.py` scores every Gemma 4 rig in the repo. GSM8K is right when the number after `Answer:` matches; BFCL simple is right when the one function call passes BFCL's own checks. Four settings point it at a hosted API:

| Setting | What it does |
| :--- | :--- |
| `JEV_CHAT_URL` | full chat URL in place of `<server>/v1/chat/completions` |
| `JEV_API_KEY_FILE` | sends the file's contents as a Bearer token |
| `JEV_REASONING_EFFORT` | sends `reasoning_effort` |
| `JEV_RETRIES`, `JEV_TIMEOUT_S` | retries on HTTP 429/5xx and network errors, and a per-request time limit |

---

#### Step 5 — Run All Seven Together

```bash
for r in 31b 26b flashlite35 flash38 hydrogen4 carbon4 argon4; do
  gemini-api-google-$r/benchmarks/runs/2026-11-14-accuracy-api/run.sh &
done
```

```
2026-11-14 09:11:16 BEGIN gemini-4-argon reasoning_effort=none
2026-11-14 09:18:51 END bfcl_simple rc=0 {"task": "bfcl_simple", "max_tokens": 512, "n": 400, "right": 387, "accuracy": 0.9675, "errors": 0, "truncated": 0, "wall_s": 454.8}
2026-11-14 09:48:10 END gsm8k rc=0 {"task": "gsm8k", "max_tokens": 2048, "n": 1319, "right": 1298, "accuracy": 0.9841, "errors": 0, "truncated": 0, "wall_s": 1759.2}
2026-11-14 09:48:10 DONE
```

Three Gemma 4 requests hit the free tier's rate limit (HTTP 429) after every retry. `resend_errors.py` sent those three again with the same settings and replaced only those rows; all three came back right. Every answer in the table below is a model answer.

---

#### Step 6 — The Results

`derived.py` builds the table from the saved replies. The cost is the token counts the API returned times the list price.

| Model | GSM8K | BFCL simple | GSM8K problems with thinking, at the lowest setting | Cost, both tasks |
| :--- | ---: | ---: | ---: | ---: |
| Gemma 4 31B | 0.9735 | 🥇 0.9675 | 0 | 🥇 free tier |
| Gemma 4 26B | 0.9742 | 0.9575 | 0 | 🥇 free tier |
| Gemini 3.5 Flash-Lite | 0.9629 | 0.9350 | 0 | $0.92 |
| Gemini 3.8 Flash | 0.9788 | 0.9400 | 1,293 | $1.81 |
| Gemini 4 Hydrogen | 🥈 0.9803 | 0.9550 | 1,319 | $5.02 |
| Gemini 4 Carbon | 🥈 0.9803 | 0.9600 | 268 | $4.02 |
| Gemini 4 Argon | 🥇 0.9841 | 🥇 0.9675 | 66 | $3.46 |

Against Gemma 4 31B on the same questions, counting the questions each got right that the other missed:

| Model | GSM8K | BFCL simple |
| :--- | :--- | :--- |
| Gemma 4 26B | level (+10 / −9) | level (+5 / −9) |
| Gemini 3.5 Flash-Lite | behind (+6 / −20) | behind (+5 / −18) |
| Gemini 3.8 Flash | level (+10 / −3) | behind (+2 / −13) |
| Gemini 4 Hydrogen | ahead (+11 / −2) | level (+5 / −10) |
| Gemini 4 Carbon | ahead (+14 / −5) | level (+5 / −8) |
| Gemini 4 Argon | ahead (+17 / −3) | level (+7 / −7) |

Ahead and behind mean the 95% range of the difference, from 2,000 resamples of the questions, sits entirely above or below zero.

---

#### Step 7 — Where the Money Goes

By list price Hydrogen is the cheapest Gemini 4 model, $7.50 per million output tokens against $10 for Carbon and Argon. It cost the most because it thought on all 1,319 GSM8K problems at its lowest setting, 248.7 thinking tokens each on average, which made thinking 55% of its GSM8K bill, against 11% for Carbon and 2% for Argon. Argon thought on 66 problems and scored the highest.

The Gemini 4 models also count more input tokens for the same text: the GSM8K prompts are 160,432 tokens to Hydrogen, Carbon and Argon and 120,902 to Gemma 4 31B, Flash-Lite and 3.8 Flash.

---

#### Step 8 — Speed

Each reply records how long its request took. All seven models ran at once, eight requests at a time each, over the same connection, so the times are comparable with each other on that morning:

| Model | GSM8K median | GSM8K 90th percentile | Whole GSM8K run | BFCL median |
| :--- | ---: | ---: | ---: | ---: |
| Gemma 4 31B | 8.96 s | 23.93 s | 2,044.1 s | 1.69 s |
| Gemma 4 26B | 8.16 s | 25.80 s | 2,022.8 s | 1.23 s |
| Gemini 3.5 Flash-Lite | 🥇 1.10 s | 🥇 1.54 s | 🥇 195.8 s | 🥇 0.61 s |
| Gemini 3.8 Flash | 3.79 s | 5.43 s | 684.9 s | 1.93 s |
| Gemini 4 Hydrogen | 🥈 2.84 s | 🥈 5.39 s | 🥈 620.8 s | 1.95 s |
| Gemini 4 Carbon | 7.34 s | 20.04 s | 1,803.7 s | 5.29 s |
| Gemini 4 Argon | 7.63 s | 17.57 s | 1,759.2 s | 6.32 s |

Hydrogen writes about 147.8 tokens per second per request, thinking included, against 26.1 for Argon and 29.7 for Carbon. It finished GSM8K in about a third of Argon's time while thinking on every problem. Flash-Lite was the fastest model overall.

---

#### Compare and Contrast

| | Gemma 4 31B | 3.5 Flash-Lite | 3.8 Flash | Hydrogen | Carbon | Argon |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GSM8K against Gemma 4 31B | — | behind | level | ahead | ahead | 🥇 ahead |
| BFCL against Gemma 4 31B | 🥇 | behind | behind | level | level | 🥇 level |
| Thinking can be turned off | 🥇 yes | 🥇 yes | no | no | no | no |
| Thinks at its lowest setting | no | no | most problems | every problem | some problems | rare |
| Cost, both tasks | 🥇 free tier | $0.92 | $1.81 | $5.02 | $4.02 | $3.46 |
| GSM8K median per request | 8.96 s | 🥇 1.10 s | 3.79 s | 🥈 2.84 s | 7.34 s | 7.63 s |
| Weights you can download | 🥇 yes | no | no | no | no | no |

---

#### So, Which One?

For the best scores, Gemini 4 Argon: highest on GSM8K, tied for highest on BFCL, and the cheapest of the three Gemini 4 models per run because it rarely thinks. Its price doubles on 2026-12-01.

For speed among the Gemini 4 models, Hydrogen: 2.84 s per GSM8K problem, level with Carbon and Argon on BFCL and ahead of Gemma 4 31B on GSM8K. Budget for its thinking: it cannot be turned off, and it made Hydrogen the most expensive model per run.

For tool calling at no cost, Gemma 4 31B: tied with Argon on BFCL, 0.0106 behind it on GSM8K, free on the Gemini API, and the same weights run on your own hardware.

Gemini 3.5 Flash-Lite is the cheapest paid model here, and it was behind Gemma 4 31B on both tests.

---

#### Summary

The goal of this article was to compare the new Gemini 4 models with Gemma 4 on equal terms. The key to the solution was running all seven models through one Gemini API endpoint, started together, with the same questions, scoring script and settings, with thinking set as low as each model accepts. The results were:

- ⚠️ Thinking cannot be turned off on Gemini 4 Hydrogen or 3.8 Flash: no level, budget or flag stops it, and it is billed as output.
- 🟢 Gemini 4 Argon scored highest: 0.9841 on GSM8K and 0.9675 on BFCL simple, for $3.46.
- 🟢 Gemma 4 31B tied Argon on BFCL at 0.9675 and scored 0.9735 on GSM8K, on the free tier.
- 🟢 Hydrogen, Carbon and Argon were ahead of Gemma 4 31B on GSM8K; on BFCL all three were level with it.
- 🟢 Gemma 4 26B was level with the 31B on both tests.
- 🟢 Hydrogen was the fastest Gemini 4 model, 2.84 s per GSM8K problem against 7.34 s for Carbon and 7.63 s for Argon.
- ⚠️ Hydrogen has the lowest Gemini 4 list price and cost the most, $5.02: thinking was 55% of its GSM8K bill.
- ⚠️ Thinking tokens are missing from `completion_tokens` on the OpenAI-compatible endpoint.
- ⚠️ Carbon and Argon have no free tier and double in price on 2026-12-01.
- ❌ Gemini 3.5 Flash-Lite and 3.8 Flash were behind Gemma 4 31B on BFCL.

Scope: one laptop in the US calling the Gemini API on 2026-11-14, one run per model, all seven started together. Thinking was set as low as each model accepts: `minimal` for Gemma 4 and Flash-Lite, `none` for the rest. The thinking controls were tried on one question. Three Gemma 4 requests that hit the free tier's rate limit were sent again with the same settings. Costs are token counts times list prices. Run times are from one run with all seven sharing one connection, and will move with load on the service. Google does not say what hardware or engine serves any of these models.

The strategy for comparing Gemini and Gemma 4 on equal terms was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [The Gemini API sibling rigs](https://github.com/xbill9/gemma4-dev/tree/main/gemini-api-google-argon4)
- [Evidence for this article](https://github.com/xbill9/gemma4-dev/tree/main/docs/gemini-vs-gemma/evidence)
- [Gemini API pricing](https://aistudio.google.com/docs/pricing)
- [GSM8K](https://github.com/openai/grade-school-math)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Gemma 4 31B on Hugging Face](https://huggingface.co/google/gemma-4-31B-it)

Hope you enjoyed it!
