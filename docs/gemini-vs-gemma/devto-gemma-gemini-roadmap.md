---
title: "Local, Hosted, Cloud Run or Your Own Accelerator? A Roadmap for Gemma 4 and the New Gemini 4 Models"
published: false
description: "Measured numbers for every way gemma4-dev runs a model: a desktop GPU, the Gemini API and AMD Token Lab, Cloud Run on an L4, and TPU, EC2 and MI300X machines. Then a roadmap for choosing between them by users, cost, speed, accuracy and ease of use."
tags: gemma, gemini, googlecloud, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemini-vs-gemma/devto-cover-roadmap.0d4945c0.jpg
---

This article provides a roadmap for choosing where to run a model: on your own machine, through a hosted API, on Cloud Run, or on an accelerator you rent. Every figure comes from a measured run in gemma4-dev, which serves Gemma 4 on more than a hundred hardware and engine combinations and now carries five Gemini API models as sibling rigs.

https://github.com/xbill9/gemma4-dev

---

#### The New Gemini Models in One Table

Gemini 4 Hydrogen, Carbon and Argon arrived on the Gemini API next to 3.5 Flash-Lite, 3.8 Flash and Gemma 4. On 2026-11-14 all seven ran through the same Gemini API endpoint, started together, with the same GSM8K (1,319 math word problems) and BFCL simple (400 function calls) questions and scoring, each at the lowest thinking level it accepts:

| Model | GSM8K | BFCL simple | Cost, both tasks | Output price, per 1M tokens |
| :--- | ---: | ---: | ---: | ---: |
| Gemma 4 31B | 0.9735 | 🥇 0.9675 | 🥇 free tier | free tier only |
| Gemma 4 26B | 0.9742 | 0.9575 | 🥇 free tier | free tier only |
| Gemini 3.5 Flash-Lite | 0.9629 | 0.9350 | $0.92 | $2.50 |
| Gemini 3.8 Flash | 0.9788 | 0.9400 | $1.81 | $3.75 |
| Gemini 4 Hydrogen | 0.9803 | 0.9550 | $5.02 | $7.50 |
| Gemini 4 Carbon | 0.9803 | 0.9600 | $4.02 | $10.00 |
| Gemini 4 Argon | 🥇 0.9841 | 🥇 0.9675 | $3.46 | $10.00 |

Hydrogen, Carbon and Argon are ahead of Gemma 4 31B on GSM8K by more than the run-to-run noise and level with it on BFCL. Output prices include thinking tokens, and thinking decides the bill: Hydrogen has the lowest Gemini 4 price but thinks on every problem, so it cost the most. Carbon and Argon double in price on 2026-12-01. The method is in the companion article, *Gemini 4 Hydrogen, Carbon and Argon Against Gemma 4*.

---

#### The Options

Four ways to run a model, each measured in this repo:

- **Local:** a GPU or CPU you own. Measured on an RTX 3050 OEM (8 GB desktop card), a GTX 1650 Ti (4 GB laptop card) and an Intel i7-10750H laptop CPU.
- **Hosted API:** someone else's server, reached with a key. Measured on the Gemini API (the five Gemini models) and AMD Token Lab (Gemma 4 31B, free with the AMD AI Developer Program).
- **Cloud Run:** Google's serverless containers with one NVIDIA L4, serving Gemma 4 E2B on vLLM.
- **Your own accelerator:** a rented machine you start and stop. Measured on a TPU v5e (one chip and four), an EC2 G6 with one L4, and an AMD MI300X on DigitalOcean.

---

#### Option 1 — Local

The RTX 3050 serves Gemma 4 E2B on six engines. One request at a time without a drafter they run from 61.9 tok/s (LiteRT-LM) to 121.1 tok/s (vLLM); with Google's speculative-decoding drafter vLLM reaches 230.5 tok/s. Under load, with 64 requests at once of 128 tokens in and 512 out, SGLang delivers 1,711.3 tok/s in total and vLLM 1,651.2.

The same card holds the larger models one request at a time: 12B at 56.4 tok/s with its drafter, 26B A4B at 40.6 tok/s on code with its experts in system RAM, and 31B at 4.0 tok/s with two thirds of it on the CPU. Quality rises with size: GSM8K 0.9014 (E2B), 0.9629 (12B), 0.9735 (31B).

The 1650 Ti laptop card decodes E2B at 73.75 tok/s for one request and peaks at 164 tok/s through its server at 16 requests. The laptop CPU decodes 16.71 to 18.86 tok/s.

Setup: llama.cpp is ready 4 s after start, vLLM 48 s. vLLM's first start compiles for about 3.5 minutes. The hardware is already paid for, and nothing is billed per hour.

---

#### Option 2 — Hosted API

**Gemini API.** A key and an HTTP call. The Gemini models and Gemma 4 26B and 31B answer the OpenAI chat format at `/v1beta/openai/chat/completions`. Estimated from the token counts returned and list prices, a full 1,319-problem GSM8K run cost $0.87 on Flash-Lite, $1.66 on 3.8 Flash, $3.15 on Argon, $3.70 on Carbon and $4.51 on Hydrogen. Gemma 4 is on the free tier only, with paid use not available, and its free-tier rate limit answered three requests in the GSM8K and BFCL runs with HTTP 429 after every retry.

**AMD Token Lab.** Gemma 4 31B on an SGLang router on AMD GPUs, free with the AMD AI Developer Program: 10 points for 30 days, with Gemma 4 31B at 0.035 points per million input tokens and 0.07 per million output. One request at a time decoded at 2.0 to 7.6 tok/s on the shared service, with up to 5 s before the first token. The limit is 128 requests at once across all users; over that, the reply is HTTP 429.

---

#### Option 3 — Cloud Run

One L4 GPU in us-east4 with 8 vCPU and 32 GiB, Gemma 4 E2B on vLLM. Output rose with concurrency to 188.46 tok/s at 4 requests and stayed there at 8 (189.53), because the service was deployed with `--concurrency=4`; more users means more instances. The same L4 on EC2 reached 360.17 tok/s at 8.

At list price the instance costs $1.4209 an hour, which works out to $7.95 per million output tokens at one request and $2.08 at eight. Cloud Run scales to zero by default, so an idle service costs nothing and the next request waits for a cold GPU start; the startup check waits 180 s before its first look. The weights load from Cloud Storage.

---

#### Option 4 — Your Own Accelerator

**TPU v5e, one chip.** Gemma 4 E2B on vLLM: 143.9 tok/s for one request, 1,689.1 tok/s at 64 requests (128 in, 512 out), GSM8K 0.9007 and BFCL 0.920. Flex-start costs $0.60 per chip-hour, about $0.089 per million output tokens at full batching and $1.31 at one request. Capacity came within two to five minutes in `us-west4-a`, and the engine took 12.6 to 14.1 minutes to start on most rigs, most of it compiling (27.2 on one).

**TPU v5e, four chips.** E2B as four one-chip engines reaches 7,788 tok/s at 64 requests and 14,189 at 256. The 12B fits with 122,624 tokens of context cache, against 9,728 on one chip.

**EC2 G6, one L4.** vLLM at 132.8 to 140.4 tok/s for one request and 3,715.7 tok/s at 64; with the drafter, 4,516.3. A spot g6.2xlarge measured on an older vLLM cost $0.9412 an hour, $0.73 per million output tokens at eight requests.

**AMD MI300X.** 192 GB on one card: E2B at 340.4 tok/s for one request and 10,293.6 tok/s at 64 (128 tokens of context). $1.99 an hour, billed while powered off too: $0.054 per million output tokens at 64 clients and $1.624 for a single stream.

---

#### 🔎 Tip: Price Per Hour Becomes Price Per Token Only Under Load

Every rented option above has two prices per million tokens, one at a single request and one at full batching: Cloud Run $7.95 and $2.08, TPU v5e $1.31 and $0.089, MI300X $1.624 and $0.054. A machine that serves one user at a time costs about what a hosted API charges, or more. A per-token API charges the same at any load.

---

#### Compare and Contrast

| | Local 3050 | Gemini API | Token Lab | Cloud Run L4 | TPU v5e-1 | MI300X |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Users at once | 64 measured | 8 measured | 128 shared | 4 per instance | 64 measured | 64 measured |
| Most output, tok/s | 1,711.3 | not measured | not measured | 189.53 | 1,689.1 | 🥇 10,293.6 |
| One request, tok/s | 230.5 | not measured | 2.0–7.6 | 49.63 | 143.9 | 340.4 |
| Cost | 🥇 card you own | per token | 🥇 free | $1.4209/hr | $0.60/hr | $1.99/hr |
| Best GSM8K | 0.9735 (31B) | 🥇 0.9841 (Argon) | not measured | not measured | 0.9007 (E2B) | not measured |
| Ready in | 🥇 4 s | 🥇 a key | 🥇 a key | minutes, cold start | 12–14 min | minutes |

---

#### The Roadmap

Pick by the axis that matters most, then check the others.

**By number of users.**

| Users at once | Start with |
| :--- | :--- |
| 1, yourself | Local card, or the Gemini or Token Lab free tier |
| 2–10 | Gemini API, or Cloud Run with more instances |
| 10–64 | A batching engine on your own card (vLLM, SGLang), a TPU v5e-1, or an L4 |
| Hundreds | TPU v5e-4 as four engines, or an MI300X |

**By cost.** Free first: your own card, Token Lab, and the Gemini free tier, which carries Gemma 4 26B and 31B, Flash-Lite, 3.8 Flash and Hydrogen. For paid use with light or uneven traffic, pay per token: both tests cost $0.92 on Flash-Lite and $3.46 on Argon. A rented machine wins only when it is kept busy, which brings TPU v5e to $0.089 and MI300X to $0.054 per million output tokens.

**By speed.** For one user, a local card with a drafter (230.5 tok/s on the 3050) or an MI300X (340.4). For many users, the MI300X or four TPU v5e chips. Cloud Run as deployed here stops at 189.53 tok/s per instance.

**By accuracy.** Gemini 4 Argon leads GSM8K at 0.9841, and Argon and Gemma 4 31B lead BFCL at 0.9675. Where a model runs did not change its score: Gemma 4 31B answered 1,284 of 1,319 GSM8K problems through the Gemini API and the same 1,284 on an RTX 3050 with llama.cpp. Model size matters more than where it runs: E2B scored 0.8954 to 0.9030 on GSM8K across six engines on the RTX 3050, the L4 and TPU v5e, and the 31B 0.9735.

**By ease of use.** A hosted API needs a key. A local card needs a driver and an engine, and llama.cpp is ready in seconds. Cloud Run needs a container and a deploy and then scales on its own. A rented accelerator needs capacity, a 12.6 to 14.1 minute engine start on TPU, and someone to turn it off.

---

#### So, Which One?

Start hosted with Gemma 4 31B on the Gemini free tier: it tied the best BFCL score and costs nothing. If the task needs the best math scores, move to Gemini 4 Argon. If the task needs Gemma 4 for its open weights, privacy or offline use, run it on your own card, 12B or 31B on 8 GB. Move to Cloud Run when a few users need an always-reachable endpoint without managing a machine. Move to a TPU v5e or an MI300X when traffic is steady enough to keep it busy, because that is where the per-token cost falls below every API.

---

#### Summary

The goal of this article was to map where each way of running a model fits. The key to the solution was a single repo that measures local cards, hosted APIs, Cloud Run and rented accelerators with the same engines, scoring script and questions. The results were:

- 🟢 Gemini 4 Argon has the best GSM8K score measured, 0.9841; Argon and Gemma 4 31B lead BFCL at 0.9675.
- 🟢 Gemma 4 31B scored the same 1,284 of 1,319 on GSM8K through the Gemini API and on an 8 GB RTX 3050.
- 🟢 Gemma 4 31B is free on the Gemini API and tied Argon on BFCL; Argon cost $3.46 for both tests.
- 🟢 A busy MI300X or TPU v5e serves Gemma 4 E2B at $0.054 and $0.089 per million output tokens.
- 🟢 AMD Token Lab and the Gemini free tier cost nothing to start.
- ⚠️ A rented machine at one request costs $1.31 (TPU v5e) to $7.95 (Cloud Run) per million output tokens.
- ⚠️ Cloud Run as deployed caps one instance at 4 requests, 189.53 tok/s.
- ❌ Token Lab's shared endpoint decoded one request at 2.0 to 7.6 tok/s.

Scope: figures come from separate runs between 2026-09-03 and 2026-11-14 in gemma4-dev, on different engine versions and days; each is quoted from its run's report, and ratios between runs on different hardware are indicative. Prices are list prices on the date each run recorded them. Accuracy was measured on GSM8K and BFCL simple only, and not on Cloud Run, Token Lab or the MI300X; the throughput figures are Gemma 4 E2B unless a size is named. Gemini API throughput under load was not measured.

The strategy for choosing where to run Gemma 4 and Gemini was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [Evidence for this article](https://github.com/xbill9/gemma4-dev/tree/main/docs/gemini-vs-gemma/evidence)
- [Gemini API pricing](https://aistudio.google.com/docs/pricing)
- [AMD Token Lab](https://notebooks.amd.com/tokenlab)
- [Cloud Run GPUs](https://cloud.google.com/run/docs/configuring/services/gpu)
- [Cloud TPU](https://cloud.google.com/tpu)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Gemma 4 on Hugging Face](https://huggingface.co/collections/google/gemma-4)

Hope you enjoyed it!
