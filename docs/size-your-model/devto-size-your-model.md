---
title: "Size Your Model to Where You Run It: Gemma 4 E2B and E4B on an AWS L4 and Two Small GPUs"
published: false
description: "A 24 GB NVIDIA L4 serves Gemma 4 E2B and E4B at only 1.2-1.3x an 8 GB desktop RTX 3050 for one user, with the same answers, because a 4-bit small model uses 13-29% of the card. Pick the model for the hardware, and the hardware for the model."
tags: gemma, llamacpp, aws, cuda
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/size-your-model/devto-cover.3f5286be.jpg
---

This article provides a step by step guide to serving Gemma 4 E2B and E4B, Google's quantization-aware trained (QAT) small models, on an NVIDIA L4 in an AWS EC2 G6 instance with llama.cpp, and compares the result with the same models on two small local GPUs: an 8 GB RTX 3050 desktop card and a 4 GB GTX 1650 Ti laptop card. Every card runs the same llama.cpp build, the same model files, the same drafter and the same tests. The measures are generation speed for one user and for many, what the drafter adds on each card and why, answer quality, and cost.

https://github.com/xbill9/gemma4-dev

**For one user, the 24 GB L4 is 1.2-1.3x an 8 GB desktop card, with the same answers.** The 4-bit small models use 13-29% of the L4's memory, and the rest of the card sits idle:

| Gemma 4, 4-bit QAT, one user | GTX 1650 Ti, 4 GB laptop | RTX 3050 OEM, 8 GB desktop | NVIDIA L4, 24 GB, AWS |
| :--- | ---: | ---: | ---: |
| E2B, no drafter | 76.5 tok/s | 108.2 tok/s | 133.4 tok/s |
| E2B, best drafter setting | 90.3 tok/s | 151.2 tok/s | **217.6 tok/s** |
| E4B, no drafter | 38.0 tok/s | 58.3 tok/s | 74.0 tok/s |
| E4B, best drafter setting | 56.5 tok/s | 100.0 tok/s | **129.7 tok/s** |
| 3050 as a share of the L4 | | 69-81% | |

The desktop card delivers 79-81% of the L4 without the drafter and 69-77% with it. The L4 earns its memory with many users at once (1.6x the 3050 at 64) and with models too large for 8 GB. The rule this article arrives at: size the model to where it runs.

---

#### What Is Being Measured?

Gemma 4 E2B and E4B are the two small models of the family, served here as exact 4-bit (Q4_0) GGUF files whose every block sits on Google's QAT grid: `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf` and `xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`. Google also ships a QAT drafter for each, a small model that proposes the next few tokens for the main model to check in one pass; it is stored the same way, as `xbill9/gemma-4-E{2,4}B-it-qat-assistant-q4_0-exact-gguf`.

When one user is generating, every token reads every weight of the model once. The speed limit is then how fast the card reads its memory, and the size of the memory plays no part as long as the model fits. That is why a card with three times the memory can be only a little faster.

The three cards:

| | GTX 1650 Ti Max-Q | RTX 3050 OEM | NVIDIA L4 |
| :--- | :--- | :--- | :--- |
| Memory | 4 GB | 8 GB | 24 GB |
| Host | laptop | i7-5820K desktop | EC2 `g6.2xlarge`, 8 vCPU |
| Where | local | local | AWS `us-east-1`, spot |

All three run llama.cpp `77c4540ce` with flash attention, a 16-bit context cache and the per-layer embedding table on the card. The L4 build uses CUDA 13.1 for sm_89.

---

#### Step 1 — Serve the Rigs

Each model has a rig directory with a `tpu.env` holding every setting and a `Makefile` to run on the instance:

```
git clone https://github.com/xbill9/gemma4-dev
cd gemma4-dev/gpu-llamacpp-g6-2b-q4_0      # or gpu-llamacpp-g6-4b-q4_0
make toolkit build download draft serve
```

The G6 instance comes from `gpu-pytorch-g6-2b-q4w4a16`'s MCP tool `create_g6_instance`, on the PyTorch Deep Learning AMI. The AMI has an NVIDIA driver (595.112.07, CUDA 13.2) and no CUDA compiler, so `make toolkit` installs CUDA 13.1 from NVIDIA's apt repository, one minor version below what the driver supports. `make build` compiles llama.cpp for sm_89 in 633 s.

The serving line, E2B:

```
llama-server -m gemma-4-E2B-it-q4_0-exact.gguf -ngl 99 -c 131072 -ctk f16 -ctv f16 -fa 1 \
  -t 4 -tb 8 --parallel 1 --reasoning off --metrics -ot 'per_layer_token_embd\.weight=CUDA0' \
  -md gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

At the full 131,072-token context with the drafter, E2B peaks at 3,952 MiB and E4B at 6,774 MiB of the L4's 23,034.

---

#### Step 2 — Measure One User

`speed.py` sends six prompts (code, chat, story, explanation, facts, summary), 256 tokens each, three greedy repeats per prompt with the prompt cache off, and reports the geometric mean over prompts. Each arm gets its own server, started on a cooled card. On the L4:

| drafted tokens | E2B tok/s | x none | E4B tok/s | x none | tokens kept |
| ---: | ---: | ---: | ---: | ---: | ---: |
| none | 133.4 | 1.000x | 74.0 | 1.000x | — |
| 1 | 185.5 | 1.391x | 106.8 | 1.443x | 0.60-0.62 |
| 2 | 211.0 | 1.582x | 121.8 | 1.646x | 0.48-0.50 |
| 3 | **217.6** | **1.631x** | **129.7** | **1.754x** | 0.40 |
| 4 | 196.6 | 1.474x | 129.7 | 1.754x | 0.30-0.35 |

A no-drafter arm run last lands 0.7% (E2B) and 1.2% (E4B) under the first. Code gains most from the drafter (2.39x E2B, 2.55x E4B at 3 tokens) and stories least (1.31x, 1.38x).

**With the drafter, E4B runs at 0.972x E2B without it**: 129.7 against 133.4 tok/s.

The same prompts on the two local cards, best setting per card:

| | 1650 Ti | 3050 | L4 |
| :--- | ---: | ---: | ---: |
| E2B best | 1 token, 1.18x | 2 tokens, 1.40x | 3 tokens, 1.63x |
| E4B best | 1 token, 1.49x | 2 tokens, 1.72x | 3 tokens, 1.75x |

---

#### 🔎 Tip: The Card Sets the Draft Length

The drafter's guesses are kept at the same rate on every card. What changes is what each guess costs. Splitting each reply into steps (one pass of the main model over the last token plus the drafted ones) shows it, here for E2B:

| drafted tokens | tokens per step, all three cards | step time vs a plain step: 1650 Ti | 3050 | L4 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.60 | 1.35x | 1.16x | 1.15x |
| 2 | 1.92-1.98 | 1.77x | 1.39x | 1.26x |
| 3 | 2.12-2.20 | 2.12x | 1.63x | 1.36x |

The speed-up is tokens per step divided by step time. On the L4 `llama-bench` times the main model over 1 to 6 tokens in one pass: 4 tokens take at most 1.04x the time of 1, at 0, 8,192 and 32,768 tokens of context. So checking 3 guesses is close to free there, and the added time is the drafter's own, 0.75-1.18 ms per guess. That is 33% of an E2B step at 3 guesses and 19% of E4B's longer step, which is why E4B gains more from its drafter. A slower card pays more per guess and peaks at fewer.

---

#### Step 3 — Hold a Long Conversation

`longctx.py` puts the first D tokens of a long document ahead of a story, a code and a summary request, at `-c 131072`. Mean speed-up over the requests at 2 drafted tokens:

| tokens ahead | E2B, L4 | E2B, 3050 | E4B, L4 | E4B, 3050 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.70x | 1.51x | 1.75x | 1.86x |
| 32,768 | 1.53x | | 1.69x | 1.64x |
| 120,000 | 1.49x | 1.21x | 1.57x | 1.55x |

Generation without the drafter slows with context (E2B story 134.1 tok/s empty, 66.7 at 120,000 ahead on the L4), and the drafter keeps paying at every depth on both cards.

---

#### Step 4 — Serve Many Users at Once

`mixed.py` sends c requests at once, each one of the six chat prompts, 256 tokens, five repeats, one shared context cache (`-kvu`) at `-c 32768 --parallel 64`. Total output tok/s:

| requests at once | E2B, 3050 | E2B, L4 | E4B, 3050 | E4B, L4 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 106.1 | 131.4 | 57.3 | 73.5 |
| 8 | 297.1 | 505.2 | 199.3 | 309.7 |
| 64 | 569.5 | **918.9** | 472.4 | **778.1** |

**This is where the L4's size shows: 1.61-1.65x the 3050 at 64 requests.** The drafter (2 tokens) helps on the L4 up to 16 requests (E2B 1.06-1.19x, E4B 1.15-1.34x) and costs output at 64 (0.87x for both).

---

#### Step 5 — Check the Answers

`gen_eval.py` runs GSM8K (1,319 grade-school math questions, up to 2,048 tokens per reply) and BFCL simple (400 single function calls) at `-c 32768`, 8 requests at once, with and without the drafter at 2 tokens:

| L4 | GSM8K, none / drafter | BFCL simple, none / drafter | GSM8K time, none / drafter |
| :--- | ---: | ---: | ---: |
| E2B | 0.8992 / 0.8984 | 0.930 / 0.925 | 1,014 s / 713 s |
| E4B | 0.9310 / 0.9287 | 0.910 / 0.915 | 1,416 s / 893 s |

Paired question by question, every drafter-against-none difference has a 95% range that includes zero. The 3050 scored E2B 0.8984 / 0.9022 and E4B 0.9340 / 0.9325 on GSM8K under the same settings: the same model files give the same answers on every card.

**E4B answers more GSM8K questions right than E2B**, +0.0318 without the drafter (range +0.0167 to +0.0470) and +0.0303 with it, and BFCL is level between the two. With its drafter E4B finishes GSM8K in 893 s, sooner than E2B without one (1,014 s).

---

#### Step 6 — Price It

The L4 instance, `g6.2xlarge` in `us-east-1`, listed at $0.9776/h on demand and $0.8393/h spot on the day of the runs. Per million output tokens, one machine kept busy:

| L4, spot | tok/s | $ per million tokens |
| :--- | ---: | ---: |
| E2B, one user, no drafter | 133.4 | $1.75 |
| E2B, one user, drafter | 217.6 | $1.07 |
| E4B, one user, drafter | 129.7 | $1.80 |
| E2B, 64 users | 918.9 | $0.25 |
| E4B, 64 users | 778.1 | $0.30 |

One user pays for the whole card while using 13-29% of its memory. Filled with 64 requests, the same card costs a sixth to a seventh as much per token. A local card's cost is the card and its electricity, neither measured here.

---

#### 🔎 Tip: Where the Bigger Card Earns Its Keep

The 8 GB 3050 runs every larger Gemma 4 too, at a price paid in speed once the model outgrows the card. From the same series on the same card:

| on the RTX 3050, 8 GB | where the model lives | generation, with drafter |
| :--- | :--- | ---: |
| 12B, 6.7 GB | all on the card | 55.1 tok/s |
| 26B A4B, 14.4 GB | experts of 21 of 30 layers in system RAM | 33.7 tok/s |
| 31B, 17.7 GB | 40 of 60 layers in system RAM | 4.14 tok/s |

All three fit a 24 GB card whole; this article measures them on the 3050 only.

---

#### Compare and Contrast

| Gemma 4 E2B / E4B, 4-bit QAT | 🥇 RTX 3050, 8 GB, local | 🥈 NVIDIA L4, 24 GB, AWS | 🥉 GTX 1650 Ti, 4 GB, local |
| :--- | ---: | ---: | ---: |
| One user, E2B / E4B, best | 151.2 / 100.0 tok/s | 217.6 / 129.7 tok/s | 90.3 / 56.5 tok/s |
| 64 users, E2B / E4B | 569.5 / 472.4 tok/s | 918.9 / 778.1 tok/s | |
| Best draft length | 2 tokens | 3 tokens | 1 token |
| GSM8K, E4B with drafter | 0.9325 | 0.9287 | |
| Cost | owned | $0.84/h spot | owned |

---

#### So, Which One?

For one user on E2B or E4B, a local 8 GB card. It delivers 69-81% of the L4's speed with the same answers, and the model uses a fraction of either card.

For many users at once, the L4. At 64 requests it serves 1.6x the 3050's output, and at about $0.25-0.30 per million tokens on spot.

For quality at small-model speed, E4B with its drafter. On the L4 it generates at 0.97x E2B without a drafter and scores 3 points higher on GSM8K.

For the 26B and 31B, a card that holds them. On 8 GB they spill into system RAM and slow to 34 and 4 tok/s.

---

#### Summary

The goal of this article was to find what a 24 GB cloud GPU buys over small local GPUs when serving Gemma 4's small models. The key to the answer was running the same llama.cpp build, model files, drafter and tests on all three cards. The results were:

- 🟢 One user: the L4 is 1.23-1.27x the 3050 without the drafter and 1.30-1.44x with it, at 13-29% of its memory
- 🟢 The same model files give the same answers on every card; the drafter costs no GSM8K or BFCL answers
- 🟢 The drafter's guesses are kept at the same rate on every card; the card's cost per guess sets the best draft length, 1 / 2 / 3 tokens on the 1650 Ti / 3050 / L4
- 🟢 E4B with its drafter generates at 0.97x E2B without one, 3 GSM8K points higher
- 🟢 64 users: the L4 serves 1.6x the 3050, $0.25-0.30 per million tokens on spot
- ⚠️ The drafter helps on the L4 up to 16 users at once and costs output at 64
- ❌ One user on the L4 pays for a whole card and uses under a third of its memory

Scope: Gemma 4 E2B-it and E4B-it QAT only, exact Q4_0 GGUF files and exact Q4_0 QAT drafters, text only, llama.cpp `77c4540ce` on all three cards, 16-bit context cache, thinking off. One L4 instance on one day (spot `g6.2xlarge`, `us-east-1d`); the 3050 (i7-5820K desktop) and 1650 Ti (laptop) runs are from 2026-10-14 and 2026-10-15 on different hosts with the same prompts and scripts, so cross-card ratios are indicative. The many-users test uses short chat prompts and a shared context cache. Larger cloud GPUs and the 12B, 26B and 31B on the L4 were not run.

The strategy for using MCP for local accelerator comparison was validated with an incremental step by step approach.

---

#### References

- Part 2, Do Drafters Work on Small Models?: https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g
- Part 6, A Dense 31B on an 8 GB GPU: https://dev.to/gde/a-dense-31b-on-an-8-gb-gpu-gemma-4-31b-on-an-rtx-3050-26p9
- Repository: https://github.com/xbill9/gemma4-dev
- E2B on the L4: https://github.com/xbill9/gemma4-dev/blob/main/gpu-llamacpp-g6-2b-q4_0/benchmarks/runs/2026-10-26-drafter-g6/REPORT.md
- E4B on the L4: https://github.com/xbill9/gemma4-dev/blob/main/gpu-llamacpp-g6-4b-q4_0/benchmarks/runs/2026-10-26-drafter-g6/REPORT.md
- Drafter cost and many users on the L4: https://github.com/xbill9/gemma4-dev/blob/main/gpu-llamacpp-g6-2b-q4_0/benchmarks/runs/2026-10-26-drafter-cost-g6/REPORT.md
- E2B GGUF: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf
- E4B GGUF: https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf
- E2B drafter: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf
- E4B drafter: https://huggingface.co/xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Amazon EC2 G6 instances: https://aws.amazon.com/ec2/instance-types/g6/
