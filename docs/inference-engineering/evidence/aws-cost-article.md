---
title: "Gemma 4 on AWS: What Each Backend Costs per Million Tokens"
published: false
description: "A dress rehearsal of six AWS inference backends behind one Strands agent, priced from the AWS Pricing API: one request at a time, 4 and 16 in parallel, and where Bedrock's per-token price wins."
tags: aws, ai, machinelearning, gemma
---

This article prices six ways to serve a model on AWS behind one Strands agent: Amazon Bedrock, a SageMaker endpoint, vLLM on EC2 g6 and g5g, and a hand-ported Gemma 4 on Inferentia2 and Trainium. Every backend was launched from nothing, checked, timed and torn down in one 26-minute run, and every price comes from the AWS Pricing API on the same day.

https://github.com/xbill9/aws-inference-strands

---

#### What Is the Problem?

An hourly price says little on its own. A $0.80 GPU that decodes 130 tokens a second and a $0.76 Neuron chip that decodes 36 are four times apart per token, and a managed endpoint on the same GPU adds both an hourly premium and a fixed cost to every call. Bedrock has no hourly price at all.

The comparison that matters is dollars per million output tokens at the load you expect, next to what an idle hour costs.

---

#### Which Backends?

| Backend | What serves the model | Model |
|---|---|---|
| Amazon Bedrock | managed API | `us.amazon.nova-micro-v1:0` |
| EC2 g6.xlarge (NVIDIA L4) | vLLM 0.30 | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` |
| SageMaker `ml.g6.xlarge` | AWS vLLM 0.31.0 container | the same repack |
| EC2 g5g.2xlarge (Graviton2 + T4G) | vLLM 0.27.2rc0, patched for sm_75 | `google/gemma-4-E2B-it`, fp16 |
| EC2 inf2.xlarge (Inferentia2) | hand-ported Gemma 4 | Gemma 4 E2B, compiled for Neuron |
| EC2 trn1.2xlarge (Trainium) | the same image | the same build |

---

#### Step 1 — Rehearse From Nothing

One command launches every backend, waits for it, opens the port forwards, checks each one, runs every demo once and tears it all down:

```
stage/rehearse.sh --down
```

The launch script tries every zone, then the next region, when one has no capacity:

```
strands-demo-g5g: launched i-0d20e4c651d93d173	us-east-1a
strands-demo-trn1: no capacity in us-east-2 (subnet-0b0f0d29473b0a624), trying the next zone
strands-demo-inf2: launched i-095444cd1e266afb1	us-east-2a
strands-demo-trn1: no capacity in us-east-2 (subnet-04bce993deca43788), trying the next zone
strands-demo-g6: no capacity in us-east-2 (subnet-0b0f0d29473b0a624), trying the next zone
strands-demo-trn1: launched i-088eb22729e0bb2ce	us-east-2c
strands-demo-g6: launched i-01ae6ce81a5cceb41	us-east-2b
```

Launch finished at 14:29Z. The check asks each backend one question and, where it supports tools, one tool call:

```
  14:39:05Z  PASS  bedrock    chat 0.851s; tool run filters=['id >= 10'] quoted=8 1.18s
  14:39:09Z  PASS  sagemaker  chat 1.662s; tool run filters=['id >= 10'] quoted=8 0.71s
  14:45:03Z  PASS  trn1       chat 1.126s
  14:46:05Z  PASS  g6         chat 1.273s; tool run filters=['id >= 10'] quoted=8 1.21s
  14:46:06Z  PASS  inf2       chat 0.388s
  14:54:09Z  PASS  g5g        chat 0.279s; tool run filters=['id >= 10'] quoted=8 2.55s
  check.py: READY
```

All six answered within 25 minutes of launch. Bedrock and SageMaker passed at the first check, 9 minutes after launch. SageMaker came up on the AWS vLLM 0.31.0 container and passed its tool call.

---

#### Step 2 — Time One Request at a Time

The same 100-token prompt, three times per backend:

```
$ python3 bench_chat.py --repeats 3
Prompt: 'Write a short paragraph about the ocean.', max_tokens=100, 3 repeats per backend

backend     median s  median tok/s  tok/s range     tokens
bedrock        0.767         130.4   126.4-141.6    100-100
g6             0.637         131.9   131.3-132.3    84-84
g5g            2.779          34.9    34.8-35.4     97-97
sagemaker      0.902          96.5    85.3-108.5    87-87
inf2           2.574          36.5    36.4-36.6     94-94
trn1           2.613          36.0    35.7-36.1     94-94
```

The g6 and SageMaker serve the same weights on the same L4. SageMaker took 0.902 s for 87 tokens; at the g6's rate those tokens take 0.660 s, so each SageMaker request carries about 0.24 s of extra time. A one-word answer in the orchestrator demo shows the same gap, 0.426 s against 0.151 s.

---

#### 🔎 Tip: Bedrock's Speed Moves Between Calls

A single `ask.py bedrock` in the same session decoded at 56.9 tok/s; the benchmark three minutes later measured 126.4 to 141.6. The self-hosted backends stay within a few tenths of a token per second between repeats. Time Bedrock over several calls before quoting one number.

---

#### Step 3 — Price Every Backend

`cost.py` fetches On-Demand prices from the AWS Pricing API, reads the benchmark table, and does the arithmetic:

```
$ python3 cost.py results/rehearsal-20261012T1429Z/bench.txt --out results/2026-10-12-cost
prices fetched 2026-10-12T15:01:41+00:00; bench results/rehearsal-20261012T1429Z/bench.txt

One request at a time (bench_chat.py, 100-token answer)

backend      tok/s  USD/hr  USD/month 24x7  USD / 1M output tokens
bedrock      130.4       -          0 idle                    0.14  (per-token list price, input 0.035)
g6           131.9  0.8048             588                    1.69
sagemaker     96.5  1.1267             822                    3.24
g5g           34.9  0.5560             406                    4.43
inf2          36.5  0.7582             553                    5.77
trn1          36.0  1.3438             981                   10.37
```

Dollars per million tokens for an instance is the hourly price over the tokens it produces in an hour, so it assumes the instance is busy the whole hour. Among the instances the g6 is the cheapest per token, at 0.38 times the g5g's cost and 0.16 times Trainium's.

---

#### Step 4 — Add Requests in Parallel

One request at a time leaves most of an L4 idle. vLLM batches requests, so the cost per token falls as load rises. The parallel figures come from the same E2B QAT build measured with `compare.py` on a g6.xlarge (2026-09-29) and an `ml.g6.xlarge` endpoint (2026-09-25), both vLLM 0.30.0, 256-token answers:

```
Requests in parallel (compare.py, 256-token answers, google/gemma-4-E2B-it-qat-w4a16-ct)

backend    parallel    tok/s  USD / 1M output tokens
g6                1    104.8                   2.132
g6                4    410.9                   0.544
g6               16   1485.9                   0.150
sagemaker         1     85.3                   3.667
sagemaker         4    328.7                   0.952
sagemaker        16   1077.2                   0.291
```

Sixteen parallel requests cut the g6's cost per token by 14x, to $0.150 per million. SageMaker scales the same way and stays at about twice the g6's cost per token: its hourly price is 1.4x the g6's, and its per-call overhead was 0.562 s against 0.008 s in those runs.

---

#### Step 5 — Against Bedrock's Per-Token Price

Bedrock bills Nova Micro at $0.035 per million input tokens and $0.14 per million output tokens, with nothing to pay while idle. A g6 running all month costs $588 whether it serves one request or a billion:

```
g6 24x7 = USD 588/month. Bedrock spend matches it at:
  0 input tokens per output token: USD 0.140 per 1M output tokens, 4196M output tokens/month = 1597 tok/s around the clock
  5 input tokens per output token: USD 0.315 per 1M output tokens, 1865M output tokens/month = 710 tok/s around the clock
```

With five input tokens per output token, a g6 has to average 710 output tokens a second, every hour of the month, before it costs less than Bedrock. That is between the 4-parallel and 16-parallel rates measured in Step 4.

---

#### 🔎 Tip: Stop the Instance When the Load Stops

The g6 numbers assume a full hour of work. At low traffic, a g6 that runs only during working hours, or a batch window, keeps its $0.150 per million tokens while busy and drops the idle hours from the bill. It became ready 16 minutes after launch in this run, so it suits scheduled load better than sudden spikes.

---

#### Compare and Contrast

| Backend | $/hr | $/month 24x7 | $ per 1M output tokens, 1 request | $ per 1M output tokens, 16 parallel | Ready after launch |
|---|---|---|---|---|---|
| 🥇 EC2 g6.xlarge, vLLM | 0.8048 | 588 | 1.69 | 0.150 | 16 min |
| 🥈 SageMaker `ml.g6.xlarge` | 1.1267 | 822 | 3.24 | 0.291 | 9 min or less |
| EC2 g5g.2xlarge, vLLM | 0.556 | 406 | 4.43 | not measured | 24 min |
| EC2 inf2.xlarge, hand-ported | 0.7582 | 553 | 5.77 | not measured | 16 min |
| EC2 trn1.2xlarge, hand-ported | 1.3438 | 981 | 10.37 | not measured | 15 min |
| Bedrock, Nova Micro | none | 0 when idle | 0.14 output, 0.035 input | 0.14 output, 0.035 input | always on |

---

#### So, Which One?

When a Bedrock model does the job, Bedrock costs less at every load measured here and needs nothing started. When you need your own weights, as with this Gemma 4 repack, a g6 running vLLM is the cheapest place to run them at every load, from one request to sixteen, and it decodes the fastest. SageMaker runs the same container on the same GPU for about twice the cost per token, in exchange for a managed endpoint. The g5g, inf2 and trn1 cost 2.6 to 6.1 times the g6 per token for this model.

---

#### Summary

The goal of this article was to price six AWS inference backends on the same day with the same prompts. The key to the solution was one rehearsal script that launches, checks and times every backend, and one cost script that pulls list prices from the AWS Pricing API and does all the arithmetic. The results were:

- 🟢 **All six backends launched, passed their checks and tore down in one 26-minute run**, with every tool call answering 8 and sending the filter `id >= 10`.
- 🟢 **A g6 with vLLM costs least per token of the five instances**: $1.69 per million output tokens one request at a time, $0.150 at sixteen in parallel.
- 🟢 **Parallel requests cut the g6's cost per token by 14x** between one and sixteen requests.
- ⚠️ **SageMaker costs about twice the g6 per token on the same GPU**, from a 1.4x hourly price and about 0.24 s of extra time per request.
- ⚠️ **First-choice zones had no capacity** for g6 and trn1 in us-east-2; the launch script placed them in the next zone.
- ❌ **A g6 running all month costs more than Bedrock Nova Micro** until it averages 710 output tokens a second around the clock, at five input tokens per output token.

Scope: one rehearsal on 2026-10-12 with three timed requests per backend, plus parallel runs of the E2B QAT build on 2026-09-25 (SageMaker) and 2026-09-29 (g6). On-demand list prices from the AWS Pricing API, us-east-2 except the g5g and Bedrock in us-east-1. Client on a laptop reaching EC2 through SSM port forwards; in the parallel runs the g6 client ran on the instance. Bedrock serves Nova Micro, a different model from the Gemma 4 builds on the other five rows, so its row compares price models, and its per-token price applies to whichever model you choose. Per-token costs for instances assume a fully busy hour.

The strategy for pricing Gemma 4 inference on AWS, from Bedrock to Trainium, was validated with an incremental step by step approach.

---

#### References

* [aws-inference-strands | GitHub](https://github.com/xbill9/aws-inference-strands)
* [Gemma 4 Inference on AWS: Bedrock, SageMaker, GPUs, Inferentia and Trainium Behind One Strands Agent | DEV](https://github.com/xbill9/aws-inference-strands/blob/main/docs/article/devto-aws-inference-strands.md)
* [gemma4-dev | GitHub](https://github.com/xbill9/gemma4-dev)
* [Strands Agents | GitHub](https://github.com/strands-agents/sdk-python)
* [AWS Price List Query API | AWS](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/using-price-list-query-api.html)
* [Amazon Bedrock pricing | AWS](https://aws.amazon.com/bedrock/pricing/)
* [Amazon EC2 G6 instances | AWS](https://aws.amazon.com/ec2/instance-types/g6/)
* [Amazon SageMaker AI real-time inference | AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
