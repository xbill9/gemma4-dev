---
title: "Inference on AWS: Precision, Memory and Six Backends Behind One Strands Agent"
published: false
description: "The companion to the talk: what bits per weight, memory bandwidth and the serving engine do to Gemma 4 on Bedrock, SageMaker, EC2 g6 and g5g, Inferentia2 and Trainium, with the cost of each, measured on one day."
tags: aws, ai, machinelearning, gemma
cover_image: https://raw.githubusercontent.com/xbill9/aws-inference-strands/main/docs/article/devto-companion-cover.5b16ff3b.jpg
---

This article is the written companion to the talk "Inference on AWS". One Strands agent sends the same question to six AWS backends: Amazon Bedrock, a SageMaker endpoint, vLLM on EC2 g6 and g5g, and a hand-ported Gemma 4 on Inferentia2 and Trainium. The talk adds two ideas that explain most of the results: the precision a model is stored and computed in, and the memory bandwidth of the chip that serves it.

https://github.com/xbill9/aws-inference-strands

The slides: https://docs.google.com/presentation/d/1Ax8KGF8T0EYffshY7WfUM61QcHmLk0q72EE7G_hIwgM/edit

---

#### What Is the Problem?

AWS sells a model as an API, as a managed endpoint, as a GPU by the hour and as two families of its own chips. The same small model, Gemma 4 E2B, runs at 36 tokens a second on one of them and 132 on another, and the hourly prices do not explain the gap. Three properties do: how many bits each weight takes, how fast the chip can read them, and how well the serving engine uses the chip.

---

#### Which Backends?

![One Strands agent sends the same question to four kinds of backend: a managed API, a managed endpoint, your own GPU and your own AWS chip](https://raw.githubusercontent.com/xbill9/aws-inference-strands/main/docs/slides/arch-ways.76cbb8fc.png)

| Backend | What serves the model | Model |
|---|---|---|
| Amazon Bedrock | managed API | `us.amazon.nova-micro-v1:0` |
| SageMaker `ml.g6.xlarge` | AWS vLLM 0.31.0 container | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` |
| EC2 g6.xlarge (NVIDIA L4) | vLLM 0.30 | the same repack |
| EC2 g5g.2xlarge (Graviton2 + T4G) | vLLM 0.27.2rc0, patched for sm_75 | `google/gemma-4-E2B-it`, fp16 |
| EC2 inf2.xlarge (Inferentia2) | hand-ported Gemma 4, `xbill9/gemma4-optb:slim` | Gemma 4 E2B, compiled for Neuron |
| EC2 trn1.2xlarge (Trainium) | the same image | the same build |

---

#### Step 1 — Precision: What You Trade

Every format below stores the same Gemma 4 E2B. Fewer bits per weight means less memory, less disk and less data to move for every token. It buys speed only where the chip multiplies that format natively, and it costs accuracy unless the weights were trained for it.

![Three panels for Gemma 4 E2B: weights in memory per format, suite accuracy per format, and which formats the L4, T4G and Neuron multiply natively](https://raw.githubusercontent.com/xbill9/aws-inference-strands/main/docs/slides/precision.1ac3aeaa.png)

| Format | E2B weights in memory | Suite accuracy, 3,880 records |
|---|---:|---:|
| bf16 | 9.42 GiB | 0.683 |
| fp8, from Google's QAT weights | 7.07 GiB | 0.670 |
| int8, from Google's QAT weights | 7.07 GiB | 0.686 |
| int8, rounded from bf16 | — | 0.670 |
| int4, QAT | 6.32 GiB | 0.678 |
| int4 with int4 embedding tables | 2.85 GiB | 0.681 |

Two results carry the slide. The int4 builds made from Google's quantization-aware-trained (QAT) checkpoint stay within half a point of bf16, while an int8 build rounded from bf16 loses 1.3 points. And E2B's embedding tables are most of its weight: packing them to int4 takes the model from 6.32 GiB to 2.85 GiB.

Native support decides whether the smaller format is also faster:

| Chip | bf16 | fp8 | int8 | int4 |
|---|---|---|---|---|
| NVIDIA L4 (g6, SageMaker) | yes | yes | yes | widened to 16-bit |
| NVIDIA T4G (g5g) | emulated | no | yes | widened to 16-bit |
| NeuronCore-v2 (inf2, trn1) | yes | in hardware; compiler support pending | yes | widened to bf16, 27x per call |

Memory figures come from the E2B data-type sweep on one AMD MI300X (2026-10-07) and accuracy from the same checkpoint files on one TPU v5e (2026-09-29), both in `gemma4-dev`.

---

#### Step 2 — Memory and Bandwidth Across AWS

Generating a token reads every weight once, so memory bandwidth sets the ceiling on tokens per second for one request. The engine decides how close a backend gets to that ceiling.

| Backend | Accelerator | Memory | Bandwidth | Host RAM | tok/s, one request |
|---|---|---|---|---|---:|
| g6.xlarge | NVIDIA L4 | 24 GB | 300 GB/s | 16 GiB | 131.9 |
| SageMaker ml.g6.xlarge | NVIDIA L4 | 24 GB | 300 GB/s | 16 GiB | 96.5 |
| g5g.2xlarge | NVIDIA T4G | 16 GB | 320 GB/s, 277 measured | 16 GiB | 34.9 |
| inf2.xlarge | Inferentia2 | 2 x 16 GiB | ~817 GB/s | 16 GiB | 36.5 |
| trn1.2xlarge | Trainium | 2 x 16 GiB | ~613 GB/s | 32 GiB | 36.0 |
| Bedrock | Nova Micro | not disclosed | not disclosed | — | 130.4 |

The Neuron chips have 2.0 to 2.7 times the L4's bandwidth and decode at 0.28 times its speed. Their 32 GB is two 16 GiB pools, one per NeuronCore, so a model either splits across both cores or fits in 16 GiB. AWS publishes Neuron bandwidth per instance; the per-chip figures here divide the largest instance's 9.8 TB/s by its chip count.

---

#### 🔎 Tip: Quota Is Permission to Ask

Quota does not reserve a machine. On the rehearsal day the first zone tried in us-east-2 had no capacity for g6 or trn1, and on an earlier run inf2 found no capacity in us-east-2a with 80 vCPUs of quota free. `stage/up.sh` tries every zone and then the next region, so a launch script that does the same saves a demo.

---

#### Step 3 — Bedrock and SageMaker: the Managed Pair

Bedrock needs an IAM permission and an inference profile id, and bills per token: $0.035 per million input tokens and $0.14 per million output tokens for Nova Micro.

```python
from strands.models import BedrockModel
model = BedrockModel(model_id="us.amazon.nova-micro-v1:0", region_name="us-east-1")
```

SageMaker runs a container you choose on an instance you choose:

![You create a model, an endpoint config and an endpoint; SageMaker runs the vLLM container on an ml.g6.xlarge with an L4, health-checks it and logs to CloudWatch; the agent calls InvokeEndpoint and the weights come from Hugging Face at start-up](https://raw.githubusercontent.com/xbill9/aws-inference-strands/main/docs/slides/sagemaker.472a0865.png)

Three API calls build it, `create-model`, `create-endpoint-config` and `create-endpoint`, and vLLM flags pass as `SM_VLLM_*` environment variables. On the rehearsal day the endpoint reached `InService` about 9 minutes after deploy on the AWS vLLM 0.31.0 container. It serves the same weights on the same GPU as the g6, and each request carries about 0.24 s of extra time through the managed path: 87 tokens took 0.902 s against 0.660 s at the g6's rate.

---

#### Step 4 — vLLM on an L4

The g6.xlarge runs vLLM 0.30 with two flags that turn on tool calls:

```
--enable-auto-tool-choice --tool-call-parser gemma4
```

| NVIDIA L4 | |
|---|---|
| Family | Ada Lovelace, compute capability 8.9 |
| Memory | 24 GB |
| Bandwidth | 300 GB/s |
| BF16 / FP16 tensor | 121 TFLOPS dense |
| FP8 tensor | 242.5 TFLOPS dense |
| Power | 72 W, single-slot PCIe |

It was ready 16 minutes after launch and decoded 131.9 tokens a second for one request, the fastest of the five instances.

---

#### Step 5 — Arm and CUDA, Then and Now

The g5g pairs a Graviton2 Arm host with an NVIDIA T4G. AWS calls G5g "the first Arm-based instances in a major cloud to feature GPU acceleration." Serving Gemma 4 on it needs vLLM built from source for sm_75 with a shared-memory patch, because the arm64 image ships without Turing kernels.

The same pairing reaches the desk this fall. NVIDIA announced RTX Spark on May 31, 2026:

| | EC2 g5g | NVIDIA RTX Spark |
|---|---|---|
| CPU | AWS Graviton2, Arm | 20-core NVIDIA Grace, Arm |
| GPU | NVIDIA T4G, Turing | Blackwell RTX, 6,144 CUDA cores |
| CPU to GPU | PCIe | NVLink-C2C |
| Memory | 16 GB GPU + 16 GB host | up to 128 GB unified |

The aarch64 and CUDA build work that the g5g required is the platform those machines run.

---

#### Step 6 — Why Neuron Is Hard to Serve On

The Inferentia2 and Trainium rows run the same hand-ported image, measured on `inf2.xlarge`:

- No vLLM release serves Gemma 4 on Neuron, so the engine is a hand port and ignores tool definitions.
- The graph compiles ahead of time for fixed shapes, 512 tokens in total and 128 for the prompt. A new shape costs about 3.9 s to compile.
- Memory is 2 x 16 GiB, one per NeuronCore. A single gather above roughly 4 to 5 GB returns zeros with no error.
- int4 weights get a dense bf16 copy before each matrix multiply: 27 times slower per call than plain bf16.

That is the 0.28x in Step 2: the chips have the bandwidth, and the software stack for this model has not caught up.

---

#### Step 7 — One Strands Agent, Every Backend

The agent asks how many of eleven ids are 10 or more and has a `count_ids` tool that computes it. Every backend that supports tool calls, three runs each on the rehearsal day:

```
bedrock    run 1/3  ok   quoted=8  filters=['id >= 10']  1.19s
g6         run 2/3  ok   quoted=8  filters=['id >= 10']  0.4s
g5g        run 1/3  ok   quoted=8  filters=['id >= 10']  2.35s
sagemaker  run 1/3  ok   quoted=8  filters=['id >= 10']  0.63s
```

All twelve runs answered 8 and sent the filter `id >= 10`. Grading the filter as well as the number matters: a wrong filter returns an exact number with a clean source.

A second agent on Bedrock calls all six backends as tools and quotes a scoreboard sorted in code:

```
| vLLM on EC2 g6 (L4) | Paris | 0.151 | 13.2 |
| SageMaker endpoint (vLLM, L4) | Paris | 0.426 | 4.7 |
| Amazon Bedrock | Paris | 0.861 | 2.3 |
```

---

#### Step 8 — What Each Backend Costs

`cost.py` pulls On-Demand prices from the AWS Pricing API and does the arithmetic:

| Backend | $/hr | $/month 24x7 | $ per 1M output tokens, 1 request | $ per 1M output tokens, 16 parallel |
|---|---:|---:|---:|---:|
| EC2 g6.xlarge | 0.8048 | 588 | 1.69 | 0.150 |
| SageMaker ml.g6.xlarge | 1.1267 | 822 | 3.24 | 0.291 |
| EC2 g5g.2xlarge | 0.556 | 406 | 4.43 | not measured |
| EC2 inf2.xlarge | 0.7582 | 553 | 5.77 | not measured |
| EC2 trn1.2xlarge | 1.3438 | 981 | 10.37 | not measured |
| Bedrock, Nova Micro | — | 0 when idle | 0.14 | 0.14 |

A g6 running all month costs more than Bedrock Nova Micro until it averages 710 output tokens a second around the clock, at five input tokens per output token.

---

#### 🔎 Tip: Debug With the Small Model

Every path in the talk was brought up with E2B first. Flags, port forwards and tool parsing fail the same way at any size, and E2B loads on Neuron in about 2 minutes against close to 4 for the 26B.

---

#### Compare and Contrast

| Backend | Precision served | Bandwidth | tok/s | $ per 1M output tokens, 1 request | Tool calls |
|---|---|---|---:|---:|---|
| 🥇 EC2 g6, vLLM | int4 weights, int4 tables | 300 GB/s | 131.9 | 1.69 | yes |
| 🥈 Bedrock | managed | not disclosed | 130.4 | 0.14 | yes |
| 🥉 SageMaker, vLLM | int4 weights, int4 tables | 300 GB/s | 96.5 | 3.24 | yes |
| EC2 g5g, vLLM | fp16 | 320 GB/s | 34.9 | 4.43 | yes |
| EC2 inf2, hand-ported | compiled for Neuron | ~817 GB/s | 36.5 | 5.77 | no |
| EC2 trn1, hand-ported | compiled for Neuron | ~613 GB/s | 36.0 | 10.37 | no |

---

#### So, Which One?

Start on Bedrock when one of its models does the job: nothing to run and the lowest cost per token here. Move to vLLM on a g6 for your own weights, quantized from a QAT checkpoint so the smaller format keeps its accuracy; it was the fastest and cheapest instance. SageMaker runs the same container as a managed endpoint for about twice the cost per token. The g5g is the Arm and CUDA path, and Inferentia2 and Trainium pay off once the serving stack supports the model.

---

#### Summary

The goal of this article was to explain the talk's results through precision, memory bandwidth and the serving engine. The key to the solution was one Strands agent driving six backends on the same day, with every count, ranking and price computed in code. The results were:

- 🟢 **int4 from QAT weights stays within half a point of bf16**, and int4 embedding tables shrink E2B from 6.32 GiB to 2.85 GiB.
- 🟢 **vLLM on a g6 decoded fastest**, 131.9 tok/s for one request at $1.69 per million output tokens, $0.150 at 16 in parallel.
- 🟢 **Every tool-calling backend picked the right filter**, 12 of 12 runs.
- ⚠️ **SageMaker adds about 0.24 s per request** on the same GPU and costs about twice the g6 per token.
- ⚠️ **Bandwidth sets a ceiling the engine has to reach**: Neuron has 2.0 to 2.7 times the L4's bandwidth and decodes at 0.28 times its speed.
- ❌ **No vLLM release serves Gemma 4 on Neuron**, so those backends run a hand port with fixed shapes and no tool calls.

Scope: one rehearsal on 2026-10-12 in us-east-2, with the g5g and Bedrock in us-east-1, three timed requests per backend, client on a laptop through SSM port forwards. Memory per format comes from one MI300X and accuracy from one TPU v5e, both serving the same Gemma 4 E2B checkpoint files, and the 16-parallel prices from earlier compare.py runs of the E2B QAT build on a g6 and an ml.g6.xlarge. Bedrock serves Nova Micro, a different model from the Gemma 4 builds on the other rows. On-demand list prices from the AWS Pricing API on 2026-10-12.

The strategy for serving Gemma 4 on AWS from Bedrock to Trainium behind one Strands agent was validated with an incremental step by step approach.

---

#### References

* [aws-inference-strands | GitHub](https://github.com/xbill9/aws-inference-strands)
* [Inference on AWS slides | Google Slides](https://docs.google.com/presentation/d/1Ax8KGF8T0EYffshY7WfUM61QcHmLk0q72EE7G_hIwgM/edit)
* [gemma4-dev | GitHub](https://github.com/xbill9/gemma4-dev)
* [Strands Agents | GitHub](https://github.com/strands-agents/sdk-python)
* [NVIDIA L4 | NVIDIA](https://www.nvidia.com/en-us/data-center/l4/)
* [NVIDIA and Microsoft Reinvent Windows PCs for the Age of Personal AI | NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-microsoft-windows-pcs-agents-rtx-spark)
* [Amazon EC2 G5g instances | AWS](https://aws.amazon.com/ec2/instance-types/g5g/)
* [Amazon EC2 Inf2 instances | AWS](https://aws.amazon.com/ec2/instance-types/inf2/)
* [Amazon EC2 Trn1 instances | AWS](https://aws.amazon.com/ec2/instance-types/trn1/)
* [AWS Neuron data types | AWS](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/data-types.html)
* [Amazon SageMaker AI real-time inference | AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
* [Amazon Bedrock pricing | AWS](https://aws.amazon.com/bedrock/pricing/)
