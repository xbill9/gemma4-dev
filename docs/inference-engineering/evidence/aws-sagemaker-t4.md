---
title: "Gemma 4 on Amazon SageMaker: The NVIDIA T4 Decodes at 0.8x of the L4 With the Same Answers"
published: false
description: "Gemma 4's 4-bit builds on SageMaker's smallest GPU, an NVIDIA T4, against the L4: a Turing patch for vLLM, the host image the CUDA 13 container needs, speed, memory, answers and cost per token."
tags: aws, sagemaker, gemma, vllm
cover_image: https://raw.githubusercontent.com/xbill9/sagemaker-gemma/main/docs/articles/sagemaker-gemma-t4/devto-cover.0adb9f63.jpg
---

This article deploys Gemma 4 to the smallest GPU Amazon SageMaker offers, an NVIDIA T4, and measures it against the L4 from the earlier parts. A suite of Python MCP tools is built to simplify management of the vLLM hosted deployment.

https://github.com/xbill9/sagemaker-gemma

| | |
|---|---|
| Models | Gemma 4 E2B, E4B, 12B and 26B A4B, 4-bit weights and 4-bit embeddings |
| Hardware | SageMaker `ml.g4dn.xlarge`, 1x NVIDIA T4, 16 GB; compared with `ml.g6.xlarge`, 1x L4 |
| Region | `us-east-2` |
| Software | AWS vLLM SageMaker container 0.30.0, with a Turing attention patch |
| Result | The T4 decodes at **0.77x to 0.82x** of the L4 and gives identical answers; 12B is the largest build that fits |

---

#### Where Do I Start?

This is part four of a series. Part one deploys Gemma 4 to a SageMaker endpoint with the aws CLI and an MCP server: https://dev.to/aws-builders/gemma-4-on-an-amazon-sagemaker-endpoint-aws-cli-nvidia-l4-and-an-mcp-server-2c9d

Part two measures Google's QAT checkpoint against the full-size bf16 release: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-qat-weights-decode-205x-faster-than-bf16-on-one-l4-318m

Part three repacks the QAT weights with 4-bit embeddings: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-4-bit-embeddings-decode-up-to-139x-faster-on-one-l4-36mf

This article takes part three's builds down one GPU generation.

---

#### Where Does This Fit?

SageMaker JumpStart lists Gemma 4 from `ml.g6e.xlarge`, one NVIDIA L40S, for E2B, and 12B only on `ml.g6e.16xlarge`; none of its Gemma 4 entries lists a T4. Gemma 4 on Turing GPUs is an open vLLM issue, #38918, "Gemma4 on Turing GPUs (SM 7.5): all attention backends hit shared memory limits", reported again on vLLM 0.29.0 in September. The builds served here, with 4-bit embeddings, are the author's repacks of Google's QAT weights on Hugging Face.

The 26B A4B build is the first W4A16 checkpoint of that model to keep Google's QAT weights on their 4-bit grid. In layer 0's query projection, 0 of its 11,534,336 weights change level, where an AWQ build of the same QAT export, published in June, moves 30.3% of them by a full level or more. `grid_check.py` reads that one tensor from each checkpoint over HTTP and compares them.

---

#### At This Point You Should Have…

- The repository cloned and `make test` passing
- An `aws login` session in an account with SageMaker endpoint quota for `ml.g4dn.xlarge`
- Part three's measurements on the L4, which this article compares against

---

#### What Is the Smallest GPU on SageMaker?

The project's MCP server has a `check_quotas` tool that reads the account's endpoint quota for each instance type in every US region:

```
| Instance | us-east-1 | us-east-2 | us-west-1 | us-west-2 |
| `ml.g4dn.xlarge` | 2 | 2 | 2 | 2 |
| `ml.g5g.xlarge` | - | - | - | - |
| `ml.g5.xlarge` | 2 | 2 | - | 2 |
| `ml.g6.xlarge` | 1 | 1 | - | 1 |
| `ml.inf2.xlarge` | 0 | 2 | - | 0 |
```

A dash means SageMaker has no quota entry for the type. `ml.g4dn.xlarge`, one T4 with 16 GB, is the smallest GPU on offer. The Graviton T4G (`g5g`) exists on EC2 only. Inferentia2 is listed, but it runs the Neuron SDK and needs a different container and a compiled model.

---

#### Step 1 — Patch the Container for Turing

Gemma 4's full-attention layers are 512 wide. vLLM serves them with its Triton attention kernel, whose tile asks for 98,304 bytes of shared memory per block. A T4 is a Turing GPU and allows 65,536, so the engine stops at start-up:

```
triton.runtime.errors.OutOfResources: shared memory,
Required: 98304, Hardware limit: 65536
```

vLLM 0.30.0, the version in the SageMaker container, has no fix for it. The author's Compute Engine T4 serves Gemma 4 with a script that halves the tiles on pre-Ampere GPUs, and the SageMaker image gets the same script, in a Dockerfile built `FROM` the stock container:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY patch_triton_turing.py /opt/turing/patch_triton_turing.py
RUN set -eu; \
    target="$(python3 -c 'import importlib.util, os; print(os.path.join(importlib.util.find_spec("vllm").submodule_search_locations[0], "v1/attention/ops/triton_unified_attention.py"))')"; \
    python3 /opt/turing/patch_triton_turing.py "$target"; \
    python3 /opt/turing/patch_triton_turing.py --check "$target"; \
    python3 -c 'import torch, sys; a = torch._C._cuda_getArchFlags(); print("torch arch:", a); sys.exit(0 if "sm_75" in a else 1)'
```

The last line refuses the build if PyTorch in the image has no Turing kernels. CodeBuild builds the image and pushes it to a private ECR repository, so the 8 GB base image never touches the local disk:

```
#8 1.092 patch_triton_turing: patched /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention.py
#8 1.092   smem budget   : 60000 B of Turing's 65536 hard limit
#8 5.652 torch arch: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
```

The patch is a no-op on an L4 or newer, and the image keeps the stock entrypoint, so every `SM_VLLM_*` setting works as before. The Dockerfile and `buildspec.yml` are in the repository, in the `turing` directory.

---

#### Step 2 — Pick the Host Image

A SageMaker production variant runs on one of several host images, each with its own NVIDIA driver, chosen by `InferenceAmiVersion`:

```
al2-ami-sagemaker-inference-gpu-2      NVIDIA driver 535, CUDA 12.2
al2-ami-sagemaker-inference-gpu-3-1    NVIDIA driver 550, CUDA 12.4
al2023-ami-sagemaker-inference-gpu-4-1 NVIDIA driver 580, CUDA 13.0
```

The vLLM 0.30.0 container is built on CUDA 13:

```
NVIDIA_REQUIRE_CUDA=cuda>=13.0 ...
CUDA_VERSION=13.0.2
```

`sm.py` passes the host image through from `.env`:

```
INFERENCE_AMI_VERSION=al2023-ami-sagemaker-inference-gpu-4-1
```

---

#### 🔎 Tip: No Log Group Means the Container Never Started

Without `InferenceAmiVersion`, an `ml.g4dn.xlarge` endpoint with this image ends about six minutes after `Creating` with:

```
CannotStartContainerError. Please ensure the model container for variant AllTraffic starts correctly when invoked with 'docker run <image> serve'
```

and no CloudWatch log group at all. A missing log group points at the host, before any code in the image runs: compare the container's `NVIDIA_REQUIRE_CUDA` with the driver of the host image. With the setting above, the same image on the same instance type reached `InService` in 13.2 minutes.

---

#### Step 3 — Run the Sweep

The sweep serves part three's text-only builds with 4-bit embeddings, one endpoint at a time: deploy, measure, tear down. Settings match the L4 runs except for the data type, since Turing has no bf16:

```
IMAGE_URI=<account>.dkr.ecr.us-east-2.amazonaws.com/sagemaker-gemma-vllm:0.30.0-sagemaker-v1.3-sm75
INFERENCE_AMI_VERSION=al2023-ami-sagemaker-inference-gpu-4-1
INSTANCE_POOLS=ml.g4dn.xlarge,ml.g4dn.2xlarge
MAX_MODEL_LEN=8192
SM_VLLM_DTYPE=float16
```

Each endpoint went from the deploy call to `InService` in 13.3 minutes. `compare.py` then measures single-request decode, 1 to 16 parallel requests and 40 questions at temperature 0, as in parts two and three, and `compare.py combine` sets each T4 run beside its L4 run:

```
decode_tokens_per_second              35.0                28.5  0.81
load_c16_tokens_per_second           411.8              216.45  0.53
quality_correct                         40                  40
identical answers: 40/40
```

---

#### T4 Against L4: Speed

| Model | Decode, T4 (tok/s) | Decode, L4 (tok/s) | T4 / L4 | 16 parallel, T4 / L4 |
|---|---:|---:|---:|---:|
| E2B | 108.5 | 141.7 | 0.77 | 0.63 |
| E4B | 65.6 | 79.9 | 0.82 | 0.57 |
| 12B | 28.5 | 35.0 | 0.81 | 0.53 |

One user at a time, the T4 runs at about 0.8x of the L4. Under load the gap widens with model size: at 16 parallel requests the T4 delivers 0.63x of the L4 at E2B and 0.53x at 12B.

The Compute Engine T4 serving the same E2B build with the same patch decodes at 109.7 tok/s, against 108.5 here. That rig runs vLLM 0.29.0 from pip with a 16,384-token context, so the agreement is a cross-check and not a controlled comparison.

---

#### Does It Still Answer Correctly?

All three models give the same 40 answers on the T4 as on the L4, byte for byte: 36 of 40 correct for E2B and E4B, 40 of 40 for 12B. The T4 computes in fp16 and runs the clamped attention tiles, and neither changed an answer.

---

#### What the Engine Allocates

| Model | Weights (GiB) | KV cache, T4 (tokens) | KV cache, L4 (tokens) |
|---|---:|---:|---:|
| E2B | 2.86 | 660,033 | 1,209,977 |
| E4B | 4.52 | 194,336 | 376,156 |
| 12B | 7.36 | 20,354 | 52,541 |

The weights take the same memory on both GPUs. The T4 has less left for the KV cache: 12B keeps 20,354 tokens, which is 2.48 requests at the full 8,192-token context.

---

#### 26B Does Not Fit

The 26B A4B build holds 14.2 GiB of weights on the L4. On the T4 it ran with the reduced settings 31B needed in part three: memory 0.97, a 1,024-token context and four sequences. The weights loaded and the engine then ran out of memory:

```
memory allocation failed with OOM on device 0 while trying to allocate 253755392 bytes (free: 191627264, total: 15636037632).
```

The T4 reports 14.56 GiB in total. Of the builds tested, 12B is the largest that serves on one T4.

---

#### And Price/Performance?

SageMaker's on-demand hosting price in `us-east-2`, from the AWS Price List API on 2026-09-30, is $0.736 an hour for `ml.g4dn.xlarge` and $1.1267 for `ml.g6.xlarge`. Per million output tokens at 16 parallel requests:

| Model | T4 | L4 | T4 / L4 |
|---|---:|---:|---:|
| E2B | $0.260 | $0.249 | 1.05 |
| E4B | $0.419 | $0.369 | 1.14 |
| 12B | $0.945 | $0.760 | 1.24 |

The T4 costs 0.65x the L4 per hour and delivers 0.53x to 0.63x of its throughput under load, so a busy T4 costs more per token, by 5% at E2B and 24% at 12B. A mostly idle endpoint is billed by the hour, and there the T4 costs 0.65x.

---

#### 🔎 Tip: Lower the Health-Check Timeout While Prototyping

`sm.py` sets `ContainerStartupHealthCheckTimeoutInSeconds` to 1800, which leaves room for a large model to download and load. A container that crashes at start holds the endpoint in `Creating` for that whole window: the 26B endpoint ran out of memory at 15:33 and was marked `Failed` at 15:59, 36.5 minutes after `Creating`. For a small model that loads in under two minutes, a few hundred seconds is enough, and a failed start comes back that much sooner.

---

#### So, Which One?

| Workload | GPU | Why |
|---|---|---|
| 🥇 E2B or E4B, light traffic | T4 | 0.65x the hourly price, 0.8x the single-user speed |
| 🥈 Any size, steady load | L4 | Lower cost per token, 1.8x to 2.6x the KV cache |
| 🥉 12B | L4 | 12B fits the T4 with room for 2.48 full-length requests |
| 26B A4B and 31B | L4 or larger | They do not fit a T4 |

---

#### What Stops the Meter

Every endpoint was deleted after its measurement, and a watchdog deleted anything left behind when the sweep exited:

```
2026-09-30T15:59:45Z gemma-4-e2b-emb4-t4: gone
2026-09-30T15:59:46Z gemma-4-e4b-emb4-t4: gone
2026-09-30T15:59:46Z gemma-4-12b-emb4-t4: gone
2026-09-30T15:59:47Z gemma-4-26b-emb4-t4: gone
```

The build resources cost nothing while idle: an ECR repository, a CodeBuild project, its role and a source bucket in `us-east-2`.

---

#### Summary

The goal of this article was to serve Gemma 4 on the smallest GPU SageMaker offers and measure what it gives up against the L4. The key to the solution was a Turing patch for vLLM in a derived container, on the host image with driver 580. The T4 results were:

- 🟢 E2B, E4B and 12B serve on one T4 and give the same 40 answers as the L4
- 🟢 Single-request decode runs at **0.77x to 0.82x** of the L4
- 🟢 E2B decodes at **108.5 tok/s**, in line with a Compute Engine T4 at 109.7
- ⚠️ At 16 parallel requests the T4 delivers **0.53x to 0.63x** of the L4, and costs 5% to 24% more per token
- ⚠️ The CUDA 13 container needs `InferenceAmiVersion` set to the driver 580 host on `ml.g4dn`
- ❌ 26B A4B does not fit one T4

Scope: one account, SageMaker `ml.g4dn.xlarge` in `us-east-2`, vLLM 0.30.0 from the AWS container with the Turing patch, fp16, one deployment per model on 2026-09-30. The L4 figures are part three's, measured on 2026-09-29 in bf16 with the stock container. Throughput was measured through the aws CLI from one client machine, and prices are on-demand list prices.

The strategy for using MCP for SageMaker deployment and benchmarking was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/sagemaker-gemma
- Part one, deploying Gemma 4 to SageMaker: https://dev.to/aws-builders/gemma-4-on-an-amazon-sagemaker-endpoint-aws-cli-nvidia-l4-and-an-mcp-server-2c9d
- Part two, QAT against bf16: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-qat-weights-decode-205x-faster-than-bf16-on-one-l4-318m
- Part three, 4-bit embeddings: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-4-bit-embeddings-decode-up-to-139x-faster-on-one-l4-36mf
- E2B emb4: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
- E4B emb4: https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4
- 12B emb4: https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct-text-emb4
- 26B A4B emb4: https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4
- 26B A4B W4A16 repack: https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct
- vLLM issue 38918, Gemma 4 on Turing: https://github.com/vllm-project/vllm/issues/38918
- SageMaker ProductionVariant: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html
- SageMaker real-time inference: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
