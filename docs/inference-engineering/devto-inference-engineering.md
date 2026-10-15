---
title: "Inference Engineering with Gemma4"
published: false
description: "A step by step tour of serving Gemma 4 on Hugging Face, Google Cloud, Cloud TPU, Colab, AWS, Azure, AMD and local hardware, built around one trade-off: what you give up for fewer bits, and where the chip, the engine and the batch decide whether it pays."
tags: ai, machinelearning, gemma, cloud
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/inference-engineering/devto-cover.f243064d.jpg
---

This article provides a step by step tour of inference engineering across every platform I have served Gemma 4 on: Hugging Face as the distribution layer, Google Cloud's managed and GPU paths, Cloud TPU, Colab, AWS from Bedrock to Trainium, Azure, an AMD Instinct MI300X, and local hardware from an 8 GB RTX 3050 down to a Snapdragon laptop. Each platform gets its own section first. A compare and contrast closes the article.

https://github.com/xbill9/gemma4-dev

---

#### What Is Inference Engineering?

Training produces a set of weights. Inference engineering is everything between those weights and a token arriving at a user: which format the weights are stored in, which chip reads them, which engine schedules the work, how many requests share a batch, and what all of it costs per hour and per token.

The same small model, Gemma 4 E2B, decoded at 36 tokens a second on one AWS chip and 132 on another on the same day. Hourly prices do not explain the gap. Four properties do: how many bits each weight takes, how fast the chip can read them, how well the serving engine uses the chip, and how many requests are served together.

---

#### The Trade-Off: Precision, What You Trade

This is the slide the rest of the article keeps coming back to. It comes from my AWS Community Builders talk, "Inference on AWS", and it holds on every platform below.

![Three panels for Gemma 4 E2B: weights in memory per format, suite accuracy per format, and which formats the L4, T4G and Neuron multiply natively](https://raw.githubusercontent.com/xbill9/aws-inference-strands/main/docs/slides/precision.1ac3aeaa.png)

Every format below stores the same Gemma 4 E2B:

| Format | E2B weights in memory | Suite accuracy, 3,880 records |
|---|---:|---:|
| bf16 | 9.42 GiB | 0.683 |
| fp8, from Google's QAT weights | 7.07 GiB | 0.670 |
| int8, from Google's QAT weights | 7.07 GiB | 0.686 |
| int8, rounded from bf16 | — | 0.670 |
| int4, QAT | 6.32 GiB | 0.678 |
| int4 with int4 embedding tables | 2.85 GiB | 0.681 |

The slide makes three claims, and each one is a column of the trade:

- **Memory.** Fewer bits per weight means less memory, less disk and less data to move for every token. E2B's embedding tables are most of its weight, so packing them to int4 takes the model from 6.32 GiB to 2.85 GiB.
- **Quality.** Formats built from Google's quantization-aware-trained (QAT) checkpoint stay within half a point of bf16. An int8 build rounded from bf16 loses 1.3 points.
- **Native math.** A smaller format is faster only where the chip multiplies it natively. Everywhere else the weights are widened back to 16 bits before the multiply, and the win is bandwidth alone.

The third column changes with every chip in this article:

| Chip | bf16 | fp8 | int8 | int4 |
|---|---|---|---|---|
| NVIDIA L4 (Ada) | yes | yes | yes | widened |
| NVIDIA T4 / T4G (Turing) | no, fp16 instead | no | yes | widened |
| AWS NeuronCore-v2 | yes | compiler support pending | yes | widened, 27x per call |
| Google TPU v5e, v6e | yes | storage only | yes, 2x bf16 | widened |
| AMD MI300X (CDNA 3) | yes | yes, `e4m3fnuz` only | runs, no win | widened |

Memory, quality and native math are the precision trade. Bandwidth, the engine and the batch decide how much of it reaches a served token, and the platform sections below measure each one.

---

#### Which Platforms?

| Platform | What serves the model | Sections |
|---|---|---|
| Hugging Face | the registry every other platform pulls from | Step 1 |
| Google Cloud | Gemini API, Vertex AI, Cloud Run, GKE, Compute Engine L4 and T4 | Step 2 |
| Cloud TPU | v5e, v6e and v5p through the TPU API, Compute Engine and GKE | Step 3 |
| Google Colab | a TPU v5e-1 notebook runtime | Step 4 |
| AWS | Bedrock, SageMaker, EC2 g6, g5g and g4dn | Step 5 |
| AWS Neuron | Inferentia2 and Trainium | Step 6 |
| Microsoft Azure | Foundry; quota for no L4- or T4-class GPU | Step 7 |
| AMD | one Instinct MI300X on AMD Developer Cloud | Step 8 |
| Local GPUs | GTX 1650 Ti, RTX 3050 | Step 9 |
| CPUs and Arm laptops | Intel laptops, Snapdragon 850, Snapdragon 7c Gen 2 | Step 10 |
| Cross-cloud | one agent and one grader over three clouds | Step 11 |

The model is Gemma 4 E2B unless a section says otherwise. The larger sizes, E4B, 12B, 26B-A4B and 31B, appear where a platform serves them.

---

#### Step 1 — Hugging Face: The Layer Every Platform Pulls From

Every self-hosted platform in this article starts with a download from Hugging Face. I use it as the registry and distribution layer: none of the serving here runs on Hugging Face's own Inference Endpoints, Spaces or Inference Providers.

What gets published under `xbill9` is the precision trade made concrete. Google ships QAT checkpoints whose weights already sit on a 4-bit grid. Google's own `-w4a16-ct` export re-rounds them with a step of max/7.5, which matches the trained step in 0.0% of groups and adds a flat 6.67% error. A repack that keeps the trained grid scores higher:

| Model | Google `-w4a16-ct` | Grid-preserving repack |
|---|---:|---:|
| E2B | 0.655 | 0.678 |
| E4B | 0.716 | 0.729 |
| 12B | 0.752 | 0.758 |

The same work on llama.cpp's format gives an exact Q4_0 E2B GGUF of 2,640,154,592 bytes against Google's 3,349,516,256, with mean KL divergence from the QAT model falling from 0.054303 to 0.001753 and the same top token rising from 87.18% to 98.06%.

Three properties of the Hub decide time to ready on every other platform:

- **Exports carry quirks.** Google's QAT exports omit `k_proj`, `v_proj` and `k_norm` for the 20 KV-shared layers, and every `-w4a16-ct` ships an `lm_head` byte-identical to `embed_tokens`, 805,306,368 bytes on E2B.
- **Pull time is part of start-up.** The 62 GB Gemma 4 31B checkpoint moved one rig's boot budget from 20 minutes to 90. Restoring the same files from an in-region Cloud Storage cache took 337 s.
- **Library versions are part of the contract.** transformers 5.15 made Gemma 4's `head_dim` per layer, which breaks older vLLM builds, and EmbeddingGemma 2 needs transformers 5.19.

The rigs pin both the revision and the SHA-256 of the file. The `draft` target in each local rig's Makefile fetches the drafter at a pinned revision and checks its hash:

```make
draft:
	@mkdir -p $(dir $(SPEC_DRAFT_MODEL))
	@echo "$(SPEC_DRAFT_SHA256)  $(SPEC_DRAFT_MODEL)" | sha256sum -c --quiet - 2>/dev/null || \
	  curl -fL -o $(SPEC_DRAFT_MODEL).part https://huggingface.co/$(SPEC_DRAFT_HF)/resolve/$(SPEC_DRAFT_HF_REVISION)/$(notdir $(SPEC_DRAFT_MODEL))
	@test ! -f $(SPEC_DRAFT_MODEL).part || mv $(SPEC_DRAFT_MODEL).part $(SPEC_DRAFT_MODEL)
	echo "$(SPEC_DRAFT_SHA256)  $(SPEC_DRAFT_MODEL)" | sha256sum -c -
```

A later card-only commit to the repository cannot change the bytes a rig serves.

---

#### 🔎 Tip: Pin the Digest, Not the Tag

The moving `vllm/vllm-tpu:nightly` tag of 2026-10-02 failed at load on every Google QAT export, because it expects the KV-shared layers' tensors on every layer. The pinned digest with a small patch serves the same checkpoint at 200.9 tok/s. On AMD the newest vendor image cannot parse Gemma 4's config at all. Pin images by digest, pin weights by hash, and check a new image before switching to it.

---

#### Step 2 — Google Cloud: Managed APIs and L4 GPUs

Google Cloud sells the model six ways short of a TPU. Every L4 path below served the same int4 E2B repack on vLLM 0.30.0, and the TPU row is in Step 3. Prices are Billing Catalog list prices fetched on 2026-10-12:

| Path | What you run | tok/s, one request | $/hr | $ per 1M output tokens |
|---|---|---:|---:|---:|
| Cloud Run, L4 | a container that scales to zero | 119.3 | 1.0465 while up | 2.44 |
| GKE Autopilot, L4 | a Deployment | 104.7 | 0.9593 | 2.55 |
| Compute Engine g2-standard-4, L4 | a VM with vLLM in Docker | 94.3 | 0.7783 | 2.29 |
| Gemini API | `gemini-3.5-flash-lite`, per token | 93.3 | — | 2.50 |
| Vertex AI endpoint, L4 | your container behind Vertex | 90.2 | 0.919 | 2.83 |
| Vertex AI MaaS | Gemma 4 26B-A4B, per token | 14.2 | — | 0.60 |

Timed on the server, the three L4s decode within 2.3% of each other: 141.9 tok/s on Cloud Run, 139.9 on GKE and 143.1 on Compute Engine. The spread in the client column is the path each request takes.

Time to ready varies far more than speed:

- **Compute Engine**: 9.6 minutes from create to serving.
- **Cloud Run**: 6.6 minutes to deploy. A cold start took 361.8 s and 483.2 s with default settings, about 116 to 119 s of it in torch.compile. A warm Cloud Storage compile cache brought a start to 240.1 s at 138.6 tok/s. `--enforce-eager` starts in about 202 to 222 s and decodes at 17.2 to 17.8 tok/s.
- **GKE Autopilot**: 8.8 minutes for the cluster and 7.7 for the Deployment.
- **Vertex AI endpoint**: 18.9 minutes for the first model upload and 24.4 to deploy it.

Batching changes the cost per token more than the platform does:

| Path | 1 request | 4 | 16 | $ per 1M at 16 |
|---|---:|---:|---:|---:|
| Compute Engine L4 | 92.2 | 304.1 | 859.6 | 0.252 |
| GKE Autopilot L4 | 88.8 | 385.0 | 1,310.0 | 0.203 |

A Compute Engine L4 running all month costs $568. Gemini Flash-Lite charges the same for 86 output tokens a second around the clock, with no input tokens.

The gotchas were all specific to a path. Cloud Run ships NVIDIA driver 580, which needs `VLLM_ENABLE_CUDA_COMPATIBILITY=0`. A Vertex shared endpoint has a 60 s timeout and no streaming, so the agent runs with `stream=False`. MaaS answered one benchmark request in five with a 429.

The precision trade shows most clearly on a Compute Engine T4, one Turing generation older than the L4, at 512 tokens in and 128 out:

| E2B build on one T4 | Weights | KV cache, tokens | tok/s, 1 request | tok/s, 8 |
|---|---:|---:|---:|---:|
| bf16 | 9.8 GiB | 315,974 | 37.04 | 164.62 |
| Google QAT w4a16 | 8.02 GiB | 519,568 | 62.27 | 215.91 |
| int4 repack with int4 tables | 2.86 GiB | 1,099,587 | 85.28 | 239.64 |

The T4 has no bf16 and no int4 math. The repack wins on bandwidth alone: fewer bytes per token, and the memory it frees becomes KV cache. Gemma 4's 512-wide attention heads need 98,304 bytes of shared memory in vLLM's Triton kernel against Turing's 65,536, so every Turing rig carries a patch that halves the tile size.

---

#### 🔎 Tip: Quota Is Permission to Ask

Quota does not reserve a machine. The project holds quota for 16 L4s in every US region, and Compute Engine still reported `STOCKOUT` across five US regions before an L4 launched in `northamerica-northeast1-b`. On the same day a TPU v6e was refused in every zone the project can use. A launch script that tries every zone, then the next region, is what makes a demo day work.

---

#### Step 3 — Cloud TPU: Where int8 Is the Only Low-Precision Win

Google's TPUs come in a scale-out "e" line and a scale-up "p" line. The two efficiency chips here:

| Spec | v5e | v6e (Trillium) |
|---|---:|---:|
| HBM | 16 GB | 32 GB |
| HBM bandwidth | 800 GiBps | 1,638 GBps |
| Peak bf16 | 197 TFLOPs | 918 TFLOPs |
| Peak int8 | 393 TOPs | 1,836 TOPs |
| fp8 in the matrix unit | no | no |

On v5e and v6e, int8 is the only low-precision format with a compute win: both quote int8 at exactly twice their bf16 peak. fp8 and int4 save memory and bandwidth and are widened to bf16 before the multiply. TPU7x (Ironwood) is the first with fp8 in the matrix unit, so these conclusions do not carry forward to it.

That shapes the precision trade on one v5e chip. Suite accuracy, then output tok/s at 1, 4 and 16 requests:

| E2B build on v5e-1 | Suite | 1 | 4 | 16 |
|---|---:|---:|---:|---:|
| bf16 | 0.683 | 144 | 560 | 2,008 |
| int8 W8A8 from QAT weights | 0.686 | 220 | 841 | 2,872 |
| int8 W8A8, int4 tables | 0.677 | 243 | 923 | 3,086 |
| fp8 | 0.670 | 182 | 702 | 2,453 |

int8 from the QAT weights is both the most accurate and about 1.5x bf16. Upstream `tpu_inference` has no int8 W8A8 method on its JAX path, so these rigs serve a pinned image with patches that add int8 W8A8, int4 embedding tables that stay packed on the device, and an int4 `lm_head`. int4 tables need padding to 128 columns: unpadded builds ran at 85 to 110 tok/s against 150 to 243 padded.

The largest model served on one v5e chip is the 12B, as int8 W8A8 with int4 tables: 11.31 GiB of weights, 57.0 / 215.7 / 713.4 tok/s at 1 / 4 / 16 requests, and 9,728 tokens of KV cache. An fp8 KV cache doubles that pool to 18,944 tokens for −0.7 suite points. On E2B with an older image the same flag gave exactly 1.000x the capacity, so a benchmark reads the KV allocation from the boot log.

**Scale out by replicas when the model has one KV head.** E2B has a single KV head, so tensor parallelism leaves attention unsplit: its attention kernel cost 803, 839 and 831 ms at TP 1, 2 and 4. On a v5e-4, four one-chip engines (`--data-parallel-size 4`) produced 1.39x the output of TP=4 at 16 requests and 3.36x at 256.

**Attention cost follows `--max-model-len`.** At 16 requests on one chip, output went from 1,387 tok/s at a 16,384 limit to 2,744 at 2,048.

How the chip is provisioned matters as much as the chip:

- **The Cloud TPU API is frozen** and TPU7x onward is Compute Engine or GKE only. v5e has no way off the TPU API: Compute Engine lists `ct5lp-hightpu-1t` and refuses to create it.
- **Serving is identical on both control planes.** A v6e-1 got exactly 1,151,744 KV tokens through each.
- **Flex-start is narrow.** v5e-1 flex-start is accepted only in `us-west4-a`, where it got a node in 0.4 minutes on 2026-10-12 and was ready 13.3 minutes later, 390.96 s of it XLA compilation.
- **GKE adds its own minutes.** A v6e-1 on GKE took about 28 minutes from nothing to the first token and served 199 / 1,180 / 1,508 tok/s at 1 / 8 / 32 requests.

A one-chip v5e at the $0.60 flex-start rate served one request at $1.58 per million output tokens and 16 at $0.138, the cheapest self-hosted figure in the Google Cloud run.

---

#### Step 4 — Google Colab: A TPU in a Notebook

Colab is the cheapest way onto a TPU. This account is eligible for a TPU v5e-1 and a T4, and is refused a v6e-1:

```
"eligibleGpus":["T4"],"ineligibleGpus":["H100","G4","A100","L4"],"eligibleTpus":["V5E1"],"ineligibleTpus":["V6E1"]
```

The runtime reports `JAX 0.7.2`, `1 device(s): TPU v5 lite` and `HBM 16.91 GB   host RAM 50.5 GB`. Notebooks are generated from `py:percent` sources and run headless with `google-colab-cli`, which wraps Colab's own assign, keep-alive and unassign calls.

The notebook compares three builds of E2B in a pure-JAX engine: Google's bf16 QAT checkpoint, Google's 4-bit export, and the grid-preserving repack from Step 1.

| Build | Disk | Decode tok/s | Mean KL from QAT | Same top token | Greedy tokens equal to QAT |
|---|---:|---:|---:|---:|---:|
| QAT, bf16 | 10.21 GB | 135.0 | — | — | 128 of 128 |
| Google 4-bit export | 8.32 GB | 93.7 | 0.07141 | 85.96% | 5 of 128 |
| Repack | 7.51 GB | 93.3 | 0.00021 | 99.29% | 128 of 128 |

The repack is 342.6x closer to the trained model at the same speed. On this engine the bf16 checkpoint is faster than both 4-bit builds, because the engine widens int4 to bf16 at every step: the precision trade with no native math under it. vLLM on a v5e-1 serves the two 4-bit builds at 136.6 and 136.5 tok/s, so the ratio between builds carries across engines and the absolute speed does not.

Two properties of the runtime shape every notebook. Host RAM is the limit before HBM: downloading 26 GB inside the kernel crashed the 50.5 GB VM, and moving downloads to a child process fixed it. And `colab new` must never be retried after a timeout, because a timed-out request can still assign a runtime that keeps billing compute units.

---

#### Step 5 — AWS: Bedrock, SageMaker and EC2 GPUs

AWS sells the model as an API, as a managed endpoint and as GPUs by the hour. One rehearsal on 2026-10-12 timed every path with the same 100-token prompt:

| Backend | What serves the model | tok/s, one request | $/hr | $ per 1M output tokens |
|---|---|---:|---:|---:|
| EC2 g6.xlarge, L4 | vLLM 0.30, int4 repack | 131.9 | 0.8048 | 1.69 |
| Bedrock | Amazon Nova Micro | 130.4 | — | 0.14 |
| SageMaker ml.g6.xlarge, L4 | AWS vLLM 0.31.0 container, same repack | 96.5 | 1.1267 | 3.24 |
| EC2 g5g.2xlarge, T4G | vLLM 0.27.2rc0, patched, fp16 | 34.9 | 0.556 | 4.43 |

**Bedrock** needs an IAM permission and an inference profile id, and bills per token: $0.035 per million input tokens and $0.14 per million output for Nova Micro. At temperature 0 it returned 9 different completions in 10 repeats, where every self-hosted backend returned one.

**SageMaker** runs the same vLLM container on the same L4 as the g6, behind three API calls: `create-model`, `create-endpoint-config` and `create-endpoint`. vLLM flags go in as `SM_VLLM_*` variables. It reached `InService` in about 8 to 9 minutes and adds about 0.24 s to each request. The repack decodes at 108.0 tok/s on it against 45.2 for the stock checkpoint, 2.39x, which is the precision trade on an L4 where int4 is widened.

**EC2 g6** was the fastest instance and the cheapest per token. Batching cut its cost 14x:

| Requests in parallel | g6 tok/s | g6 $ per 1M | SageMaker tok/s | SageMaker $ per 1M |
|---:|---:|---:|---:|---:|
| 1 | 104.8 | 2.132 | 85.3 | 3.667 |
| 4 | 410.9 | 0.544 | 328.7 | 0.952 |
| 16 | 1,485.9 | 0.150 | 1,077.2 | 0.291 |

A g6 running all month costs more than Nova Micro until it averages 710 output tokens a second around the clock, at five input tokens per output token.

**EC2 g5g** pairs a Graviton2 Arm host with an NVIDIA T4G. The arm64 vLLM image ships without Turing kernels, so it runs vLLM built from source for sm_75 with the shared-memory patch from Step 2. It took 23.7 minutes from launch to ready, most of it reading 9.54 GiB of weights with 11.19 GiB of free RAM. The same chip ran three engines on the same checkpoint:

| Engine on g5g, one request | Decode tok/s | Cold boot | Change-to-retest cycle |
|---|---:|---:|---:|
| vLLM 0.27.2rc0 | 32.53 | 1,417.8 s | about 67 minutes, a source rebuild |
| JAX | 12.69 | 242.2 s | 83 s |
| PyTorch + transformers | 10.24 | 195.2 s | 25 s |

vLLM is 3.18x PyTorch for one request, and 100.28 tok/s at eight against PyTorch's 10.08. The engine is the biggest lever on this chip, and the slowest engine to iterate on.

**EC2 g4dn**, a T4, served the stock checkpoint at 42.36 tok/s for one request and plateaued at 243.93 from 16.

---

#### 🔎 Tip: Debug With the Small Model

Every path in this article was brought up with E2B first. Flags, port forwards, tool parsing and drivers fail the same way at any size, and E2B fails faster: it is ready on Trainium in 121.2 s against 220.1 s for the 26B.

---

#### Step 6 — AWS Neuron: Inferentia2 and Trainium

AWS's own chips have the bandwidth and lack the software for this model. The Neuron rows run a hand-ported Gemma 4 image, the same digest on both chips:

| Backend | tok/s, one request | $/hr | $ per 1M output tokens |
|---|---:|---:|---:|
| inf2.xlarge, Inferentia2 | 36.5 | 0.7582 | 5.77 |
| trn1.2xlarge, Trainium | 36.0 | 1.3438 | 10.37 |

AWS's per-instance figures put Inferentia2 at about 817 GB/s per chip and Trainium at about 613, 2.0 to 2.7 times the L4's 300. They decode at 0.28 times the L4's speed. The reasons are all in the serving stack:

- **No vLLM release serves Gemma 4 on Inf2 or Trn1.** The newest vLLM-Neuron plugin supports Trn2 and Trn3 only, and Trn1 and Inf2 stop at vLLM 0.16, whose transformers has no `gemma4` model type. The hand port ignores tool definitions.
- **Everything compiles ahead of time for fixed shapes.** The image is built for 512 tokens in total and 128 of prompt. A new shape costs about 3.9 s to compile.
- **Memory is per core.** The 32 GB is two 16 GiB pools, one per NeuronCore. A single gather above roughly 4 to 5 GB returns zeros with no error, and the server then answers 200 with no tokens.
- **int4 is widened.** int4 weights get a dense bf16 copy before each multiply, 27 times slower per call than plain bf16.
- **The host needs swap.** E2B peaks at 19.53 GB of host memory, more than the 16 GiB on an inf2.xlarge.

Capacity was the other constraint. inf2 returned `InsufficientInstanceCapacity` in us-east-2a with 80 vCPUs of quota free, and on one run it was refused in five zones across two regions before launching in us-west-2a.

---

#### Step 7 — Microsoft Azure: Foundry, and No L4- or T4-Class GPU

Azure's limit was quota, before capacity was ever asked. The subscription holds no quota in the T4 or A100 VM families, and Container Apps refused a serverless T4 outright:

```
ContainerAppInvalidScaleExceedsQuota … maxium quota value of '0' for 'ManagedEnvironmentConsumptionT4Gpus'
```

**Foundry** is the Azure path that matches the rest of this article. It serves `gpt-4.1-nano` and `Phi-4-mini` as deployments on an existing account, reached through the OpenAI v1 endpoint:

| Model | tok/s, one request | $ per 1M output tokens | Notes |
|---|---:|---:|---|
| gpt-4.1-nano | 49.7 | 0.40 | 50K tokens per minute |
| Phi-4-mini | 35.4 | 0.30 | answered 12 of 20; 8 timed out at 60 s |

**The only GPU VM the subscription can rent is an A10, and it belongs to a different class.** The NVadsA10 v5 family holds 36 vCPUs of quota in eastus, and `Standard_NV36ads_A10_v5` is the only size with a whole A10: the smaller sizes get a sixth, a third or half of the GPU. A whole A10 therefore comes as a 36-vCPU, 440 GB machine at $3.20 an hour, against $0.8048 for the L4 in an EC2 g6.xlarge. It has no L4- or T4-class equivalent on Azure here, so its numbers stay in this section and out of the comparisons below.

What the A10 run shows is operational. The GPU is vGPU-backed, so it needs NVIDIA's GRID driver; the datacenter driver does not bind. The QAT checkpoint without int4 tables ran out of memory at `--gpu-memory-utilization 0.90`, where the 2.86 GiB int4 repack loaded in 22.46 s. Deleting the VM leaves its NIC, public IP and disk behind, so the teardown script deletes them by name.

---

#### Step 8 — AMD: One Instinct MI300X

AMD Developer Cloud rents one MI300X as a DigitalOcean GPU droplet at $1.99 an hour: 191.7 GiB of HBM3, 304 compute units, a 64-wide wavefront, and 3.76 TB/s of measured bandwidth. It is the first chip here where fp8 is a compute format, at twice the bf16 peak.

The platform has its own operations surface:

- **The console orders the card; the API cannot.** `POST /v2/droplets` with the devcloud size returns `422: This size is unavailable`.
- **A powered-off droplet keeps billing.** Only destroying it stops the meter.
- **A fresh droplet needs firmware.** `amdgpu` does not bind until `firmware-amd-graphics` is installed and the droplet rebooted, and `/dev/kfd` appears either way. `rocminfo` reporting a gfx agent is the reliable check.

**Newer images are not safer.** Gemma 4's sliding layers have 256-wide heads and its full-attention layers 512, and transformers 5.15 raises on a global `head_dim` read:

| Image | vLLM | transformers | Gemma 4 |
|---|---|---|---|
| `vllm/vllm-openai-rocm:nightly-rocm100` | 0.29.1rc1.dev187 | 5.17.0 | loads |
| `rocm/vllm`, newest AMD build | 0.27.1.dev5 | 5.16.1 | fails on `head_dim` |
| `rocm/vllm`, oldest | 0.19.1 | 5.8.1 | loads |

No ROCm image ships vLLM's audio extras, so audio is unreachable on this card.

**Memory is never the limit.** E2B's KV pool is 9,026,017 tokens, and the heaviest benchmark cell, 64 requests at 8,192 tokens of context, uses 5.8% of it. Long context flattens on prefill, so total throughput at 64 requests matched 16. At 128 tokens of context and 64 requests it produced 10,293.6 output tok/s, $0.054 per million.

The data-type sweep asked whether the precision trade pays on a card where everything fits. Output relative to bf16 across nine cells per size:

| Size | bf16 tok/s, one request | fp8 vs bf16 | fp8 cells faster | int4 W4A16 vs bf16 |
|---|---:|---|---:|---|
| E2B | 343 | 0.75–1.08x | 5 of 9 | 0.14–0.63x |
| E4B | 239 | 0.89–1.19x | 6 of 9 | 0.14–0.54x |
| 12B | 134 | 1.12–1.41x | 9 of 9 | 0.16–0.57x |
| 26B-A4B | 224 | 0.99–1.23x | 7 of 9 | 0.27–0.69x |
| 31B | 58 | 1.20–1.45x | 9 of 9 | 0.19–0.59x |

**fp8 overtakes bf16 from 12B up**, where the weights are large enough for the fp8 matrix cores to matter. **int4 lost everywhere.** The only ROCm W4A16 kernel for this card widens to bf16 before every multiply. A preregistration predicted the 31B int4 repack would cut time per output token by 1.5x to 3x. It measured 91.88 ms against bf16's 17.19.

Two more AMD results follow the same rule, that an accepted setting is not a faster kernel:

- **`VLLM_ROCM_USE_AITER=1` made the 12B fp8 slower in all nine cells**, 0.974x to 0.994x. Attention stays on Triton for Gemma 4's mixed head sizes, and the fp8 matrix multiplies moved to AITER kernels that logged 924 "not found tuned config" lines.
- **`vllm bench serve` seeds every prompt at 0 by default**, and prefix caching is on. At 8,192 tokens of context that measured the cache: 2.25x the throughput of unique prompts.

The card also served EmbeddingGemma 2 at 828.7 requests a second for 128-token inputs, and beside the 12B fp8 generator on the same card.

---

#### 🔎 Tip: Give Every Benchmark Cell a Unique Prompt Seed

A prefix cache that sees the same prompt twice answers it from memory. On the MI300X a default seed inflated long-context throughput 2.25x, and on a GTX 1650 Ti the llama.cpp prompt cache answered 656 of 661 tokens in a sweep. Seed every cell differently, or turn the cache off for the run.

---

#### Step 9 — Local GPUs: A Laptop 1650 Ti and an 8 GB RTX 3050

Local inference runs llama.cpp on the exact Q4_0 GGUF from Step 1. The two cards bracket a generation:

| Card | E2B decode tok/s | E2B prefill tok/s | VRAM used |
|---|---:|---:|---:|
| GTX 1650 Ti Max-Q, 4 GiB, Turing, no tensor cores | 81.59–81.92 | about 345 | 1,480 MiB |
| RTX 3050 OEM, 8 GiB, Ampere | 116.14 | 3,859.96 | 1,510 MiB |

The 3050 decodes about 1.42x faster and prefills 11x faster, and the prefill gap decides concurrency. llama.cpp prefills one request at a time on both cards. On the 1650 Ti that serial prefill fills the time, so 512-token prompts scale only 1.47x with more users. On the 3050 it is fast enough for decode batching to show:

| RTX 3050, E2B | 1 user | 64 users | Gain |
|---|---:|---:|---:|
| 512 tokens in, 128 out | 93.28 | 358.05 | 3.84x |
| 128 tokens in, 512 out | 104.74 | 618.12 | 5.90x |

**Speculative decoding** is the local trade that pays most. Google's QAT assistant drafters guess the next tokens and the full model checks them in one pass:

| Card and model | No drafter | With drafter | Draft length |
|---|---:|---:|---:|
| RTX 3050, E2B | — | 1.386x | 2 tokens |
| RTX 3050, E4B | 58.2 tok/s | 1.723x | 2 tokens |
| GTX 1650 Ti, E2B | 79.0 tok/s | 1.18x | 1 token |
| GTX 1650 Ti, E4B | 38.5 tok/s | 1.43x | 2 tokens |

The drafter's precision costs nothing. An exact Q4_0 drafter of 59.2 MB accepts the same 0.475 of drafted tokens as the 170.2 MB bf16 one, and runs at 151.1 tok/s against Q8_0's 150.1. The gain shrinks with context, E2B from 1.51x with nothing ahead of the question to 1.21x at 120k tokens, while acceptance barely moves: verification gets more expensive, the guesses do not get worse.

Greedy text with a drafter can differ from greedy text without one, and the cause is prompt length: E4B matched with no text ahead and differed from about 2,048 tokens of prompt on. The prompt cache changes greedy text by itself. A benchmark that needs byte-identical output turns off both.

On quality, E4B beats E2B on the 3,880-record suite, 0.730 against 0.683, and on GSM8K, 0.933 against 0.898. Each build matches its TPU reference, so the gap is the model and the quantization costs nothing measurable.

---

#### Step 10 — CPUs and Arm Laptops

The CPU is where the bandwidth half of the trade is most visible. On an i7-10750H, llama.cpp serves E2B at 16.5 to 18.9 tok/s end to end. PyTorch with transformers serves bf16 at 4.85. vLLM's CPU backend cannot serve at all on that host: the w4a16 build needs 10.46 GB against 9.25 GB available. On the same laptop the 1650 Ti decodes 4.37x faster than the CPU.

Arm laptops add a fourth variable, the instruction set:

| Laptop | Cores used | E2B decode tok/s | Prefill tok/s | Note |
|---|---|---:|---:|---|
| Snapdragon 850, Yoga C630 | 4 Gold | 5.00 | 9.22 on all 8 | no dot product |
| Snapdragon 850, Yoga C630 | all 8 | 2.18 | — | Silver cores are stragglers |
| Snapdragon 7c Gen 2, Galaxy Book Go | 2 Gold | 8.33 | 46.68 on all 8 | dot product; every cell swapped |
| Intel i7-1360P | 4 threads | 22.60 | 106–119 | |

The 7c Gen 2 has the Arm dot-product instructions, so llama.cpp repacks Q4_0 for them and two cores outrun the Snapdragon 850's four. The repack costs 1.44 GB of memory against 192 MB on the 850, on a laptop with 3.4 GiB, so memory is that host's ceiling. On both Snapdragons decode wants only the fast cores and prefill wants all of them.

---

#### Step 11 — Cross-Cloud: One Agent, One Grader

The cloud sections share a method. Three repositories, one per cloud, use the same Strands agent, prompts and grader, and a fourth loads them all into one agent. Every GPU row serves the same int4 repack with vLLM's Gemma 4 tool parser.

The agent asks how many of eleven ids are 10 or more and has a `count_ids` tool that computes it. The grader checks the number and the filter sent, because a wrong filter returns an exact number with a clean source:

```
backend          engine  filter  med s    rows  filters sent
aws.g6            20/20   20/20    0.4   20/20  id >= 10
gcp.cloudrun      20/20   20/20   0.52   20/20  id >= 10
gcp.tpu           20/20   20/20   1.65   20/20  id >= 10
gcp.maas          19/20   19/20   13.0    0/20  id >= 10
azure.foundry     20/20   20/20   3.19   19/20  id >= 10
```

With the engine counting, every backend except MaaS was right 20 times in 20 with the right filter. When the model counted the rows itself, MaaS was right 0 times in 20. Counting belongs in code, and the grade belongs on the filter.

---

#### Compare and Contrast

One request at a time, the same 100-token prompt, list prices on the day:

| Backend | Hardware | tok/s | $ per 1M output tokens |
|---|---|---:|---:|
| 🥇 AWS EC2 g6 | L4 | 132.45 | 1.69 |
| 🥈 Cloud Run | L4 | 119.3 | 2.44 |
| 🥉 AWS Bedrock | Nova Micro | 115.85 | 0.14 |
| Cloud TPU | v5e-1, flex-start | 105.3 | 1.58 |
| AWS SageMaker | L4 | 105.2 | 2.98 |
| Compute Engine | L4 | 94.3 | 2.29 |
| AWS EC2 inf2 | Inferentia2 | 36.6 | 5.75 |
| AWS EC2 g5g | T4G | 35.8 | 4.31 |

How each platform answers the precision trade:

| Platform | Native low-precision compute | What paid | Biggest trap |
|---|---|---|---|
| Hugging Face | — | grid-preserving repacks | lossy official exports |
| Google Cloud L4 | fp8, int8 | int4 repack for bandwidth | stockouts with quota |
| Cloud TPU v5e, v6e | int8 only | int8 W8A8, 1.5x bf16 | moving nightly images |
| Colab TPU | int8 only | repack accuracy | orphaned runtimes |
| AWS L4 | fp8, int8 | int4 repack, 2.39x stock | SageMaker per-request overhead |
| AWS Neuron | int8 | — | no vLLM for Gemma 4 |
| Azure | — | Foundry per token | quota of 0 for T4 |
| AMD MI300X | fp8 | fp8 from 12B up | newest image cannot load |
| Local GPUs | int8 | Q4_0 plus a drafter | prompt cache in benchmarks |
| CPUs and Arm | dot product where present | Q4_0 repack | memory on small hosts |

---

#### So, Which One?

Start with a per-token API when one of its models does the job: Bedrock's Nova Micro was the cheapest per token in this article and costs nothing idle. Move to vLLM on an L4 for your own weights, on whichever cloud has capacity that day, and quantize from a QAT checkpoint so the smaller format keeps its accuracy. On Cloud TPU, serve int8 W8A8 and scale E2B with replicas. On an MI300X, serve bf16 below 12B and fp8 from 12B up. Neuron pays off once the serving stack supports the model. Locally, an 8 GB RTX 3050 with an exact Q4_0 model and drafter serves one user at over 100 tokens a second.

---

#### Summary

The goal of this article was to cover inference engineering across every platform I have served Gemma 4 on, with each platform in depth before comparing them. The key to the solution was one trade-off applied everywhere: what fewer bits save in memory, what they cost in quality, and whether the chip multiplies them natively. The results were:

- 🟢 **Formats built from QAT weights keep their accuracy**: int8 at 0.686 against bf16's 0.683, and the grid-preserving int4 repack 342.6x closer to the trained model than Google's 4-bit export.
- 🟢 **Fewer bits pay through bandwidth even with no native math**: 2.39x on a SageMaker L4 and 2.30x on a T4 for the int4 repack.
- 🟢 **Native math pays where it exists**: int8 at about 1.5x bf16 on a TPU v5e, fp8 ahead of bf16 in every cell from 12B up on an MI300X.
- 🟢 **Batching cuts cost per token by an order of magnitude**: $2.132 to $0.150 per million on an EC2 g6 from 1 to 16 requests.
- ⚠️ **Bandwidth sets a ceiling the engine has to reach**: Neuron has 2.0 to 2.7 times the L4's bandwidth and decodes at 0.28 times its speed.
- ⚠️ **Quota is permission to ask**: Google Cloud, AWS and Azure each refused hardware on the day, in different ways.
- ❌ **int4 lost to bf16 on every MI300X cell**, because the only ROCm kernel widens it before every multiply.

Scope: each platform was measured on its own days between July and October 2026, mostly on Gemma 4 E2B. The cross-cloud tables come from 2026-10-12, with AWS from 2026-10-05, and their per-token rows serve each cloud's own model, so those rows compare paths. vLLM versions differ between platforms (0.27 to 0.31, plus patched TPU and Turing builds and a hand-ported Neuron engine), local results come from llama.cpp builds at different commits, and Snapdragon and cross-card ratios are indicative. Accuracy figures come from one TPU v5e or one RTX 3050 per build, and no accuracy benchmark ran on the MI300X. Prices are on-demand or flex-start list prices on the day each run was made.

The strategy for engineering Gemma 4 inference across Hugging Face, Google Cloud, Cloud TPU, Colab, AWS, Azure, AMD and local hardware was validated with an incremental step by step approach.

---

#### References

* [gemma4-dev | GitHub](https://github.com/xbill9/gemma4-dev)
* [aws-inference-strands | GitHub](https://github.com/xbill9/aws-inference-strands)
* [Inference on AWS slides | Google Slides](https://docs.google.com/presentation/d/1Ax8KGF8T0EYffshY7WfUM61QcHmLk0q72EE7G_hIwgM/edit)
* [gcp-inference | GitHub](https://github.com/xbill9/gcp-inference)
* [azure-inference | GitHub](https://github.com/xbill9/azure-inference)
* [cross-cloud-inference | GitHub](https://github.com/xbill9/cross-cloud-inference)
* [colab-jax | GitHub](https://github.com/xbill9/colab-jax)
* [vLLM | GitHub](https://github.com/vllm-project/vllm)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
* [Strands Agents | GitHub](https://github.com/strands-agents/sdk-python)
* [Cloud TPU v6e | Google Cloud](https://cloud.google.com/tpu/docs/v6e)
* [Amazon EC2 Inf2 instances | AWS](https://aws.amazon.com/ec2/instance-types/inf2/)
* [AMD Instinct MI300X | AMD](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
* [NVIDIA L4 | NVIDIA](https://www.nvidia.com/en-us/data-center/l4/)
