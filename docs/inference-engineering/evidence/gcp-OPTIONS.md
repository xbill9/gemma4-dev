# Serving an LLM on Google Cloud: every option, hosted to self-run

Each option below has the facts that decide whether it fits, each with the Google page it came from (read 2026-10-12), and what this repo measured on it. "Measured" points into `results/`; anything without a source or a measurement is marked untested.

**Naming.** Vertex AI is now **Gemini Enterprise Agent Platform**: Model Garden, Model as a Service (MaaS), online inference and Agent Engine (now Agent Runtime) all moved under that name. The API host `aiplatform.googleapis.com` and the `gcloud ai` commands are unchanged, so this repo still says Vertex AI where it names a command. [Name changes](https://docs.cloud.google.com/gemini-enterprise-agent-platform/vertex-ai-name-changes)

---

#### The map

| Option | You run | Billed | Tool calls | Measured here |
|---|---|---|---|---|
| Gemini on Agent Platform | nothing | per token | yes | `gemini` |
| Open models as a service (MaaS) | nothing | per token | yes | `maas`, Gemma 4 26B A4B |
| Agent Platform endpoint, your container | a model upload and an endpoint | per node hour | when vLLM has them on | `vertex`, L4 |
| Cloud Run with a GPU | a container image | per second while an instance exists | when vLLM has them on | `cloudrun`, L4 |
| GKE Autopilot with a GPU | a Deployment | node hour plus premium, plus cluster fee | when vLLM has them on | `gke`, L4 |
| Compute Engine GPU VM | the VM and everything on it | per hour | when vLLM has them on | `gce`, L4 |
| Cloud TPU | the TPU VM and everything on it | per chip hour | yes, vLLM TPU | `tpu`, v5e-1 |

---

#### 1 — Gemini on Agent Platform

- Five ways to pay: Provisioned Throughput, Standard, Priority, Flex and Batch. Flex and Batch are half the Standard rate; Priority costs more for steadier latency. [Consumption options](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deploy/consumption-options)
- Standard has no fixed quota: the throughput ceiling rises with 30-day spend, from 2,000,000 tokens per minute for Flash and Flash-Lite at the lowest tier. A 429 means the shared pool is busy; back off and retry. [Standard PayGo](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/standard-paygo)
- The `global` endpoint has the most headroom; `us` and `eu` keep processing in that jurisdiction; single regions have the least. [Standard PayGo](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/standard-paygo)
- Flex is Preview, global only, with request timeouts up to 30 minutes. [Flex PayGo](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/flex-paygo)
- Gemini 3.5 Flash-Lite list prices in the Cloud Billing catalog, global, Standard: USD 0.30 per million input tokens and 2.50 per million output tokens (`results/2026-10-12/prices.json`).
- Strands: `GeminiModel` with `client_args={"vertexai": True, ...}`.
- **Measured:** `results/2026-10-12/demo1-*-gemini.txt`, `bench-gemini.txt`.

#### 2 — Open models as a service (MaaS)

- Serverless open models on the same API, billed per token: Gemma 4 26B A4B, gpt-oss 120B and 20B among them. [Use open models](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/use-open-models)
- Gemma 4 26B A4B is **Experimental**, on the `global` endpoint only, with function calling, no thinking mode and no batch. [Model card](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/google/gemma-4-26b-a4b-it)
- OpenAI-compatible chat completions at `.../locations/global/endpoints/openapi/chat/completions`, model `google/gemma-4-26b-a4b-it-maas`, with a Google access token as the API key. [Call open model APIs](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/call-open-model-apis)
- List price in the billing catalog: USD 0.15 per million input tokens and 0.60 per million output tokens (`prices.json`).
- **Measured:** five tool-call requests five seconds apart before the run got two answers, two `429 The request queue is full.` and one 60 s client timeout; one demo-1 run took 258 s while the OpenAI client retried. That message appears in no Google doc; the documented 429 fixes are backoff, the global endpoint, gradual ramp-up, or Provisioned Throughput. [429 errors](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deploy/error-code-429)

#### 3 — Agent Platform endpoint with your container

- Four ways to serve an open model here: MaaS, one-click deploy from Model Garden, Google's prebuilt serving containers, or your own container. [Choose a serving option](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/choose-serving-option)
- Model Garden's one-click deploy for Gemma 4 E2B offers `g4-standard-48` (RTX PRO 6000) or `a3-highgpu-1g` (H100) only (`gcloud ai model-garden models list-deployment-config --model=google/gemma4@gemma-4-e2b-it`). An L4 needs your own container.
- Images must come from Artifact Registry in the model's region. A remote repository that proxies Docker Hub works: this repo serves `vllm/vllm-openai:v0.30.0` through one. [Custom container requirements](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/custom-container-requirements)
- Shared public endpoints: 60 s request timeout, 1.5 MB payloads, no streaming. Dedicated endpoints: 1 hour, 10 MB, streaming, and the OpenAI-style `invoke` routes. [Endpoint types](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/choose-endpoint-type)
- This repo uses a shared endpoint with the predict route set to vLLM's `/v1/chat/completions`, called through `:rawPredict`; `backends.py` rewrites the OpenAI client's path onto it.
- List price, us-central1, g2-standard-4 with one L4: node plus management fee, computed in `cost.py`.
- **Measured:** the first model upload took 19 minutes while the proxy copied the image; a second upload of the same image took 2.5 minutes. Deploy-model to serving took 24.4 minutes (`results/2026-10-12/launch-to-ready.txt`). Strands' OpenAI client streams by default; on a shared endpoint it needs `stream=False`.

#### 4 — Cloud Run with a GPU

- L4 (24 GB) or RTX PRO 6000 (96 GB), one per instance. An L4 needs at least 4 CPU and 16 GiB. Driver 580.x, CUDA 13.0. [GPU services](https://docs.cloud.google.com/run/docs/configuring/services/gpu)
- Scales to zero. Billing is per instance lifetime, with nothing charged while no instance exists. Google quotes about 5 seconds to start an instance before the container and model load. [GPU services](https://docs.cloud.google.com/run/docs/configuring/services/gpu)
- Zonal redundancy is on by default and costs more per GPU-second; off is cheaper, and its quota is more likely to be granted. The first deployment gets 3 L4s per region with redundancy off. [GPU services](https://docs.cloud.google.com/run/docs/configuring/services/gpu)
- Set Cloud Run concurrency to vLLM's `--max-num-seqs`: requests queued inside vLLM do not trigger scale-out. [Run Gemma on Cloud Run](https://docs.cloud.google.com/run/docs/run-gemma-on-cloud-run)
- Weights load faster from Cloud Storage than from Hugging Face; Google recommends a Cloud Storage volume or a parallel copy. [GPU best practices](https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices)
- Private by default here: `--no-allow-unauthenticated`, called with `gcloud auth print-identity-token`.
- **Measured:** deploy to serving 6.6 minutes; from zero, the waking request was answered after 361.8 s and 483.2 s with default settings, 240.1 s with warm caches on a Cloud Storage volume (`results/2026-10-12/launch-to-ready.txt`, `coldstart.txt`, `coldstart-all.txt`). The deploy uses `VLLM_ENABLE_CUDA_COMPATIBILITY=0`: with the image's forward-compatibility libraries switched on, vLLM fails on Cloud Run's 580 driver with CUDA error 803.

#### 5 — GKE Autopilot with a GPU

- GPU Pods are billed per node, plus an Autopilot accelerator premium per vCPU, GiB and GPU, plus the cluster fee. [Autopilot GPUs](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/autopilot-gpus)
- **Measured:** a Pod asking for 4 vCPU, 16 GiB and one L4 got a `g2-standard-8` node, twice the vCPU and memory asked for (`results/2026-10-12/backends-where.txt`). `cost.py` prices that node.
- Driver: `default` or `latest`, by node selector. The `default` driver measured 580.173.02 here.
- GKE Inference Gateway adds prefix-cache-aware and model-aware routing, LoRA serving and autoscaling on server metrics, at no extra cost. [About Inference Gateway](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway) Untested here.
- GKE Inference Quickstart lists Google-benchmarked model, server and accelerator profiles and writes the manifests: `gcloud container ai profiles list`. [Inference Quickstart](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/inference/inference-quickstart) Untested here.
- This project's `default` network is a legacy network, which GKE cannot use; the cluster runs on the auto-mode `vpc-glitnir` network.

#### 6 — Compute Engine GPU VM

- G2 (L4) and G4 (RTX PRO 6000) are Google's picks for real-time serving; A3 (H100) and up for larger models. [Accelerator-optimized machines](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines)
- `g2-standard-4`: 4 vCPU, 16 GB, one L4 with 24 GB at 300 GBps. [GPUs](https://docs.cloud.google.com/compute/docs/gpus)
- Spot (up to 91% off, preemptible), flex-start VMs (up to 7 days, waits for capacity) and calendar-mode reservations change price and availability. [Spot](https://docs.cloud.google.com/compute/docs/instances/spot), [Flex-start VMs](https://docs.cloud.google.com/compute/docs/instances/about-flex-start-vms), [Calendar mode](https://docs.cloud.google.com/compute/docs/instances/future-reservations-calendar-mode-overview)
- The `common-cu129-ubuntu-2204-nvidia-580` Deep Learning VM image booted on `g2-standard-4` with driver 580.178.04 and the NVIDIA container toolkit installed, and Docker missing. Google's machine page says Deep Learning VM images cannot boot G2; on 2026-10-12 this one did.
- No public port: the VM is reached through an IAP TCP tunnel.
- **Measured:** 13 zones in 5 regions refused `g2-standard-4` with `STOCKOUT` before `northamerica-northeast1-b` launched one, with 16 L4s of quota in every US region (`results/2026-10-12/capacity.txt`).

#### 7 — Cloud TPU

- The Cloud TPU API is no longer under active development; TPU7x (Ironwood) and later are Compute Engine or GKE only. Compute Engine supports v5p, v6e and TPU7x, so v5e stays on the TPU API or GKE. [TPUs in Compute Engine](https://docs.cloud.google.com/compute/docs/tpus/tpu-resources-in-compute-engine)
- v6e: 32 GB HBM at 1638 GBps; bf16 918 TFLOPs, int8 1836 TOPs; fp8 at the bf16 rate. TPU7x: 192 GB at 7380 GBps, fp8 at twice bf16. [v6e](https://docs.cloud.google.com/tpu/docs/v6e), [TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x)
- vLLM runs on TPU through the `tpu-inference` plugin, tool calls included. Its JAX path serves int4-weight (`w4a16`) checkpoints for dense layers since 2026-09-27; this repo serves the bf16 checkpoint on TPU. [tpu-inference](https://github.com/vllm-project/tpu-inference)
- Flex-start (DWS) draws on a separate pool, stops by itself, and costs less. [DWS on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/dws)
- **Measured:** v6e-1 was refused in every zone this project can use, on-demand and Spot, and a v6e flex-start request stayed queued. A v5e-1 flex-start request in us-west4-a got a node in 22 seconds; vLLM TPU 0.30.0 served the bf16 checkpoint with tool calls, 20 of 20, at 144.1 tok/s server-side (`results/2026-10-12/capacity.txt`, `server-side-tpu.txt`).

#### 8 — Also on the list, untested here

- Agent Platform batch inference for Gemini, half price. [Batch inference](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference)
- The Gemini Developer API (AI Studio key) with the same Standard, Flex, Batch and Priority tiers. [Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Cloud Run with Ollama. [Tutorial](https://docs.cloud.google.com/run/docs/tutorials/gpu-gemma-with-ollama)
- Cloud Run jobs with a GPU, for offline inference. [GPU jobs](https://docs.cloud.google.com/run/docs/configuring/jobs/gpu)
- Agent Platform online inference on TPU (v5e, v6e, TPU7x, single host). [Use TPUs](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/use-tpu)
- Google's Gemma 4 deployment list. [Gemma on Google Cloud](https://ai.google.dev/gemma/docs/integrations/google-cloud)
