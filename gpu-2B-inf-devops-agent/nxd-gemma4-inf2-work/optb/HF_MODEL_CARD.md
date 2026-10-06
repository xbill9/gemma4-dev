---
license: apache-2.0
base_model: google/gemma-4-E2B-it
tags:
  - gemma
  - gemma-4
  - aws-inferentia
  - inferentia2
  - neuron
  - torch-neuronx
  - text-generation
library_name: torch-neuronx
pipeline_tag: text-generation
---

# Gemma-4-E2B-it on AWS Inferentia2 (Option B / torch_neuronx)

Compiled **AWS Neuron** artifacts + a self-contained server that run
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) **coherently and
fast on a single AWS Inferentia2 device** (`inf2.8xlarge`), at **~44 tokens/sec**.

This is a **runnable inference port**, not a fine-tune — the weights are Google's,
unmodified. What's new is the *compilation recipe*: a way to express Gemma-4's
**cross-layer KV-sharing** on Neuron, which AWS's own `NxD` / `optimum-neuron` stack
cannot currently do (there is no Gemma-4 in `optimum-neuron`, and NxD can't represent the
KV-sharing graph).

## TL;DR

| | |
|---|---|
| Base model | `google/gemma-4-E2B-it` (~5B params, 2B *effective* via MatFormer + Per-Layer Embeddings), Apache-2.0 |
| Hardware | AWS Inferentia2 — `inf2.8xlarge`, one logical NeuronCore (cores 0–1) |
| Precision | bf16 (fp32 neffs overflow the 16 GB core) |
| Throughput | **~44 tok/s** (~23 ms/token), measured |
| Context | `max_total_tokens=512`, `max_prompt_tokens=128` (baked into the neffs) |
| Cold start | ~100 s to load both neffs onto the core, then instant |
| Output parity | Greedy decode is **token-for-token identical to the CPU reference** |

## Why this exists

Gemma-4-E2B shares Key/Value projections across groups of layers. On TPU that's a free
graph view; in AWS's NxD framework it can't be expressed, so the vendor path either
refuses the model or emits gibberish. **Option B** sidesteps NxD entirely: it
`torch_neuronx.trace()`s the Hugging Face `transformers` (5.13) Gemma-4 **text forward
pass** directly, so KV-sharing traces as ordinary live graph dependencies — exactly like
it does on TPU/XLA.

Key ingredients that make the trace compile and match the reference:

- **eager attention** (not the fused/SDPA path)
- **host-side PLE (Per-Layer Embedding) lookup** — the 262144×8960 embedding table is kept
  off-device and gathered on CPU, fed in as activations (it otherwise trips the compiler and
  blows the 16 GB core)
- plain `DynamicCache` at trace time; a fixed static KV buffer at decode time
- elementary **tanh-GELU** and attention **softcap = 30** written out explicitly
- language model + head registered as real submodules so they compile into the graph

## Two-graph KV-cache design

Inference is split into two compiled graphs sharing one static KV buffer:

- **Prefill** (`kv_pre_512.pt`) — consumes a padded prompt (≤ `KV_BUCKET` = 128 tokens),
  returns the 15 non-shared layers' K/V (shared layers never write cache).
- **Decode** (`kv_dec_512.pt`) — single-token forward against a fixed `KV_MAX` = 512-length
  KV buffer. Each step writes the new K/V via a one-hot masked scatter
  (`buf*(1-oh)+k*oh`, pure arithmetic → trace-safe); KV tensors are graph inputs/outputs.

`KV_MAX` = maximum total tokens (buffer length, baked into both neffs).
`KV_BUCKET` = maximum prompt tokens (baked into the prefill neff).

## Files

| File | What it is |
|---|---|
| `kv_pre_512.pt` | Prefill neff (TorchScript, bf16) |
| `kv_dec_512.pt` | Decode neff (TorchScript, bf16) |
| `optb_kv.py` | Builds/compiles both neffs (`cpu` = reference check, `trace` = compile) |
| `optb_server.py` | Stdlib-only HTTP server (full model on host); loads both neffs once, then serves |
| `optb_server_slim.py` | Low-RAM server — loads only the embedding/PLE tables on the host (bf16, ~6 GB) so it fits **inf2.xlarge** (16 GB). Same API. |
| `optb_gen.py` | Minimal standalone greedy-generation example |
| `Dockerfile` / `Dockerfile.slim` | Reproducible runtime images (full / slim) |

The neffs embed the base weights in bf16, so they are Apache-2.0 derivatives of
`google/gemma-4-E2B-it` — see **License** below.

## Run the prebuilt Docker image (fastest)

A ready-to-run image with the neffs + Neuron runtime + server baked in is published on
Docker Hub — no compilation or Python setup needed:

```bash
docker pull xbill9/gemma4-optb:latest
# on an AWS inf2 instance:
docker run --rm -p 8080:8080 --device=/dev/neuron0 xbill9/gemma4-optb:latest
# then: curl -s localhost:8080/health
```

Image tags on **`docker.io/xbill9/gemma4-optb`** (~16 GB each, Apache-2.0):
- **`latest`** / `512-128` — full server, for **inf2.8xlarge** (128 GB host RAM), **~44 tok/s**.
- **`slim`** — low-RAM server for **inf2.xlarge** (16 GB host RAM), **~24 tok/s**.

### inf2.xlarge (the cheap box) needs swap ⚠️
The slim server fits 16 GB *serving-time* (~3.6 GB), but loading the two 3.4 GB neffs briefly peaks
at **~14.5 GB**. On a host with **no swap** (e.g. a stock Neuron DLAMI) that OOM-kills the container.
Add swap **before** running:

```bash
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
docker run --rm -p 8080:8080 --device=/dev/neuron0 xbill9/gemma4-optb:slim
```

Both images serve the same OpenAI-compatible routes below, plus a Prometheus **`/metrics`** endpoint
(requests, tokens, tokens/sec, errors, resident memory).

## Run from these files

**Requirements:** an AWS `inf2` instance with the Neuron runtime, plus:

```
transformers==5.13.0
torch-neuronx==2.8.0.2.12.22436
neuronx-cc==2.23.6484.0
libneuronxla==2.2.15515.0
```

**Serve (recommended):**

```bash
export KV_MAX=512 KV_BUCKET=128 \
       KV_PRE_OUT=./kv_pre_512.pt KV_DEC_OUT=./kv_dec_512.pt PORT=8080
python optb_server.py           # ~100 s warmup, then serves on :8080
```

The server exposes OpenAI-compatible routes (no auth — see Limitations):

```bash
curl -s localhost:8080/health
curl -s -X POST localhost:8080/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"What is AWS Inferentia?","max_tokens":64}'
# also: /v1/chat/completions, /v1/completions, /v1/models
```

**Recompile for a different context window** (the sizes are baked into the neffs):

```bash
KV_MAX=1024 KV_BUCKET=256 \
KV_PRE_OUT=./kv_pre_1024.pt KV_DEC_OUT=./kv_dec_1024.pt \
python optb_kv.py trace
```

Compilation runs on CPU (`neuronx-cc`) and does **not** need a NeuronCore, so you can
recompile on any box. Larger buffers cost more device memory — fp32 neffs already exceed
the 16 GB core, which is why these ship as bf16.

## Benchmarks

- **~44 tok/s** sustained greedy decode (~23 ms/token), single logical NeuronCore.
- Prefill ~0.06 s after warmup; one-time neff load ~100 s (load once, serve many).
- End-to-end examples (e.g. *"What is Gemma?"* → coherent ~115-token answer) are fluent
  and on-topic.
- **Correctness:** device greedy argmax matches the CPU float reference token-for-token
  (validated on multi-sentence generations, e.g. *"The capital of France is **Paris**."*).

## Limitations

- **Batch size 1**, single-stream greedy/sampled decode. No continuous batching / paged
  attention.
- Context is fixed at compile time (512 total / 128 prompt here). Longer needs a recompile.
- Server ships with **no authentication** — put it behind your own gateway before exposing
  it beyond a trusted network.
- Compiled specifically for Inferentia2 + the Neuron SDK versions pinned above.

## License & attribution

- **Base model:** `google/gemma-4-E2B-it`, © Google, **Apache-2.0**.
- **These artifacts** (compiled neffs + scripts) are Apache-2.0 derivatives; the neffs
  contain the base weights in bf16. Redistributed under Apache-2.0 with attribution to
  Google — include the upstream `LICENSE`/`NOTICE` when redistributing.
- The Option B recipe, two-graph KV-cache implementation, and server are released under
  Apache-2.0.

*Not affiliated with or endorsed by Google or AWS. Provided for research/testing.*
