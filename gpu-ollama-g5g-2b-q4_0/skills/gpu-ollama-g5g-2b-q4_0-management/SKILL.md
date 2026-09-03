---
name: gpu-ollama-g5g-2b-q4_0-management
description: Manage AWS EC2 G5g capacity (Graviton2 + NVIDIA T4G) and Gemma 4 E2B Q4_0 serving under Ollama. Use when the user asks about provisioning, launching, listing, or terminating G5g instances, installing or debugging Ollama on T4G / Turing / aarch64, serving a GGUF or QAT 4-bit checkpoint through Ollama, G-family quotas, or the gpu-ollama-g5g-2b-q4_0 devops MCP agent. Triggers include "G5g", "T4G", "Graviton", "Turing", "SM 7.5", "arm64 GPU", "ollama", "ollama serve", "gemma4:e2b", "GGUF", "Q4_0".
---

# gpu-ollama-g5g-2b-q4_0 management

Provision and operate **EC2 G5g** (Graviton2 host + NVIDIA **T4G**, Turing SM 7.5) serving
`gemma4:e2b-it-qat` through the **Ollama daemon**, via the `gpu-ollama-g5g-2b-q4_0` MCP server.

## What has been measured

**Nothing. This rig has served no tokens.** Created 2026-09-02; `benchmarks/` is empty on
purpose. Any figure in this rig's prose is a property of the bundle, arithmetic from the
artifact, or a *sibling's* measurement, and says which.

## Why it exists

It shares an engine with **`gpu-llamacpp-g5g-2b-q4_0`** — Ollama links `libllama.so`, and the
Gemma 4 graph both run is upstream `src/models/gemma4.cpp`. **Slot 2 names the front end, not
the decoder.** They are two rigs because four differences are visible to a benchmark; read THE
NOTE in either `CLAUDE.md` before comparing them.

The headline asymmetry: **Ollama ships a working aarch64 CUDA binary with native sm_75 SASS;
llama.cpp publishes none, so the sibling compiles.** This rig has no build step.

## Order of operations

```
create_g5g_instance  ->  get_install_progress  ->  verify_gpu_arch
                     ->  verify_model_health   ->  get_metrics
```

No deploy step: cloud-init downloads the bundle, starts the daemon, pulls the tag and asserts
the model reached VRAM.

## The one thing that goes wrong silently

**A CPU-resident model.** Ollama chooses its own offload and cannot be told to fail — there is
no `--n-gpu-layers` equivalent. It will serve correctly, several times slower, and log nothing.

The only honest check is `size_vram` from `/api/ps`: **0 means CPU**, and less than `size` means
a partial offload. `verify_gpu_arch` reads it. If a decode number looks low, run that **first**.

## Three settings that are pinned, and why

| Setting | Ollama's default | Why pinned |
| --- | --- | --- |
| `OLLAMA_LLM_LIBRARY` | auto by driver | `cuda_v13` is **PTX-only for every arch** and JITs at load; `cuda_v12` has native sm_75 CUBIN. Pinning is what makes the pair comparable |
| `OLLAMA_CONTEXT_LENGTH` | `0` = 4k/32k/256k **by VRAM** | two instance sizes would silently get two contexts |
| `OLLAMA_KEEP_ALIVE` | `5m` | a sweep pausing longer reloads the model and records it as latency |

`ollama serve` takes **no arguments** — all of this is environment, and Ollama *ignores* an
unknown variable rather than rejecting it. A typo produces a working rig at the wrong settings.

## Reading a number off this rig

There is **no `/metrics` endpoint** — Ollama registers no Prometheus route. `get_metrics`
**probes**: it runs one generate and reads `eval_count` / `eval_duration`. Those are Go
`time.Duration`s, i.e. **nanoseconds**.

A probe measures one request, not a run. Quote decode, never `total_duration` — it carries
prefill, sampling and HTTP, and the PyTorch sibling measured the two disagreeing by 36%.

Health is `GET /` ("Ollama is running"). There is no `/health`, and `/` answers 200 before any
model is loaded.

## Cost and capacity

Spot by default. **Termination is cheaper here than on the llama.cpp sibling** — that rig loses
a compile, this one loses a 1.5 GB download and a 4.3 GB pull.
`InsufficientInstanceCapacity` is transient; loop the AZs.
