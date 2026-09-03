---
name: gpu-llamacpp-g5g-2b-q4_0-management
description: Manage AWS EC2 G5g capacity (Graviton2 + NVIDIA T4G) and Gemma 4 E2B Q4_0 GGUF serving under llama.cpp. Use when the user asks about provisioning, launching, listing, or terminating G5g instances, building or debugging llama.cpp on T4G / Turing / aarch64, serving a GGUF or QAT 4-bit checkpoint on a GPU, G-family quotas, or the gpu-llamacpp-g5g-2b-q4_0 devops MCP agent. Triggers include "G5g", "T4G", "Graviton", "Turing", "SM 7.5", "arm64 GPU", "llama.cpp", "llama-server", "GGUF", "Q4_0", "gguf on GPU".
---

# gpu-llamacpp-g5g-2b-q4_0 management

Provision and operate **EC2 G5g** (Graviton2 host + NVIDIA **T4G**, Turing SM 7.5) serving
`google/gemma-4-E2B-it-qat-q4_0-gguf` through **`llama-server`**, via the
`gpu-llamacpp-g5g-2b-q4_0` MCP server.

## What has been measured

**Nothing. This rig has served no tokens.** Scaffolded 2026-09-02; `benchmarks/` is empty on
purpose. Any figure you find in this rig's prose is either arithmetic from the GGUF's own
tensor table or a measurement from a *sibling* rig, and says which. Do not report one as this
rig's result.

## Why it exists

It is the only rig in the monorepo serving **4-bit weights on a GPU**. vLLM 0.26.0 has no
`gguf` module at all (CUDA build or TPU), no GGUF reader exists in the JAX ecosystem, and
transformers 5.12.1 reads the file but dequantizes to fp32 and silently drops 35
`layer_scalar` tensors. llama.cpp is what is left. Root `QUANTIZATION.md` has the evidence.

Its sibling is **`gpu-ollama-g5g-2b-q4_0`** — same silicon, same weights, **same engine**
(Ollama links llama.cpp). They still differ in four ways a benchmark can see; that note is in
both rigs' `CLAUDE.md` and you should read it before comparing them.

## Order of operations

```
create_g5g_instance  ->  get_install_progress  ->  verify_gpu_arch
                     ->  verify_model_health   ->  get_metrics
```

**There is no deploy step.** Cloud-init builds llama.cpp and enables the unit itself, and
llama-server fetches its own checkpoint. Nothing of ours ships to the box.

## The three things that go wrong silently

Each produces a server that starts, binds, and answers **correctly**:

1. **A CPU-only build** — several times slower, no error anywhere. `verify_gpu_arch` greps the
   built binary's own `--list-devices`; `verify_model_health` flags decode under 3 tok/s.
2. **A partial GPU offload** — same shape of failure, caught by `--n-gpu-layers 999`.
3. **No `nvcc`** — the bootstrap exits 1 rather than letting cmake configure without CUDA.

If a decode number looks low, run `verify_gpu_arch` **before** investigating anything else.

## Reading a number off this rig

Quote the decode rate `get_metrics` derives from `llamacpp:tokens_predicted_total` /
`llamacpp:tokens_predicted_seconds_total`. **Never** an end-to-end rate: the PyTorch sibling
measured the two disagreeing by up to 36% on the same rows. Prefill is reported separately
(`llamacpp:prompt_*`) and must never be folded in.

`--metrics` is off by default in llama.cpp; the rig always passes it. A 404 from `/metrics`
means the unit is missing that flag.

For a sweep, `sweep.py --decode-source auto` resolves to `stream` here — the client-side
inter-token statistic — because llama-server does not emit our servers' invented
`usage.decode_tokens_per_second`. That is the cross-rig statistic, and it is the right one.

## Cost and capacity

Spot by default. **Termination costs more here than on the sibling rigs**: they lose a pip
install and a model cache, this one loses a compile as well. `InsufficientInstanceCapacity` is
transient — loop the AZs rather than giving up.

## Configuration

`tpu.env` is the source of truth and every key in it is read by `server.py` (a test asserts
this). The directory name is documentation, not config — never copy a slot value into a flag.

Key knobs: `LLAMA_CPP_REF` (pinned, never `master`), `CUDA_ARCH` (75), `CONTEXT_SIZE`,
`PARALLEL_SLOTS` (1 — raise it only in a run whose report says so), `MODEL_FILE` (names *which*
GGUF, because a repo can hold several).
