---
name: gpu-pytorch-g6-2b-q4w4a16-management
description: Manage AWS EC2 G6 capacity (x86_64 + NVIDIA L4) and Gemma 4 E2B QAT int4 serving through the hand-built PyTorch int4 engine (Triton GEMV, CUDA-graph decode). Use when the user asks about provisioning, launching, listing, or terminating G6 instances for the int4 PyTorch rig, debugging the int4 engine on L4 / Ada / SM 8.9, or the gpu-pytorch-g6-2b-q4w4a16 devops MCP agent. Triggers include "G6 int4", "q4w4a16 on L4", "PyTorch int4 on G6", "e2b_engine on AWS".
---

# gpu-pytorch-g6-2b-q4w4a16 management

Provision and operate **EC2 G6** (x86_64 host + NVIDIA **L4** GPU) serving Gemma 4 E2B
with Google's QAT 4-bit weights (`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`, vocabulary
tables from `-text-emb4`) through the `gpu-pytorch-g6-2b-q4w4a16` MCP server.

The engine is `local-pytorch-1650ti-2b-q4w4a16`'s: every int4 weight packed on the GPU, a
Triton int4 GEMV, split decode attention, and the decode step compiled with Inductor and
replayed as a CUDA graph. The AWS half is forked from `gpu-pytorch-g6-2b` (dense bf16,
stock transformers, same instance family).

## Order of operations

`create_g6_instance` → `get_install_progress` → `verify_gpu_arch` → `deploy_torch_server`
→ `get_torch_logs` → `verify_model_health`

Cloud-init installs the runtime only (pinned `transformers==5.16.1`,
`compressed-tensors==0.17.0`, `accelerate==1.14.0` into the DLAMI's torch, and checks
Triton imports). `deploy_torch_server` ships the payload (`openai_server.py`,
`e2b_engine.py`, `w4gemv.py`, `w4attn.py`, `gemma4_tools.py`) over SSM.

**Always `make skill` before `deploy_torch_server`**: the MCP server ships the payload next
to itself. `verify_model_health` compares the served build id with the local digest.

Engine start-up downloads both checkpoints, times an fp16 and an fp32 GEMM to choose the
prompt-read dtype, loads, and compiles and captures the decode step; follow it with
`get_torch_logs`. The first JSON line it prints carries the device, the chosen dtype, the
GEMM rates, weights and K/V MiB, and the load time.

## What carries from the 1650 Ti and what does not

- Compute dtype is **float16 on every GPU** (fixed in `e2b_engine.py`), even though the L4
  has bf16.
- `w4gemv.py`'s per-shape tilings were tuned on the **GTX 1650 Ti**; they run here but are
  not tuned for the L4.
- Prompt reads multiplied in fp32 on the 1650 Ti because it has no tensor cores; the L4
  has them, and `--prefill-dtype auto` decides on the device.
- The memory-efficient SDPA fix for 32k+1-query chunks is in the engine and applies to the
  DLAMI's torch 2.13.

## Serving facts

- One request at a time, greedy. `MAX_MODEL_LEN=131072`.
- `/metrics` carries the `tpu_jax_*` series `get_metrics` parses; quote the
  `tpu_jax_decode_tokens_per_second` gauge.
- `tpu_jax_degenerate_responses_total` is what `verify_model_health` reads either side
  of its probe; a non-empty reply is not evidence of health.

## Lifecycle guardrails

- Spot by default; surface capacity errors.
- Explicit subnet, security-group and instance-profile ids. No broad network or IAM policy.
- HF token from Secrets Manager at boot (`save_hf_token`), never in user data.
- Instance discovery is scoped to `ManagedBy=gpu-pytorch-g6-2b-q4w4a16`.
- The unit is `torch-g6-q4.service`, under `/opt/torch-g6-q4`.

Record results in `benchmarks/runs/<date>-<what>-g6/`.
