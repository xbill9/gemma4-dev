---
name: tpu-pytorch-trn1-2b-q4w4a16-management
description: Provision and operate AWS EC2 Trainium (trn1) instances for the Gemma-4 E2B q4w4a16 checkpoint (xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text) through the tpu-pytorch-trn1-2b-q4w4a16 MCP server.
---

# AWS Trainium (trn1) management, `q4w4a16` checkpoint

Use the `tpu-pytorch-trn1-2b-q4w4a16` MCP tools for trn1 lifecycle work.

## Workflow

1. Call `get_help` and `check_trn1_quotas`. A new account has 0 vCPUs of Trn
   spot quota and 8 of on-demand, which is enough for one `trn1.2xlarge`.
2. Use `get_deployment_config` for a read-only review of launch settings.
3. Confirm the subnet, security group, and IAM instance profile. `trn1.2xlarge`
   is offered in one us-east-2 zone (`us-east-2c`) and not in us-east-1 or
   us-west-1 (checked 2026-10-04); the subnet has to be in a zone that offers it.
4. Call `create_trn1_instance`; creation starts billing. Launches default to
   spot; pass `spot=False` when the user asks for on-demand or the spot quota
   is still 0. On a capacity or quota error, report it and offer on-demand
   instead of retrying silently.
5. Poll `list_trn1_instances`, then call `verify_neuron_health`.
6. Reach the API through an SSM port forward (the security group should have no
   inbound rules), then call `query_model` after health passes.
7. `stop_trn1_instance` works for on-demand instances only.
   `terminate_trn1_instance` is permanent and requires explicit approval.

## What it serves

The Gemma-4 images compiled for Inferentia2 run unchanged on trn1: both chips
carry two NeuronCore-v2 cores and 32 GiB of device memory. The default image is
`docker.io/xbill9/gemma4-optb:slim` (E2B); pass `image=` for another
single-device build. Measured on `trn1.2xlarge` (2026-10-05): ready 121.2 s
after a 365 s image pull, 36.0 to 38.1 tok/s on six answers of 94 to 123 tokens,
19.53 GB peak host memory during the neff load.

Each image's graph is traced at a fixed geometry (E2B: 512 total, 128 prompt
tokens), so a longer prompt is refused rather than truncated. The server
ignores `tools` in a chat request, so it cannot drive an agent.

There is **no vLLM path**: vLLM-Neuron 0.24 and 0.21 support Trn2/Trn3 only,
the 0.5.3 line that still covers Trn1 pins vLLM 0.16, and no release has a
Gemma-4 model class.

## This rig's checkpoint: `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`

The image above serves the dense E2B reference. This rig exists for `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` (the QAT W4A16 repack that keeps the QAT grid, text only), which runs only through the native engine below. Never report the image's output as this checkpoint's. `ct_load.py` expands the weights to bf16 on the host at load; NeuronCore-v2 has no low-precision matmul the trace path can reach.

## The native engine (`torch_openai_server.py`): not deployed by any tool

The skill ships `torch_generate.py`, `torch_openai_server.py`, `ct_load.py` and `bench_arm.py`, the same
traced-graph design as the image but tracing from a checkpoint at start-up.
**No tool starts it** and it has not been run on trn1. If asked to run it by
hand: `pip install -r requirements-serving.txt` with torch pinned by a constraint
(never `pip install torch`; `compressed-tensors` would otherwise replace it), pass
`--model xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`, set
`--neff-dir`, and treat an **empty completion as the signature Neuron fault**:
an oversized device gather returns zeros, which decode to an EOS, so the
endpoint answers `200 OK` with nothing in it.

## Guardrails

- Accept only `trn1.*` types; never silently fall back to GPU or inf2.
- Use Systems Manager for remote commands. Do not expose SSH or the API port.
- Scope discovery to `ManagedBy=tpu-pytorch-trn1-2b-q4w4a16`.

The caller needs EC2, SSM and Service Quotas permissions. The instance profile
needs `AmazonSSMManagedInstanceCore`.
