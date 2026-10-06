---
name: tpu-vllm-trn1-2b-management
description: Provision and operate AWS EC2 Trainium (trn1) instances running Gemma-4 through the vLLM-Neuron DLC, via the tpu-vllm-trn1-2b MCP server.
---

# AWS Trainium (trn1) management, vLLM-Neuron DLC

Use the `tpu-vllm-trn1-2b` MCP tools for trn1 lifecycle work.

## Workflow

1. Call `get_help` and `check_trn1_quotas`. A new account has 0 vCPUs of Trn
   spot quota and 8 of on-demand, which is enough for one `trn1.2xlarge`.
2. Call `save_hf_token` if the secret does not exist yet; Gemma checkpoints are
   gated and the instance reads the token from Secrets Manager at boot.
3. Use `get_deployment_config` for a read-only review of launch settings.
4. Confirm the subnet, security group, and IAM instance profile. The profile
   needs `AmazonSSMManagedInstanceCore` and `secretsmanager:GetSecretValue` on
   the token secret. `trn1.2xlarge` is offered in one us-east-2 zone
   (`us-east-2c`); the subnet has to be in a zone that offers it.
5. Call `create_trn1_instance`; creation starts billing. Launches default to
   spot; pass `spot=False` when the user asks for on-demand or the spot quota
   is still 0. On a capacity or quota error, report it and offer on-demand
   instead of retrying silently.
6. Poll `list_trn1_instances`, follow `get_server_logs` through the Neuron
   compile, then call `verify_neuron_health`.
7. Reach the API through an SSM port forward (the security group should have no
   inbound rules), then call `query_model`.
8. `stop_trn1_instance` works for on-demand instances only.
   `terminate_trn1_instance` is permanent and requires explicit approval.

## What to expect

The DLC is vLLM-Neuron 0.16 (SDK 2.31.1), the newest line that still covers
Trn1; 0.21 and 0.24 list Trn2/Trn3 only. **No vLLM-Neuron release has a
Gemma-4 model class**, and this DLC came up healthy on inf2 and served
gibberish for Gemma-4. **On trn1 (2026-10-05) it never started**: the image's
Transformers 4.57.6 has no `gemma4` model type and vLLM 0.16 has no Gemma-4
class, so the container exits during configuration and restarts. A crash loop
with `does not recognize this architecture` in `get_server_logs` is this known
result, not a deploy fault.

**Judge the endpoint by the text it returns, never by its status.** Ask a
question with a known answer ("What is the capital of France?") and report the
answer verbatim. A `200 OK` with fluent nonsense, a repeated token, or an empty
completion is a failed model, not a working deploy. For a working Gemma-4
endpoint on trn1, point the user at `tpu-pytorch-trn1-2b`.

## Guardrails

- Accept only `trn1.*` types; never silently fall back to GPU or inf2.
- Use Systems Manager for remote commands. Do not expose SSH or the API port.
- Never put the HF token in user data.
- Scope discovery to `ManagedBy=tpu-vllm-trn1-2b`.

The caller needs EC2, SSM, Secrets Manager and Service Quotas permissions.
