---
name: tpu-jax-inf2-2b-management
description: Provision and operate AWS EC2 Inferentia2 instances and Neuron vLLM serving through the tpu-jax-inf2-2b MCP server.
---

# AWS Inferentia2 management

Use the `tpu-jax-inf2-2b` MCP tools for AWS Inf2 lifecycle work.

## Workflow

1. Call `get_help` and `check_inf2_quotas`.
2. Use `get_deployment_config` for a read-only review of launch settings.
3. Confirm the subnet, security group, and IAM instance profile.
4. Call `create_inf2_instance`; creation starts billing. Launches default
   to spot capacity — pass `spot=False` only when the user asks for
   on-demand. Spot capacity can be interrupted or unavailable; on an
   `InsufficientInstanceCapacity` / spot error, report it and offer
   on-demand instead of retrying silently.
5. Poll `list_inf2_instances`, then call `verify_neuron_health`.
6. Resolve the API with `get_endpoint`; call `query_model` after health passes.
7. Prefer `stop_inf2_instance` when preserving storage is useful.
   `terminate_inf2_instance` is permanent and requires explicit approval.

## Serving Gemma-4 (`serving="optb"`)

The Neuron vLLM DLC cannot serve Gemma-4 (`optimum-neuron` has no Gemma-4
model class; the endpoint comes up healthy but generates gibberish). For
`google/gemma-4-E2B-it`, pass `serving="optb"` to `create_inf2_instance` or
`get_deployment_config`. This deploys a prebuilt `torch_neuronx` container
(default `docker.io/xbill9/gemma4-optb:slim`, override with `OPTB_IMAGE`)
with compiled neffs and weights baked in — no Hugging Face token required.
It is a single-device build: only `inf2.xlarge` or `inf2.8xlarge`. Cloud-init
adds a 16 GB swapfile because the one-time neff load peaks ~14.5 GB of host
RAM on the 16 GB `inf2.xlarge`. Expect a long first start (image pull +
neff load) before health passes.

## Guardrails

- Accept only supported `inf2.*` types; never silently fall back to GPU.
- Never place Hugging Face tokens in user data or output. Use Secrets Manager.
- Use Systems Manager for remote commands. Do not expose SSH to the internet.
- Scope discovery to `ManagedBy=tpu-jax-inf2-2b`.
- Restrict port 8000 to trusted networks or a private load balancer.
- Match tensor parallelism to NeuronCore count; the server derives it.
- Upgrade the Neuron DLC as a tested SDK/container compatibility set.

The caller needs EC2, SSM, Secrets Manager, and Service Quotas permissions.
The instance profile needs SSM core permissions, ECR Public read access, and
read access to the configured Hugging Face secret.
