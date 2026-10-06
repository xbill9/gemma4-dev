# CLAUDE.md — AWS Trainium (trn1) Gemma-4 on the vLLM-Neuron DLC

This rig packages the `tpu-vllm-trn1-2b-management` skill and the `tpu-vllm-trn1-2b` MCP server (`MCPServer`) for AWS EC2 trn1 instances. It was forked from `tpu-pytorch-trn1-2b` on 2026-10-05 and **differs from it in slot 2 only**: it serves through vLLM instead of the prebuilt `torch_neuronx` images.

## What it serves, and what is expected

`create_trn1_instance` runs `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.31.1-ubuntu24.04` with `vllm serve google/gemma-4-E2B-it`. That is the newest vLLM-Neuron line that still covers Trn1: the 0.21 and 0.24 images (SDK 2.31 and 2.32) list Trn2/Trn3 only, and the 0.24 plugin registers five architectures (Llama, Eagle3 Llama, GPT-OSS, Qwen3, Qwen3-VL) and nothing for Gemma. NxD Inference 0.10 (SDK 2.32) has `gemma3` and no `gemma4`.

**No vLLM-Neuron release has a Gemma-4 model class**, and the same 0.16 DLC on inf2 (`tpu-pytorch-inf2-2b`, `serving="vllm"`) came up healthy and served gibberish. **Measured on trn1 (2026-10-05, `benchmarks/runs/2026-10-05-smoke-vllm-trn1`): the server never starts.** vLLM exits during configuration because the image's Transformers 4.57.6 has no `gemma4` model type, and vLLM 0.16's registry stops at Gemma 3n, so upgrading Transformers alone would not help. No Neuron compile ran. **A healthy `/health` and a `200 OK` are not evidence the model works — read the text `query_model` returns.** `tpu-pytorch-trn1-2b` is the rig that serves Gemma-4 on trn1.

## Authoritative files

- `server.py`: trn1 MCP server and AWS lifecycle. It shares its lifecycle tools with `tpu-pytorch-trn1-2b` and adds `save_hf_token`. The tools take `model=` where the sibling takes `image=`.
- `.claude/skills/tpu-vllm-trn1-2b-management/SKILL.md`: workflow and guardrails. The `mcp/` copies beside it are generated: `make skill`.
- `docs/neuron-jax-quirks.md`: NeuronCore-v2 failure modes, carried from the sibling, useful for reading wrong output.

## Engineering rules

- boto3 and the standard credential chain; SSM Run Command for remote administration (dash, no bashisms, PATH lacks `/opt/aws/neuron/bin`); no inbound ports.
- The HF token lives in Secrets Manager (`HF_SECRET_ID`, default `tpu-vllm-trn1-2b/hf-token`) and is read at boot. Never put it in user data. The instance profile needs `secretsmanager:GetSecretValue` on it as well as `AmazonSSMManagedInstanceCore`.
- `TENSOR_PARALLEL_SIZE` defaults to 2, one device. The container gets every device on the instance but vLLM uses only that many cores; E2B's heads do not split 32 ways on a `trn1.32xlarge`. The value is validated against the instance's core count.
- Keep Neuron SDK, vLLM and DLC versions moving as one tested set; check the image's supported instance list before changing `VLLM_IMAGE`.
- Discovery is scoped to `ManagedBy=tpu-vllm-trn1-2b`.
- Only `trn1.2xlarge`, `trn1.32xlarge`, `trn1n.32xlarge`. `trn1.2xlarge` is offered in **us-east-2c only** among us-east-1, us-east-2 and us-west-1 (2026-10-04); the region default is us-east-2.
- Launches default to spot. Trn spot quota (`L-6B0D517C`) was 0 vCPUs on this account until an increase was requested on 2026-10-04; on-demand (`L-2C3B7624`) is 8, one trn1.2xlarge, shared by every trn1 rig.

## Tests

`python3 -m unittest discover -s tests -v` (offline). `make lint`. Never pytest.
