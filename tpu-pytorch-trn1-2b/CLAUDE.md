# CLAUDE.md — AWS Trainium (trn1) Gemma-4 DevOps MCP

This rig packages the `tpu-pytorch-trn1-2b-management` skill and the `tpu-pytorch-trn1-2b` MCP server (`MCPServer`) for AWS EC2 trn1 instances. It was forked from `tpu-pytorch-inf2-2b` on 2026-10-05 and **differs from it in slot 3 only**, so the pair is an Inferentia2-vs-Trainium A/B on the same image.

## What it serves

`create_trn1_instance` launches the prebuilt Gemma-4 Option-B image (`docker.io/xbill9/gemma4-optb:slim`, E2B, override with `image=` or `OPTB_IMAGE`) on one Neuron device. The image was compiled for **inf2** and runs on trn1 unchanged: both chips carry two NeuronCore-v2 cores and 32 GiB of device memory (`aws ec2 describe-instance-types`). Measured in `benchmarks/runs/2026-10-05-smoke-trn1/`.

**There is no vLLM path, and none can be added by changing a flag.** vLLM-Neuron 0.24 and 0.21 list Trn2/Trn3 only; the 0.5.3 line that still covers Trn1 and Inf2 pins vLLM 0.16; no release has a Gemma-4 model class. The inf2 sibling's `serving="vllm"` path came up healthy and served gibberish for Gemma-4, which is why it was not carried over.

## Authoritative files

- `server.py`: trn1 MCP server and AWS lifecycle. Tool names say `trn1`; the inf2 sibling's say `inf2`.
- `torch_generate.py` / `torch_openai_server.py`: the native engine, copied from the inf2 rig. **No tool deploys it and it has not run on trn1.** The engine rules in `tpu-pytorch-inf2-2b/CLAUDE.md` (host-side embedding gather, traced shapes, one engine, check content not token count) apply unchanged, because they are NeuronCore-v2 rules.
- `.claude/skills/tpu-pytorch-trn1-2b-management/SKILL.md`: workflow and guardrails. The `mcp/` copies beside it are generated: `make skill`.

## Engineering rules

- boto3 and the standard credential chain; SSM Run Command for remote administration (dash, no bashisms, PATH lacks `/opt/aws/neuron/bin`); no inbound ports.
- Discovery is scoped to `ManagedBy=tpu-pytorch-trn1-2b`. This rig does **not** inherit the inf2 rig's legacy `inf2-devops` tag.
- Only `trn1.2xlarge`, `trn1.32xlarge`, `trn1n.32xlarge`. The images are single-device builds, so the 32xlarge sizes leave 15 devices idle.
- `trn1.2xlarge` is offered in **us-east-2c only** among us-east-1, us-east-2 and us-west-1 (2026-10-04); the region default is us-east-2.
- Launches default to spot. Trn spot quota (`L-6B0D517C`) was 0 vCPUs on this account until an increase was requested on 2026-10-04; on-demand (`L-2C3B7624`) is 8, one trn1.2xlarge.
- Host RAM is 32 GiB, against inf2.xlarge's 16: E2B's neff load peaked at 19.53 GB, which needs no swap here. Cloud-init still adds 16 GB as insurance for the larger images.

## Tests

`python3 -m unittest discover -s tests -v` (offline; mocks nothing in the cloud because no test calls it). `make lint`. Never pytest.
