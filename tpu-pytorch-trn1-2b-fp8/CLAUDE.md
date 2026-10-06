# CLAUDE.md — AWS Trainium (trn1) Gemma-4 DevOps MCP, `fp8` checkpoint

This rig packages the `tpu-pytorch-trn1-2b-fp8-management` skill and the `tpu-pytorch-trn1-2b-fp8` MCP server (`MCPServer`) for AWS EC2 trn1 instances. It was forked from `tpu-pytorch-trn1-2b` on 2026-10-05 and **differs from it in slot 5 only**; it **differs from `tpu-pytorch-inf2-2b-fp8` in slot 3 only**. Its engine payload (`torch_generate.py`, `torch_openai_server.py`, `ct_load.py`, `bench_arm.py`, `requirements-serving.txt`) is a copy of that inf2 rig's, taken 2026-10-05 while the inf2 rig was still untracked. A later change on either side does not propagate.

## What it serves

The checkpoint is `xbill9/gemma-4-E2B-it-qat-q4_0-fp8-text`: fp8 W8A8 (e4m3, one scale per output channel) from Google's QAT weights, text only. `CHECKPOINT` in `server.py` holds it.

**No tool deploys that checkpoint.** `create_trn1_instance` launches the prebuilt Option-B image (`docker.io/xbill9/gemma4-optb:slim`), which serves the **dense** E2B reference, the same as the base trn1 rig. The checkpoint runs through the native engine by hand on the instance: `torch_generate.py --model xbill9/gemma-4-E2B-it-qat-q4_0-fp8-text` for parity, `bench_arm.py` for speed and host memory, `torch_openai_server.py` to serve it. **None of the three has run on trn1 yet**, and this rig has no measurements.

`ct_load.py` expands the compressed-tensors weights to bf16 on the host at load, because NeuronCore-v2 has no int4, int8 or fp8 matmul the `torch_neuronx` trace path can reach. The traced graph is the same dense bf16 graph either way; the encoding changes the weight values and the download size. Trainium1 is NeuronCore-v2 as well, so this holds unchanged here.

**There is no vLLM path in this rig.** `tpu-vllm-trn1-2b` holds the vLLM-Neuron attempt for trn1.

## Authoritative files

- `server.py`: trn1 MCP server and AWS lifecycle. Identical to `tpu-pytorch-trn1-2b/server.py` except the module docstring, `RIG_NAME`, `CHECKPOINT` and the `get_help` line that reports it.
- `torch_generate.py` / `torch_openai_server.py` / `ct_load.py` / `bench_arm.py`: the native engine and its loader and benchmark, copied from `tpu-pytorch-inf2-2b-fp8`. The engine rules in `tpu-pytorch-inf2-2b/CLAUDE.md` (host-side embedding gather, traced shapes, one engine, check content not token count) apply unchanged, because they are NeuronCore-v2 rules.
- `requirements-serving.txt`: what gets installed on the instance. `compressed-tensors` pulls a newer torch than the DLAMI's Neuron-matched one, so install it with a constraint that pins torch to the version `torch-neuronx` requires.
- `.claude/skills/tpu-pytorch-trn1-2b-fp8-management/SKILL.md`: workflow and guardrails. The `mcp/` copies beside it are generated: `make skill`.

## Engineering rules

- boto3 and the standard credential chain; SSM Run Command for remote administration (dash, no bashisms, PATH lacks `/opt/aws/neuron/bin`); no inbound ports.
- Discovery is scoped to `ManagedBy=tpu-pytorch-trn1-2b-fp8`. Instances launched by `tpu-pytorch-trn1-2b` are invisible here, and the reverse.
- Only `trn1.2xlarge`, `trn1.32xlarge`, `trn1n.32xlarge`. The engine is single-device, so the 32xlarge sizes leave 15 devices idle.
- `trn1.2xlarge` is offered in **us-east-2c only** among us-east-1, us-east-2 and us-west-1 (2026-10-04); the region default is us-east-2.
- Launches default to spot. Trn spot quota (`L-6B0D517C`) was 0 vCPUs on this account until an increase was requested on 2026-10-04; on-demand (`L-2C3B7624`) is 8, one trn1.2xlarge. All trn1 rigs share that quota.

## Tests

`python3 -m unittest discover -s tests -v` (offline). `make lint`. Never pytest.
