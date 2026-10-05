# tpu-pytorch-trn1-2b

Gemma 4 E2B on one AWS Trainium chip (`trn1.2xlarge`), served by the hand-ported Neuron image built for Inferentia2. An MCP server provisions the instance through EC2 and Systems Manager, and serves the image on an OpenAI-compatible endpoint.

The images compiled for inf2 run on trn1 unchanged: both chips have two NeuronCore-v2 cores and 32 GiB of device memory.

| | inf2.xlarge | trn1.2xlarge |
|---|---|---|
| Neuron device | 1 Inferentia2, 2 cores, 32 GiB | 1 Trainium, 2 cores, 32 GiB |
| Host | 4 vCPU, 16 GiB | 8 vCPU, 32 GiB |
| On-demand, us-east-1 | $0.7582/hr | $1.3438/hr |

## Quick start

```bash
pip install -r requirements.txt
./project-setup.sh . --region us-east-2
```

Then, from Claude Code: `check_trn1_quotas`, `get_deployment_config`, `create_trn1_instance` (pass `spot=False` until a Trn spot quota exists), `verify_neuron_health`, `query_model`, `terminate_trn1_instance`.

There is no vLLM path: no vLLM-Neuron release supports both Trn1 and Gemma 4.

## Measured

`benchmarks/runs/2026-10-05-smoke-trn1/`: E2B ready 121.2 s after a 365 s image pull, 36.0 to 38.1 tok/s on six answers of 94 to 123 tokens, 19.53 GB peak host memory. The 26B A4B MoE image built for one inf2.xlarge (`xbill9/gemma4-optb-26b:xlarge`) also serves unchanged: ready 220.1 s after start, 5.5 tok/s decode, 0.31 to 0.34 s prefill, `Paris` 3 of 3.
