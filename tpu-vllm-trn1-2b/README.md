# tpu-vllm-trn1-2b

Gemma 4 E2B on one AWS Trainium chip (`trn1.2xlarge`) through the vLLM-Neuron Deep Learning Container. An MCP server provisions the instance through EC2 and Systems Manager and starts `vllm serve` on an OpenAI-compatible endpoint.

The DLC is `pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.31.1-ubuntu24.04`, the newest vLLM-Neuron line that still covers Trn1. No vLLM-Neuron release has a Gemma 4 model class, and this DLC served gibberish for Gemma 4 on Inferentia2, which has the same NeuronCore-v2. For a working Gemma 4 endpoint on trn1, use `tpu-pytorch-trn1-2b`.

## Quick start

```bash
pip install -r requirements.txt
./project-setup.sh . --region us-east-2
```

Then, from Claude Code: `save_hf_token`, `check_trn1_quotas`, `get_deployment_config`, `create_trn1_instance` (pass `spot=False` until a Trn spot quota exists), `get_server_logs` to follow compilation, `verify_neuron_health`, `query_model`, `terminate_trn1_instance`.

## Measured

`benchmarks/runs/2026-10-05-smoke-vllm-trn1/`: the server exits before it touches the Neuron device. The image's Transformers 4.57.6 has no `gemma4` model type, and vLLM 0.16's model registry stops at Gemma 3n. It restarted 8 times and never answered `/health`.
