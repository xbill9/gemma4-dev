# Gemma 4 E2B through the vLLM-Neuron DLC on trn1.2xlarge — smoke test

- **Date:** 2026-10-05 (UTC), us-east-2c, one spot `trn1.2xlarge` (`i-04293aceb3a1d7ea3`), launched by this rig's `create_trn1_instance` with its defaults
- **Host:** 1 Trainium device (2 NeuronCore-v2, 32 GB), Ubuntu 24.04.4, AMI `Deep Learning AMI Neuron (Ubuntu 24.04) 20260818` (`ami-0222021b369f03219`)
- **Image:** `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.31.1-ubuntu24.04` (`sha256:bc617207…`)
- **Command:** `vllm serve google/gemma-4-E2B-it --tensor-parallel-size 2 --max-num-seqs 4 --max-model-len 4096`
- **Question:** does the newest vLLM-Neuron line that still covers Trn1 serve Gemma 4?
- **Answer:** no. The server exits during configuration, before it touches the Neuron device, and never opens its port.

## Stack inside the image

| Package | Version |
|---|---|
| vllm | 0.16.0 |
| vllm-neuron | 0.5.3 |
| transformers | 4.57.6 |
| torch / torch-neuronx | 2.9.1 / 2.9.0.2.15.32035 |
| neuronx-distributed-inference | 0.10.18399 |
| neuronx-cc | 2.26.6360.0 |
| optimum-neuron | not installed |

## Result

| Stage | Time (UTC) |
|---|---|
| Launch | 23:11:52 |
| SSM online | 23:12:49 |
| Image pulled, container running | by 23:18:17 |
| First exit | before 23:18:17 |
| Container stopped by hand | after 8 restarts, `/health` never answered |

Every start fails the same way (`vllm-server.log`, first crash):

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, The checkpoint you are trying to load has model type `gemma4` but Transformers does not recognize this architecture.
```

The image lacks Gemma 4 at two layers, checked inside the same image:

- **Transformers 4.57.6 has no `gemma4` model type**, which is the error above.
- **vLLM 0.16's model registry stops at Gemma 3n**: `Gemma2ForCausalLM`, `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`, `Gemma3nForCausalLM`, `Gemma3nForConditionalGeneration` and older; no `Gemma4ForConditionalGeneration`.

Upgrading Transformers inside the image would clear the first and stop at the second: vLLM would have no class to build the model from. vllm-neuron 0.5.3 loads models through NxD Inference, whose 0.10 release has `gemma3` and no `gemma4`.

## What this does and does not show

- It shows this DLC cannot start a Gemma 4 server on trn1 with the stock checkpoint. No Neuron compile ran, so nothing here is evidence about Trainium itself.
- The failure is in Python packages, ahead of any instance-specific code.
- `tpu-pytorch-trn1-2b` serves Gemma 4 E2B on the same instance type through the prebuilt `torch_neuronx` image (`benchmarks/runs/2026-10-05-smoke-trn1` in that rig).
