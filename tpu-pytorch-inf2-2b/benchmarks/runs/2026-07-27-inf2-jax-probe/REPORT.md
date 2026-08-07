# JAX-Neuron Gemma decoder probe on AWS Inferentia2

**Run:** 2026-07-27  
**Instance:** `inf2.xlarge` spot, `us-east-1d`  
**Instance ID:** `i-0f0fe6e6aafac60f6`  
**JAX:** 0.9.0  
**Backend:** JAX-Neuron PJRT, platform `neuron`

## Purpose

Establish that the current JAX-Neuron stack can compile and execute the main
structural operations required by a fixed-shape Gemma 4 E2B decoder before
porting model weights:

- BF16 RMSNorm
- rotary position embedding
- grouped-query attention
- functional static-KV cache update with `jax.lax.dynamic_update_slice`
- gated GELU MLP

The source is `jax_neuron/probe.py`. This is a compatibility probe, not a model
correctness or performance benchmark.

## Result

```json
{
  "device_count": 2,
  "devices": [
    "NeuronCore(id=0, process_index=0, local_id=0)",
    "NeuronCore(id=1, process_index=0, local_id=1)"
  ],
  "finite": true,
  "first_call_s": 3.0174,
  "jax_version": "0.9.0",
  "output_shape": [1, 512],
  "platform": "neuron",
  "warm_call_s": 0.0009
}
```

The probe passed. JAX discovered both logical NeuronCores, compiled the
fixed-shape decoder step, executed two positions while carrying the KV cache,
and returned finite output.

## Operational finding

The first attempt failed because the existing Option-B PyTorch container owned
both logical NeuronCores:

```text
Logical Neuron Core(s) not available - Requested:2 Available:0
```

Stopping `vllm-neuron` released the device and the unchanged JAX probe passed.
A future JAX deployment mode must not launch the Option-B container alongside
the JAX process.

## Warnings

- AWS labels the JAX Neuron platform experimental.
- EFA/OFI initialization warnings are expected on single-device
  `inf2.xlarge`; the probe does not use multi-instance collectives.
- The runtime fell back to synchronous tensor I/O because the asynchronous
  path is Trn2-specific. This does not invalidate correctness, but real decode
  benchmarking must include host/device KV transfer costs.

## Next milestone

Load `google/gemma-4-E2B-it-qat-q4_0-unquantized` into a pure-JAX Gemma 4
implementation and compare fixed-prompt greedy tokens against the proven
PyTorch reference before optimizing or attempting compressed W4A16 execution.

