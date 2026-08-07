# Gemma 4 E2B QAT on AWS Inferentia2

**Run:** 2026-07-26  
**Instance:** `inf2.xlarge` spot, `us-east-1f`  
**Model:** `google/gemma-4-E2B-it-qat-q4_0-unquantized`  
**Runtime:** Neuron Runtime 2.30.51, `torch_neuronx` Option-B two-graph KV-cache path  
**Shapes:** `KV_BUCKET=32`, `KV_MAX=128`

## Acceptance

- QAT checkpoint, host tables, and precompiled prefill/decode graphs loaded.
- Neuron health passed with 2 visible NeuronCores and 32 GB device memory.
- Built-in deterministic self-test passed:
  - Prompt: `What is the capital of France?`
  - Output: `Paris`
- Independent OpenAI-compatible inference passed.
- Warm start after container creation: 273.7 seconds.
- Peak host RSS: 14.36 GB.

## Prompt smoke test

| Prompt | Result | Output tokens | Throughput |
| --- | --- | ---: | ---: |
| Explain quantization-aware training in one sentence | Coherent and accurate | 34 | 20.4 tok/s |
| Explain AWS Inferentia in one sentence | Coherent; called it “AWS Inferential” | 23 | 22.2 tok/s |
| `17 * 23`, number only | Correct: `391` | 3 | 16.0 tok/s |
| Write a Python `clamp` function | Coherent but over-explained; hit output cap | 81 | 23.3 tok/s |

The short-response measurements are latency-dominated. The 81-token request is
the best sustained-decode estimate from this smoke run: **23.3 tok/s**.

## Baseline comparison

The repository's established stock Gemma 4 E2B Option-B baseline on
`inf2.xlarge` is approximately **44–46 tok/s**. This QAT graph currently
delivers about half that rate.

The deployment uses the expected Option-B layout: separate prefill and decode
graphs, both NeuronCores visible to the process, host-side embeddings/PLE, and
KV tensors passed as graph I/O. There is no evidence that the slowdown is
caused by a missing device or failed graph load.

## Next investigation

1. Compare the stock and QAT trace recipes and compiler flags.
2. Inspect generated graph/compiler reports for inserted casts or unfused
   operations in the QAT decode graph.
3. Run a fixed-length deterministic decode benchmark to remove EOS and HTTP
   variance.
4. Preserve this working QAT deployment while developing a separate optimized
   graph artifact.

