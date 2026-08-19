---
pretty_name: Gemma 4 E2B QAT on TPU v6e-1 — JAX Benchmark Artifacts
language:
- en
license: other
tags:
- benchmark
- tpu
- jax
- gemma
- llm-inference
- quantization
- performance
task_categories:
- text-generation
---

# Gemma 4 E2B QAT on TPU v6e-1

This dataset packages measurements and reproducibility artifacts for a pure-JAX
Gemma 4 E2B QAT inference path on one Google Cloud TPU v6e-1
(`ct6e-standard-1t`, 32 GB HBM3). It contains no model weights, prompts from
users, or credentials.

## What is included

- `kv-quant-v6e1/`: corrected KV-capacity, buffer-donation, quantization,
  prefill-admission, quality, and roofline results plus the benchmark scripts.
- `real-http-v6e1/`: real-checkpoint OpenAI-compatible HTTP latency and
  concurrency measurements.
- `vllm-baseline/`: the independently measured vLLM BF16 serving baseline.
- `article/`: the accompanying long-form analysis and cover image.

The tested checkpoint is
[`google/gemma-4-E2B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct).
Access and use of that checkpoint remain governed by its upstream terms.

## Results at a glance

| Finding | Measured result |
|---|---|
| Buffer donation | 1.60–1.62× faster for BF16 KV |
| INT8 KV | 1.17–1.19× faster than donated BF16; 1.82–1.98× KV capacity |
| INT4 PLE | 53% smaller parameter tree; no statistically meaningful throughput gain |
| Best kernel configuration | 2,888 aggregate tok/s at B=32, context 8,192 |
| Real HTTP serving | approximately 139–141 aggregate tok/s |

## Essential interpretation

The 2,888 tok/s result is a **static-shape decode-kernel benchmark** using
architecture-shaped synthetic parameter values. Real checkpoint weights were
used for correctness and quality validation, and values do not change the
compiled static-shape work, but the kernel number is not an HTTP serving result.

The current server executes each HTTP request as an independent B=1 program. It
does not perform continuous or device-level request batching. Concurrent HTTP
requests therefore remain near the single-stream aggregate rate while latency
grows with contention.

Several earlier conclusions were withdrawn as the measurement method improved.
Read `kv-quant-v6e1/REPORT.md`, especially its correction notice and final
addenda, before quoting capacity or throughput figures.

## Reproduction

The source implementation and tests are maintained at
[`xbill9/gemma4-dev`](https://github.com/xbill9/gemma4-dev/tree/main/tpu-jax-v6e1-2b). The principal corrected
harness is:

```bash
python3 ports/gemma4/jax_e_benchmark_sweep_v2.py \
  --batch-sizes 1,2,4,8,16,32,64 \
  --contexts 8,128,512,2048 \
  --json-out results.json
```

TPU performance, HBM ceilings, Pallas/Mosaic behavior, and compilation timings
cannot be reproduced meaningfully on CPU. CPU is suitable for endpoint, SSE,
scheduler, and numerical-correctness tests using the repository's tiny
synthetic checkpoint.

## Licensing

No license is asserted here for third-party model weights, which are not
included. Source files retain the terms of their originating repository.
Benchmark measurements and prose are provided for research and reproducibility;
contact the author before redistribution if you require explicit licensing
terms.
