# Gemma 4 E2B QAT on a Single TPU with Pure JAX

This repository is an experimental, inspectable inference path for
`google/gemma-4-E2B-it-qat-w4a16-ct` on a single-chip Cloud TPU. It loads the QAT
checkpoint directly from safetensors, executes Gemma 4 in JAX without PyTorch,
and exposes an OpenAI-compatible HTTP/SSE server.

**Target and measurements now agree.** The deployment target is a **v6e-1
flex-start Compute Engine instance** — `ct6e-standard-1t`, 32 GB HBM3,
`europe-west4-a` — which is what `server.py`, `set_env.sh` and
[`deploy.md`](deploy.md) provision, and it is the chip every *measured* number on
this page was taken on. Those measurements predate the migration off the Cloud TPU
API; the control plane changes provisioning and not serving, so they still hold.
With 32 GB the checkpoint choice is free: `deploy.md` works through why the faster
bf16 `-q4_0-unquantized` is the right default here, where on a 16 GB chip it is not.

The project began with a practical incompatibility: the tested vLLM TPU stack
could not load this QAT export. Building the missing path uncovered a broader
measurement story about KV-cache accounting, JAX buffer donation, quantization,
static-shape batching, and the gap between a fast kernel and a useful server.

## What is verified

- The QAT checkpoint loads without PyTorch in the path.
- Cached decode matches full-sequence re-forward within float32 tolerance.
- Padding to 128-aligned TPU buckets does not change model output.
- INT8 KV attention matches a dequantize-first reference.
- Chunked prefill is token-exact against one-shot prefill.
- Greedy SSE and non-streaming HTTP responses produce identical text.
- CPU tests use a tiny synthetic checkpoint; TPU claims were measured on
  `ct6e-standard-1t` with 32 GB HBM3.

## Corrected v6e-1 results

The final 2026-07-29 revalidation used checkpoint-shaped static programs,
isolated processes, warmup, and 15 timed samples.

| Finding | Measured result |
| :--- | :--- |
| Buffer donation | **1.60–1.62×** faster with BF16 KV |
| INT8 KV | **1.17–1.19×** faster than donated BF16 |
| INT8 KV capacity | **1.82–1.98×** donated BF16 capacity |
| INT4 PLE | Parameter tree **53% smaller**; no meaningful throughput gain |
| Best static decode kernel | **2,888 aggregate tok/s**, B=32, context 8,192 |
| Real-checkpoint HTTP server | Approximately **139–141 aggregate tok/s** |

The largest optimization was buffer donation. Without `donate_argnums`, a
single-token `dynamic_update_slice` could leave two full KV caches live during
decode. Donation removed that copy, increased throughput, and nearly doubled
the resident-token ceiling.

INT8 KV then reduced bandwidth and approximately doubled capacity. Its real
checkpoint quality check measured 28.41 versus 28.73 perplexity for BF16, with
97.08% greedy-token agreement over 583 teacher-forced steps. Error did not grow
through a separate 968-step continuous decode.

Read the correction history before quoting an absolute capacity number:
[`benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md`](benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md).
Earlier results were withdrawn after finding an artificial power-of-two capacity
invariant, an undonated cache copy, and an incorrect roofline accounting of the
PLE gather.

## Kernel speed is not serving speed

The 2,888 tok/s point is a static-shape decode-kernel measurement using
architecture-shaped synthetic parameter values. Static values do not change the
compiled work, but this is not an HTTP benchmark or a direct vLLM comparison.

The real checkpoint served through `jax_openai_server.py` measured:

| Prompt tokens | Prefill | Decode |
| ---: | ---: | ---: |
| 506 | 9.3 ms | 141.1 tok/s |
| 2,045 | 32.0 ms | 140.5 tok/s |
| 7,679 | 318.6 ms | 138.5 tok/s |

At HTTP concurrency 2/4/8, aggregate throughput remained
128.7/133.6/143.3 tok/s while median request latency rose
497/952/1,775 ms. Every request succeeded, but none formed a device batch.

The current server runs independent B=1 executions. Reaching the batched-kernel
rate requires a request batcher, batched KV ownership, continuous admission, and
prefix reuse. Until then, this is a validated experimental engine—not a vLLM
replacement. Full results:
[`benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md`](benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md).

## Repository map

| Path | Purpose |
| :--- | :--- |
| `ports/gemma4/jax_e_model.py` | Gemma 4 E2B forward, cached decode, quantized KV, and Pallas experiments |
| `ports/gemma4/jax_e_loader.py` | Torch-free safetensors/QAT loader |
| `ports/gemma4/jax_e_benchmark_sweep_v2.py` | Corrected prefill and cached-decode benchmark |
| `jax_engine.py` | Stateful generation engine |
| `jax_openai_server.py` | OpenAI-compatible completions, chat, SSE, health, and metrics |
| `tests/` | CPU correctness and API regression suite |
| `benchmarks/runs/` | Raw logs, JSON, scripts, reports, and correction notes |
| `benchmarks/queued/` | Hardware profiling and queued benchmark utilities |

## Run the engine

Provision the default target and confirm JAX sees the chip — via the MCP agent,
`create_tpu_vm_instance(workload="jax")` then `wait_for_jax_ready` and
`verify_jax_tpu`, or the equivalent raw `gcloud` in [deploy.md](deploy.md). That
gives a TPU VM with a current `jax[tpu]` installation and no docker. With
authenticated access to the gated Gemma checkpoint, install the HTTP and
checkpoint dependencies, then:

```bash
python3 -m pip install fastapi uvicorn pydantic huggingface_hub safetensors \
  flax transformers sentencepiece 'jinja2>=3.1'

python3 jax_openai_server.py \
  --model google/gemma-4-E2B-it-qat-w4a16-ct \
  --kv-cache-dtype int8 \
  --quant-mode w4a16 \
  --max-model-len 8192 \
  --port 8000
```

Query the streaming endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-E2B-it-qat-w4a16-ct",
    "messages": [{"role": "user", "content": "Explain TPU buffer donation."}],
    "stream": true
  }'
```

Run the corrected kernel sweep:

```bash
python3 ports/gemma4/jax_e_benchmark_sweep_v2.py \
  --batch-sizes 1,2,4,8,16,32,64 \
  --contexts 8,128,512,2048 \
  --json-out results.json
```

Run CPU correctness tests:

```bash
python3 -m unittest discover -s tests
```

CPU is suitable for numerical correctness, scheduling, endpoint, and SSE tests.
It cannot validate TPU throughput, HBM capacity, compilation timing, or
Pallas/Mosaic performance.

## Current limitations

- No continuous batching or prefix cache.
- Static `(batch, sequence)` shapes can trigger first-touch compilation.
- Chunked prefill improves admission but does not yet compose with ring-buffer
  windowing.
- Greedy decoding is implemented; production grammar-constrained tool output is
  not.
- The Pallas W4A16 experiment regressed performance and is not the recommended
  path.
- Results describe this implementation and workload on a v6e-1, not the hardware
  limit of that chip.

## Supporting TPU infrastructure

The repository also carries the `tpu-jax-v6e1-2b-management` skill and `tpu-jax-v6e1-2b` MCP
server used to provision flex-start TPU VMs, verify JAX devices, probe zones for real
capacity, inspect logs, run vLLM baselines, and clean up capacity. It provisions
**through Compute Engine only** — there is no queued-resource path — and defaults to
this repo's target (`ACCELERATOR_TYPE=v6e-1`, `TENSOR_PARALLEL_SIZE=1`,
`GOOGLE_CLOUD_ZONE=europe-west4-a`, `MODEL_NAME=google/gemma-4-E2B-it`,
`PROVISIONING_MODEL=flex-start`), overridable per env var:

```bash
./project-setup.sh /path/to/project --project <gcp-project-id>
make skill
make skill-install
```

See [SKILL.md](.claude/skills/tpu-jax-v6e1-2b-management/SKILL.md) for the infrastructure
tool catalog. Root sources (`server.py`, `project-setup.sh`, and `tpu.md`) remain
authoritative; generated skill snapshots are refreshed with `make skill`.

## Results and writing

- [Corrected TPU report](benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md)
- [Real-weight HTTP report](benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md)
- [vLLM baseline](benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json)
- [Long-form article draft](devto-jax-gemma4-e2b.md)
- [Hugging Face benchmark dataset](https://huggingface.co/datasets/xbill9/gemma4-e2b-tpu-v6e-benchmarks)

The Hugging Face dataset is currently private. It contains reports, raw JSON,
benchmark scripts, and the article, but no model weights or credentials.

## Security

Never commit Hugging Face tokens, GCP credentials, model caches, or generated
server logs. Use scoped credentials and Google Secret Manager for remote TPU
deployments. The upstream Gemma checkpoint remains governed by its own access
and license terms.
