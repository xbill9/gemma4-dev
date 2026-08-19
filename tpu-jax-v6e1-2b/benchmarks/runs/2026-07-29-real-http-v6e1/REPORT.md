# Real-checkpoint HTTP validation — Gemma 4 E2B QAT, TPU v6e-1

**Date:** 2026-07-29 · **Hardware:** one `ct6e-standard-1t` in
`europe-west4-a` · **Model:** `google/gemma-4-E2B-it-qat-w4a16-ct`

## Configuration

- Current `jax_openai_server.py` and `JaxGemmaEngine`
- Real checkpoint weights, W4A16 reference path
- int8 KV cache, buffer donation enabled
- `max_model_len=8192`, greedy decoding, 32 requested output tokens
- Requests sent to the OpenAI-compatible `/v1/completions` endpoint

The test server was staged separately under `/home/xbill/gemma-e2e-test`; the
existing VM checkout was not modified. First-touch static-bucket compilation took
41.8–47.3 seconds. Results below are warm runs after compilation.

Prompts repeat a fixed technical sentence to reach the target token counts. This
is a latency and scheduling test, not a generation-quality evaluation.

## Single-stream context scaling

Median of three warm requests:

| Prompt tokens | HTTP wall time | Prefill | Decode |
|---:|---:|---:|---:|
| 506 | 240.5 ms | 9.3 ms | 141.1 tok/s |
| 2,045 | 267.7 ms | 32.0 ms | 140.5 tok/s |
| 7,679 | 570.9 ms | 318.6 ms | 138.5 tok/s |

Decode throughput changes by only 1.8% from 506 to 7,679 prompt tokens. The
long-context latency increase is overwhelmingly prefill.

## Concurrent HTTP requests

Two repetitions per point, each request using 2,045 prompt tokens and generating
32 tokens:

| Concurrent requests | Success | Batch wall time | Aggregate | Median request | Effective per stream |
|---:|:---:|---:|---:|---:|---:|
| 2 | 2/2 | 497.3 ms | 128.7 tok/s | 496.5 ms | 64.4 tok/s |
| 4 | 4/4 | 957.9 ms | 133.6 tok/s | 952.1 ms | 33.6 tok/s |
| 8 | 8/8 | 1,786.2 ms | 143.3 tok/s | 1,774.9 ms | 18.0 tok/s |

All requests succeeded, but aggregate throughput remains near the B=1 rate while
request latency grows almost linearly with concurrency. This is expected from the
current implementation: `generate_stream` is explicitly single-sequence and the
FastAPI server launches independent B=1 executions. Concurrent HTTP calls queue
device work; they do not form a JAX batch.

## Conclusion

The real-weight OpenAI path is correct and stable through a 7.7K prompt, but it
does **not** realize the synthetic batched-kernel results. Production agent
throughput requires a real request batcher, batched KV ownership, continuous
admission, and prefix reuse. Until those exist, the defensible serving figure is
approximately **139–141 tok/s aggregate**, not 2,888 tok/s.

Raw files: `http_b1_results.json`, `http_b1_metrics.json`,
`http_concurrency_results.json`, and `server.log`.
