# gemma4-optb — Gemma-4-E2B-it on AWS Inferentia2

Prebuilt, ready-to-run image that serves **`google/gemma-4-E2B-it`** on a single **AWS
Inferentia2** device (`inf2.8xlarge`) at **~44 tokens/sec**, via an OpenAI-compatible HTTP
server. The compiled Neuron artifacts (neffs) + runtime + server are baked in — no
compilation or Python setup required.

This is a runnable inference port, not a fine-tune. What's novel is the compilation recipe:
it expresses Gemma-4's cross-layer KV-sharing on Neuron, which AWS's own NxD /
optimum-neuron stack cannot currently do.

## Quick start (on an AWS inf2 instance)

```bash
docker pull xbill9/gemma4-optb:latest
docker run --rm -p 8080:8080 --device=/dev/neuron0 xbill9/gemma4-optb:latest
# health:
curl -s localhost:8080/health
# generate:
curl -s -X POST localhost:8080/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"What is AWS Inferentia?","max_tokens":64}'
```

Routes: `/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`
(no auth — put it behind your own gateway before exposing it).

## Details

| | |
|---|---|
| Tags | `latest`, `512-128` |
| Context | max_total_tokens 512, max_prompt_tokens 128 |
| Precision | bf16 |
| Throughput | ~44 tok/s (~23 ms/token) |
| Warmup | ~100 s to load neffs, then instant |
| Hardware | AWS Inferentia2 (`inf2`), Neuron runtime |
| License | Apache-2.0 (base model © Google) |

## Recipe, neffs, and full model card

🤗 **https://huggingface.co/xbill9/gemma-4-E2B-it-inferentia2** — the compilation scripts,
the raw neffs, the two-graph KV-cache writeup, and how to recompile for a different context
window.

*Not affiliated with or endorsed by Google or AWS. Provided for research/testing.*
