# Stock Gemma 4 E2B vs the emb4 repack on SageMaker

- **Question:** on the same SageMaker endpoint, is `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` better than stock `google/gemma-4-E2B-it`?
- **Answer:** same answers, 2.39x the tokens per second. Both scored 20 of 20 on demo 1 in both modes; the repack served 108.0 tok/s against 45.2.

## Setup

Both endpoints were identical except for `MODEL_ID`:

- `ml.g6.xlarge` (one NVIDIA L4), us-east-2
- image `763104351884.dkr.ecr.us-east-2.amazonaws.com/vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.4`
- `SM_VLLM_ENABLE_AUTO_TOOL_CHOICE=true`, `SM_VLLM_TOOL_CALL_PARSER=gemma4`, max model length 8192, GPU memory 0.9
- client: this laptop, temperature 0

The us-east-2 quota is one `ml.g6.xlarge` endpoint, so they ran one after the other on 2026-10-08: stock 14:39 to 14:41 UTC, repack 14:50 to 14:51 UTC. `run.sh` holds the exact commands.

## Results

| | stock `google/gemma-4-E2B-it` | repack `-text-emb4` |
|---|---|---|
| launch to InService | 10.2 min | 8.4 min or less |
| chat, median tok/s (10 requests, 100-token cap) | 45.2 (41.4–46.5) | 108.0 (106.0–110.0) |
| chat, median seconds | 2.148 (97 tokens) | 0.777 (84 tokens) |
| demo 1, engine counts: correct / right filter | 20/20, 20/20 | 20/20, 20/20 |
| demo 1, engine counts: median s per run | 1.92 | 0.52 |
| demo 1, model counts rows: correct | 20/20 | 20/20 |
| demo 1, model counts rows: median s per run | 1.42 | 1.05 |

Every filter sent in engine mode was `id >= 10`. Raw runs are in the `*.json` files beside this report.

## Scope

One run per model, ten timed chat requests and twenty agent runs per mode, on one GPU type in one region. The stock checkpoint is the multimodal bf16 model, with its vision and audio towers loaded. The repack is text-only and 4-bit throughout. At temperature 0 the two wrote different paragraphs, 97 and 84 tokens, neither at the 100-token cap. Demo 1 is one eleven-id question, so it shows that the repack calls tools and answers it as well as stock, not a broad quality comparison. The repack's InService time is an upper bound: the endpoint was already in service at the first check, 8.4 minutes after deploy.
