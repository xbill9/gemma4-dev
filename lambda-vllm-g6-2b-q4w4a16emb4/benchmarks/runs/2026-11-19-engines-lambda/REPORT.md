# 2026-11-19 — five serving engines as AWS Lambda GPU functions (Lambda Managed Instances, one NVIDIA L4)

The G6 serving-engine flight (`gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-05-engines-g6`) as Lambda
functions: each engine in its own container image, one execution environment on one L4, reached with a
synchronous `Invoke` from a laptop. One engine ran at a time; each was scaled to 0 before the next started.

| rig | engine | image base | Init (engine healthy) |
|---|---|---|---|
| `lambda-vllm-g6-2b-q4w4a16emb4` | vLLM 0.33.0, W4A16 (run 2026-11-18) | AWS Lambda vLLM | see that rig |
| `lambda-sglang-g6-2b-q4w4a16emb4` | SGLang 0.5.23 + 2 patches, W4A16 | AWS Lambda SGLang | 347–372 s |
| `lambda-llamacpp-g6-2b-q4_0` | llama.cpp `7182ef71e`, exact Q4_0, drafter at 2 tokens | AWS Lambda base | publish to Active 696 s |
| `lambda-pytorch-g6-2b-q4w4a16` | the hand-built int4 engine, torch 2.14.1 | AWS Lambda PyTorch | 198 s |
| `lambda-jax-g6-2b` | the pure-JAX port, bf16, jax 0.11.1 | AWS Lambda base | 42 s |

Requests (`bench.py` in each rig): one prompt with a unique nonce, greedy, thinking off, 256 tokens with
`ignore_eos`; 1 request, then 4 at once; four runs per rig (`run1`–`run4`). Output tokens per second over the
wall time of each level, measured on the client, so the Lambda invoke round trip is included.

## Output tok/s, medians of four runs

| engine | 1 request | 4 at once | native G6, 1 / 4 |
|---|---|---|---|
| vLLM, drafter at 4 (2026-11-18, 3 / 2 runs) | 212.8 | 554.8 | 275.3 / 711.3 |
| vLLM, no drafter (2026-11-18, 3 / 2 runs) | 129.9 | 495.2 | 140.4 / 537.2 |
| llama.cpp, drafter at 2 | 177.4 | 405.7 | 207.7 / 373.2 |
| SGLang | 125.3 | 458.5 | 136.5 / 475.4 |
| PyTorch int4 | 127.0 | 138.1 | 134.7 / 134.5 |
| JAX bf16 | 47.9 | 54.6 | 48.3 / – |

Every run: llama.cpp 169.1 / 192.6 / 183.5 / 171.2 and 382.5 / 409.4 / 406.4 / 405.0; SGLang 124.8 / 125.1 / 126.7 /
125.5 and 170.7 / 461.7 / 455.2 / 472.0; PyTorch 107.7 / 128.8 / 126.2 / 127.8 and 135.5 / 138.3 / 138.1 / 138.1;
JAX 6.7 / 47.9 / 48.0 / 47.8 and 46.2 / 54.7 / 54.6 / 54.6. The first JAX request compiled (XLA's cache is in
`/tmp`, empty in a new environment).

#### Findings

- **Every engine serves on Lambda at its native G6 speed less the invoke round trip**: one request reaches
  0.92–0.99x of the native single-stream number without a drafter (SGLang 0.92, PyTorch 0.94, JAX 0.99).
- **The native ranking holds**: vLLM with its drafter is fastest for one request, then llama.cpp with its drafter;
  vLLM, SGLang and llama.cpp batch 4 requests at 3.2–3.8x one; PyTorch and JAX serve one at a time.
- **Start-up is minutes, set by the engine**: JAX 42 s and PyTorch 198 s to a healthy server, SGLang 347–372 s.
  Lambda can report a publish as Failed (internal error) while a slow engine is still in Init; the version
  turns Active by itself once Init succeeds.

## Setup

- Capacity provider `gemma4-e2b-g6-l4` (g6.2xlarge or g6.4xlarge, one L4 either way) for SGLang, PyTorch and
  JAX; llama.cpp ran on `gemma4-e2b-g6` (g6.2xlarge only). Lambda had no g6.2xlarge in any us-east-1 zone
  17:01–17:38 UTC; the instance type behind each run is not visible from the account.
- Images built on one c7i.4xlarge (`builder.py`) and pushed to ECR, 458–612 s each.
- `sweep.sh` and `sweep-pytorch-jax.log` here are the driver and its output for PyTorch and JAX.
