# 2026-09-29 — E2B QAT on a raw G6, against the same build on SageMaker

First serve of `gpu-vllm-g6-2b-w4a16`: `google/gemma-4-E2B-it-qat-w4a16-ct` on one on-demand
`g6.xlarge` (1× NVIDIA L4) in us-east-2, `vllm/vllm-openai:v0.30.0`, `max_model_len` 8192,
GPU memory utilization 0.90, vLLM's default `max_num_seqs` (`settings.txt`). Measured with
sagemaker-gemma's `compare.py` — the script that measured the SageMaker endpoint
`gemma-4-e2b-qat` (`ml.g6.xlarge`, vLLM 0.30.0, 2026-09-25) — run **on the instance against
localhost**, over Systems Manager. The security group has no inbound rules.

## Results (`measure-gemma-4-e2b-qat-g6.json`)

| Measure | SageMaker `ml.g6.xlarge` | Raw EC2 `g6.xlarge` | EC2 / SageMaker |
| --- | ---: | ---: | ---: |
| vLLM | 0.30.0 | 0.30.0 | – |
| Weights in GPU memory (GiB) | 8.01 | 8.01 | – |
| KV cache (tokens) | 867,999 | 873,318 | – |
| Weight load (s) | 66.2 | 45.8 | – |
| Decode, one request (tokens/s) | 105.1 | 105.1 | 1.00 |
| Per-call fixed cost (s) | 0.562 | 0.008 | – |
| 1 request, 256 tokens (tokens/s) | 85.35 | 104.85 | 1.23 |
| 4 parallel (tokens/s) | 328.7 | 410.85 | 1.25 |
| 16 parallel (tokens/s) | 1077.25 | 1485.9 | 1.38 |
| Questions correct (of 40) | 37 | 37 | – |
| Launch → serving (min) | 10.1 | 8.1 | – |

- The model server is the same on both: identical weights, decode rate, and all 40 answers
  byte-identical. The KV cache differs by 0.6 %.
- The per-call fixed cost is where they part. SageMaker's includes starting the `aws` CLI,
  signing the request and the round trip from the client; here the client is on the instance.
  That cost is what separates the 1- to 16-parallel rates, so the 1.23–1.38× measures the
  invocation path, not the GPU. A client off the instance would pay its own network cost; this
  run does not measure that.
- vLLM's own "Avg generation throughput" peaked at 409.6 tokens/s on both; those lines are
  10-second averages over batches of 2–4 s and do not resolve the difference.

## Cost

On-demand hosting, us-east-2 (AWS Pricing API, 2026-09-29): `g6.xlarge` $0.8048/h,
`ml.g6.xlarge` $1.1267/h.

| | SageMaker | Raw EC2 |
| --- | ---: | ---: |
| $ per hour | 1.1267 | 0.8048 |
| $ per million tokens at the 16-parallel rate measured on each | 0.291 | 0.150 |
| $ per million tokens at SageMaker's measured rate | 0.291 | 0.208 |

The second row mixes the price difference with the invocation difference; the third holds
throughput equal and shows the price alone (0.71×).

## Timeline (`timeline.txt`)

Launched 19:43:47Z, image pull 19:45:22–19:46:59Z, vLLM healthy 19:51:51Z, measured
19:51:56–19:54:17Z, terminate called 19:54:18Z, state `terminated` 20:00:51Z. Running
time to the terminate call 10.5 min, about $0.14.

## Files

- `settings.txt` — the rig values and exact serve flags
- `measure-gemma-4-e2b-qat-g6.json`, `.log` — compare.py output
- `vllm-engine.log`, `bootstrap.log` — container and cloud-init logs
- `driver.log`, `timeline.txt`, `instance-id.txt`, `watchdog.sh`, `watchdog.log`
- `../../ec2_measure.py` — the driver
