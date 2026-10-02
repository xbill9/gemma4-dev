# 2026-10-02-rig-boot-v5e1

`tpu-vllm-v5e1-2b-w8a8` booted through its own code path: `create_tpu_queued_resource` in `server.py`, flex-start,
`v5litepod-1` in `us-west4-a`, runtime `v2-alpha-tpuv5-lite`, `--tensor-parallel-size 1`, serving `xbill9/gemma-4-E2B-it-qat-w8a8-int8`.
Queued Resource created 2026-10-02T21:57:29.352194899Z.

This run is the same-method baseline for `../../../../tpu-vllm-v5e4-2b-w8a8/benchmarks/runs/2026-10-02-rig-boot-v5e4`:
both rigs boot through the same `server.py` and startup script with the same serving arguments (`--max-model-len 16384`, `--max-num-batched-tokens 512`, `--gpu-memory-utilization 0.8`, `--max-num-seqs` at the vLLM default,
tool-call and reasoning parsers on) and differ only in chip count and tensor parallelism. The load client, its method and
the VM-local measurement point are the same.

| | |
|---|---|
| Weights | `[(6.88, 15.75)]` GiB |
| KV cache | 333,344 tokens (v5e-4 at TP=4: 631,968, so 1.90x) |

#### Verification

- `/v1/models` lists `xbill9/gemma-4-E2B-it-qat-w8a8-int8` (`verify-models.json`).
- Chat, "What is the capital of Australia? Answer in one sentence.", greedy: `The capital of Australia is Canberra.`.
- Tool call, one `get_weather(city)` tool: `finish_reason` `tool_calls`, `get_weather({"city": "Paris"})`.

#### Throughput, short prompts

`../../../../jev-tpu-v5e1/tpu/w4a16_client.py load` on the VM against `localhost:8000`: N parallel chat requests of
exactly 256 output tokens (`ignore_eos`, greedy), one untimed warm-up pass, then three timed passes; median output tok/s.
The v5e-4 column is `load.vm.c*.json` from the v5e-4 run above; the last column is the earlier `../../../../jev-tpu-v5e1` sweep
(`2026-09-29-qatw8-e2b-qat-w8a8-v5e1`), which served with `--max-model-len 2048`, `--max-num-seqs 16` and no parsers.

| Requests | v5e-1, this run | v5e-4, TP=4 | v5e-4 / v5e-1 | sweep v5e-1 (other settings) |
|---:|---:|---:|---:|---:|
| 1 | **174.7** (174.7–174.7) | 190.3 | 1.09x | 220 |
| 4 | **581.1** (579.7–581.1) | 643.5 | 1.11x | 841 |
| 16 | **1,386.0** (1,360.3–1,386.8) | 1,464.5 | 1.06x | 2,872 |

Parentheses: slowest and fastest of the three timed passes.

#### Throughput, long prompts

`w4a16_client.py loadlong` with `JEV_LOAD_CONCURRENCY=16`: 16 parallel streamed requests, each with a unique prefix so
the prefix cache never serves a prompt, prompt length calibrated to `JEV_LOAD_INPUT_TOKENS`, 256 output tokens; one
untimed warm-up pass, three timed passes, median. The v5e-4 rig has no loadlong run in its 2026-10-02 record.

| Target input | Prompt tokens / request | Output tok/s | Total tok/s | Median TTFT, s |
|---:|---:|---:|---:|---:|
| 2,048 | 2,096 | **1,006.8** (1,006.7–1,007.0) | 9,251.8 | 0.57 |
| 8,192 | 8,388 | **498.9** (498.7–499.0) | 16,846.9 | 2.513 |

#### Files

| File | What |
|---|---|
| `startup.log` | the boot script's output (`/var/log/vllm-startup.log`) |
| `vllm-container.log` | `docker logs vllm-gemma4`, HTTP access lines removed, addresses redacted |
| `queued-resource.json` | the Queued Resource as described at collection time |
| `load.vm.c{1,4,16}.json` | load client on the VM |
| `loadlong.vm.c16.in{2048,8192}.json` | long-prompt load client on the VM, 16 requests |
| `verify-*.json` | the three verification responses |
