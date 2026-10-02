# 2026-10-02-rig-boot-v5e4

`tpu-vllm-v5e4-2b-w8a8` booted through its own code path: `create_tpu_queued_resource` in `server.py`, flex-start,
`v5litepod-4` in `us-west4-a`, runtime `v2-alpha-tpuv5-lite`, `--tensor-parallel-size 4`, serving `xbill9/gemma-4-E2B-it-qat-w8a8-int8`.
Queued Resource created 2026-10-02T18:18:34.629194250Z. Capacity was granted within 5 minutes.

| | |
|---|---|
| Weights per chip | `[(1.75, 15.75), (1.75, 15.75), (1.75, 15.75), (1.75, 15.75)]` GiB (v5e-1: 6.88 GiB on one chip) |
| KV cache | 631,968 tokens (v5e-1 rig: 333,344, so 1.90x) |

#### Verification

- `/v1/models` lists `xbill9/gemma-4-E2B-it-qat-w8a8-int8` (`verify-models.json`).
- Chat, "What is the capital of Australia? Answer in one sentence.", greedy: `The capital of Australia is Canberra.`.
- Tool call, one `get_weather(city)` tool: `finish_reason` `tool_calls`, `get_weather({"city": "Paris"})`.

#### Throughput

`../../../../jev-tpu-v5e1/tpu/w4a16_client.py load` on the VM against `localhost:8000`: N parallel chat requests of
exactly 256 output tokens (`ignore_eos`, greedy), one untimed warm-up pass, then three timed passes; median.
The v5e-1 column is the v5e-1 rig booted the same day through the same code path with identical serving arguments
(`../../../../tpu-vllm-v5e1-2b-w8a8/benchmarks/runs/2026-10-02-rig-boot-v5e1`). The sweep column is the `../jev-tpu-v5e1`
harness, which serves with `--max-model-len 2048`, `--max-num-seqs 16` and no parsers; those settings are worth about
2x at 16 requests on one chip, so it does not pair with either rig run.

| Requests | v5e-1 rig | v5e-4 rig, TP=4 | v5e-4 / v5e-1 | v5e-1 sweep (other settings) |
|---:|---:|---:|---:|---:|
| 1 | 174.7 | **190.3** | 1.09x | 220 |
| 4 | 581.1 | **643.5** | 1.11x | 841 |
| 16 | 1,386.0 | **1,464.5** | 1.06x | 2,872 |

KV cache: 631,968 tokens against 333,344 on the v5e-1 rig (1.90x).

#### Files

| File | What |
|---|---|
| `startup.log` | the boot script's output (`/var/log/vllm-startup.log`) |
| `vllm-container.log` | `docker logs vllm-gemma4`, HTTP access lines removed, addresses redacted |
| `queued-resource.json` | the Queued Resource as described at collection time |
| `load.vm.c{1,4,16}.json` | load client on the VM |
| `verify-*.json` | the three verification responses |
