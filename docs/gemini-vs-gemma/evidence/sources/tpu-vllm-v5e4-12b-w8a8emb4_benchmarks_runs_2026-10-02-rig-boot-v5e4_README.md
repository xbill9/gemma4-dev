# 2026-10-02-rig-boot-v5e4

`tpu-vllm-v5e4-12b-w8a8emb4` booted through its own code path: `create_tpu_queued_resource` in `server.py`, flex-start,
`v5litepod-4` in `us-west4-a`, runtime `v2-alpha-tpuv5-lite`, `--tensor-parallel-size 4`, serving `xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4`.
Queued Resource created 2026-10-02T18:16:59.956032967Z. Capacity was granted within 5 minutes.

| | |
|---|---|
| Weights per chip | `[(3.25, 15.75), (3.25, 15.75), (3.25, 15.75), (3.25, 15.75)]` GiB (v5e-1: 11.31 GiB on one chip) |
| KV cache | 122,624 tokens (v5e-1: 9,728, so 12.61x) |

#### Verification

- `/v1/models` lists `xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4` (`verify-models.json`).
- Chat, "What is the capital of Australia? Answer in one sentence.", greedy: `The capital of Australia is Canberra.`.
- Tool call, one `get_weather(city)` tool: `finish_reason` `tool_calls`, `get_weather({"city": "Paris"})`.

#### Throughput

`../../../../jev-tpu-v5e1/tpu/w4a16_client.py load` on the VM against `localhost:8000`: N parallel chat requests of
exactly 256 output tokens (`ignore_eos`, greedy), one untimed warm-up pass, then three timed passes; median.
The v5e-1 column is the v5e-1 rig's own boot, `2026-09-30-rig-boot-v5e1`, same client and method.

| Requests | v5e-1 | v5e-4, TP=4 | v5e-4 / v5e-1 |
|---:|---:|---:|---:|
| 1 | 57.0 | **128.4** | 2.25x |
| 4 | 215.7 | **450.9** | 2.09x |
| 16 | 713.4 | **1,212.9** | 1.70x |

#### Files

| File | What |
|---|---|
| `startup.log` | the boot script's output (`/var/log/vllm-startup.log`) |
| `vllm-container.log` | `docker logs vllm-gemma4`, HTTP access lines removed, addresses redacted |
| `queued-resource.json` | the Queued Resource as described at collection time |
| `load.vm.c{1,4,16}.json` | load client on the VM |
| `verify-*.json` | the three verification responses |
