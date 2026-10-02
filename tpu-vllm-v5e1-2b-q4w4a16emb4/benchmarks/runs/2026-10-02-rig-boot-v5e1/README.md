# 2026-10-02-rig-boot-v5e1

`tpu-vllm-v5e1-2b-q4w4a16emb4` booted through its own code path: `create_tpu_queued_resource` in `server.py`, flex-start,
`v5litepod-1` in `us-west4-a`, runtime `v2-alpha-tpuv5-lite`, `--tensor-parallel-size 1`, serving `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`.
Queued Resource created 2026-10-02T21:57:50.336431919Z.

This run is the same-method baseline for `../../../../tpu-vllm-v5e4-2b-q4w4a16emb4/benchmarks/runs/2026-10-02-rig-boot-v5e4`:
both rigs boot through the same `server.py` and startup script with the same serving arguments (`--max-model-len 16384`, `--max-num-batched-tokens 512`, `--gpu-memory-utilization 0.8`, `--max-num-seqs` at the vLLM default,
tool-call and reasoning parsers on) and differ only in chip count and tensor parallelism. The load client, its method and
the VM-local measurement point are the same.

| | |
|---|---|
| Weights | `[(2.84, 15.75)]` GiB |
| KV cache | 568,224 tokens (v5e-4 at TP=4: 620,512, so 1.09x) |

#### Verification

- `/v1/models` lists `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` (`verify-models.json`).
- Chat, "What is the capital of Australia? Answer in one sentence.", greedy: `The capital of Australia is Canberra.`.
- Tool call, one `get_weather(city)` tool: `finish_reason` `tool_calls`, `get_weather({"city": "Paris"})`.

#### Throughput, short prompts

`../../../../jev-tpu-v5e1/tpu/w4a16_client.py load` on the VM against `localhost:8000`: N parallel chat requests of
exactly 256 output tokens (`ignore_eos`, greedy), one untimed warm-up pass, then three timed passes; median output tok/s.
The v5e-4 column is `load.vm.c*.json` from the v5e-4 run above; the last column is the earlier `../../../../jev-tpu-v5e1` sweep
(2026-09-29, tables padded to 128 columns), which served with `--max-model-len 2048`, `--max-num-seqs 16` and no parsers.

| Requests | v5e-1, this run | v5e-4, TP=4 | v5e-4 / v5e-1 | sweep v5e-1 (other settings) |
|---:|---:|---:|---:|---:|
| 1 | **128.2** (128.2–128.3) | 187.2 | 1.46x | 150 |
| 4 | **446.0** (446.0–446.0) | 613.2 | 1.37x | 583 |
| 16 | **1,171.0** (1,169.4–1,171.5) | 1,424.0 | 1.22x | 2,067 |

Parentheses: slowest and fastest of the three timed passes.

#### Throughput, long prompts

`w4a16_client.py loadlong` with `JEV_LOAD_CONCURRENCY=16`: 16 parallel streamed requests, each with a unique prefix so
the prefix cache never serves a prompt, prompt length calibrated to `JEV_LOAD_INPUT_TOKENS`, 256 output tokens; one
untimed warm-up pass, three timed passes, median. The v5e-4 rig has no loadlong run in its 2026-10-02 record.

| Target input | Prompt tokens / request | Output tok/s | Total tok/s | Median TTFT, s |
|---:|---:|---:|---:|---:|
| 2,048 | 2,096 | **745.3** (745.2–745.4) | 6,848.2 | 1.084 |
| 8,192 | 8,388 | **332.4** (332.4–332.4) | 11,223.4 | 4.52 |

#### Files

| File | What |
|---|---|
| `startup.log` | the boot script's output (`/var/log/vllm-startup.log`) |
| `vllm-container.log` | `docker logs vllm-gemma4`, HTTP access lines removed, addresses redacted |
| `queued-resource.json` | the Queued Resource as described at collection time |
| `load.vm.c{1,4,16}.json` | load client on the VM |
| `loadlong.vm.c16.in{2048,8192}.json` | long-prompt load client on the VM, 16 requests |
| `verify-*.json` | the three verification responses |
