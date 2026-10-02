# 2026-10-02-longctx-v5e4

What four v5e chips buy the 12B for long context. `tpu-vllm-v5e4-12b-w8a8emb4` booted once through `create_tpu_queued_resource`
(flex-start `v5litepod-4`, `us-west4-a`, Queued Resource created 2026-10-02T21:57Z, ACTIVE 22:01Z, delete issued 23:37Z). On that
VM the serving container was then re-created with the boot script's image (`vllm-tpu-w8a8emb4:patched`), environment and
arguments, changing only `--max-model-len` (`relaunch.py`; each `len*/docker-run.txt` is the exact command, token masked):
`--tensor-parallel-size 4 --max_num_batched_tokens 512 --gpu-memory-utilization 0.92`, `MIN_TOKEN_BUCKET=64`.
`len8192/` is the boot script's own container.

The model allows 262,144 positions (`max_position_embeddings`, `config.json`; sliding window 1024). The KV pool sets the
limit instead: at 131,072 vLLM refused to start, "48.0 GiB KV cache is needed, which is larger than the available KV cache
memory (44.85 GiB) ... the estimated maximum model length is 122368" (`len131072/vllm-container.log`). 122,368 served.

#### KV pool and boot

| MAX_MODEL_LEN | Boot | KV cache (tokens) | Full-length requests the pool holds |
|---:|---|---:|---:|
| 8,192 | served | 122,624 | 14.97 |
| 32,768 | served | 122,688 | 3.74 |
| 65,536 | served | 122,624 | 1.87 |
| 122,368 | served | 122,624 | 1.00 |
| 131,072 | failed | — | — |

vLLM sizes the pool by memory, so it stays at ~122,600 tokens whatever the maximum length; a larger maximum only changes how
many full-length requests fit. At 122,368 one full-length request takes the whole pool.

#### Long prompts

`../../../../jev-tpu-v5e1/tpu/w4a16_client.py loadlong` on the VM against `localhost:8000`: N parallel streamed chat
requests, each prompt unique so the prefix cache never serves it, 256 output tokens (`ignore_eos`, greedy), one untimed
warm-up pass, three timed passes, median. Output tok/s counts only generated tokens over the wall time, so it includes
prefill; total tok/s counts prompt plus output. Prompt length is calibrated by the client and lands 2-5% above the target
(`target_input_tokens` in each JSON). At 8,192 a prompt of 8,192 plus 256 output tokens exceeds the limit, so the
nearest feasible length, 7,876, stands in for it; 7,876 was measured at every setting so the settings compare directly.

| MAX_MODEL_LEN | Prompt tokens | Requests | Output tok/s | Total tok/s | Median TTFT (s) |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 3,611 | 1 | 110.5 | 1,668.4 | 0.227 |
| 8,192 | 3,611 | 16 | 557.4 | 8,420.6 | 2.021 |
| 8,192 | 6,279 | 1 | 99.7 | 2,546.1 | 0.391 |
| 8,192 | 6,279 | 16 | 363.1 | 9,270.0 | 3.709 |
| 8,192 | 7,876 | 1 | 94.4 | 2,998.7 | 0.485 |
| 8,192 | 7,876 | 16 | 293.7 | 9,329.0 | 4.765 |
| 32,768 | 7,876 | 1 | 85.9 | 2,730.2 | 0.501 |
| 32,768 | 7,876 | 16 | 282.9 | 8,986.6 | 4.899 |
| 32,768 | 8,392 | 1 | 82.4 | 2,784.4 | 0.534 |
| 32,768 | 8,392 | 16 | 246.6 | 8,330.3 | 5.452 |
| 32,768 | 25,375 | 1 | 59.5 | 5,956.8 | 1.671 |
| 32,768 | 25,375 | 16 | 108.1 | 10,828.2 | 19.727 |
| 65,536 | 7,876 | 1 | 85.9 | 2,729.0 | 0.5 |
| 65,536 | 7,876 | 16 | 282.9 | 8,986.6 | 4.898 |
| 65,536 | 8,392 | 1 | 82.6 | 2,791.4 | 0.532 |
| 65,536 | 8,392 | 16 | 248.9 | 8,407.8 | 5.432 |
| 65,536 | 51,135 | 1 | 42.1 | 8,445.0 | 3.566 |
| 65,536 | 51,135 | 16 | 56.2 | 11,285.7 | 39.564 |
| 122,368 | 7,876 | 1 | 86.5 | 2,747.4 | 0.499 |
| 122,368 | 7,876 | 16 | 283.6 | 9,010.1 | 4.89 |
| 122,368 | 8,392 | 1 | 83.2 | 2,810.1 | 0.531 |
| 122,368 | 8,392 | 16 | 249.4 | 8,426.2 | 5.427 |
| 122,368 | 94,616 | 1 | 25.4 | 9,396.1 | 7.33 |
| 122,368 | 94,616 | 16 | 25.8 | 9,566.7 | 86.562 |

A 94,616-token prompt is prefilled in 7.3 s on one request (9,396 total tok/s). At 16 such requests the pool holds one at a
time: total throughput stays ~9,500 tok/s and the median request waits 86.6 s for its first token. At 51,135 tokens two
fit and the median wait is 39.6 s; at 25,375 tokens four fit and it is 19.7 s.

#### Short requests

`w4a16_client.py load`: N parallel chats, 256 output tokens, same method.

| MAX_MODEL_LEN | Requests | Output tok/s (min-max) |
|---:|---:|---:|
| 8,192 | 16 | 1,213.4 (1,195.9-1,216.2) |
| 8,192 | 64 | 2,129.3 (2,129.2-2,129.8) |
| 32,768 | 16 | 1,094.1 (1,088.9-1,094.1) |
| 32,768 | 64 | 1,931.0 (1,929.3-1,941.9) |
| 65,536 | 16 | 1,118.9 (1,118.8-1,119.0) |
| 65,536 | 64 | 2,017.4 (1,992.2-2,023.9) |
| 122,368 | 16 | 1,135.1 (1,134.9-1,136.5) |
| 122,368 | 64 | 2,066.6 (2,046.2-2,069.8) |

#### Cost of a larger maximum at the same prompt

At 7,876-token prompts, raising the maximum from 8,192 to any of 32,768 / 65,536 / 122,368 costs about 9% at one request
(94.4 -> 85.9-86.5 output tok/s) and about 4% at 16 (293.7 -> 282.9-283.6); short-request throughput drops 6.5-9.8% at
16 and 2.9-9.3% at 64. On long prompts the three raised settings are within 1% of each other; on short requests they
spread by up to 7%, and 122,368 is the fastest of the three on both. The cost is paid on leaving 8,192 and does not grow
with the maximum. The logged KV block size differs by setting (256, 64,
128, 256), so the block size alone does not explain the drop.

#### Against v5e-1

The v5e-1 rig served at `--max-model-len 4096` with a 9,728-token pool; its longest measured prompt is 3,611 tokens
(`../../../tpu-vllm-v5e1-12b-w8a8emb4/benchmarks/runs/2026-09-30-long12b-12b-w8a8-emb4-v5e1/logs/2026-09-30-long12b-v5e1-logs/12b-w8a8-emb4.long3584.c*.json`),
same client and method. On v5e-1, 16 such requests exceed the pool (16 x 3,867 tokens against 9,728) and queue; on v5e-4
they all fit.

| Prompt tokens | Requests | v5e-1 output tok/s | v5e-4 output tok/s | v5e-4 / v5e-1 | v5e-1 TTFT (s) | v5e-4 TTFT (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 3,611 (v5e-1 3,611) | 1 | 51.5 | 110.5 | 2.15x | 0.352 | 0.227 |
| 3,611 (v5e-1 3,611) | 16 | 94.7 | 557.4 | 5.89x | 21.96 | 2.021 |

v5e-1 has no measurement above 3,611 tokens; everything from 6,279 tokens up in this run is a context v5e-1 cannot serve.

#### Boot time

From each container's log. The first three starts ran without a compile cache. From the 131,072 attempt on, the container
mounted a persistent XLA cache (`-e VLLM_XLA_CACHE_PATH=/xla-cache -v /opt/xla-cache:/xla-cache`, the method in
`jev-tpu-31b/tpu/serve.sh`). The 131,072 start failed at KV sizing, before compiling anything, so the first 122,368 start
filled the cache (61 MB, 26 entries) and the second, `len122368-warm/`, read it. Engine init to KV sized is weight loading
and memory profiling; KV sized to ready is compilation and warm-up.

| Start | MAX_MODEL_LEN | XLA cache | KV page size | Engine init to KV sized (s) | KV sized to ready (s) | Sum of logged compile steps (s) | First log line to ready (s) |
|---|---:|---|---:|---:|---:|---:|---:|
| `len8192/vllm-container.log` | 8,192 | none | 256 | 110 | 542 | 526 | 711 |
| `len32768/vllm-container.log` | 32,768 | none | 64 | 97 | 555 | 540 | 709 |
| `len65536/vllm-container.log` | 65,536 | none | 128 | 99 | 552 | 538 | 708 |
| `len131072/vllm-container.log` | 131,072 | mounted, empty for these shapes | 256 | — | — | 0 | — |
| `len122368/vllm-container.log` | 122,368 | mounted, empty for these shapes | 256 | 96 | 547 | 530 | 699 |
| `len122368-warm/vllm-container.log` | 122,368 | warm (same shapes compiled once before) | 256 | 96 | 61 | 46 | 214 |

A warm cache cut the compile phase from 547 s to 61 s and the whole container start from 699 s to 214 s. The warm
start served chat ("The capital of Australia is Canberra.") and a tool call (`get_weather({"city": "Paris"})`,
`finish_reason` `tool_calls`) (`len122368-warm/verify-*.json`).

#### Recommendation

**`MAX_MODEL_LEN=122368`** for this rig on v5e-4: the largest value that boots at `GPU_MEMORY_UTILIZATION=0.92`, at the same
throughput as 32,768 and 65,536 or better, and 14.9x the current limit. Its cost against 8,192, on work 8,192 can already
serve, is 8.4% at one 7,876-token request, 3.4% at 16, and 6.5% / 2.9% at 16 / 64 short requests. Keep 8,192 only for traffic that is all short. `tpu.env` and `server.py` are unchanged.

#### Files

| File | What |
|---|---|
| `startup.log` | the boot script's output (`/var/log/vllm-startup.log`), token lines removed |
| `queued-resource.json` | the Queued Resource as described after boot, addresses redacted |
| `len<N>/vllm-container.log` | `docker logs vllm-gemma4` for the start at `--max-model-len N`, HTTP access lines removed, addresses redacted |
| `len<N>/docker-run.txt` | the `docker run` command for that start, token masked (`len8192` is the boot script's, in `startup.log`) |
| `len<N>/loadlong.in<T>.c<C>.json` | `loadlong` at target prompt length T and C requests |
| `len<N>/load.c<C>.json` | `load` at C requests |
| `len<N>/progress.txt` | completion time and exit code of each client run |
| `len122368-warm/` | the warm-cache restart: log, command, chat and tool-call verification |
| `relaunch.py` | re-creates the container from the boot script's, changing only `--max-model-len`, with the XLA cache mount |
| `measure.sh` | the client runs for one setting |
| `tables.py` | generates every table above from the JSON and logs (`python3 tables.py .`) |
