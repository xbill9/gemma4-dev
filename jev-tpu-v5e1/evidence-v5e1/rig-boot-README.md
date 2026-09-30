# 2026-09-30-rig-boot-v5e1

The first boot of this rig through its own code path: `create_tpu_queued_resource` in `server.py` rendered
`startup_script_template.sh` with `patches/` embedded, the VM built `vllm-tpu-w8a8emb4:patched` from the pinned
`vllm/vllm-tpu@sha256:19a1a052…`, and served `xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4` from Hugging Face.
The endpoint was found by `_discover_vllm_node()`.

| | |
|---|---|
| Queued Resource | `tpu-vllm-v5e1-12b-w8a8emb4`, node `tpu-vllm-v5e1-12b-w8a8emb4-node` |
| Zone, shape, model | `us-west4-a`, `v5litepod-1`, flex-start, runtime `v2-alpha-tpuv5-lite` |
| Serving settings (`tpu.env`, `server.py`) | `TENSOR_PARALLEL_SIZE=1`, `GPU_MEMORY_UTILIZATION=0.92`, `MAX_MODEL_LEN=8192`, `MAX_NUM_BATCHED_TOKENS=512`, `LIMIT_MM_PER_PROMPT={"image":0,"audio":0,"video":0}`, `MIN_TOKEN_BUCKET=64`, `GMM_V2_TILE_VMEM_FRACTION=0.85`, `--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4` |
| Patches applied on the VM | `kvshare.diff`, `wna16.diff`, `moe.diff`, `lowmem.diff`, `textonly.diff` (all five, `startup-excerpt.log`) |
| Weights on the chip | 11.31 GiB of 15.75 (`Init model | hbm=[(11.31, 15.75)]GiB`) |
| KV cache | 9,728 tokens, 38 blocks, 1.19x concurrency at 8,192 tokens |
| Checkpoint download | 28 s (11.21 GiB), weight load 52 s |
| Engine init | 590 s, of which compile 512 s |

#### Timeline (UTC)

| Event | Time | From create |
|---|---|---:|
| Queued Resource created | 15:42:16 | 0:00 |
| VM kernel boot | 15:43:58 | 1:42 |
| Container started (image pulled and patched) | 15:46:56 | 4:40 |
| Weights resident | 15:49:39 | 7:23 |
| `Application startup complete.` | 15:59:35 | 17:19 |

Capacity was granted within two minutes of the request. Container start to ready took 12 min 39 s.

#### Verification

- `/v1/models` lists `xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4` with `max_model_len` 8192 (`verify-models.json`).
- Chat completion, "What is the capital of Australia? Answer in one sentence.", greedy: `The capital of Australia is Canberra.` (`verify-chat.json`).
- Tool call, one `get_weather(city)` tool, "What is the weather in Paris right now?": `finish_reason` `tool_calls`, `get_weather({"city": "Paris"})` (`verify-tool-call.json`).

#### Throughput

`../../../../jev-tpu-v5e1/tpu/w4a16_client.py load`, unchanged except that its URL is read from the environment:
N parallel chat requests of exactly 256 output tokens (`ignore_eos`, greedy), one untimed warm-up pass, then
three timed passes; output tokens per second from the server's usage counts, median of the three.

| Requests | On the VM, `localhost:8000` | Remote, over the internet | Sweep (`2026-09-30-12bctx5-v5e1`) |
|---:|---:|---:|---:|
| 1 | **57.0** (56.9–57.0) | 54.8 | 57.2 |
| 4 | **215.7** (215.7–215.8) | 207.9 | 217.1 |
| 16 | **713.4** (713.0–713.9) | 692.1 | 722.5 |

`load.vm.c*.json` ran on the VM, which is how the sweep ran, so that column is the one to compare. It is within
0.4%, 0.6% and 1.3% of the sweep. The remote column pays about 80 ms of round trip plus a new connection per
request, which at one request is 0.19 s on a 4.5 s request.

#### Differences from the sweep

- **Compile time doubled, 512 s against 237 s.** The sweep served with `--max-num-seqs 16`; the rig leaves vLLM's
  default of 256, so it precompiles 44 backbone programs against the sweep's 20. Throughput at up to 16
  requests is unchanged. The sweep also loaded the checkpoint from local disk and served with no tool or
  reasoning parser.
- **Weights, KV cache and block count match the sweep exactly**: 11.31 GiB, 9,728 tokens, 38 blocks.
- **`lowmem.diff` is a different file** from `jev-tpu-v5e1/patches/lowmem.diff`: the rig's version is written on
  top of `kvshare.diff`, `wna16.diff` and `moe.diff` and adds `WNA16EmbedMethod` and the embedding `decode`
  path. The serving result is the same.

#### Files

| File | What |
|---|---|
| `startup-excerpt.log` | the boot script's progress lines: image, patches with their sha256 prefixes, settings, readiness |
| `vllm-container.log` | `docker logs vllm-gemma4` from start to ready and through the load runs; HTTP access lines removed, addresses redacted |
| `load.vm.c{1,4,16}.json` | load client on the VM |
| `load.remote.c{1,4,16}.json` | load client from outside Google Cloud against the external IP |
| `verify-*.json` | the three verification responses |
