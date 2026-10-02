# 2026-10-02-rig-boot-patched-v5e1

`tpu-vllm-v5e1-2b-q4_0` booted through its own code path on the patched image: `create_tpu_queued_resource` in `server.py`, flex-start, `v5litepod-1` in `us-west4-a`, runtime `v2-alpha-tpuv5-lite`, `--tensor-parallel-size 1`, serving `google/gemma-4-E2B-it-qat-q4_0-unquantized`. Queued Resource created 2026-10-02T21:59:38.737271305Z.

Image: the boot script pulled `vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507`, applied the rig's `patches/` in `patches/ORDER` and served the result as `vllm-tpu-q4_0:patched`. Patches applied (`startup.log`):

| Patch | sha256 (first 16) |
|---|---|
| `kvshare.diff` | `92b65901961d05fa` |
| `wna16.diff` | `a6f537de1ff9de41` |
| `moe.diff` | `a954f893fa163a05` |
| `lowmem.diff` | `27e40c1b6412efb0` |
| `textonly.diff` | `7ab13a4356202400` |

Serving arguments are the rig's own, unchanged by the image move: `--max-model-len 16384`, `--max_num_batched_tokens 4096`, `--disable_chunked_mm_input`, `--limit-mm-per-prompt '{"image":4,"audio":1}'`, gemma4 tool-call and reasoning parsers.

Engine start (profile, KV cache, warmup) took 14.1 minutes, 12.6 of them compilation, with a cold compile cache (the rig's startup script sets no `VLLM_XLA_CACHE_PATH`).

#### Verification

- Weights per chip (used, total GiB): `[(8.94, 15.75)]`. KV cache: 323,424 tokens.
- `/v1/models`: `google/gemma-4-E2B-it-qat-q4_0-unquantized`, `max_model_len` 16384.
- Chat, greedy: `The capital of Australia is Canberra.`
- Tool call: finish `tool_calls`, `get_weather({"city": "Paris"})`.

#### Load

`jev-tpu-v5e1/tpu/w4a16_client.py load` run on the VM against `localhost:8000`: N parallel chats, 256 output tokens each (`ignore_eos`), 3 passes per level. Output tok/s is the median pass; min and max alongside. Prompt tokens are summed over the level's requests.

| Requests | output tok/s | min | max | prompt tokens |
|---:|---:|---:|---:|---:|
| 1 | **124.7** | 124.7 | 124.7 | 26 |
| 4 | **435.6** | 435.3 | 435.6 | 107 |
| 16 | **1,136.8** | 1,115.6 | 1,154.0 | 444 |

#### Files

| File | What |
|---|---|
| `load.vm.c1.json` | load client result, c1 |
| `load.vm.c16.json` | load client result, c16 |
| `load.vm.c4.json` | load client result, c4 |
| `queued-resource.json` | the Queued Resource as described at collection time |
| `startup.log` | the boot script's output (`/var/log/vllm-startup.log`), token lines removed |
| `verify-chat.json` | greedy chat completion |
| `verify-models.json` | `GET /v1/models` |
| `verify-tool-call.json` | greedy chat completion with one `get_weather` tool |
| `vllm-container.log` | `docker logs vllm-gemma4`, HTTP access lines removed, addresses redacted |
