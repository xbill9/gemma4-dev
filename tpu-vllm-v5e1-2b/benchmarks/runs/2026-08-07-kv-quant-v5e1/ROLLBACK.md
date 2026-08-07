# Rollback state for the fp8 KV cache experiment

Captured 2026-08-07 on `tpu-2B-v5e1-devops-agent` (us-west4-a, `aisprint-491218`) **before** any
change. This rig serves live demos; treat this file as the way back.

## Original container, exactly as it ran

| Property | Value |
|---|---|
| Name | `vllm-gemma4` |
| Image tag | `vllm/vllm-tpu:nightly` |
| **Image ID** | `sha256:2a4a1f82793f748e02af54d77a62e590d34d2c9c68e833a8bb00d26a878a684c` |
| Entrypoint | `/entrypoint.sh` |
| Privileged | `true` |
| Network | `host` |
| ShmSize | `10737418240` (10 GB) |
| Binds | `/dev/shm:/dev/shm` |
| Restart policy | `no` |
| Env keys | `HF_HOME=/dev/shm`, `HF_TOKEN` (present — never log or copy it), plus image defaults |

```
vllm serve google/gemma-4-E2B-it \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --disable_chunked_mm_input \
  --max_num_batched_tokens 4096 \
  --limit-mm-per-prompt {"image":4,"audio":1} \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

Baseline serving state: `cache_dtype=auto` (bf16), `kv_cache_size_tokens=321376`,
`num_gpu_blocks=10043`, `block_size=32`, `kv_cache_max_concurrency=19.615`,
`gpu_memory_utilization=0.92`, `enable_prefix_caching=True`, `calculate_kv_scales=False`.

## Why rollback is a rename, not a rebuild

The original container is **preserved, not deleted**. It is stopped and renamed
`vllm-gemma4-bf16`, and the fp8 arm runs as a separate container. Reverting is
`docker start` on the untouched original, which replays its exact config — no risk of
mistranscribing a flag, and `HF_TOKEN` never has to be read out or re-supplied.

It also keeps the original container's **JAX compile cache** on its own filesystem layer, so the
revert boot is warm. Building a fresh bf16 container would pay the full ~738 s compile again.

## Procedure

Forward:

```
sudo docker stop vllm-gemma4
sudo docker rename vllm-gemma4 vllm-gemma4-bf16
# create vllm-gemma4-fp8 from the pinned IMAGE ID (never the :nightly tag - it moves)
```

Back:

```
sudo docker stop vllm-gemma4-fp8
sudo docker start vllm-gemma4-bf16
sudo docker rename vllm-gemma4-bf16 vllm-gemma4
```

Then confirm recovery before declaring done:

```
curl -s localhost:8000/v1/models
curl -s localhost:8000/metrics | grep '^vllm:cache_config_info'   # expect cache_dtype="auto", 321376
```

## Notes

- Boot is slow. The bf16 boot took **814 s** to `Application startup complete`, 738 s of it
  compilation. The fp8 boot is a guaranteed JAX compile-cache miss (the KV dtype changes cache
  shapes), so budget the full ~14 min.
- Pin the image by **ID**. `:nightly` moves, and a newer image pulled between arms would
  confound the comparison with an engine change.
- Do not `docker rm` the bf16 container until the experiment is written up and reverted.
