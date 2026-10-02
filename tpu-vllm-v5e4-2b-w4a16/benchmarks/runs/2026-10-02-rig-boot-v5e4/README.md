# 2026-10-02-rig-boot-v5e4

`tpu-vllm-v5e4-2b-w4a16` booted through its own code path: `create_tpu_queued_resource` in `server.py`, flex-start,
`v5litepod-4` in `us-west4-a`, runtime `v2-alpha-tpuv5-lite`, `--tensor-parallel-size 4`, serving `google/gemma-4-E2B-it-qat-w4a16-ct`.
Queued Resource created 2026-10-02T18:18:21.569626026Z. Capacity was granted within 5 minutes.

#### Result: the container exits during model load

Image `vllm/vllm-tpu:nightly` as pulled 2026-10-02, digest `sha256:106a30b605d48adfc5f555fe20dbca9e0d3cef4f1af89cc239e743f161e650ff`. At TP=4:

```
(EngineCore pid=651) ERROR 10-02 18:29:03 [core.py:1436] TypeError: Argument 'model.states[0][378]' of shape bfloat16[256] of type <class 'jax.ShapeDtypeStruct'> is not a valid JAX type.
```

**The same container at `--tensor-parallel-size 1`** on the same VM, same image, every other argument identical,
fails the same way (`vllm-container-tp1.log`). The failure belongs to the image and checkpoint, independent of
chip count. This rig's README already expected the stock image to fail on this W4A16 compressed-tensors checkpoint; the
same checkpoint served in the `../jev-tpu-v5e1` sweep only on the patched image (`wna16.diff`).

#### Diagnosis

Google's QAT exports omit `k_proj`, `v_proj` and `k_norm` for layers 15-34, the 20 layers that reuse an earlier
layer's KV cache and never compute their own. Read from the safetensors headers on 2026-10-02:

| Checkpoint | `k_norm` layers | `k_proj` layers |
|---|---|---|
| `google/gemma-4-E2B-it` | 0-34 (35) | 35 |
| `google/gemma-4-E2B-it-qat-q4_0-unquantized` | 0-14 (15) | 15 |
| `google/gemma-4-E2B-it-qat-w4a16-ct` | 0-14 (15) | 15 |
| `xbill9/gemma-4-E2B-it-qat-w8a8-int8` | 0-14 (15) | 15 |

The 2026-10-02 nightly (tpu_inference `431287b09`, committed 2026-10-02 06:56 UTC) allocates all three on every layer;
its `gemma4.py` says "we still allocate q_proj/k_proj/v_proj because gemma-4's checkpoint stores full Q/K/V weights
for shared layers", which holds for the base export only. The bf16 loader checks every parameter and rejects the 20
missing `k_norm` weights (`2b-q4_0`: exactly layers 15-34). The compressed-tensors loader skips that check, so the
unloaded `k_norm` stays an abstract `ShapeDtypeStruct` of shape `bfloat16[256]` and fails at JIT (`2b-w4a16`).
The v5e-1 rigs served on an earlier nightly; the tag is unpinned and has moved.

#### The same checkpoint on the patched image

The pinned `vllm/vllm-tpu@sha256:19a1a052…` with `../../../../tpu-vllm-v5e4-2b-q4w4a16emb4/patches/` applied, on the
same VM at TP=4, every serving argument unchanged from the failed container (multimodal limits included).
`kvshare.diff` builds no K/V weights for KV-shared layers. It loads and serves (`vllm-container-patched.log`):

- Weights per chip `[(1.83, 15.75), (1.83, 15.75), (1.83, 15.75), (1.83, 15.75)]` GiB; KV cache 737,248 tokens.
- Chat, greedy: `The capital of Australia is Canberra.`. Tool call: `tool_calls`, `get_weather({"city": "Paris"})`.

| Requests | output tok/s | v5e-1 sweep (other settings) | v5e-4 / sweep |
|---:|---:|---:|---:|
| 1 | **116.7** | 136.6 | 0.85x |
| 4 | **409.1** | 532.3 | 0.77x |
| 16 | **1,083.5** | 1,911 | 0.57x |

The v5e-1 column is the `../jev-tpu-v5e1` serving run `2026-09-30-gspeedb-e2b-google-v5e1`. That harness serves with
`--max-model-len 2048`, `--max-num-seqs 16` and no parsers, worth about 2x at 16 requests on one chip, so the ratio
measures the settings along with the chips; a same-settings v5e-1 run is in `../../../../tpu-vllm-v5e1-2b-w4a16/benchmarks/runs/`.

**Fix for the rig:** serve on the pinned digest with `patches/`, as `tpu-vllm-v5e4-2b-w8a8` does, instead of
`vllm/vllm-tpu:nightly`. The v5e-1 sibling pulls the same tag and fails the same way on its next boot.

Added files: `vllm-container-patched.log`, `verify-*-patched.json`, `load.vm.patched.c{1,4,16}.json`.

#### Files

| File | What |
|---|---|
| `startup.log` | the boot script's output (`/var/log/vllm-startup.log`) |
| `vllm-container.log` | `docker logs vllm-gemma4`, HTTP access lines removed, addresses redacted |
| `queued-resource.json` | the Queued Resource as described at collection time |
| `vllm-container-tp1.log` | the TP=1 control container's log |
