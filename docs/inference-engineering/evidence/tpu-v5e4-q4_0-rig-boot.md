# 2026-10-02-rig-boot-v5e4

`tpu-vllm-v5e4-2b-q4_0` booted through its own code path: `create_tpu_queued_resource` in `server.py`, flex-start,
`v5litepod-4` in `us-west4-a`, runtime `v2-alpha-tpuv5-lite`, `--tensor-parallel-size 4`, serving `google/gemma-4-E2B-it-qat-q4_0-unquantized`.
Queued Resource created 2026-10-02T18:18:09.639566842Z. Capacity was granted within 5 minutes.

#### Result: the container exits during model load

Image `vllm/vllm-tpu:nightly` as pulled 2026-10-02, digest `sha256:106a30b605d48adfc5f555fe20dbca9e0d3cef4f1af89cc239e743f161e650ff`. At TP=4:

```
(EngineCore pid=651) ERROR 10-02 18:25:04 [core.py:1436] ValueError: Following weights were not initialized from checkpoint: {'model.language_model.layers.22.self_attn.k_norm.weight', 'model.language_model.layers.31.self_attn.k_norm.weight', 'model.language_model.layers.17.self_attn.k_norm.weight', 'model.language_model.layers.27.self_attn.k_norm.weight', 'model.language_model.layers.24.self_attn.
```

**The same container at `--tensor-parallel-size 1`** on the same VM, same image, every other argument identical,
fails the same way (`vllm-container-tp1.log`). The failure belongs to the image and checkpoint, independent of
chip count. The patched rigs serve E2B on the pinned `vllm-tpu@sha256:19a1a052…`, where `kvshare.diff` handles
the KV-shared layers.

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

- Weights per chip `[(2.31, 15.75), (2.31, 15.75), (2.31, 15.75), (2.31, 15.75)]` GiB; KV cache 709,152 tokens.
- Chat, greedy: `The capital of Australia is Canberra.`. Tool call: `tool_calls`, `get_weather({"city": "Paris"})`.

| Requests | output tok/s |
|---:|---:|
| 1 | **200.9** |
| 4 | **716.5** |
| 16 | **1,515.3** |

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
