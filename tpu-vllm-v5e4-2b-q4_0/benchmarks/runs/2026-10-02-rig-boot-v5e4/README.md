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

#### Files

| File | What |
|---|---|
| `startup.log` | the boot script's output (`/var/log/vllm-startup.log`) |
| `vllm-container.log` | `docker logs vllm-gemma4`, HTTP access lines removed, addresses redacted |
| `queued-resource.json` | the Queued Resource as described at collection time |
| `vllm-container-tp1.log` | the TP=1 control container's log |
