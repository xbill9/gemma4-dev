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

#### Files

| File | What |
|---|---|
| `startup.log` | the boot script's output (`/var/log/vllm-startup.log`) |
| `vllm-container.log` | `docker logs vllm-gemma4`, HTTP access lines removed, addresses redacted |
| `queued-resource.json` | the Queued Resource as described at collection time |
| `vllm-container-tp1.log` | the TP=1 control container's log |
