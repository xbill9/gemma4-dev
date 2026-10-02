# 2026-10-02-rig-boot-patched-v5e1

`tpu-vllm-v5e1-2b-w4a16` booted twice through its own `create_tpu_queued_resource` on the patched image: flex-start
`v5litepod-1` in `us-west4-a`, runtime `v2-alpha-tpuv5-lite`, `--tensor-parallel-size 1`, serving
`google/gemma-4-E2B-it-qat-w4a16-ct`. The boot script pulled `vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507`,
applied `patches/` and started `vllm-tpu-w4a16:patched`. The checkpoint loads on this image; **neither boot reached
serving**, so this run has no verification or load numbers.

| Patch | sha256 (first 16) |
|---|---|
| `kvshare.diff` | `92b65901961d05fa` |
| `wna16.diff` | `a6f537de1ff9de41` |
| `moe.diff` | `a954f893fa163a05` |
| `lowmem.diff` | `27e40c1b6412efb0` |
| `textonly.diff` | `7ab13a4356202400` |

#### Boot 1: `--gpu-memory-utilization` 0.90 (vLLM's default), program load fails on HBM

Weights loaded in 142.39 s; `Init model | hbm=[(7.17, 15.75)]` GiB; KV cache 426,112 tokens. Compilation ran from
10-02 22:08:36 to the failure at 10-02 22:36:42, then loading the compiled program failed:

```
RESOURCE_EXHAUSTED: E0101: RuntimeProgramAllocationFailure:
Error loading program 'jit_run_model_impl': Attempting to allocate 1.15G. That was not possible. There are 1.13G free.
```

The KV pool at 0.90 leaves too little HBM for the compiled program. The rig now sets `GPU_MEMORY_UTILIZATION=0.80`,
the value the other patched v5e-1 rigs use (`vllm-container-gmu090.log`, `startup-gmu090.log`).

#### Boot 2: `--gpu-memory-utilization` 0.80, the host kills the engine during compilation

Queued Resource created 2026-10-02T22:39:25.032695931Z. Weights loaded in 142.85 s; `Init model | hbm=[(7.17, 15.75)]` GiB; KV cache 316,000 tokens.
Compilation started 10-02 22:48:11; at Fri Oct  2 23:22:36 2026 (UTC) the kernel's OOM killer stopped `VLLM::EngineCor` with
35.1 GiB resident (59.0 GiB virtual) on the 48 GiB `ct5lp-hightpu-1t` host
(`dmesg-oom.log`), and the API server exited with `Engine core initialization failed ... {'EngineCore': -9}`.
The model files also sit in `/dev/shm` (`HF_HOME=/dev/shm`), which counts against the same host RAM.

Boot 1 compiled every bucket on the same host with the same `--max_num_batched_tokens 4096`, so compile-time host
memory on this checkpoint sits close to the host's limit. The patched v5e-1 rigs that serve
(`../../../../tpu-vllm-v5e1-2b-w8a8`, `../../../../tpu-vllm-v5e1-2b-q4w4a16emb4`) cap batched tokens at 512 with
`MIN_TOKEN_BUCKET=64`, which compiles four backbone buckets instead of eight to keep compile host memory down; this
rig keeps 4096 for its multimodal limits. Compile cache: cold on both boots (the startup script sets no
`VLLM_XLA_CACHE_PATH`).

#### Files

| File | What |
|---|---|
| `startup.log` | boot 2's script output (`/var/log/vllm-startup.log`), token lines removed |
| `vllm-container.log` | boot 2's `docker logs vllm-gemma4`, HTTP access lines removed, addresses redacted |
| `dmesg-oom.log` | the kernel OOM-killer lines from boot 2 |
| `queued-resource.json` | boot 2's Queued Resource as described at collection time |
| `startup-gmu090.log` | boot 1's script output |
| `vllm-container-gmu090.log` | boot 1's container log |
