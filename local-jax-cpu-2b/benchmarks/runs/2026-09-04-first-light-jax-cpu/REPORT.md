# First light — `local-jax-cpu-2b`, 2026-09-04

This repo's own JAX port (`ports/gemma4/` via `jax_engine.py` behind
`jax_openai_server.py`), `google/gemma-4-E2B-it` at `ple_bits=4`, on the host CPU
(Intel i7-10750H, 6 physical cores, AVX2 only). jax/jaxlib **0.11.2.dev20260904** (nightly).
No accelerator. **The rig's first recorded run** — its `benchmarks/runs/` was empty until now.

## Result

| | |
| :--- | ---: |
| load (download → read_shards → convert_params → device_put) | **59 s** |
| weights placed | 5.75 GB on `cpu:0` |
| tower weights skipped at load | **0.95 GB** |
| cold query, 128 tok | 295.4 s = 0.43 tok/s ⚠️ compiled during the request |
| warm query 1 | 271.5 s = **0.47 tok/s** |
| warm query 2 | 279.1 s = **0.46 tok/s** |
| engine decode gauge | **0.50 tok/s** (`tpu_jax_decode_tokens_per_second`) |
| engine RSS | 4.72 GB (`tpu_jax_host_rss_bytes`) |

**Compilation was only ~8% of the cold request** — 0.43 → 0.47 tok/s warm. This rig is genuinely
slow, not slow-to-start. The engine stamps `⚠️ COLD` on a request that compiled, unprompted, which
is what makes that split visible at all.

## Against the PyTorch rig on the same host and checkpoint

| rig | tok/s | note |
| :--- | ---: | :--- |
| `local-pytorch-cpu-2b` | **4.85** | stock transformers, dense bf16 |
| `local-jax-cpu-2b` | **0.47** | this repo's JAX port, `ple_bits=4` |

**~10x, in transformers' favour — and the number is confounded, so do not publish the ratio.**
Three reasons, in order of how much they could move it:

1. **This run paged and the PyTorch run did not.** `SwapFree` fell 15.94 → 7.04 GB across load and
   first query. Worse, the engine reports **RSS 4.72 GB against 5.75 GB of placed weights**, so
   roughly a gigabyte of the weight set itself is on disk — and decode streams weights every token.
   The PyTorch run had ~12 GB free and never touched swap.
2. **XLA:CPU has no bf16 datapath.** It upconverts to fp32 in front of every use. `verify_cpu_backend`
   says so at startup. bf16 is not a speed choice here — fp32 *storage* would need 18.51 GB against
   16.42 GB of host RAM, so it is the only option that fits.
3. **`ple_bits=4` is not free on CPU.** It buys 3.5 GB of residency and costs a dequant on every
   gather. On the TPU parent it measured 0.0% decode impact; that result was taken on hardware with a
   very different memory hierarchy and does not transfer here unmeasured.

**What would settle it:** free the host, restart, and re-measure with `SwapFree` unchanged
throughout. Not done — that is the obvious next run.

## `check_host_capacity` under-predicts, and this run proves it

The tool cleared the load explicitly:

```
Weights (ple_bits=4)                 5.75 GB
Prefill transient at bucket 2048     1.61 GB
Total                                7.37 GB
✅ Fits in available RAM with 4.01 GB to spare. No swap needed.
```

Then `MemAvailable` fell to **0.87 GB** during `convert_params` and ~5.5 GB went to swap, with a
further ~3.4 GB during the first query. **Its model is `weights + prefill transient`; the
dtype-conversion transient is not in it**, and neither is whatever the first XLA compilation holds.

This is exactly the failure the rig's own `CLAUDE.md` says the tool exists to prevent — "exceeding
host RAM is *accepted* and paid for in swap, so a thrashing serve is indistinguishable from a loading
one." The tool was right about the steady state and wrong about the peak, and the peak is what
decides whether you swap.

## Confirmations worth keeping

- **`skipped 0.95 GB of non-text tower weights`** at load. That is the fourth independent arrival at
  the same figure today, from a fourth code path — the Ollama projector blob (0.951 GB measured), the
  w4a16 safetensors header (0.945 GB), `@MODELS.md`'s per-tensor table, and now this engine.
- **Gemma 4 answered directly**, no thinking block, at a 128-token budget — matching
  `local-pytorch-cpu-2b` and unlike the llama.cpp and Ollama sweeps. Both rigs that answer directly
  are the two applying the checkpoint's own chat template.

## Caveats

- **`detect_hardware_profile()` reports `tpu-v6e-1` on this CPU-only host.** The shared port falls
  back to the TPU profile off-accelerator. It is a default, not a reading of this machine; any figure
  the engine derives from it is not about this hardware. `verify_cpu_backend` warns about this.
- **One prompt, one shape, three requests.** First light, not a sweep.
- **A prior session exists in the server log** (09:41 today) that loaded 5.75 GB and was never
  recorded. This is the rig's first *recorded* run, not necessarily its first execution.
