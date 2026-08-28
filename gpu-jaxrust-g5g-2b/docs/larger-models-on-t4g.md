# How large a Gemma 4 will this rig serve? Measured 2026-08-23

Device budget is **14.07 GB** (`tpu_jax_hbm_limit_bytes`, = 0.90 of the T4G's 15360 MiB), and
**no G5g size changes it**: the engine is single-device (`jax.devices()[0]`, no mesh/pmap/
shard_map/NamedSharding anywhere in the payload), so the second T4G on `g5g.16xlarge` and
`g5g.metal` idles. Bigger instances buy host RAM, not device memory.

## Result

| Model | QAT w4a16 export | Loads? | Serves? | Resident weights | Blocker |
| --- | --- | --- | --- | ---: | --- |
| **E2B** | yes | **yes** | **yes** | **3.05 GB** (`ple_bits=4`) | — works |
| **E4B** | yes | **no** | — | — | OOM **5.25 GiB** during load |
| **12B** | yes | **yes** | **no** | **8.15 GB** | OOM **12.61 GiB** per request |
| 26B A4B | **none (404)** | — | — | — | 15.27 GiB resident > budget |
| 31B | yes | — | — | — | ~15.5 GB int4 > budget, before calibration |

**E2B is the ceiling today.** But note *why* E4B and 12B fail: not resident weight size —
both are comfortably inside 14.07 GB — but **transient** allocations. That is a different and
more tractable class of problem than 26B/31B, which are hard-blocked on residency.

## E4B fails during load, not because the model would not fit

Both variants died at the same 5.25 GiB allocation, at different points:

- `ple_bits=0` — at `jax.block_until_ready(self.params)` (`jax_engine.py:352`), i.e. placing
  the unquantised parameters on device. E4B's QAT export is **11 GB on disk**, larger than
  12B's 9.6 GB, because it carries an unquantised PLE table.
- `ple_bits=4` — inside the quantisation itself (`executable_name='jit_dynamic_slice'`).

That second one is the interesting one. `quantize_ple_table` runs *before* `device_put`, but
jnp ops default to the GPU, so quantising a multi-GB table needs the source and the result
resident on a 14 GB device at once. **The quantised E4B would very likely fit** — it is the
act of quantising that does not. A host-side or chunked PLE quantisation is the obvious
route, and nothing in the current code exposes one.

## 12B loads and cannot serve

`Loaded 8.15 GB of parameters on cuda:0 in 170.7s` — the weights fit with ~5.9 GB to spare.
Every `/v1/chat/completions` then returns 500:

```
Allocator (GPU_0_bfc) ran out of memory trying to allocate 12.61GiB (rounded to 13541437696)
```

12B has **no PLE** (`hidden_size_per_layer_input=0`), so the `ple_bits=4` lever that rescues
E2B does not exist here. There is no equivalent knob.

## The transient allocations are unidentified, and they scale with model size

| model | resident (w4a16) | unexplained transient |
| --- | ---: | ---: |
| E2B | 6.56 GB (`ple_bits=0`) | 4.52 GiB |
| E4B | — (fails first) | 5.25 GiB |
| 12B | 8.15 GB | 12.61 GiB |

They grow roughly with model size, which is consistent with the reference w4a16 path
materialising a large fraction of the weights in dense form — `qat_w4a16_reference_linear_jax`
does say *"materialize the BF16 weight, then matmul"*. **But this is not established.** For
E2B the largest single quantised Linear dequantises to only 0.005 GB, so a per-matmul
explanation does not account for 4.52 GiB, and no single tensor of the observed size has been
identified at any model size. Treat the cause as open; only the correlation is measured.

Identifying it is the highest-value next step: it is what stands between this rig and both
E4B and 12B.

## Calibration: `MODELS.md`'s int4 column under-predicts

Measured against the table's arithmetic columns:

- **int4 under-predicts by 19%** — E2B measured 3.054 GB against ~2.4 GiB (2.58 GB) predicted.
  The column quarters everything, but `embed_tokens` stays bf16 (0.805 GB on E2B) and the
  per-group scales add overhead.
- **bf16 over-predicts by 9%** — E2B measured 9.257 GB against 10.2 GB.

E2B QAT+`ple_bits=4` decomposes as: PLE table 1.175 GB (4-bit, from 4.698 GB bf16) +
`embed_tokens` 0.805 GB + quantised Linears ~1.06 GB.

## Measurement trap found here: `max_tokens` is part of the compiled shape

`max_new_tokens` is a `static_argnames` entry on the jitted decode step, so **changing
`max_tokens` between warm-up and measurement forces a recompile and the measured request is
cold.** This run warmed at 32 tokens and measured at 48, and E2B reported **3.4 tok/s with an
11,494 ms prefill** — against **13.5 tok/s / 160 ms** for the identical configuration warmed
at the same shape. A 4x error, from a warm-up that looked correct.

Warm up at the *exact* `max_tokens` you intend to measure. This is a sharper form of the
existing "warm up before recording anything" rule, and it invalidates the throughput numbers
in this particular matrix (the weight figures are unaffected).
