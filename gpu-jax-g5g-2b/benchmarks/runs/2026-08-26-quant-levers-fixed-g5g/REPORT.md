# 2026-08-26 — the quantization levers, fixed and measured

**`g5g.2xlarge` spot (`i-03fb4f6bd619179a2`), build id `de4af73c873f`.**

The earlier run today (`2026-08-26-config-sweep-g5g`) found that **none** of this rig's
quantization levers could load. This run fixes all three bugs and measures the levers for the
first time. **5/5 configs load, 15/15 cells.**

## The matrix

| Config | Weights | vs base | Load s | Decode tok/s (128 / 1024 / 4096) |
| --- | ---: | ---: | ---: | --- |
| `ple0` — **shipped default** | 9.257 GB | — | 184.3 | 12.80 / 12.77 / 12.60 |
| `ple8` | 6.927 GB | **−2.330** | 80.5 | 12.80 / 12.70 / 12.60 |
| `ple4` | 5.752 GB | **−3.505** | 126.8 | 12.80 / 12.70 / 12.60 |
| `ple0+int8head` | 9.660 GB | +0.403 | 94.8 | **13.10 / 13.00 / 12.80** |
| **`ple4+int8head`** | **6.155 GB** | **−3.102** | 95.3 | **13.10 / 13.00 / 12.80** |

**Every prediction written in the source comments is confirmed:**

| Claim in `jax_e_model.py` | Measured | Error |
| --- | ---: | ---: |
| `ple_bits=8` → −2.35 GB | −2.330 GB | 0.8% |
| `ple_bits=4` → −3.51 GB | −3.505 GB | 0.1% |
| `int8_lm_head` adds `262144×1536×1 B` | +0.403 GB | exact |

**PLE is memory-only, exactly as documented.** Decode is identical across `ple0`/`ple8`/`ple4`,
which corroborates the port's reasoning — the table is a gather, never a matmul, so decode
never streams it. The win is HBM headroom, not throughput.

**The shipped default is dominated.** `ple4+int8head` is strictly better on both axes: **33%
less memory and +2.3% throughput.** `int8_lm_head` is not numerics-preserving (~0.8% logit
error), so this is a deliberate trade rather than a free win — but nothing else here moves
throughput at all.

## `int8_lm_head` does not do an int8 matmul

The prediction going in was that `int8_lm_head` would win by removing the LM-head conversion
kernel, which was 14.3% of decode on its own. **xprof says it does not.**

```
ple4+int8head:   conversion 54.2%   fp32 33.8%   fp16 0.0%
baseline:        conversion 54.4%   fp32 33.3%   fp16 0.0%

wrapped_convert_61   9.452 ms   12.8%   1 call    <-- still there (was 10.636 ms)
```

The consuming path explains it:

```python
logits = jnp.matmul(h, params["embed_tokens_q8"].T.astype(h.dtype)) * scale
```

The int8 table is **dequantized to fp16 in full — 0.75 GiB — on every decode step**, and the
matmul that follows is the same one as before. The lever halves the bytes *read* (int8 vs
bf16) and pays a full-table convert regardless. That is the entire +2.3%, and it is why the
conversion kernel shrank by 11% rather than disappearing.

This is the same shape as the W4A16 result: what is labelled quantized execution is really
**dequantize-then-matmul**. The memory claim lands to the byte; the speed claim mostly does
not.

**Turing has int8 tensor cores (~130 TOPS on T4) and this path never touches them** — the
kernel table reports `fp16 kernels 0.0%`, and int8 is 0.0% too. A genuine int8 matmul is the
unexploited win here, not a larger PLE.

## The three bugs this run fixed

All three were hard startup failures measured earlier today; each failing allocation matched
its tensor byte-for-byte.

1. **Swap gate excluded the rig's own default size.** `_needs_swap` was
   `0 < host_ram < 16` and `g5g.2xlarge` has exactly 16 GiB, so it got no swapfile while
   `quantize_ple_table` needed >15 GiB of host RSS. Now `<=`. The remedy already existed in
   `_user_data`; only the threshold was wrong.
2. **`quantize_lm_head` upcast the whole table on the device** — `emb.astype(jnp.float32)`
   over `[262144, 1536]`, twice, = 1.50 GiB each on top of a resident tree. Now host-side and
   chunked, matching `quantize_ple_table` 1,200 lines up, which had the correct pattern all
   along.
3. **`quantize_ple_table` placed the copy while the source was resident.** Popping the key
   from the returned dict frees nothing — the caller still holds the device buffer. Now takes
   `release_source=True`, which `load()` passes.

**`release_source` is opt-in for a reason.** The first version deleted unconditionally, and a
CPU test caught it within seconds: `.delete()` invalidates the *caller's* array, so any
caller reusing its params dict got `Array has been deleted`. A test now asserts `load()`
actually opts in — otherwise the fix would sit in the tree doing nothing.

`profile_decode.py` also gained `--int8-lm-head`; it had no such flag, so the one config most
worth profiling could not be profiled at all.

## What to do next

1. **Change the default to `PLE_BITS=4` + `INT8_LM_HEAD=1`** in `tpu.env`, or document why
   not. It is strictly better on both axes. The ~0.8% logit error is the only reason to
   hesitate, and it should be an explicit choice rather than an untested default.
2. **The dtype tax is still 87% of decode and still the only thing that matters.** Neither
   lever touches it. The bf16→float16 weight conversion remains the single largest available
   win, and the bit-shift path (bf16→f32 is a 16-bit left shift, f32→f16 is a native
   vectorised cast) remains untried.
3. **Investigate an actual int8 matmul for the LM head.** The table is already int8 in
   memory; the dequantize-then-matmul in `jax_e_model.py:1232` is what throws the advantage
   away, and Turing has the int8 units to use.
4. **PLE quantization makes loads FASTER** — 80–127 s against 184 s for baseline. Worth
   noting: less to place on the device outweighs the host-side quantization cost.

## Artifacts

| File | What |
| --- | --- |
| `config_sweep.json` | per-config results, checkpointed |
| `driver.log` | full driver console |
| `sweep_*.json`, `requests_*.jsonl` | per-config sweeps and per-request logs |
| `profile_ple4_int8head.txt` | `profile_decode.py` kernel table for the winning config |

The xprof JSON bundle for this config was not retrieved — AWS credentials expired during
extraction. The `profile_decode.py` kernel table above carries the finding; the xprof bundle
for the baseline config is in `2026-08-25-context-sweep-g5g/xprof/`.
