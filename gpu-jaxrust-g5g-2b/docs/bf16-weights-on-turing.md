# The weights are bf16 on a chip with no bf16 datapath

**Measured 2026-08-24 on `i-0bca12be1046b5faf` (g5g.2xlarge, T4G, jax 0.11.1, CUDA 13,
Python 3.14). Root cause confirmed. NOT fixed — two fixes were tried on hardware and
both are worse than the disease.**

## What was being looked for

`docs/larger-models-on-t4g.md` recorded per-request transient allocations that scale with
model size — E2B 4.52 GiB, E4B 5.25 GiB, 12B 12.61 GiB — and could not attribute any of
them to a tensor. It guessed at "the reference w4a16 path materialising a large fraction of
the weights in dense form" and correctly marked the cause open.

## What it actually is

`ports/gemma4/jax_e_loader.py` stores every float parameter as **bfloat16**
(`get_arr(key, default_dtype=jnp.bfloat16)`), while `COMPUTE_DTYPE` on this chip is
**float16** — the rig's whole documented dtype policy. A stored dtype the device cannot
compute in is not free: XLA inserts a convert in front of every use, and the converted copy
is a transient the size of the weight.

The optimized HLO for a prefill names it directly:

```
1.500 GiB  f32[262144,1536]   wrapped_convert.231      <- embed_tokens, upcast
1.500 GiB  f32[1536,262144]   bitcast.3677.0           <- the same buffer, transposed
768.0 MiB  bf16[262144,1536]  params__embed_tokens__   <- what is actually stored
4.375 GiB  bf16[262144,8960]  params__embed_tokens_per_layer__
```

`1536` is E2B's `hidden_size`, not a sequence length. Both 1.5 GiB entries are the **LM head
weight**, upcast bf16 → f32, materialised per call.

Two independent lines of evidence that it is a weight and not an activation:

- **The transient is flat in the prompt bucket.** 1.504 GiB at 512, 1.504 GiB at 1,536,
  1.742 GiB at 4,096. An attention score matrix would grow quadratically and an activation
  linearly. `profile_prefill.py --sweep` exists to make exactly this distinction.
- **It is the same conversion `profile_decode.py` already found**, where `wrapped_convert`
  was 55% of decode time on 2026-08-23 and every matmul kernel was `float` rather than
  `__half`. Same cause, two symptoms, measured a day apart by different tools.

`embed_tokens_per_layer` at bf16[262144, 8960] = 4.375 GiB also accounts for the "4.52 GiB
unexplained transient" the earlier doc could not place.

## Why it is not fixed

Changing the loader default to `COMPUTE_DTYPE` is one line and is correct. The cast then has
to happen somewhere, and both places were tried on the instance:

- **On the device** (`arr.astype()` after the tree is on GPU) — OOMs. Source and destination
  are resident together: 8.76 GiB for the PLE table alone, against a 14.07 GB budget.
- **On the host at shard-load time** — correct, frees each source immediately, and
  **unusably slow**: `ndarray.astype(float16)` on an `ml_dtypes.bfloat16` array did not
  finish E2B's 4.7 GB table in 10 minutes on Graviton2. ml_dtypes casts are not vectorised.

A third option was tried and rejected for a different reason: building the whole tree on the
host under `jax.default_device(cpu)` and placing it at the end. That is the obviously right
shape and it fails here — `device_put` of the finished 9.26 GB tree cannot find a contiguous
4.38 GiB block for the PLE table inside a 14.07 GB budget, and ordering the puts
largest-first does not rescue it.

**Promising direction, untried:** bf16 → f16 is a pure bit operation (truncate the mantissa,
same exponent bias), so a `view(uint16)` shift-and-round on the host would avoid ml_dtypes
entirely and run at memory bandwidth. Overflow only matters for values outside f16 range,
which weights are not.

## A numpy trap that cost a hardware round-trip

`ml_dtypes.bfloat16` is an **extension dtype**: `np.dtype(bfloat16).kind` is `'V'`, and
`np.issubdtype(bfloat16, np.floating)` is **False**.

```python
raw[key] = arr.astype(want) if arr.dtype.kind == "f" else arr   # skips bf16 silently
raw[key] = arr if arr.dtype.kind in "iub" else arr.astype(want) # correct
```

The first form looks like it converts the floats and leaves the packed integers alone. It
converts float16 and float32 — which need nothing — and skips the one dtype in the
checkpoint that has to be converted, with no error.

## What DID land

`prefill_with_kv_cache` now selects the last real token **before** the LM head
(`logits_at` on `Gemma4EModelJAX.__call__`) rather than computing `[B, S, vocab]` and
slicing one row out of it. Confirmed in the HLO: no sequence-sized logits tensor appears at
bucket 4,096 any more. It is a strict reduction in both memory and FLOPs on the largest
matmul in the model, and it is orthogonal to the dtype problem above.
