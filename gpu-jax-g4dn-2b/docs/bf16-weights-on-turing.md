> # SUPERSEDED 2026-08-28 — the thesis of this document is WRONG
>
> **The dtype tax is real (86.9% of decode). Its cause is not what this document says.**
> Storing the weights as bf16 while the device computes float16 was tested directly, by
> converting the whole tree to float16 in the loader and re-profiling. The convert and GEMV
> kernels came back **identical to the microsecond**:
>
> ```
>                      bf16 weights      float16 weights
>   wrapped_convert      19.95 ms/step     19.94 ms/step   n=40
>   (second)              9.98             9.98            n=20
>   (LM head)             9.32             9.32            n=1
>   gemvx                11.92             11.90           n=40
>   throughput          12.9/13.0 tok/s   12.9/13.0 tok/s  (+0.0%)
> ```
>
> **The reason is in the kernel signature.** At `B=1` decode is a matrix-*vector* product, and
> cuBLAS dispatches `gemvx::kernel<int, int, float, float, float, float, ...>` — every template
> parameter **fp32** — reporting `is_op_tensor_core_eligible = False`. **A GEMV has no
> half-precision path**, so the weights are promoted to fp32 whatever they are stored as. The
> `wrapped_convert` kernels are that promotion **to fp32**, not a bf16→float16 fixup.
>
> So: the storage dtype was never the lever, `0.0% TensorCore` is structural rather than a
> misconfiguration, and the "2.2x from removing the converts" estimate was predicated on a fix
> that does not exist at `B=1`. Everything below is kept as the record of how the wrong cause
> was arrived at, and because its *measurements* remain sound — only the attribution was wrong.
>
> Full run: `benchmarks/runs/2026-08-28-f16-weights-g5g/`.

# The weights are bf16 on a chip with no bf16 datapath

**Measured 2026-08-24 on `i-0bca12be1046b5faf` (g5g.2xlarge, T4G, jax 0.11.1, CUDA 13,
Python 3.14). Root cause confirmed. NOT fixed — two fixes were tried on hardware and
both are worse than the disease.**

## What was being looked for

`../gpu-jax-g5g-2b/docs/larger-models-on-t4g.md` (parent rig) recorded per-request transient allocations that scale with
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

## 2026-08-28: the host-side reasoning above is wrong for this loader

**Everything in this document that reasons about "casting on the host" assumes the weights are
on the host. They are not.** MEASURED on `i-0e629977053de0e57`:

```
safetensors.flax.load_file -> jaxlib._jax.ArrayImpl
  dtype    bfloat16
  devices  {CudaDevice(id=0)}
  sharding SingleDeviceSharding(device=CudaDevice(id=0), memory_kind=device)
```

`jax_engine.load()` uses `from safetensors.flax import load_file`, so **every weight is already
resident on the GPU the moment the shard is read.** There is no host-side stage to put a cast
in. (`safetensors.numpy.load_file` cannot even open this checkpoint — it raises
`TypeError: data type 'bfloat16' not understood`.)

**That explains the 2026-08-27 failure exactly, and both halves of it.** The attempt added a
host-side chunked cast in the shard loop, which called `np.asarray()` on device arrays:

- **The 54 s was device→host transfer.** Measured D2H on this box is **1.25 GB/s**, so dragging
  the 9.26 GB tree back costs **~7.4 s** of pure transfer before any conversion, plus the cast,
  plus re-upload, plus allocator churn. Removing the host cast returns `read_shards` to
  **25.6 s** against the 24.7 s baseline — **the regression was entirely the round-trip.**
- **The OOM was both copies resident.** The bf16 tree stayed on device while f16 copies
  accumulated beside it.

### The loader-only change is correct and still not sufficient

Pointing `get_arr`'s default at `COMPUTE_DTYPE` and adding **no** host cast makes the conversion
an ordinary on-device `astype`. `read_shards` is then normal (25.6 s) — but it still OOMs,
now on a mere **384 MiB**, because `raw_weights` holds every bf16 source for the whole
conversion while the f16 tree is built beside it. **bf16 and float16 are both 2 bytes, so the
finished tree is the same size** — the problem is purely that both exist at once:
9.26 GB + 9.26 GB against a 14.07 GB budget.

### The remedy, and the pattern already exists in this repo

**Release each source as its converted copy is made**, so peak is `finished tree + one tensor`
rather than `2x tree`. That is exactly the `release_source=True` pattern
`quantize_ple_table` already uses, added 2026-08-26 for the same class of failure.

Two cautions, both learned the expensive way:

- **`.delete()` invalidates the CALLER's array.** The first version of `release_source` deleted
  unconditionally and a CPU test caught it in seconds — any caller reusing its params dict got
  `Array has been deleted`. Make release opt-in and assert the caller opts in.
- **Order matters at this budget.** Peak is `9.26 GB + largest single tensor`, and the largest
  is `embed_tokens_per_layer` at **4.375 GiB** — about 13.6 GB against 14.07. Converting that
  tensor early, while little else is resident, is the difference between fitting and not.

**Do not attempt this without watching `read_shards` and the allocation log together.** The
failure mode is a startup OOM, not a slow serve.

**Superseded above:** the "three placements tried and rejected" reasoning, and the
`view(uint16)` bit-shift as the untried direction. The bit-shift is real and bit-identical
(re-measured 2026-08-27) but irrelevant here — it is a *host* technique for weights that never
reach the host.

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
