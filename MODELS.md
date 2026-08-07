# Gemma 4 model characteristics

Properties of the **checkpoints themselves** — layer structure, attention shape, KV cost, weight
footprint. These are the same whatever serves them, so this file is canonical for the whole monorepo
and every rig should read it rather than re-deriving the numbers.

Anything that depends on a runtime, an engine build, or a chip generation does **not** belong here.
`QUANTIZATION.md` covers what the serving stack supports, `HARDWARE.md` what the silicon can compute in,
and measured throughput lives with the rig that measured it, under its `benchmarks/runs/`.

Read out of `config.json` and the **safetensors tensor headers** on 2026-08-07, cross-checked against
`tpu_inference` source and boot-time allocation logs. Where a claim is inferred rather than measured it
says so. **Config fields alone were not sufficient here** — `head_dim` is a single value for a model with
two attention geometries, and trusting it produced a 17% KV sizing error that only the weights exposed.

## E2B — `google/gemma-4-E2B-it`

| Field | Value |
| :--- | ---: |
| `num_hidden_layers` | 35 |
| `num_kv_shared_layers` | 20 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | **1** |
| `head_dim` | 256 |
| `global_head_dim` | 512 |
| `hidden_size` | 1536 |
| `intermediate_size` | 6144 |
| `vocab_size` | 262,144 |
| `sliding_window` | 512 |
| `layer_types` | 4x `sliding_attention`, 1x `full_attention`, repeating (period 5) |

Also `tie_word_embeddings=True`, `final_logit_softcapping=30.0`, and per-layer embeddings
(`vocab_size_per_layer_input=262144`, `hidden_size_per_layer_input=256`).

### Two attention geometries, not one

**Verified by reading the safetensors headers** (`model.safetensors`, filtered to
`model.language_model.layers.*.self_attn.*`; script at
`tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-07-kv-quant-v5e1/inspect_weights.py`). Shapes are `[out, in]`:

| Layer type | Count | `q_proj` | `k_proj` / `v_proj` | `o_proj` | `q_norm` / `k_norm` | head_dim |
| :--- | ---: | :--- | :--- | :--- | :--- | ---: |
| `sliding_attention` | 28 | 2048x1536 | **256**x1536 | 1536x2048 | 256 | **256** |
| `full_attention` | 7 | 4096x1536 | **512**x1536 | 1536x4096 | 512 | **512** |

Both keep 8 query heads and **1 KV head**; what changes is head_dim. `global_head_dim=512` is therefore
real and applies to Q, K, V **and** the norms in full-attention layers — it is not a Q-only field.

**All 35 layers carry `q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm` and `k_norm`. Nothing is missing
from the base checkpoint** — `layers missing k_proj: []`, `layers missing k_norm: []`.

> **Correction (2026-08-07).** An earlier version of this file claimed full-attention layers were
> allocated at 256 like the rest, that KV cost 15 KiB/token, and that layers 15-34 "legitimately have no
> K projection and no `k_norm`". The weights refute all three. The `k_norm`-missing failure on the QAT
> exports is therefore **not** explained by the architecture and needs re-diagnosing against that
> checkpoint; do not repeat the old explanation.

**A caution when reading tensor names:** the checkpoint also contains `model.audio_tower.layers.N.*` and
a vision tower, each with its own independent layer numbering and its own `self_attn.*`. A regex matching
`layers\.(\d+)\.` collides with them and silently overwrites language-model values for low indices.
Always anchor on `model.language_model.`.

### Which layers hold a cache

The split is `first_shared = num_hidden_layers - num_kv_shared_layers` = **35 - 20 = 15**:

- **Layers 0-14** own the 15 KV cache tensors the runtime allocates.
- **Layers 15-34** are marked KV-shared and read an earlier layer's cache.

This is a **runtime** property, not a checkpoint one — the weights above show K/V projections present for
all 35 layers, so the sharing is in how the model is executed, not in what was shipped. Whether layers
15-34's `k_proj`/`v_proj` are loaded-but-unused (~38 MB of dead weights at bf16) has **not** been
verified.

The source is *the last preceding layer of the same attention type* (sliding vs full). With the period-5
`layer_types` pattern, full-attention layers sit at 4, 9, 14, 19, 24, 29, 34, so within layers 0-14 the
last full is **14** and the last sliding is **13**. Therefore:

> **All 20 shared layers resolve to exactly two source caches** — layer 13 for the 16 sliding ones,
> layer 14 for the 4 full ones (19, 24, 29, 34).

Not a rolling window and not paired layers. Twenty layers reading two tensors.

### KV cost: 18 KiB/token at bf16

The 15 cached layers are **not** homogeneous. Layers 0-14 contain three full-attention layers (4, 9, 14)
at head_dim 512 and twelve sliding layers at 256:

```
12 sliding x 1 KV head x 2 (K,V) x 256 x 2 bytes = 12 x 1,024 = 12,288 B
 3 full    x 1 KV head x 2 (K,V) x 512 x 2 bytes =  3 x 2,048 =  6,144 B
                                                    total     = 18,432 B = 18 KiB/token
```

**Independently cross-checked on two generations:**

| | tokens | x 18,432 B | matches |
| :--- | ---: | ---: | :--- |
| v5e-1 | 321,376 | 5.52 GiB | `total_hbm_avail_gb=5.52GiB` in the boot log — exact |
| v6e-1 | 1,151,744 | 19.77 GiB | 19.79 GiB measured pool — 0.1% |

At the old 15 KiB/token figure the v5e number would be 4.60 GiB against 5.52 GiB available, i.e. 17% of
the KV budget unexplained. **18 KiB/token is right and 15 KiB/token was wrong**, on both chips.

> **Two derived figures elsewhere inherit the old error and need correcting:** the "KV cache 4.60 GiB
> (derived)" row in `tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-06-vllm-sweep-v5e1/REPORT.md` (should be
> ~5.52 GiB), and any per-token sizing that used 15 KiB.

**Unresolved, and the reason to distrust the runtime here.** The v5e boot log reports a *single* KV
layout for all 15 tensors — `num_kv_cache_groups=1`, `regular_attn_shape=(num_blocks, (32, 1, 2, 256))`
— for a model that demonstrably needs two shapes. The memory arithmetic says 18 KiB/token was actually
allocated, so the printed 256 shape is at best one representative tensor and at worst evidence the
allocator is flattening a hybrid cache to a single geometry. **Resolve this against a per-tensor
allocation dump before trusting any KV sizing on a hybrid-attention model.** If a runtime really does
allocate 256 for the full-attention layers, their K/V is being truncated and that is a correctness bug,
not a sizing one.

**Do not extrapolate 18 KiB/token to other sizes either.** A 35-layer model paying KV for only 15 layers
with a single KV head is still an extraordinarily cheap configuration. Any model without KV sharing, or
with real KV heads, costs multiples of this per token.

### Three head mismatches

1. **Query:KV is 8:1** — `num_attention_heads=8` against `num_key_value_heads=1`. This is full **MQA**,
   not merely GQA.
2. **Heads do not tile the hidden size** — `8 x 256 = 2048` against `hidden_size = 1536`. The Q
   projection is rectangular. Anything computing `head_dim = hidden_size / num_heads` gets 192 and is
   wrong.
3. **`global_head_dim=512` vs `head_dim=256` is real, and it applies to K/V.** Confirmed in the weights:
   full-attention layers ship `k_proj`/`v_proj` at 512x1536 and norms at 512, against 256 for sliding.
   **A single `head_dim` does not describe this model.** Anything that reads one value and applies it to
   all 35 layers under-counts the seven full-attention layers by 2x — which is exactly how the
   15 KiB/token error arose. The v6e reconciliation that first flagged this was correct.

### Single KV head does not shard

`num_key_value_heads=1` cannot be split across chips. Runtimes pad `num_kv_heads` up to a multiple of
the tensor-parallel size, so at TP=4 you pay **4x the KV memory to store the same head replicated**.
A larger topology does not divide E2B's KV cost; it multiplies it. Check the target model's
`num_key_value_heads` before assuming more chips solves a memory problem.

## Weight footprints

bf16 weight sizes, as recorded in `tpu-jax-v5e1-2b/server.py:806`:

```python
_BF16_WEIGHTS_GB = {"E2B": 10.2, "E4B": 16.0, "12B": 24.0, "26B": 52.0, "31B": 62.0}
```

| Model | Params | bf16 | GiB | int8 | int4 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| E2B | 2B effective / ~5B total | 10.2 GB | 9.5 | ~4.8 | ~2.4 |
| **E4B** | 4.5B effective / 8.0B total | 16.0 GB | **14.9** | ~7.5 | ~3.7 |
| 12B | 12B | 24.0 GB | 22.4 | ~11.2 | ~5.6 |

int8/int4 columns are the arithmetic halving/quartering, not measured.

**The `E` prefix is load-bearing.** E4B is *not* a 4B dense model — 4.5B effective, 8.0B total. Reading
`E4B` as "4B" understates its weights by roughly 2x, which is the difference between fitting a 16 GB
accelerator and not.

E2B's measured on-device figure is **8.97 GiB**, about 6% under the table's 10.2 GB — so treat these as
close estimates, not exact allocations.

## Naming

The repo's directory slot is size only — `2b`, `4b`, `12b`, lowercase, no `E` prefix — while
`MODEL_NAME` carries the real checkpoint id (`google/gemma-4-E2B-it`). Weight encoding is a separate
optional slot. See `NAMING.md`; do not encode a model characteristic in a directory name from memory.
