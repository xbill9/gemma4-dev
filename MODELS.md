# Gemma 4 model characteristics

Properties of the **checkpoints themselves** — layer structure, attention shape, KV cost, weight
footprint. These are the same whatever serves them, so this file is canonical for the whole monorepo
and every rig should read it rather than re-deriving the numbers.

Anything that depends on a runtime, an engine build, or a chip generation does **not** belong here.
Quantization support, kernel behaviour and measured throughput live with the rig that measured them —
see `tpu-vllm-v5e1-2b/gemma4-quantization.md` for the vLLM-on-TPU quantization landscape.

Read out of `config.json` in the running container on 2026-08-07, cross-checked against
`tpu_inference` source. Where a claim is inferred rather than measured it says so.

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

### 35 layers, 15 KV caches

The split is `first_shared = num_hidden_layers - num_kv_shared_layers` = **35 - 20 = 15**:

- **Layers 0-14** compute and store their own K/V — 15 cache tensors.
- **Layers 15-34** compute no K/V at all and reuse an earlier layer's cache.

The source is *the last preceding layer of the same attention type* (sliding vs full). With the period-5
`layer_types` pattern, full-attention layers sit at 4, 9, 14, 19, 24, 29, 34, so within layers 0-14 the
last full is **14** and the last sliding is **13**. Therefore:

> **All 20 shared layers resolve to exactly two source caches** — layer 13 for the 16 sliding ones,
> layer 14 for the 4 full ones (19, 24, 29, 34).

Not a rolling window and not paired layers. Twenty layers reading two tensors.

**This is why loaders trip on layers 15-34.** Those layers legitimately have no K projection and no
`k_norm`, because they never compute K. A loader expecting per-layer norms across all 35 layers asks for
weights the architecture correctly does not have — the "`k_norm.weight` missing for layers 15-34"
failure seen on the QAT exports. That is a loader bug, not a broken checkpoint.

### KV cost: 15 KiB/token at bf16

```
1 KV head x 2 (K,V) x 256 head_dim x 2 bytes = 1,024 B/token/layer
                          x 15 cached layers = 15,360 B = 15 KiB/token
```

**Do not extrapolate this figure to other sizes.** A 35-layer model paying KV for only 15 layers with a
single KV head is an extraordinarily cheap configuration. Any model without KV sharing, or with real KV
heads, costs multiples of this per token.

### Three head mismatches

1. **Query:KV is 8:1** — `num_attention_heads=8` against `num_key_value_heads=1`. This is full **MQA**,
   not merely GQA.
2. **Heads do not tile the hidden size** — `8 x 256 = 2048` against `hidden_size = 1536`. The Q
   projection is rectangular. Anything computing `head_dim = hidden_size / num_heads` gets 192 and is
   wrong.
3. **`global_head_dim=512` vs `head_dim=256`** — full-attention layers carry a different head dim from
   sliding ones. Observed KV allocation is uniform at 256 across all 15 tensors and the 15 KiB/token
   arithmetic only closes at 256, so `global_head_dim` does not enlarge KV. Where it *is* applied has
   not been traced — treat as unresolved.

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
