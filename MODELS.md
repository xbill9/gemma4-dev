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

## Family overview

Nothing structural is shared across sizes. Every row below differs from E2B in a way that changes
loading, KV sizing, or both.

| | E2B | **E4B** | 12B | **26B A4B** | **31B** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Layers | 35 | **42** | — | **30** | **60** |
| Attention pattern | `i%5==4` full (28s/7f) | **`i%6==5` full (35s/7f)** | — | `i%6==5` full (25s/5f) | `i%6==5` full (50s/10f) |
| `num_kv_shared_layers` | 20 | **18** | — | **0** | **0** |
| Layers owning KV | 15 of 35 | **24 of 42** | — | all 30 | all 60 |
| `num_key_value_heads` | 1 | **2** | — | 8 | — |
| `num_global_key_value_heads` | null (→1) | **null (→2)** | — | **2** | **4** |
| `head_dim` / `global_head_dim` | 256 / 512 | 256 / 512 | — | 256 / 512 | — / 512 |
| `hidden_size` | 1536 | **2560** | — | 2816 | — |
| `sliding_window` | 512 | 512 | — | **1024** | — |
| `attention_k_eq_v` | false | false | — | **true** | **true** |
| Per-layer embeddings (PLE) | yes | yes | — | **no** (`hidden_size_per_layer_input=0`) | — |
| Dense or sparse | dense | dense | dense | **sparse MoE** | dense |
| **KV per token, bf16** | **18 KiB** | **56 KiB** | — | ~cheap (see §26B) | — |

Dashes are unrecorded, not "same as E2B". **`num_kv_shared_layers=0` on both large models means the
KV-sharing logic above simply does not apply to them** — every layer owns its KV.

E2B/E4B config values read from the published `config.json` (public even though the weights are gated).

Sources: `~/tpu-jax-26b/docs/gemma4-quirks.md` §15–21 and `~/tpu-jax-31b/docs/gemma4-quirks.md` §12,
verified against the HF reference and by reading checkpoint bytes on a CPU box. Those repos sit outside
this monorepo and predate the naming scheme; the facts are reproduced here so the monorepo is
self-contained.

## E4B — KV is 3.1x E2B's, not comparable

E4B shares the 256/512 split by layer type, so it *looks* like a scaled E2B. It is not: **three separate
things move at once, and they multiply.**

| | E2B | E4B | effect on KV |
| :--- | ---: | ---: | :--- |
| Layers | 35 | 42 | more |
| `num_kv_shared_layers` | 20 | 18 | **fewer shared** |
| Layers owning KV | **15** | **24** | 1.6x |
| `num_key_value_heads` | **1** | **2** | **2x per layer** |
| head_dim by type | 256 / 512 | 256 / 512 | unchanged |

**Cache sharing.** `first_shared = 42 - 18 = 24`, so layers **0-23 own a cache** and layers **24-41**
share. E4B caches 57% of its layers against E2B's 43%. The mapping rule is the same — last preceding
layer of matching type — and within 0-23 the last full is **23** and the last sliding is **22**, so all 18
shared layers again resolve to just **two** source caches (3 full → L23, 15 sliding → L22).

**Head type.** The 256/512 split is identical to E2B: full-attention layers are 512-wide, sliding are 256.
The `layer_types` array is `i % 6 == 5` — full at 5, 11, 17, 23, 29, 35, 41 — so **4 of the 24 cached
layers are full-attention** (5, 11, 17, 23) and 20 are sliding. What changed is the head *count*:
`num_key_value_heads = 2`, and `num_global_key_value_heads` is null so full layers fall back to 2 as well.
E2B's single KV head is the anomaly, not the family norm.

```
20 sliding x 2 KV heads x 2 (K,V) x 256 x 2 B = 20 x 2,048 = 40,960 B
 4 full    x 2 KV heads x 2 (K,V) x 512 x 2 B =  4 x 4,096 = 16,384 B
                                                 total     = 57,344 B = 56 KiB/token
```

**56 KiB/token against E2B's 18 — 3.1x.** Sizing E4B's context budget from E2B experience overstates it
by more than 3x. On a v5e-1 with int8 weights (~7.5 GiB, leaving ~7.0 GiB of the 14.49 usable) that is
**~131,000 KV tokens**, against E2B's measured 321,376 — so roughly 8 concurrent streams at 16K context,
not 20.

Not verified against an allocation log yet: this is derived from the published config using the same
arithmetic that reproduced E2B's measured 18 KiB/token to the byte on two chips. Confirm against
`GPU KV cache size` on first boot.

## `attention_k_eq_v` — full-attention layers ship no `v_proj`

**Set `true` on both the 26B and the 31B, `false` on E2B.** Where set, the full-attention layers carry
`q_proj`, `k_proj`, `k_norm`, `o_proj` and **no `v_proj` at all** — one projection feeds both K and V.

- 31B: all **ten** full-attention layers, verified key by key on the checkpoint.
- 26B: all **five** full-attention layers.

Loading either without handling it yields exactly ten (or five) missing tensors, and **a loader that
tolerates `None` produces a silently broken model that still emits fluent text**. The fix is to alias V to
K — the same arrays, not copies.

This is a checkpoint-shape fact, so it is the first thing to check when a big-model load reports missing
tensors. It is **not** the explanation for E2B: E2B sets the flag `false` and ships `v_proj` on all
fifteen non-shared layers.

The KV cache still stores K and V separately where the flag is set — redundant but correct. Collapsing it
would save one of the two planes on those layers, worth ~4.5% of the 31B's KV.

## 26B A4B — sparse MoE, and the odd one out twice over

`google/gemma-4-26B-A4B-it` — **26.5B total, ~4B active**, 128 experts, top-8.

| field | value |
| :--- | ---: |
| `num_hidden_layers` | 30 (25 sliding / 5 full) |
| `hidden_size` | 2816 |
| `intermediate_size` (dense MLP) | 2112 |
| `num_experts` / `top_k_experts` | 128 / 8 |
| `moe_intermediate_size` (per expert) | 704 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `num_global_key_value_heads` | 2 |
| `sliding_window` | 1024 |

### The MoE block runs *alongside* the dense MLP, not instead of it

`enable_moe_block: true` does **not** swap the feed-forward block for an expert bank. Every layer keeps
its ordinary `mlp` **and** gains a 128-expert bank, and the two outputs are summed before a shared
post-norm:

```
residual = h
dense = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(h) ) )
moe   = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(h) ) )
h     = residual + post_feedforward_layernorm(dense + moe)
```

Both branches read `h`, the post-attention residual — not each other's output. That is why the 26B
carries a full-width dense MLP *on top of* 128 experts, and why its feed-forward block has **five** norms
rather than two.

**The router reads the RAW residual; the experts read a normalized copy.** Passing the normalized tensor
to both is the obvious simplification and it is wrong — the router opens with its own scale-less RMSNorm,
so composing it with a learned-weight RMSNorm first reweights the channels the router sees and changes
which experts fire. **Measured cost of getting this wrong: 0.36 relative error, with router parity tests
and expert parity tests both still green.** Only end-to-end comparison caught it.

In the router tail, the per-expert scale is applied **after** renormalization, so the final top-k weights
do *not* sum to 1. "Fixing" that changes the model.

### KV is not the constraint here — prefill temporaries are

Because the full-attention layers carry only 2 KV heads and the sliding layers window at 1024, KV on the
26B is unusually cheap: ~110 KiB/token at int8 unwindowed, ~0.15 GB at 4K context with windowed KV. This
is the opposite of the usual failure mode and inverts how you size the model.

### No `-w4a16-ct` checkpoint exists for this size

Enumerated from the Hub 2026-07-31 — **the suffix set is not uniform across sizes**:

| size | `-w4a16-ct` | `-q4_0-unquantized` | `-q4_0-gguf` | mobile |
| :--- | :---: | :---: | :---: | :---: |
| E2B, E4B | yes | yes | yes | yes |
| 12B, 31B | yes | yes | yes | — |
| **26B A4B** | **no** | yes | yes | — |

`google/gemma-4-26B-A4B-it-qat-w4a16-ct` 404s. GGUF targets llama.cpp, so the only usable export is
`-q4_0-unquantized`: **51.61 GB of BF16**.

It fits a v6e-1 anyway because **"unquantized" describes the container, not the values.** Those are QAT
weights already sitting on a Q4_0 grid — verified by range-reading the shards, with all 256 sampled groups
of 32 lying exactly on a 4-bit grid across expert, attention, MLP, router and embedding tensors. Group
size 64 fails the same test (3/128), which pins the group at **32** rather than leaving it assumed.
Repacking to W4A16 at load gives **15.27 GB resident**.

Two ways to destroy those weights while "just repacking" them, both silent:

1. **`d = amax/8` is the wrong step.** Many blocks peak below level ±8, so the derived step is a fraction
   of the true one and the block gets requantized onto a grid that does not contain its own values —
   median relative error 4.9e-2. Searching for the level the peak actually occupies, plus least-squares
   refinement, gives exactly 0 error for 93.1% of values.
2. **Packing after the transpose.** W4A16 packs nibbles along the last axis and the Q4_0 grid runs along
   `in`. Packing a transposed weight groups across `out`, where no grid exists — a real requantization
   dressed as a repack. Packing must happen in the loader, before the transpose.

## 31B — dense, 60 layers

`google/gemma-4-31B-it` — 31.0B, 62 GB at bf16. Has `-w4a16-ct` and `-q4_0-unquantized` QAT exports.

- **60 layers**, `[s,s,s,s,s,f]` repeating — full attention at `i % 6 == 5`, so **10 full / 50 sliding**.
- `num_kv_shared_layers = 0` — every layer owns its KV. Sliding layers dominate the KV budget.
- `num_global_key_value_heads = 4` (the 26B uses 2 — **do not carry it over**), `global_head_dim = 512`.
  At layer 5 the packed `k_proj` is `[2048, 672]`, i.e. 4 x 512, with `k_norm` `[512]`.
- `attention_k_eq_v = true` — see above; ten missing `v_proj` on load is expected, not corruption.
- `use_bidirectional_attention = "vision"` — selects bidirectional attention for **image** tokens only.
  Text decoding is unaffected, so the causal-only text path is correct. Absent/`null` on E2B.
- `store_full_length_kv` is **not a checkpoint field** in any config — it is a reference-implementation
  concept. Don't look for it.

**Not verified on TPU.** The 26B facts above are checked against the reference and the checkpoint bytes
on CPU; as of the source doc, nothing was measured on a TPU. Treat performance claims for either large
model as unmeasured.

## Weight footprints

bf16 weight sizes, as recorded in `tpu-jax-v5e1-2b/server.py:806`:

```python
_BF16_WEIGHTS_GB = {"E2B": 10.2, "E4B": 16.0, "12B": 24.0, "26B": 52.0, "31B": 62.0}
```

| Model | Params | bf16 | GiB | int8 | int4 | v5e-1 (14.49 GiB)? | v6e-1 (~28 GiB)? |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| E2B | 2B effective / ~5B total | 10.2 GB | 9.5 | ~4.8 | ~2.4 | bf16 fits (8.97 measured) | yes |
| **E4B** | 4.5B effective / 8.0B total | 16.0 GB | **14.9** | ~7.5 | ~3.7 | needs int8 | bf16 fits |
| 12B | 12B | 24.0 GB | 22.4 | ~11.2 | ~5.6 | needs int4 | bf16 fits |
| **26B A4B** | 26.5B total / **~4B active** | 51.61 GB | 48.1 | ~24 | **15.27 measured** | no | **yes, repacked** |
| 31B | 31.0B | 62.0 GB | 57.7 | ~29 | ~14.4 | no | multi-chip |

int8/int4 columns are arithmetic halving/quartering **except** the 26B, whose 15.27 GB is the measured
resident size after load-time Q4_0→W4A16 repacking of the `-q4_0-unquantized` export.

**The `E` prefix is load-bearing.** E4B is *not* a 4B dense model — 4.5B effective, 8.0B total. Reading
`E4B` as "4B" understates its weights by roughly 2x, which is the difference between fitting a 16 GB
accelerator and not.

**Sparse ≠ small on disk.** The 26B's ~4B *active* parameters set its compute cost, not its memory: all
26.5B must be resident because any token can route to any expert. It is the largest checkpoint here after
the 31B, and the "A4B" in the name describes throughput, not footprint.

E2B's measured on-device figure is **8.97 GiB**, about 6% under the table's 10.2 GB — so treat the
arithmetic entries as close estimates, not exact allocations. `~/tpu-jax-v5e1-2b/server.py` reserves
3.5 GB per chip for the libtpu/XLA runtime plus the activation working set (measured 2.0 + 1.5 GB on a
v6e-1), which is why the v6e-1 column above is ~28 GiB rather than 31.24.

Host RAM does not predict HBM. For CPU debugging: ~8 GiB handles E2B/E4B, ~64 GiB loads the 31B and lets
you inspect its parameter tree but OOMs on a forward pass, ~128 GiB runs it. XLA:CPU allocates roughly 2x
what the weights occupy.

## Naming

The repo's directory slot is size only — `2b`, `4b`, `12b`, lowercase, no `E` prefix — while
`MODEL_NAME` carries the real checkpoint id (`google/gemma-4-E2B-it`). Weight encoding is a separate
optional slot. See `NAMING.md`; do not encode a model characteristic in a directory name from memory.
