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
from the base checkpoint** — `layers missing k_proj: []`, `layers missing k_norm: []`. Note the scope:
this is the **base export**, `google/gemma-4-E2B-it`. The QAT exports are a different artifact and are
**not** described by this row — see below.

> **Correction (2026-08-07).** An earlier version of this file claimed full-attention layers were
> allocated at 256 like the rest, that KV cost 15 KiB/token, and that layers 15-34 "legitimately have no
> K projection and no `k_norm`". The weights refute all three **as claims about the base checkpoint**.

> **Amended 2026-08-07: that correction was then over-applied.** It was read as evidence that the
> `k_norm`-missing failure on the **QAT** exports is unexplained. It is not evidence about those
> exports at all — the headers above are the base repo's. Reading *both* repos shows the base ships
> `self_attn.k_norm` for all 35 layers while **the QAT export ships it only for the 15 non-KV-shared
> layers**, both configs declaring `num_kv_shared_layers: 20`. Those readings are compatible: the base
> carries tensors layers 15-34 never use, and the QAT export drops them.
>
> Since KV sharing is a **runtime** property, a base-checkpoint header count can never settle a
> question about the QAT export in either direction. That inference is what produced both the original
> error and the overshoot; don't make it a third time. `QUANTIZATION.md` holds the full re-diagnosis
> and the loading evidence — this file states only what the checkpoints contain.

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

**Resolved 2026-08-07: the allocation is correct; the log line is misleading.** The boot log reports a
single KV layout for all 15 tensors — `num_kv_cache_groups=1`,
`regular_attn_shape=(num_blocks, (32, 1, 2, 256))` — for a model that demonstrably needs two shapes,
which looked like it could be silent truncation of the full-attention layers. It is not. In
`tpu_inference/runner/kv_cache_manager.py` each layer's cache is built from **that layer's own spec**:

```python
kv_cache = create_kv_caches(..., num_kv_heads=layer_spec.num_kv_heads,
                                 head_size=layer_spec.head_size, ...)   # per layer
kv_caches.append(kv_cache)
metadata["regular_attn"].count += 1
if metadata["regular_attn"].shape is None:        # <- first layer only, never updated
    metadata["regular_attn"].shape = kv_cache.shape
```

`count` increments for all 15 tensors but `shape` is written **once, from the first layer** — layer 0,
which is sliding, hence 256. So the printed shape describes layer 0 alone and says nothing about layers
4, 9 and 14. Heterogeneous per-layer caches are allocated correctly, and the 18 KiB/token memory
arithmetic is the accurate reading.

**Take this as a warning about the log, not the allocator:** `regular_attn_shape` is a first-wins sample
presented as if it described the group. On any hybrid-attention model it under-reports, and it is what
produced the 15 KiB/token error above. Size KV from the config geometry and check it against
`total_hbm_avail_gb`, not from this line.

**Do not extrapolate 18 KiB/token to other sizes either.** A 35-layer model paying KV for only 15 layers
with a single KV head is still an extraordinarily cheap configuration. Any model without KV sharing, or
with real KV heads, costs multiples of this per token.

> **OPEN DISCREPANCY — the vLLM CUDA path reports about half this, 2026-08-30.** Serving E2B under
> stock vLLM 0.28.0 on an NVIDIA L4 (`gpu-vllm-g6-2b`, `benchmarks/runs/2026-08-30-first-serve-g6/`),
> the engine allocated a **9.65 GiB KV pool** and reported **1,076,849 tokens** of capacity —
> **9,622 B/token, a 1.92x gap against the 18,432 B derived above.**
>
> **Neither number has been shown wrong, and neither should be edited to match the other.** The
> 18 KiB derivation is geometry plus two exact TPU cross-checks. The 9.4 KiB figure is a
> *divided-out average* — pool ÷ reported capacity — taken on a **different serving stack**
> (`vllm` on CUDA, not `tpu_inference`). The most likely reconciliation is that vLLM v1 charges
> sliding-window layers only their **window** rather than the full context, in which case the two
> figures answer different questions and 1.92x is close to what a 12-sliding/3-full split would
> predict. That is a hypothesis, **not a measurement** — it needs vLLM's KV-cache-group accounting
> read before anyone relies on it.
>
> Practical consequence for sizing: **on the vLLM CUDA path, do not budget KV at 18 KiB/token** —
> you will under-provision context by ~2x. Size from the engine's own allocation log, which is the
> rule this section already gives for the TPU path.

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

### RMSNorm has NO `1 + weight` convention — unlike Gemma 1, 2 and 3

**Measured 2026-09-02.** `Gemma4RMSNorm.forward` is `normed_output * self.weight` — a plain scale.
Gemma 1/2/3 all use `(1 + weight)`, which is why every GGUF converter for those adds 1 when writing and
every reader subtracts 1 when loading (`Gemma2TensorProcessor` in transformers is exactly that `-1`).

**Gemma 4 does neither, and the artifacts confirm it rather than merely implying it.** The F32 norm
tensors in `google/gemma-4-E2B-it-qat-q4_0-gguf` are bit-identical to the bf16 tensors in the
`-qat-q4_0-unquantized` safetensors — `blk.0.attn_norm` and `layers.0.input_layernorm` both start
9.375, 7.9375, 10.6875 with mean +10.67993. No offset is applied in either direction.

Two consequences:

- **transformers omitting `gemma4` from `TENSOR_PROCESSORS` is correct, not a bug.** It falls back to the
  identity processor, which is what Gemma 4 wants. Do not "fix" it by pointing `gemma4` at
  `Gemma2TensorProcessor`; that would subtract 1 from every norm in the model.
- **Never carry a Gemma 2/3 loader convention into a Gemma 4 port.** The values look plausible either way —
  a norm weight near 10 is not obviously wrong at 9 — so this fails silently as a quality regression rather
  than as an error.

## Family overview

Nothing structural is shared across sizes. Every row below differs from E2B in a way that changes
loading, KV sizing, or both.

| | E2B | **E4B** | 12B | **26B A4B** | **31B** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Layers | 35 | **42** | **48** | **30** | **60** |
| Attention pattern | `i%5==4` full (28s/7f) | **`i%6==5` full (35s/7f)** | `i%6==5` full (40s/8f) | `i%6==5` full (25s/5f) | `i%6==5` full (50s/10f) |
| `num_kv_shared_layers` | 20 | **18** | **0** | **0** | **0** |
| Layers owning KV | 15 of 35 | **24 of 42** | all 48 | all 30 | all 60 |
| `num_key_value_heads` | 1 | **2** | 8 | 8 | 16 |
| `num_global_key_value_heads` | null (→1) | **null (→2)** | **1** | **2** | **4** |
| `head_dim` / `global_head_dim` | 256 / 512 | 256 / 512 | 256 / 512 | 256 / 512 | 256 / 512 |
| `hidden_size` | 1536 | **2560** | **3840** | 2816 | **5376** |
| `intermediate_size` | 6144 | — | **15360** | 2112 | — |
| `sliding_window` | 512 | 512 | **1024** | **1024** | **1024** |
| `attention_k_eq_v` | false | false | **true** | **true** | **true** |
| Per-layer embeddings (PLE) | yes | yes | **no** | **no** (`hidden_size_per_layer_input=0`) | **no** |
| `use_double_wide_mlp` | true | — | **false** | **false** | **false** |
| Dense or sparse | dense | dense | dense | **sparse MoE** | dense |
| **KV per token, bf16** | **18 KiB** | **56 KiB** | see caution below | ~cheap (see §26B) | — |

Dashes are unrecorded, not "same as E2B". **`num_kv_shared_layers=0` on every size above E4B means the
KV-sharing logic above simply does not apply to them** — every layer owns its KV.

**`num_global_key_value_heads` is the one field with no pattern to extrapolate** — null, null, 1, 2, 4
across the five sizes. Never carry one size's value onto another.

E2B/E4B config values read from the published `config.json` (public even though the weights are gated).

Sources: `~/tpu-jax-26b/docs/gemma4-quirks.md` §15–21 and `~/tpu-jax-31b/docs/gemma4-quirks.md` §12,
verified against the HF reference and by reading checkpoint bytes on a CPU box; the 12B column from
`tpu-jax-v6e1-12b-w4a16/docs/12b-exploration-2026-07-31.md`, read off `config.json` on the device.
The `~/tpu-jax-*` repos sit outside this monorepo and predate the naming scheme; the facts are
reproduced here so the monorepo is self-contained.

> **One line in the 12B exploration note contradicts this table, and the table wins.** That note's
> comparison lists E2B at `num_key_value_heads` 4 and `num_global_key_value_heads` 4. The safetensors
> headers say **1** — full MQA — and the boot-time allocation arithmetic agrees to 0.1% on two chip
> generations. Its 12B column was read from `config.json` and is sound; its E2B column was filled in
> from memory.

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

**Set `true` on the 12B, the 26B and the 31B; `false` on E2B and E4B.** Where set, the full-attention
layers carry `q_proj`, `k_proj`, `k_norm`, `o_proj` and **no `v_proj` at all** — one projection feeds
both K and V.

- 31B: all **ten** full-attention layers, verified key by key on the checkpoint.
- 26B: all **five** full-attention layers.
- 12B: all **eight** full-attention layers.

**Every dense size at 12B and above sets it** — treat it as the family default above E4B rather than a
big-model curiosity, and expect it on any new size until the config says otherwise.

Loading any of them without handling it yields exactly ten (or five, or eight) missing tensors, and
**a loader that tolerates `None` produces a silently broken model that still emits fluent text**. The
fix is to alias V to K — the same arrays, not copies.

This is a checkpoint-shape fact, so it is the first thing to check when a big-model load reports missing
tensors. It is **not** the explanation for E2B: E2B sets the flag `false` and ships `v_proj` on all
fifteen non-shared layers.

The KV cache still stores K and V separately where the flag is set — redundant but correct. Collapsing it
would save one of the two planes on those layers, worth ~4.5% of the 31B's KV.

## 12B — the MatFormer features switched off

`google/gemma-4-12B-it-qat-w4a16-ct`, ported to the pure-JAX engine on a v6e-1, 2026-07-31.
Artifacts in `tpu-jax-v6e1-12b-w4a16/`.

**It needed zero code changes to load.** The 12B is the same `gemma4_unified` architecture as E2B with
every MatFormer feature switched *off* — no PLE, no KV sharing, no double-wide MLP. `config_from_hf`
already resolved all of it, because `pick()` tests `is not None` and so the `0`s that disable those
features survive rather than being treated as absent. RoPE, logit softcapping (30.0), tied embeddings
and the 262,144 vocab are identical to E2B. See the family table above for the full field list.

`attention_k_eq_v` is **true** here too, so the eight full-attention layers ship no `v_proj` — the same
load-time surprise documented for the 26B and 31B above.

There is no `lm_head` tensor and no `embed_tokens_per_layer`; the checkpoint is one 10.26 GB
`model.safetensors` including the vision and audio towers. At W4A16 it is **8.15 GB resident**.

**The `'111111'` digit output on bare prompts is not an engine defect.** Without `<bos>` and the Gemma 4
chat template, the HF PyTorch reference emits the same digit strings. It is a prompt-formatting
requirement of the IT QAT checkpoint. Formatted, JAX and the reference reach 100% exact token parity.
**The 31B does not share this** — it recovers the `<|channel>thought` scaffolding on its own from a
bare prompt, so a bare-prompt smoke test that passes on the 31B proves nothing about the 12B.

> **Do not quote that rig's per-token KV figure.** Its `REPORT.md` charges the `attention_k_eq_v`
> layers for a V it also states is free (16 KiB/token where 8 KiB follows if V really is K), and adds
> that to a *window-capped* sliding-layer figure as though both were uncapped rates. The stated
> 336 KiB/token inherits both errors. Derive KV from this file and confirm against a boot allocation
> log.

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

## 26B A4B — expert tensor layout

Supplements the §26B section above with the shape traps, which are checkpoint facts rather than wiring.

Expert weights are `nn.Parameter`, not `nn.Linear`, so **the keys carry no `.weight` suffix**:

| key | shape | meaning |
| :--- | :--- | :--- |
| `layers.N.experts.gate_up_proj` | `[128, 1408, 2816]` | `[E, 2*moe_inter, hidden]` |
| `layers.N.experts.down_proj` | `[128, 2816, 704]` | `[E, hidden, moe_inter]` |

- **Gate and up are fused** into one `[2I, H]` tensor per expert — first `I` rows gate, last `I` up.
  Splitting on the wrong axis or the wrong half silently swaps them, and `gelu(up) * gate` is a
  plausible-looking model that is not this one.
- **They stay in `[E, out, in]` orientation**, unlike every rank-2 projection, which loaders transpose
  to `[in, out]`. Transposing them is actively harmful — it is what makes the Q4_0 repack group across
  `out`, where no grid exists (see `QUANTIZATION.md`).

**Prefill does 16x the expert FLOPs.** Decode gathers only the 8 selected banks per token; prefill
cannot — at T tokens that would re-read T x 8 banks — so it dequantizes the whole 128-expert bank once
and masks. Optimal in bytes moved, 128/8 = **16x wasteful in FLOPs**. That ratio is arithmetic from E
and K, not a measurement. Fixing it needs expert-sorted dispatch, deliberately not attempted because
capacity-padded dispatch **drops tokens** when an expert is oversubscribed — a silent drop being
precisely the failure mode this codebase keeps producing.

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

## 31B — the residual stream is clamped at both ends

Behavioural structure measured on `gemma-4-31B-it-qat-w4a16-ct`, v6e-1, 2026-08-01. Workings in
`tpu-jax-v6e1-31b-w4a16/benchmarks/runs/2026-07-31-gemma4-31b-v6e1/MODEL-INTEL.md`. The config facts
are in the §31B section above; this is what the model *does* when it runs.

Each layer ends with `h *= layer_scalar`, one learned scalar per layer over the whole residual stream.
It is a **three-phase clamp**, not a smooth schedule: ~10-14x suppression at layers 0-1, essentially
pass-through at 2-4, and **~31x suppression at layer 59** (`layer_scalar` = 0.0317). Full-attention
layers are damped 14% harder than sliding ones (0.6891 vs 0.8010).

It exists because **the pre-norms are enormous**: `input_layernorm` averages 23.5 with individual
channels reaching **1248**, `pre_feedforward_layernorm` averages 40.8 — while the post-norms are
sub-unit (0.47, 0.74). Each block amplifies hugely going in and is scaled back coming out.

Consequently the residual stream **shrinks** with depth — RMS 10.06 at layer 0 to 1.239 at layer 59,
**0.12x**. Most transformers grow it.

### Massive activations, and why the first two tokens are load-bearing

- **max/median ratio in the residual stream: mean 1,214x, peak 15,665x** (layer 7).
- **78% of layers put their peak activation on `<bos>` or `<|turn>`** — position 1 in 32/60 layers,
  position 0 in 15/60. Two channels (ch3770, ch1682) account for 57% of layers.
- The sinks are **sparse in (position, channel) space** — mean |h| at position 0 is only 1.3x the other
  positions. Whole positions are not large; specific channels at those positions are.
- **The norm-weight outlier channels and the activation outlier channels do not overlap.** Static
  top channels are ch1081/1400/1924/4206 (pre-norms) and a tight band at ch33-47 (post-norms); runtime
  ones are ch3770/1682/2130/3067. You cannot predict one population from the other.

Three consequences:

1. **Per-tensor activation quantization will not survive this model.** A 15,665x in-tensor dynamic
   range leaves nothing for the other channels. Per-(batch, head, position) KV scales are the right
   shape, and this is the reason.
2. **Do not trim or drop the leading tokens**, and never evict positions 0-1 from the **full**
   layers' cache — layer 11 spends **44% of its entire attention budget** there.
3. **Windowing the sliding layers' KV is provably safe.** A sliding layer is masked to
   `(p - 1024, p]` regardless of `window_kv` — the ring buffer is memory, the mask is semantics — so
   past 1024 tokens **50 of 60 layers cannot attend to position 0 at all**. Measured sink mass on
   sliding layers is *exactly* 0.000000, against 0.2365 on the full layers.

That last point explains the rest: **the 10 full-attention layers are the model's only sink pathway and
only long-range pathway beyond 1024 tokens**, which is why they produce 33-37% larger outputs and are
damped harder. The design tension is that this pathway runs through the *narrowest* KV in the model —
4 KV heads against the sliding layers' 16, plus `attention_k_eq_v`. Most load-bearing, least capacity,
no redundancy. If a mixed-precision KV scheme is ever worth doing on this model, **this** is the split
that matters.

### Output is a point mass on real prompts

top-1 probability **1.0000**, entropy **0.000 nats**, top-1-to-top-2 margin **15.28**, max logit 27.9
against the 30.0 softcap (so `tanh` is well into saturation without clipping any logit).

**Greedy decoding is therefore extremely stable on real input and meaningless to evaluate on random
tokens**, where the margin collapses toward zero and any numerical perturbation flips the argmax. A ~3%
fusion-drift divergence between two `window_kv` settings looked like a cache-correctness bug and was
this. Hold `window_kv` fixed across any A/B — greedy decoding is not bit-reproducible across it.

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

**The int4 column UNDER-predicts by ~19%, and the reason generalises to every size.** MEASURED
2026-08-23 on a T4G (`gpu-jax-g5g-2b/docs/larger-models-on-t4g.md`): E2B at `ple_bits=4` came to
**3.054 GB against the 2.4 GiB (= 2.58 GB) this table predicts**. Mind the units — the int8/int4
columns quarter the **GiB** column, not the GB one, and comparing 3.054 GB against "2.4" as though it
were GB turns a 19% error into an apparent 27% one. Quartering everything is wrong in two places at
once — **`embed_tokens` stays at the storage dtype** rather than being quantized, and **the per-group
scales are extra**. Budget an int4 target as:

```
quartered transformer weights  +  embed_tokens at full width  +  per-group scales
```

not as `bf16 / 4`. The bf16 column is the safer of the two: it **over**-predicts, by ~6-9% (E2B
measured 8.97 GiB on a v5e-1 and 9.257 GB on a T4G against 10.2 GB here). **For sizing a new rig,
treat the bf16 column as a ceiling and the int4 column as a floor** — the error runs in opposite
directions, so a plan that survives both is safe and one that needs the int4 figure to be exact is not.

### On llama.cpp, 58% of the E2B GGUF never reaches the accelerator

**MEASURED 2026-09-03 on `local-llamacpp-1650ti-2b-q4_0`** (GTX 1650 Ti, 4096 MiB). This is a
property of how llama.cpp loads E2B, so it holds wherever llama.cpp serves this checkpoint.

`google/gemma-4-E2B-it-qat-q4_0-gguf` is 3.334 GB of tensor data across 541 tensors, and
`per_layer_token_embd.weight` alone is **1.927 GB of it — 58%**. That tensor is
`[8960, 262144]`: `hidden_size_per_layer_input=256` x 35 layers = 8960, against
`vocab_size_per_layer_input=262144`. It is the per-layer-embedding table this file describes
above, and it is most of the gap between E2B's ~5B total and 2B effective params.

**llama.cpp does not make it resident.** `src/models/gemma4.cpp` creates it with
`TENSOR_READ_LAZY` — "read rows on demand instead of loading whole tensor; requires mmap for now"
(`src/llama-model-loader.h`) — and it is a `GGML_OP_GET_ROWS` lookup rather than a matmul
(`src/llama-arch.cpp`), so rows are pulled from the mapped file on the host as tokens need them.

Measured, `-ngl 99 -c 8192 -ctk f16 -ctv f16`:

```
1342 MiB  resident weights (3.334 GB total - 1.927 GB lazy PLE)
+ 144 MiB  KV, 8192 tokens x 18 KiB/token
+ ~130 MiB  CUDA context + compute buffers
= 1616 MiB predicted        1618 MiB measured (nvidia-smi, per-process)
```

**Two consequences, and both have already caught a rig:**

- **Never size a llama.cpp E2B deployment from the file size on disk.** 3.35 GB against a 4 GiB
  card reads as "will not fit"; the real requirement is ~1.6 GiB and it fits with 2.3 GiB spare.
  A rig that lowers `--n-gpu-layers` on the strength of the disk figure is slower for no reason.
- **`--no-mmap` inverts the result.** `TENSOR_READ_LAZY` requires mmap, so disabling it forces the
  1.93 GB tensor to be materialised and turns a comfortable fit into an OOM.

**This corrects a derived figure in `gpu-llamacpp-g5g-2b-q4_0`**, whose table gives
*Resident: 3.35 GB* for this artifact and computes "freeing ~6.9 GB of a 14.07 GB budget" from it.
Resident is ~1.4 GB and the freeing is ~8.8 GB. Note that rig's *other* row is right and agrees
exactly: 1.407 GB "streamed per decode step" is the same set of tensors under the correct label.

**Do not extrapolate the 58% to other sizes.** It is a property of the `E` checkpoints — 12B, 26B
and 31B have `hidden_size_per_layer_input=0` and no such tensor at all (see the family table
above), so their GGUF footprint is resident in full.

**The `E` prefix is load-bearing.** E4B is *not* a 4B dense model — 4.5B effective, 8.0B total. Reading
`E4B` as "4B" understates its weights by roughly 2x, which is the difference between fitting a 16 GB
accelerator and not.

**Sparse ≠ small on disk.** The 26B's ~4B *active* parameters set its compute cost, not its memory: all
26.5B must be resident because any token can route to any expert. It is the largest checkpoint here after
the 31B, and the "A4B" in the name describes throughput, not footprint.

### Resident is not streamed — the number a decode roofline needs

**A decode ceiling is measured bandwidth ÷ bytes STREAMED per token, and on the `E` and `A` sizes that
is nowhere near the resident footprint.** Dividing the resident figure by bandwidth understates the
ceiling, and the error is large enough to invert a conclusion: it did exactly that in
`gpu-pytorch-g5g-2b` on 2026-08-29, where a 31.3 tok/s "floor" made a rig look near the hardware limit
when it was at 13% of it, and briefly made a sibling's legitimate measurement look impossible.

**E2B, computed from the config fields above and cross-checked against measurement:**

| Component | Params | fp16 | Streamed per decode step? |
| :--- | ---: | ---: | :--- |
| Transformer matmuls, 35 layers | 1.854 B | 3.709 GB | yes |
| LM head / `embed_tokens` (tied) | 0.403 B | 0.805 GB | yes — full-vocab matmul |
| **PLE table** (262144 x 256 x 35) | **2.349 B** | **4.698 GB** | **no — indexed gather** |
| | | **4.514 GB streamed** | of 10.209 GB resident |

**Do not forget `use_double_wide_mlp`.** E2B sets it `true`, which **doubles `intermediate_size`
on the `num_kv_shared_layers` (20 of 35) layers** — see `tpu-jax/ports/gemma4/jax_e_model.py` and
tpu-inference's `test_double_wide_mlp`. Using the plain `3 x intermediate_size x hidden_size` MLP
for all 35 layers understates the streamed figure by 1.13 GB, and was the first error made when
this section was written.

**The arithmetic reconciles against a measurement.** Text-only total is 1.854 + 0.403 + 2.349 =
4.606 B params = **9.212 GB**, against the JAX sibling's **measured 9.257 GB** of text-only
resident weights — **0.49% apart**. That reconciliation is the check that catches a missed
structural field; without it the double-wide error is invisible.

**Confirmed two ways, from a rig that changed it and measured.**
`gpu-jax-g5g-2b/benchmarks/runs/2026-08-26-quant-levers-fixed-g5g/` quantised the PLE table and
recorded `ple0` → `ple4` at **−3.505 GB** of resident weights (9.257 → 5.752); the arithmetic above
predicts −3.523 GB, 0.5% off. And **decode did not move**: 12.80 / 12.80 / 12.80 across a 38%
reduction in resident weights. That report's conclusion — *"the table is a gather, never a matmul, so
decode never streams it"* — is the general rule, not a JAX-port detail.

A third check on the same rig: `int8_lm_head` halves the tied head (805 → 403 MB, −11.9% of streamed
bytes) for **+2.3%** throughput (12.80 → 13.10), not the ~13% a bandwidth-bound decode would give. On
T4G none of the three runtimes tested is actually bandwidth-bound at `B=1`.

**Per size — and note what is NOT known:**

| Size | PLE? | Resident vs streamed |
| :--- | :--- | :--- |
| **E2B** | yes | **4.514 GB streamed of 10.209 GB resident** — computed and cross-checked above |
| **E4B** | yes | **Same trap, magnitude UNRECORDED.** `hidden_size_per_layer_input` and `intermediate_size` are dashes in the family table, so this cannot be computed here. Read them off `config.json` before building a roofline. |
| 12B, 31B | no | resident ≈ streamed; the plain division is correct |
| **26B A4B** | no | **Same trap, different cause.** ~4B active of 26.5B resident — decode gathers only the 8 selected expert banks per token (see §26B). All 26.5B must be resident; far less streams. |

**The `E` prefix is load-bearing in two directions.** It understates weights when you read `E4B` as
4B (above) — and it *over*states streamed bytes when you feed the resident figure into a roofline.
`RIG-ANALYSIS.md` carries the method rule.

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
