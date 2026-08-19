# Gemma 4 E2B quirks

**Part I (1–15) — the architecture.** Every entry is checked against
`transformers.models.gemma4.modeling_gemma4` (transformers 5.12+). That module is
readable locally — **no torch install, no TPU, no checkpoint download needed** — and
reading it is faster and more reliable than inferring architecture from tensor
shapes. Five of those bugs were found the slow way first; the rest by diffing.

**Part II (16–21) — this engine's serving path.** Not in the reference; each entry
says how it was checked instead. Two of them are live wrong-output bugs found with
the whole test suite green; 21 is the opposite — the things that look wrong and are
not, with the reason, so they stop getting re-investigated.

Status legend: **✅ verified** against the reference (Part I) or by direct
experiment (Part II) · **⚠️ inferred** from shapes or measurement only.

---

## 1. The checkpoint is multimodal ✅

The text decoder lives under `model.language_model.`, alongside `model.audio_tower.`
(751 tensors) and vision towers (659). A loader that assumes a bare `model.` prefix
finds **nothing** — and `get_arr()` returning `None` for every key produces a
parameter tree that "loads" in seconds and holds zero bytes.

`jax_e_loader.detect_text_prefix()` tries `model.language_model.`, `language_model.`,
`model.` and raises listing the candidates if none matches.

## 2. Sandwich norms ✅

`post_attention_layernorm` normalizes the attention **output**, before the residual
add — it is *not* a pre-norm for the MLP. The feed-forward block is wrapped on both
sides:

```
residual = h
h = attn(input_layernorm(h))
h = residual + post_attention_layernorm(h)

residual = h
h = mlp(pre_feedforward_layernorm(h))
h = residual + post_feedforward_layernorm(h)
```

Getting this wrong still runs and still emits fluent-looking tokens. They are just
the wrong tokens.

## 3. `layer_scalar` scales the whole residual stream ✅

Each decoder layer ends with `hidden_states *= self.layer_scalar` — a single learned
scalar per layer, applied to the **entire** hidden state after every residual add
(including the PLE injection), not to the layer's delta.

It is the counterweight to this checkpoint's large RMSNorm weights (`final_norm`
mean ≈ 14, max 118). Without it the residual stream grows layer over layer and the
output logits pin against the ±30 softcap. Observed values on E2B run 0.02–0.79.

## 4. RMSNorm is `x_normed * weight` ✅

**Not** the `x_normed * (1 + weight)` convention used by earlier Gemma generations.
The weights here are not zero-centred (means of +5 to +19 are normal), which is the
quick way to tell the two conventions apart.

`v_norm` is constructed `with_scale=False` — it normalizes with no weight at all, and
the checkpoint ships no `v_norm` tensor. Passing `weight=None` to an RMSNorm that
multiplies only when a weight is present reproduces this.

## 5. Attention: no score softcap, and `scaling = 1.0` ✅

- The config declares `final_logit_softcapping: 30.0` and **no**
  `attn_logit_softcapping`. Gemma 3+ dropped softcapping of attention *scores*;
  applying it saturates `tanh` and destroys the attention distribution.
- The reference sets `self.scaling = 1.0` — **not** `head_dim ** -0.5`. `q_norm` and
  `k_norm` already normalize query and key before the dot product, so the usual
  `1/sqrt(d)` is not applied on top. (I "fixed" this once by adding it. That was the
  regression, not the fix.)
- `q_norm`/`k_norm` are applied **before** RoPE.

## 6. RoPE: concatenated frequency layout, partial rotary by masking ✅

`rotate_half` pairs channel *i* with *i + d/2*, so the frequency table must be
`cat(freqs, freqs)`. Building it with `repeat_interleave` pairs every channel against
the wrong frequency — the model still generates, but it echoes and repeats.

Per-layer-type RoPE comes from a nested `rope_parameters` dict:

| layer type | `rope_theta` | `rope_type` | `partial_rotary_factor` |
| --- | ---: | --- | ---: |
| `sliding_attention` | 10,000 | `default` | 1.0 (absent) |
| `full_attention` | 1,000,000 | `proportional` | 0.25 |

`proportional` keeps the **full** `head_dim` and zeroes the inverse frequencies past
the factor, rather than slicing channels: with `global_head_dim` 512 and factor 0.25,
the first 64 frequency pairs are rotated and the remaining 192 get `inv_freq = 0`
(i.e. `cos = 1`, `sin = 0`, identity). Masking the frequency table is equivalent and
keeps the `cat(freqs, freqs)` pairing intact.

For text-only inputs `apply_multidimensional_rope` falls through to standard
`apply_rotary_pos_emb` — the "multidimensional" path is for image/audio position dims.

## 7. KV sharing is keyed by layer *type* ✅

`first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers` (35 − 20 = 15
on E2B). Layers at or above it carry **no** `k_proj`, `v_proj` or `k_norm` — the
omission upstream's loader mishandles ([tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225)).

The reference stores `shared_kv_states[layer_type] = (k, v)` and every non-shared
layer overwrites its type's entry, so a shared layer reads the **last non-shared
layer of its own type**. Since E2B interleaves sliding and full attention, there are
two independent sources. `Gemma4EConfig.kv_share_map()` computes exactly this.

## 8. Per-Layer Embeddings ✅

Two components, combined:

```
token identity  = embed_tokens_per_layer[ids] * sqrt(D_ple)     -> [B,S,L,D_ple]
context         = per_layer_model_projection(inputs_embeds) * hidden_size**-0.5
                  reshaped to [B,S,L,D_ple], then per_layer_projection_norm
per_layer_input = (context + token_identity) * 2**-0.5
```

Verified constants: `embed_scale = hidden_size**0.5`, PLE `embed_scale =
hidden_size_per_layer_input**0.5`, `per_layer_model_projection_scale =
hidden_size**-0.5`, `per_layer_input_scale = 2**-0.5`.

The trap: `per_layer_projection_norm` has shape **`[D_ple]`**, so the norm applies to
the reshaped `[B,S,L,D_ple]` tensor per layer-slice — *not* across the flattened
`[L*D_ple]` projection.

In the QAT checkpoint all three PLE projections (`per_layer_model_projection`,
`per_layer_input_gate`, `per_layer_projection`) ship **W4A16-packed**, not dense. The
global one is the exact tensor #3225 reports as unimplemented in vLLM's TPU loader.

## 9. Double-wide MLP on shared layers ✅

`use_double_wide_mlp=True` means KV-shared layers use `2 * intermediate_size`
(12288 = 2 × 6144 on E2B). Confirmed in the checkpoint: layer 20's `gate_proj` is
`[12288, 192]` packed.

## 10. The tokenizer does not add BOS ⚠️

`tok("hello")` → `[23391]`, no `<bos>`. Gemma expects one, and without it the model
**echoes the prompt** instead of answering. `JaxGemmaEngine` now prepends it.
(Measured behaviour, not read from the reference.)

## 11. Config values that differ from intuition ✅

| field | value | note |
| --- | ---: | --- |
| `hidden_size` | 1536 | not 2048 |
| `num_key_value_heads` | 1 | a single 256-dim KV head |
| `num_global_key_value_heads` | `null` | falls back to 1 |
| `sliding_window` | 512 | 28 of 35 layers (`i % 5 != 4`) |
| `tie_word_embeddings` | true | `lm_head.weight` is a materialized copy |
| `layer_types` | explicit list | full attention at 4, 9, 14 … (`i % 5 == 4`) |

## 12. `attention_k_eq_v`: V *is* K, and the checkpoint omits `v_proj` ✅

Set `True` on **gemma-4-31B**, `False` on E2B. Where it is set, the affected layers
ship no `v_proj` at all — one projection feeds both K and V.

Verified by reading the 31B checkpoint on a CPU box: all ten **full-attention**
layers (`i % 6 == 5` in its 60-layer `[s,s,s,s,s,f]` pattern) carry `q_proj`,
`k_proj`, `k_norm`, `o_proj` and **no `v_proj`**, while every sliding layer carries
both. Shapes at layer 5 confirm the geometry — `k_proj` is `[2048, 672]` packed,
i.e. `num_global_key_value_heads` (4) × `global_head_dim` (512), and `k_norm` is
`[512]`.

Loading the 31B without handling this yields exactly ten missing tensors and, in a
loader that tolerates `None`, a silently broken model. `jax_e_loader` aliases V to
K (the same arrays, not copies) when `Gemma4EConfig.attention_k_eq_v` is set.

The KV cache still stores K and V separately, which is redundant but correct.
Collapsing it would save one of the two planes on those layers — worth ~4.5% of the
31B's KV, since sliding layers dominate its budget.

**This does not explain E2B's KV-bytes discrepancy below.** E2B sets the flag
`False` and ships `v_proj` on all fifteen non-shared layers, checked key by key.

## 13. `use_bidirectional_attention` is a vision setting ✅

`"vision"` on the 31B, absent/`null` on E2B. It selects bidirectional attention for
image tokens; text decoding is unaffected, so the causal-only text path is correct
for both. `store_full_length_kv` is **not present in either config** — it is a
reference-implementation concept, not a checkpoint field.

## 14. KV is 18.0 KiB/token, and full-attention layers really are 512-dim ✅

Settled by reading the checkpoint. The three full-attention layers among the fifteen
that own KV (`i % 5 == 4`, so L4/L9/L14) carry a **512-wide** K projection, and their
`k_norm` is `[512]` to match:

| layer type | count | `k_proj` out | `k_norm` | KV head dim |
|---|---:|---:|---:|---:|
| `sliding_attention` | 12 | 256 | `[256]` | 256 |
| `full_attention` | 3 | **512** | **`[512]`** | **512** |

So `global_head_dim` is the KV head dim on those layers, not merely the query head
dim. `init_kv_cache` allocates **18.00 KiB/token** at every context length, matching
`12 × 1 × 256 × 2 + 3 × 1 × 512 × 2 = 9,216` elements × 2 B exactly.

An earlier note recorded a 15.0 KiB/token reading and concluded our estimates were
"~20% pessimistic". That was backwards. 15.0 KiB is precisely what a **uniform
256-dim** assumption produces (`15 × 1 × 256 × 2 × 2 B`), and the checkpoint
contradicts it. Our figure is the correct one; nothing needs adjusting.

Worth checking upstream: an allocator that sizes E2B's KV uniformly at `head_dim`
would under-provision the three full-attention layers by half. The provenance of
that 15.0 KiB reading is not recorded here, so this is a lead rather than a report.

## 15. Unresolved ⚠️

- **`store_full_length_kv` behaviour.** The reference marks the last non-shared layer
  of each type as storing full-length KV. Our windowed-KV ring windows every sliding
  layer including the source. Self-consistent (windowed and full-length outputs match
  in `tests/test_windowed_kv.py`), but whether it matches Gemma's intent for shared
  sliding layers is unverified.

## How to check the next one

```python
import transformers.models.gemma4.modeling_gemma4 as m; print(m.__file__)
```

Read it before inferring anything from tensor shapes. Every fix in sections 2–6 came
from that file after hours of guessing produced nothing but plausible-looking garbage.

---
---

# Part II — serving-path quirks

Sections 1–15 are properties of the *model*. What follows are properties of *this
engine* — `jax_engine.py`, `jax_openai_server.py` and the mask/cache plumbing in
`jax_e_model.py`. They are not in the reference, and the reference cannot settle
them; each entry below states how it was checked here.

Every one of these was live with the full suite green (127 tests, 12 modules, all
passing as of 2026-07-31). That is the theme: the suite is built almost entirely out
of **parity assertions between two of our own code paths**, and a wrong assumption
shared by both paths is invisible to every one of them.

## 16. Bucket padding is charged against the sliding window ⚠️ **wrong output, silent**

Verified by experiment (below) and by reading the reference mask builder.

Positions are tracked in two different spaces and only one of them is used
consistently:

| quantity | space | where |
| --- | --- | --- |
| RoPE `position_ids` at decode | **logical** (`prompt_len + step`) | `jax_engine.py:434` |
| KV write index / `cache_slot` | **slot** (`bucket_s + step`) | `jax_engine.py:434` |
| `valid` mask index | slot | `make_decode_mask` |
| **sliding-window cutoff** | **slot** | `make_decode_mask`, `make_ring_decode_mask` |

The prompt is right-padded to a static bucket, so slot space is logical space plus
`bucket − prompt_len` dead slots wedged between the prompt and the first generated
token. Everything above tolerates that except the last row: the window is a
*distance*, and it is measured across the padding. The reference measures it in
position space — `dist = q_idx - kv_idx` in `sliding_window_mask_function`
(modeling_gemma4.py:1916) — so the padding should not count.

Real prompt tokens still visible to a sliding layer at decode step *k*:

```
visible(k) = max(0, P − (bucket − 512) − k)          P = real prompt length (BOS included)
```

Since `TPUv6eHardwareProfile.static_sequence_buckets` doubles (see quirk 19), the
padding reaches ~50% of the bucket, and the window vanishes entirely:

| prompt `P` | bucket | pads | real prompt tokens the 28 sliding layers see at step 0 |
| ---: | ---: | ---: | ---: |
| 100 | 128 | 28 | 100 — all of it (no windowing below 512) |
| 513 | 1024 | 511 | **1** |
| 600 | 1024 | 424 | 88 of 600 |
| 1025–1536 | 2048 | ≥512 | **0** |
| 2049–3584 | 4096 | ≥512 | **0** |
| 3585 | 4096 | 511 | **1** |

In the blackout ranges every sliding layer — 28 of 35 — attends to nothing but the
token currently being generated. The 7 full-attention layers still carry the prompt,
so the model does not produce noise; it produces fluent text that has largely
forgotten the prompt. With the server's default `max_model_len=4096`, prompts of
1025–1536 and 2049–3584 tokens land in it.

Reproduction (tiny CPU config, `window=4`): run the same 5-token prompt unpadded
(`S=5`) and padded to a bucket (`S=8`, `real_len=5`), decoding the same logical
positions in both.

```
window=4 real_len=5 bucket=8 pads=3
  window_kv=False  unpadded=[93, 43, 98, 122]  padded=[93, 70, 13, 94]  max|dlogit|=1.56
  window_kv=True   unpadded=[93, 43, 98, 122]  padded=[93, 70, 13, 94]  max|dlogit|=1.56
  padded: window_kv False vs True   max|dlogit|=1.3e-06  same tokens   <- what the suite checks
```

Note the third line. `tests/test_windowed_kv.py` asserts the ring cache agrees with
the full-length cache, and it does — **both** measure the window in slot space, so
they agree on the same wrong answer. The prefill logits agree too (token `93`); the
divergence starts at the first decode step, which is where the pads enter the span.

Not the same bug as commit `005eaa3`. That one had windowed layers *attending to pad
K/V*; the fix (consult `valid`) is correct and holds. This is the pads still
*occupying window budget* after being correctly excluded from the softmax.

Fix direction, not yet applied: left-pad instead of right-pad. Put the pads at slots
`0 … bucket−P−1`, the prompt at `bucket−P … bucket−1`, and offset `position_ids` by
`−pad_len` so real tokens keep logical positions `0 … P−1`. Slot distance then equals
logical distance for every real token and the pads fall out of the window on their
own. This is the same reason HF requires `padding_side='left'` for decoder-only
generation. Note that a fix must be verified against an **unpadded** run — no
comparison between two padded paths can see it.

## 17. Chunked prefill takes the last-real-token logits from the final chunk only ⚠️

`chunked_prefill_with_kv_cache` keeps only the last chunk's logits, then indexes
within it (`jax_e_model.py:1386`):

```python
within = prompt_lens - 1 - (S - chunk_size)
within = jnp.clip(within, 0, chunk_size - 1)
```

The comment justifies this with "every row's last real token lives there when
prompts are right-padded to the same bucket" — true *across rows of a batch*, but it
does not put the token in the **final** chunk. Whenever `P ≤ S − chunk_size` — i.e.
the bucket padding is at least one chunk wide, which quirk 19 makes routine — the
subtraction goes negative and the clip silently returns the logits of a **pad
position**. The first sampled token is then garbage, and the rest of the generation
follows from it.

Measured against one-shot prefill on the tiny config:

```
S= 8 real=7 chunk=4   final chunk [4,8) holds token 6   max|dlogit|=1.0e-06  argmax 6  vs 6   ok
S= 8 real=3 chunk=4   final chunk [4,8) holds no real   max|dlogit|=1.45     argmax 62 vs 48  WRONG
S=16 real=5 chunk=4   final chunk [12,16) holds no real max|dlogit|=1.39     argmax 12 vs 36  WRONG
```

`tests/test_chunked_prefill.py` passes because every prompt it builds is fully valid
(`jnp.ones` for `prompt_valid`), so the last real token is always the last slot.

Reachable from `JaxGemmaEngine(prefill_chunk_size=…)` only; `jax_openai_server.py`
exposes no flag for it, so nothing served today hits this. The fix is to keep each
chunk's logits, or re-run the row's last real position, rather than clipping.

## 18. The `prefill_chunk_size` + `window_kv` guard checks the wrong value ⚠️

Chunked prefill writes `chunk_size` contiguous slots at an arbitrary offset, which a
shorter ring buffer would wrap — so the two are mutually exclusive, and
`JaxGemmaEngine.__init__` rejects the combination:

```python
if prefill_chunk_size is not None and window_kv:      # jax_engine.py:238
```

`window_kv` there is the **constructor argument**, whose default is `None` meaning
"decide at load". `load()` then resolves it (`jax_engine.py:319`) to `True` whenever
`max_model_len > sliding_window`. So the default path —
`JaxGemmaEngine(prefill_chunk_size=256)`, `max_model_len=4096` — sails past the guard
and builds a full-length cache with a windowed decode step.

It fails loudly rather than silently, which is the one piece of luck here:

```
TypeError: add got incompatible shapes for broadcasting: (1, 4, 1, 12), (1, 1, 1, 4)
```

— the ring mask is `min(total, window)` keys wide while the cache is full-length. The
check belongs after the resolution in `load()`, not on the raw argument.

## 19. The "128-aligned" bucket ladder is a doubling ladder ✅

`TPUv6eHardwareProfile.static_sequence_buckets` is `(64, 128, 256, 512, 1024, 2048,
4096, 8192)` and `get_nearest_bucket` returns the first entry `>= seq_len`. "Nearest"
is a misnomer: the next bucket is the next **power of two**, so padding runs up to
just under 50% of the bucket. A 513-token prompt is padded to 1024.

Two consequences, one benign and one not:

- prefill computes over the whole bucket, so that 513-token prompt pays a
  1024-token prefill, and the KV cache is allocated at `bucket + max_new_tokens`;
- it is what makes quirk 16 catastrophic rather than merely lossy. A ladder in steps
  of 128 would cap the window loss at 127 tokens instead of 2047.

The doubling ladder is a real trade — it bounds the number of XLA retraces to eight
— but the padding it buys is charged somewhere, and right now part of the bill lands
on the sliding window.

## 20. `convert_tokens_to_ids` does not raise on an unknown token ✅

It returns `unk_token_id`, which is a perfectly ordinary non-negative int. A guard of
the form `if isinstance(tid, int) and tid >= 0` therefore accepts it. In `_eos_ids`
that put `<unk>` in the stop set while leaving the real turn terminator out of it, so
generation ran to `max_tokens` on every request. Fixed in `35459ab`/`bb5a5f1` by
comparing against `unk_token_id`; the trap generalizes to any vocab lookup by name.
The spelling differs by checkpoint (`<end_of_turn>` on some, `<turn|>` on the QAT E2B
ones), which is why the lookup is attempted at all rather than hardcoded.

`_eos_ids` also folds `pad_token_id` into the stop set. Harmless here — the engine
pads inputs with id 0 and pad ids are masked out of attention rather than sampled —
but it is a deliberate choice, not an accident, and it means a checkpoint whose
`pad_token_id` collides with a real generable token would truncate.

## 21. Audited and found correct ✅

The negative space of the same pass. Each of these looks like a bug, or sits next to
one, and is not — recorded so the next reader does not spend the afternoon I did.

- **`make_ring_decode_mask`'s gather** (`jax_e_model.py:376`). `pos = s - ((s - r) mod
  window)` is the most recent absolute position living in ring slot `r`; the
  `pos >= 0` term kills slots the ring has not reached. Checked against an independent
  simulation that physically writes each position into a ring array — 20,416
  `(window, real_len, bucket, step)` combinations, zero mismatches. (A first attempt
  at this check recomputed the same modular formula and so proved nothing; the
  simulation has to not know the closed form.) This is commit `005eaa3`'s fix and it
  is right — quirk 16 is a *different* defect in the same function's caller, so do not
  read 16 as a regression of this.
- **RoPE stays in logical space** at both prefill and decode, for prompt and generated
  tokens alike, while the cache is indexed in slot space. That split is deliberate and
  correct: rotary phase must reflect what position a token actually *is*, not where it
  was stored. Quirk 16's fix must preserve it. Do not "make them consistent" by moving
  RoPE to slot space — that breaks the one row of the quirk-16 table that currently
  works.
- **Softcap is applied before the mask is added** (`jax_e_model.py:574`, then `:578`).
  The order is load-bearing: `tanh(-1e9 / 30) * 30` is `-30`, a perfectly attendable
  score, so capping after masking would silently un-mask every pad. Moot on E2B, which
  sets `attn_logit_softcapping = 0.0` (quirk 5), but the code path is live for any
  checkpoint that sets it.
- **Quantized-KV scales are folded into the contraction** rather than dequantizing
  K/V first (`:565`, `:583`). Algebraically identical — the K scale is per-key so it
  commutes with the score matmul, and the V scale is applied in f32 before the
  downcast. For grouped-query heads the scale is *reshaped* into KV groups, never
  broadcast up to the query-head count.
- **`gather_ple`'s int4 unpack** (`:1197`) derives its group count from an array shape
  instead of a Python `int` in the params pytree. That is not stylistic: an `int`
  there raises `TracerBoolConversionError` under `jit`.
- **Chunked prefill marks pad slots valid mid-loop** (`:1329`) and restores the real
  mask afterwards (`:1378`). Safe, but only because pads are always to the *right* of
  real tokens, so a chunk that includes pads is the last one and nothing later reads
  them. A left-padding fix for quirk 16 would invalidate exactly this argument — the
  two changes must land together.

## What the test suite can and cannot catch

Worth stating plainly, since quirks 16 and 17 both survived a green suite:

- **Parity between two of our own paths** — windowed vs full-length cache, chunked
  vs one-shot prefill, cached decode vs full re-forward. This is most of the suite,
  and it cannot see an assumption both paths share.
- **Fully-valid prompts.** Almost every test builds `prompt_valid = jnp.ones(...)`.
  Padding is the input that finds these bugs, and it is barely represented —
  `test_right_padded_prompt_does_not_attend_to_pads` (added in `005eaa3`) is the only
  decode-path test that pads at all.

The check that would have caught both: run the same prompt **unpadded** and
**padded**, and assert the logits match. It is not a parity test between two of our
implementations; it is a test against what the prompt actually means.
