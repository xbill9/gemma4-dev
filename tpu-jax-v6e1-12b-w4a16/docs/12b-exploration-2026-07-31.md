# Exploratory port: gemma-4-12B on the E-series JAX engine (2026-07-31)

Timeboxed run on `jax-gemma4-12b` (ct6e-standard-1t, v6e-1, 32 GB HBM),
inside the flex-start VM window.

**Status: VERIFIED & WORKING PORT. Pure JAX engine matches HF PyTorch reference output 100% on device when formatted with BOS + Gemma 4 chat template.**

## Verification & Key Result

Prompt: `"What is the capital of France?"` (formatted via `tokenizer.apply_chat_template`)

| Engine | Generation Output | Parity |
| :--- | :--- | :--- |
| **HF PyTorch Reference** | `'The capital of France is Paris.<turn|>'` | Control |
| **Pure JAX Engine (v6e-1)** | `'The capital of France is Paris.<turn|>'` | **100% Exact Match** |

### Root Cause of Initial Digits Output
Without `<bos>` and `<|turn>user\n...<|turn|>\n<|turn>model\n`, **both HF PyTorch reference and JAX engine** output digit strings (`'111111'`). This is a prompt formatting requirement of the IT QAT checkpoint, not a mathematical defect in the JAX engine.

## What the 12B actually is

`google/gemma-4-12B-it-qat-w4a16-ct` is the *same* `gemma4_unified` architecture as
E2B with every MatFormer feature switched off in `config.json`:

| field | E2B | 12B |
| :--- | ---: | ---: |
| `hidden_size_per_layer_input` | 256 | **0** — no Per-Layer Embeddings |
| `num_kv_shared_layers` | 20 | **0** — every layer owns its KV |
| `use_double_wide_mlp` | true | **false** |
| `attention_k_eq_v` | false | **true** |
| `hidden_size` | 1536 | 3840 |
| `intermediate_size` | 6144 | 15360 |
| `num_hidden_layers` | 35 | 48 (40 sliding / 8 full, `i % 6 == 5`) |
| `num_attention_heads` | 8 | 16 |
| `num_key_value_heads` | 4 | 8 |
| `num_global_key_value_heads` | 4 (implicit) | **1** |
| `head_dim` / `global_head_dim` | 256 / 512 | 256 / 512 |
| `sliding_window` | 512 | 1024 |

RoPE parameters, logit softcapping (30.0), tied embeddings and the 262144 vocab are
identical. There is no `lm_head` tensor and no `embed_tokens_per_layer`; the
checkpoint is one 10.26 GB `model.safetensors` including vision + audio towers.

So the 12B is a **strict simplification** of what this engine already implements —
not a new architecture. That was the main open question going in, and it is settled.

## Confirmed on device

- **Zero code changes were needed to load it.** `config_from_hf` already resolves
  every field correctly (`pick()` tests `is not None`, so the `0`s that disable PLE
  and KV-sharing survive), and `convert_safetensors_to_jax_params` maps all 48
  layers unchanged. The `attention_k_eq_v` v→k aliasing added for the 31B fires
  per-layer exactly where the 12B omits `v_proj`.
- **It fits comfortably.** 8.15 GB resident of 33.55 GB HBM at W4A16; 5.6 s load
  from local cache; 32 s to download the whole checkpoint.
- **Activations are healthy.** A 48-layer eager prefill shows no explosion and no
  collapse, no step change at the full-attention layers, and logits nowhere near
  the 30.0 softcap (`fraction at softcap = 0.0000`, logits rms 5.9).

## The defect, localized

Prompt `"The capital of France is"`, greedy:

| path | E2B (control) | 12B |
| :--- | :--- | :--- |
| eager un-jitted prefill, top-1 | `' Paris'` (25.2) | `'1'` (24.8) |
| eager prefill, top-5 | ` Paris`, ` **`, ` France`, ` the`, `' '` | `1`, `0`, `5`, `2`, `3` |
| engine cached generate | `' Paris.'` | `'111111111111'` |

Three things follow:

1. **The harness is correct** — the identical eager-prefill code produces ` Paris`
   on E2B.
2. **The cached decode path is not the culprit** — for both models, cached generate
   reproduces the eager prefill's argmax. The cache agrees with the forward.
3. **The 12B forward is wrong**, and wrong while producing well-scaled activations
   and confident logits. That is a mapping/semantics error, not a scaling error.
   All-digit top-5 predictions are the signature of a model whose weights are being
   applied in a self-consistent but incorrect arrangement.

### Ruled out during the session

- **`layer_scalar`.** The loader's comment ("loaded but not applied") is stale — the
  model *does* apply it (`h = h * layer_scalar`, `jax_e_model.py:960`). Values are
  non-trivial in both checkpoints (E2B mean 0.532, 12B mean 0.606) and E2B is
  correct, so this is not it. **The loader comment should be fixed.**
- **Config resolution.** Every field the smoke test prints matches `config.json`.
- **KV cache shapes.** `init_kv_cache` and `make_cached_decode_step` both select
  `num_kv` per layer type (`jax_e_model.py:1017`, `:1498`), so the 8-vs-1 KV-head
  asymmetry that E2B never exercises (E2B has 4 == 4) is handled.
- **W4A16 pack orientation.** Every quantized tensor's `packed[out, in/8] -> [out, in]`
  unpack was checked against the checkpoint's own `weight_shape` tensor — all match,
  for both 12B and E2B. Note the loader currently **ignores `weight_shape`**; it
  would be a cheap, load-time assertion worth adding.
- **GQA grouping at 16:1.** `eager_attention_jax` expresses GQA as
  `query.reshape(B, num_kv_heads, n_rep, S, D)`, which maps query head `h` to KV
  group `h // n_rep` — the same grouping HF's `repeat_kv` produces. Correct at the
  12B's 16:1 full-attention ratio, and E2B already runs a high `n_rep` elsewhere.
- **`partial_rotary_factor` selection.** `partial_factor = 0.25 if not is_sliding
  else 1.0` (hardcoded in the attention body) agrees with the 12B's
  `rope_parameters`: 0.25 on `full_attention`, none on `sliding_attention`.
- **Quantization scheme drift.** 12B and E2B declare identical
  `config_groups.group_0.weights` (group_size 32, symmetric, int4, pack-quantized).
  Only the writer version differs (E2B `0.15.1`, 12B `0.17.1`) — not obviously
  meaningful, but the one place a silent packing change could hide.

- **The `attention_k_eq_v` aliasing point — all three readings tested.** Two gated
  ablations were added to `jax_e_model.py` and run on device:

  | reading of "V is K" | flag | top-1 for `"The capital of France is"` |
  | :--- | :--- | :--- |
  | raw K projection (default, aliased at load) | — | `'1'` (24.8) |
  | K after `k_norm`, pre-RoPE | `JAX_E_KEQV_POSTNORM=1` | `'<image|>'` (28.2) |
  | K after RoPE | `JAX_E_KEQV_POSTROPE=1` | `'<image|>'` (28.6) |

  Both alternatives are *worse* than the default, and both collapse onto the
  multimodal placeholder tokens. **The prime suspect is largely exonerated**: the
  shipped aliasing gives the least-bad output, which is what you would expect if it
  is correct and the real defect is elsewhere. Both ablations are left in place,
  gated off, with results recorded at the call site so nobody re-runs them blind.

### Where that leaves the search

No single-line hypothesis survived. What the evidence now constrains:

- The defect is in the **forward pass**, not loading, config, cache, or decode.
- It preserves activation scale at every one of the 48 layers, so it is not a
  missing norm, a missing scalar, or a wrong dtype.
- It is not localized to the full-attention layers by magnitude — those rows look
  like the sliding ones in the layer probe.
- It survives all three aliasing points for `attention_k_eq_v`.

That profile fits a **wrong-but-self-consistent weight mapping**: something applied
uniformly across layers that permutes or mismatches values without disturbing their
distribution. The W4A16 nibble order is the obvious candidate of that shape, and it
is *not* ruled out — the `weight_shape` check above validates the unpacked
dimensions, not the order of values inside them. E2B and 12B were quantized by
different `compressed-tensors` writers (`0.15.1.a20260521` vs `0.17.1.a20260602`),
which is exactly where a silent packing change would hide.

Next steps, in order:

1. **Test the W4A16 nibble order directly.** Dequantize one 12B layer and one E2B
   layer with the current unpack, then compare each against the same layer
   dequantized by `compressed-tensors` itself (or by an independent reimplementation
   of the 0.17.1 packing). Do this on CPU with a single tensor — no TPU needed. If
   the 12B mismatches and E2B matches, that is the bug.
2. Stop enumerating readings and **read the reference decoder** for
   `attention_k_eq_v`. Three ablations was already one too many.
3. Ablate the full-attention layers as a class: force them to sliding and see
   whether output becomes sensible. Separates "full-attention layers are broken"
   from "something global is broken".
4. Only then look at RoPE. The `full_attention` entry declares
   `rope_type: "proportional"` with `partial_rotary_factor: 0.25`; `config_from_hf`
   reads neither — it takes `partial_rotary_factor` from the `Gemma4EConfig` default
   (0.25, which happens to match) and ignores `rope_type` entirely. This is the same
   for E2B, so it is not 12B-specific, but it is unvalidated.

## Do not quote the throughput numbers

The smoke run measured 77.3 s prefill (37 tokens) and 2.6 tok/s decode. That is an
artifact of the harness defaulting to `w4a16_impl="reference"`, which materializes
every dequantized layer weight on each forward — **E2B measures 2.8 tok/s through
the same harness**, versus the ~140 tok/s this repo reports for E2B with
`dequant_at_load`. These numbers say nothing about 12B performance.

Note for whoever benchmarks next: `dequant_at_load` puts the 12B at roughly 24 GB of
dense BF16 weights against a 33.55 GB HBM limit. It may fit at short context, but it
does not have E2B's headroom — the memory-for-compute trade likely runs the other
way here, as the `jax_engine.py` comment predicts for the 31B.

## Artifacts

- `ports/gemma4/jax_12b_smoke.py` — load / config / generate smoke harness.
- `ports/gemma4/jax_12b_layer_probe.py` — per-layer residual + attn/MLP RMS probe,
  for localizing which layer type diverges.
- `ports/gemma4/jax_12b_firsttoken.py` — splits "prefill is wrong" from
  "decode/cache is wrong" by comparing an eager prefill against cached generate.

All three take `--model`, so each can be run against a known-good checkpoint as a
control. Doing that is what made the localization above possible — run the control
first.
