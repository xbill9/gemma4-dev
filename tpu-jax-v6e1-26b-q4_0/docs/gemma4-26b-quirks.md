# gemma-4-26B-A4B quirks: the sparse checkpoint

Companion to [`tpu-jax-v5e1-2b/docs/gemma4-quirks.md`](../../tpu-jax-v5e1-2b/docs/gemma4-quirks.md),
which covers the **E2B** architecture against the `transformers` reference. This file covers
**`gemma-4-26B-A4B`**, which is the odd one out twice over: it is the only sparse checkpoint in the
family and the only size with no `-w4a16-ct` release.

Section numbers are kept from the original combined document (they start at 15) so that references
from `benchmarks/runs/2026-07-31-gemma4-26b-v6e1/REPORT.md` still resolve. Sections 1–14 there are
E2B architecture and live in the file linked above.

Verified against `transformers.models.gemma4.modeling_gemma4` (v5.12.1) and against the checkpoint
bytes, on CPU. **Nothing here is measured on a TPU** — see §22.

Status legend: **✅ verified** against the reference or the checkpoint · **⚠️ inferred** from shapes
or measurement only.

---

## 15. The MoE block runs *alongside* the dense MLP, not instead of it ✅

`enable_moe_block: true` does not swap the feed-forward block for an expert bank.
Every layer keeps its ordinary `mlp` **and** gains a 128-expert bank, and the two
outputs are summed before the shared post-norm. That is why the 26B carries
`mlp.gate_proj` / `up_proj` / `down_proj` at full `intermediate_size` (2112) on
top of 128 experts of `moe_intermediate_size` (704).

It also gains three extra norms, for five in the feed-forward block total:

| tensor | normalizes |
| --- | --- |
| `pre_feedforward_layernorm` | input to the **dense MLP** |
| `pre_feedforward_layernorm_2` | input to the **experts** |
| `post_feedforward_layernorm_1` | output of the **dense MLP** |
| `post_feedforward_layernorm_2` | output of the **experts** |
| `post_feedforward_layernorm` | the **sum**, before the residual add |

```
residual = h
dense = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(h) ) )
moe   = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(h) ) )
h     = residual + post_feedforward_layernorm(dense + moe)
h    *= layer_scalar
```

Note both branches read `h` — the post-attention residual — not each other's
output. The dense branch is §2's sandwich, unchanged; the MoE branch is a second
sandwich in parallel with it.

## 16. The router reads the RAW residual; the experts read a normalized copy ✅

The single most dangerous line in this architecture:

```python
hidden_states_flat = residual.reshape(-1, residual.shape[-1])
_, top_k_weights, top_k_index = self.router(hidden_states_flat)      # RAW
hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)  # NORMED
```

Two different tensors. Passing the normalized one to both is the obvious
simplification and it is wrong: the router opens with its own scale-less RMSNorm,
so composing it with a *learned-weight* RMSNorm first does not cancel — it
reweights the channels the router sees and changes which experts fire.

**Measured cost of getting this wrong: 0.36 relative error on a 1-layer model,
with every unit test still green.** Router parity passed. Expert parity passed.
Only end-to-end comparison against `Gemma4TextModel` caught it. Guarded now by
`test_router_reads_the_raw_residual_not_the_normed_one` and
`test_moe_input_is_the_residual_not_the_normed_mlp_input`.

`moe_block_jax` therefore takes `router_in` and `expert_in` as separate
arguments, which looks redundant until you know why.

## 17. Router internals ✅

```
x -> RMSNorm(with_scale=False)          # no weight
  -> * router.scale                     # [H], learned, separate from the norm
  -> * hidden_size ** -0.5              # fixed
  -> router.proj                        # [E, H] -> E logits
  -> softmax over ALL E
  -> top_k                              # k = top_k_experts = 8
  -> renormalize so the k weights sum to 1
  -> * router.per_expert_scale[top_k_index]
```

Order matters at the tail: the per-expert scale is applied **after**
renormalization, so the final weights do *not* sum to 1. Renormalizing afterwards
"to fix" that changes the model.

Softmax-then-top-k-then-renormalize is mathematically a softmax over just the
top-k logits, so either is fine — but only if the per-expert scale stays last.

`jax_e_model.moe_router_jax` computes this in **float32** even when activations
are BF16. The router picks *which experts run at all*, so a tie resolved
differently at BF16 resolution swaps an expert and moves the output
discontinuously, unlike a matmul where BF16 noise stays proportional. 128 logits
per token is nothing to compute in f32.

## 18. Expert tensors ship without a `.weight` suffix, stacked and fused ✅

They are `nn.Parameter`, not `nn.Linear`, so the keys have no `.weight`:

| key | shape | meaning |
| --- | --- | --- |
| `layers.N.experts.gate_up_proj` | `[128, 1408, 2816]` | `[E, 2*moe_inter, hidden]` |
| `layers.N.experts.down_proj` | `[128, 2816, 704]` | `[E, hidden, moe_inter]` |

Two traps:

- **Gate and up are fused** into one `[2I, H]` tensor per expert; the first `I`
  rows are gate, the last `I` are up (`.chunk(2, dim=-1)` on the projection
  output). Splitting on the wrong axis or the wrong half silently swaps them —
  and `gelu(up) * gate` is a plausible-looking model that is not this one.
- **They stay in `[E, out, in]` orientation.** Unlike every rank-2 projection,
  which the loader transposes to `[in, out]`, these are consumed as-is by
  einsums that contract `in` on the last axis. See §20 for why transposing them
  would be actively harmful.

## 19. There is no `-w4a16-ct` checkpoint for this size ✅

Enumerated from the Hub, 2026-07-31 — do not assume the suffix set is uniform:

| size | `-w4a16-ct` | `-q4_0-unquantized` | `-q4_0-gguf` | mobile |
| :--- | :---: | :---: | :---: | :---: |
| E2B, E4B | ✅ | ✅ | ✅ | ✅ |
| 12B, 31B | ✅ | ✅ | ✅ | — |
| **26B A4B** | **❌** | ✅ | ✅ | — |

`google/gemma-4-26B-A4B-it-qat-w4a16-ct` does not exist. GGUF targets llama.cpp,
so the only usable export is `-q4_0-unquantized`: 51.61 GB of BF16 against a
v6e-1's 33.55 GB.

It fits anyway because **"unquantized" describes the container, not the values**.
These are QAT weights already sitting on a Q4_0 grid — measured by range-reading
the shards, all 256 sampled groups of 32 lie exactly on a 4-bit grid, for expert,
attention, MLP, router and embedding tensors alike. Group size **64** fails the
same test (3/128), which is what pins the group at 32 rather than leaving it
assumed. Repacking to W4A16 at load gives 15.27 GB.

## 20. Two ways to destroy those weights while "just repacking" them ✅

**(a) `d = amax / 8` is the wrong step.** The textbook Q4_0 rule assumes each
block's largest magnitude sits at level ±8. Many blocks here peak lower. When
they do, the derived step is a fraction of the true one, `round(x/d)` lands
between grid points, and the block is requantized onto a grid that does not
contain its own values:

| step rule | median relative error, `experts.gate_up_proj` |
| :--- | ---: |
| `d = amax/8` | **4.9e-2** |
| `d = amax/m`, m searched over 1..8 | 0 for 78.7% of values |
| ...plus least-squares refinement over the group | **0 for 93.1% of values** |

Nothing raises in the first case. `ports/gemma4/jax_q4_0.py` searches for the
level the peak actually occupies and returns a count of groups it could not
place; the loader raises on any nonzero count.

**(b) Packing after the transpose.** W4A16 packs nibbles along the **last** axis,
and the Q4_0 grid runs along `in`. The loader transposes dense rank-2 weights
`[out, in] -> [in, out]`, so packing a *transposed* weight groups across `out`,
where no grid exists — a real requantization dressed up as a repack. Packing must
happen in the loader, before the transpose, which is why `requantize_q4_0` is a
loader argument rather than a post-processing pass over the parameter tree.

Residual after doing both correctly: 89–93% of values bit-identical, worst case
~1.6 BF16 ULP. That gap is scale precision — Q4_0 carries an fp16 block scale and
this format stores BF16 — not level assignment; refining the step moves zero
levels. Same regime as the shipped `-w4a16-ct` checkpoints.

## 21. 26B config values that differ from the dense sizes ✅

| field | 26B A4B | note |
| --- | ---: | --- |
| `num_hidden_layers` | 30 | 25 sliding / 5 full, full at `i % 6 == 5` |
| `hidden_size` | 2816 | |
| `intermediate_size` | 2112 | the dense MLP, which still exists |
| `num_experts` / `top_k_experts` | 128 / 8 | ~4B active of 26.5B total |
| `moe_intermediate_size` | 704 | per expert |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 8 | sliding layers |
| `num_global_key_value_heads` | **2** | 31B uses 4 — do not carry it over |
| `global_head_dim` | 512 | vs `head_dim` 256 |
| `num_kv_shared_layers` | **0** | no KV sharing (§7 does not apply) |
| `hidden_size_per_layer_input` | **0** | no PLE (§8 does not apply) |
| `use_double_wide_mlp` | false | §9 does not apply |
| `attention_k_eq_v` | **true** | as the 31B — the 5 full layers ship no `v_proj` |
| `sliding_window` | 1024 | vs E2B's 512 |

Consequence worth noting: because the full-attention layers carry only 2 KV heads
and the sliding layers window at 1024, KV on this model is unusually cheap —
110 KiB/token at int8 unwindowed, and 0.15 GB at 4K context with `window_kv`. On
the 26B, KV is not the constraint; prefill temporaries are.

## 22. Unresolved ⚠️

- **`store_full_length_kv` behaviour.** The reference marks the last non-shared layer
  of each type as storing full-length KV. Our windowed-KV ring windows every sliding
  layer including the source. Self-consistent (windowed and full-length outputs match
  in `tests/test_windowed_kv.py`), but whether it matches Gemma's intent for shared
  sliding layers is unverified.
- **26B on hardware.** Everything in §15–21 is verified against the reference or
  against the checkpoint bytes on CPU. Nothing is measured on a TPU yet — the
  15.27 GB resident figure is arithmetic, and the prefill ceiling is unknown. See
  `benchmarks/runs/2026-07-31-gemma4-26b-v6e1/REPORT.md`.

## How to check the next one

```python
import transformers.models.gemma4.modeling_gemma4 as m; print(m.__file__)
```

Read it before inferring anything from tensor shapes. Every fix in sections 2–6 came
from that file after hours of guessing produced nothing but plausible-looking garbage.

And reading it is necessary but not sufficient. §16 was found by *diffing against a
running reference model*, not by reading — the wiring it describes is two adjacent
lines that look interchangeable, and every unit test of the parts in isolation
passed while the whole was wrong. For anything with more than one input tensor,
build the parity harness before trusting the read.
