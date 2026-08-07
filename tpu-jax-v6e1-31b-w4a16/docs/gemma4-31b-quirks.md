# Gemma 4 31B on TPU: quirks found the hard way

Companion to [`gemma4-quirks.md`](gemma4-quirks.md), which covers the E2B
architecture against the `transformers` reference. This one covers what the **31B**
does differently, what the **W4A16 checkpoint format** does, what **XLA on v6e**
does, and — the section that cost the most time — **how the measurements lie**.

Everything here was found while getting `google/gemma-4-31B-it-qat-w4a16-ct` running
on a single v6e-1 on 2026-07-31/08-01. Sources:
[`REPORT.md`](../benchmarks/runs/2026-07-31-gemma4-31b-v6e1/REPORT.md) (engine) and
[`MODEL-INTEL.md`](../benchmarks/runs/2026-07-31-gemma4-31b-v6e1/MODEL-INTEL.md)
(model).

Status legend: **✅ measured** · **⚠️ inferred** · **❌ retracted** (kept because the
wrong version is instructive).

---

# A. The 31B itself

## A1. It needs no new code, and that is the whole point ✅

60 layers, `hidden_size` 5376, 50 sliding / 10 full attention, no PLE
(`hidden_size_per_layer_input: 0`), no KV sharing (`num_kv_shared_layers: 0`), no
double-wide MLP. It is a **strict simplification** of what the E-series engine
already implements — architecturally *simpler* than E2B. All 15 asserted config
fields resolved through the existing `config_from_hf` unchanged.

Every difficulty was scale, never architecture.

## A2. `layer_scalar` is a three-phase clamp, not a smooth schedule ✅

Mean 0.7823 over 60 layers, but the shape matters:

| L0 | L1 | L2–L4 | body | L58 | **L59** |
| --: | --: | --: | --: | --: | --: |
| 0.0986 | 0.0732 | ~0.99 | 0.45–0.9 | 0.6953 | **0.0317** |

**~12x suppression on entry, near pass-through at L2–L4, ~31x suppression on exit.**
Full-attention layers are damped harder than sliding ones (0.6891 vs 0.8010).

## A3. The pre-norms are enormous and the post-norms are not ✅

| norm | mean abs | max abs |
| :--- | ---: | ---: |
| `input_layernorm` | 23.475 | **1248.0** |
| `pre_feedforward_layernorm` | 40.797 | 860.0 |
| `post_attention_layernorm` | 0.473 | 19.6 |
| `post_feedforward_layernorm` | 0.737 | 63.0 |

Pre-norms amplify by 23–41 on average; post-norms are sub-unit. `layer_scalar` (A2)
is the counterweight. Any quantization scheme assuming a well-behaved per-tensor
range will be destroyed by the 1248 channel.

## A4. The residual stream SHRINKS with depth ✅

RMS 10.06 at layer 0 → **1.239** at layer 59, i.e. **0.12x**. Most transformers grow
it. Early layers restructure rather than refine: attention output *exceeds* the
residual at layers 2, 3, 4 (`attn/resid` 1.92, 1.85, 2.32).

## A5. Massive activations at 15,665x, parked on the first two tokens ✅

max/median ratio in the residual stream: mean **1,214x**, peak **15,665x** (layer 7).
The peak sits on position 1 (`<|turn>`) in 32/60 layers and position 0 (`<bos>`) in
15/60 — **78% of layers on the first two tokens**. Two channels (ch3770, ch1682)
account for 57%.

Mean `|h|` at position 0 is only **1.3x** the rest, so the sinks are sparse in
(position, channel) space — whole positions are not large, specific channels at
those positions are.

**Layer 59 peaks at only 18.33** against hundreds elsewhere: A2's exit clamp exists
to crush these before the lm_head. A2 and A5 are one mechanism.

## A6. Only 10 of 60 layers can see the sink past 1024 tokens ✅

A sliding layer is masked to `(p − 1024, p]` **regardless of `window_kv`** — the ring
buffer is memory, the mask is semantics. At a 1536-token prompt, attention mass on
positions 0–1 for the last query:

| | sink mass | top key is sink |
| :--- | ---: | ---: |
| sliding (50) | **exactly 0.000000** | — |
| full (10) | **0.236535** | **70%** |

Layer 11 spends **44%** of its attention budget on the first two tokens. This
explains why full-attention layers produce 33–37% larger outputs and are damped
harder: **they are the model's only sink pathway and only long-range pathway beyond
1024 tokens.**

The tension: that pathway runs through the **narrowest KV in the model** — 4 KV
heads vs the sliding layers' 16, plus `attention_k_eq_v`. Most load-bearing, least
capacity, no redundancy.

## A7. The norm-weight outlier channels are NOT the activation outlier channels ✅

Static, by norm weight — pre-norms: ch1081, ch1400, ch1924, ch4206 (recurring across
both); post-norms: a tight band at **ch33–47**, ch39 the argmax in 12/60 and 16/60
layers. Runtime, by activation: ch3770, ch1682, ch2130, ch3067.

**No overlap.** You cannot predict one from the other.

Also: 13.9% of `input_layernorm` and 23.0% of `pre_feedforward_layernorm` channels
exceed 10x the median, versus **0.13–0.17%** for post-norms. Broad heavy tail vs a
few localized outliers — two different designs in one layer.

## A8. On real prompts the output is a point mass ✅

top-1 probability **1.0000**, entropy **0.000 nats**, top-1-to-top-2 margin
**15.28**, max logit 27.9 against a 30.0 softcap (so `tanh` is well into saturation
without clipping).

Consequence: greedy decoding is extremely stable on real input — and **meaningless
to evaluate on random tokens**, where the margin collapses to ~0. See D1.

## A9. Unlike the 12B, bare prompts work ✅

The 12B emits digit strings (`'111111'`) without the chat template, on the HF
PyTorch reference *and* this engine alike. The 31B recovers the `<|channel>thought`
scaffolding on its own and answers correctly from a bare prompt.

---

# B. The W4A16 checkpoint format

## B1. int4 nibbles are BIASED, not two's complement ✅

Stored nibble is `value + 8`: 0 → −8, 8 → 0, 15 → +7. See `jax_e_model.py:305`
(`((pc >> (4*i)) & 0xF) - 8`).

Sign-extending as two's complement instead scrambles the weights and yields
**SNR ≈ −8 dB**. Cost: two wrong measurement rounds.

## B2. Packed words are int32 and must not pass through float32 ✅

An `I32` packing eight nibbles routinely exceeds 2²⁴, which float32's mantissa
cannot hold — the bit pattern is **silently rounded** and the nibbles are destroyed.
Any reader that normalizes tensors to float32 must expose a raw path.

## B3. `safetensors` cannot decode bf16 on a bare JAX VM ✅

Both `framework="np"` and `framework="flax"` raise
`data type 'bfloat16' not understood`, and torch is not installed. The container
format is trivial (8-byte header length, JSON header, raw buffer) and bf16 → f32 is
a 16-bit left shift — see `SafeReader` in `ports/gemma4/jax_31b_model_intel.py`.

## B4. Measured W4A16 error is 6.67% and FLAT ✅

Relative Frobenius error **0.0667**, SNR **23.52 dB**, across 20 (layer, projection)
pairs — min 0.06663, max 0.06671, a spread of **0.12%**.

Control: every tensor neither variant quantizes — all four RMSNorms at layers
0/30/59, `layer_scalar`, `final_norm`, all 1.4B parameters of `embed_tokens` — is
**bit-identical** between `-qat-w4a16-ct` and `-qat-q4_0-unquantized`. Same base
model.

## B5. ❌ Scale dynamic range does NOT predict quantization damage

The proxy ranked `v_proj` hardest (4.22) and `up_proj` easiest (2.41), a 1.75x
spread, and suggested spending extra bits on late-layer attention. **B4 measured
flat error and `v_proj` is among the lowest.** Retracted.

The proxy measured the spread of the *scales*, and the scales exist to absorb that
spread — one bf16 scale per 32 columns adapts to whatever local range it finds. A
wide scale distribution means the mechanism is working.

**Consequence: there is no cheap mixed-precision win on the weights here.**

## B6. Packed inner dim 672 is not lane-aligned ⚠️

`in/8` for hidden-input projections is 5376/8 = **672 = 5.25 × 128**. Roughly 2/3 of
packed weight bytes live in 672-last-dim tensors. It costs **no HBM** — resident
bytes match the unpadded arithmetic exactly — but whether it costs VMEM tiling
efficiency is unmeasured.

## B7. The multimodal towers are only 1.15 GB ✅

Of the 23.3 GB checkpoint, vision+audio is 1.15 GB; text is 22.11 GB on host and
19.36 GB as a JAX parameter tree (dtype casts at load). Resident is 19.297 GB — the
60 MB gap is `attention_k_eq_v` aliasing `v_proj` onto `k_proj` across 10 layers,
which the arithmetic confirms (10 × ~6.2 MB).

---

# C. XLA and the TPU

## C1. Sequence buckets make 4097–8192 tokens guaranteed-OOM ✅

`static_sequence_buckets` is powers of two to 8192, and the 128-aligned fallback only
applies **above** 8192. The 31B's one-shot prefill ceiling is ~4096 tokens, so a
4,200-token prompt rounds up to the 8192 bucket and dies — a shape that cannot run.
Between 4096 and 8192 there is no bucket to land on.

## C2. Prefill temp is ~10 GB before sequence length matters ✅

W4A16 unpack/dequantize working set: **147.3 MB/layer**, 8.836 GB of the 9.174 GB
floor at 60 layers, confirmed by layer ablation *and* by named buffers
(`s32[out, in/8, 8]` nibble expansions at 168 MiB, `bf16[out, in/32, 32]`
dequantized weights at 220.5 MiB, dozens live).

Weights (19.36 GB) + that floor ≈ 30 GB of a 33.55 GB chip **before** context.

## C3. The 4K→8K cliff is the attention softmax ✅

`subtract`, `exponential`, `divide` each grow **exactly 4.00x** per 2x in S — dense
`f32[16,2,S,S]` scores, 2.147 GB each at 4K, 8.59 GB at 8K. Temp goes 10.34 GB (4K)
→ 41.10 GB (8K), a 3.98x jump. It is a **cliff, not a curve**: XLA tiles below and
stops above.

## C4. Chunked prefill was missing `donate_argnums` ✅

Without it XLA allocates a fresh KV cache per chunk and the schedule **collapses at
depth**: 34.70 GB vs 8.013 GB of temp at `total_len=4096`, the difference between
not compiling and compiling. Below 4096 it is scheduling noise and sometimes worse.
Fixed in `chunked_prefill_with_kv_cache`.

## C5. Query-chunking cannot fix the 8192 cliff ✅

Chunk sizes 128 / 256 / 1024 all require 32.44 / 32.41 / 32.98 GiB — an **8x range
in chunk size moves it 1.7%**. The cliff is driven by the **key** dimension, not the
query dimension chunking bounds.

## C6. A ring buffer for chunked prefill must be `window + chunk`, not `window` ⚠️

A chunk's queries span `[slot, slot+chunk)` and each attends back `window`, so the
chunk collectively reads `window + chunk − 1` positions. A ring of exactly `window`
is filled by the current chunk alone and **destroys the previous chunk's keys before
the chunk's early rows read them** — it compiles, runs, and computes the wrong thing.

An earlier draft of the report proposed `chunk_size == sliding_window` as the safe
case. It is the worst case.

## C7. The fused Pallas W4A16 kernel is unusable at this width ✅

`CompileTimeScopedVmemOom` at 5376×21504 (32 MB scoped-VMEM limit), and **5.4x
slower** at the small buckets where it does compile. The exception surfaces at
*compile* time, so the try/except fallback in `qat_w4a16_linear_jax` does **not**
catch it — `auto` fails the cell rather than degrading to `reference`.

## C8. `--xla_dump_hlo_pass_re=buffer-assignment` silently emits nothing ✅

It matches no pass name. Plain `--xla_dump_to` produces the whole set including
`after_optimizations-buffer-assignment.txt` and `memory-usage-report.txt`. The
filtered form fails with no error and an empty directory.

## C9. XLA prints GiB; `memory_analysis()` returns bytes ✅

`(38.28G)` in an OOM message is **38.28 GiB = 41.10 GB**. Mixing that with
`temp_size_in_bytes / 1e9` produced a wrong ratio (3.70x instead of 3.98x) that
disagreed with an independent measurement until the units were fixed.

Also: `available HBM (31.24G)` = 31.24 GiB = 33.55 GB — the **whole chip**. XLA
compares **temp alone** against it and does not subtract resident weights.

## C10. `peak_bytes_in_use` does not capture intra-call transients ✅

It returned exactly `bytes_in_use` on every sample. Resident-memory series cannot
be used to infer peak; use `memory_analysis()` on the compiled executable.

## C11. Instruction output size ≠ allocated memory ✅

Ranking HLO instructions by output shape suggested the lm_head's
`f32[262144,5376]` (5.637 GB, three instructions) was the temp floor. The **decode
step contains the same three instructions with a total temp of 0.146 GB** — they are
fused, not materialized. Fusion means an instruction's shape says nothing about
allocation.

---

# D. How the measurements lie

The most expensive section. Every wrong conclusion in this work came from here.

## D1. Never judge correctness on random-token prompts ✅

Random tokens are out of distribution: logits go near-flat, top-1 margins collapse
to ~0, and any numerical perturbation flips the argmax. Measured on random tokens,
two `window_kv` settings differed by **12% with a different top-1 token** and looked
like a cache-correctness bug. On a real prompt: **same argmax**, 3.0% divergence
against a **15.28** margin (A8).

It was fusion drift — different buffer sizes compile to different HLO, so
accumulation order differs, and ~3% accumulates over 60 layers. Ordinary
floating-point non-associativity.

Corollary: **hold `window_kv` fixed across any A/B.** Greedy decoding is not
bit-reproducible across it.

## D2. A weights-only bandwidth floor does not prove compute-boundness ✅

19.30 GB / 78.45 ms = 246 GB/s, ~15% of a published 1640 GB/s — reported as "decode
is not bandwidth-bound". **This repository had already made and retracted that exact
error on E2B**, noting the floor "omitted KV-cache traffic and the logits
convert/copy". 1640 GB/s is also published, not calibrated here.

What survives is narrower: the fused kernel is *measurably* slower (C7), which needs
no floor.

## D3. Bitwise on CPU is not bitwise on TPU ✅

The `logits_at` change is bitwise-identical on CPU with a 4-layer f32 model, and on
a TPU with 60 layers at W4A16 it differs in ~every logit (max 3.7e-2, argmax
preserved). `[B,1,H] × [H,V]` tiles differently from `[B,S,H] × [H,V]` on the MXU.

Use "mathematically exact, numerically different on TPU".

## D4. Extrapolating from a smaller shape fails at every cliff ✅

Prefill temp is 9.17 / 10.19 / 10.34 GB at 1K/2K/4K — nearly flat, `temp/token`
*falling* — then **41.10 GB** at 8K. Any per-token model fitted below 4K predicts
off the end of a cliff. Same for chunked prefill: fine to 30 layers, 5.3x worse at
60.

**Measure at the target shape.**

## D5. Byte counts computed by hand do not predict what the compiler does ✅

The `logits_at` saving was estimated at 4.29–8.59 GB from first principles. Measured
on the same chip: **0.4–1.4 GB** at B=1 (XLA had already fused most of it away),
**~21.7 GiB** at B=2×2K (where it had not), and **0.01 GiB at 8K** (nothing left).
No hand-computed number predicted any of these.

## D6. Negative SNR is always a decoder bug ✅

A quantizer cannot produce error larger than signal. SNR ≈ −8 dB caught B1 and B2.
Useful tripwire: if a "quantization error" exceeds 100%, stop and check the decoder.

## D7. Design a control that can actually fail ✅

The B4 control compared tensors that **neither** variant quantizes. Those matching
proves a shared base, but does not by itself prove the QAT-trained projections were
meant to match. The 6.67% result is quantizer-shaped, which is the real evidence.
Ask what a passing control would look like if the hypothesis were false.

---

## How to check the next one

The pattern across every error above: reasoning about something that had not been
measured *under conditions matching the real case*. The cheap defenses, in order —

1. **Reproduce small on CPU first.** If it does not reproduce, that is the clue.
2. **Ask the compiler**, not the resident-memory series: `memory_analysis()` on the
   compiled executable predicted every pass/fail exactly.
3. **Use a real prompt.**
4. **Check the physical bound** — bandwidth, SNR, information content — before
   believing a ratio.
5. **Triangulate against a known-good implementation** instead of guessing a third
   time. The engine's own unpack is what settled B1.
