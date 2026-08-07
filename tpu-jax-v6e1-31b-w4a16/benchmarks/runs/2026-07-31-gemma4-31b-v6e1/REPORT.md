# Gemma 4 31B QAT on the pure-JAX engine — TPU v6e-1

**Date:** 2026-07-31
**Hardware:** Cloud TPU v6e single chip (`ct6e-standard-1t`, 33.55 GB HBM), `europe-west4-a`
**VM:** `jax-gemma4-12b` (flex-start, 4h window — the 12B VM, reused)
**Model:** `google/gemma-4-31B-it-qat-w4a16-ct` (60 layers, W4A16 `compressed-tensors`)
**JAX:** 0.11.0
**Harness:** `ports/gemma4/jax_31b_port.py`

**Status: LOADS, FITS, AND ANSWERS CORRECTLY ON A SINGLE v6e CHIP — with zero engine changes.**

---

## Executive summary

1. **Zero code changes.** All 15 asserted config fields resolve correctly through the
   existing `config_from_hf`, and `convert_safetensors_to_jax_params` maps all 60
   layers unchanged. The 31B is the third checkpoint in a row (E2B → 12B → 31B) to
   need nothing new. The `attention_k_eq_v` branch written for this model finally
   ran on the model it was written for.
2. **It fits one chip, comfortably.** 19.36 GB of W4A16 text weights, 19.30 GB
   resident, leaving **14.25 GB of headroom (42.5%)** on 33.55 GB. Multi-chip
   sharding is not required to serve this model — which was the open question.
3. **It is correct.** Greedy decode answers `The capital of France is Paris.` — and
   unlike the 12B, it answers correctly on the *bare* prompt as well as the
   chat-templated one.
4. **Decode sits far off a weights-only bandwidth floor — which is NOT proof it is
   compute-bound.** 78.45 ms/step at B=1 against 19.30 GB of weights implies
   246 GB/s, ~15% of v6e's published ~1640 GB/s. **An earlier draft of this report
   called that "decode is not bandwidth-bound" and made it the headline. That was
   wrong, and wrong in a way this repository had already documented and retracted
   once** — see `benchmarks/runs/2026-07-28-jax-e2b-v6e1/REPORT.md`, which found the
   identical ~6x gap on E2B and then corrected itself:

   > The floor we compared against counted **weights and the LM head only**. It
   > omitted KV-cache traffic and the logits convert/copy, so "decode is not
   > bandwidth-bound" overstates what the gap proves.

   The same three omissions apply here, and Addendum 2 shows one of them is large:
   the decode step's lm_head converts the tied `bf16[262144,5376]` table (2.82 GB)
   to f32 — **the numerator is missing several GB per step**, so the 246 GB/s
   denominator is simply not the traffic the step actually moves. 1640 GB/s is also
   a published figure, not calibrated on this chip.

   **What survives:** the gap is *unattributed*. Reducing weight traffic is
   nonetheless measurably not the win — the fused int4 kernel is 5.4x slower where
   it compiles (below), which is a direct measurement rather than an inference from
   a floor. Profile before optimizing further.
5. **The fused Pallas kernel does not close that gap — it is 5.4x worse.** Tested,
   because the roofline gap made it the obvious candidate. It is not the fix. See
   the A/B below; this is a negative result and it is the useful kind.
6. **The binding constraint is prefill, not weights.** One-shot prefill OOMs above
   roughly 4K total prompt tokens; decode at 4K is fine, and chunked prefill does
   not rescue it. A memory probe shows resident HBM at 4K is only 20.63 GB of
   33.55 — so the ceiling is a **transient**, and the biggest avoidable one is the
   `[B, S, 262144]` logits tensor prefill builds and then throws all but one row of.
   Fixing that is cheap and is the top next action.

**This model is dense.** Architecturally it is *simpler* than E2B — PLE off, KV
sharing off, double-wide MLP off — but every one of its 31B parameters is read on
every decode step, where E2B reads 2.3B effective plus a PLE gather. That is the
whole story of this report: nothing new to implement, and no headroom to waste.

---

## Memory: what 19.36 GB buys and what is left

| Component | HBM | Notes |
| :--- | ---: | :--- |
| W4A16 packed weights + scales | **19.36 GB** | parameter-tree bytes |
| Actually resident | **19.30 GB** | 60 MB less — see aliasing note |
| Free headroom | **14.25 GB** | 42.5% of the chip |
| HBM limit | 33.55 GB | `ct6e-standard-1t` |

Load read **22.11 GB of text tensors from host** and skipped **1.15 GB of
vision/audio towers**. Load wall time was **8.7 s** against a warm page cache
(the 23.3 GB download preceded it separately).

### The 60 MB gap is `attention_k_eq_v` working

Parameter-tree bytes (19.36 GB) exceed resident bytes (19.30 GB) by ~60 MB. That is
the loader aliasing `v_proj` onto `k_proj` on the 10 full-attention layers: the tree
holds two references, the device holds one buffer. Per layer that is a `[2048, 672]`
int32 packed tensor plus its `[2048, 168]` bf16 scale ≈ 6.2 MB, times 10 layers ≈
62 MB. The arithmetic lands where it should, which is a decent independent check
that the aliasing is real and not just declared.

### KV cache, derived from `init_kv_cache`

With `window_kv=True` (automatic here, since `max_model_len` 4096 > `sliding_window` 1024):

| Layer class | Count | Per token (bf16) | Behaviour |
| :--- | ---: | ---: | :--- |
| sliding (`num_key_value_heads`=16, `head_dim`=256) | 50 | 16 KiB/layer | ring buffer, capped at 1024 slots → **838.9 MB fixed** |
| full (`num_global_key_value_heads`=4, `global_head_dim`=512) | 10 | 8 KiB/layer | grows with context → **80 KiB/token** |

So per stream: **838.9 MB + 80 KiB/token**. At 128K context that is **11.58 GB** —
it fits inside the 14.25 GB headroom, but only just, and only for a single stream.

Against the 12B's 336 KiB/token this is much heavier, and the reason is not depth:
the 31B carries 16 sliding KV heads to the 12B's 8, and 4 global KV heads to the
12B's 1.

**Optimization worth taking, quantified:** `init_kv_cache` allocates separate K and V
buffers unconditionally, so on the 10 full-attention layers — where `attention_k_eq_v`
guarantees V *is* K — half of that 80 KiB/token is an exact duplicate. Collapsing it
saves 40 KiB/token: 22% of total KV at 8K, and **~46% at 128K (5.37 GB)**. Note this
is arithmetic from the allocation code, not a measurement. It also corrects a stale
figure — `jax_e_loader.py:177` estimates this saving at "~4.5% of the 31B's KV", which
is far too low at any long context.

---

## Correctness

Prompt: `What is the capital of France?`, greedy (`temperature=0.0`), 32 new tokens.

Chat template resolves to:
`'<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'`

| Prompt form | Output | `finish_reason` |
| :--- | :--- | :--- |
| chat-templated | `'The capital of France is Paris.'` | `stop` |
| bare | `'\n<|channel>thought\n<channel|>The capital of France is Paris.'` | `stop` |

Both correct, both terminating on their own. **This differs from the 12B**, where a
bare prompt produced digit strings (`'111111'`) on the HF PyTorch reference *and* on
this engine alike. The 31B recovers the channel scaffolding on its own and answers
anyway.

The transcripts above are from the second (`auto`) run. The first run did not pass
`eos_token_ids`, so it produced the same correct answer and then rambled to the
32-token limit repeating itself — which reads like a defect and is not one. The
harness now derives stop tokens from the tokenizer and passes them to `generate`;
that is the only difference between the two transcripts.

---

## Benchmark sweep — `w4a16_impl="reference"`

Median of 3, 1 warmup discarded. Same methodology as the 12B report, so the two are
directly comparable.

| Users (B) | Context (S) | Prefill (TTFT) | Decode step | Aggregate | Per-user |
| :---: | :---: | ---: | ---: | ---: | ---: |
| 1 | 1K | 296.4 ms | 78.67 ms | 12.7 tok/s | 12.7 tok/s |
| 1 | 2K | 489.3 ms | 78.69 ms | 12.7 tok/s | 12.7 tok/s |
| 1 | 4K | 1085.3 ms | 78.45 ms | 12.7 tok/s | 12.7 tok/s |
| 1 | 8K | **OOM** | — | — | — |
| 2 | 1K | 468.3 ms | 131.27 ms | 15.2 tok/s | 7.6 tok/s |
| 2 | 2K | **OOM** | — | — | — |
| 2 | 4K | **OOM** | — | — | — |
| 2 | 8K | **OOM** | — | — | — |
| 4 | 1K | **OOM** | — | — | — |
| 4 | 2K | **OOM** | — | — | — |
| 4 | 4K | **OOM** | — | — | — |
| 4 | 8K | **OOM** | — | — | — |

Only two cells of the concurrency grid survive one-shot prefill. B=2 at 1K is the
practical concurrency ceiling on this chip without chunking, and it buys only
1.20x aggregate throughput for a 1.73x per-user latency penalty — a bad trade.
At this weight footprint the 31B on a single v6e-1 is a **latency-optimized,
low-concurrency deployment**, not a throughput one.

Decode step is **flat to within 0.3% from 1K to 4K**, the same signature the 12B
showed — windowed sliding KV means context costs almost nothing per step until the
full-attention layers' share grows.

### Versus the 12B, on the same chip and harness

| | 12B | 31B | ratio |
| :--- | ---: | ---: | ---: |
| resident weights | 8.15 GB | 19.30 GB | 2.37x |
| decode step, B=1 | 29.50 ms | 78.45 ms | 2.66x |
| decode tok/s, B=1 | 33.9 | 12.7 | 0.37x |
| implied HBM read rate | 276 GB/s | 247 GB/s | — |

The step ratio (2.66x) exceeds the weight ratio (2.37x), and both models sit at
15–17% of the chip's ~1640 GB/s. A decode step that were actually bandwidth-bound
would land near 12 ms for the 31B, not 78 ms.

---

## The fused Pallas W4A16 kernel is unusable here — negative result

The roofline gap above made `w4a16_impl="auto"` (the fused Pallas kernel, `plane`
layout) the obvious candidate, so it was run as a straight A/B: same checkpoint,
same harness, same chip, only the flag changed. Load was identical to the byte
(19.357 GB tree / 19.297 GB resident), confirming the flag changes only the matmul.

| | `reference` | `auto` (fused Pallas) | |
| :--- | ---: | ---: | :--- |
| parity, bare prompt — decode | **13.42 tok/s** | **2.47 tok/s** | **5.4x slower** |
| parity, bare prompt — prefill | 217.3 ms | 403.6 ms | 1.9x slower |
| sweep cell B=1, ctx 1K | 78.67 ms/step | **fails to compile** | `CompileTimeScopedVmemOom` |

Two distinct failures, and they are consistent with each other:

1. **Where it compiles, it is slower.** At the parity prompt's tiny bucket the
   kernel fits scoped VMEM and runs — 5.4x worse than the reference dequantize.
2. **At any realistic context it does not compile at all.** `ctx=1024` exceeds the
   32 MB scoped-VMEM limit. `set_w4a16_impl`'s docstring already records this
   failure mode for the `interleaved` layout; the 31B's projections are wide enough
   (`5376 x 21504` on the MLP) that `plane` hits it too.

Note the exception surfaces at *compile* time, not trace time, so the try/except
fallback inside `qat_w4a16_linear_jax` does not catch it — `auto` does not silently
degrade to `reference` here, it fails the cell. Worth knowing before anyone reaches
for this flag on a large checkpoint.

**So the gap against a weights-only floor is real but unattributed** (see the
retraction in the executive summary — that floor omits KV traffic and the logits
pipeline, so it does not establish that decode is compute-bound). The fused kernel is
not the way to claim it. `dequant_at_load` is also out — 31B dense BF16 is ~62 GB
against 33.55 GB of HBM. What is left is a kernel that fits VMEM at these widths,
which is real work, not a flag.

### The prefill ceiling

One-shot prefill OOMs above roughly 4K total prompt tokens:

- B=1 × 4K = 4096 tokens — **passes**
- B=2 × 2K = 4096 tokens — **OOMs**

So the budget is not a pure token product: the batch dimension also multiplies the
KV allocation. The 12B's equivalent ceiling was ~8192 tokens.

**Chunking does not fix it.** Three configurations were tried with
`prefill_chunk_size` (which forces `window_kv=False`), all at 8K, all OOM:

| ctx | chunk | result |
| ---: | ---: | :--- |
| 8192 | 1024 | OOM |
| 8192 | 512 | OOM |
| 8192 | 256 | OOM |
| 16384 | 1024 | OOM (`RuntimeBufferAllocationFailure`) |

A 256-token chunk's own temporaries are under 2 GB, so chunk size is not the
binding term — and turning windowing off to enable chunking costs 7.4 GB of
unwindowed KV at 8K, which is most of the headroom. Chunking as currently wired
trades away more than it saves at this model size.

### What actually sets the ceiling

`ports/gemma4/jax_31b_prefill_probe.py` measures resident HBM after prefill at
increasing S (B=1, windowed):

| S | resident | above weights | marginal |
| ---: | ---: | ---: | ---: |
| 512 | 19.811 GB | 0.514 GB | — |
| 1024 | 20.306 GB | 1.010 GB | 0.968 MB/token |
| 2048 | 20.427 GB | 1.130 GB | **0.118 MB/token** |
| 4096 | 20.631 GB | 1.334 GB | **0.100 MB/token** |
| 8192 | **OOM** | — | — |

(The device's `peak_bytes_in_use` counter returned exactly `bytes_in_use` on every
row, so it is not capturing intra-call transients on this backend. That is why the
transient below is bounded by inference from the resident series rather than simply
read off — worth knowing before trusting that counter for this purpose.)

The marginal cost collapses by ~9x once S passes `sliding_window=1024` — windowed KV
is doing exactly what it claims, and **retained** memory is not the problem. At
S=4096 only 20.63 GB is resident, leaving **12.9 GB free**, and extrapolating the
0.100 MB/token slope puts retained memory at 8K near 21.0 GB — still ~12.5 GB
short of the limit. **So the 8K OOM is entirely transient, and the transient must
exceed ~12.5 GB, i.e. >1.5 MB per prompt token.**

That last number is the useful one, because it is a budget the candidate has to fill.
The largest avoidable transient is visible in `jax_e_model.py:1306-1317`: prefill
calls the model, gets logits for **every** position — `[B, S, 262144]` — and then
`take_along_axis`es out the single last real row. At vocab 262144 that discarded
tensor costs **0.52 MB/token at bf16, 1.05 MB/token at f32**: 4.29 GB / 8.59 GB at
S=8192. At f32 it alone is **~70% of the >1.5 MB/token that has to be accounted
for** — by far the largest single term, and the only large one that is pure waste.

Slicing the hidden state to the last real position *before* the lm_head would remove
that allocation entirely and cost nothing — prefill already discards the rest.

**Scope of this claim:** the transient-vs-retained split is measured; that the
full-sequence logits tensor is the largest avoidable transient is a code-read plus
arithmetic. It has *not* been proven to be the specific allocation that trips the
OOM — other prefill temporaries (the 21504-wide MLP intermediates are ~705 MB per
tensor at 8K) are live at the same time. Implementing the slice and re-running the
8K cell is the measurement that would settle it, and is the top next action.

---

---

# Addendum: the `logits_at` prefill fix, and what it did not do

**Second run, 2026-07-31 evening. VM `jax-gemma4-31b` (fresh flex-start v6e-1,
europe-west4-a). Same checkpoint, same harness, same `reference` impl.**

The action item above ("slice the hidden state before the lm_head") was implemented
and measured. **The prediction attached to it was wrong and the record should say so
plainly: this did not make the 8K cell pass.**

## The change

`Gemma4EModelJAX.__call__` takes an optional `logits_at: [B]`; when given, the
lm_head runs only at those positions and returns `[B, 1, V]`.
`prefill_with_kv_cache` now passes `prompt_lens - 1` instead of building
`[B, S, 262144]` and discarding all but one row.

**Bitwise on CPU; NOT bitwise on TPU.** `tests/test_prefill_logits_slice.py` (6
tests, padded/unpadded, B=1..3, `array_equal` against the pre-change path) passes on
CPU with a 4-layer f32 model, and all 86 pre-existing tests still pass. An earlier
draft of this report called the change "bitwise identical" on that basis. Checked
afterwards on the actual deployed path — TPU, 60 layers, W4A16 — it is not:

| shape | bitwise | max abs diff | differing logits | argmax preserved |
| :--- | :--- | ---: | ---: | :--- |
| B=1, S=512 | **no** | 3.70e-2 | 262143 / 262144 | yes |
| B=1, S=1024 | **no** | 3.70e-2 | 262138 / 262144 | yes |
| B=2, S=1024 | **no** | 5.72e-6 | 385150 / 524288 | yes |

The change is exact in real arithmetic — matmul rows are independent, the softcap is
elementwise — but contracting `[B,1,H] x [H,V]` tiles and accumulates differently
from `[B,S,H] x [H,V]` on the MXU. Argmax was preserved at every shape tested, so
greedy output is unaffected there, **but a near-tie could in principle flip and this
has not been swept.** Use "mathematically exact, numerically different on TPU" —
not "bitwise".

## Measured effect: real, and smaller than claimed

| cell | pre-fix | post-fix |
| :--- | :--- | :--- |
| B=1, 1K | 296.4 ms / 78.67 ms | 294.9 ms / 78.31 ms |
| B=1, 2K | 489.3 ms / 78.69 ms | 481.9 ms / 78.26 ms |
| B=1, 4K | 1085.3 ms / 78.45 ms | 1065.3 ms / 78.41 ms |
| B=1, 8K | **OOM** | **OOM** ← the prediction that failed |
| B=1, 16K | not run | **OOM** |
| B=2, 1K | 468.3 ms / 131.27 ms | 460.1 ms / 129.19 ms |
| **B=2, 2K** | **OOM** | **856.4 ms / 130.12 ms** ← newly feasible |
| B=2, 4K | **OOM** | **OOM** |

Two things are true at once:

1. **The fix bought real headroom.** B=2 x 2K went from OOM to passing. At that
   shape the freed tensor is `[2, 2048, 262144]` f32 = **4.29 GB**, and that is
   exactly the margin that had been missing. Prefill and decode timings are
   unchanged everywhere else, as a pure-memory change should be.
2. **It did not move the token ceiling.** Both before and after, one-shot prefill
   tops out at **~4096 total prompt tokens**. What changed is that B=2 can now
   *reach* 4096; before, the second sequence's overhead pushed it over.

## Why the prediction was wrong

The reasoning was: at 8K the freed tensor is 8.59 GB, retained is ~21 GB, so ~12.5 GB
of transient budget minus 8.59 GB should fit. It didn't — so the pre-fix transient at
8K must have exceeded **21 GB**, not the ~12.5 GB the retained-series bound implied.
Removing 8.59 GB from a >21 GB peak still leaves it over the limit.

The report above hedged this correctly ("has *not* been proven to be the specific
allocation that trips the OOM") and the hedge earned its keep. The lesson is the one
in CLAUDE.md: the arithmetic identified a real 8.59 GB of waste, and being real waste
did not make it the *binding* term. A candidate that fits the budget is not thereby
the thing filling it.

## Settled: ask the compiler, not the resident-memory series

`ports/gemma4/jax_31b_hlo_memory.py` compiles prefill without running it (so it works
at shapes that OOM) and reads `memory_analysis()` off the executable:

| B | S | temp (GB) | temp (GiB) | vs 31.24 GiB budget | sweep said |
| ---: | ---: | ---: | ---: | :--- | :--- |
| 1 | 1024 | 9.174 | 8.54 | fits | OK |
| 1 | 2048 | 10.191 | 9.49 | fits | OK |
| 1 | 4096 | 10.336 | 9.63 | fits | OK |
| 1 | 8192 | **41.10** | **38.28** | **OVER** | OOM |
| 2 | 1024 | 9.921 | 9.24 | fits | OK |
| 2 | 2048 | 10.408 | 9.69 | fits | OK |
| 2 | 4096 | 36.90 | 34.37 | **OVER** | OOM |

**Units matter here and an earlier draft of this table got them wrong.** XLA prints
bytes in **GiB** ("38.28G"), while `memory_analysis()` returns raw bytes that this
report divides by 1e9 to get GB. Both columns are shown above so the comparison
against XLA's own message is direct. The earlier draft also carried a "peak" column
adding `args + temp + output` and comparing it to 33.55 GB — that was wrong
accounting and has been removed: XLA compares **temp alone** against a budget it
reports as 31.24 GiB, which *is* 33.55 GB, i.e. essentially the whole chip. The
weights are not subtracted from it in that message.

That correction also removes a hollow claim. The earlier draft said the peak column
"predicts the sweep exactly" — but **temp alone already separates every pass from
every fail**, so the extra arithmetic was doing no work and merely looked validated
by agreeing.

XLA's own words at B=1, S=8192:

```
Ran out of memory on HBM, the total memory required for HLO temporaries
(38.28G) exceeds available HBM (31.24G).
```

Two things this settles:

1. **The ceiling is a discontinuity, not a curve.** Temp is 9.17 / 10.19 / 10.34 GB
   at 1K / 2K / 4K — nearly flat, with `temp/token` *falling* (8.96 → 4.98 → 2.52 MB)
   — and then **41.10 GB at 8K**. That is a **3.98x jump for a 2x in S**, which is
   4.00x to within measurement noise, and it agrees with the independently measured
   per-opcode softmax scaling in Addendum 2 (`subtract`/`exponential`/`divide` all at
   exactly 4.00x). Below 4K a large S-independent floor dominates and masks the
   quadratic term; at 8K the quadratic takes over. Every per-token model in this
   report (mine included) was fitting the flat region and predicting off the end of
   it. (The earlier draft said "3.70x", a consequence of the GB/GiB error above;
   correcting the units makes this number agree with the opcode measurement rather
   than merely being near it.)
2. **It contradicts a documented assumption in this repo.** `jax_engine.py` records
   prefill temporaries as "LINEAR in the total prompt tokens ... a flat ~2.13 MB per
   token from 1K to 8K (not quadratic — XLA tiles the softmax, so B x H x S x S
   scores are never materialized)". That was measured on E2B and **does not hold for
   the 31B**, which has 32 attention heads to E2B's 8. At 8K, something S x S clearly
   is being materialized. The comment should be scoped to E2B.

The practical consequence: **weights (19.36 GB) + a ~10 GB near-fixed prefill temp
consume ~30 GB of the 33.55 GB chip before sequence length matters at all.** Every
feasible cell sits within 1.8–4.1 GB of the limit. That is why the ceiling is so
abrupt, why chunking (which forces `window_kv=False` and adds 7.4 GB of KV) cannot
help, and why shaving per-token costs moves the boundary only at the margin — which
is precisely what the `logits_at` fix did.

## Verdict on the change

Keep it. It is mathematically exact (numerically different on TPU — see the table
above), costs nothing, and **measurably** widened the feasible batch/context region:
B=2 x 2K goes from not compiling to compiling, controlled on one chip. It is simply
not the fix for the 4K ceiling — at 8K it saves 0.01 GiB — and nothing should be
planned on the assumption that it was.

---

# Addendum 2: the buffers, named — and a controlled A/B

**Same VM (`jax-gemma4-31b`), one process, `ports/gemma4/jax_31b_buffers_ab.py`.**
Run because everything above about *why* the ceiling exists was inferred, and two
such inferences had already proven wrong.

## ⚠️ Read the caveat below before using this section

**`largest_buffers()` ranks HLO instructions by OUTPUT SHAPE, not by allocated
memory.** A fused instruction produces no buffer at all. This distinction was not
made when the section below was first written, and it invalidates its headline
conclusion — see "What this section does and does not establish" at the end.

## Candidate for the ~10 GB "fixed floor": the lm_head upcasting the embedding table

Largest buffers in the optimized HLO at B=1, S=4096 (temp 10.336 GB):

```
5.637 GB  broadcast     f32[262144,5376]
5.637 GB  convert       f32[262144,5376]
5.637 GB  multiply      f32[262144,5376]
2.819 GB  parameter    bf16[262144,5376]     <- the table as stored
2.147 GB  convolution   f32[16,2,4096,4096]
2.147 GB  exponential   f32[16,2,4096,4096]
2.147 GB  subtract      f32[16,2,4096,4096]
   ... 8+ more f32[16,2,4096,4096] at 2.147 GB each
```

The tied embedding table is **2.819 GB in bf16 and appears three times as 5.637 GB
in f32**. `jnp.matmul(h, params["embed_tokens"].T)` (jax_e_model.py, end of
`__call__`) runs with an f32 `h`, so XLA emits a convert of the entire 262144x5376
table to f32. These instructions are S-independent, which is *suggestive* given the
temp floor is also S-independent.

**But they are almost certainly fused, not allocated.** The decode-step check
(`jax_31b_review_checks.py`, CHECK 2) compiles the cached decode step and finds the
**same three `f32[262144,5376]` instructions — with a total temp of 0.146 GB.** If
those instructions forced allocation, decode's temp could not be under 0.2 GB. So
XLA streams the convert through the matmul rather than materializing 5.6 GB.

Two consequences:

- **The "10 GB floor is the lm_head upcast" conclusion does not hold.** It is a
  candidate, not a finding. What creates the S-independent floor in *prefill* is
  still unidentified.
- The convert is therefore **not extra HBM traffic** — it is VPU work proportional
  to the matmul. Whether it costs meaningful time is unmeasured. Note the E2B report
  already found an int8 lm_head worth 0% at B=1 and 4-5% at B>=2, which argues
  against the lm_head being a major decode cost on that model.

## The S² term is the attention softmax over dense scores

Aggregate bytes by opcode, S=2048 -> S=4096 (S ratio 2x):

| opcode | 2K | 4K | growth | class |
| :--- | ---: | ---: | ---: | :--- |
| `subtract` | 64.42 GB | 257.70 GB | **4.00x** | S² |
| `exponential` | 64.42 GB | 257.70 GB | **4.00x** | S² |
| `divide` | 32.21 GB | 128.85 GB | **4.00x** | S² |
| `fusion` | 155.99 GB | 317.22 GB | 2.03x | S |
| `bitcast` | 272.74 GB | 319.44 GB | 1.17x | fixed |
| `and`, `shift-right-arithmetic` | 117.6 GB | 117.6 GB | 1.00x | fixed |

(Aggregates over all instructions, not simultaneous residency — the *growth* is the
signal.) `subtract` / `exponential` / `divide` are softmax's `x-max`, `exp`,
normalize, growing at exactly 4.00x for a 2x in S. The buffer is
`f32[16, 2, S, S]` — 16 KV heads x n_rep 2 = the 32 query heads — **2.147 GB each at
4K, 8.59 GB each at 8K**, with 8+ of them in the program.

`shift-right-arithmetic` and `and` at a flat 117.6 GB are the int4 unpack: fixed, as
expected, and untouched by sequence length.

**50 of the 60 layers are sliding with a 1024 window and can never attend outside
it, so they need `[16,2,S,1024]`, not `[16,2,S,S]` — 8x smaller at 8K.** The dense
`[B,1,S,S]` mask from `make_prefill_causal_mask` is what makes them compute it
anyway. That is the ceiling, and it is now a named tensor rather than a guess.

## Controlled A/B of the `logits_at` fix

Addendum 1's pre/post comparison ran on two different VMs, one sample each. Redone
here with both prefill bodies compiled back-to-back in one process on one chip
(old body reconstructed locally, shipped file untouched):

| cell | OLD temp | NEW temp | saved |
| :--- | ---: | ---: | ---: |
| B=1, S=2048 | 11.596 GB | 10.191 GB | 1.405 GB |
| B=1, S=4096 | 10.745 GB | 10.336 GB | 0.409 GB |
| **B=2, S=2048** | **31.42 GiB — OOM** | **10.408 GB — fits** | **~21.7 GiB** |
| B=2, S=4096 | 42.86 GiB | 34.37 GiB | 8.49 GiB (still over) |
| **B=1, S=8192** | **38.29 GiB** | **38.28 GiB** | **0.01 GiB** |

### What this section does and does not establish

| Claim | Status |
| :--- | :--- |
| softmax opcodes scale at exactly 4.00x per 2x in S | **measured** — a growth rate, immune to the fusion caveat |
| `f32[16,2,S,S]` score tensors exist in the program | **measured** (they appear in the HLO) |
| dense scores are what makes 8K OOM | **strongly supported**, not proven — the S² growth rate is the evidence |
| the ~10 GB S-independent floor is the lm_head f32 convert | **NOT established** — those instructions appear in a decode step whose total temp is 0.146 GB |
| any single instruction's output size equals allocated memory | **false in general** — fusion |

To settle the floor properly, use a buffer-assignment dump
(`--xla_dump_hlo_pass_re=buffer-assignment`) which reports allocated buffers rather
than instruction outputs. Not run here.

### Controlled A/B conclusions, both measured

1. **The B=2 x 2K improvement is real and attributable to the fix**, not to VM
   variance. Addendum 1 guessed right without having earned it; this controls it.
2. **At S=8192 the fix saves 0.01 GiB — nothing.** XLA already fuses the
   full-sequence logits away entirely at that shape, in both paths. *That* is why
   removing it did not move the 8K cell. Not a competing allocation — nothing left
   to remove.

This also corrects Addendum 1's own arithmetic in the other direction: the fix was
estimated at 4.29-8.59 GB from first principles, and is worth **0.4-1.4 GB** at B=1
because the compiler had already eliminated most of it — while being worth **~21.7
GiB** at B=2 x 2K, where it had not. The value is strongly shape-dependent and no
hand-computed byte count predicts it. Ask the compiler.

---

# Addendum 3: the floor, identified — it is the W4A16 dequantize working set

Three attempts failed to name the ~9.2 GB S-independent prefill floor: inference
from resident memory, ranking HLO instructions by output shape, and a guess at the
lm_head. This one is settled two independent ways.

## Layer-count ablation

`ports/gemma4/jax_31b_floor_ablate.py` truncates the layer stack while leaving
`embed_tokens`, `final_norm` and the whole lm_head path intact, then reads
`temp_size_in_bytes`. If the floor were the lm_head it would be flat.

| layers | temp | args | Δtemp/layer |
| ---: | ---: | ---: | ---: |
| 1 | 0.485 GB | 3.088 GB | — |
| 2 | 0.678 GB | 3.357 GB | 192.5 MB |
| 6 | 2.184 GB | 4.472 GB | 376.6 MB |
| 15 | 5.209 GB | 6.935 GB | 336.1 MB |
| 30 | 7.443 GB | 11.088 GB | 148.9 MB |
| 60 | **9.174 GB** | 19.357 GB | 57.7 MB |

Linear fit over 1..60 layers: **147.3 MB/layer, intercept 0.338 GB.** So
**8.836 GB of the 9.174 GB floor is per-layer** and only ~0.34 GB is layer
independent. **The lm_head hypothesis from Addendum 2 is refuted** — it cannot
account for more than ~0.34 GB. The sublinear slope (377 → 58 MB/layer) is XLA
reusing buffers once the rolling working set saturates.

## Buffer assignment, at last

The dump works with plain `--xla_dump_to`; `--xla_dump_hlo_pass_re=buffer-assignment`
matches no pass and silently emits nothing, which is what defeated the first
attempt. `after_optimizations-memory-usage-report.txt` (kept as
`logs/memusage_report.txt`) lists every allocation. The recurring large ones:

| shape | size each | what it is |
| :--- | ---: | :--- |
| `s32[5376,1024,8]`, `s32[8192,672,8]`, `s32[672,8,32,256]` | **168 MiB** | int4 **nibble expansion**, `[out, in/8, 8]` |
| `bf16[21504,168,32]`, `bf16[5376,672,32]` | **220.5 MiB** | **dequantized weights**, grouped `[out, in/32, 32]` |
| `f32[16,2,1024,1024]` | 134 MiB | attention scores — the S² term |
| `f32[1024,21504]` | 88 MiB | MLP activations |

Dozens of the 168–220 MiB unpack/dequantize buffers are live simultaneously. The
`in/8` and `in/32` groupings match the checkpoint's pack layout exactly, so these
are unambiguously the W4A16 path materializing its intermediates.

## Why this ties the whole report together

The `reference` W4A16 implementation is now implicated in **both** headline
problems:

- it spends **~8.8 GB of prefill working set** on unpack/dequantize intermediates,
  which is most of what leaves only ~3 GB for sequence-dependent work; and
- it is the reason decode sits far off any weights-only bandwidth floor.

The fused Pallas kernel exists precisely to avoid materializing these — and it
**cannot be used**, because it exceeds the 32 MB scoped-VMEM limit at the 31B's
5376x21504 projections (and is 5.4x slower at the small shapes where it does
compile).

So the highest-value engineering item is not a flag and not a per-token saving:
**make a dequantize path that tiles within VMEM at this model's projection widths.**
That one fix addresses the prefill ceiling and the decode gap at the same time.
Everything else in the next-actions list is second order to it.

---

# Addendum 4: the dequantize penalty, measured on the 31B — and it compounds with depth

Addendum 3 named the floor as the W4A16 working set. This quantifies what that path
costs in *time*, measured on the 31B rather than inferred from E2B's
`dequant_at_load` result.

The 31B cannot run dense at full depth (~62 GB against 33.55 GB), but a **truncated**
31B can, so w4a16 and dense are compared at identical depth on the same chip.
`ports/gemma4/jax_31b_dequant_cost.py`, decode step, median of 7, S=1024, B=1:

| layers | mode | weights | step | tok/s | weight GB/s | % of 1640 GB/s |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 8 | w4a16 | 5.01 GB | 12.17 ms | 82.2 | 411.9 | 25.1% |
| 8 | **dense** | 10.61 GB | 10.03 ms | 99.7 | **1057.9** | **64.5%** |
| 15 | w4a16 | 6.93 GB | 20.91 ms | 47.8 | 331.6 | 20.2% |
| 15 | dense | — | — | — | — | OOM (`RuntimeProgramAllocationFailure`) |

**The dense path reaches 64.5% of the published roofline; the W4A16 reference path
reaches 20–25%.** Per byte, dense is **2.57x** more efficient at 8 layers. An earlier
draft estimated ~5.9x by extrapolating E2B's `dequant_at_load` figure — **that was
too high**, and this is the direct measurement in the 31B's own geometry. (The 2.57x
is diluted: the unquantized 2.82 GB embedding is identical in both configurations and
is a large share of an 8-layer model.)

## The finding that matters: efficiency degrades with depth

| depth | w4a16 weight GB/s | % roofline |
| ---: | ---: | ---: |
| 8 layers | 411.9 | 25.1% |
| 15 layers | 331.6 | 20.2% |
| 60 layers (full) | 246 | 15.0% |

Monotonic, and it matches the working-set growth measured in Addendum 3
(147.3 MB/layer). More depth means more live unpack/dequantize buffers and less
reuse. It also explains the cross-model ordering at full depth — E2B (35 layers,
~276+ GB/s) > 12B (48 layers, 276 GB/s) > 31B (60 layers, 246 GB/s): **the deeper
the model, the worse this path performs, and the 31B is the deepest.**

The 31B is therefore the worst case on both axes at once: deepest stack, and the only
one of the three that cannot fall back to `dequant_at_load` because dense does not
fit.

## The size of the prize

If a VMEM-tiled kernel reached the dense path's measured per-byte efficiency
(1058 GB/s) on the full 19.36 GB: **18.3 ms/step ≈ 55 tok/s, a 4.3x speedup** over
the measured 12.8. That is an upper bound from a measurement taken at 8 layers, and
the depth curve above says a real kernel will land below it — but it sizes the
opportunity, and it is the same fix that returns ~8.8 GB of prefill working set.

---

# Addendum 5: chunked prefill hits a compilation cliff at 60 layers

Earlier, chunked prefill was written off: it OOMed at 8K for chunk sizes 1024, 512
and 256, and the explanation offered was that chunking forces `window_kv=False` and
so adds 7.4 GB of unwindowed KV. **That explanation was incomplete.**

`ports/gemma4/jax_31b_chunk_temp.py` compiles the chunk *step* alone and reads its
temp, separating the step's cost from the KV cost. At `total_len=4096, chunk=256` the
KV is only 3.69 GB — but the step needs **32.32 GiB (34.70 GB) of temporaries**,
against 9.63 GiB for the *unchunked* full prefill at the same 4096. Chunking is
**3.4x worse**, which is the opposite of its purpose.

## Where it goes wrong: between 30 and 60 layers

| layers | chunked step temp | one-shot prefill temp (S=4096) |
| ---: | ---: | ---: |
| 6 | 1.988 GB | 2.184 GB |
| 15 | 5.090 GB | 5.209 GB |
| 30 | **6.601 GB** | 7.443 GB |
| 60 | **OOM — needs 34.70 GB** | 9.174 GB |

Chunked tracks the one-shot path closely through 30 layers and is *slightly better*
there. Then it regresses **5.3x for a 2x increase in depth**, while the one-shot path
grows 1.2x over the same step.

**So chunked prefill is not fundamentally broken — it hits a compilation cliff at
this model's depth.** That is investigable and plausibly avoidable (buffer donation
across chunks, `lax.scan` instead of a Python loop over jitted calls, or simply a
different schedule), and it is a far more tractable problem than the one that was
assumed.

## The meta-finding

This is the second cliff of exactly this shape found today:

| cliff | below | above |
| :--- | :--- | :--- |
| attention tiling, 4K → 8K | temp 10.34 GB | temp 41.10 GB (3.98x) |
| chunked step, 30 → 60 layers | temp 6.60 GB | temp 34.70 GB (5.3x) |

In both, XLA behaves well up to a threshold and then changes strategy, and memory
explodes. **The 31B sits just past both thresholds** — 60 layers deep, and needing
8K+ contexts to be useful. Neither cliff is visible on E2B (35 layers) or the 12B (48
layers), which is why nothing in this repository predicted them.

That is the real character of this port: the architecture needed no new code, and
almost every difficulty came from being the first model here big enough to fall off
XLA's well-behaved region. Anyone extending this work should assume more cliffs
exist, and should measure with `memory_analysis()` at the *actual* target shape
rather than extrapolating from a smaller one — every extrapolation attempted in this
report was wrong.

---

# Addendum 6: the chunked cliff is a missing `donate_argnums` — fixed

Addendum 5 found chunked prefill costing 34.70 GB of temp at 60 layers where the
one-shot path costs 9.17 GB, and called it "a scheduling regression, not a design
flaw". That was right, and the cause turned out to be one missing argument.

`ports/gemma4/jax_31b_chunk_cliff.py`, 60 layers, chunk=256, `window_kv=False`,
compile-time temp for ONE chunk step:

| total_len | KV | `donate=False` | `donate=True` |
| ---: | ---: | ---: | ---: |
| 1024 | 0.92 GB | 8.785 GB | 8.520 GB |
| 2048 | 1.84 GB | 8.027 GB | 8.793 GB |
| **4096** | 3.69 GB | **OOM — 34.70 GB** | **8.013 GB** |
| 8192 | 7.38 GB | OOM — 41.87 GB | OOM — 34.80 GB |

Below 4096 donation is scheduling noise and occasionally *worse*. At 4096 it is the
difference between "does not compile" and 8.013 GB — a **4.3x** reduction.

`make_chunked_prefill_step` threads `caches` and `valid` through and reassigns both
every chunk, exactly as `make_cached_decode_step` does. The decode step is jitted
with `donate_argnums=(1, 2)` and `jax_engine.py` documents a measured 1.62x for it.
**The chunked prefill step was simply never given the same treatment**, so XLA had to
allocate a fresh copy of the entire KV cache per chunk — and at 60 layers x 4096 that
cascaded into a catastrophic schedule rather than merely costing 3.69 GB.

**Fixed** in `chunked_prefill_with_kv_cache` (jax_e_model.py): the step is now jitted
with `donate_argnums=(1, 2)`, with the measurement table recorded at the call site.
Safe because nothing outside the loop holds a reference to `caches` or `valid` across
iterations. `tests/test_chunked_prefill.py` and `tests/test_donation.py` pass (13
tests), as do all 86.

## It does not unlock 8K — and the reason is now isolated

At `total_len=8192` donation cuts the requirement from 41.87 GB to 34.80 GB — still
above the 33.55 GB budget, by **~1.2 GiB**. Reducing KV dtype will not help: KV is an
*argument*, and the 34.80 GB is *temp*.

**The 8192 cliff is driven by the key dimension, not the query dimension.** Sweeping
chunk size at `total_len=8192` with donation on:

| chunk | temp required |
| ---: | ---: |
| 128 | 32.44 GiB |
| 256 | 32.41 GiB |
| 1024 | 32.98 GiB |

**An 8x range in chunk size moves the requirement by 1.7%.** Chunking bounds the
query count; it does nothing here. So no chunk-size tuning will ever reach 8K, and
the trigger is `total_len` — the extent of keys each layer attends over.

That is a useful negative result with a positive corollary: the lever that *would*
matter is capping the key extent, which is precisely what windowed KV does for the
50 sliding layers (1024 keys instead of 8192). See the windowed-chunked-prefill
proposal in the next-actions list — this measurement is what promotes it from a
guess to the indicated next experiment.

So the honest scoring of this fix: it removes a real pathology and makes the chunked
path usable at 4096 where it previously would not compile. It does **not** by itself
extend context, because the one-shot path already handles 4096 more cheaply
(29.7 GB total vs 31.1 GB). Its value is that chunking is now a *viable* mechanism to
build on rather than a dead end — and it is 1.2 GiB away from working at 8K.

**This is the third cliff of the same shape** (after attention 4K→8K and chunked-step
30→60 layers), and the first one with a known, one-line cause. That is weak evidence
the other two may also be schedule-level rather than fundamental.

---

# Addendum 7: KV buffer size changes prefill logits — numerical, not a logic bug

**RESOLVED, and the first version of this section was wrong.** It was written as
"an open correctness bug ... the most serious finding in this report" and committed
in that form. Two follow-up measurements retracted it. The measurements below are
kept because the *sensitivity* is real and worth knowing; the alarm was not.

## What was measured

`prefill_with_kv_cache` only — no chunking involved. Random-token prompts, greedy,
w4a16, bf16 cache.

**(1) `window_kv=False` vs `window_kv=True`, identical input:**

| S | sliding buf F/T | max abs diff | rel | argmax match |
| ---: | :--- | ---: | ---: | :--- |
| 512 | 520 / 520 | **0.00000** | 0 | yes (buffers identical) |
| 1024 | 1032 / **1024** | 2.41263 | 12.0% | **no** |
| 2048 | 2056 / 1024 | 3.54227 | 13.6% | **no** |

At S=512 the two configurations allocate the *same* buffer and agree **bitwise** —
so the harness is sound and the computation is deterministic. At S=1024 the buffers
differ by **eight slots**, both take the same `S <= buf_len` store path (a plain
`dynamic_update_slice` at offset 0), and the logits move **12%** with a different
top-1 token.

**(2) Ground truth — the same model with `kv_caches=None`, S=1024:**

| path | top-1 | vs no-cache |
| :--- | ---: | :--- |
| no cache (reference) | 237221 | — |
| `window_kv=True` | **237221** | **argmax MATCHES** |
| `window_kv=False` | 759 | argmax differs |

## The two measurements that resolved it

**(3) Tiny-scale CPU reproduction — `tests/test_kv_buffer_size_invariance.py`:
6/6 PASS.** At 4 layers with an fp32 cache there is no buffer-size dependence at
all, to within 1e-5. A logic bug in shared engine code would have reproduced here.

**(4) The same TPU A/B with a REAL prompt** instead of random tokens (chat-templated
and repeated to 1024 tokens so the window binds):

| | top-1 | |
| :--- | ---: | :--- |
| `window_kv=False` | 624 `'and'` | |
| `window_kv=True` | 624 `'and'` | **MATCH** |

`max|diff| = 0.6887` (rel 3.0%), **top-1-to-top-2 margin = 1.565**. The divergence
is well under the margin, so the argmax is stable.

## What was actually going on

Different buffer sizes produce different HLO, therefore different fusion and
accumulation order, therefore ~3% bf16 divergence accumulated over 60 layers. That
is ordinary floating-point non-associativity, not a leak of cache contents into the
attention.

**The 12% / flipped-argmax result came from using random-token prompts.** Those are
far out of distribution: the logits are near-flat, top-1 margins are ~0, and any
reordering flips the argmax. The measurement was correct; the interpretation was
not. At S=512 the two configurations compile to the *same* HLO (identical buffers),
which is why they agreed bitwise — that was the clue, and it was misread as
evidence of a bug rather than evidence of a compilation difference.

## What remains true and worth knowing

- **Greedy decoding is not bit-reproducible across `window_kv` settings**, and on
  out-of-distribution input the sampled token can genuinely differ. Anyone doing
  A/B evaluation must hold `window_kv` fixed.
- **Never benchmark correctness on random tokens.** A flat-logit prompt makes
  argmax comparison meaningless and manufactured a false alarm here.
- The invariance test is retained as a regression guard: it encodes the property
  that *should* hold, and it would catch the real bug this was mistaken for.

---

## Next actions, in order

0. ~~The KV-buffer-size dependence in Addendum 7.~~ **RESOLVED — not a bug.** It is
   fusion-order divergence (~3% in bf16 over 60 layers), and the alarming 12% /
   flipped-argmax reading came from benchmarking correctness on random tokens.
   `tests/test_kv_buffer_size_invariance.py` is retained as a regression guard.
   Standing rule it produced: **hold `window_kv` fixed across any A/B, and never
   judge correctness on out-of-distribution prompts.**

1. ~~Slice the hidden state before the lm_head in prefill.~~ **DONE — see addendum.**
   Mathematically exact (not bitwise on TPU), worth ~21.7 GiB at B=2 x 2K and
   **0.01 GiB at 8K**, so it made B=2 x 2K feasible but did **not** break the 4K
   ceiling. Superseded by (2).
2. ~~Run XLA `memory_analysis` on the compiled 8K prefill HLO.~~ **DONE — see above.**
   Temp is 10.34 GB at 4K and 41.10 GB at 8K (3.98x); the ceiling is XLA ceasing to
   tile the attention, not any single removable buffer.
3. ~~Find where the tiling stops, and why.~~ **DONE — see Addendum 2.** It is the
   softmax over dense `f32[16,2,S,S]` scores, 8.59 GB per buffer at 8K.

   **3a. Window the sliding layers' prefill attention.** 50 of 60 layers have a
   1024 window and need `[16,2,S,1024]`, not `[16,2,S,S]` — **8x less at 8K**. The
   dense `[B,1,S,S]` mask from `make_prefill_causal_mask` is what makes them compute
   the full thing. Biggest single lever on the ceiling.

   **3b.** ~~Buffer-assignment dump to identify the floor.~~ **DONE — Addendum 3.**
   It is the W4A16 unpack/dequantize working set: 147.3 MB/layer, 8.836 GB of the
   9.174 GB floor, confirmed by layer ablation and by named buffers.

**THE ONE THAT MATTERS: make the W4A16 dequantize tile within VMEM.** It is the
common cause of both headline problems — ~8.8 GB of prefill working set *and* the
decode gap. The fused Pallas kernel is the right shape of answer and currently
exceeds the 32 MB scoped-VMEM limit at 5376x21504; fixing its tiling is worth more
than every other item on this list combined. Measured ceiling on the prize
(Addendum 4): the dense path reaches 1058 GB/s vs this path's 246 GB/s at full
depth, so ~4.3x on decode, plus the 8.8 GB back in prefill.

**RUNNER-UP:** ~~find the chunked-prefill cliff between 30 and 60 layers~~ **DONE —
Addendum 6.** It was a missing `donate_argnums=(1, 2)` on the chunk step; fixed,
34.70 GB → 8.013 GB at 4096. Chunking is now viable rather than a dead end, but is
still ~1.2 GiB short of compiling at 8192.

**NEW RUNNER-UP: close the last ~1.2 GiB at `total_len=8192`, chunk=256.** With
donation the chunked step needs 34.80 GB against a 33.55 GB budget. The attention
term there is only 268 MB, so the gap is elsewhere in the ~8 GB dequant working set
plus whatever cliffs at 8192. Closing it is the difference between a 4K-context 31B
and a long-context one, and it is the smallest remaining gap in this report.

**The specific idea worth trying first — chunked prefill WITH windowed KV, when
`chunk_size == sliding_window`.** (Design proposal, arithmetic only, NOT measured.)

`chunked_prefill_with_kv_cache` hard-requires `window_kv=False`, and
`jax_engine.py` explains why: "a chunk writes contiguous slots at an arbitrary
offset that a shorter ring buffer would wrap."

**⚠️ An earlier draft of this section proposed `chunk_size == sliding_window` as the
safe special case. That is WRONG, and dangerously so — do not implement it.** The
wrap alignment is fine; the *history* is not:

- a query at absolute position `p` must attend to keys in `(p - window, p]`
- a chunk covers `p` in `[slot, slot + chunk)`, so the keys it collectively needs
  span `(slot - window, slot + chunk)` — **`window + chunk - 1` positions**
- with `buf_len == window == chunk`, writing the current chunk fills the entire
  ring, **destroying the previous chunk's keys before the chunk's early rows can
  read them**

Rows near the start of each chunk would silently lose up to `window - 1` positions
of history. It would compile, run, and compute the wrong thing — the failure mode
this codebase keeps producing.

**The correct condition is a buffer, not a chunk size:** the sliding cache must hold
at least `window + chunk_size - 1` positions. Allocating
`buf_len = sliding_window + chunk_size` makes plain `p % buf_len` ring indexing
correct, because the entries a write evicts are exactly those `window + chunk`
positions old, which no query in this chunk can reach.

Two things would follow at `total_len=8192` with `chunk=256`, sliding buffers sized
`window + chunk = 1280`:

| | `window_kv=False` (today) | ring buffer of 1280 (corrected proposal) |
| :--- | ---: | ---: |
| KV (argument) | 7.38 GB | **~1.02 GB** |
| sliding-layer attention extent | 8192 keys | **1280 keys** |

The KV saving is in *arguments*, which does not touch the 34.80 GB temp. But the
second column might: **50 of the 60 layers would attend over 1280 keys instead of
8192**, a 6.4x reduction in the attention extent for 5/6 of the model. The
chunk-size sweep above established the 8192 cliff is key-driven, so this is the only
identified lever on it.

**Attempted, and it is NOT a guard relaxation.** Passing a windowed cache
(`window_kv=True`, `total_len=8192`, `chunk=1024`) into `make_chunked_prefill_step`
fails at trace time:

```
window_kv=False   KV 7.38 GB   OOM, needs 32.98 GiB
window_kv=True    KV 1.51 GB   ERROR: add got incompatible shapes for broadcast
```

Not an OOM — a **shape error**. The chunked step's mask and cache indexing assume
full-length buffers, so the ring case is unimplemented rather than merely forbidden.
The KV saving is confirmed (7.38 → **1.51 GB**, 5.9 GB off the arguments), but
whether it also removes the temp cliff is **unknown**: it fails before compiling, so
no temp figure exists.

Correctly scoped, this is: implement ring-aware masking and slot indexing in
`make_chunked_prefill_step` for the `chunk_size % sliding_window == 0` case, then
measure. Still the best-indicated next experiment — the chunk-size sweep shows the
cliff is key-driven and this is the only lever on the key extent — but it is a real
change with a KV-parity test attached, not a one-line guard edit. An earlier draft of
this section called it "cheap to try"; that was wrong.
4. **Scope the `jax_engine.py` "prefill is linear, ~2.13 MB/token" comment to E2B.**
   It is false for the 31B and it is the assumption that misled this investigation.
5. **Collapse the duplicated V buffer on full-attention layers.** `attention_k_eq_v`
   makes V identical to K, but `init_kv_cache` allocates both: 40 KiB/token of pure
   duplication, ~46% of KV at 128K.
6. **Leave `w4a16_impl="auto"` alone on this model** until the kernel fits scoped
   VMEM at 5376x21504. It is slower where it compiles and does not compile where it
   matters.
7. **Profile the decode step before optimizing it.** The gap against a weights-only
   floor is unattributed, not evidence of compute-boundness — that inference was
   made and retracted here, and on E2B before that. A profile that attributes the
   78.4 ms across weight reads, KV traffic, and the logits pipeline is the
   prerequisite for any decode work.

## Reproduce

```bash
# main run: config, load, parity, sweep
python3.13 ports/gemma4/jax_31b_port.py \
  --stages config,load,parity,sweep \
  --batch-sizes 1,2 --contexts 1024,2048,4096 \
  --json-out gemma4_31b_results.json

# fused-kernel A/B
python3.13 ports/gemma4/jax_31b_port.py --stages load,parity,sweep \
  --w4a16-impl auto --batch-sizes 1 --contexts 1024

# prefill memory probe
python3.13 ports/gemma4/jax_31b_prefill_probe.py --contexts 512,1024,2048,4096,8192
```

## Artifacts

- `ports/gemma4/jax_31b_port.py` — staged harness (config / load / parity / sweep /
  chunked), writes JSON after every cell so an expiring flex-start VM still leaves
  results. Subclasses `JaxGemmaEngine` to stream the checkpoint through host RAM;
  the base `load()` puts all 23.3 GB on device before filtering the towers, which is
  affordable at E2B scale and not at this one.
- `ports/gemma4/jax_31b_prefill_probe.py` — resident-HBM-vs-S probe.
- `results/gemma4_31b_results.json` — reference-impl run (config, load, parity, sweep).
- `results/gemma4_31b_results_fused.json` — `auto` / Pallas A/B.
- `results/gemma4_31b_prefill_probe.json` — memory probe.
- `results/gemma4_31b_chunked*.json` — chunked-prefill attempts (all OOM).
- `logs/` — console transcripts for each.

## Caveats

- Single flex-start VM, single run per cell, median of 3 with 1 warmup. No
  cross-VM replication.
- The two parity transcripts come from different runs (`reference` without stop
  tokens, `auto` with them); only the stop-token handling differs.
- `run31b_fused.log` is truncated: a duplicate launcher raced the surviving process
  and clobbered the file. The JSON results are intact and are what the report uses.
