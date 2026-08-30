# Right-padding evicts real tokens from the sliding-window KV ring

**Status: FIXED 2026-08-24, verified end-to-end on CPU and on a T4G.**
Root cause measured 2026-08-23 on `i-02f74ac9b944576c5` (g5g.2xlarge, T4G). Nothing in the
mechanism is chip-specific — which is what made a CPU reproduction possible, see "The fix".

**On the TPU rig — inspected, not tested.** `ports/gemma4/jax_e_model.py` is NOT byte-identical
between the rigs: 1,842 lines here against 1,570 in `tpu-jax-v5e1-2b`, so "it is shared,
therefore it is affected" is not a valid argument and an earlier revision of this file made it.
What is true is that every ingredient is present in that copy too — `make_ring_decode_mask`,
the same `layer_len = min(max_seq_len, sliding_window)` ring sizing, the same pad-gap
docstring, and the same bucket-derived decode slot (`jnp.int32(S + t)`, line 1463). So it very
likely reproduces there. **It has not been run on TPU hardware to confirm.**

## Symptom

The model returns fluent output for most prompts and, for some, a token loop:

```
'TheTheTheTheTheTheTheTheTheTheTheTheTheThe...'
```

It is recorded as `status="success"`. Nothing in `/health`, the metrics, or the benchmark
harness distinguishes it from a good answer. That is what makes it dangerous: it is a
silent correctness failure in the path every benchmark number goes through.

## It is not a long-context bug

This was the first wrong conclusion, and it is worth stating plainly because the symptom
invites it. Degeneracy is **non-monotonic in prompt length**:

| prompt tokens | 2,015 | 2,615 | 3,015 | 3,315 | 3,515 | 4,015 |
| --- | --- | --- | --- | --- | --- | --- |
| result | ok | **loop** | **loop** | **loop** | **loop** | ok |

It is also **not numerical**. `JAX_E_COMPUTE_DTYPE=bfloat16` — which XLA emulates through
fp32 on Turing, i.e. strictly more headroom than float16 — reproduces the table
byte-for-byte. fp16 range is not involved.

## The actual variable is padding

`pad_to_bucket` right-pads every prompt to the next power-of-two bucket. Sorting the same
data by `pad_len = bucket - tokens` makes it exact:

| tokens | bucket | pad | result |
| ---: | ---: | ---: | --- |
| 1,415 | 2,048 | 633 | **loop** |
| 1,515 | 2,048 | 533 | **loop** |
| 1,595 | 2,048 | 453 | ok |
| 2,015 | 2,048 | 33 | ok |
| 3,415 | 4,096 | 681 | **loop** |
| 3,515 | 4,096 | 581 | **loop** |
| 3,815 | 4,096 | 281 | ok |
| 4,055 | 4,096 | 41 | ok |

Predicting `pad_len >= 512 => degenerate` scored **14/14** on a sweep across two buckets.
The boundary is bracketed between 453 (ok) and 533 (loop), and `sliding_window = 512`.

**A 1,415-token prompt fails while a 4,055-token prompt succeeds.** Length is irrelevant;
padding is the whole effect. Long prompts merely make large padding likely.

## Mechanism

`window_kv` resolves to **True** whenever `max_model_len > sliding_window`
(`jax_engine.py:363`), which is every real configuration here — `MAX_MODEL_LEN=8192`
against a 512 window. Under it, `init_kv_cache` gives sliding layers a ring of exactly
`sliding_window` slots:

```python
if window_kv and is_sliding and config.sliding_window:
    layer_len = min(max_seq_len, int(config.sliding_window))   # 512
```

> The buffers are ring buffers indexed by `position % buf_len`.

Prefill writes all `S` padded slots into that ring, so it retains only positions
`S-512 … S-1`. Real tokens occupy `0 … S-pad_len-1`. A real token is still resident only if

```
pad_len <= 511
```

At `pad_len >= 512` the ring holds **nothing but padding**. `make_ring_decode_mask` then
correctly masks every entry as invalid, and the sliding layers attend to an entirely masked
window — no real context at all.

E2B declares `sliding_window: 512` for **28 of its 35 layers** (read off the loaded config on
the instance, not off a docstring: `sliding_window=512`, 28 of 35 `layer_types` are
`sliding_attention`, `first_kv_shared_layer_idx=15`). The remaining full-attention
layers use `make_decode_mask(valid)` with no window and are unaffected, which is why the
output is a degenerate loop rather than noise: a fifth of the network still sees the prompt.

## Why the existing guard does not catch it

`make_ring_decode_mask` already documents the pad gap:

> The cache is NOT filled contiguously: the server right-pads every prompt to a static
> bucket and then decodes at `bucket + step` while the logical position tracks the real
> length, so the pad slots `[real_len, bucket)` sit *inside* `[0, slot)`. Assuming a
> contiguous fill attends to pad K/V and corrupts the output with no error at all.

That fix is real and it works — it stops the model **attending to** pad K/V. It does not
address pad K/V **evicting** real K/V from a ring shorter than the padding. The hazard was
identified and hardened in one dimension only.

The non-ring path (`window_kv=False`) is **predicted** to fail at the same threshold for a
sibling reason — `make_decode_mask(valid, window, slot)` masks with `idx > slot - window`,
which measures distance in **slot space**, and slot space includes the pad gap. **This has not
been tested.** Every run here had `window_kv=True` (verified at runtime: the engine resolves it
from `max_model_len 8192 > sliding_window 512`). `--window-kv off` now exists precisely so this
can be checked, and until someone runs it the slot-space explanation is inference.

## Where the gap comes from

Two call sites pass a decode slot derived from the bucket while passing the logical position
derived from the real length:

- `jax_engine.py:478` — `prompt_lens + step_idx` (position, correct) vs
  `jnp.int32(bucket_s + step_idx)` (slot, skips the gap)
- `ports/gemma4/jax_e_model.py:1667` — the same mismatch in `generate_with_kv_cache`

## The failure is decided at the FIRST decode step

Padding does **not** progressively consume the window, and an early hypothesis here that it
did (`pad_len + k >= 511`) is **refuted by measurement**:

| pad | generated tokens | result |
| ---: | ---: | --- |
| 407 | 320 | coherent |
| 407 | **600** | **coherent** |
| 2,035 | 320 | loops from the first token |
| 2,035 | 600 | loops from the first token |

Each *generated* token is written into the ring at `slot % 512`, so decoding progressively
refills the ring with real tokens. If any real prompt token is resident at step 0, generation
bootstraps and stays coherent for as long as it runs. If the ring is entirely padding at step
0, the first token is produced with no real context and the model then loops on its own
output — which is why the degenerate cases begin at character 0 rather than degrading part
way through.

Practical consequence: **guaranteeing `pad_len < 512` appears sufficient to prevent the
failure**, not merely to postpone it. Note the limits of the evidence: padding values actually
tested clean run up to 453, and the long-generation check used pad=407. The range 454-511 is
predicted safe by the mechanism but was never exercised. That is what makes option 4 below a real fix for correctness
rather than a delay. It remains a partial fix for *quality*: every pad slot still costs a slot
of real sliding context at step 0, so pad=500 leaves only ~12 real tokens visible where
pad=0 would leave 512.

## The fix, applied 2026-08-24

**Option 2 as written below is NOT sufficient on its own, and that is the main thing learned
in fixing this.** Gating the prefill write by `prompt_valid` leaves the pad *indices* in the
cache's coordinate space, and `make_ring_decode_mask` reads `valid` at those indices and
correctly reports them invalid — so the ring is still unattendable, now holding stale zeros
instead of pad K/V. Masking cannot repair a layout problem. The gap has to be removed from the
coordinate space, not skipped over.

What landed is options 1 + 2 together, as a single invariant:

> **A cache index is an absolute real position.** Padding never occupies an index that a real
> position uses.

Three changes carry it:

- `_ring_store_one(buf, val, real_len)` — ring slot `j` receives the most recent *real*
  position `p < real_len` with `p % buf_len == j`, by gather rather than by the old two-slice
  copy (`real_len` is traced, so the split point can no longer be static). `real_len=None`
  keeps the old padded-space behaviour for the chunked-prefill path, whose `val` carries no
  padding. Cost is a gather of `buf_len` rows out of `S`, negligible beside the prefill.
- `cache_valid` threaded through `Gemma4EModelJAX.__call__` into the attention layer, so
  prefill can pass `prompt_valid` down to that store. Defaults to `None` everywhere.
- **Decode writes at `prompt_len + t`, not `bucket + t`** — `jax_engine.py` and
  `generate_with_kv_cache`. This is option 1, and with the gap gone it is what keeps the
  invariant true after prefill. Generated tokens now overwrite the former pad slots.

`make_ring_decode_mask` and `make_decode_mask` are **unchanged** and stay correct under the
new invariant: with no gap, every index they consult is a real position, so their `valid`
lookups simply always succeed. They remain the defence if the invariant is ever broken again.

**`B > 1` now raises `NotImplementedError` rather than silently reverting.** The decode slot is
a scalar shared by every row, and a row's real length only coincides with the bucket at
`B == 1`. Supporting `B > 1` needs a per-row scatter in the attention cache write. Both engines
in this tree serve `MAX_NUM_SEQS=1`.

Option 4 (the bucket ladder) **also landed, as defence in depth**: `static_sequence_buckets` is
now `(64, 128, 256)` plus 128-steps to 16384, so worst-case padding is **127 tokens** at every
length instead of `B/2`. That keeps `pad_len` below every `sliding_window` Gemma 4 declares
even if the store regresses, and preserves ~385 of the 512 ring slots for real context. Cost is
one compile per newly seen bucket, amortised by the persistent compilation cache. A test
asserts the 127 bound across all 16,384 lengths rather than sampling.

### Verification

Reproduced **and fixed** on CPU, which the "nothing here is chip-specific" claim predicted and
which no longer requires an instance. `tests/test_engine.py` builds a four-layer random model —
three sliding layers with `window=8`, one full-attention layer, the structure that makes the
failure a loop rather than noise — and generates from a 20-token prompt at several paddings:

| pad | pre-fix output | post-fix |
| ---: | --- | --- |
| 0 | `28 13 59 54 13 21 39 24 …` | same |
| 4 | `28  4 20 42  0 36 28 22 …` diverged | **matches pad=0** |
| 8 | `28 45  1 23 23 44 44 44 44 …` | **matches pad=0** |
| 12 | `28 45  1 23 23 44 44 44 44 …` | **matches pad=0** |
| 28 | `28 45  1 23 23 44 44 44 44 …` | **matches pad=0** |
| 44 | `28 45  1 23 23 44 44 44 44 …` | **matches pad=0** |

Note the pre-fix column reproduces the reported signature exactly: every padding at or above
the window returns the *same* sequence as every other — the prompt has stopped mattering — and
that sequence repeats a token four times running. Paddings below the window each diverge
differently, which is the graded quality loss rather than the cliff.

The property asserted is **padding invariance**: the generated tokens must not depend on how
much the prompt was padded. That is stronger than "does not loop" and it is what the old code
violated.

**Confirmed on hardware 2026-08-24**, `i-02f74…` replaced by `i-0bca12be1046b5faf`
(g5g.2xlarge, T4G, fp16, `window_kv=True`, real `sliding_window=512`). The check forces the
OLD power-of-two ladder back in, so `pad_len >= 512` is reproduced rather than avoided — which
is what isolates the cache-store fix from the ladder:

| ladder | tokens | bucket | pad | result |
| --- | ---: | ---: | ---: | --- |
| new | 1,515 | 1,536 | 21 | ok |
| new | 3,515 | 3,584 | 69 | ok |
| **old** | **1,515** | **2,048** | **533** | **ok** — looped on 2026-08-23 |
| old | 1,595 | 2,048 | 453 | ok (was ok before too) |
| **old** | **3,515** | **4,096** | **581** | **ok** — looped on 2026-08-23 |

So the ladder really is only defence in depth; the store fix is what removes the failure.

Two caveats on that table. The prompt is repeated filler ("The quick brown fox…"), so
"coherent" means the model resumes the pattern mid-sentence — ` dog.The quick brown fox…`,
with correct word boundaries and punctuation, against the 2026-08-23 signature of
`TheTheTheThe` with no spaces at all. A non-repetitive prompt would be a sharper test. And
nothing here was run on TPU, so `tpu-jax-v5e1-2b` remains untested and unfixed.

**The TPU rig is not fixed.** `tpu-jax-v5e1-2b` has its own diverged copy of
`jax_e_model.py` (1,570 lines against 1,842 here) and nothing here touched it.

## Fix options as originally written, for the record

1. **Close the gap.** Decode into `prompt_len + k` rather than `bucket_s + k`. Correct and
   minimal for `B=1` (the only shape either rig runs — `MAX_NUM_SEQS=1`, and the engine
   calls `ids[None, :]`). Needs per-row scatter for `B>1`, which is why it is not a
   one-liner in the shared port. **Does not by itself fix the ring**, because the pads are
   written during *prefill*, before any decode slot is chosen.
2. **Do not commit pad K/V to the cache at all.** Gate the prefill cache write by
   `prompt_valid`. Described here as "the fix that actually addresses the ring, and it
   subsumes (1)" — **that was wrong**, see above: it addresses the ring's *contents* and not
   its *coordinates*, and the mask then rejects the slots anyway. 1 and 2 are both required.
3. **Left-pad instead of right-pad.** Real tokens land at the end and survive the ring
   naturally, but position ids and the causal mask both have to change with it. Not taken.
4. **Cap `pad_len < 512` via a finer bucket ladder.** Taken, as defence in depth rather than
   as the remedy.

## Reproducing

```
--model google/gemma-4-E2B-it-qat-w4a16-ct --quant-mode auto --ple-bits 4 --max-model-len 8192
```

Send a prompt of 1,515 tokens (loop) and one of 1,595 tokens (fine). Both sit in the 2,048
bucket; only the padding differs. `--window-kv on|off|auto` selects the path.

`tpu_jax_degenerate_responses_total` counts occurrences; it is observational and does not
change the response or the status code. It is kept after the fix precisely because it does not
depend on the eviction explanation being the only cause.
