# 2026-09-29 — n-gram speculative decoding for the single-user demo, GTX 1650 Ti

**Question:** does llama-server's n-gram speculative decoding make the one-user demo faster?

**Answer: not for the demo. Leave it off.** On a prompt seen for the first time, the best case is
**1.16x on a code rewrite** (`ngram-simple`), where the answer copies the input. Text edits and
open-ended answers are unchanged or slower, by up to 25%. Every variant that drafts also changes
the greedy output slightly.

## Setup

v2 exact GGUF, this rig's binary (llama.cpp `f95b0d9`), the demo serving flags (`-ngl 99 -fa 1
-c 8192 -t 6 -tb 12 --parallel 1 --reasoning off`), plus `--spec-type <type>` at its defaults.
Four prompts, each sent 3 times in a row per server start, at temperature 0 with the prompt cache
off:

- `edit-code`: add type hints to a 52-line Python file (`sample_code.py`) and return it whole
- `edit-text`: fix three spelling mistakes in a paragraph and return it whole
- `open-explain`: the demo's quantization question
- `open-story`: a 200-word story

Configs ran in ABBA order (none, simple, map-k, mod, cache, then reversed), with the card cooled
to ≤50 °C before each. Tokens/s is the server's own `predicted_per_second`. `spec_ab.py` is the
harness; `spec_ab.json` holds every row and every text; `summary.md` is its all-repeats table.

## First request (what a live user sees)

The first of the three repeats after each server start, averaged over the two passes, relative
to no speculation:

| | edit-code | edit-text | open-explain | open-story |
|---|---:|---:|---:|---:|
| none (tok/s) | 77.28 | 78.31 | 78.31 | 78.26 |
| `ngram-simple` | **1.159** | 0.752 | 0.994 | 0.994 |
| `ngram-map-k` | 1.112 | 0.995 | 0.993 | 0.995 |
| `ngram-mod` | 0.994 | 0.841 | 0.998 | 0.992 |
| `ngram-cache` | 1.111 | 0.900 | 0.808 | 0.927 |

## Repeats inflate two of them

`ngram-mod` and `ngram-map-k` keep state across requests, so the second and third identical
prompt can copy the first answer. `ngram-mod` reads 152.8 tok/s (1.98x) on repeated code edits,
against 0.99x the first time; it even gains on the open-ended prompts once they repeat (1.28x).
`summary.md` mixes all repeats and overstates both. A live user does not send the same prompt
twice.

## Speculation changes the output

With greedy sampling, speculative decoding is meant to reproduce the plain output exactly. Here it
does not. Against the no-speculation text:

- `edit-text`, char 356: "recogni**s**e" becomes "recogni**z**e" under `ngram-simple`, `ngram-mod`
  and `ngram-cache`.
- `open-explain` and `open-story` drift after 110–490 characters under `ngram-mod` and
  `ngram-cache`.
- `edit-code` output was identical under all four.

`ngram-cache` changed the text while accepting **zero** drafts. So the verification pass itself
is the cause, not a wrong draft: checking several tokens at once runs a batched forward pass on
different CUDA kernels than one-token decoding, and the logits differ enough to flip near-ties.
This is the same class of effect as the exact GGUF's CUDA-only chat flip in
`2026-09-29-exact-gguf-1650ti`. The changed answers are not worse, just different, but a demo that
wants reproducible answers should not have speculation on.

## Recommendation

- **Demo:** off, as `tpu.env` has it. The demo's prompts are conversational, which is where
  every variant is neutral or a loss.
- **If a demo step is "rewrite this code":** `--spec-type ngram-map-k` is the safe choice. It
  gave 1.11x on the code rewrite and was never more than 0.7% slower elsewhere in this set.
  `ngram-simple` is faster on code (1.16x) but lost 25% on the text edit.
- **Not adopted into `tpu.env`.** It would also change what the paired sweep measures.
