# 2026-10-16 — exact Q4_0 12B drafter, RTX 3050 OEM

**Answer: Google's 12B drafter doubles generation on this card (2.01x at 2 drafted tokens), and
the exact Q4_0 build is 1.06x faster than the Q8_0 conversion at the same acceptance**, using
202 MiB less VRAM.

## The build

Google's `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant` (revision `18934064`) is 4 layers
of width 1024 (3 sliding, 1 full attention) with queries only, two projections to the target's
3840-wide state, and a tied 262,144 × 1024 embedding, all bf16 on a 4-bit grid.

- `convert_hf_to_gguf.py --outtype q8_0` (llama.cpp `7182ef71e`, `convert-q8_0.log`): 49 tensors,
  465,128,096 B. This is the metadata template.
- `gguf_exact.py`, the E2B/E4B drafter script from
  `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem` unchanged:
  **253,708,960 B**, every weight matrix Q4_0, `general.file_type` set to Q4_0. No group off the
  grid; 96.70–97.13% of values per tensor kind reconstruct to the bf16 source exactly
  (`build-12B.json`). Hashes in `sha256.txt`.
- Unlike the E2B and E4B drafters, this checkpoint ships no `masked_embedding` tensors, so the
  converter dropped nothing.

## Speed and acceptance (`spec_ab.py`, `summarize.py`, `summary.txt`)

The 2026-10-14 method: one llama-server per arm at `-c 8192 --parallel 1`, target the exact 12B
GGUF, six prompts, greedy ×3 plus one sampled pass (temperature 0.8, seed 1234), 256 tokens,
prompt cache off, GPU cooled to ≤40 °C per arm. No drafter ran first and last (28.1 both times).
Largest within-cell spread 0.5%.

| greedy, geometric mean of 6 prompts | tok/s | x | acceptance | VRAM |
| --- | ---: | ---: | ---: | ---: |
| no drafter | 28.1 | 1.00 | — | 7,255 MiB |
| Q8_0, 2 tokens | 53.4 | 1.90 | 0.661 | 7,771 MiB |
| **exact Q4_0, 2 tokens** | **56.4** | **2.01** | **0.658** | **7,569 MiB** |
| exact Q4_0, 1 token | 45.6 | 1.62 | 0.759 | 7,569 MiB |
| exact Q4_0, 3 tokens | 56.7 | 2.02 | 0.561 | 7,569 MiB |

Sampled: Q8_0 1.81x, exact Q4_0 1.90x (2 tokens), 1.64x (1), 1.90x (3).

- **Per prompt at 2 tokens** (exact Q4_0): code 2.40x, facts 2.30x, chat 1.94x, explain 1.93x,
  summary 1.82x, story 1.75x.
- **Q4_0 beats Q8_0 here**, where on E2B and E4B they were level: 1.057x overall at equal acceptance,
  1.036–1.069x per prompt. This drafter's Q8_0 file is 4.7x theirs (1024 wide, a 268M-value embedding scored
  over the full vocabulary), so its own weight reads are likely a larger share of a drafted step (inferred, not measured).
- **2 and 3 tokens are level overall**; 3 wins on code and facts and loses on story and summary.
  2 is the rig's setting.
- **The gain is larger than E4B's 1.72x** on the same card: acceptance is higher (0.66 against
  0.48), and each 12B target step likely costs more relative to a drafted step (inferred, not measured).

## Greedy text

With the drafter, greedy replies repeat exactly across the three runs, but **differ from the
no-drafter replies in 18 of 18 at 2 and 3 tokens, and 9 of 18 at 1**. Q8_0 and Q4_0 drafters at
2 tokens agree on 15 of 18. E2B showed the same divergence with no change in GSM8K or BFCL scores;
**answer quality with the drafter is not measured for the 12B.**
