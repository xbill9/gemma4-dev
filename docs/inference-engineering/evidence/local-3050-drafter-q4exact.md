# 2026-10-14 — exact Q4_0 drafter against Q8_0 and bf16, RTX 3050 OEM

**Answer: the exact Q4_0 drafter is as fast as the Q8_0 one, guesses the same tokens,
and scores the same, at 60% of the file size.** Drafter precision does not move acceptance at
all: Google's bf16 weights, unchanged, accept the same share of drafted tokens as either
4-bit or 8-bit, and run slower.

## The three drafters

Google's QAT assistant drafters (`google/gemma-4-E{2,4}B-it-qat-q4_0-unquantized-assistant`,
revisions in each rig's `tpu.env`) ship bf16 weights that already lie on a 4-bit grid: at most
16 distinct values per group of 32 in every weight matrix.

| | Q8_0 (`make draft`) | exact Q4_0 (`gguf_exact.py`) | bf16 (`gguf_exact.py --bf16`) |
| --- | ---: | ---: | ---: |
| E2B file | 97,836,704 B | 59,236,000 B | 170,213,024 B |
| E4B file | 98,672,288 B | 59,678,368 B | 171,785,888 B |
| relative L2 error vs source (E2B / E4B) | 0.00494 / 0.00486 | 0.00138 / 0.00138 | 0 |
| values bit-identical, dequantized in fp32 | 11.7% / 11.4% | 29.4% / 28.7% | 100% |
| values bit-identical after rounding to bf16 | — | 96.6–97.1% per tensor | 100% |

`gguf_exact.py` is the 2026-09-29 exact-GGUF builder
(`local-llamacpp-i71360p-2b-q4_0/benchmarks/runs/2026-09-29-exact-gguf-i71360p/`) with the
drafter's tensor names and the Q8_0 GGUF as the metadata template; it rewrites
`general.file_type` to Q4_0. No group was off the grid. Hashes in `sha256.txt`, per-tensor
results in `build-*.json`. llama.cpp's converter drops the checkpoint's `masked_embedding`
tensors (`conversion/gemma.py:964`) in all three files.

## Speed and acceptance (`spec_ab.py`, `summarize.py`)

One llama-server per arm at `-c 8192 --parallel 1`, the rigs' serving flags, six prompts,
greedy ×3 plus one sampled pass (temperature 0.8, seed 1234), 256 tokens, GPU cooled to
≤40 °C per arm. The Q8_0 arm ran first and again last. Largest within-cell spread 0.7%.

| greedy, geometric mean of 6 prompts | E2B tok/s | x | acceptance | E4B tok/s | x | acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no drafter | 108.2 | 1.00 | — | 58.2 | 1.00 | — |
| Q8_0, 2 tokens | 150.1 | 1.388 | 0.475 | 100.4 | 1.723 | 0.487 |
| **exact Q4_0, 2 tokens** | **151.1** | **1.397** | **0.475** | **100.0** | **1.717** | **0.481** |
| bf16, 2 tokens | 133.1 | 1.231 | 0.475 | 92.3 | 1.585 | 0.484 |
| exact Q4_0, 3 tokens | 143.0 | 1.322 | 0.384 | 99.3 | 1.705 | 0.398 |
| bf16, 3 tokens | 122.0 | 1.128 | 0.383 | 89.4 | 1.536 | 0.403 |
| Q8_0, 2 tokens, repeated last | 150.3 | 1.390 | 0.475 | 100.4 | 1.724 | 0.487 |

Sampled: Q8_0 1.360 / 1.706x, exact Q4_0 1.373 / 1.707x, bf16 1.209 / 1.571x (E2B / E4B).
Tables per prompt in `e2b/summary.txt`, `e4b/summary.txt`.

- **Q4_0 and Q8_0 are level on speed**: +0.7% on E2B, −0.3% on E4B, the size of the Q8_0
  arm's own first-to-last difference (0.1%) plus prompt-level noise.
- **bf16 is slowest at equal acceptance**: 0.88x / 0.92x the Q8_0 arm.
- **Two drafted tokens stay best**; three win only on code, as on 2026-10-13.
- **VRAM at `-c 8192`**: E2B 2,795 / 2,915 / 2,875 / 2,983 MiB and E4B 4,415 / 4,537 / 4,499 /
  4,607 MiB for none / Q8_0 / Q4_0 / bf16. The exact Q4_0 drafter saves 38–40 MiB.
- **Greedy text**: every drafted arm produces the same 18 greedy replies as every other drafted
  arm of the same model, whatever the drafter's precision or draft length. On E4B these equal
  the no-drafter replies; on E2B all 18 differ from them. E2B's divergence therefore comes from
  drafting itself, not from quantizing the drafter.

## Answer quality with the drafter serving (`run_accuracy.sh`, `compare.py`)

Server flags of the 2026-10-13 quality run (`-c 32768 --parallel 8`, ...) plus the drafter at
2 tokens; GSM8K (1,319, 2,048-token limit) and BFCL simple (400) through `jev-tpu-v5e1/gen_eval.py`,
8 at a time. The suite is not rerun: it reads next-token probabilities, where no draft is made.
Pairing and 95% ranges are `gen_compare.compare`. Drafting works with 8 slots.

E2B, against the 2026-10-13 no-drafter run:

| | no drafter | Q8_0 | exact Q4_0 | Q4_0 − Q8_0 (95% range) |
| --- | ---: | ---: | ---: | --- |
| GSM8K | 0.8976 | 0.8992 | 0.9022 | +0.0030 (−0.0061 to +0.0129) |
| BFCL simple | 0.9250 | 0.9225 | 0.9225 | +0.0000 (identical answers) |
| GSM8K wall time | 1,760 s | 1,150 s | 1,167 s | |
| acceptance over GSM8K + BFCL | — | 0.7974 | 0.7972 | |

E4B, against the 2026-10-13 no-drafter run:

| | no drafter | Q8_0 | exact Q4_0 | Q4_0 − Q8_0 (95% range) |
| --- | ---: | ---: | ---: | --- |
| GSM8K | 0.9325 | 0.9310 | 0.9325 | +0.0015 (−0.005 to +0.008) |
| BFCL simple | 0.9100 | 0.9150 | 0.9150 | +0.0000 (−0.007 to +0.007) |
| GSM8K wall time | 2,267 s | 1,342 s | 1,345 s | |
| acceptance over GSM8K + BFCL | — | 0.8374 | 0.8382 | |

## Same-day no-drafter control (`run_control.sh`)

Both models again with no drafter, same flags, after the drafted arms. Full tables in
`accuracy/COMPARE.md`, numbers in `accuracy/compare.json`.

| | E2B GSM8K | E2B BFCL | E4B GSM8K | E4B BFCL |
| --- | ---: | ---: | ---: | ---: |
| no drafter, 2026-10-13 | 0.8976 | 0.9250 | 0.9325 | 0.9100 |
| no drafter, today | 0.8984 | 0.9250 | 0.9340 | 0.9100 |
| Q8_0 drafter | 0.8992 | 0.9225 | 0.9310 | 0.9150 |
| exact Q4_0 drafter | 0.9022 | 0.9225 | 0.9325 | 0.9150 |

- **Every drafted arm is level with today's no-drafter run**: each paired difference's 95% range
  includes zero, except E4B BFCL with Q8_0 (+0.005, range +0.000 to +0.013, 2 problems).
- **The no-drafter baseline reproduces in score but not in text.** Two no-drafter E2B runs disagree on
  59 GSM8K problems (29 against 30) and give the same text on only 175 of 1,319; E4B, 20 problems and
  366. At 8 concurrent requests, batch composition changes greedy text from run to run. Q8_0 and Q4_0
  drafters disagree on 42 E2B problems and 16 E4B problems, which is within that run-to-run variation.
- **Drafting finishes the work faster at 8 concurrent requests**: GSM8K 1.54x (Q8_0) and 1.51x (Q4_0)
  on E2B, 1.69x on E4B for both; BFCL 1.18–1.22x.
