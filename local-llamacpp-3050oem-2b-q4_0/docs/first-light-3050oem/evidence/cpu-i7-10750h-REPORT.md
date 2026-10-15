# 2026-09-29 — Google's GGUF vs the v2 exact rebuild, CPU arm, i7-10750H

**Why this run exists:** in `2026-09-29-paired-sweep-cpu`, this arm's prompt time (TTFT) came out 9%
worse than in 2026-09-22's sweep. Two things had changed: the weights (Google's GGUF → v2) and the
heat (5x the throttle events). This run separates them. Both files are measured in one invocation,
alternating which runs first, with the same temperature gate.

**Answer: the file does not slow CPU prefill; the heat did. CPU decode is 1.06x faster on v2.**

## Setup

The CPU arm's binary (`~/llama.cpp/build-cpu`, llama.cpp `f95b0d9`), `CUDA_VISIBLE_DEVICES` empty,
`-ngl 0 -fa 1`, `-t 6` and `-t 12` (the server runs `-t 6 -tb 12`: decode on 6 threads, prefill
on 12), pp512/tg128, r=5. Four passes in ABBA order (Google, v2 / v2, Google / v2, Google /
Google, v2). Each pass waits 60 s and then for the package to reach ≤50 °C. `bench_abba.sh` →
`bench-abba.md`; `summary.md` is the table below.

## Result

| test | threads | Google p1–p4 | v2 p1–p4 | v2/Google (mean) | v2 ran first | Google ran first |
|---|---:|---|---|---:|---:|---:|
| pp512 | 6 | 90.28 / 85.72 / 85.04 / 88.34 | 86.24 / 89.03 / 88.90 / 84.37 | 0.998 | 1.042 | 0.955 |
| pp512 | 12 | 86.66 / 86.37 / 86.38 / 87.79 | 87.18 / 88.05 / 87.60 / 82.57 | 0.995 | 1.017 | 0.973 |
| tg128 | 6 | 17.93 / 17.22 / 17.86 / 17.89 | 19.02 / 19.14 / 19.03 / 18.17 | **1.063** | 1.089 | 1.038 |
| tg128 | 12 | 14.34 / 14.90 / 14.77 / 14.65 | 15.49 / 15.44 / 15.75 / 14.15 | 1.037 | 1.051 | 1.023 |

- **Prefill: no difference.** Whichever file runs first in a pass is ~4% faster at prefill,
  and the ABBA mean cancels that to within 0.5%.
- **Decode: v2 is faster in both orders**, by 3.8–8.9% at 6 threads. Less of the file is read per
  token: `token_embd`, the tied output projection read in full every token, is 31% smaller at Q4_0
  than at Q6_K.
- **12 threads decode slower than 6** for both files (14–16 against 17–19 tok/s). This matches
  `THREADS=6` in `tpu.env`: on this 6-core part the SMT siblings hurt decode.
- The package reached 77–86 °C by the end of every pass, logging 18,600–27,700 throttle events.
  Those counts are why only the within-pass ratios are quoted.
