# 2026-10-16 — answer quality, 12B on the RTX 3050 OEM, with and without the drafter

`run.sh` serves the exact Q4_0 12B as the rig does (`-c 24576 --parallel 1`, f16 cache, thinking
off, prompt cache at the server default), once with the exact Q4_0 12B drafter at 2 tokens and
once without, and runs `jev-tpu-v5e1/gen_eval.py` one request at a time: GSM8K (1,319 questions,
2,048 tokens) and BFCL simple (400). Same method as
`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem`, whose E4B
scores are the reference here (llama.cpp `77c4540ce`, `-c 131072`, so a different build and
context). Paired figures from `compare.py`, raw output `compare.txt`. Finished 2026-10-17 00:00;
the rig's live server was restored after.

| | 12B, drafter | 12B, no drafter | E4B, drafter | E4B, no drafter |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | **0.9598** (1,266 / 1,319) | **0.9629** (1,270 / 1,319) | 0.9295 | 0.9310 |
| BFCL simple | **0.9500** (380 / 400) | **0.9475** (379 / 400) | 0.9125 | 0.9125 |

**The 12B answers more questions than E4B on both tests, and the drafter changes neither score.**

- **12B against E4B, paired by question:** GSM8K +0.0303 (95% range +0.0182 to +0.0432) with the
  drafter and +0.0318 (+0.0190 to +0.0440) without; BFCL +0.0375 (+0.0100 to +0.0650) with the
  drafter and +0.0350 (+0.0075 to +0.0625) without. Every range is above zero.
- **Drafter against no drafter:** GSM8K −0.0030 (−0.0083 to +0.0023), 12 questions apart (4 right
  only with the drafter, 8 only without); BFCL +0.0025 (0 to +0.0075), one question apart
  (`simple_382`: without the drafter the 12B passed `room_type` as `"single room"`). Both ranges
  include zero.
- **Identical replies:** GSM8K 332 of 1,319 word for word; BFCL 396 of 400 tool calls.
- **Time:** without the drafter GSM8K took 2.34x as long (15,361.7 s against 6,565.9 s) and BFCL
  1.73x (595.9 s against 344.1 s).
- Replies stopped at the 2,048-token limit: 3 with the drafter, 2 without.
- Memory at load: 7,820 MiB with the drafter, 7,500 MiB without (`driver.log`).
