# 2026-10-23 — Colab notebook on a T4 (in progress)

`colab/src/01_gemma4_e2b_int4_on_4gb.py` run whole with `colab exec` on a Colab Tesla T4 (driver 580.82.07, torch
2.11.0+cu130, triton 3.6.0, transformers 5.16.1), engine fetched at `c44bbf37`, PyTorch capped at 3,732 MiB.
Log: `t4-exec-1.log.gz` (gzipped; most of its lines are one torch warning, see below).

- Default paths: 16-bit load and Transformers' own 4-bit path both out of memory in the budget (4-bit: loaded
  1,775 MiB, peak 3,454 at the first forward).
- Prompt-read multiplies timed: fp16 36.57, fp32 6.47 TFLOP/s, so fp16 was kept.
- Engine: weights 2,498 MiB, K/V cache (32,768) 198 MiB, 2,714 MiB allocated after load; load 283 s.
- Chat, 3 prompts: decode 101.9 / 117.9 / 117.6 tok/s; peak 2,793 MiB.
- 32,491-token recall prompt (1,082 records): prompt read 1,539.9 tok/s, decode 69.7 tok/s, answer correct
  (999494), peak 2,837 MiB.

On torch 2.11 the Triton mutation analysis raised on `scale = scale.to(tl.float32)` in `w4attn._dec_split` (scale
arrives as a Python float) and fell back to treating every input as mutated: about 19,000 warning lines, results
unaffected. The line is removed after `c44bbf37`; the rerun on the fixed engine is still to come.
