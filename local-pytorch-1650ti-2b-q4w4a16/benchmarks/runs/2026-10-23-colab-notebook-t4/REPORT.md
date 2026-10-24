# 2026-10-23 — the Colab notebook on a T4 and on the GTX 1650 Ti

`colab/src/01_gemma4_e2b_int4_on_4gb.py` run whole with PyTorch capped at 3,732 MiB, the memory a GTX 1650 Ti shows.

- **Colab T4** (`colab exec`, driver 580.82.07, torch 2.11.0+cu130, triton 3.6.0, transformers 5.16.1), engine fetched
  at `a836a6aa`: `t4-exec-final.log` (progress bars removed).
- **GTX 1650 Ti** (driver 615.71.10, torch 2.13.0, triton 3.7.1), engine files from the working tree at the same
  content: `local-1650ti.log.gz`.

| | Colab T4 | GTX 1650 Ti |
| --- | ---: | ---: |
| 16-bit weights in the cap | out of memory | out of memory |
| Transformers' 4-bit path, first forward | out of memory (loaded 1,775 MiB, peak 3,454) | out of memory (loaded 1,775, peak 3,363) |
| GEMM `[256,1536]x[1536,24576]`, fp16 / fp32 TFLOP/s | 36.58 / 6.68 | 0.42 / 1.58 |
| Prompt-read multiplies chosen | fp16 | fp32 |
| Engine load | 280 s | 41 s (downloads cached) |
| Weights + 32k K/V cache | 2,498 + 198 MiB | 2,499 + 198 MiB |
| Decode, 3 chat prompts | 101.6 / 117.5 / 117.3 tok/s | 86.1 / 87.5 / 87.4 tok/s |
| Peak, chat | 2,793 MiB | 2,852 MiB |
| 32,491-token recall prompt: read | 1,508.2 tok/s | 147.2 tok/s |
| Decode with 32k held | 69.2 tok/s | 50.8 tok/s |
| Peak, 32k | 2,836 MiB | 2,908 MiB |
| Recall (record 270 of 1,082, code 999494) | found | found |

The notebook filters one `torch._dynamo` warning ("Encountered an exception in identify_…"): on torch 2.11 Inductor's
mutation analysis cannot trace `scale.to(tl.float32)` in `w4attn._dec_split` when `scale` is a Python float, falls
back to treating every input as mutated, and logs about 19,000 lines; results are unaffected. The cast stays: on torch
2.13 the float arrives as fp64 and the cast keeps the softmax state fp32. Earlier T4 runs (`t4-exec-1.log.gz`, engine
`c44bbf37`) gave the same picture.

Compute: four T4 sessions, 64.34 to 61.06 Colab compute units.
