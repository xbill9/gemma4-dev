# 2026-11-02 — context size against memory and speed, LiteRT-LM on the RTX 3050 OEM

**Question:** the rig holds about 1.4 GB of the 8 GB card. Does a larger context use the rest, and what
does it cost?

**Answer:** a 32k context takes about 4.0 GB instead of 1.4 GB, and on the same short prompt reads it at
0.40x and generates at 0.65x. Larger values behave exactly like 32k. `MAX_NUM_TOKENS` stays at the
engine default for chat speed; set 32768 for long conversations.

## Method

`litert-lm benchmark --backend gpu --speculative-decoding on -p 1024 -d 256 --max-num-tokens N`, one run
each, rig model and `LITERT_LM_DIR`. GPU memory sampled every 300 ms (`smi-N.txt`); the rig's own server
held 1,375 MiB on the card throughout, so the benchmark's share is peak − 1,375.

| `--max-num-tokens` | peak MiB | benchmark's share | prompt reading | generation |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 2,787 | 1,412 | 5,107 tok/s | 167.6 tok/s |
| 32,768 | 5,379 | 4,004 | 2,029 tok/s | 108.7 tok/s |
| 65,536 | 5,322 | 3,947 | 2,148 tok/s | 109.5 tok/s |
| 131,072 | 5,332 | 3,957 | 2,112 tok/s | 111.6 tok/s |

- Only 1,280 tokens are ever in context, so the slowdown at 32k comes from the size of the cache allocated,
  not from text held.
- 65,536 and 131,072 match 32,768 in memory and speed, and the engine logs no limit. Google's model card says
  the model supports up to 32k context; that the engine caps it there is inferred.
- The rig's server, started with no `max_num_tokens`, holds 1,316–1,381 MiB (first-light run), the same as the
  4,096 benchmark, so the server's default is likely 4,096 (inferred).
- Generation figures are with the built-in drafter on the benchmark's synthetic text; chat speeds are lower
  (first-light `REPORT.md`). The ratios between rows are what this run measures.
