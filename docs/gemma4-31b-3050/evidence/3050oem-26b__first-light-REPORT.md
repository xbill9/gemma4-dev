# 2026-10-18 — first light: Gemma 4 26B A4B QAT on the RTX 3050 OEM (8 GB)

llama.cpp `7182ef71e` (CUDA, sm_86), host `gpu` (i7-5820K, 6 cores / 12 threads, 46 GiB RAM).
Target: Google's `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` at `d1c082be`, unmodified,
14,439,363,584 B (SHA-256 in `sha256.txt`, matches the Hub's LFS oid). Drafter: Google's
`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` at `9537141`, stored as exact Q4_0 with
the 12B sister's `gguf_exact.py`, published as
`xbill9/gemma-4-26B-A4B-it-qat-assistant-q4_0-exact-gguf` at `b3116a27`.

## Result

**The 26B fits the 8 GB card with its experts in system RAM**, and generates about as fast as
the dense 12B without a drafter. At the serving setting (`-ncmoe 21`, q4_0 cache, `-c 131072`,
drafter at 2 tokens; 7,850 MiB):

| tokens held | code, drafter | story, drafter | no drafter |
|---:|---:|---:|---:|
| 0 | 40.6 tok/s (1.44x) | 27.9 (0.98x) | 28.2 |
| 16,384 | 35.5 (1.43x) | 26.3 (1.06x) | 24.9 |
| 32,768 | 34.1 (1.54x) | 24.0 (1.09x) | 22.2 |
| 65,536 | 28.4 (1.63x) | 19.5 (1.12x) | 17.5 |
| 98,304 | 23.6 (1.67x) | 16.6 (1.17x) | 14.1 |

Summaries of the document held gain 1.16–1.29x from 16k on (`depth/depth-table.txt`).

## The file

30 layers, 128 experts, 8 used per token. Read from the GGUF's tensors: expert weights
12,251 MiB (Q4_0, 408 MiB per layer), everything else 1,505 MiB (Q4_0 883, `token_embd` Q6_K
578, F32 44). `-ngl 99 -ncmoe N` puts every layer's attention, norms and router on the card and
keeps the experts of the first N layers in RAM, where the CPU runs them.

## Expert layers in RAM against speed (`bench-*.md`)

llama-bench, `-ngl 99 -fa 1 -t 6`, three repeats, default 512-token context:

| `-ncmoe` | 30 (`--cpu-moe`) | 24 | 20 | 18 | 17 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| tg128 tok/s | 24.35 | 27.66 | 30.65 | 32.27 | 33.31 | out of memory |
| pp512 tok/s | 264 | 313 | 365 | 394 | 408 | |

Each layer moved to the card adds about 0.5–0.6 tok/s. At 16 the 544 MiB compute buffer does
not fit. `-ub 256` loads 16 (34.17 tok/s) but reads prompts at 243 tok/s (`bench-ub256.md`).

Threads at `-ncmoe 17` (`bench-threads.md`), tg128: 4 / 5 / 6 / 8 / 12 threads = 28.49 / 31.92 /
33.03 / 33.97 / 34.65 (±2.98). Prompt reading is level (400–408). The rig serves `-t 8`.

## Context that fits (`fit.md`, `fit-draft.txt`, `fit-*.log`)

llama-server, `--parallel 1`. Full-attention KV is 20 KiB/token at f16 (the failed allocations:
2,560 MiB at 131,072, 5,120 at 262,144). The compute buffer grows with context (475 MiB at 32,768,
1,467 at 262,144).

| `-ncmoe` | f16 | q8_0 | q4_0 | q4_0 + drafter |
|---:|---:|---:|---:|---:|
| 17 | — (32,768 fails) | | | |
| 18 | — (32,768 fails) | | 98,304 | |
| 19 | 32,768 | 65,536 | 131,072 | 65,536 |
| 20 | 32,768 | 65,536 | 131,072 | 98,304 |
| 21 | 32,768 | 131,072 | 131,072 | **131,072 (7,836 MiB)** |
| 22 | | 131,072 | **262,144 (7,874 MiB)** | 131,072 |
| 23 | | 131,072 | 262,144 | 262,144 fails |
| 24 | | | | 262,144 fails |

Blank cells were not tried. Values are the largest context tried that loaded.

## Drafter (`ab-n20/`, `spec_ab.py`, `summarize.py`)

The 12B sister's method: one llama-server per arm on port 8099 at `-ncmoe 20 -c 8192`, f16
cache, `-t 8`; six prompts, greedy x3 plus one sampled pass, 256 tokens, prompt cache off. No
drafter ran first and last (29.6 and 28.7 tok/s).

| arm | greedy | x none | accepted | sampled x none | VRAM |
|---|---:|---:|---:|---:|---:|
| none | 29.6 | 1.00x | — | 1.00x | 6,608 MiB |
| exact Q4_0, 1 token | 37.0 | 1.247x | 0.760 | 1.252x | 6,916 |
| **exact Q4_0, 2 tokens** | **36.8** | **1.241x** | 0.654 | 1.219x | 6,916 |
| exact Q4_0, 3 tokens | 30.1 | 1.014x | 0.579 | 0.614x | 6,916 |
| Q8_0, 2 tokens | 35.6 | 1.203x | 0.657 | 1.171x | 7,116 |

Per prompt at 2 tokens: code 1.51x, facts 1.43x, explanation 1.17x, chat 1.16x, story and
summary 1.12x. The 12B's drafter gives 2.01x on the same card. Each drafted token the target
checks routes to its own experts, which the CPU reads, and that likely limits the gain here
(inferred, not measured). 2 tokens are served: level with 1 overall, ahead on code and lists.
Greedy replies differ from no drafter in 18 of 18 at 2 tokens, 12 of 18 at 1.

The exact Q4_0 drafter is 251,939,648 B against 461,785,920 for the converter's Q8_0; no block
was off the grid, and 96.6–97.0% of values per tensor match the bf16 source exactly
(`build-drafter.log`, `build-drafter.json`).

## By conversation length (`depth/`, `depth_sweep.py`)

The 12B sister's `kv_sweep.py` method at the serving setting, two arms (drafter / none), both at
`-ncmoe 21 -c 131072` q4_0: the first D tokens of the NVIDIA driver README, then a story, code or
summary request, greedy, 256 tokens, prompt cache on. Table above. 131,072 itself was skipped (no
room for the reply). Prompt reading of new text fell from 302 to 227 tok/s between 8k and 98k.

## Not measured

Answer quality (GSM8K, BFCL) for this target, the q4_0 cache or the drafter on the 26B; the
exact Q4_0 rebuild of the target (Google's `token_embd` is Q6_K); concurrency; flash attention
off.
