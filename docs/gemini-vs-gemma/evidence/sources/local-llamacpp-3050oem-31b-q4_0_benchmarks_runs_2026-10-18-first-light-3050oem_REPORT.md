# 2026-10-18 — first light: Gemma 4 31B QAT on the RTX 3050 OEM (8 GB)

llama.cpp `7182ef71e` (CUDA, sm_86), host `gpu` (i7-5820K, 6 cores / 12 threads, 46 GiB RAM).
Target: Google's `google/gemma-4-31B-it-qat-q4_0-gguf` at `59dde245`, unmodified,
17,651,001,568 B (SHA-256 in `sha256.txt`, matches the Hub's LFS oid). Drafter: Google's
`google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` at `96d4c8c`, stored as exact Q4_0 with
the 12B sister's `gguf_exact.py`, published as `xbill9/gemma-4-31B-it-qat-assistant-q4_0-exact-gguf`
at `f1c0932a`.

## Result

**The 31B runs on the 8 GB card with two thirds of its layers on the CPU, at about 4 tok/s with
the drafter.** It serves `-ngl 20` (20 of 60 layers on the card), q4_0 cache, `-c 32768`, drafter
at 1 token: 7,734 MiB, and the check reply through `make serve` generated at 4.14 tok/s
(`serve-chat.json`, 27 of 36 drafted tokens accepted).

## The file (`gguf-layout.txt`)

Dense: 60 layers of 257–287 MiB, 15,716 MiB in all, Q4_0; `token_embd` 1,102 MiB, Q6_K. Every
token reads every layer, so the layers on the CPU set the speed.

## Layers on the card against speed (`bench-ngl*.md`)

llama-bench `-fa 1 -t 6`, two repeats, pp512 / tg64:

| `-ngl` | 0 | 12 | 16 | 18 | 20 | 22 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tg64 tok/s | 1.68 | 2.13 | 2.30 | 2.39 | 2.52 | 2.62 | out of memory |
| pp512 tok/s | 116 | 133 | 141 | 145 | 149 | 154 | |

Each layer on the card adds about 0.05 tok/s. Threads at `-ngl 22` (`bench-threads.md`): 4 / 6 /
8 / 12 = 2.09 / 2.62 / 2.66 / 2.63 tok/s; the rig serves `-t 8`.

## Prompt reading and `-ub` (`bench-ub*.md`)

pp2048, `-b 2048 -t 8`: `-ub 512` at `-ngl 20` 145 tok/s (137 at `-ngl 16`); `-ub 1024` at
`-ngl 16` 195 (fails at `-ngl 20`); `-ub 2048` at `-ngl 12` 238. A bigger `-ub` reads prompts up
to 1.64x faster, and its compute buffer costs layers on the card, so generation drops from 2.52
to 2.13 tok/s. The rig keeps the default 512.

## Drafter (`ab-ngl20/`, `spec_ab.py`, `summarize.py`)

The 26B sister's method cut to fit a 2.5 tok/s target: one llama-server per arm on port 8099 at
`-ngl 20 -c 8192`, f16 cache, `-t 8`; six prompts, greedy, **one run each**, 128 tokens, prompt
cache off. No drafter ran first and last (2.5 tok/s both). `summarize.py`'s header says
"median of 3"; here each cell is one run.

| drafted tokens | none | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| tok/s | 2.5 | **4.0** | 3.1 | 3.5 | 2.7 |
| x none | 1.00x | **1.585x** | 1.211x | 1.373x | 1.085x |
| accepted | — | 0.757 | 0.655 | 0.558 | 0.485 |

1 token gains 1.47–1.71x on every prompt (facts and code most, summary least). Checking several
drafted tokens at once costs the CPU-side layers more than checking one, so longer drafts lose;
2 tokens fall below 3 on every prompt, which this run does not explain. VRAM 7,020 MiB without
the drafter, 7,382 with. Greedy replies differ from no drafter in 4 of 6 at 1 token.

The exact Q4_0 drafter is 279,956,288 B against 514,706,240 for the converter's Q8_0; no block
was off the grid, and 96.6–97.1% of values per tensor match the bf16 source exactly
(`build-drafter.log`, `build-drafter.json`).

## Context that fits (`fit-draft.txt`, `fit.txt`, `fit-*.log`)

llama-server, `--parallel 1`, `-ub 512`; values are the largest context tried that loaded.

| `-ngl` | drafter | f16 | q4_0 |
|---:|---|---:|---:|
| 20 | yes | 16,384 (7,694 MiB) | **32,768 (7,734 MiB)**; 65,536 fails |
| 18 | yes | | 65,536 fails |
| 16 | yes | | 65,536 (7,448 MiB) |
| 12 | yes | | 131,072 (7,810 MiB) |
| 20 | no | | 65,536 (7,752 MiB) |
| 16 | no | | 131,072 (7,682 MiB) |
| 10 | no | | 262,144 (7,802 MiB) |

q8_0 at `-ngl 20` with the drafter fails at 32,768.

## Not measured

Generation with a long conversation held; answer quality (GSM8K, BFCL) for the target, the q4_0
cache or the drafter; the Q8_0 drafter against the exact one on this target; an exact Q4_0
rebuild of the target; concurrency.
