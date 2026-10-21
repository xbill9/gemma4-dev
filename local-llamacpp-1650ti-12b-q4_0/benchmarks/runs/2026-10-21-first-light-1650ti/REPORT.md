# 2026-10-21 — first light: Gemma 4 12B QAT on the GTX 1650 Ti Max-Q (4 GB)

llama.cpp `7182ef71e` (CUDA, sm_75, `GGML_NATIVE=ON`, FORCE_MMQ off), host `yogagpu` (Lenovo Yoga 9
15IMH5: i7-10750H, 6 cores / 12 threads, 15 GiB RAM; GTX 1650 Ti with Max-Q Design, TU117, 4096 MiB,
driver 615.71.09). Target: `xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf` at `85e797ef`, SHA-256
`038337ee…` (checked by `make download`). Drafter: `xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf`
at `5fc2e287`, `f22f6366…` (`make draft`). Google's `google/gemma-4-12B-it-qat-q4_0-gguf` file for the
comparison was already on the host. The same files and llama.cpp commit as
`local-llamacpp-3050oem-12b-q4_0`; a different card, CPU and host.

## Result

**The 12B runs on the 4 GB card, too slowly for interactive chat.** About half the 48 layers fit on
the card; the rest run on the CPU and set the speed. At the serving setting (20 layers on the card,
`-c 32768`, q4_0 cache, drafter at 3 tokens) a short question generates at 7.4–10.1 tok/s and a
128-token reply takes 15–19 s. Prompt reading is about 60 tok/s, so a 4,096-token document takes
100 s before the first word and a 16,384-token one 5 minutes. For scale, the same files on the
RTX 3050 OEM (all layers on the card) generate at 56.4 tok/s with the drafter and read prompts at
987 tok/s (the 3050 sister's `CLAUDE.md`, a different host).

## Layers on the card against speed (`bench.out`, `bench-*.md`)

llama-bench `-fa 1 -t 6`, two repeats, pp512 / tg64, 11:41–11:55, each value started with the GPU at
or below 50 C:

| `-ngl` | 0 | 12 | 16 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tg64, `-ub 512` | 3.63 | 4.72 | 5.06 | 5.20 | 5.31 | 5.43 | 5.64 | | | | |
| pp512, `-ub 512` | 58.1 | 59.6 | 60.1 | 60.4 | 60.5 | 60.7 | 60.8 | | | | |
| tg64, `-ub 128` | | | | | | 5.43 | 5.49 | 5.64 | 5.80 | 5.98 | out of memory |
| pp512, `-ub 128` | | | | | | 50.5 | 51.2 | 51.5 | 51.8 | 52.2 | |

Each layer on the card adds about 0.1 tok/s of generation, and 21 layers are 1.55x none. Prompt
reading does not move with the layers on the card (58–61 tok/s), and a smaller `-ub` costs it 15%.

**Exact rebuild against Google's file**, same session: tg64 5.06 / 5.09 at 16 layers, 5.31 / 5.34 at
19, 5.64 / 5.69 at 22 (`-ub 128`), level. The rebuild is 6.24 GiB against 6.48, which is what it
buys here: more layers on the card. At `-ub 512` the rebuild loaded 21 layers where Google's file
failed at 20 in a run just before this session (19 loaded); at `-ub 128` the rebuild loads 24, and
Google's file was not tried above 22.

## Prompt batch size (`bench2.out`, `bench-ub*.md`)

pp2048, `-b 2048 -t 6`, at the most layers that loaded: `-ub 512` 56.6 tok/s at 18 layers, `-ub 1024`
59.4 at 14, `-ub 2048` 60.5 at 6. A larger `-ub` does not read prompts faster on this card and costs
layers; the rig keeps 512. Two of the failed loads aborted with a core dump instead of a clean
out-of-memory (`-ub 1024` at 16 layers, `-ub 2048` at 8).

## Threads (`bench-threads.out`)

tg64 at 21 layers, one llama-bench per value, each started with the CPU package at or below 55 C, in
the mirrored order 6 4 8 12 12 8 4 6:

| `-t` | 4 | 6 | 8 | 12 |
|---|---:|---:|---:|---:|
| tok/s, two passes | 4.90 / 4.88 | 5.03 / 4.66 | 4.91 / 4.97 | 4.05 / 3.94 |

4–8 level, 12 slower; the rig keeps 6. These ran after 12:00, when a browser tab was using about
40% of a core; the same setting gave 5.64 tok/s at 11:47 before it started, so absolute speed on this
laptop moves with whatever else it is doing. The settings compared ran under the same conditions.

## Context that fits (`fit-all.out`, `fit-*.log`)

llama-server, `--parallel 1`, `-ub 512`: the most layers on the card that load.

| context | q4_0 + drafter | q4_0 | f16 + drafter | f16 |
|---|---:|---:|---:|---:|
| 8,192 | 22 | 24 | 20 | 23 |
| 32,768 | **20** | 23 | 19 | 21 |
| 131,072 | 14 | 19 | | |

The cache for layers on the CPU stays in system RAM, so a longer context costs only a few layers and,
at about 0.1 tok/s each, little generation speed. Two loads aborted with a core dump instead of a
clean out-of-memory (24 layers at 32,768, with and without the drafter).

## Drafter (`ab-rounds-c32768/`, `spec_ab_rounds.py`, `summarize.py`)

One llama-server per arm on port 8099, `-ngl 20 -c 32768`, q4_0 cache, `-t 6`; six prompts, greedy,
128 tokens, prompt cache off; **three rounds, the arm order rotated each round**, cell = median of the
three (`summary.txt`'s header says "one run": it is the median of three rounds).

| | none | 1 token | 2 tokens | 3 tokens | none at 23 layers |
|---|---:|---:|---:|---:|---:|
| tok/s (geomean of prompts) | 5.4 | 8.1 | 6.2 | **8.5** | 5.9 |
| x none | 1.00x | 1.50x | 1.15x | **1.56x** | 1.09x |
| accepted | — | 0.756 | 0.655 | 0.524 | — |

By round, mean of six prompts: none 5.31 / 5.43 / 5.56; 1 token 7.68 / 8.26 / 8.16; 3 tokens 8.76 /
8.37 / 8.44. 3 tokens lead 1 in all three rounds; code and explain gain most (1.97x and 1.68x at 3),
story and summary least (1.35x, 1.38x). 2 tokens fall below both 1 and 3 on every prompt, as on the
3050's partly-offloaded 31B; this run does not explain it. The drafter is worth far more than the
three layers it costs (1.56x against 1.09x). Greedy replies differ from no drafter in 18 of 18 at
every draft length, and no drafter at 23 layers differs from no drafter at 20 in 18 of 18: moving
layers between card and CPU changes greedy text by itself.

A single pass first (`ab-c32768/`, arms in a fixed order) drifted 1.14x between its first and last
no-drafter arms while the GPU clock moved from 1035 to 1380–1590 MHz; it is kept as the record of
why the rounds design was needed and its numbers are not used.

## With a long conversation held (`depth.out`, `depth.jsonl`, `depth.py`)

The 3050 sister's method (same document and questions, greedy, prompt cache on), 128 tokens. The
first question at each depth reads the document; the next two reuse it from the cache.

| tokens held | drafter at 20 layers: story / code / summary | no drafter at 23 layers | reading the document |
|---:|---|---|---|
| 0 | 7.37 / 10.12 / — | 5.63 / 5.64 / — | |
| 4,096 | 5.77 / 9.27 / 6.98 | 4.91 / 4.99 / 4.97 | 53 tok/s, 100 s |
| 16,384 | 4.11 / 6.54 / 4.43 | 4.32 / 4.30 / 4.31 | 46 tok/s, 297 s |
| 30,000 | 3.15 / 5.03 / 3.63 | 3.64 / 3.61 / 3.62 | 38 tok/s, 397 s |

From 0 to 30,000 tokens generation slows by 35% without the drafter and by 50–57% with it. The drafter's gain shrinks with the
conversation: from 16,384 tokens it is behind on story, level on summary and ahead only on code.
Prompt reading falls from 60 to 38 tok/s as the document grows.

## Not measured

Flash attention off; `THREADS_BATCH`; concurrency; f16 or q8_0 cache speed; the Q8_0 drafter; a
CPU-only build of the same commit (`-ngl 0` on the CUDA build still reads prompts on the card);
E2B/E4B on this commit. Answer quality: `../2026-10-21-accuracy-12b-1650ti`.
