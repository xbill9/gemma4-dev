# 2026-10-16 — first light, 12B on the RTX 3050 OEM

`local-llamacpp-3050oem-12b-q4_0`: the exact Q4_0 12B GGUF (`gemma-4-12B-it-q4_0-exact.gguf`,
SHA-256 `038337ee…`, built in `../2026-10-16-exact-gguf-12b-3050oem`) against Google's
`gemma-4-12b-it-qat-q4_0.gguf` (SHA-256 `93567e57…`), llama.cpp `7182ef71e`, on the card and host
of the E2B and E4B sister rigs. Scripts: `bench.sh` (raw output `results.md`), `fit.sh` (`fit.md`);
allocation logs `alloc-*.log`. Every server: `-ngl 99 -fa 1 -t 6 -tb 12 --parallel 1`.

## Question: does a QAT 12B fit an 8 GB card?

**Yes, every layer on the card, at up to 40,960 tokens of context.** Google's file fits too, at up
to 28,672.

| | Google's GGUF | exact rebuild |
| --- | ---: | ---: |
| file | 6,975,879,296 B | 6,716,356,736 B |
| CUDA0 model buffer | 6637.69 MiB | **6390.19 MiB** |
| `token_embd` input copy, read from the mmap | 787.50 MiB (Q6_K) | 540.00 MiB (Q4_0) |
| largest `-c` that loads, f16 KV | 28,672 (7,814 MiB); 32,768 fails | **40,960** (7,772 MiB); 49,152 fails |
| largest `-c` that loads, q8_0 KV | — | 65,536 (7,550 MiB); 98,304 fails |
| with the exact Q4_0 drafter, f16 KV | — | 24,576 (7,820 MiB); 28,672 fails |
| with the exact Q4_0 drafter, q8_0 KV | — | 49,152 (7,790 MiB) |

The 12B has no per-layer table, so unlike E2B and E4B nothing large stays in the mmap: the card
holds the whole model. The output layer is the tied `token_embd`, so it is on the card as well as
read from the mmap for input, and the rebuild's Q4_0 copy saves 247.5 MiB of VRAM there.

**Context cache, from the allocation logs:** 16 KiB per token for the 8 full-attention layers
(1 KV head of width 512, K and V in f16: 512 MiB at 32,768) plus a fixed 480 MiB for the 40
sliding-window layers, which hold 1,536 cells (the 1,024 window plus a 512-token batch). q8_0
halves both (255 MiB sliding). The compute buffer is 130–160 MiB at f16 and 257 MiB at q8_0.

## Speed: exact vs Google, ABBA (llama-bench r=5, cooled to ≤40 °C)

| pass | file | pp512 t/s | tg128 tok/s |
| --- | --- | ---: | ---: |
| 1 | Google | 981.57 | 27.68 |
| 2 | exact | 987.49 | 28.64 |
| 3 | exact | 986.97 | 28.64 |
| 4 | Google | 977.83 | 27.68 |

**Generation 1.035x Google's file** (28.64 against 27.68 in both passes of each), prefill level
(1.006x). Smaller than E4B's 1.06x and E2B's 1.11x on the 1650 Ti: the tied output layer is a
smaller share of what a 12B reads per token.

## Speed by conversation length (exact, r=3)

| tokens already in context | prompt t/s | generate tok/s |
| ---: | ---: | ---: |
| 0 | 981.09 | 28.63 |
| 8,192 | 801.58 | 26.81 |
| 16,384 | 692.32 | 26.33 |

The 32,768 depth did not run: llama-bench sizes its context to depth plus test and stopped there.

## Against E4B on the same binary

E4B exact, same card and build, pass 5: pp512 2,240.65, tg128 61.19 (61.2 against the E4B rig's
60.5 on `77c4540ce`). **The 12B generates at 0.47x E4B and reads prompts at 0.44x.**

## Served

`make serve` with the rig's `tpu.env` (drafter on, `-c 24576`): healthy, **7,820 MiB**. Three
greedy chat replies in `serve-chat.jsonl`: a correct one-sentence TPU definition, `391` for
17 × 23, and a haiku; 51–68 tok/s with the drafter, 0.56–1.00 of drafted tokens accepted.
