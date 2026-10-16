# 2026-10-13 — concurrency sweep, RTX 3050 OEM

`local-llamacpp-3050oem-2b-q4_0`, exact Q4_0 E2B GGUF (v2, `419db9a6…`), llama.cpp `77c4540ce`.
`sweep.py` concurrency mode at two shapes, levels 1–64, 5 repeats per level (median reported,
min–max and spread recorded), unique prompts, `--decode-source auto` → `stream`,
`--expect-device gpu` (attested `device=gpu`, `-ngl 99`, CUDA backend mapped).

## Server

One server for both shapes (`run.sh`):

```
llama-server -ngl 99 -c 65536 --parallel 64 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --reasoning off --metrics
```

64 slots × 1024 tokens (`n_ctx_slot = 1024` in `llama-server.log`); both shapes fit a slot. The
parent's sweeps used the same 1024 tokens per slot. **VRAM at 64 slots: 2612 MiB of 8192**
(`vram.txt`; 2637 MiB peak in `gpu.csv`).

## 512 in / 128 out

| c | aggregate tok/s | vs c=1 | spread | TTFT ms | TPOT ms | per stream |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 93.28 | 1.00x | 0.4% | 213 | 9.34 | 107.03 |
| 2 | 156.85 | 1.68x | 0.9% | 357 | 10.27 | 97.35 |
| 4 | 196.09 | 2.10x | 1.4% | 669 | 15.64 | 63.95 |
| 8 | 224.18 | 2.40x | **21.5%** | 1373 | 25.64 | 38.99 |
| 16 | 316.07 | 3.39x | **25.6%** | 2702 | 30.12 | 33.20 |
| 32 | 344.98 | 3.70x | **14.7%** | 5348 | 51.71 | 19.34 |
| 64 | 358.05 | 3.84x | 0.9% | 8516 | 113.20 | 8.84 |

## 128 in / 512 out

| c | aggregate tok/s | vs c=1 | spread | TTFT ms | TPOT ms | per stream |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 104.74 | 1.00x | 0.2% | 135 | 9.34 | 107.02 |
| 2 | 189.18 | 1.81x | 1.7% | 194 | 10.23 | 97.77 |
| 4 | 259.73 | 2.48x | 0.7% | 330 | 14.74 | 67.83 |
| 8 | 310.25 | 2.96x | 4.3% | 607 | 24.56 | 40.71 |
| 16 | 505.42 | 4.83x | 0.7% | 1114 | 29.32 | 34.10 |
| 32 | 586.41 | 5.60x | 0.8% | 2127 | 49.98 | 20.01 |
| 64 | 618.12 | 5.90x | 1.1% | 3890 | 95.33 | 10.49 |

Generations stopped on their own before the 512 cap: 440 tokens per request at c=1, 28,130 over
64 requests at c=64. Full numbers in `summary.txt`; raw per-level JSON in `in*/concurrency.json`.

## Findings

1. **Concurrency pays on this card at both shapes.** Aggregate output reaches 3.84x c=1 at
   512/128 and 5.90x at 128/512, both at c=64. Per-stream speed falls from 107 tok/s to 9–10.
   Past c=16 the gains flatten: c=16 → c=64 adds 13% at 512/128 and 22% at 128/512 while
   per-stream speed falls 3.3–3.8x.
2. **Prefill still runs one request at a time.** TTFT roughly doubles with each doubling of c
   (ratios 1.59–2.05 at 512/128, 1.44–1.91 at 128/512), the same pattern the parent measured.
   This card prefills fast enough (3,860 t/s in llama-bench) that decode batching shows through
   anyway.
3. **512/128 has wide spreads at c=8, 16 and 32** (21.5%, 25.6%, 14.7%), and in each the median
   sits near the maximum (226.6, 318.1, 348.9), so most repeats are fast and one or more slow.
   128/512 stays under 5% everywhere. The parent saw two discrete speeds in its mid-range levels
   on the 128/512 shape. Here the per-repeat values are not stored, only min, max and median, so
   whether this is the same two-state behaviour is not settled. Quote c=1, c=4 and c=64 at
   512/128 with confidence; treat c=8–32 as medians with a known low tail.
4. **No throttling.** 822 one-second samples: SM clock 1890 MHz in every sample above 50%
   utilisation, 68 °C peak, 110.9 W peak of the 120 W limit.
5. **The prompt cache answered nothing:** `llamacpp:prompt_tokens_cached_total` moved by 0 at
   both shapes, against 328,193 and 97,961 prompt tokens.

## Against the parent

Same file, same tokens per slot, same harness; different host, llama.cpp commit and day, so
indicative only. Parent figures from `local-llamacpp-1650ti-2b-q4_0` runs
`2026-09-03-full-sweep-1650ti` (512/128) and `2026-09-08-conc-shortprompt-1650ti` (128/512).

| | 1650 Ti | 3050 OEM | ratio |
| --- | ---: | ---: | ---: |
| 512/128, c=1 | 32.74 | 93.28 | 2.85x |
| 512/128, c=32 | 48.19 | 344.98 | 7.16x |
| 128/512, c=1 | 63.77 | 104.74 | 1.64x |
| 128/512, c=16 | 164.07 | 505.42 | 3.08x |

On the 1650 Ti, 512/128 output topped out at 1.47x c=1 because serial prefill filled the time.
On this card the same shape scales 3.70x by c=32.

## Not measured

c above 64, longer contexts per slot, and a context sweep (`sweep.py` without `--concurrency`).
