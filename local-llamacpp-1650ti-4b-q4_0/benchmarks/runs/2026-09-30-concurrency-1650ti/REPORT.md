# 2026-09-30 — concurrency sweep, Gemma 4 E4B exact Q4_0, GTX 1650 Ti

Both shapes the E2B rig swept, on E4B: **long** 512 in / 128 out (its 2026-09-03 shape) and **short**
128 in / 512 out (its 2026-09-08 shape), concurrency 1–16, 3 repeats per level, **every level cooled
first** (package ≤50 °C, GPU ≤45 °C; `driver.log`). One server for both: `-c 16384 --parallel 16`,
1024 tokens per slot. Machine-readable: `../../reports/2026-09-30-concurrency-1650ti.json`.

**Headline: concurrency buys 1.5x on long prompts and 2.5x on short ones, and both plateau by c=8.**

## 32 slots does not fit E4B — 16 is the ceiling on this card

The E2B rig ran concurrency at `-c 32768 --parallel 32`. On E4B that fails to load (`probe/`):

| | `-c 32768 --parallel 32` | `-c 16384 --parallel 16` |
|---|---:|---:|
| full-attention KV (4 layers) | 512 MiB | 256 MiB |
| sliding-window KV (20 layers) | **1280 MiB — cudaMalloc failed** | 640 MiB |
| `nvidia-smi` | — | **3568 MiB of 4096** |

llama.cpp gives **every slot its own 1024-cell sliding-window cache** (20 layers × 2 KiB × 1024 =
40 MiB/slot), whatever `-c` is. So on this model the slot count, not the context, is what spends VRAM,
and 16 slots leaves ~100 MiB. E2B's sliding KV is 12 layers × 1 KiB — 12 MiB/slot — which is why 32 slots
cost it nothing. `--kv-unified` might change this; not tried.

## Long prompt: 512 in / 128 out

| c | aggregate tok/s | spread | TTFT ms | TPOT ms | per stream |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.09 | 0.6% | 3,133 | 26.11 | 38.30 |
| 2 | 26.38 | 0.9% | 6,176 | 28.45 | 35.15 |
| 4 | 28.98 | 0.2% | 12,240 | 43.61 | 22.93 |
| 8 | 30.82 | 7.3% | 24,453 | 70.56 | 14.17 |
| 16 | 30.39 | 0.4% | 48,903 | 148.41 | 6.74 |

**1.53x at best, flat from c=8.** TTFT doubles exactly with every doubling of `c` (3.1 → 6.2 → 12.2 →
24.5 → 48.9 s): prefill does not batch across slots, the same wall the E2B rig recorded, at half the height
(E2B plateaued at ~45–48 tok/s, 1.47x). A 16th concurrent user waits 49 s for a first token.

## Short prompt: 128 in / 512 out

| c | aggregate tok/s | spread | TTFT ms | TPOT ms | per stream |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.32 | 1.0% | 1,109 | 25.94 | 38.55 |
| 2 | 60.42 | 4.4% | 1,957 | 27.96 | 35.77 |
| 4 | 76.06 | 3.7% | 3,757 | 42.77 | 23.38 |
| 8 | **88.91** | 1.7% | 7,326 | 69.09 | 14.47 |
| 16 | 87.26 | 2.0% | 15,511 | 144.73 | 6.91 |

**2.52x at c=8, then flat.** c=2 is nearly free (1.71x aggregate, per stream 38.6 → 35.8); from c=4 the
per-stream rate falls roughly as 1/c.

**No bimodality.** The E2B rig's short-prompt sweep found c=2, 4 and 8 flipping between two discrete
decode rates (spreads 17–57%) and quoted only c=1 and c=16. Here the worst spread in either shape is 7.3%
(long c=8, one slow repeat) and every other level is under 5%. This run used a newer llama.cpp (`f95b0d9`
against `95ef7fc`) and a different model, so which of the two removed the instability is not separated.

## Against E2B — rough, not paired

E2B's concurrency runs are on the older binary with `-t 4`; these are ratios across binaries and sessions.
Long c=1: 20.09 vs 32.74 (0.61). Short c=1: 35.32 vs 63.77 (0.55). Short plateau: 88.9 at c=8 vs 164.1 at
c=16 (0.54). E4B holds roughly half E2B's throughput at every shape, as in the paired single-stream sweep.

## Method

`conc.sh`: one server, then for each shape and level a cooldown gate and
`sweep.py --concurrency c --conc-input … --conc-output … --repeats 3 --prompt-mode unique`, one output
directory per level (`long/c*/`, `short/c*/`, each with `concurrency.json`, `metrics.prom`, `sweep.log`).
Prompt cache: 0 of 65,830 prompt tokens cached (`metrics-final.prom`). Thinking on per request. Arm
attested every level (`gpu`, sha256 `9f2a8b0b…`, the E4B exact file). The E2B demo server was stopped
before and restored after, with its original argv.
