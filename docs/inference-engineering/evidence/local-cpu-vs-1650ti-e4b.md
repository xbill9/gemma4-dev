# 2026-09-30 — paired CPU/GPU sweep, ABBA, Gemma 4 E4B exact Q4_0

**The E2B pair's 2026-09-29 protocol, run on E4B.** The two arms are `local-llamacpp-cpu-4b-q4_0` and
`local-llamacpp-1650ti-4b-q4_0`, both forked that day from the E2B pair with slot 4 the only change.
Same binaries (by sha256), flags, harness (`sweep.py`, byte-identical in both arms), prompts and ABBA
protocol as `2026-09-29-paired-sweep`. The checkpoint is the only planned change. The report is identical
in both arms' run directories; the driver, `analyze.py` and `make_report.py` live in the GPU arm's.

## Result

GPU decode is **4.20x** the CPU, prefill (TTFT) **3.84x**, end-to-end **3.96x**: medians over 8 paired
cells, each cell the mean of two passes per arm. 32/32 cells ok.

| in tok | out tok | CPU decode | GPU decode | **x** | CPU TTFT ms | GPU TTFT ms | **x** | CPU e2e | GPU e2e | **x** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 94 | 32 | 9.50 | 39.56 | **4.17** | 2540 | 731 | **3.48** | 5.85 | 22.24 | **3.80** |
| 94 | 128 | 9.31 | 39.05 | **4.19** | 2644 | 737 | **3.58** | 8.02 | 32.72 | **4.08** |
| 516 | 32 | 9.02 | 37.91 | **4.20** | 12393 | 3159 | **3.92** | 2.08 | 8.21 | **3.96** |
| 516 | 128 | 9.02 | 37.79 | **4.19** | 12297 | 3170 | **3.88** | 4.93 | 19.84 | **4.03** |
| 998 | 32 | 8.84 | 37.16 | **4.20** | 23923 | 6298 | **3.80** | 1.18 | 4.54 | **3.83** |
| 998 | 128 | 8.76 | 37.34 | **4.26** | 23983 | 6296 | **3.81** | 3.36 | 13.31 | **3.96** |
| 1959 | 32 | 8.48 | 37.01 | **4.36** | 48478 | 12339 | **3.93** | 0.62 | 2.44 | **3.95** |
| 1959 | 128 | 8.29 | 36.96 | **4.46** | 50129 | 12341 | **4.06** | 1.97 | 8.15 | **4.13** |

- **decode**: client-side inter-chunk rate off the SSE stream (`--decode-source auto -> stream`) on both
  arms. Recomputed as tokens over the measured decode span, the arms read CPU 8.49–10.51 and GPU
  37.86–43.79 tok/s, and the median ratio is 4.20 either way.
- **TTFT ratio** is CPU/GPU, so higher is a bigger GPU win.
- **Thinking was on for every request** (`sweep.py` asks for it; the servers default to off). Every cell
  measures the thinking phase, as in every E2B paired run.
- **Prompt cache defeated**: 0 of 28,553 prompt tokens cached in every pass (`pass*/metrics.prom`).
- **Attested** from `/proc` on every pass: CPU arm `build-cpu` sha256 `9c88c782…`, `-ngl 0`; GPU arm
  `build` sha256 `9f2a8b0b…`, `-ngl 99`; both `-m` the E4B exact file.

## Protocol: ABBA with a temperature gate

| pass | arm | start | end | throttle events during | duration |
| :--- | :--- | :--- | :--- | ---: | ---: |
| 1 | CPU | 40 °C pkg / 38 °C gpu | 85 / 52 | 115,310 | 15m23 |
| 2 | GPU | 44 / 40 | 74 / 57 | 6,373 | 4m05 |
| 3 | GPU | 42 / 40 | 77 / 58 | 6,980 | 4m06 |
| 4 | CPU | 48 / 42 | 88 / 52 | 135,858 | 17m31 |

Every pass waited at least 120 s and until the package was ≤50 °C and the GPU ≤45 °C (`driver.log`).
The CPU passes run twice as long as E2B's and log about twice the throttle events.

## Is the difference real?

Yes. The decode ratio spans 4.17–4.46 across cells. The GPU arm is very steady; the CPU arm is not.

| | decode drift, pass 1 → 2 | within-cell spread (max) |
| :--- | :--- | :--- |
| GPU | median −0.2%, range −0.7% to +0.5% | 2.62% / 0.51% |
| CPU | **median −8.1%, range −9.8% to −6.5%** | 3.22% / 12.46% |

**The CPU arm lost 8% between its passes**, and its TTFT rose 14–23%: the second CPU pass started 8 °C
warmer and throttled more. This is what ABBA is for. The one-order estimates bracket the paired figure:
CPU-then-GPU reads 4.05x decode and 3.58x prefill, GPU-then-CPU 4.40x and 4.09x — a single-order run
would have misstated the decode ratio by 4–5%, several times the E2B pair's ~1% order effect.

## Against the E2B pair (2026-09-29-paired-sweep)

Same binaries, flags, harness, prompts and protocol; the checkpoint differs. The runs are one day apart,
not interleaved, so these ratios carry the two sessions' thermal difference.

| arm | E4B / E2B decode | E4B / E2B prefill speed (TTFT inverse) |
| :--- | :--- | :--- |
| GPU | **0.489** (0.487–0.498) | 0.503 (0.498–0.519) |
| CPU | **0.507** (0.496–0.513) | 0.495 (0.473–0.510) |

| | E2B pair | E4B pair |
| :--- | ---: | ---: |
| decode x | 4.37 | 4.20 |
| prefill x | 3.80 | 3.84 |
| end-to-end x | 3.99 | 3.96 |

**E4B costs half the speed on both devices**, and the GPU's advantage over the CPU is essentially the
same as for E2B. Decode's ratio is 4% lower on E4B; that is inside what the CPU arm's 8% pass-to-pass
drift can move, and the E2B session ran hotter on the CPU, so it is not claimed as a model effect.

The ~0.5x fits the bytes each token reads: the resident part of the file (everything but the lazy
per-layer table) is 2.61 GB for E4B against 1.30 GB for E2B, a factor of 2.0. Memory traffic was not
measured directly.

## Files

`paired.sh` (driver; it expects port 8080 free — the E2B server was stopped by hand first — and restores
the E2B server at the end), `driver.log`, `driver.out`, `pass{1,2}/` per arm (`sweep.json`, `sweep.log`, `metrics.prom`),
`pass{1,2}-server.log`, `analyze.py` (2026-09-29's with the paths changed), `make_report.py` (unchanged)
→ `benchmarks/reports/2026-09-30-paired-sweep-{1650ti,cpu}.json` in each arm.
