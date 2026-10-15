# 2026-09-29 — paired CPU/GPU sweep, ABBA, Gemma 4 E2B exact Q4_0 (v2)

**The 2026-09-22 paired A/B, re-run after both arms moved to the v2 exact GGUF.** Same protocol,
same binaries (by sha256), same flags except `-ngl`, same harness and prompts. The weights are the
only planned change. Identical report in both arms' run directories.

## Result

GPU decode is **4.37x** the CPU, prefill (TTFT) **3.80x**, end-to-end **3.99x**. These are
medians over 8 paired cells, each cell the mean of two passes per arm.

| in tok | out tok | CPU decode | GPU decode | **x** | CPU TTFT ms | GPU TTFT ms | **x** | CPU e2e | GPU e2e | **x** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 94 | 32 | 18.86 | 79.42 | **4.21** | 1201 | 366 | **3.28** | 11.88 | 44.54 | **3.75** |
| 94 | 128 | 18.30 | 79.13 | **4.32** | 1295 | 369 | **3.51** | 15.94 | 66.08 | **4.15** |
| 516 | 32 | 17.83 | 77.79 | **4.36** | 6188 | 1574 | **3.93** | 4.12 | 16.54 | **4.01** |
| 516 | 128 | 17.68 | 77.42 | **4.38** | 5970 | 1582 | **3.77** | 9.88 | 40.20 | **4.07** |
| 998 | 32 | 17.43 | 76.08 | **4.37** | 11940 | 3186 | **3.75** | 2.36 | 9.00 | **3.81** |
| 998 | 128 | 17.35 | 76.54 | **4.41** | 12223 | 3188 | **3.83** | 6.61 | 26.62 | **4.03** |
| 1959 | 32 | 16.52 | 75.45 | **4.57** | 24566 | 6404 | **3.84** | 1.22 | 4.72 | **3.88** |
| 1959 | 128 | 16.71 | 74.93 | **4.48** | 24602 | 6410 | **3.84** | 3.99 | 15.87 | **3.97** |

- **decode**: the client-side inter-chunk rate off the SSE stream (`--decode-source auto ->
  stream`) on both arms. Recomputed as tokens over the measured decode span, the arms read
  CPU 17.12–20.88 and GPU 76.73–87.90 tok/s, and the **median ratio is 4.37 either way**.
- **TTFT ratio** is CPU/GPU, so higher is a bigger GPU win in every column.
- **Thinking was on for every request**, as in every earlier paired run, so every cell measures
  the thinking phase. The server default has since changed to `--reasoning off` for the demo, and
  `sweep.py` now asks for thinking explicitly per request, so later runs keep measuring the same
  thing. Checked live: with the server at `--reasoning off`, a request carrying
  `"enable_thinking": true` returned 708 characters of reasoning.

## Protocol: ABBA with a temperature gate

| pass | arm | start | end | throttle events during |
| :--- | :--- | :--- | :--- | ---: |
| 1 | CPU | 46 °C pkg / 42 °C gpu | 90 / 53 | 50,918 |
| 2 | GPU | 45 / 41 | 78 / 56 | 3,209 |
| 3 | GPU | 44 / 41 | 80 / 58 | 4,932 |
| 4 | CPU | 48 / 41 | 90 / 53 | 59,190 |

Every pass waited at least 120 s, and until the package was ≤50 °C and the GPU ≤45 °C.
`driver.log` holds the timeline; `pass{1,2}/` hold each arm's raw `sweep.json`, `sweep.log` and
`metrics.prom`. `analyze.py` is 2026-09-22's with the date changed. `make_report.py` builds the
schema reports in `benchmarks/reports/`; it was checked by regenerating 2026-09-22's sweep entries
exactly from that run's passes.

## Is the difference real?

Yes. The decode ratio spans 4.21–4.57 across cells, and both arms are steadier than on 2026-09-22.

| | decode drift, pass 1 → 2 | within-cell spread (max) |
| :--- | :--- | :--- |
| GPU | median −0.6%, range −1.8% to +0.1% | 0.86% / 0.97% |
| CPU | median +0.8%, range −1.7% to +3.3% | 6.22% / 5.00% |

The one-order estimates agree within 2%: a CPU-then-GPU pair reads 4.42x decode and 3.80x prefill,
a GPU-then-CPU pair 4.34x and 3.80x.

## Against 2026-09-22: what moved, and what it can be blamed on

Binaries, flags, harness, prompts and protocol are all the same, which 2026-09-22 could not say
of its own predecessor. Two things differ: **the weights** (Google's GGUF → v2) and **the heat**
(CPU passes logged 5x the throttle events and ended at 90 °C).

| arm | decode, 09-29 / 09-22 | TTFT, 09-29 / 09-22 |
| :--- | :--- | :--- |
| GPU | **1.101** (1.098–1.106) | 0.982 (0.958–0.987) |
| CPU | 1.030 (1.019–1.082) | **1.094** (1.063–1.119): slower |

The GPU arm moved exactly as the file-level llama-bench A/B predicted (`2026-09-29-exact-gguf-v2-1650ti`:
tg128 1.11–1.12x, pp512 1.02x). Its passes barely heat the package, so this is the file.

**The CPU arm's 9% slower TTFT is the heat, not the file.** A controlled llama-bench check on the
CPU arm's binary, Google against v2, in ABBA order with the same gate
(`local-llamacpp-cpu-2b-q4_0/benchmarks/runs/2026-09-29-google-vs-v2-cpu`), found:

| | v2 / Google, mean of 4 passes | v2 ran first | Google ran first |
| :--- | ---: | ---: | ---: |
| pp512, 6 threads | 0.998 | 1.042 | 0.955 |
| pp512, 12 threads | 0.995 | 1.017 | 0.973 |
| tg128, 6 threads | **1.063** | 1.089 | 1.038 |

CPU prefill is unchanged by the file; whichever file runs second in a pass loses about 4% to heat.
CPU decode gains 6% in that check, and 3% here, where the package ran hotter.

**So the GPU's lead widened (4.14x → 4.37x decode, 3.42x → 3.80x prefill) for two reasons.** The file
helps the GPU more than the CPU (1.10x against 1.03–1.06x), and this session's heat cost the CPU
arm prefill speed that 2026-09-22's did not. Only the first is a property of the weights. Quote
this run's ratios with the throttle counts beside them.

## What was controlled

| | |
| --- | --- |
| checkpoint | same file in both arms, `gemma-4-E2B-it-q4_0-exact.gguf`, SHA-256 `419db9a6…`, attested path |
| engine | llama.cpp `f95b0d9` (b318), the 2026-09-22 binaries: CPU exe `9c88c7821fcc11a6`, GPU exe `9f2a8b0b6ed5365d` |
| flags | `-c 8192 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --parallel 1 --metrics`, differing ONLY in `-ngl` (0 vs 99); from each rig's `tpu.env`, no command-line override this time |
| prompts | device-neutral filler, `sweep.py` byte-identical in both rigs; all 8 cells paired on equal `input_len` of 94/516/998/1959 |
| prompt cache | verified defeated: `prompt_tokens_cached_total 0` of 28,553 in every pass |
| order / thermals | ABBA, with a temperature-gated cooldown before every pass |

## What was NOT controlled

- **Heat within a pass.** The gate sets the starting temperature, but both CPU passes ended at 90 °C
  with far more throttling than on 2026-09-22. The weather, the machine's position or a background
  job could all do that; none was recorded.
- **Page cache.** The GGUF was hot throughout. Nothing here measures a cold start.
- **Concurrency.** Single stream only.

## Reproduce

`paired.sh` in this directory is the driver. For each arm it runs `make serve` in the arm's rig,
waits for `/health`, then runs:

```bash
python3 sweep.py --base http://127.0.0.1:8080/v1 --out <run>/passN --rig <rig> --expect-device <cpu|gpu>
```

It then stops the server and waits out the gate. It uses the default contexts (64, 512, 1024,
2048) and outputs (32, 128), with 3 repeats. Both `tpu.env` files now carry `REASONING=off`,
which `make serve` passes as `--reasoning off`; `sweep.py` overrides it per request.
