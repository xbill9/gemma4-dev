# 2026-09-22 — paired CPU/GPU sweep, ABBA, Gemma 4 E2B q4_0

**The re-run of the 2026-09-16 paired A/B on the rebuilt binaries**, and the first
one here that controls for order and thermal state instead of listing them as
uncontrolled. One GGUF, one llama.cpp commit (`f95b0d9`, built under Debian sid,
gcc 16.2, CUDA 13.4), one port, one harness, one prompt set, **one set of flags
except `-ngl`**. Identical report in both arms' run directories.

## Result

GPU decode is **4.14x** the CPU, prefill (TTFT) **3.42x**, end-to-end
**3.62x** — medians over 8 paired cells, each cell the mean of two passes per arm.

| in tok | out tok | CPU decode | GPU decode | **x** | CPU TTFT ms | GPU TTFT ms | **x** | CPU e2e | GPU e2e | **x** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 94 | 32 | 18.31 | 71.89 | **3.93** | 1084 | 381 | **2.84** | 12.24 | 41.51 | **3.39** |
| 94 | 128 | 17.96 | 71.57 | **3.98** | 1171 | 385 | **3.04** | 15.85 | 60.43 | **3.81** |
| 516 | 32 | 17.06 | 70.61 | **4.14** | 5528 | 1608 | **3.44** | 4.47 | 15.97 | **3.57** |
| 516 | 128 | 17.19 | 70.35 | **4.09** | 5527 | 1610 | **3.43** | 10.00 | 37.94 | **3.80** |
| 998 | 32 | 16.10 | 69.25 | **4.30** | 11071 | 3242 | **3.41** | 2.51 | 8.78 | **3.50** |
| 998 | 128 | 16.77 | 69.38 | **4.14** | 11057 | 3246 | **3.41** | 6.94 | 25.43 | **3.66** |
| 1959 | 32 | 16.14 | 68.52 | **4.25** | 23118 | 6497 | **3.56** | 1.29 | 4.63 | **3.60** |
| 1959 | 128 | 16.37 | 68.22 | **4.17** | 22718 | 6496 | **3.50** | 4.22 | 15.40 | **3.65** |

- **decode** — client-side inter-chunk rate off the SSE stream, `--decode-source
  auto -> stream` on both arms. Chunks undercount tokens here (see the rig
  `CLAUDE.md`); recomputed as tokens over the measured decode span the arms read
  CPU 16.77-20.27 / GPU 69.87-79.58 tok/s, and the **median ratio is 4.14
  either way** — the undercount is the same engine on both sides and cancels.
- **TTFT ratio** is CPU/GPU, so higher is a bigger GPU win in every column.

## Protocol: ABBA with a temperature gate

| pass | arm | start | end | throttle events during |
| :--- | :--- | :--- | :--- | ---: |
| 1 | CPU | 38 °C pkg / 36 °C gpu | 85 / 52 | 9,604 |
| 2 | GPU | 45 / 40 | 79 / 56 | 420 |
| 3 | GPU | 42 / 39 | 78 / 55 | 2,879 |
| 4 | CPU | 41 / 39 | 82 / 53 | 19,279 |

Every pass waited at least 120 s and until the package was ≤50 °C and the GPU
≤45 °C. Order CPU, GPU, GPU, CPU means each arm has one early and one late pass,
so a drift that is linear in time cancels in the pair. `driver.log` holds the
timeline; `pass{1,2}/` hold each arm's raw `sweep.json`, `sweep.log` and
`metrics.prom`.

## Is the difference real?

Yes, by an order of magnitude. The decode ratio spans 3.93-4.30 across cells.
Pass-to-pass drift:

| | decode drift, pass 1 → 2 | worst cell |
| :--- | :--- | :--- |
| GPU | median **+0.6%**, range 0.0 to +1.3% | — |
| CPU | median **-1.1%**, range -9.3% to +0.1% | 998/32: 16.89 → 15.32 |

Within-cell spread over 3 repeats: GPU ≤1.14% on both passes; CPU ≤2.69% on pass
1 but up to **14.05%** on pass 2, the pass with twice the throttle events. CPU
TTFT also rose 3-7% on pass 2 at the two long contexts (e.g. 1959/32: 22289 →
23948 ms), while GPU TTFT moved ≤0.3%.

**What the order effect is worth, now measured instead of asserted:** a single
CPU-then-GPU pair (the 2026-09-16 design) reads 4.09x decode / 3.38x prefill; the
GPU-then-CPU pair reads 4.17x / 3.47x. So a one-order run misstates the ratio by
~1-2%. That is real but small — the 2026-09-16 warning that order could swing it
was right in direction and much larger than the effect turned out to be.

**The CPU arm is the noisy one, and its noise is thermal.** The GPU arm barely
heats the package (420 throttle events in a pass), the CPU arm saturates it.

## What was controlled

| | |
| --- | --- |
| checkpoint | same file, `gemma-4-E2B_q4_0-it.gguf`, byte-identical path |
| engine | llama.cpp `f95b0d9` (b318), both arms rebuilt from clean on 2026-09-22 |
| flags | `-c 8192 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --parallel 1 --metrics`, differing ONLY in `-ngl` (0 vs 99) |
| prompts | device-neutral filler, `sweep.py` byte-identical in both rigs; all 8 cells paired on equal `input_len` — 94/516/998/1959 |
| prompt cache | verified defeated: `prompt_tokens_cached_total 0` of 28,553 in every pass |
| order / thermals | ABBA, temperature-gated cooldown before every pass |
| arm identity | attested from `/proc` on every pass: CPU exe `9c88c7821fcc11a6`, GPU exe `9f2a8b0b6ed5365d` |

**`-t 6 -tb 12` on the GPU arm is an override.** This rig's `tpu.env` still says
`THREADS=4`, `THREADS_BATCH=8`; the CPU arm's were re-derived to 6/12 from the
real topology on 2026-09-22, and the control requires the flags to match. GPU
decode was measured insensitive to threads on 2026-09-03 (73.75 / 73.08 / 72.97
at 4/6/8).

## What was NOT controlled

- **Page cache.** The GGUF was hot throughout. Nothing here measures cold start.
- **Concurrency.** Single stream only.
- **Throttling within a pass.** The gate controls the starting state, not what
  happens during an 8-minute CPU pass. ABBA cancels its linear part; the 14%
  within-cell spread on CPU pass 2 is the non-linear part showing through.

## Against 2026-09-16 — not a comparison, a caution

The 2026-09-16 run read 4.27x decode / 3.63x prefill; this one reads 4.14x /
3.42x. **Do not attribute that to anything.** Three things changed at once: the
llama.cpp commit (`c6824a9` → `f95b0d9`), the thread flags (`-t 4 -tb 8` → `-t 6
-tb 12`), and the protocol. The direction is at least consistent with the CPU
arm's own spot check — more threads help CPU prefill (its TTFT fell 5-8% here) —
but that is a hypothesis, not a finding of this run.

## Reproduce

`paired.sh` (the driver) and `analyze.py` (the pairing) are in this directory. For each arm it runs, `make serve` in the arm's rig
(the GPU arm with `THREADS=6 THREADS_BATCH=12` on the command line), wait for
`/health`, then

```bash
python3 sweep.py --base http://127.0.0.1:8080/v1 --out <run>/passN --rig <rig> --expect-device <cpu|gpu>
```

then stop the server and wait out the gate. Default contexts (64, 512, 1024,
2048), outputs (32, 128), 3 repeats.
