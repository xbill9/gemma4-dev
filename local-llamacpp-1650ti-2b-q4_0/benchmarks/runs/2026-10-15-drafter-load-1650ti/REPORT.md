# 2026-10-15 — the drafter under load and answer quality as the rigs serve, GTX 1650 Ti

**In progress.** Committed partway through; part B and the `-kvu` rerun of part A are still running
(`driver.log`). This report will be completed when they finish.

## Setup

The laptop `yogagpu` (GTX 1650 Ti Max-Q, 4096 MiB, driver 615.71.09), llama.cpp `77c4540ce`
(build 813, `host.txt`) from `~/llama.cpp-77c4540ce`. Drafter: the exact Q4_0 one at each rig's
`tpu.env` `SPEC_DRAFT_N_MAX` (E2B 1, E4B 2), i.e. as the rigs serve. `run.sh` runs A then B;
`kvu.sh` reruns A afterwards.

- **A. Several requests at once**: `sweep.py`, 1, 2, 4, 8 and 16 requests, 5 repeats, unique
  prompts, 512 in / 128 out and 128 in / 512 out, `-c 16384 --parallel 16`, drafter off and on.
- **B. Answer quality as the rigs serve**: `-c 8192 --parallel 1`, one request at a time,
  GSM8K (1,319, 2,048-token limit) and BFCL simple (400) through `jev-tpu-v5e1/gen_eval.py`.

## A. Several requests at once, per-slot KV cache (`conc/summary.txt`)

Aggregate output, drafter divided by no drafter:

| requests at once | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B, 512 in / 128 out | 1.137 | 1.013 | 1.011 | 0.947 | 0.968 |
| E2B, 128 in / 512 out | 1.324 | 1.115 | 1.080 | 0.941 | 0.976 |
| E4B, 512 in / 128 out | 1.277 | 1.066 | 0.905 | 0.930 | 0.969 |
| E4B, 128 in / 512 out | 1.658 | 1.232 | 0.797 | 0.850 | 0.939 |

This ran with llama-server's default per-slot KV cache. On this build that cache halved decode
speed at random from 4 requests up on the RTX 3050
(`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-mixed-3050oem`), and the
no-drafter arms here spread 28–36% between repeats at 2 requests. `kvu.sh` reruns A with one
shared cache (`-kvu`) into `conc-kvu/`; read these figures as provisional until it finishes.

## B. Answer quality, one request at a time (`acc/`)

| | with drafter | without drafter |
| --- | ---: | ---: |
| E2B BFCL simple | 0.9225, 337 s | 0.9250, 377 s |
| E2B GSM8K | 0.8969, 5,503 s | running |
| E4B | queued | queued |
