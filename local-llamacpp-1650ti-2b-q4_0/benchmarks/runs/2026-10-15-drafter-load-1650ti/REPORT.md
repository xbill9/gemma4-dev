# 2026-10-15 — the drafter under load and answer quality as the rigs serve, GTX 1650 Ti

**Answer: as the rigs serve, one request at a time, the drafter costs no answers and finishes
GSM8K 1.34x (E2B) and 1.78x (E4B) faster. With several requests at once it pays at 1–2 and
costs from 4 (E4B) or 8 (E2B) on.** On this card one shared KV cache (`-kvu`) removes the
random slowdowns of the per-slot cache but gives 0.59–0.81x the output at 16 requests, the
reverse of the RTX 3050; the drafter's ratio is the same under either cache.

## Setup

The laptop `yogagpu` (GTX 1650 Ti Max-Q, 4096 MiB, driver 615.71.09), llama.cpp `77c4540ce`
(build 813, `host.txt`) from `~/llama.cpp-77c4540ce`. Drafter: the exact Q4_0 one at each rig's
`tpu.env` `SPEC_DRAFT_N_MAX` (E2B 1, E4B 2), as the rigs serve. Server flags `-ngl 99 -ctk f16
-ctv f16 -fa 1 -t 6 -tb 12 --reasoning off`. `run.sh` ran A then B (2026-10-15 13:50 to
2026-10-16 01:34); `kvu.sh` reran A with `-kvu` (01:34 to 03:13). `driver.log` has both.

## B. Answer quality, one request at a time (`acc/summary.txt`)

`-c 8192 --parallel 1`, `gen_eval.py --concurrency 1`, prompt cache at the server default:
GSM8K (1,319, 2,048-token limit) and BFCL simple (400) through `jev-tpu-v5e1/gen_eval.py`. Pairing
and 95% ranges from `gen_compare.compare` (10,000 resamples, seed 0) in `summarize_acc.py`.

| | no drafter | drafter | difference (95% range) | only no drafter right | only drafter right | wall s | x faster |
|---|---:|---:|---|---:|---:|---|---:|
| E2B GSM8K, 1 token | 0.8939 | 0.8969 | +0.0030 (−0.0083 to +0.0136) | 25 | 29 | 7,374 → 5,503 | 1.34 |
| E2B BFCL, 1 token | 0.9250 | 0.9225 | −0.0025 (−0.0125 to +0.0050) | 2 | 1 | 377 → 336 | 1.12 |
| E4B GSM8K, 2 tokens | 0.9363 | 0.9310 | −0.0053 (−0.0121 to +0.0015) | 15 | 8 | 12,790 → 7,177 | 1.78 |
| E4B BFCL, 2 tokens | 0.9125 | 0.9125 | 0.0000 | 0 | 0 | 772 → 618 | 1.25 |

- **Every range includes zero.** E4B BFCL gets the same 400 verdicts either way.
- **The drafter's speed-up holds over a full benchmark** as the rigs serve, against 0.88–0.96x
  at 8 requests at once (`../2026-10-14-drafter-accuracy-1650ti`). On the RTX 3050 at its serving
  setting GSM8K finished 1.76x (E2B) and 2.23x (E4B) faster
  (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem`).
- **Scores sit with the earlier runs**: E2B GSM8K 0.8939–0.8992 and E4B 0.9295–0.9363 across
  this run and the 8-at-once run.

## A. Several requests at once (`conc*/summary.txt`)

`sweep.py`, 1, 2, 4, 8 and 16 requests, 5 repeats, unique prompts, 512 in / 128 out and 128 in /
512 out, `-c 16384 --parallel 16`, drafter off and on. Run twice: llama-server's default per-slot
KV cache (`conc/`, 1,024 tokens per slot) and one shared cache (`conc-kvu/`, `-kvu`, every slot
may use the full 16,384).

Aggregate output, drafter divided by no drafter, shared cache:

| requests at once | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B, 512 in / 128 out | 1.146 | 1.039 | 1.028 | 0.949 | 0.969 |
| E2B, 128 in / 512 out | 1.326 | 1.123 | 1.094 | 0.905 | 0.960 |
| E4B, 512 in / 128 out | 1.284 | 1.067 | 0.904 | 0.931 | 0.923 |
| E4B, 128 in / 512 out | 1.650 | 1.175 | 0.785 | 0.834 | 0.829 |

With per-slot caches the same ratios are within 0.04 of these except E4B at 16 requests (0.969
and 0.939) and E4B 128/512 at 2 (1.232); `conc/summary.txt`.

**The cache setting on this card (`conc-kvu/compare.txt`):**

- **The shared cache removes the random slowdowns.** Per-slot, the no-drafter arms spread
  28.2% (E2B) and 35.6% (E4B) between repeats at 2 requests, 512 in; shared, 0.7% and 1.0%.
- **It also costs output as requests rise**: shared over per-slot is 1.01–1.05x at 1 request,
  0.75–0.92x at 8 and 0.59–0.81x at 16. On the RTX 3050 the shared cache gave 1.2–1.9x more at
  16–64 requests (`.../2026-10-15-concurrency-mixed-3050oem`). Likely because each slot attends
  over a cache sized for 16,384 tokens instead of 1,024 (inferred, not measured).
- **No-drafter peak output**: E2B 163.6 tok/s per-slot and 136.6 shared (128 in / 512 out);
  E4B 87.4 and 80.3.

The 1650 Ti rigs serve one request at a time (`PARALLEL_SLOTS=1`), where neither cache setting
nor the drafter's load behaviour applies.
