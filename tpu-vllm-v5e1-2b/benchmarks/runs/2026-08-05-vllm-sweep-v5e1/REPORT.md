# Serving Sweep — Gemma 4 E2B on TPU v5e-1: vLLM, context × concurrency

**Run:** 2026-08-05 (see *Provenance* — the date is from file timestamps, not recorded in the data)
**Matrix:** 1 model × 4 concurrency levels {1, 4, 16, 64} × 5 context lengths {128, 1024, 4096, 8192, 15000}
**Coverage: 20 measured cells, 0 failed, 0 infeasible.** Every cell in the matrix produced a result.
Full matrices in [`tables.md`](tables.md), machine-readable cells in [`results/summary.json`](results/summary.json).

Regenerate the derived files with `python3 aggregate.py`.

## Headline

| | value |
| :--- | ---: |
| Peak aggregate output | **1,878 tok/s** (ctx 128, c=64) |
| Single-stream output (ctx 128) | 127 tok/s @ 14.4 ms median TTFT |
| Aggregate at 15K context | 203 tok/s (c=64) |
| Best total (prefill+decode) throughput | 28,260 tok/s (ctx 8192, c=16) |

## Context is the binding axis, not user count

Aggregate output tok/s, from [`tables.md`](tables.md):

| ctx \ users | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| 128 | 127 | 432 | 1,077 | **1,878** |
| 1024 | 123 | 435 | 939 | 1,475 |
| 4096 | 111 | 400 | 613 | 823 |
| 8192 | 106 | 356 | 435 | 340 |
| 15000 | 90 | 185 | 169 | 203 |

Concurrency scales output cleanly only while the KV budget is slack. At ctx 128 the chip returns
14.8× single-stream throughput at 64 users; by ctx 4096 that has fallen to 7.4×, and at ctx 8192
**adding users past 16 makes aggregate throughput go down** (435 → 340 tok/s), the signature of KV
pressure forcing preemption. At 15000 the curve is flat-to-noisy across all user counts — the chip
is serving a context-determined ceiling of roughly 170–200 tok/s regardless of how many streams ask.

Total (prefill+decode) throughput peaks at 28,260 tok/s at ctx 8192 / c=16, an order of magnitude
above the output-only figure, because at long context the great majority of tokens processed are
prefill. Quoting that number as "throughput" would flatter the chip; output tok/s is what a user
waits on.

## Provenance and what is not recorded

This run predates the standard, so several fields a v1.1 report requires were never captured:

| Field | Status |
| :--- | :--- |
| Run date | **Inferred** from file mtimes (`sweep_results_v5e1.csv` 2026-08-04 21:35, plots 2026-08-05 09:30). Not recorded in the data. |
| Engine version | Not recorded. |
| `max_model_len`, time to healthy | Not recorded. The 15000 top context suggests a 16384 limit, but that is an inference, not a measurement. |
| Zone, provisioning model, instance | Not recorded. |
| Median / p90 / p99 TPOT and ITL | Not measured — the CSV carries **mean only** for TPOT and ITL. `per_stream_tok_per_s` in `tables.md` is therefore derived from mean TPOT, not the median the schema prefers. |

**Because of that, this run has no `reports/*.json`.** Emitting one would require inventing
`software.version`, which the schema requires. The numbers here are real; the stack identity behind
them is not reconstructable. The first conformant v5e-1 report is
`../../reports/2026-08-06-gemma4-e2b-v5e1.json`, measured on a stack whose identity was captured.

Treat this directory as a measurement whose provenance is partial — usable for shape, not for
citing against another chip's numbers.
