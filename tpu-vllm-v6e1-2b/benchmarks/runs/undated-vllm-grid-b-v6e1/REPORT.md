# Grid run B — Gemma 4 on TPU v6e-1 (undated)

**Run date: unknown.** Nothing in the data records one, and the files carry no reliable mtime
(they were re-written by a bulk copy). The directory is deliberately named `undated-` rather than
given an inferred date — see *Why two grids* below for why guessing here would have been wrong.

**Hardware:** v6e-1, per the title of `benchmark_tables.md` ("Gemma 4 TPU Benchmarking Sweep Report
(v6e-1)"). This is a v5e-1 rig; the file records a v6e measurement that travelled with a fork.

**Matrix:** 12 concurrency levels {1…2048} × 13 context lengths {4…16384} = 156 cells.

| Status | Cells |
|---|---:|
| `success` | 129 |
| `skipped_capacity_limit` | 21 |
| `failed` | 6 |

That three-way split is exactly the distinction schema 1.1 added `throughput.sweep[].status` for
(`ok` / `infeasible` / `failed`). The data has always carried it; the 1.0 report format could not
express it, which is part of why this run never became a report.

## Why two grids

`undated-vllm-grid-a-v6e1/` covers the **same 12 × 13 matrix** but is a different measurement:

| | grid A | grid B (this one) |
|---|---|---|
| Source | `benchmark_results.json` | `grid_benchmark_results.csv` |
| Throughput at c=1, ctx=4 | 1.29 | 5.52 |
| Throughput at c=8, ctx=128 | 5.26 | 24.07 |
| Non-success vocabulary | `timeout` (18) | `skipped_capacity_limit` (21), `failed` (6) |

**Zero of the 138 comparable cells agree**, and B is roughly 4× A throughout. They are two runs of
one matrix, not two views of one run. Which is more representative is not recoverable from what was
kept — no engine version, no serve flags, no date for either.

## Files

| File | Produced by |
|---|---|
| `grid_benchmark_results.csv` | `run_grid_benchmark.py` / `run_fast_sweep.py` (rig root) |
| `benchmark_tables.md` | `generate_report.py` |
| `throughput_heatmap.png`, `latency_heatmap.png` | `plot_grid.py` |
| `throughput_line.png` | attributed by content; generator not identified |

Those scripts now read this directory by default. They still carry v6e-era hardcoded titles and
sibling paths (`../tpu-2B-v6e4-devops-agent/`) that no longer exist — see the rig `CLAUDE.md`.

## Status

**No `reports/*.json`.** A conformant report requires `software.engine` and `software.version`,
neither of which was recorded. Usable for shape and for the status-vocabulary example; not citable
against another chip.
