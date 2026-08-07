# Grid run A — Gemma 4 on TPU v6e-1 (undated)

**Run date: unknown.** Not recorded in the data. Named `undated-` rather than given an inferred
date — the only dated artifact that was sitting beside this one at the rig root
(`benchmark_results.csv`, 2026-04-28) turned out to be a *different, 5-point* run, so borrowing its
date would have attached a real timestamp to the wrong measurement.

**Hardware:** v6e-1, inherited from the same fork-era batch as `undated-vllm-grid-b-v6e1`. This is
a v5e-1 rig; the file records a v6e measurement.

**Matrix:** 12 concurrency levels {1…2048} × 13 context lengths {4…16384} = 156 cells.

| Status | Cells |
|---|---:|
| `success` | 138 |
| `timeout` | 18 |

## Relationship to grid B

`undated-vllm-grid-b-v6e1/` covers the identical matrix with entirely different numbers — see the
comparison table in that directory's `REPORT.md`. Zero of the 138 comparable cells match, and this
run reads roughly 4× *lower* throughput throughout. Its non-success cells are recorded as `timeout`,
where B distinguishes `skipped_capacity_limit` from `failed`; a `timeout` does not say whether the
cell was infeasible or merely slow, so this run's coverage is the weaker record of the two.

## Files

| File | Produced by |
|---|---|
| `benchmark_results.json` | `run_sweep.py` (rig root) |
| `sweep_throughput_heatmap.png`, `sweep_throughput_lineplot.png` | `plot_sweep_results.py` |
| `comparison_plot.png`, `_v2`, `_v3` | `compare_benchmarks.py` (v3 is the current output; v1/v2 are superseded) |

`compare_benchmarks.py` also reads sibling directories (`../tpu-12B-v6e1-devops-agent/`,
`../gpu-12B-L4-devops-agent/`, `/home/xbill/gemma4-tips-aws/…`) that do not exist in this monorepo,
so it cannot run as written regardless of this move.

## Status

**No `reports/*.json`** — no engine version or serve flags were recorded, and a conformant report
requires them. Given the unexplained 4× disagreement with grid B, treat this directory as
provenance-unknown data retained for the record, not as a measurement to cite.
