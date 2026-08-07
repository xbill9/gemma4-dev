# Concurrency run — Gemma 4 on TPU v6e-1

**Run: 2026-04-28 18:20 UTC.** This is the only legacy artifact in this rig that records its own
date — every row of `benchmark_results.csv` carries an ISO timestamp, and they span 51 seconds
(18:20:01 → 18:20:52). The date here is read from the data, not inferred.

**Hardware:** v6e-1, inherited from the same fork-era batch as the two `undated-vllm-grid-*-v6e1`
directories. This is a v5e-1 rig; the file records a v6e measurement.

**Matrix:** 5 concurrency levels, single context. 5 cells, all measured, none failed.

| Column | Meaning |
|---|---|
| `concurrency` | simultaneous requests |
| `total_requests`, `success_rate` | 20 requests per level, all succeeded |
| `avg_latency`, `p95_latency` | seconds |
| `req_per_sec`, `tokens_per_sec` | throughput |

## Why this is its own directory

These five rows sat at the rig root beside two 156-cell grids, and the whole set was initially
filed under one date. It is a separate run: different shape (1-D concurrency, not context ×
concurrency), different column vocabulary, and the only one with timestamps. Its date was the one
piece of real provenance in the group, and folding it in with the grids would have spent it on data
it does not describe.

## Files

| File | Produced by |
|---|---|
| `benchmark_results.csv` | `benchmarking_suite.py --output` (rig root), default filename |

`plot_benchmark.py` reads this file and now points here by default.

## Status

**No `reports/*.json`.** The schema requires `software.engine` and `software.version`; neither was
recorded, and the CSV's `avg_latency` / `p95_latency` columns do not map onto the schema's
TTFT/TPOT/ITL breakdown without knowing how they were measured. Retained as a dated record.
