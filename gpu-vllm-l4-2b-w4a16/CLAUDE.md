# CLAUDE.md — `gpu-vllm-l4-2b-w4a16`

## This is an artifact rig, not a serving rig

No `server.py`, no MCP server, no skill, no plugin manifest, no `tpu.env` — and none is owed. It holds
migrated measurements for **google/gemma-4-E2B-it-qat-w4a16-ct** on an **NVIDIA L4**, and nothing else. Do not scaffold it
into a full rig; there is no deployment path here and never was one in this monorepo.

`benchmarks/rollup.py` discovers rigs by globbing `*/benchmarks/`, so this one appears in `ROLLUP.md`
and gets a generated `INDEX.md`. `NAMING.md` has the rule for artifact rigs.

**Platform slot `gpu`, hardware slot `l4`.** Per `NAMING.md`, the cloud is deliberately not a slot —
Cloud Run, GCE and EC2 all reduce to `l4`, and the host is recorded in the run directory name and in
each report's `Endpoint:` line instead.

## Provenance — read before citing anything here

Migrated on 2026-08-07 from `~/gemma4-tips` / `~/gemma4-tips-aws`, a tree of ~31 legacy
`<platform>-<model>-<host>-devops-agent` directories that **mass-duplicated each other's benchmark
artifacts**. Measured at migration time: **82 `benchmark_report*.md` files reduced to 20 unique, and
109 CSVs to 32.** One 12B Cloud Run report sat in 13 directories spanning 2B through 31B.

**The old directory names are not evidence of anything.** `g2-48-26B-qat-L4` and `g2-96-26B-qat-L4`
both held the *31B* report; `g2-4-2B-qat-L4` held a 12B one. Do not go back to that tree and read a
model or a chip off a directory name.

Only reports whose own `Model:` **and** `Endpoint:` lines are present and agree were migrated — 10 of
the 20 unique. The other 10 carry no model line at all, or name an endpoint where a model should be,
and are unattributable. They were left behind deliberately; they are not missing.

**The CSVs are weaker evidence than the reports.** One 145-row CSV appeared in eight directories,
including as the *primary* `benchmark_sweep_results.csv` beside both the 12B and the 31B Cloud Run
reports — it cannot be both. Only CSVs unique to their source directory came across, and even those
are paired with their report by **co-location only**: nothing inside a CSV identifies a model, a chip,
or a run. Where a run directory here has no CSV, that is why.

## What is trustworthy in these reports

The latency and throughput matrices, and the `Model:` / `Endpoint:` / `Generated at:` header. That is
the whole list.

The parenthetical host label after the model is **not** reliable — the 12B EC2 run is labelled
"(NVIDIA L4 GPU Cloud Run)" while its endpoint is a bare public IP in the AWS tree, not a `run.app`
URL. Where the label and the endpoint disagree, the endpoint wins; the run directory names here record
the host resolved that way.

These are **single-run** grids with no repeat count and no variance figure, on shared-tenancy Cloud
Run and GCE. Treat a cell as one sample. Several grids also show a first-column anomaly — a
multi-second latency at concurrency 1 that vanishes at concurrency 2 — which is cold-start, not a
model property. Do not read the `users=1` column as single-stream latency.
