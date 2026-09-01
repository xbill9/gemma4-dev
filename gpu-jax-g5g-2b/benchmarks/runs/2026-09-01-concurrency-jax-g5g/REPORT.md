# 2026-09-01 — concurrency leg

This rig's contribution to the three-runtime concurrency sweep. Full analysis, the comparison
table and the scope caveats are in
`gpu-pytorch-g5g-2b/benchmarks/runs/2026-09-01-concurrency-torch-g5g/REPORT.md`.

Driven by `gpu-pytorch-g5g-2b/sweep.py --concurrency 1,2,4,8,16,32` at 512 in / 128 out, the
same command and shape against all three rigs. Raw per-level results are in
`concurrency.json`.
