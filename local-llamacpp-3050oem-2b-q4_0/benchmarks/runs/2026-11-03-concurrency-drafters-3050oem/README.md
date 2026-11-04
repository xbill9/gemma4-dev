# Concurrency sweep with the exact Q4_0 drafter, 2026-11-03

`run.sh` repeats `../2026-11-03-concurrency-3050oem` (the same 64-slot server and sweep.py arguments) at 1 and
2 drafted tokens (`q4x-n1/`, `q4x-n2/`), on llama.cpp `7182ef71e` (`version.txt`). It ran beside vLLM's drafter
sweep on the same card the same evening. The report comparing both engines, with and without drafters, is
`local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-concurrency-drafters-3050oem/REPORT.md`.
