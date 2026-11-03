# 2026-11-03 — concurrency sweep, RTX 3050 OEM, re-run beside vLLM

The 2026-10-13 sweep (`../2026-10-13-concurrency-3050oem`), same `run.sh` and `sweep.py` arguments, on the
binary now in place: llama.cpp `7182ef71e` (`version.txt`; `tpu.env` still names `77c4540ce`). Run the
same day as `local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-concurrency-3050oem`, which
pairs the two engines.

| requests | 512/128 tok/s | spread | 128/512 tok/s | spread |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 95.35 | 0.4% | 107.15 | 0.9% |
| 2 | 160.85 | 1.6% | 193.16 | 1.3% |
| 4 | 201.18 | 1.0% | 262.82 | 1.4% |
| 8 | 228.08 | 21.1% | 312.59 | 1.1% |
| 16 | 321.54 | 26.0% | 508.89 | 0.2% |
| 32 | 354.67 | 14.9% | 592.74 | 0.5% |
| 64 | 367.60 | 0.8% | 633.26 | 0.7% |

- Every level is 0.7–2.8% above 2026-10-13 (`77c4540ce`), both shapes.
- The 21–26% spreads at 8–32 requests on 512 in / 128 out are back, as on 2026-10-13; the other shape
  stays within 1.4%.
- VRAM at 64 slots: 2,612 MiB (`vram.txt`), 2,621 MiB peak (`gpu.csv`), as before.
