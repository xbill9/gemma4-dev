# gce-vllm-v6e1-26b-q4w4a16

Results for one cell of the Gemma 4 repack sweeps: **`/work/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`** through vLLM on **TPU v6e-1 (`ct6e-standard-1t`)**, provisioned as a Compute Engine instance. An artifact rig: measurements only.

Slot 5, `q4w4a16`: W4A16 holding the QAT Q4_0 grid exactly (xbill9 repack of `-qat-q4_0-unquantized`).

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-26-kvbf16-26b-q4w4-kvbf16-v6e1`](benchmarks/runs/2026-09-26-kvbf16-26b-q4w4-kvbf16-v6e1/) | `/work/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct` | 0.755 (3,880 records) | `jev-tpu-31b` |
| [`2026-09-26-moe2-26b-q4w4-v6e1`](benchmarks/runs/2026-09-26-moe2-26b-q4w4-v6e1/) | `/work/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct` | 0.753 (3,880 records) | `jev-tpu-31b` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-26-moe-v6e1`](benchmarks/runs/2026-09-26-moe-v6e1/)
- [`2026-09-26-repack-v6e1`](benchmarks/runs/2026-09-26-repack-v6e1/)
