# gce-vllm-v6e1-12b-w4a16

Results for one cell of the Gemma 4 repack sweeps: **`google/gemma-4-12B-it-qat-w4a16-ct`** through vLLM on **TPU v6e-1 (`ct6e-standard-1t`)**, provisioned as a Compute Engine instance. An artifact rig: measurements only.

Slot 5, `w4a16`: Google's `-qat-w4a16-ct`, the QAT weights re-rounded to a min-max scale per group of 32.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-25-w4a16-12b-w4a16-v6e1`](benchmarks/runs/2026-09-25-w4a16-12b-w4a16-v6e1/) | `google/gemma-4-12B-it-qat-w4a16-ct` | 0.750 (3,880 records) | `jev-tpu-31b` |
| [`2026-09-26-override-12b-w4a16-ovr-v6e1`](benchmarks/runs/2026-09-26-override-12b-w4a16-ovr-v6e1/) | `google/gemma-4-12B-it-qat-w4a16-ct` | 0.751 (3,880 records) | `jev-tpu-31b` |
| [`2026-09-26-q4_0-12b-w4a16-ovr-v6e1`](benchmarks/runs/2026-09-26-q4_0-12b-w4a16-ovr-v6e1/) | `google/gemma-4-12B-it-qat-w4a16-ct` | 0.751 (3,880 records) | `jev-tpu-31b` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-25-followup-v6e1`](benchmarks/runs/2026-09-25-followup-v6e1/)
- [`2026-09-28-pr3660-v6e1`](benchmarks/runs/2026-09-28-pr3660-v6e1/)
