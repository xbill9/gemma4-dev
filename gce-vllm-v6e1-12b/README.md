# gce-vllm-v6e1-12b

Results for one cell of the Gemma 4 repack sweeps: **`google/gemma-4-12B-it`** through vLLM on **TPU v6e-1 (`ct6e-standard-1t`)**, provisioned as a Compute Engine instance. An artifact rig: measurements only.

Slot 5, none: the reference bf16 release.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-24-12b-v6e1`](benchmarks/runs/2026-09-24-12b-v6e1/) | `google/gemma-4-12B-it` | 0.762 (3,880 records) | `jev-tpu` |
| [`2026-09-25-w4a16-12b-bf16-v6e1`](benchmarks/runs/2026-09-25-w4a16-12b-bf16-v6e1/) | `google/gemma-4-12B-it` | 0.760 (3,880 records) | `jev-tpu-31b` |
| [`2026-09-26-override-12b-bf16-ovr-v6e1`](benchmarks/runs/2026-09-26-override-12b-bf16-ovr-v6e1/) | `google/gemma-4-12B-it` | 0.760 (3,880 records) | `jev-tpu-31b` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-25-followup-v6e1`](benchmarks/runs/2026-09-25-followup-v6e1/)
- [`2026-09-26-override-v6e1`](benchmarks/runs/2026-09-26-override-v6e1/)
- [`2026-09-26-q4_0-v6e1`](benchmarks/runs/2026-09-26-q4_0-v6e1/)
