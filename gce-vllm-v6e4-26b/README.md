# gce-vllm-v6e4-26b

Results for one cell of the Gemma 4 repack sweeps: **`google/gemma-4-26B-A4B-it`** through vLLM on **TPU v6e-4 (`ct6e-standard-4t`, TP=4)**, provisioned as a Compute Engine instance. An artifact rig: measurements only.

Slot 5, none: the reference bf16 release.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-26-tp4-26b-bf16-v6e4`](benchmarks/runs/2026-09-26-tp4-26b-bf16-v6e4/) | `google/gemma-4-26B-A4B-it` | 0.764 (3,880 records) | `jev-tpu-31b` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.
