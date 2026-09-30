# gce-vllm-v6e1-12b-q4_0

Results for one cell of the Gemma 4 repack sweeps: **`google/gemma-4-12B-it-qat-q4_0-unquantized`** through vLLM on **TPU v6e-1 (`ct6e-standard-1t`)**, provisioned as a Compute Engine instance. An artifact rig: measurements only.

Slot 5, `q4_0`: Google's `-qat-q4_0-unquantized`, the QAT weights stored bf16.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-26-q4_0-12b-q4_0u-ovr-v6e1`](benchmarks/runs/2026-09-26-q4_0-12b-q4_0u-ovr-v6e1/) | `google/gemma-4-12B-it-qat-q4_0-unquantized` | 0.757 (3,880 records) | `jev-tpu-31b` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.
