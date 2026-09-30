# tpu-vllm-v5e1-2b-w8a8rtn

Results for one cell of the Gemma 4 repack sweeps: **`glenic/gemma-4-E2B-it-W8A8-INT8`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.

Slot 5, `w8a8rtn`: int8 W8A8 rounded to nearest from the bf16 release.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-29-w8a8b-e2b-w8a8-v5e1`](benchmarks/runs/2026-09-29-w8a8b-e2b-w8a8-v5e1/) | `glenic/gemma-4-E2B-it-W8A8-INT8` | 0.670 (3,880 records) | `jev-tpu-v5e1` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-29-w8a8-v5e1`](benchmarks/runs/2026-09-29-w8a8-v5e1/)
