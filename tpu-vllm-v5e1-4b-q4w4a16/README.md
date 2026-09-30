# tpu-vllm-v5e1-4b-q4w4a16

Results for one cell of the Gemma 4 repack sweeps: **`/work/models/gemma-4-E4B-it-qat-q4_0-w4a16-ct`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.

Slot 5, `q4w4a16`: W4A16 holding the QAT Q4_0 grid exactly (xbill9 repack of `-qat-q4_0-unquantized`).

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-29-e4br-e4b-repack-v5e1`](benchmarks/runs/2026-09-29-e4br-e4b-repack-v5e1/) | `/work/models/gemma-4-E4B-it-qat-q4_0-w4a16-ct` | 0.729 (3,880 records) | `jev-tpu-v5e1` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-29-e4b-v5e1`](benchmarks/runs/2026-09-29-e4b-v5e1/)
