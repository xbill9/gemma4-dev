# tpu-vllm-v5e1-2b-q4w4a16

Results for one cell of the Gemma 4 repack sweeps: **`/work/models/gemma-4-E2B-it-qat-q4_0-w4a16-ct`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.

Slot 5, `q4w4a16`: W4A16 holding the QAT Q4_0 grid exactly (xbill9 repack of `-qat-q4_0-unquantized`).

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-29-e2br-e2b-repack-v5e1`](benchmarks/runs/2026-09-29-e2br-e2b-repack-v5e1/) | `/work/models/gemma-4-E2B-it-qat-q4_0-w4a16-ct` | 0.678 (3,880 records) | `jev-tpu-v5e1` |
| [`2026-09-30-fillB-e2b-w4a16text-v5e1`](benchmarks/runs/2026-09-30-fillB-e2b-w4a16text-v5e1/) | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` | 0.678 (3,880 records); did not serve: see its boot log | `jev-tpu-v5e1` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-29-e2b-v5e1`](benchmarks/runs/2026-09-29-e2b-v5e1/)
- [`2026-09-29-mtpg1-v5e1`](benchmarks/runs/2026-09-29-mtpg1-v5e1/)
- [`2026-09-29-mtpg2-v5e1`](benchmarks/runs/2026-09-29-mtpg2-v5e1/)
- [`2026-09-29-mtpq-v5e1`](benchmarks/runs/2026-09-29-mtpq-v5e1/)
- [`2026-09-29-w8a8-v5e1`](benchmarks/runs/2026-09-29-w8a8-v5e1/)
- [`2026-09-30-fillE-e2b-repack-v5e1`](benchmarks/runs/2026-09-30-fillE-e2b-repack-v5e1/): GSM8K 0.879 (768-token limit); BFCL 0.920
- [`2026-09-30-gen2048a-e2b-repack-v5e1`](benchmarks/runs/2026-09-30-gen2048a-e2b-repack-v5e1/): GSM8K 0.901 (2,048-token limit); BFCL 0.920
- [`2026-09-30-gspeedb-e2b-repack-v5e1`](benchmarks/runs/2026-09-30-gspeedb-e2b-repack-v5e1/): output tok/s at 1 / 4 / 16: 136 / 532 / 1,910
