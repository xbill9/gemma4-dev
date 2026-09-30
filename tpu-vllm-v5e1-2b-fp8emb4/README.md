# tpu-vllm-v5e1-2b-fp8emb4

Results for one cell of the Gemma 4 repack sweeps: **`/work/models/gemma-4-E2B-it-qat-q4_0-fp8-text-emb4`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.

Slot 5, `fp8emb4`: `fp8` with the vocabulary tables int4.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-29-e2c-e2b-fp8-emb4-v5e1`](benchmarks/runs/2026-09-29-e2c-e2b-fp8-emb4-v5e1/) | `/work/models/gemma-4-E2B-it-qat-q4_0-fp8-text-emb4` | 0.677 (3,880 records) | `jev-tpu-v5e1` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-29-speed-v5e1`](benchmarks/runs/2026-09-29-speed-v5e1/)
