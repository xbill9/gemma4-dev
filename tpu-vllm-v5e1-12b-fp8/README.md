# tpu-vllm-v5e1-12b-fp8

Results for one cell of the Gemma 4 repack sweeps: **`gemma-4-12B-it-qat-q4_0-fp8-text`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.

Slot 5, `fp8`: fp8 W8A8 from the QAT weights, text only (local build).

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-30-fillA-12b-fp8-v5e1`](benchmarks/runs/2026-09-30-fillA-12b-fp8-v5e1/) | `/work/models/gemma-4-12B-it-qat-q4_0-fp8-text` | 0.757 (3,880 records); output tok/s at 1 / 4 / 16: 43 / 165 / 518; GSM8K 0.949 (768-token limit); BFCL 0.950 | `jev-tpu-v5e1` |

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-30-gen2048b2-12b-fp8-v5e1`](benchmarks/runs/2026-09-30-gen2048b2-12b-fp8-v5e1/): GSM8K 0.961 (2,048-token limit); BFCL 0.945
