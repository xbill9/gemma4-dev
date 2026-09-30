# tpu-vllm-v5e1-12b-w8a8rtn

Results for one cell of the Gemma 4 repack sweeps: **`glenic/gemma-4-12B-it-W8A8-INT8`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.

Slot 5, `w8a8rtn`: int8 W8A8 rounded to nearest from the bf16 release, served with its stored `lm_head` (byte-identical to the tied `embed_tokens`) removed by `jev-tpu-v5e1/strip_tied_head.py`.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-30-fillA-12b-w8a8rtn-v5e1`](benchmarks/runs/2026-09-30-fillA-12b-w8a8rtn-v5e1/) | `/work/models/gemma-4-12B-it-W8A8-INT8-glenic-notiedhead` | 0.752 (3,880 records); output tok/s at 1 / 4 / 16: 53 / 201 / 623; GSM8K 0.957 (768-token limit); BFCL 0.877 | `jev-tpu-v5e1` |

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-30-gen2048b2-12b-w8a8rtn-v5e1`](benchmarks/runs/2026-09-30-gen2048b2-12b-w8a8rtn-v5e1/): GSM8K 0.956 (2,048-token limit); BFCL 0.882
