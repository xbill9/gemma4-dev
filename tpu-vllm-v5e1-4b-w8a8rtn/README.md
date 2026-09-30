# tpu-vllm-v5e1-4b-w8a8rtn

Results for one cell of the Gemma 4 repack sweeps: **`glenic/gemma-4-E4B-it-W8A8-INT8`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.

Slot 5, `w8a8rtn`: int8 W8A8 rounded to nearest from the bf16 release, served with its stored `lm_head` (byte-identical to the tied `embed_tokens`) removed by `jev-tpu-v5e1/strip_tied_head.py`.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-30-fillB-e4b-w8a8rtn-v5e1`](benchmarks/runs/2026-09-30-fillB-e4b-w8a8rtn-v5e1/) | `/work/models/gemma-4-E4B-it-W8A8-INT8-glenic-notiedhead` | 0.730 (3,880 records); output tok/s at 1 / 4 / 16: 119 / 459 / 1,597 | `jev-tpu-v5e1` |
