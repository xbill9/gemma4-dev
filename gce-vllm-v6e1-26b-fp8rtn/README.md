# gce-vllm-v6e1-26b-fp8rtn

Results for one cell of the Gemma 4 repack sweeps: **`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`** through vLLM on **TPU v6e-1 (`ct6e-standard-1t`)**, provisioned as a Compute Engine instance. An artifact rig: measurements only.

Slot 5, `fp8rtn`: fp8 W8A8 (`FP8-dynamic`) rounded from the bf16 release.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-24-26b-fp8-v6e1`](benchmarks/runs/2026-09-24-26b-fp8-v6e1/) | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | 0.760 (3,880 records) | `jev-tpu` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-25-followup-v6e1`](benchmarks/runs/2026-09-25-followup-v6e1/)
- [`2026-09-26-moe-v6e1`](benchmarks/runs/2026-09-26-moe-v6e1/)
