# tpu-vllm-v5e1-2b-w8a8emb4

Results for one cell of the Gemma 4 repack sweeps: **`/work/models/gemma-4-E2B-it-qat-w8a8-int8-emb4`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.

Slot 5, `w8a8emb4`: `w8a8` with `embed_tokens` (and the per-layer embeddings, where present) and an untied `lm_head` int4.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-29-e4a-e2b-w8a8-emb4-v5e1`](benchmarks/runs/2026-09-29-e4a-e2b-w8a8-emb4-v5e1/) | `/work/models/gemma-4-E2B-it-qat-w8a8-int8-emb4` | 0.679 (3,880 records) | `jev-tpu-v5e1` |
| [`2026-09-29-w8e4-e2b-w8a8-emb4-v5e1`](benchmarks/runs/2026-09-29-w8e4-e2b-w8a8-emb4-v5e1/) | `/work/models/gemma-4-E2B-it-qat-w8a8-int8-emb4` | 0.677 (3,880 records) | `jev-tpu-v5e1` |
| [`2026-09-30-fillB-e2b-w8a8-emb4-hf-v5e1`](benchmarks/runs/2026-09-30-fillB-e2b-w8a8-emb4-hf-v5e1/) | `xbill9/gemma-4-E2B-it-qat-w8a8-ct-text-emb4` | 0.673 (3,880 records) | `jev-tpu-v5e1` |

Suite accuracy is the share of the 3,880-record public suite read correctly, computed by the sweep's
`quant_compare.py`. Differences against bf16 are in the paired comparisons each run's `README.md`
lists; they cover two cells, so they stay with the sweep.

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-29-speed-v5e1`](benchmarks/runs/2026-09-29-speed-v5e1/)
- [`2026-09-30-gen2048a-e2b-w8a8-emb4-v5e1`](benchmarks/runs/2026-09-30-gen2048a-e2b-w8a8-emb4-v5e1/): GSM8K 0.895 (2,048-token limit); BFCL 0.920
- [`2026-09-30-longb-e2b-w8a8-emb4-v5e1`](benchmarks/runs/2026-09-30-longb-e2b-w8a8-emb4-v5e1/): 1,043-token prompts at 16: 2,256 tok/s, first token 0.24 s; 3,645-token prompts at 16: 1,249 tok/s, first token 0.82 s; GSM8K 0.873 (768-token limit); BFCL 0.920
