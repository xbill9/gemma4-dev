# tpu-vllm-v5e1-26b-q4w4a16emb4

Results for one cell of the Gemma 4 repack sweeps: **`xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4`** through vLLM on **TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: measurements only.

Slot 5, `q4w4a16emb4`: the QAT W4A16 repack with `embed_tokens` and an untied `lm_head` int4, text only.

On this chip it serves only with the KV cache capped at 17 blocks (2,176 tokens), as the 26B repack does: its weights take 13.72 GiB against the repack's 13.58, so int4 tables free no memory for context. 3,072 and 2,560 tokens failed; see Other runs.

| Run | Checkpoint | Suite accuracy | Sweep |
|---|---|---:|---|
| [`2026-09-30-fillC-26b-emb4-v5e1`](benchmarks/runs/2026-09-30-fillC-26b-emb4-v5e1/) | `xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4` | 0.755 (3,880 records); output tok/s at 1 / 4 / 16: 36 / 120 / 228 | `jev-tpu-v5e1` |

## Other runs

No suite; see each run's `README.md`.

- [`2026-09-30-fillC-26b-emb4-ctx-v5e1`](benchmarks/runs/2026-09-30-fillC-26b-emb4-ctx-v5e1/): did not serve: HTTP status client error (429 Too Many Requests)
- [`2026-09-30-fillG-26b-emb4-ctx8k-v5e1`](benchmarks/runs/2026-09-30-fillG-26b-emb4-ctx8k-v5e1/): refused at 8,192 tokens before compiling: 1.72 GiB of KV needed, 1.51 GiB available inside the 0.97 cap
- [`2026-09-30-fillG2-26b-emb4-ctx2560-v5e1`](benchmarks/runs/2026-09-30-fillG2-26b-emb4-ctx2560-v5e1/): did not serve at 2,560 tokens (20 blocks): `RuntimeProgramAllocationFailure`
- [`2026-09-30-fillG2-26b-emb4-ctx3k-v5e1`](benchmarks/runs/2026-09-30-fillG2-26b-emb4-ctx3k-v5e1/): did not serve at 3,072 tokens (24 blocks): `CompileTimeHbmOom` (run.log)
