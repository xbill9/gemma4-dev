# Serving Sweep — Gemma 4 E2B / E4B / 12B on TPU v6e-1: vLLM × the PyTorch TPU backend

**Run:** 2026-07-25 · `vllm-sweep-vm` (ct6e-standard-1t, flex-start, europe-west4-a, `comglitn`), deleted after collection
**Matrix:** 3 models × 7 concurrency levels {1,2,4,8,16,32,64} × 13 context lengths {8…65536}, output 128 tok/cell, greedy (`--ignore-eos`)
**Coverage: 252 measured cells, 0 failed, 42 infeasible-by-hardware** (recorded, not silently skipped). Full matrices in [`tables.md`](tables.md), raw JSONs in `results/`, per-cell logs on request from `sweep-main.tgz`/`sweep-fixup.tgz`.
**Companion:** the PyTorch TPU backend single-stream + batched-loop numbers from [`../2026-07-25-tpu-e2b-v6e1/REPORT.md`](../2026-07-25-tpu-e2b-v6e1/REPORT.md) and `logs/batch-sweep.log` (same day, same chip type, same zone).

## 1. The chip decides what each model is

| | E2B | E4B | 12B |
| :--- | ---: | ---: | ---: |
| Bootable max-model-len (bf16, 32 GB HBM) | 65536 | 65536 | **8192** |
| Time to healthy | 490 s | 524 s | 622 s |
| Single-stream (ctx 128) | 211 tok/s @ 11.6 ms TTFT | 111 tok/s @ 15.2 ms | 45 tok/s @ 27.6 ms |
| Peak aggregate output | **2,501 tok/s** (c=64) | 2,138 tok/s (c=64) | 990 tok/s (c=64) |
| Aggregate at 64K context | 147 tok/s (c=8) | 82 tok/s (c=8) | — infeasible |
| Cost/M output tokens (peak / single-stream) | $0.15 / $1.77 | $0.18 / $3.37 | $0.38 / $8.28 |

12B on one v6e-1 is a **short-context model**: it boots only at 8K, and KV pressure flattens throughput to ~135 tok/s by 2K context and ~76 by 4K regardless of user count. E2B holds >1,900 tok/s aggregate even at 4K context and still serves 8 users at 64K. If long context matters, the answer on this chip is E2B/E4B — or a bigger slice for 12B.

## 2. Context is the real axis, not users

For every model, throughput scales almost linearly with users until the KV budget binds, then the curves collapse onto a context-determined ceiling (full matrices in `tables.md`):

- **E2B**: ctx ≤ 512 scales cleanly to c=64 (~2,450 tok/s). At 2048 ctx the ceiling is ~2,100; at 4096 ~1,900; at 64K the chip sustains ~100–147 tok/s aggregate with TTFT rising to ~1.4 s single-stream (65K-token prefill).
- **E4B**: same shape, roughly half the throughput, and the 4K-context ceiling (789 tok/s) arrives one row earlier than E2B's.
- **12B**: user-scaling stops mattering past c≈16 for short contexts (990 tok/s ceiling), and past c≈2 for 4K context (76 tok/s ceiling — the KV cache simply can't hold more concurrent 4K streams).

TTFT stays interactive (< 200 ms median) everywhere except the ≥16K rows, where prefill dominates (E2B 64K: 1.4–17 s depending on load).

## 3. the PyTorch TPU backend vs vLLM — the two regimes

**Batch 1: identical.** Same chip, same models, measured hours apart:

| Model | the PyTorch TPU backend compiled loop | vLLM engine (ctx 128, c=1) |
| :--- | ---: | ---: |
| E2B | 225 tok/s (4.5 ms/step) | 211 tok/s (4.7 ms TPOT) |
| E4B | 114 tok/s (8.8 ms) | 111 tok/s (8.9 ms) |
| 12B | 45.2 tok/s (22.1 ms) | 45.3 tok/s (22.0 ms) |

Single-stream decode is compute-bound; the stack doesn't matter. (the PyTorch TPU backend eager, for contrast: ~3 tok/s on all three.)

**Concurrency: two different worlds.** E2B aggregate output tok/s at matching user counts:

| Users | the PyTorch TPU backend static batch (no KV cache) | vLLM continuous batching | vLLM advantage |
| ---: | ---: | ---: | ---: |
| 1 | 225 | 211 | 0.9× |
| 2 | 390 | 419 | 1.1× |
| 4 | 450 | 828 | 1.8× |
| 8 | **493 (peak)** | 1,554 | 3.2× |
| 16 | 474 | 2,224 | 4.7× |
| 32 | 427 | 2,451 | 5.7× |
| 64 | 392 | 2,452 | **6.3×** |

The static-batch loop doubles once (B=2) then saturates near 500 tok/s and *declines* — without a KV cache, every added stream re-pays full-sequence attention every step. vLLM keeps scaling to 2,500. The lesson is crisp: **the PyTorch TPU backend's batch story ends where KV caching begins**; beyond ~2–4 lockstep streams you either implement a real KV cache or use the engine that has one.

## 4. Practical guidance

- **Shared/multi-user endpoint** → vLLM, any model. 6× throughput at c=64, $0.15/M tokens at E2B saturation.
- **Single stream, latency-sensitive, or vLLM-unsupported checkpoints (QAT)** → the PyTorch TPU backend compiled loop is engine-equivalent at batch 1 and up to ~2 lockstep users.
- **Model pick on one v6e-1**: E2B is the throughput/context king; E4B costs ~2× per token for one quality step up; 12B costs ~5× per token, caps at 8K context, and really wants a v6e-4.
- **Interactive long-context (≥16K)**: budget seconds of TTFT per request; keep concurrency ≤ 8 on E2B/E4B.

## 5. Method notes & caveats

- Random-token prompts (`vllm bench serve --dataset-name random`), fixed 128-token output, `--ignore-eos`; one run per cell (the 07-21 baseline measured run-to-run cv ≤ 0.3 % for this stack). Prompt-count per cell scales down with context (8×users → 1×users) to bound prefill time; absolute aggregates at high ctx are conservative.
- The "64K" row is effective input 65,376 (max-model-len 65,536 − output − margin); the naive 65,536-input cells are impossible by construction and were re-measured in a fix-up pass.
- 12B rows ≥ 8K context are infeasible at bf16 on 32 GB (vLLM cannot boot the KV cache) — a quantized 12B would change this picture.
- the PyTorch TPU backend batch numbers use the no-KV-cache 256-token loop; they measure that design's ceiling, not the PyTorch TPU backend's theoretical best with a hand-built KV cache.
- vLLM: `vllm/vllm-tpu:nightly` (tpu-inference JAX backend); the PyTorch TPU backend: `0.1.1.dev20260725090141` + torch 2.11.0+cpu.

## 6. Session cost

Two flex-start v6e-1 VMs, both deleted after collection: `tpu-vm` ~1.6 h + `vllm-sweep-vm` ~4.1 h ≈ **$7.70 total** for 252 serving cells + 13 the PyTorch TPU backend benchmark configurations.
