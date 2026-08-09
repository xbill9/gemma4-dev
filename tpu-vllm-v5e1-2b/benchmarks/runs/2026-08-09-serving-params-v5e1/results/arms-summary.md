# Arms run 2026-08-09, node tpu-vllm-v5e1-2b (spot v5litepod-1, us-west4-a)
Image vllm/vllm-tpu:nightly @ sha256:5b63034a1d04e6f9f3232f0920b81462da2e2d3d721b33cc41033b9eb38f712f
vLLM 0.26.1rc1.dev125+ga7a204cc6 (git a7a204cc6ec99b375985e564f4f271e68ee8f5b8, 2026-07-30)

| arm | container | config delta vs production | boot s | outcome |
|---|---|---|---|---|
| B | mnbt-unset | `--max_num_batched_tokens` removed | ~60 (crash) | ValueError: mm floor 2496 > default 2048 |
| A | prod | none (control) | 1089 | reproduces 2026-08-06 archive to the digit |
| C | prop | mml 32768 + mns 64 + MNBT 2496 | 1701 | WORSE on all 5 cells (-24.7% @8k/64) |
| D | mml32k | mml 32768 only | lost* | >= A on all cells (+4.4% @1k/16) |
| E | rec | mml 32768 + 3 no-op pins + cache mount | **857** | recommended config; 12 cells x3 |

*Arm D boot time lost: no polling loop was set and the uvicorn "Application startup complete"
line carries no timestamp. Recorded as unmeasured rather than estimated.
