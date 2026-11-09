# SGLang-JAX first light: Gemma 4 E2B on one TPU v5e chip (2026-11-08/09)

One flex-start `v5litepod-1` in `us-west4-a`, queued resource `tpu-sglangjax-v5e1-2b`, run unattended by
`flight/run-sgl.sh` from 01:44 to 03:00 UTC on 2026-11-09 and deleted by `finish.sh` (`finish-vm.log`:
`delete requested (rc=0)`). Requests, prompts and statistics are the v5e-1 ladder's
(`tpu-vllm-v5e1-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-ladder-v5e1`). Every figure below is printed by
`analyze.py` into `SUMMARY.txt`; this file quotes it.

Software (`versions.txt`): sglang-jax `b3e3e16b2` with `patches/gemma4-e2b.diff` (sha256 `d6ac1d7a2f8f6e55…`),
jax / jaxlib 0.11.1, libtpu 0.0.46.3. Checkpoint `google/gemma-4-E2B-it` at `4adf1574`, bf16. Server flags
(`one-sglang/command.txt`): `--context-length 16384 --max-running-requests 64 --mem-fraction-static 0.8
--page-size 128 --watchdog-timeout 7200 --reasoning-parser gemma4`. KV pool 103,040 tokens at 43,008 bytes a
token; 64 running requests (`one-sglang/server.log`).

#### One request at a time

Six prompts, 256 tokens, greedy, thinking off, 3 repeats; geometric mean over prompts of each prompt's median.
Ladder arms as measured 2026-11-06/07 on other VMs.

| Arm | Weights | e2e tok/s | decode tok/s | TTFT ms | 4 at once tok/s | x vllm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sglangjax` | bf16 | 159.9 | 161.7 | 20.2 | 631.5 | 1.111 |
| `vllm` | int4 `-text-emb4` | 143.9 | 145.1 | 15.4 | 477.2 | 1.000 |
| `jaxrust` | bf16 | 145.2 | 145.5 | 9.4 | 145.4 | 1.009 |
| `jax-bf16` | bf16 | 134.3 | 131.7 | 11.3 | 150.1 | 0.933 |

- 🟢 SGLang-JAX is the fastest engine measured on one v5e chip for one request: 1.11x vLLM and 1.19x the bf16
  JAX engine it was ported from.
- 🟢 It repeats its greedy replies on 6 of 6 prompts.
- ⚠️ Its greedy replies match none of the other arms (0 of 6 against `jax-bf16`, `jaxrust` and `vllm`);
  `jax-bf16` and `jaxrust` match each other on 3 of 6. GSM8K below is the check on its answers.

#### Many requests at once

Output tok/s, median of 5 repeats, unique prompts (`sweep.py`). `vllm` is the ladder's int4 arm.

| Shape | Arm | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 in / 128 out | `sglangjax` | 154.2 | 117.4 | 552.0 | 915.6 | 1,663.0 | 2,358.4 | 2,758.9 |
| 512 in / 128 out | `vllm` | 140.6 | 257.4 | 440.2 | 682.2 | 937.9 | 1,156.8 | 1,288.4 |
| 128 in / 512 out | `sglangjax` | 160.3 | 237.2 | 380.4 | 943.1 | 1,671.9 | 2,550.2 | 4,249.4 |
| 128 in / 512 out | `vllm` | 144.5 | 270.2 | 476.4 | 778.4 | 1,112.8 | 1,462.5 | 1,689.1 |

- 🟢 From 8 requests up SGLang-JAX leads, 2.14x vLLM at 64 on 512 / 128 and 2.52x on 128 / 512.
- ❌ At 2 to 4 requests it trails on two of four cells (0.46x at 2 on 512 / 128).
- ⚠️ Its repeats are far apart: the spread from slowest to fastest repeat is 49–205% of the slowest, against
  0.0–5.0% for vLLM. The medians above are its typical figure; any single run may sit well below it.

#### Answers

GSM8K, 1,319 questions, 2,048 tokens, one request at a time, thinking off; paired by question with a 95%
paired bootstrap range (2,000 resamples, seed 0).

| Arm | GSM8K | sglangjax − arm, paired [95%] | differ |
| --- | ---: | --- | ---: |
| `sglangjax` | 0.9143 (1,206) | – | – |
| `jax-bf16` | 0.9128 | +0.0015 [−0.0091, +0.0121] | 50 |
| `jaxrust` | 0.9113 | +0.0030 [−0.0076, +0.0144] | 58 |
| `vllm` | 0.9007 | +0.0136 [+0.0023, +0.0258] | 66 |
| `vllm-text` | 0.8992 | +0.0152 [+0.0023, +0.0273] | 70 |

- 🟢 Level with both bf16 engines (ranges include zero), so the E2B patch gives the same answers as the
  reference port it was written from.
- 🟢 Ahead of the int4 vLLM arms (ranges above zero). The bf16 engines led vLLM by about one point in the
  ladder too, short of clear there.
- GSM8K took 3,748 s against 3,946 for `jax-bf16` and 3,630 for `vllm`; 3 replies hit the token limit.

#### Start-up

`ready_s` 270.2: weights load and 7 prefill plus 7 decode shapes compile. The VM's JAX compile cache held the
prefill shapes compiled during debugging on the same VM, so prefill shapes took about 13 s each (94 s for 7);
decode shapes took 88 s for 7. A fresh VM has no cache and compiles each 512-wide attention shape for about
300 s (`debug/sgl-dbg*.log`).

#### Scope

The sweep compares bf16 SGLang-JAX with int4 vLLM, so it is an engine-plus-weights comparison; no bf16 vLLM
sweep with these requests exists on v5e-1. Different VMs and days from the ladder, jax 0.11.1 here against
0.12.0.1 for the ladder's JAX arms. No BFCL, no drafter, no image or audio input (text only). Context 16,384.
