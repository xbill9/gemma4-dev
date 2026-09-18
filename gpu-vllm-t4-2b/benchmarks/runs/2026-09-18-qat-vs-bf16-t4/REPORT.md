# 2026-09-18 — Gemma 4 E2B, QAT w4a16 vs bf16, on one Tesla T4

**QAT w4a16 is faster than bf16 on this T4 at every 512-token cell measured:
1.79x per stream at c=1, 1.31x total throughput at c=8.** At 4096-token prompts
the two are within 5% of each other except at c=1 (QAT 14.00 vs 12.11 tok/s),
because the T4's prefill, not its decode, sets the rate there — and prefill is
equally slow for both builds.

## Setup

| | |
| --- | --- |
| Host | GCE `n1-standard-2` (2 vCPU, 7.80 GB RAM), `us-west2-b`, 16 GB swapfile on `/opt1` |
| GPU | Tesla T4, SM 7.5, 15360 MiB, driver 615.71.09, 70 W cap (`evidence/gpu.txt`) |
| Engine | vLLM 0.29.0, torch 2.13.0+cu130, Turing clamp applied (`evidence/packages.txt`) |
| Flags (both) | `--dtype float16 --kv-cache-dtype auto --gpu-memory-utilization 0.9 --max-model-len 16384 --max-num-seqs 8`, TP=1 |
| bf16 | `google/gemma-4-E2B-it` — fp16 cuBLAS linears |
| QAT | `google/gemma-4-E2B-it-qat-w4a16-ct` — int4 symmetric, group 32, `MarlinLinearKernel` |
| Harness | `vllm bench serve`, random dataset, output 128, `--ignore-eos --temperature 0 --num-warmups 2` (`sweep.sh`) |
| Grid | input {512, 4096} x concurrency {1, 4, 8, 16}, 3 repeats, unique seed per cell and repeat, identical seeds across builds |

Coverage: **16 cells measured, 0 failed, 0 infeasible**; 48 bench runs, 0 failed
requests. Worst-cell cv of output throughput 2.0%. `results.csv` is produced by
`aggregate.py` from the raw `bf16/*.json` and `qat/*.json`.

## Results (mean of 3 repeats)

| input | c | bf16 out tok/s | QAT out tok/s | bf16 per-stream | QAT per-stream | bf16 TTFT | QAT TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 37.04 | **62.27** | 40.44 | **72.31** | 300 ms | 316 ms |
| 512 | 4 | 109.05 | **157.49** | 32.25 | **54.50** | 791 ms | 937 ms |
| 512 | 8 | 164.62 | **215.91** | 25.49 | **37.83** | 1,212 ms | 1,301 ms |
| 512 | 16 | 164.36 | **213.79** | 24.97 | **34.69** | 7,270 ms | 5,903 ms |
| 4096 | 1 | 12.11 | 14.00 | 30.00 | 46.18 | 7,199 ms | 7,243 ms |
| 4096 | 4 | 15.16 | 15.82 | 6.40 | 7.23 | 15,304 ms | 15,391 ms |
| 4096 | 8 | 15.66 | 15.95 | 2.53 | 2.59 | 16,779 ms | 16,879 ms |
| 4096 | 16 | 15.46 | 15.75 | 2.42 | 2.48 | 81,551 ms | 80,119 ms |

Per-stream is 1000 / median TPOT. c=16 exceeds `--max-num-seqs 8`, so its extra
requests queue: throughput plateaus at the c=8 value and TTFT absorbs the wait.

## Memory

| | bf16 | QAT | source |
| --- | ---: | ---: | --- |
| Model loading | 9.8 GiB | 8.02 GiB | `evidence/engine-startup-*.log` |
| GPU KV cache | 315,974 tokens | 519,568 tokens | same |
| Max concurrency at 16,384 tokens | 19.29x | 31.71x | same |

**GPU capacity never binds at this serving shape, for either build.** Eight
sequences of 4,224 tokens need 33,792 KV tokens (ARITHMETIC), about a tenth of
bf16's pool. QAT's larger pool matters only if `MAX_NUM_SEQS` is raised.

## Why QAT is faster: bytes per token

ARITHMETIC from the safetensors headers (`evidence/bytes_per_token.py`,
`evidence/bytes-per-token.txt`). A text decode step reads the decoder layers and
the tied embedding (the LM head); the per-layer-embedding table is a row lookup
and the vision/audio towers are idle.

| | bytes read per token | bound at 277.0 GB/s |
| --- | ---: | ---: |
| bf16 (run as fp16) | 4.597 GB | 60.3 tok/s |
| QAT | 1.862 GB | 148.8 tok/s |

277.0 GB/s is the T4's measured streaming read from `@HARDWARE.md`, not a
measurement from this run. Measured per-stream decode at 512/c=1 reaches 67% of
the bf16 bound (40.44 / 60.3) and 49% of the QAT bound (72.31 / 148.8). During a
bf16 decode the GPU sat at 100% utilization, pinned at its 70 W cap with the SW
power-cap throttle active and the SM clock at 1275-1290 MHz of 1590
(`evidence/bringup.txt` §6), so the shortfall is not the host CPU.

## Prefill is the T4's weak point, for both builds

TTFT at c=1 goes from 300 ms at 512 tokens to 7,199 ms at 4096 — **24x the time
for 8x the tokens** (bf16; QAT 316 → 7,243 ms, 23x). Prefix caching may make
these TTFTs *optimistic* — see Scope.

The cause is **not established.** It is not the Marlin kernel, since bf16 is
equally slow. The superlinear growth points at attention; the candidate is the
Turing shared-memory clamp (`patch_triton_turing.py`), which on SM 7.5 shrinks
the Triton attention tiles and sets `num_stages=1` in every attention layer,
including the 512-wide global layers whose cost grows with the square of the
prompt. That is an INFERENCE, not a profile.

## Against the g4dn twin

`gpu-vllm-g4dn-2b` measured bf16 at 512/128 on the same T4 part on 2026-08-30:
42.36 tok/s at c=1 and 242.47 tok/s at c=8. **Not a controlled comparison:** it
ran vLLM 0.28.0 in the published docker image with its own harness (English
filler sized via `/tokenize`) on a 4 vCPU host, where this ran vLLM 0.29.0 under
`vllm bench serve` with random tokens. Its c=8 figure is 47% above this run's
bf16 164.62; which of those differences accounts for it is unknown.

## Scope

One T4 on one GCE VM, one engine version, three repeats per cell, random-token
prompts. Output length is fixed at 128 with `--ignore-eos`. QAT quality was
spot-checked only (`evidence/bringup.txt` §7), not evaluated.

**The prefix cache was not fully defeated.** Every cell had a unique seed, yet the
engine logged prefix-cache hit rates averaging about 8% and peaking at 53% (QAT)
and 62% (bf16) in individual 10-second windows (`evidence/bringup.txt` §8). The
source is not identified. Both builds ran identical seeds, so the QAT-vs-bf16
ratios stand; absolute TTFT and prefill-bound throughput may be understated.
Re-run with `--no-enable-prefix-caching` before quoting prefill figures as the
T4's own.
