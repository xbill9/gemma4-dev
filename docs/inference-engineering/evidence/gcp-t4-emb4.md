# 2026-09-29 — Gemma 4 E2B with int4 embeddings (`-emb4`), concurrency sweep on one Tesla T4

**The `-emb4` build is the fastest of the three E2B builds measured on this T4 at every
cell: +37% output throughput over Google's QAT export at 512-token prompts and c=1
(2.30x bf16), narrowing to +11% at c=8.** At 4096-token prompts the gain is 3-7%,
because the T4's prefill sets the rate there and int4 embeddings do not touch prefill.

## Setup

| | |
| --- | --- |
| Host | GCE `n1-standard-2` (2 vCPU, 7.80 GB RAM), `us-west2-b`, 16 GB swapfile on `/opt1` |
| GPU | Tesla T4, SM 7.5, 15360 MiB, driver 615.71.09, 70 W cap (`evidence/setup.txt`) |
| Engine | vLLM 0.29.0, torch 2.13.0+cu130, transformers 5.17.0, triton 3.7.1, Turing clamp applied |
| Flags | `--dtype float16 --kv-cache-dtype auto --gpu-memory-utilization 0.9 --max-model-len 16384 --max-num-seqs 8 --language-model-only`, TP=1 |
| Model | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`: Google's QAT weights, linears int4 group 32 (`MarlinLinearKernel`), PLE table + `embed_tokens` int4 (`CompressedTensorsEmbeddingWNA16Int`), untied int4 `lm_head`, fp16 scales, text only |
| Harness | `vllm bench serve`, random dataset, output 128, `--ignore-eos --temperature 0 --num-warmups 2` (`sweep.sh`, copied verbatim from `gpu-vllm-t4-2b`'s 2026-09-18 run) |
| Grid | input {512, 4096} x concurrency {1, 4, 8, 16}, 3 repeats, **the same seeds as the 2026-09-18 run**, so every cell sends the same prompts as that run's bf16 and qat cells |

Coverage: **8 cells measured, 0 failed, 0 infeasible**; 24 bench runs, 0 failed
requests. Worst-cell cv of output throughput 2.5% (the 2026-09-18 run: 2.0%).
`results.csv` is produced by `aggregate.py` from the raw `emb4/*.json`.

## Results (mean of 3 repeats)

bf16 = `google/gemma-4-E2B-it`, QAT = `google/gemma-4-E2B-it-qat-w4a16-ct`, both from
[`gpu-vllm-t4-2b/benchmarks/runs/2026-09-18-qat-vs-bf16-t4`](../../../../gpu-vllm-t4-2b/benchmarks/runs/2026-09-18-qat-vs-bf16-t4/REPORT.md).
emb4 is this run.

| input | c | bf16 out tok/s | QAT out tok/s | emb4 out tok/s | emb4 vs QAT | emb4 vs bf16 | bf16 per-stream | QAT per-stream | emb4 per-stream | QAT TTFT | emb4 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 37.04 | 62.27 | **85.28** | +37% | 2.30x | 40.44 | 72.31 | **99.94** | 316 ms | 252 ms |
| 512 | 4 | 109.05 | 157.49 | **183.08** | +16% | 1.68x | 32.25 | 54.50 | **67.71** | 937 ms | 971 ms |
| 512 | 8 | 164.62 | 215.91 | **239.64** | +11% | 1.46x | 25.49 | 37.83 | **44.68** | 1,301 ms | 1,394 ms |
| 512 | 16 | 164.36 | 213.79 | **237.08** | +11% | 1.44x | 24.97 | 34.69 | **40.05** | 5,903 ms | 5,369 ms |
| 4096 | 1 | 12.11 | 14.00 | **15.03** | +7% | 1.24x | 30.00 | 46.18 | **56.67** | 7,243 ms | 7,138 ms |
| 4096 | 4 | 15.16 | 15.82 | **16.36** | +3% | 1.08x | 6.40 | 7.23 | **7.35** | 15,391 ms | 15,020 ms |
| 4096 | 8 | 15.66 | 15.95 | **16.38** | +3% | 1.05x | 2.53 | 2.59 | **2.66** | 16,879 ms | 16,479 ms |
| 4096 | 16 | 15.46 | 15.75 | **16.17** | +3% | 1.05x | 2.42 | 2.48 | **2.55** | 80,119 ms | 78,074 ms |

Per-stream is 1000 / median TPOT. c=16 exceeds `--max-num-seqs 8`, so its extra
requests queue: throughput plateaus at the c=8 value and TTFT absorbs the wait.

## Reading it

- **Short prompts gain most, and the gain shrinks with concurrency.** At c=1 each decode
  step reads every weight once for one token, so the int4 `lm_head` — 0.21 GiB instead of
  the fp16 0.75 GiB tied table, the one full-vocab matmul per token — is a large share of
  the step. At c=8 the same weight read serves eight tokens and other work grows, so the
  saving is a smaller share: +37% at c=1, +16% at c=4, +11% at c=8.
- **Long prompts are prefill-bound and barely move.** A 4096-token prompt takes ~7.1 s to
  first token on all three builds; that is compute on the prompt, which this build does not
  change. The 3% at c>=4 is small but above this run's noise (cv 0.0-0.2% at 4096).
- **TTFT is unchanged within noise except at 512/c=1** (252 against 316 ms).

## What this comparison is and is not

It is **not a one-variable A/B**. emb4 differs from Google's QAT export in three ways at
once: the exact-grid repack (Google's export re-quantizes the QAT weights by min-max
round-to-nearest, `@QUANTIZATION.md`), int4 embeddings plus an int4 `lm_head`, and text
only. The 2026-09-18 runs also served the multimodal checkpoints without
`--language-model-only`, which changes memory and startup but should not change text
throughput. Same host, GPU, vLLM version, flags, harness and prompts; run 11 days apart.
The single-stream check on 2026-09-29 isolates the embedding step on its own: the
bf16-embedding `-text` build 81.6 tok/s, `-emb4` 109.7 tok/s at c=1
(`../../../evidence/2026-09-29-embed-int4.txt`).

## Memory

| | bf16 | QAT | emb4 | source |
| --- | ---: | ---: | ---: | --- |
| Model loading | 9.8 GiB | 8.02 GiB | **2.86 GiB** | bf16, QAT: the 2026-09-18 run's `evidence/engine-startup-*.log` (multimodal on); emb4: `evidence/setup.txt` |
| KV cache | 315,974 tokens | 519,568 tokens | **1,099,587 tokens** | same |

At this serving shape (8 sequences, 16,384-token limit) the KV pool never binds for any of
the three, so the memory saving shows up as headroom, not throughput.

## Files

`sweep.sh` (the harness), `aggregate.py` (labels discovered, otherwise as in the
2026-09-18 run), `results.csv`, `emb4/*.json` (raw `--save-result`), `emb4/*.log`,
`sweep.out`, `evidence/setup.txt`. `build_report.py` writes the schema report,
`../../reports/2026-09-29-emb4-sweep-t4.json`, from `emb4/` only.

## `w8a8/` — a partial run, not part of these results

`w8a8/` and `sweep-w8a8.out` are the same harness against
`xbill9/gemma-4-E2B-it-qat-w8a8-ct-text-emb4` (int8 weights and activations, the same
int4 embeddings), **rep 1 only, stopped by hand during 4096/c=4** because W8A8 was
slower than emb4 in every cell completed:

| input | c | W8A8 out tok/s (1 rep) | emb4 out tok/s (mean of 3) |
| ---: | ---: | ---: | ---: |
| 512 | 1 | 45.67 | 85.28 |
| 512 | 4 | 125.94 | 183.08 |
| 512 | 8 | 181.25 | 239.64 |
| 512 | 16 | 182.89 | 237.08 |
| 4096 | 1 | 12.94 | 15.03 |

One repeat, so the W8A8 column carries no spread. **Do not re-run `aggregate.py` as
it stands**: it discovers every label directory, so it would now add single-repeat
`w8a8` rows to `results.csv` beside the three-repeat `emb4` ones.
