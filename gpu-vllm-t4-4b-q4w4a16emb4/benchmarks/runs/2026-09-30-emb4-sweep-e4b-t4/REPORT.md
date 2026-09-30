# 2026-09-30 — Gemma 4 E4B with int4 embeddings (`-emb4`), concurrency sweep on one Tesla T4

**E4B `-emb4` runs at 0.53-0.56x the E2B `-emb4` build's output throughput at
512-token prompts, and 0.74-0.77x at 4096.** It peaks at 131 tok/s (512 in, c>=8)
against E2B's 240; at 4096-token prompts both are prefill-bound on this T4 and the
gap narrows. Same host, GPU, engine, flags, harness and prompts as the E2B run.

## Setup

| | |
| --- | --- |
| Host | GCE `n1-standard-2` (2 vCPU, 7.80 GB RAM), `us-west2-b`, swapfile on `/opt1` |
| GPU | Tesla T4, SM 7.5, 15360 MiB, driver 615.71.09, 70 W cap (`evidence/setup.txt`) |
| Engine | vLLM 0.29.0, torch 2.13.0+cu130, transformers 5.17.0, triton 3.7.1, Turing clamp applied |
| Flags | `--dtype float16 --kv-cache-dtype auto --gpu-memory-utilization 0.9 --max-model-len 16384 --max-num-seqs 8 --language-model-only`, TP=1, **drafter off** (`SPECULATIVE_MODEL= ./vllm-t4 start`) |
| Model | `xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4`: Google's QAT weights, linears int4 group 32, PLE table + `embed_tokens` int4, untied int4 `lm_head`, text only |
| Harness | `vllm bench serve`, random dataset, output 128, `--ignore-eos --temperature 0 --num-warmups 2` (`sweep.sh`, verbatim from the E2B run) |
| Grid | input {512, 4096} x concurrency {1, 4, 8, 16}, 3 repeats, **the same seeds as the E2B emb4 run**, so every cell sends the same prompts |

Coverage: **8 cells measured, 0 failed, 0 infeasible**; 24 bench runs, 0 failed
requests. Worst-cell cv of output throughput 1.4%. `results.csv` is produced by
`aggregate.py` from the raw `emb4/*.json`.

The drafter is off because a random-prompt sweep measures nothing useful about draft
acceptance. `evidence/cold-start-drafter.txt` records the drafter-on start for
reference; it is not the configuration measured.

## Results (mean of 3 repeats)

E2B emb4 is `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` from
[`gpu-vllm-t4-2b-w4a16/benchmarks/runs/2026-09-29-emb4-sweep-t4`](../../../../gpu-vllm-t4-2b-w4a16/benchmarks/runs/2026-09-29-emb4-sweep-t4/REPORT.md).

| input | c | E2B out tok/s | E4B out tok/s | E4B / E2B | E2B per-stream | E4B per-stream | E2B TTFT | E4B TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 85.28 | **45.30** | 0.53x | 99.94 | 51.4 | 252 ms | 401 ms |
| 512 | 4 | 183.08 | **102.9** | 0.56x | 67.71 | 36.1 | 971 ms | 1,548 ms |
| 512 | 8 | 239.64 | **129.9** | 0.54x | 44.68 | 23.1 | 1,394 ms | 2,238 ms |
| 512 | 16 | 237.08 | **130.8** | 0.55x | 40.05 | 20.6 | 5,369 ms | 9,534 ms |
| 4096 | 1 | 15.03 | **11.1** | 0.74x | 56.67 | 32.9 | 7,138 ms | 8,640 ms |
| 4096 | 4 | 16.36 | **12.5** | 0.76x | 7.35 | 5.4 | 15,020 ms | 18,689 ms |
| 4096 | 8 | 16.38 | **12.6** | 0.77x | 2.66 | 2.0 | 16,479 ms | 20,513 ms |
| 4096 | 16 | 16.17 | **12.4** | 0.77x | 2.55 | 1.9 | 78,074 ms | 100,993 ms |

Per-stream is 1000 / median TPOT. c=16 exceeds `--max-num-seqs 8`, so its extra
requests queue: throughput plateaus at the c=8 value and TTFT absorbs the wait.
Exact E4B values are in `results.csv`.

## Reading it

- **Decode-bound cells cost E4B about half.** At 512-token prompts E4B delivers
  0.53-0.56x E2B at every concurrency. Its loaded weights are 1.6x E2B's (4.54 against
  2.86 GiB), which alone would predict about 0.63x; the rest is not accounted for here.
- **Prefill-bound cells cost it a quarter.** At 4096-token prompts both builds spend most
  of each request on the prompt, and E4B's 8.6 s to first token at c=1 against E2B's
  7.1 s is a smaller ratio than its decode cost.
- **Both plateau at c=8**, the `--max-num-seqs` limit, not the KV pool.

## One cell, one repeat: `c4-in4096` rep 2

Its median TPOT is 155 ms against 200 ms in reps 1 and 3, while its throughput
(12.46 tok/s) matches theirs (12.52, 12.48) and its median TTFT is higher (21.1 s
against 17.4 and 17.5). The same wall time was split differently between prefill and
decode, not spent faster. It was the first cell on the restarted server (below), which
is the likeliest cause. The mean TPOT in the table (185 ms) includes it.

## The run was interrupted and resumed

`sweep.sh` died at 14:31 during `c4-in4096` rep 2, with 4 of 16 requests sent (no
error in its log; it is kept as `evidence/aborted-c4-in4096-out128.rep2.log`).
`sweep.out` is that first pass: rep 1 and six rep-2 cells.

The aborted cell had already put seed 240964's prompts into the prefix cache, so
rerunning it on the same server would have measured cache hits. **The server was
restarted with the same flags before resuming** (`evidence/restart-before-resume.txt`:
same args, same 310,499-token KV cache), and `resume.sh` — the same grid, seeds and
flags as `sweep.sh`, skipping cells that already had a result — ran the remaining ten
cells. Its output is `resume.out`. Every rep-1 cell ran on the first server; every
rep-3 cell on the second; rep 2 is split. Repeats agree within 1.4% across the restart.

## Memory

| | E2B emb4 | E4B emb4 | source |
| --- | ---: | ---: | --- |
| Model loading | 2.86 GiB | **4.54 GiB** | each run's `evidence/setup.txt` |
| KV cache | 1,099,587 tokens | **310,499 tokens** | same |

At this serving shape (8 sequences, 16,384-token limit) neither KV pool binds.

## Files

`sweep.sh` and `resume.sh` (the harness), `aggregate.py`, `results.csv`,
`emb4/*.json` (raw `--save-result`), `emb4/*.log`, `sweep.out`, `resume.out`,
`evidence/`. `build_report.py` writes the schema report,
`../../reports/2026-09-30-emb4-sweep-e4b-t4.json`, from `emb4/`.
