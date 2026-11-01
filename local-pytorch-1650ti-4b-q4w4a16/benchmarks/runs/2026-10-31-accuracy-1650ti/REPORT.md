# 2026-10-31 — Gemma 4 E4B int4 in PyTorch on the GTX 1650 Ti

GTX 1650 Ti Max-Q (TU117, 3,732 MiB usable), torch 2.13.0+cu130, transformers 5.16.1. Decoder linears from
`xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text`, vocabulary and per-layer table int4 from `-text-emb4`. llama.cpp
`7182ef7` with `gemma-4-E4B-it-q4_0-exact.gguf`, same card, same night. Engine scripts and raw output are in this
directory and in `../2026-10-31-{first-light,correctness,tune}-1650ti/`.

#### Memory

2,499 MiB of weights on the GPU, 660 MiB of K/V at 40,960 tokens, 3,304 MiB peak filled to 40,021 tokens. The per-layer
table (1.5 GiB in int4) stays in CPU RAM and one row is copied to a GPU buffer per decode step.

49,152 tokens loads and fails on the first long prompt read; 65,536 fails at load. llama-server loads `-c 56320`
(3,677 MiB) and not `-c 65536`; a filled 56,320 was not run, and `llama-bench -d 40320` failed to create its context.

#### Decode, one request

| Tokens held | Engine tok/s | llama.cpp tok/s | Ratio |
| ---: | ---: | ---: | ---: |
| short | 43.69 (41.66–45.01) | 39.57 | 1.10x |
| ~4k | 42.03 | 36.53 | 1.15x |
| ~16k | 38.37 | 32.17 | 1.19x |
| ~32k | 34.75 | 27.47 | 1.27x |
| ~40k | 33.25 | not run | |

Engine short: eight prompts, two passes, up to 128 tokens (`run-final.jsonl`). Engine at depth: one 128-token reply.
llama.cpp: `llama-bench -ngl 99 -fa 1 -t 6`, 128 tokens, 5 repeats short; `-n 64`, 2 repeats at depth.

**With its drafter llama.cpp is ahead at short context**: `llama-server`, the same eight prompts twice, 39.0 tok/s
without and 63.5 with the exact Q4_0 drafter at 2 tokens (`spec.jsonl`). The engine has no drafter.

#### Prompt reads

Engine, whole prompt from position 0: 139 / 121 / 103 / 97 tok/s to ~4k / 16k / 32k / 40k. llama-bench, 512 tokens read
at depth: 154 / 120 / 92 tok/s at 4k / 16k / 32k. The two are different measures. A short prompt costs the engine about
0.6 s, because each chunk unpacks every layer.

#### Correctness

Against stock transformers on `google/gemma-4-E4B-it-qat-q4_0-unquantized`, bf16 on the CPU: 9 of 10 replies
token-identical, including prompts of 631 and 2,401 tokens, past the 512-slot ring. The tenth splits at token 4 on two
candidates that stock bf16 scores equal at 3 decimals (−0.8335, `margin.jsonl`); the engine has them 0.047 apart.
Recall of one code from a document was right at all four depths (one question each).

#### Answers

Through `openai_server.py` at `--max-ctx 8192`, one request at a time, `gen_eval.py`, against llama.cpp without a drafter
(`local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-accuracy-1650ti`, `--parallel 8`, llama.cpp `77c4540ce`):

| Set | n | Engine | llama.cpp | Difference (95% range) | Only engine right / only llama.cpp right |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFCL simple | 400 | 0.9100 | 0.9100 | 0.0000 (−0.0075 to +0.0075) | 1 / 1 |
| GSM8K | 1,319 | 0.9325 | 0.9318 | +0.0008 (−0.0061 to +0.0076) | 12 / 11 |

Level. BFCL took 934 s, GSM8K 11,470 s, 2 GSM8K replies truncated, 0 errors.

#### Not measured

Thinking on; more than one request at a time (the server serves one); llama.cpp decode at 40k; recall over enough prompts
to give a rate; `compare.py` pairs against a different llama.cpp commit and `--parallel`.
