# 2026-11-04 — mistral.rs on the 3050 against llama.cpp, same GGUF, with and without drafters

mistral.rs v0.9.5 (CUDA + flash-attn, sm_86) and llama.cpp `7182ef71e` serving the same exact QAT Q4_0 E2B GGUF on the RTX 3050 OEM, through the runner of `local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem` (`engines.py`, `summarize.py`): six chat prompts, greedy, thinking off, up to 256 tokens, no prompt cache, back to back on one card. llama.cpp is the same-day reference: 108.4 tok/s against 108.8 the day before.

| arm | tok/s, one request | vs llama.cpp | decode tok/s | first token | 4 at once (total) | 4 at once vs one | GPU memory |
|---|---|---|---|---|---|---|---|
| mistral.rs | 118.9 | 1.096 | 119.7 | 22 ms | 316.8 | 2.66 | 4,531 MiB |
| mistral.rs + drafter, 1 token | 85.8 | 0.791 | 86.0 | 21 ms | 141.9 | 1.66 | 4,587 MiB |
| mistral.rs + drafter, 2 tokens | 75.3 | 0.695 | 75.3 | 22 ms | 157.5 | 2.09 | 4,623 MiB |
| mistral.rs + drafter, 3 tokens | 64.4 | 0.594 | 65.9 | 19 ms | 113.1 | 1.76 | 4,623 MiB |
| llama.cpp | 108.4 | 1.000 | 107.8 | 41 ms | 271.2 | 2.50 | 2,809 MiB |
| llama.cpp + Q4_0 drafter, 2 tokens | 146.0 | 1.346 | 148.3 | 43 ms | 279.7 | 1.92 | 2,885 MiB |

Drafted tokens accepted on mistral.rs (its server log, last line of each arm): 61.6% at 1 token, 49.4% at 2, 42.7% at 3.

## Findings

- **Without drafters mistral.rs is ahead of llama.cpp on the same file**: 1.10x for one request, 1.17x with four at once, and its first token comes in about half the time (22 against 41 ms).
- **Google's drafter slows mistral.rs at every draft length** (0.79x to 0.59x llama.cpp; 0.72x to 0.54x its own speed without it), at acceptance close to llama.cpp's and vLLM's for the same lengths (about 0.60 / 0.47 / 0.37, `2026-11-03-drafters-3050oem`). The cost of a drafted step, not acceptance, decides it here. llama.cpp with its drafter stays fastest of the two for one request (146.0).
- **mistral.rs holds 1,722 MiB more than llama.cpp** (4,531 against 2,809): it keeps the per-layer table and the image and audio encoders on the card.

## Settings

mistral.rs from `local-mistralrs-3050oem-2b-q4_0/tpu.env`: `--max-seqs 4 --max-model-len 8192 --kv-cache 8192tokens --prefix-cache-n 0`; with the drafter (`--mtp-model`, Google's `gemma-4-E2B-it-qat-q4_0-unquantized-assistant`, bf16 safetensors, `--mtp-n-predict N`) the cache is 4,096 tokens, because 8,192 does not load beside it. llama.cpp: `-c 8192 --parallel 4`, f16 cache, flash attention, exact Q4_0 drafter at 2 tokens.

mistral.rs rejects a body carrying both `max_tokens` and `max_completion_tokens` (it reads the second as an alias of the first), so the runner sends only `max_tokens` to it.

Files: `results.jsonl`, `SUMMARY.txt`, `server-*.log`, `run.out`.
