# 2026-11-19 — Gemini 4 Neon, GSM8K and BFCL simple, lowest thinking level

The settings of the 2026-11-14 seven-model run (`gemini-api-google-*/benchmarks/runs/2026-11-14-accuracy-api`):
same endpoint, `gen_eval.py`, questions, token limits (GSM8K 2,048, BFCL 512), temperature 0, concurrency 8,
timeout and retries; `reasoning_effort: none`, which Neon accepts and runs with thinking on every problem.
Neon ran alone, five days later. `compare.py` writes `compare.txt`.

| | GSM8K | BFCL simple | Both tasks | Thinking per GSM8K problem |
| :--- | ---: | ---: | ---: | ---: |
| Gemini 4 Neon | 1,295 / 1,319, 0.9818 | 390 / 400, 0.9750 | $10.06 | 252.2 (all 1,319) |

Paired against the 11-14 models: GSM8K above Gemma 4 31B, 26B and Flash-Lite, level with 3.8 Flash and the other
three Gemini 4 models; BFCL above Flash-Lite, 3.8 Flash, Hydrogen and Carbon, level with Gemma 4 31B, 26B and Argon.
Median 13.98 s per GSM8K request with the connection to itself, so not comparable with the 11-14 times.
