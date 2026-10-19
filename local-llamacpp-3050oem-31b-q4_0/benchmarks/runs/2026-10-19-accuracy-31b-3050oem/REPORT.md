# 2026-10-19 — answer quality: Gemma 4 31B QAT on the RTX 3050 OEM, with and without the drafter

**In progress.** BFCL is complete for both arms; GSM8K with the drafter is running, GSM8K without
it follows. `compare.txt` is a snapshot taken at commit time; rerun `compare.py` for current figures.

llama.cpp `7182ef71e`, host `gpu`. Target: Google's `google/gemma-4-31B-it-qat-q4_0-gguf`,
unmodified. Drafter: `xbill9/gemma-4-31B-it-qat-assistant-q4_0-exact-gguf` at 1 token.

## Method (`run.sh`, `resumable_eval.py`)

The rig's serving setting: `-ngl 20 -c 32768 --parallel 1 -ctk q4_0 -ctv q4_0 -fa 1 -t 8 -tb 12
--reasoning off`, prompt cache at the server default; one arm with the drafter
(`--spec-type draft-mtp -ngld all --spec-draft-n-max 1`), one without. BFCL simple (400) both arms,
then GSM8K (1,319, up to 2048 tokens) with the drafter, then without; one request at a time.
Requests and scoring are `jev-tpu-v5e1/gen_eval.py`'s own functions, called per question by
`resumable_eval.py`, which writes each record as it lands so a multi-day run survives a restart;
its time figure is the sum of request times. The 26B sister's method
(`local-llamacpp-3050oem-26b-q4_0/benchmarks/runs/2026-10-18-accuracy-26b-3050oem`).

## BFCL simple (complete)

| | 31B | 26B | 12B |
|---|---:|---:|---:|
| with drafter | **0.9600** (384) | 0.9450 | 0.9500 |
| without | 0.9575 (383) | 0.9450 | 0.9500 |

Drafter − without: +0.0025 [+0.0000, +0.0075], one answer differs (wrong → right), 397 of 400
replies word-for-word identical, 1.39x faster (7,571 / 5,428 s). 31B − 26B: +0.0150
[−0.0050, +0.0350] with the drafter, +0.0125 [−0.0075, +0.0325] without; 31B − 12B: +0.0100
[−0.0075, +0.0275] and +0.0075 [−0.0100, +0.0250]. Level with both sisters.

## GSM8K (running)

Partial; see `compare.txt` for the questions done at commit time and their paired differences
against the 26B and 12B on the same questions. About 59 s per question with the drafter.
