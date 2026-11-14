# 2026-10-19 — answer quality: Gemma 4 31B QAT on the RTX 3050 OEM, with and without the drafter

llama.cpp `7182ef71e`, host `gpu`. Target: Google's `google/gemma-4-31B-it-qat-q4_0-gguf`,
unmodified. Drafter: `xbill9/gemma-4-31B-it-qat-assistant-q4_0-exact-gguf` at 1 token.

GSM8K with the drafter is the complete run (1,319). GSM8K without the drafter is a 263-question
sample: it adds nothing to the drafter finding the full BFCL arm and these 263 already establish, so
it was not carried to the end. `compare.txt` is the current figures; rerun `compare.py` to reproduce.

## Method (`run.sh`, `resumable_eval.py`)

The rig's serving setting: `-ngl 20 -c 32768 --parallel 1 -ctk q4_0 -ctv q4_0 -fa 1 -t 8 -tb 12
--reasoning off`, prompt cache at the server default; one arm with the drafter
(`--spec-type draft-mtp -ngld all --spec-draft-n-max 1`), one without. BFCL simple (400) both arms,
then GSM8K (1,319, up to 2048 tokens) with the drafter, then without; one request at a time.
Requests and scoring are `jev-tpu-v5e1/gen_eval.py`'s own functions, called per question by
`resumable_eval.py`, which writes each record as it lands so a multi-day run survives a restart;
its time figure is the sum of request times. The 26B sister's method
(`local-llamacpp-3050oem-26b-q4_0/benchmarks/runs/2026-10-18-accuracy-26b-3050oem`).

## The drafter matches the model it drafts for

Across both tasks the drafter changes wording but not correctness. BFCL simple: 397 of 400 replies
word-for-word identical, one answer differs (wrong → right), +0.0025 [+0.0000, +0.0075]. GSM8K on
the 263 paired questions: every one scores the same with the drafter and without, 0 disagreements,
+0.0000 [+0.0000, +0.0000], with 96 of 263 word-for-word identical — the rest reword the reasoning
and reach the same answer. The drafter runs 1.39x faster on BFCL and 1.73x on GSM8K
(sum of request times, same questions).

## GSM8K (1,319 with the drafter)

| | 31B | 26B | 12B |
|---|---:|---:|---:|
| accuracy | **0.9735** (1284) | 0.9651 | 0.9613 |

Paired on the full 1,319: 31B − 26B +0.0083 [+0.0008, +0.0167] (+17 / −6), 31B − 12B +0.0121
[+0.0045, +0.0205] (+22 / −6). The 31B leads both sisters, interval clear of zero. No truncations.

## BFCL simple (400, both arms)

| | 31B | 26B | 12B |
|---|---:|---:|---:|
| with drafter | **0.9600** (384) | 0.9450 | 0.9500 |
| without | 0.9575 (383) | 0.9450 | 0.9500 |

31B − 26B: +0.0150 [−0.0050, +0.0350] with the drafter, +0.0125 [−0.0075, +0.0325] without.
31B − 12B: +0.0100 [−0.0075, +0.0275] and +0.0075 [−0.0100, +0.0250]. Level with both sisters;
intervals span zero.
