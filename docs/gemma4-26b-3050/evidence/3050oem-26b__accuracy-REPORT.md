# 2026-10-18 — answer quality: Gemma 4 26B A4B QAT on the RTX 3050 OEM, with and without the drafter

llama.cpp `7182ef71e`, host `gpu`. Target: Google's `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`,
unmodified. Drafter: `xbill9/gemma-4-26B-A4B-it-qat-assistant-q4_0-exact-gguf` at 2 tokens.

## Result

**The 26B scores level with the 12B sister on GSM8K and BFCL, and the drafter changes neither
score.** GSM8K 0.965 with the drafter and 0.962 without; BFCL 0.945 both. Against the 12B at its
own serving setting the paired differences are +0.004 (GSM8K) and −0.005 (BFCL), both ranges
including zero. The 26B takes 1.76x the 12B's time for the same GSM8K run (12,346 s against
7,032 s, both with their drafters).

## Method (`run.sh`)

The rig's serving setting: `-ngl 99 -ncmoe 21 -c 131072 --parallel 1 -ctk q4_0 -ctv q4_0 -fa 1
-t 8 -tb 12 --reasoning off`, prompt cache at the server default. One arm with the drafter
(`--spec-type draft-mtp -ngld all --spec-draft-n-max 2`), one without. Each arm ran BFCL simple
(400) then GSM8K (1,319, up to 2048 tokens), one request at a time, through
`jev-tpu-v5e1/gen_eval.py` (greedy, scored in code). The 12B sister's method
(`local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-accuracy-12b-3050oem`).
`compare.py` pairs the arms by question with a 95% bootstrap range; its output is `compare.txt`.

## Scores

| | GSM8K | BFCL simple |
|---|---:|---:|
| 26B with drafter | **0.9651** (1273, 2 cut at 2048) | **0.9450** (378) |
| 26B without | 0.9621 (1269, 1 cut) | 0.9450 (378) |
| drafter − without | +0.0030 [−0.0023, +0.0083] | +0.0000 [−0.0100, +0.0100] |
| answers that differ | 12 (+8 / −4) | 4 (+2 / −2) |
| word-for-word identical replies | 160 of 1,319 | 392 of 400 |
| time, without / with drafter | 1.35x (16,663 / 12,346 s) | 1.11x (1,229 / 1,104 s) |

GSM8K generated about 426k tokens per arm: 34.5 tok/s with the drafter and 25.5 without,
counting prompt reading.

## Against the 12B (`local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem`)

The 12B at its serving setting: q4_0 cache, `-c 81920`, exact Q4_0 12B drafter / none, same
binary, card and harness; context size differs.

| | 12B | 26B − 12B (paired) |
|---|---:|---|
| GSM8K, drafter | 0.9613 | +0.0038 [−0.0045, +0.0129] (+18 / −13) |
| GSM8K, none | 0.9621 | +0.0000 [−0.0083, +0.0091] (+17 / −17) |
| BFCL, drafter | 0.9500 | −0.0050 [−0.0250, +0.0150] (+7 / −9) |
| BFCL, none | 0.9500 | −0.0050 [−0.0250, +0.0150] (+8 / −10) |

## Not measured

The q4_0 cache against f16 on the 26B (both arms here use q4_0); answer quality with a long
conversation held; Nimble suite.
