# Thinking and answer tokens, wave 2

Counted by `tokens.py` with Gemma 4 E4B's tokenizer.

## gsm8k

| arm | thinking | mean thinking tokens | mean answer tokens | median answer tokens | generated tokens, on vs off: Wilcoxon p | problems with more tokens when on |
|---|---|---|---|---|---|---|
| repack-none | off | 0 | 342 | 305 |  |  |
| repack-none | on | 514 | 264 | 247 | 4.46e-215 | 1309/1319 |
| repack-q4x | off | 0 | 340 | 305 |  |  |
| repack-q4x | on | 514 | 264 | 246 | 1.20e-216 | 1313/1319 |
| google-none | off | 0 | 344 | 308 |  |  |
| google-none | on | 505 | 263 | 249 | 6.32e-217 | 1310/1319 |
| google-q4x | off | 0 | 344 | 309 |  |  |
| google-q4x | on | 506 | 263 | 248 | 1.02e-214 | 1311/1319 |

## aime2025

| arm | thinking | mean thinking tokens | mean answer tokens | median answer tokens | generated tokens, on vs off: Wilcoxon p | problems with more tokens when on |
|---|---|---|---|---|---|---|
| repack-none | off | 0 | 3582 | 3274 |  |  |
| repack-none | on | 3843 | 1336 | 1138 | 2.51e-04 | 26/30 |
| repack-q4x | off | 0 | 3983 | 3505 |  |  |
| repack-q4x | on | 3760 | 1200 | 1205 | 3.74e-03 | 25/30 |
| google-none | off | 0 | 3969 | 3784 |  |  |
| google-none | on | 3947 | 1138 | 1099 | 5.38e-03 | 24/30 |
| google-q4x | off | 0 | 4174 | 3805 |  |  |
| google-q4x | on | 4288 | 1215 | 1185 | 2.19e-03 | 23/30 |

## Control

completion_tokens minus counted tokens, on replies that stopped normally: thinking off: 1 on 5393 replies, thinking off: 2 on 3 replies, thinking on: 5 on 5393 replies, thinking on: 6 on 3 replies

