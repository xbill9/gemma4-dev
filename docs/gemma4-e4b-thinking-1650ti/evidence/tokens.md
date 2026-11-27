# Thinking and answer tokens, E4B on the GTX 1650 Ti

Counted by `tokens.py` with Gemma 4 E4B's tokenizer.

## gsm8k

| arm | thinking | mean thinking tokens | mean answer tokens | median answer tokens | generated tokens, on vs off: Wilcoxon p | problems with more tokens when on |
|---|---|---|---|---|---|---|
| repack-none | off | 0 | 342 | 302 |  |  |
| repack-none | on | 516 | 265 | 250 | 8.84e-214 | 1307/1319 |
| repack-q4x | off | 0 | 342 | 303 |  |  |
| repack-q4x | on | 511 | 263 | 248 | 2.92e-214 | 1313/1319 |
| google-none | off | 0 | 342 | 308 |  |  |
| google-none | on | 503 | 263 | 249 | 5.40e-213 | 1313/1319 |
| google-q4x | off | 0 | 346 | 307 |  |  |
| google-q4x | on | 508 | 262 | 248 | 3.48e-214 | 1309/1319 |

## aime2025

| arm | thinking | mean thinking tokens | mean answer tokens | median answer tokens | generated tokens, on vs off: Wilcoxon p | problems with more tokens when on |
|---|---|---|---|---|---|---|
| repack-none | off | 0 | 3494 | 3274 |  |  |
| repack-none | on | 3868 | 1207 | 1206 | 6.92e-06 | 26/30 |
| repack-q4x | off | 0 | 3698 | 3616 |  |  |
| repack-q4x | on | 3838 | 1270 | 1209 | 6.92e-06 | 26/30 |
| google-none | off | 0 | 3852 | 3728 |  |  |
| google-none | on | 4351 | 1178 | 1113 | 6.89e-05 | 26/30 |
| google-q4x | off | 0 | 4201 | 3775 |  |  |
| google-q4x | on | 3947 | 1218 | 1166 | 1.64e-02 | 21/30 |

## Control

completion_tokens minus counted tokens, on replies that stopped normally: thinking off: 1 on 5392 replies, thinking off: 2 on 4 replies, thinking on: 5 on 5393 replies, thinking on: 6 on 2 replies, thinking on: 7 on 1 replies

