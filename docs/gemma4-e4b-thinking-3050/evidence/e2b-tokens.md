# Thinking and answer tokens, wave 1

Counted by `tokens.py` with Gemma 4 E2B's tokenizer.

## gsm8k

| arm | thinking | mean thinking tokens | mean answer tokens | median answer tokens | generated tokens, on vs off: Wilcoxon p | problems with more tokens when on |
|---|---|---|---|---|---|---|
| repack-none | off | 0 | 399 | 344 |  |  |
| repack-none | on | 606 | 280 | 263 | 1.79e-214 | 1307/1319 |
| repack-q4x | off | 0 | 401 | 344 |  |  |
| repack-q4x | on | 601 | 279 | 263 | 1.75e-211 | 1302/1319 |
| google-none | off | 0 | 408 | 345 |  |  |
| google-none | on | 615 | 270 | 254 | 8.21e-211 | 1303/1319 |
| google-q4x | off | 0 | 407 | 345 |  |  |
| google-q4x | on | 602 | 271 | 253 | 1.00e-211 | 1301/1319 |

## aime2025

| arm | thinking | mean thinking tokens | mean answer tokens | median answer tokens | generated tokens, on vs off: Wilcoxon p | problems with more tokens when on |
|---|---|---|---|---|---|---|
| repack-none | off | 0 | 6259 | 3278 |  |  |
| repack-none | on | 7668 | 1105 | 1014 | 7.73e-04 | 26/30 |
| repack-q4x | off | 0 | 5812 | 3588 |  |  |
| repack-q4x | on | 7926 | 963 | 929 | 1.13e-03 | 26/30 |
| google-none | off | 0 | 6445 | 3736 |  |  |
| google-none | on | 8290 | 1037 | 924 | 4.07e-04 | 26/30 |
| google-q4x | off | 0 | 6548 | 3598 |  |  |
| google-q4x | on | 10240 | 846 | 820 | 6.75e-05 | 26/30 |

## Control

completion_tokens minus counted tokens, on replies that stopped normally: thinking off: 1 on 5381 replies, thinking off: 2 on 4 replies, thinking on: 5 on 5378 replies, thinking on: 6 on 4 replies, thinking on: 7 on 1 replies

