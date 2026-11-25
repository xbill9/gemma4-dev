# Thinking and answer tokens, E2B on the GTX 1650 Ti

Counted by `tokens.py` with Gemma 4 E2B's tokenizer.

## gsm8k

| arm | thinking | mean thinking tokens | mean answer tokens | median answer tokens | generated tokens, on vs off: Wilcoxon p | problems with more tokens when on |
|---|---|---|---|---|---|---|
| repack-none | off | 0 | 424 | 344 |  |  |
| repack-none | on | 601 | 278 | 262 | 1.88e-209 | 1301/1319 |
| repack-q4x | off | 0 | 404 | 343 |  |  |
| repack-q4x | on | 613 | 280 | 264 | 5.71e-210 | 1303/1319 |
| google-none | off | 0 | 408 | 347 |  |  |
| google-none | on | 633 | 272 | 256 | 3.30e-213 | 1300/1319 |
| google-q4x | off | 0 | 427 | 347 |  |  |
| google-q4x | on | 631 | 273 | 258 | 4.03e-211 | 1299/1319 |

## aime2025

| arm | thinking | mean thinking tokens | mean answer tokens | median answer tokens | generated tokens, on vs off: Wilcoxon p | problems with more tokens when on |
|---|---|---|---|---|---|---|
| repack-none | off | 0 | 5425 | 3646 |  |  |
| repack-none | on | 7075 | 1016 | 906 | 3.06e-04 | 28/30 |
| repack-q4x | off | 0 | 6497 | 3066 |  |  |
| repack-q4x | on | 9050 | 1735 | 986 | 5.56e-05 | 26/30 |
| google-none | off | 0 | 6391 | 3708 |  |  |
| google-none | on | 9113 | 919 | 996 | 1.04e-03 | 26/30 |
| google-q4x | off | 0 | 5553 | 3649 |  |  |
| google-q4x | on | 8301 | 1058 | 1041 | 4.17e-05 | 28/30 |

## Control

completion_tokens minus counted tokens, on replies that stopped normally: thinking off: 1 on 5380 replies, thinking off: 2 on 4 replies, thinking on: 5 on 5376 replies, thinking on: 6 on 6 replies, thinking on: 7 on 1 replies

