# Function calling on the 3050: results

BFCL v3 simple (400), multiple (200), irrelevance (240); exact Q4_0, llama.cpp d1044bc82, no drafter, one request at a time. `sampled` = temperature 1.0, top_k 64, top_p 0.95, seed 1. `thought` = replies carrying thinking text. Hours = sum of reply times.

## Per cell

| model | task | setting | right | score | 95% range | thought | cut at 8,192 | mean tokens | hours |
|---|---|---|---|---|---|---|---|---|---|
| e2b | simple | greedy thinkoff | 369/400 | 0.9225 | 0.8921-0.9449 | 0/400 | 0 | 32 | 0.040 |
| e2b | multiple | greedy thinkoff | 180/200 | 0.9000 | 0.8506-0.9343 | 0/200 | 0 | 32 | 0.024 |
| e2b | irrelevance | greedy thinkoff | 199/240 | 0.8292 | 0.7765-0.8715 | 0/240 | 0 | 211 | 0.133 |
| e2b | simple | sampled thinkoff | 355/400 | 0.8875 | 0.8528-0.9149 | 121/400 | 0 | 104 | 0.110 |
| e2b | multiple | sampled thinkoff | 170/200 | 0.8500 | 0.7939-0.8929 | 1/200 | 0 | 40 | 0.028 |
| e2b | irrelevance | sampled thinkoff | 177/240 | 0.7375 | 0.6784-0.7891 | 16/240 | 0 | 228 | 0.143 |
| e2b | simple | sampled thinkon | 360/400 | 0.9000 | 0.8667-0.9257 | 254/400 | 0 | 181 | 0.192 |
| e2b | multiple | sampled thinkon | 183/200 | 0.9150 | 0.8681-0.9463 | 193/200 | 0 | 272 | 0.145 |
| e2b | irrelevance | sampled thinkon | 208/240 | 0.8667 | 0.8178-0.9039 | 210/240 | 0 | 464 | 0.287 |
| e2b | simple | greedy thinkon | 362/400 | 0.9050 | 0.8723-0.9300 | 400/400 | 0 | 323 | 0.335 |
| e2b | multiple | greedy thinkon | 183/200 | 0.9150 | 0.8681-0.9463 | 200/200 | 0 | 348 | 0.184 |
| e2b | irrelevance | greedy thinkon | 212/240 | 0.8833 | 0.8366-0.9180 | 240/240 | 0 | 522 | 0.322 |
| e4b | simple | greedy thinkoff | 365/400 | 0.9125 | 0.8807-0.9364 | 0/400 | 0 | 32 | 0.075 |
| e4b | multiple | greedy thinkoff | 184/200 | 0.9200 | 0.8740-0.9502 | 0/200 | 0 | 32 | 0.043 |
| e4b | irrelevance | greedy thinkoff | 209/240 | 0.8708 | 0.8225-0.9075 | 0/240 | 0 | 232 | 0.272 |
| e4b | simple | sampled thinkoff | 339/400 | 0.8475 | 0.8090-0.8794 | 343/400 | 0 | 167 | 0.330 |
| e4b | multiple | sampled thinkoff | 164/200 | 0.8200 | 0.7609-0.8671 | 125/200 | 0 | 137 | 0.143 |
| e4b | irrelevance | sampled thinkoff | 195/240 | 0.8125 | 0.7583-0.8568 | 28/240 | 0 | 252 | 0.296 |
| e4b | simple | sampled thinkon | 366/400 | 0.9150 | 0.8836-0.9385 | 139/400 | 0 | 121 | 0.243 |
| e4b | multiple | sampled thinkon | 185/200 | 0.9250 | 0.8800-0.9540 | 35/200 | 0 | 75 | 0.087 |
| e4b | irrelevance | sampled thinkon | 210/240 | 0.8750 | 0.8272-0.9110 | 116/240 | 1 | 429 | 0.503 |
| e4b | simple | greedy thinkon | 366/400 | 0.9150 | 0.8836-0.9385 | 400/400 | 0 | 280 | 0.546 |
| e4b | multiple | greedy thinkon | 185/200 | 0.9250 | 0.8800-0.9540 | 200/200 | 0 | 341 | 0.370 |
| e4b | irrelevance | greedy thinkon | 219/240 | 0.9125 | 0.8699-0.9421 | 239/240 | 1 | 555 | 0.749 |
| 12b | simple | greedy thinkoff | 380/400 | 0.9500 | 0.9240-0.9674 | 0/400 | 0 | 32 | 0.162 |
| 12b | multiple | greedy thinkoff | 183/200 | 0.9150 | 0.8681-0.9463 | 0/200 | 0 | 32 | 0.094 |
| 12b | irrelevance | greedy thinkoff | 193/240 | 0.8042 | 0.7493-0.8494 | 0/240 | 0 | 230 | 0.578 |
| 12b | simple | sampled thinkoff | 374/400 | 0.9350 | 0.9065-0.9553 | 0/400 | 0 | 32 | 0.162 |
| 12b | multiple | sampled thinkoff | 178/200 | 0.8900 | 0.8391-0.9262 | 0/200 | 0 | 32 | 0.095 |
| 12b | irrelevance | sampled thinkoff | 188/240 | 0.7833 | 0.7270-0.8308 | 0/240 | 0 | 235 | 0.590 |
| 12b | simple | sampled thinkon | 378/400 | 0.9450 | 0.9181-0.9634 | 398/400 | 0 | 176 | 0.743 |
| 12b | multiple | sampled thinkon | 188/200 | 0.9400 | 0.8981-0.9653 | 200/200 | 0 | 179 | 0.393 |
| 12b | irrelevance | sampled thinkon | 185/240 | 0.7708 | 0.7136-0.8195 | 240/240 | 4 | 648 | 1.619 |
| 12b | simple | greedy thinkon | 372/400 | 0.9300 | 0.9007-0.9511 | 400/400 | 0 | 161 | 0.682 |
| 12b | multiple | greedy thinkon | 185/200 | 0.9250 | 0.8800-0.9540 | 200/200 | 0 | 168 | 0.371 |
| 12b | irrelevance | greedy thinkon | 178/240 | 0.7417 | 0.6828-0.7929 | 240/240 | 19 | 1081 | 2.737 |

## Pooled over all 840 problems

| model | setting | right | score | 95% range | thought | hours |
|---|---|---|---|---|---|---|
| e2b | greedy thinkoff | 748/840 | 0.8905 | 0.8676-0.9098 | 0/840 | 0.197 |
| e2b | sampled thinkoff | 702/840 | 0.8357 | 0.8091-0.8592 | 138/840 | 0.282 |
| e2b | sampled thinkon | 751/840 | 0.8940 | 0.8714-0.9131 | 657/840 | 0.624 |
| e2b | greedy thinkon | 757/840 | 0.9012 | 0.8791-0.9196 | 840/840 | 0.841 |
| e4b | greedy thinkoff | 758/840 | 0.9024 | 0.8804-0.9207 | 0/840 | 0.390 |
| e4b | sampled thinkoff | 698/840 | 0.8310 | 0.8041-0.8548 | 496/840 | 0.769 |
| e4b | sampled thinkon | 761/840 | 0.9060 | 0.8843-0.9239 | 290/840 | 0.832 |
| e4b | greedy thinkon | 770/840 | 0.9167 | 0.8960-0.9335 | 839/840 | 1.665 |
| 12b | greedy thinkoff | 756/840 | 0.9000 | 0.8779-0.9185 | 0/840 | 0.834 |
| 12b | sampled thinkoff | 740/840 | 0.8810 | 0.8573-0.9011 | 0/840 | 0.847 |
| 12b | sampled thinkon | 751/840 | 0.8940 | 0.8714-0.9131 | 838/840 | 2.755 |
| 12b | greedy thinkon | 735/840 | 0.8750 | 0.8509-0.8957 | 840/840 | 3.790 |

## Paired tests (same problems; exact two-sided McNemar)

| comparison | scope | A right only | B right only | p |
|---|---|---|---|---|
| e2b thinkoff: greedy (A) vs sampled (B) | simple | 19 | 5 | 0.006611 |
| e2b thinkoff: greedy (A) vs sampled (B) | multiple | 14 | 4 | 0.03088 |
| e2b thinkoff: greedy (A) vs sampled (B) | irrelevance | 41 | 19 | 0.006218 |
| e2b thinkoff: greedy (A) vs sampled (B) | all 840 | 74 | 28 | 5.908e-06 |
| e2b thinkon: greedy (A) vs sampled (B) | simple | 13 | 11 | 0.8388 |
| e2b thinkon: greedy (A) vs sampled (B) | multiple | 3 | 3 | 1 |
| e2b thinkon: greedy (A) vs sampled (B) | irrelevance | 5 | 1 | 0.2188 |
| e2b thinkon: greedy (A) vs sampled (B) | all 840 | 21 | 15 | 0.405 |
| e2b greedy: thinking on (A) vs off (B) | simple | 12 | 19 | 0.281 |
| e2b greedy: thinking on (A) vs off (B) | multiple | 10 | 7 | 0.6291 |
| e2b greedy: thinking on (A) vs off (B) | irrelevance | 17 | 4 | 0.007197 |
| e2b greedy: thinking on (A) vs off (B) | all 840 | 39 | 30 | 0.3356 |
| e2b sampled: thinking on (A) vs off (B) | simple | 22 | 17 | 0.5224 |
| e2b sampled: thinking on (A) vs off (B) | multiple | 17 | 4 | 0.007197 |
| e2b sampled: thinking on (A) vs off (B) | irrelevance | 42 | 11 | 2.248e-05 |
| e2b sampled: thinking on (A) vs off (B) | all 840 | 81 | 32 | 4.547e-06 |
| e4b thinkoff: greedy (A) vs sampled (B) | simple | 34 | 8 | 6.877e-05 |
| e4b thinkoff: greedy (A) vs sampled (B) | multiple | 23 | 3 | 8.798e-05 |
| e4b thinkoff: greedy (A) vs sampled (B) | irrelevance | 29 | 15 | 0.04877 |
| e4b thinkoff: greedy (A) vs sampled (B) | all 840 | 86 | 26 | 1.101e-08 |
| e4b thinkon: greedy (A) vs sampled (B) | simple | 18 | 18 | 1 |
| e4b thinkon: greedy (A) vs sampled (B) | multiple | 6 | 6 | 1 |
| e4b thinkon: greedy (A) vs sampled (B) | irrelevance | 9 | 0 | 0.003906 |
| e4b thinkon: greedy (A) vs sampled (B) | all 840 | 33 | 24 | 0.2892 |
| e4b greedy: thinking on (A) vs off (B) | simple | 16 | 15 | 1 |
| e4b greedy: thinking on (A) vs off (B) | multiple | 9 | 8 | 1 |
| e4b greedy: thinking on (A) vs off (B) | irrelevance | 13 | 3 | 0.02127 |
| e4b greedy: thinking on (A) vs off (B) | all 840 | 38 | 26 | 0.1686 |
| e4b sampled: thinking on (A) vs off (B) | simple | 42 | 15 | 0.00046 |
| e4b sampled: thinking on (A) vs off (B) | multiple | 27 | 6 | 0.0003241 |
| e4b sampled: thinking on (A) vs off (B) | irrelevance | 28 | 13 | 0.02753 |
| e4b sampled: thinking on (A) vs off (B) | all 840 | 97 | 34 | 3.322e-08 |
| 12b thinkoff: greedy (A) vs sampled (B) | simple | 7 | 1 | 0.07031 |
| 12b thinkoff: greedy (A) vs sampled (B) | multiple | 6 | 1 | 0.125 |
| 12b thinkoff: greedy (A) vs sampled (B) | irrelevance | 15 | 10 | 0.4244 |
| 12b thinkoff: greedy (A) vs sampled (B) | all 840 | 28 | 12 | 0.01659 |
| 12b thinkon: greedy (A) vs sampled (B) | simple | 4 | 10 | 0.1796 |
| 12b thinkon: greedy (A) vs sampled (B) | multiple | 0 | 3 | 0.25 |
| 12b thinkon: greedy (A) vs sampled (B) | irrelevance | 13 | 20 | 0.2962 |
| 12b thinkon: greedy (A) vs sampled (B) | all 840 | 17 | 33 | 0.03284 |
| 12b greedy: thinking on (A) vs off (B) | simple | 4 | 12 | 0.07681 |
| 12b greedy: thinking on (A) vs off (B) | multiple | 6 | 4 | 0.7539 |
| 12b greedy: thinking on (A) vs off (B) | irrelevance | 14 | 29 | 0.03154 |
| 12b greedy: thinking on (A) vs off (B) | all 840 | 24 | 45 | 0.01543 |
| 12b sampled: thinking on (A) vs off (B) | simple | 10 | 6 | 0.4545 |
| 12b sampled: thinking on (A) vs off (B) | multiple | 12 | 2 | 0.01294 |
| 12b sampled: thinking on (A) vs off (B) | irrelevance | 16 | 19 | 0.7359 |
| 12b sampled: thinking on (A) vs off (B) | all 840 | 38 | 27 | 0.2145 |
| 12b (A) vs e4b (B), greedy thinkoff | simple | 24 | 9 | 0.01353 |
| 12b (A) vs e4b (B), greedy thinkoff | multiple | 7 | 8 | 1 |
| 12b (A) vs e4b (B), greedy thinkoff | irrelevance | 2 | 18 | 0.0004025 |
| 12b (A) vs e4b (B), greedy thinkoff | all 840 | 33 | 35 | 0.9036 |
| 12b (A) vs e4b (B), sampled thinkoff | simple | 45 | 10 | 2.057e-06 |
| 12b (A) vs e4b (B), sampled thinkoff | multiple | 24 | 10 | 0.02431 |
| 12b (A) vs e4b (B), sampled thinkoff | irrelevance | 24 | 31 | 0.4188 |
| 12b (A) vs e4b (B), sampled thinkoff | all 840 | 93 | 51 | 0.0005851 |
| 12b (A) vs e4b (B), sampled thinkon | simple | 20 | 8 | 0.0357 |
| 12b (A) vs e4b (B), sampled thinkon | multiple | 7 | 4 | 0.5488 |
| 12b (A) vs e4b (B), sampled thinkon | irrelevance | 2 | 27 | 1.624e-06 |
| 12b (A) vs e4b (B), sampled thinkon | all 840 | 29 | 39 | 0.275 |
| 12b (A) vs e4b (B), greedy thinkon | simple | 17 | 11 | 0.3449 |
| 12b (A) vs e4b (B), greedy thinkon | multiple | 3 | 3 | 1 |
| 12b (A) vs e4b (B), greedy thinkon | irrelevance | 1 | 42 | 1e-11 |
| 12b (A) vs e4b (B), greedy thinkon | all 840 | 21 | 56 | 8.216e-05 |
| e4b (A) vs e2b (B), greedy thinkoff | simple | 14 | 18 | 0.5966 |
| e4b (A) vs e2b (B), greedy thinkoff | multiple | 10 | 6 | 0.4545 |
| e4b (A) vs e2b (B), greedy thinkoff | irrelevance | 13 | 3 | 0.02127 |
| e4b (A) vs e2b (B), greedy thinkoff | all 840 | 37 | 27 | 0.2604 |
| e4b (A) vs e2b (B), sampled thinkoff | simple | 17 | 33 | 0.03284 |
| e4b (A) vs e2b (B), sampled thinkoff | multiple | 14 | 20 | 0.3915 |
| e4b (A) vs e2b (B), sampled thinkoff | irrelevance | 39 | 21 | 0.02734 |
| e4b (A) vs e2b (B), sampled thinkoff | all 840 | 70 | 74 | 0.8027 |
| e4b (A) vs e2b (B), sampled thinkon | simple | 24 | 18 | 0.4408 |
| e4b (A) vs e2b (B), sampled thinkon | multiple | 8 | 6 | 0.7905 |
| e4b (A) vs e2b (B), sampled thinkon | irrelevance | 7 | 5 | 0.7744 |
| e4b (A) vs e2b (B), sampled thinkon | all 840 | 39 | 29 | 0.275 |
| e4b (A) vs e2b (B), greedy thinkon | simple | 15 | 11 | 0.5572 |
| e4b (A) vs e2b (B), greedy thinkon | multiple | 5 | 3 | 0.7266 |
| e4b (A) vs e2b (B), greedy thinkon | irrelevance | 8 | 1 | 0.03906 |
| e4b (A) vs e2b (B), greedy thinkon | all 840 | 28 | 15 | 0.06599 |
| 12b (A) vs e2b (B), greedy thinkoff | simple | 20 | 9 | 0.06143 |
| 12b (A) vs e2b (B), greedy thinkoff | multiple | 13 | 10 | 0.6776 |
| 12b (A) vs e2b (B), greedy thinkoff | irrelevance | 8 | 14 | 0.2863 |
| 12b (A) vs e2b (B), greedy thinkoff | all 840 | 41 | 33 | 0.416 |
| 12b (A) vs e2b (B), sampled thinkoff | simple | 30 | 11 | 0.004324 |
| 12b (A) vs e2b (B), sampled thinkoff | multiple | 16 | 8 | 0.1516 |
| 12b (A) vs e2b (B), sampled thinkoff | irrelevance | 34 | 23 | 0.1849 |
| 12b (A) vs e2b (B), sampled thinkoff | all 840 | 80 | 42 | 0.0007408 |
| 12b (A) vs e2b (B), sampled thinkon | simple | 28 | 10 | 0.005098 |
| 12b (A) vs e2b (B), sampled thinkon | multiple | 8 | 3 | 0.2266 |
| 12b (A) vs e2b (B), sampled thinkon | irrelevance | 3 | 26 | 1.524e-05 |
| 12b (A) vs e2b (B), sampled thinkon | all 840 | 39 | 39 | 1 |
| 12b (A) vs e2b (B), greedy thinkon | simple | 25 | 15 | 0.1539 |
| 12b (A) vs e2b (B), greedy thinkon | multiple | 7 | 5 | 0.7744 |
| 12b (A) vs e2b (B), greedy thinkon | irrelevance | 3 | 37 | 1.947e-08 |
| 12b (A) vs e2b (B), greedy thinkon | all 840 | 35 | 57 | 0.02802 |
| 12b greedy thinkoff (A) vs e4b greedy thinkon (B) | all 840 | 27 | 41 | 0.1143 |
| 12b greedy thinkoff (A) vs e4b sampled thinkon (B) | all 840 | 31 | 36 | 0.6254 |
| 12b sampled thinkon (A) vs e4b greedy thinkon (B) | all 840 | 24 | 43 | 0.02712 |
| 12b sampled thinkon (A) vs e4b sampled thinkon (B) | all 840 | 29 | 39 | 0.275 |

## Sampled, thinking off: right rate by whether the reply carried thinking

| model | task | thought: right/n | no thought: right/n |
|---|---|---|---|
| e2b | simple | 114/121 | 241/279 |
| e2b | multiple | 1/1 | 169/199 |
| e2b | irrelevance | 12/16 | 165/224 |
| e4b | simple | 324/343 | 15/57 |
| e4b | multiple | 120/125 | 44/75 |
| e4b | irrelevance | 12/28 | 183/212 |
| 12b | simple | 0/0 | 374/400 |
| 12b | multiple | 0/0 | 178/200 |
| 12b | irrelevance | 0/0 | 188/240 |

## Wrong replies by kind

| model | task | setting | kinds |
|---|---|---|---|
| e2b | simple | greedy thinkoff | value 22, 0 tool calls 5, missing 3, name restaurant_search_find_closest 1 |
| e2b | multiple | greedy thinkoff | value 12, 0 tool calls 6, 2 tool calls 1, missing 1 |
| e2b | irrelevance | greedy thinkoff | 1 tool calls 40, 2 tool calls 1 |
| e2b | simple | sampled thinkoff | 0 tool calls 22, value 21, missing 1, name restaurant_search_find_closest 1 |
| e2b | multiple | sampled thinkoff | 0 tool calls 17, value 10, missing 2, 2 tool calls 1 |
| e2b | irrelevance | sampled thinkoff | 1 tool calls 63 |
| e2b | simple | sampled thinkon | value 28, 0 tool calls 6, missing 4, 2 tool calls 1, name restaurant_search_find_closest 1 |
| e2b | multiple | sampled thinkon | value 10, 0 tool calls 5, 2 tool calls 1, missing 1 |
| e2b | irrelevance | sampled thinkon | 1 tool calls 32 |
| e2b | simple | greedy thinkon | value 25, 0 tool calls 10, missing 2, name restaurant_search_find_closest 1 |
| e2b | multiple | greedy thinkon | value 10, 0 tool calls 5, 2 tool calls 2 |
| e2b | irrelevance | greedy thinkon | 1 tool calls 28 |
| e4b | simple | greedy thinkoff | value 17, 0 tool calls 14, missing 2, 4 tool calls 1, name restaurant_search_find_closest 1 |
| e4b | multiple | greedy thinkoff | 0 tool calls 8, value 6, 2 tool calls 1, name random_forest_regression 1 |
| e4b | irrelevance | greedy thinkoff | 1 tool calls 30, 2 tool calls 1 |
| e4b | simple | sampled thinkoff | 0 tool calls 38, value 20, 4 tool calls 1, name restaurant_search_find_closest 1, missing 1 |
| e4b | multiple | sampled thinkoff | 0 tool calls 28, value 8 |
| e4b | irrelevance | sampled thinkoff | 1 tool calls 43, 2 tool calls 2 |
| e4b | simple | sampled thinkon | value 24, missing 5, 0 tool calls 3, 4 tool calls 1, name restaurant_search_find_closest 1 |
| e4b | multiple | sampled thinkon | value 11, 0 tool calls 3, 2 tool calls 1 |
| e4b | irrelevance | sampled thinkon | 1 tool calls 30 |
| e4b | simple | greedy thinkon | value 20, 0 tool calls 12, name restaurant_search_find_closest 1, missing 1 |
| e4b | multiple | greedy thinkon | value 9, 0 tool calls 4, 2 tool calls 1, name random_forest_regression 1 |
| e4b | irrelevance | greedy thinkon | 1 tool calls 21 |
| 12b | simple | greedy thinkoff | value 19, name restaurant_search_find_closest 1 |
| 12b | multiple | greedy thinkoff | value 13, 0 tool calls 2, 2 tool calls 1, name database_create_backup 1 |
| 12b | irrelevance | greedy thinkoff | 1 tool calls 46, 2 tool calls 1 |
| 12b | simple | sampled thinkoff | value 24, 0 tool calls 1, name restaurant_search_find_closest 1 |
| 12b | multiple | sampled thinkoff | value 18, 0 tool calls 2, 2 tool calls 1, name database_create_backup 1 |
| 12b | irrelevance | sampled thinkoff | 1 tool calls 51, 2 tool calls 1 |
| 12b | simple | sampled thinkon | value 19, missing 1, 0 tool calls 1, name restaurant_search_find_closest 1 |
| 12b | multiple | sampled thinkon | value 8, 0 tool calls 2, 2 tool calls 1, name game_missions_list 1 |
| 12b | irrelevance | sampled thinkon | 1 tool calls 54, 4 tool calls 1 |
| 12b | simple | greedy thinkon | value 24, 0 tool calls 2, 4 tool calls 1, name restaurant_search_find_closest 1 |
| 12b | multiple | greedy thinkon | value 10, 0 tool calls 3, 2 tool calls 1, name game_missions_list 1 |
| 12b | irrelevance | greedy thinkon | 1 tool calls 59, 3 tool calls 2, 2 tool calls 1 |
