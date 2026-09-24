# Bespoke Labs' 13-subset public suite: Gemma 4 read by label probabilities

Jev 1.13.0 and Nimble-9B columns are as published by Bespoke Labs on the same records; Gemma columns computed by `score_suite.py`. ECE: 10 equal-width bins; Brier: multiclass mean.

| Subset | type | n | Jev acc | Nimble acc | gemma-4-12B-it ar acc | gemma-4-26B-A4B-it-FP8-dynamic ar acc | gemma-4-E2B-it ar acc | gemma-4-E4B-it ar acc |
|---|---|---|---|---|---|---|---|---|
| aegis2 | noul | 250 | 0.804 | 0.812 | 0.768 | 0.760 | 0.748 | 0.768 |
| boolq | noul | 300 | 0.897 | 0.860 | 0.863 | 0.837 | 0.647 | 0.763 |
| civil_comments | noul | 300 | 0.810 | 0.703 | 0.893 | 0.887 | 0.903 | 0.913 |
| helpsteer2 | score | 249 | 0.341 | 0.390 | 0.426 | 0.373 | 0.386 | 0.386 |
| massive-de-DE | choice | 350 | 0.869 | 0.834 | 0.829 | 0.837 | 0.769 | 0.811 |
| massive-en-US | choice | 350 | 0.874 | 0.869 | 0.854 | 0.869 | 0.823 | 0.857 |
| multinli | choice | 299 | 0.829 | 0.853 | 0.809 | 0.849 | 0.706 | 0.793 |
| paws | noul | 250 | 0.892 | 0.828 | 0.856 | 0.828 | 0.760 | 0.808 |
| pubmedqa | choice | 250 | 0.772 | 0.756 | 0.712 | 0.664 | 0.492 | 0.548 |
| squad2 | noul | 299 | 0.829 | 0.806 | 0.863 | 0.883 | 0.749 | 0.813 |
| summeval-consistency | score | 144 | 0.812 | 0.757 | 0.833 | 0.785 | 0.785 | 0.826 |
| summeval-relevance | score | 240 | 0.350 | 0.492 | 0.404 | 0.392 | 0.312 | 0.383 |
| vitaminc-dev | choice | 599 | 0.801 | 0.766 | 0.726 | 0.760 | 0.698 | 0.699 |

| Subset | Jev ECE | Nimble ECE | gemma-4-12B-it ar ECE (after 50 labels) | gemma-4-26B-A4B-it-FP8-dynamic ar ECE (after 50 labels) | gemma-4-E2B-it ar ECE (after 50 labels) | gemma-4-E4B-it ar ECE (after 50 labels) |
|---|---|---|---|---|---|---|
| aegis2 | 0.048 | 0.102 | 0.229 (0.096) | 0.233 (0.120) | 0.246 (0.170) | 0.218 (0.079) |
| boolq | 0.038 | 0.109 | 0.135 (0.059) | 0.157 (0.070) | 0.347 (0.257) | 0.218 (0.104) |
| civil_comments | 0.056 | 0.133 | 0.105 (0.059) | 0.107 (0.048) | 0.094 (0.042) | 0.082 (0.056) |
| helpsteer2 | 0.261 | 0.343 | 0.490 (0.079) | 0.573 (0.196) | 0.553 (0.092) | 0.471 (0.088) |
| massive-de-DE | 0.071 | 0.079 | 0.167 (0.078) | 0.162 (0.080) | 0.225 (0.080) | 0.156 (0.077) |
| massive-en-US | 0.075 | 0.061 | 0.142 (0.068) | 0.129 (0.064) | 0.178 (0.081) | 0.123 (0.073) |
| multinli | 0.061 | 0.083 | 0.184 (0.070) | 0.149 (0.059) | 0.274 (0.095) | 0.159 (0.068) |
| paws | 0.041 | 0.100 | 0.136 (0.061) | 0.163 (0.066) | 0.209 (0.072) | 0.164 (0.066) |
| pubmedqa | 0.129 | 0.173 | 0.286 (0.126) | 0.322 (0.127) | 0.453 (0.103) | 0.379 (0.138) |
| squad2 | 0.042 | 0.139 | 0.130 (0.052) | 0.110 (0.059) | 0.228 (0.092) | 0.169 (0.076) |
| summeval-consistency | 0.079 | 0.177 | 0.149 (0.120) | 0.187 (0.133) | 0.170 (0.089) | 0.078 (0.095) |
| summeval-relevance | 0.232 | 0.102 | 0.488 (0.096) | 0.527 (0.095) | 0.491 (0.105) | 0.342 (0.132) |
| vitaminc-dev | 0.104 | 0.172 | 0.267 (0.044) | 0.233 (0.060) | 0.273 (0.063) | 0.257 (0.076) |

Pooled and macro accuracy, by question type:

| Group | Jev micro | Jev macro | gemma-4-12B-it ar micro | gemma-4-12B-it ar macro | gemma-4-26B-A4B-it-FP8-dynamic ar micro | gemma-4-26B-A4B-it-FP8-dynamic ar macro | gemma-4-E2B-it ar micro | gemma-4-E2B-it ar macro | gemma-4-E4B-it ar micro | gemma-4-E4B-it ar macro |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.773 | 0.760 | 0.762 | 0.757 | 0.760 | 0.748 | 0.685 | 0.675 | 0.728 | 0.721 |
| noul | 0.846 | 0.846 | 0.851 | 0.849 | 0.842 | 0.839 | 0.762 | 0.761 | 0.815 | 0.813 |
| choice | 0.828 | 0.829 | 0.781 | 0.786 | 0.797 | 0.796 | 0.708 | 0.697 | 0.745 | 0.742 |
| score | 0.452 | 0.501 | 0.510 | 0.554 | 0.474 | 0.517 | 0.449 | 0.494 | 0.485 | 0.532 |
