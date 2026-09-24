# Bespoke Labs' 13-subset public suite: Gemma 4 read by label probabilities

Jev 1.13.0 and Nimble-9B columns are as published by Bespoke Labs on the same records; Gemma columns computed by `score_suite.py`. ECE: 10 equal-width bins; Brier: multiclass mean.

| Subset | type | n | Jev acc | Nimble acc | diffusiongemma-26B-A4B-it-AWQ-INT4 dg1 acc | diffusiongemma-26B-A4B-it-AWQ-INT4 dg4 acc | gemma-4-26B-A4B-it-AWQ-4bit ar acc | gemma-4-E4B-it ar acc |
|---|---|---|---|---|---|---|---|---|
| aegis2 | noul | 250 | 0.804 | 0.812 | 0.772 | 0.768 | 0.784 | 0.768 |
| boolq | noul | 300 | 0.897 | 0.860 | 0.850 | 0.850 | 0.840 | 0.760 |
| civil_comments | noul | 300 | 0.810 | 0.703 | 0.877 | 0.873 | 0.870 | 0.913 |
| helpsteer2 | score | 249 | 0.341 | 0.390 | 0.382 | 0.386 | 0.414 | 0.390 |
| massive-de-DE | choice | 350 | 0.869 | 0.834 | 0.837 | 0.834 | 0.840 | 0.811 |
| massive-en-US | choice | 350 | 0.874 | 0.869 | 0.851 | 0.849 | 0.863 | 0.854 |
| multinli | choice | 299 | 0.829 | 0.853 | 0.753 | 0.756 | 0.816 | 0.793 |
| paws | noul | 250 | 0.892 | 0.828 | 0.848 | 0.848 | 0.840 | 0.812 |
| pubmedqa | choice | 250 | 0.772 | 0.756 | 0.668 | 0.664 | 0.640 | 0.560 |
| squad2 | noul | 299 | 0.829 | 0.806 | 0.880 | 0.873 | 0.893 | 0.833 |
| summeval-consistency | score | 144 | 0.812 | 0.757 | 0.868 | 0.861 | 0.771 | 0.833 |
| summeval-relevance | score | 240 | 0.350 | 0.492 | 0.475 | 0.475 | 0.308 | 0.404 |
| vitaminc-dev | choice | 599 | 0.801 | 0.766 | 0.748 | 0.748 | 0.746 | 0.706 |

| Subset | Jev ECE | Nimble ECE | diffusiongemma-26B-A4B-it-AWQ-INT4 dg1 ECE (after 50 labels) | diffusiongemma-26B-A4B-it-AWQ-INT4 dg4 ECE (after 50 labels) | gemma-4-26B-A4B-it-AWQ-4bit ar ECE (after 50 labels) | gemma-4-E4B-it ar ECE (after 50 labels) |
|---|---|---|---|---|---|---|
| aegis2 | 0.048 | 0.102 | 0.184 (0.085) | 0.184 (0.079) | 0.218 (0.094) | 0.218 (0.077) |
| boolq | 0.038 | 0.109 | 0.118 (0.061) | 0.114 (0.060) | 0.155 (0.064) | 0.221 (0.104) |
| civil_comments | 0.056 | 0.133 | 0.077 (0.069) | 0.074 (0.072) | 0.122 (0.048) | 0.083 (0.057) |
| helpsteer2 | 0.261 | 0.343 | 0.446 (0.106) | 0.441 (0.102) | 0.526 (0.149) | 0.473 (0.095) |
| massive-de-DE | 0.071 | 0.079 | 0.114 (0.088) | 0.113 (0.096) | 0.157 (0.080) | 0.157 (0.077) |
| massive-en-US | 0.075 | 0.061 | 0.108 (0.079) | 0.108 (0.074) | 0.136 (0.071) | 0.125 (0.068) |
| multinli | 0.061 | 0.083 | 0.169 (0.069) | 0.164 (0.064) | 0.180 (0.084) | 0.160 (0.072) |
| paws | 0.041 | 0.100 | 0.073 (0.072) | 0.070 (0.073) | 0.157 (0.060) | 0.173 (0.071) |
| pubmedqa | 0.129 | 0.173 | 0.243 (0.098) | 0.248 (0.095) | 0.354 (0.159) | 0.375 (0.134) |
| squad2 | 0.042 | 0.139 | 0.072 (0.062) | 0.066 (0.060) | 0.104 (0.056) | 0.145 (0.074) |
| summeval-consistency | 0.079 | 0.177 | 0.072 (0.082) | 0.076 (0.078) | 0.211 (0.128) | 0.079 (0.092) |
| summeval-relevance | 0.232 | 0.102 | 0.210 (0.093) | 0.205 (0.093) | 0.607 (0.167) | 0.323 (0.133) |
| vitaminc-dev | 0.104 | 0.172 | 0.170 (0.065) | 0.164 (0.060) | 0.247 (0.076) | 0.260 (0.080) |

Pooled and macro accuracy, by question type:

| Group | Jev micro | Jev macro | diffusiongemma-26B-A4B-it-AWQ-INT4 dg1 micro | diffusiongemma-26B-A4B-it-AWQ-INT4 dg1 macro | diffusiongemma-26B-A4B-it-AWQ-INT4 dg4 micro | diffusiongemma-26B-A4B-it-AWQ-INT4 dg4 macro | gemma-4-26B-A4B-it-AWQ-4bit ar micro | gemma-4-26B-A4B-it-AWQ-4bit ar macro | gemma-4-E4B-it ar micro | gemma-4-E4B-it ar macro |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.773 | 0.760 | 0.761 | 0.754 | 0.759 | 0.753 | 0.753 | 0.740 | 0.733 | 0.726 |
| noul | 0.846 | 0.846 | 0.848 | 0.845 | 0.845 | 0.842 | 0.848 | 0.845 | 0.819 | 0.817 |
| choice | 0.828 | 0.829 | 0.774 | 0.771 | 0.773 | 0.770 | 0.783 | 0.781 | 0.748 | 0.745 |
| score | 0.452 | 0.501 | 0.528 | 0.575 | 0.528 | 0.574 | 0.455 | 0.498 | 0.496 | 0.542 |
