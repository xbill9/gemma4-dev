# Jev-style reads: DiffusionGemma vs Gemma 4 26B — run `2026-09-23-l4-awq`

Computed by `score.py` from the records in this directory. Escalation threshold 0.1 (structured_server default); ECE over 15 equal-width bins; ranges are 95% over 2000 resamples of examples.

## ag_news (n=300)

| readout | accuracy | ECE | Brier | AUROC conf→correct | median label mass | escalated | acc kept / escalated | median ms |
|---|---|---|---|---|---|---|---|---|
| ar | 86.7% | 0.129 | 0.255 | 0.738 | 0.999 | 6.0% | 88.7% / 55.6% | 248 |
| dg1 | 83.3% | 0.133 | 0.281 | 0.727 | 0.369 | 100.0% | — / 83.3% | 281 |
| dg4 | 83.0% | 0.119 | 0.279 | 0.735 | 0.370 | 100.0% | — / 83.0% | 281 |
| dgauto | 83.0% | 0.119 | 0.279 | 0.735 | 0.370 | 100.0% | — / 83.0% | 576 |

Majority-class accuracy: 25.0%.
- dgauto − ar, accuracy: -0.037 (95% range -0.067 to -0.007)
- dgauto − ar, ece: -0.010 (95% range -0.030 to +0.020)
- dgauto − ar, brier: +0.024 (95% range -0.015 to +0.063)
- DiffusionGemma spread across noise draws: median 0.003; low spread predicts correct with AUROC 0.685; 60.8% of wrong answers had spread under 0.02

Calibration on the held-out half after fitting one temperature on N labels:

| N labels | ar T | ar ECE | dg1 T | dg1 ECE | dg4 T | dg4 ECE |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.139 | 1.00 | 0.151 | 1.00 | 0.132 |
| 25 | 6.90 | 0.089 | 2.04 | 0.065 | 1.94 | 0.095 |
| 50 | 5.96 | 0.064 | 1.76 | 0.054 | 1.76 | 0.061 |
| 100 | 5.96 | 0.064 | 1.76 | 0.054 | 1.76 | 0.061 |
| 150 | 6.26 | 0.060 | 1.85 | 0.064 | 1.76 | 0.061 |

Options listed in reverse order: ar changed 14 of 300 answers (4.7%); dg1 changed 21 of 300 answers (7.0%)

## emotion (n=300)

| readout | accuracy | ECE | Brier | AUROC conf→correct | median label mass | escalated | acc kept / escalated | median ms |
|---|---|---|---|---|---|---|---|---|
| ar | 58.3% | 0.386 | 0.783 | 0.699 | 1.000 | 16.7% | 63.2% / 34.0% | 167 |
| dg1 | 60.7% | 0.262 | 0.666 | 0.660 | 0.366 | 99.0% | 66.7% / 60.6% | 255 |
| dg4 | 60.3% | 0.260 | 0.664 | 0.667 | 0.392 | 99.0% | 66.7% / 60.3% | 255 |
| dgauto | 60.3% | 0.261 | 0.664 | 0.667 | 0.392 | 99.0% | 66.7% / 60.3% | 556 |

Majority-class accuracy: 35.0%.
- dgauto − ar, accuracy: +0.020 (95% range -0.010 to +0.050)
- dgauto − ar, ece: -0.125 (95% range -0.154 to -0.089)
- dgauto − ar, brier: -0.118 (95% range -0.164 to -0.074)
- DiffusionGemma spread across noise draws: median 0.008; low spread predicts correct with AUROC 0.635; 63.9% of wrong answers had spread under 0.02

Calibration on the held-out half after fitting one temperature on N labels:

| N labels | ar T | ar ECE | dg1 T | dg1 ECE | dg4 T | dg4 ECE |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.440 | 1.00 | 0.332 | 1.00 | 0.311 |
| 25 | 7.24 | 0.102 | 2.36 | 0.105 | 2.25 | 0.103 |
| 50 | 7.24 | 0.102 | 2.48 | 0.122 | 2.36 | 0.115 |
| 100 | 6.57 | 0.122 | 2.25 | 0.145 | 2.14 | 0.125 |
| 150 | 6.57 | 0.122 | 2.36 | 0.105 | 2.36 | 0.115 |

Options listed in reverse order: ar changed 27 of 300 answers (9.0%); dg1 changed 26 of 300 answers (8.7%)

## irony (n=300)

| readout | accuracy | ECE | Brier | AUROC conf→correct | median label mass | escalated | acc kept / escalated | median ms |
|---|---|---|---|---|---|---|---|---|
| ar | 89.7% | 0.098 | 0.199 | 0.851 | 1.000 | 5.3% | 91.2% / 62.5% | 186 |
| dg1 | 86.7% | 0.049 | 0.170 | 0.899 | 0.567 | 96.3% | 100.0% / 86.2% | 260 |
| dg4 | 87.3% | 0.043 | 0.153 | 0.911 | 0.593 | 96.3% | 100.0% / 86.9% | 260 |
| dgauto | 87.3% | 0.043 | 0.153 | 0.911 | 0.594 | 96.3% | 100.0% / 86.9% | 557 |

Majority-class accuracy: 60.3%.
- dgauto − ar, accuracy: -0.023 (95% range -0.057 to +0.010)
- dgauto − ar, ece: -0.055 (95% range -0.077 to -0.014)
- dgauto − ar, brier: -0.046 (95% range -0.100 to +0.005)
- DiffusionGemma spread across noise draws: median 0.007; low spread predicts correct with AUROC 0.873; 15.8% of wrong answers had spread under 0.02

Calibration on the held-out half after fitting one temperature on N labels:

| N labels | ar T | ar ECE | dg1 T | dg1 ECE | dg4 T | dg4 ECE |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.088 | 1.00 | 0.057 | 1.00 | 0.055 |
| 25 | 3.48 | 0.054 | 0.81 | 0.078 | 0.73 | 0.055 |
| 50 | 4.03 | 0.042 | 1.13 | 0.071 | 0.77 | 0.054 |
| 100 | 4.90 | 0.033 | 1.38 | 0.053 | 1.19 | 0.029 |
| 150 | 4.90 | 0.033 | 1.25 | 0.053 | 1.08 | 0.050 |

## sst2 (n=300)

| readout | accuracy | ECE | Brier | AUROC conf→correct | median label mass | escalated | acc kept / escalated | median ms |
|---|---|---|---|---|---|---|---|---|
| ar | 95.0% | 0.051 | 0.099 | 0.850 | 1.000 | 1.0% | 95.3% / 66.7% | 189 |
| dg1 | 93.3% | 0.037 | 0.106 | 0.889 | 0.499 | 100.0% | — / 93.3% | 275 |
| dg4 | 93.3% | 0.045 | 0.105 | 0.893 | 0.503 | 100.0% | — / 93.3% | 275 |
| dgauto | 93.3% | 0.045 | 0.105 | 0.893 | 0.503 | 100.0% | — / 93.3% | 563 |

Majority-class accuracy: 51.0%.
- dgauto − ar, accuracy: -0.017 (95% range -0.040 to +0.007)
- dgauto − ar, ece: -0.005 (95% range -0.021 to +0.020)
- dgauto − ar, brier: +0.006 (95% range -0.025 to +0.036)
- DiffusionGemma spread across noise draws: median 0.001; low spread predicts correct with AUROC 0.891; 65.0% of wrong answers had spread under 0.02

Calibration on the held-out half after fitting one temperature on N labels:

| N labels | ar T | ar ECE | dg1 T | dg1 ECE | dg4 T | dg4 ECE |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.040 | 1.00 | 0.033 | 1.00 | 0.047 |
| 25 | 4.90 | 0.028 | 0.93 | 0.033 | 0.77 | 0.043 |
| 50 | 5.15 | 0.029 | 1.52 | 0.014 | 1.52 | 0.025 |
| 100 | 5.15 | 0.029 | 1.68 | 0.016 | 1.60 | 0.026 |
| 150 | 4.90 | 0.028 | 1.52 | 0.014 | 1.45 | 0.029 |

Options listed in reverse order: ar changed 6 of 300 answers (2.0%); dg1 changed 7 of 300 answers (2.3%)
