# Gemma 4 E4B and E2B, bf16, plain label read — computed by score.py functions (exploratory arms, see PREREGISTRATION.md addendum)

| Model | Task | Accuracy | Majority | ECE raw | ECE after 50 labels (20-split mean) | Fitted T at 50 | Reversed-order flips | Most-picked option |
|---|---|---|---|---|---|---|---|---|
| 26B AWQ | sst2 | 95.0% | 51.0% | 0.051 | 0.039 | 4.65 | 6/300 | positive 150/300 |
| 26B AWQ | ag_news | 86.7% | 25.0% | 0.129 | 0.066 | 6.30 | 14/300 | business 87/300 |
| 26B AWQ | emotion | 58.3% | 35.0% | 0.386 | 0.089 | 7.05 | 27/300 | sadness 138/300 |
| 26B AWQ | irony | 89.7% | 60.3% | 0.098 | 0.060 | 5.20 | — | no 170/300 |
| E4B bf16 | sst2 | 94.3% | 51.0% | 0.050 | 0.042 | 2.32 | 2/300 | negative 156/300 |
| E4B bf16 | ag_news | 83.7% | 25.0% | 0.143 | 0.071 | 2.33 | 35/300 | business 85/300 |
| E4B bf16 | emotion | 54.0% | 35.0% | 0.401 | 0.089 | 3.77 | 24/300 | sadness 154/300 |
| E4B bf16 | irony | 84.3% | 60.3% | 0.085 | 0.081 | 1.89 | — | no 164/300 |
| E2B bf16 | sst2 | 88.7% | 51.0% | 0.114 | 0.059 | 7.46 | 19/300 | positive 173/300 |
| E2B bf16 | ag_news | 30.3% | 25.0% | 0.692 | 0.475 | 7.99 | 28/300 | world 277/300 |
| E2B bf16 | emotion | 53.3% | 35.0% | 0.431 | 0.092 | 7.80 | 86/300 | sadness 122/300 |
| E2B bf16 | irony | 72.7% | 60.3% | 0.229 | 0.091 | 7.33 | — | no 257/300 |
