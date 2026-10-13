# Answer quality: E2B vs E4B, exact Q4_0 on the RTX 3050 OEM

## Nimble 13-subset suite

| run | records | accuracy | all labels returned |
|---|---:|---:|---:|
| E2B Q4 (3050) | 3880 | 0.6827 | 96.2% |
| E4B Q4 (3050) | 3880 | 0.7304 | 82.0% |
| E2B bf16 (TPU v5e) | 3880 | 0.6827 | 86.5% |
| E4B bf16 (TPU v6e) | 3880 | 0.7307 | 82.7% |

Paired on the same records (difference = B - A; 95% range from 2,000 bootstrap draws):

| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |
|---|---|---:|---:|---:|---:|---|---:|---:|
| E2B Q4 (3050) | E4B Q4 (3050) | 3880 | 0.683 | 0.730 | **+0.048** | +0.035 to +0.061 | 278 | 463 |
| E2B bf16 (TPU v5e) | E2B Q4 (3050) | 3880 | 0.683 | 0.683 | **+0.000** | -0.010 to +0.010 | 170 | 170 |
| E4B bf16 (TPU v6e) | E4B Q4 (3050) | 3880 | 0.731 | 0.730 | **-0.000** | -0.007 to +0.006 | 91 | 90 |
| E2B bf16 (TPU v5e) | E4B bf16 (TPU v6e) | 3880 | 0.683 | 0.731 | **+0.048** | +0.035 to +0.061 | 251 | 437 |

Per subset, E4B - E2B on the 3050 (secondary; ranges not corrected for 13 comparisons):

| subset | n | E2B | E4B | E4B - E2B | 95% range |
|---|---:|---:|---:|---:|---|
| aegis2 | 250 | 0.752 | 0.764 | +0.012 | -0.024 to +0.052 |
| boolq | 300 | 0.623 | 0.760 | +0.137 | +0.087 to +0.183 |
| civil_comments | 300 | 0.903 | 0.923 | +0.020 | -0.007 to +0.043 |
| helpsteer2 | 249 | 0.369 | 0.386 | +0.016 | -0.044 to +0.076 |
| massive-de-DE | 350 | 0.746 | 0.820 | +0.074 | +0.034 to +0.114 |
| massive-en-US | 350 | 0.806 | 0.863 | +0.057 | +0.029 to +0.086 |
| multinli | 299 | 0.682 | 0.776 | +0.094 | +0.033 to +0.151 |
| paws | 250 | 0.748 | 0.800 | +0.052 | +0.008 to +0.100 |
| pubmedqa | 250 | 0.608 | 0.564 | -0.044 | -0.108 to +0.020 |
| squad2 | 299 | 0.686 | 0.856 | +0.171 | +0.117 to +0.224 |
| summeval-consistency | 144 | 0.757 | 0.833 | +0.076 | +0.007 to +0.146 |
| summeval-relevance | 240 | 0.292 | 0.367 | +0.075 | +0.000 to +0.146 |
| vitaminc-dev | 599 | 0.736 | 0.694 | -0.042 | -0.080 to -0.005 |

## GSM8K (2,048-token limit) and BFCL simple

| task | A | B | n | A | B | difference | 95% range | A only | B only |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| gsm8k | E2B Q4 (3050) | E4B Q4 (3050) | 1319 | 0.898 | 0.933 | **+0.035** | +0.020 to +0.050 | 28 | 74 |
| gsm8k | E2B bf16 (TPU v5e) | E2B Q4 (3050) | 1319 | 0.910 | 0.898 | **-0.012** | -0.025 to +0.001 | 47 | 31 |
| gsm8k | E4B QAT repack W4A16 (TPU v5e) | E4B Q4 (3050) | 1319 | 0.934 | 0.933 | **-0.002** | -0.009 to +0.005 | 13 | 11 |
| bfcl_simple | E2B Q4 (3050) | E4B Q4 (3050) | 400 | 0.925 | 0.910 | **-0.015** | -0.043 to +0.013 | 19 | 13 |
| bfcl_simple | E2B bf16 (TPU v5e) | E2B Q4 (3050) | 400 | 0.927 | 0.925 | **-0.003** | -0.020 to +0.015 | 7 | 6 |
| bfcl_simple | E4B QAT repack W4A16 (TPU v5e) | E4B Q4 (3050) | 400 | 0.910 | 0.910 | **+0.000** | -0.010 to +0.010 | 2 | 2 |
