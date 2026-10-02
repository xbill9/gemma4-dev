# 2026-10-02-suite-tp4-v5e4: GSM8K and BFCL simple at TP=4 on v5e-4 against TP=1 on v5e-1, paired record for record

`gen_compare.py` (10,000 bootstrap resamples, seed 0). Reference: `2026-09-30-gen2048a-v5e1` (E2B) and `2026-09-30-gen2048c-v5e1` (12B), same flags (`tools,mml=4096`, gmu 0.80 / 0.92, GSM8K limit 2,048 tokens).

| comparison | n | v5e-1 | v5e-4 TP=4 | difference | 95% range | v5e-1 right only | v5e-4 right only |
|---|---:|---:|---:|---:|---|---:|---:|
| E2B W8A8 gsm8k | 1319 | 0.889 | 0.903 | +0.014 | +0.002 to +0.025 | 22 | 40 |
| 12B W8A8 + int4 tables gsm8k | 1319 | 0.964 | 0.955 | -0.009 | -0.017 to -0.002 | 20 | 8 |
| E2B W8A8 bfcl_simple | 400 | 0.915 | 0.927 | +0.013 | +0.000 to +0.025 | 1 | 6 |
| 12B W8A8 + int4 tables bfcl_simple | 400 | 0.955 | 0.948 | -0.007 | -0.018 to +0.000 | 3 | 0 |
