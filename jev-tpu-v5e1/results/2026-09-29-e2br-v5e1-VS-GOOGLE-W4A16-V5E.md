# 2026-09-29-e2br-v5e1: 4-bit against bf16, paired on the same records

## e2b-repack against results/2026-09-29-e2b-v5e1-e2b-google

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.870 | 0.893 | +0.023 | +0.003 to +0.047 | 2 | 9 | 100.0% / 100.0% | 300 | +0.023 | +0.003 to +0.047 |
| ag_news | 300 | 0.513 | 0.573 | +0.060 | +0.023 to +0.100 | 9 | 27 | 100.0% / 100.0% | 300 | +0.060 | +0.023 to +0.100 |
| emotion | 300 | 0.437 | 0.460 | +0.023 | -0.010 to +0.053 | 8 | 15 | 100.0% / 100.0% | 300 | +0.023 | -0.010 to +0.053 |
| irony | 300 | 0.797 | 0.787 | -0.010 | -0.047 to +0.027 | 16 | 13 | 100.0% / 100.0% | 300 | -0.010 | -0.047 to +0.027 |
| reversed options (3 choice tasks) | 900 | 0.602 | 0.611 | +0.009 | -0.009 to +0.026 | 26 | 34 | 100.0% / 100.0% | 900 | +0.009 | -0.009 to +0.026 |
| suite (all subsets) | 3880 | 0.655 | 0.678 | +0.024 | +0.014 to +0.034 | 149 | 241 | 95.2% / 96.5% | 3666 | +0.023 | +0.013 to +0.035 |

