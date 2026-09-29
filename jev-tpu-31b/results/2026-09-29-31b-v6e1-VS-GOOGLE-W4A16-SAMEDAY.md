# 2026-09-29-31b-v6e1: 4-bit against bf16, paired on the same records

## 31b-repack against results/2026-09-29-31b-v6e1-31b-google

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.953 | 0.947 | -0.007 | -0.017 to +0.000 | 2 | 0 | 60.3% / 68.3% | 180 | -0.011 | -0.028 to +0.000 |
| ag_news | 300 | 0.870 | 0.873 | +0.003 | +0.000 to +0.010 | 0 | 1 | 44.3% / 44.3% | 131 | +0.008 | +0.000 to +0.023 |
| emotion | 300 | 0.620 | 0.620 | +0.000 | -0.013 to +0.013 | 2 | 2 | 72.3% / 78.3% | 216 | +0.000 | -0.019 to +0.019 |
| irony | 300 | 0.900 | 0.903 | +0.003 | +0.000 to +0.010 | 0 | 1 | 100.0% / 97.3% | 292 | +0.003 | +0.000 to +0.010 |
| reversed options (3 choice tasks) | 900 | 0.807 | 0.811 | +0.004 | -0.002 to +0.011 | 3 | 7 | 64.0% / 64.7% | 557 | +0.004 | -0.007 to +0.014 |
| suite (all subsets) | 3880 | 0.777 | 0.782 | +0.005 | +0.001 to +0.010 | 26 | 47 | 70.4% / 67.3% | 2520 | +0.008 | +0.001 to +0.014 |

