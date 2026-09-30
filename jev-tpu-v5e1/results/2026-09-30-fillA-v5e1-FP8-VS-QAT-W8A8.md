# 2026-09-30-fillA-v5e1: 4-bit against bf16, paired on the same records

## 12b-fp8 against results/2026-09-30-12bgap-v5e1-12b-w8a8

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.940 | 0.953 | +0.013 | +0.003 to +0.027 | 0 | 4 | 100.0% / 100.0% | 300 | +0.013 | +0.003 to +0.027 |
| ag_news | 300 | 0.863 | 0.860 | -0.003 | -0.020 to +0.013 | 4 | 3 | 80.7% / 84.3% | 241 | -0.004 | -0.025 to +0.017 |
| emotion | 300 | 0.597 | 0.593 | -0.003 | -0.017 to +0.010 | 3 | 2 | 99.7% / 99.0% | 297 | -0.003 | -0.020 to +0.010 |
| irony | 300 | 0.867 | 0.867 | +0.000 | -0.023 to +0.023 | 6 | 6 | 100.0% / 99.7% | 299 | +0.000 | -0.023 to +0.023 |
| reversed options (3 choice tasks) | 900 | 0.798 | 0.803 | +0.006 | -0.003 to +0.014 | 6 | 11 | 91.8% / 94.3% | 822 | +0.006 | -0.004 to +0.017 |
| suite (all subsets) | 3880 | 0.757 | 0.757 | -0.000 | -0.006 to +0.005 | 66 | 65 | 85.2% / 85.7% | 3258 | +0.001 | -0.006 to +0.007 |

