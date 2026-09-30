# 2026-09-30-fillA-v5e1: 4-bit against bf16, paired on the same records

## 12b-w8a8rtn against results/2026-09-30-12bgap-v5e1-12b-w8a8

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.940 | 0.957 | +0.017 | +0.000 to +0.033 | 1 | 6 | 100.0% / 85.0% | 255 | +0.020 | +0.000 to +0.039 |
| ag_news | 300 | 0.863 | 0.853 | -0.010 | -0.033 to +0.017 | 9 | 6 | 80.7% / 60.7% | 180 | -0.022 | -0.061 to +0.017 |
| emotion | 300 | 0.597 | 0.593 | -0.003 | -0.030 to +0.020 | 7 | 6 | 99.7% / 73.0% | 219 | +0.005 | -0.023 to +0.037 |
| irony | 300 | 0.867 | 0.843 | -0.023 | -0.050 to +0.000 | 11 | 4 | 100.0% / 100.0% | 300 | -0.023 | -0.050 to +0.000 |
| reversed options (3 choice tasks) | 900 | 0.798 | 0.790 | -0.008 | -0.020 to +0.004 | 19 | 12 | 91.8% / 69.2% | 620 | -0.005 | -0.021 to +0.011 |
| suite (all subsets) | 3880 | 0.757 | 0.752 | -0.005 | -0.013 to +0.002 | 121 | 102 | 85.2% / 75.1% | 2907 | -0.007 | -0.017 to +0.002 |

