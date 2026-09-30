# 2026-09-30-fillA-v5e1: 4-bit against bf16, paired on the same records

## 12b-fp8 against ../jev-tpu-31b/results/2026-09-26-override-12b-bf16-ovr

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.950 | 0.953 | +0.003 | -0.013 to +0.020 | 2 | 3 | 80.7% / 100.0% | 242 | +0.004 | -0.012 to +0.021 |
| ag_news | 300 | 0.863 | 0.860 | -0.003 | -0.017 to +0.010 | 3 | 2 | 77.7% / 84.3% | 233 | -0.004 | -0.021 to +0.013 |
| emotion | 300 | 0.593 | 0.593 | +0.000 | -0.017 to +0.020 | 4 | 4 | 84.0% / 99.0% | 252 | -0.004 | -0.024 to +0.016 |
| irony | 300 | 0.847 | 0.867 | +0.020 | +0.000 to +0.040 | 2 | 8 | 100.0% / 99.7% | 299 | +0.020 | +0.000 to +0.040 |
| reversed options (3 choice tasks) | 900 | 0.796 | 0.803 | +0.008 | -0.001 to +0.017 | 5 | 12 | 80.0% / 94.3% | 718 | +0.010 | +0.000 to +0.021 |
| suite (all subsets) | 3880 | 0.760 | 0.757 | -0.003 | -0.009 to +0.004 | 98 | 88 | 78.0% / 85.7% | 3023 | +0.000 | -0.008 to +0.009 |

