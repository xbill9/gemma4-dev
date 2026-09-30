# 2026-09-30-fillA-v5e1: 4-bit against bf16, paired on the same records

## 12b-w8a8rtn against ../jev-tpu-31b/results/2026-09-26-override-12b-bf16-ovr

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.950 | 0.957 | +0.007 | -0.007 to +0.020 | 1 | 3 | 80.7% / 85.0% | 237 | +0.008 | -0.008 to +0.025 |
| ag_news | 300 | 0.863 | 0.853 | -0.010 | -0.030 to +0.010 | 7 | 4 | 77.7% / 60.7% | 175 | -0.023 | -0.057 to +0.011 |
| emotion | 300 | 0.593 | 0.593 | +0.000 | -0.023 to +0.023 | 7 | 7 | 84.0% / 73.0% | 213 | -0.005 | -0.038 to +0.028 |
| irony | 300 | 0.847 | 0.843 | -0.003 | -0.023 to +0.017 | 5 | 4 | 100.0% / 100.0% | 300 | -0.003 | -0.023 to +0.017 |
| reversed options (3 choice tasks) | 900 | 0.796 | 0.790 | -0.006 | -0.016 to +0.003 | 12 | 7 | 80.0% / 69.2% | 595 | -0.002 | -0.013 to +0.012 |
| suite (all subsets) | 3880 | 0.760 | 0.752 | -0.007 | -0.014 to -0.001 | 114 | 86 | 78.0% / 75.1% | 2863 | -0.008 | -0.017 to +0.002 |

