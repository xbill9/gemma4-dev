# 2026-09-30-fillB-v5e1: 4-bit against bf16, paired on the same records

## e4b-w8a8rtn against results/2026-09-29-e4b-v5e1b-e4b-w8a8

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.923 | 0.927 | +0.003 | -0.013 to +0.020 | 3 | 4 | 100.0% / 100.0% | 300 | +0.003 | -0.013 to +0.020 |
| ag_news | 300 | 0.827 | 0.827 | +0.000 | -0.027 to +0.030 | 9 | 9 | 97.3% / 77.3% | 231 | -0.004 | -0.039 to +0.030 |
| emotion | 300 | 0.557 | 0.550 | -0.007 | -0.027 to +0.013 | 6 | 4 | 97.0% / 86.3% | 259 | -0.008 | -0.031 to +0.012 |
| irony | 300 | 0.883 | 0.837 | -0.047 | -0.080 to -0.017 | 19 | 5 | 100.0% / 100.0% | 300 | -0.047 | -0.080 to -0.017 |
| reversed options (3 choice tasks) | 900 | 0.727 | 0.739 | +0.012 | -0.001 to +0.027 | 13 | 24 | 100.0% / 96.7% | 870 | +0.013 | -0.001 to +0.026 |
| suite (all subsets) | 3880 | 0.727 | 0.730 | +0.003 | -0.004 to +0.011 | 102 | 114 | 82.0% / 81.9% | 3167 | +0.002 | -0.007 to +0.010 |

