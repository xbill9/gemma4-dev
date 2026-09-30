# 2026-09-29-e4br-v5e1: 4-bit against bf16, paired on the same records

## e4b-repack against results/2026-09-29-e4b-v5e1-e4b-google

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.927 | 0.930 | +0.003 | -0.010 to +0.017 | 2 | 3 | 100.0% / 100.0% | 300 | +0.003 | -0.010 to +0.017 |
| ag_news | 300 | 0.840 | 0.840 | +0.000 | -0.020 to +0.020 | 4 | 4 | 98.3% / 100.0% | 295 | +0.000 | -0.020 to +0.017 |
| emotion | 300 | 0.547 | 0.547 | +0.000 | -0.020 to +0.020 | 5 | 5 | 98.3% / 97.3% | 292 | +0.000 | -0.021 to +0.021 |
| irony | 300 | 0.857 | 0.867 | +0.010 | -0.013 to +0.037 | 6 | 9 | 100.0% / 100.0% | 300 | +0.010 | -0.013 to +0.037 |
| reversed options (3 choice tasks) | 900 | 0.740 | 0.730 | -0.010 | -0.021 to +0.001 | 19 | 10 | 100.0% / 100.0% | 900 | -0.010 | -0.021 to +0.001 |
| suite (all subsets) | 3880 | 0.716 | 0.729 | +0.013 | +0.007 to +0.020 | 55 | 105 | 82.0% / 82.0% | 3180 | +0.014 | +0.006 to +0.022 |

