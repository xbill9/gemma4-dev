# 2026-09-30-fillB-v5e1: 4-bit against bf16, paired on the same records

## e4b-w8a8rtn against ../jev-tpu-31b/results/2026-09-25-followup-e4b-bf16

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.940 | 0.927 | -0.013 | -0.027 to -0.003 | 4 | 0 | 100.0% / 100.0% | 300 | -0.013 | -0.027 to -0.003 |
| ag_news | 300 | 0.833 | 0.827 | -0.007 | -0.023 to +0.010 | 4 | 2 | 87.3% / 77.3% | 220 | -0.009 | -0.032 to +0.014 |
| emotion | 300 | 0.543 | 0.550 | +0.007 | -0.010 to +0.023 | 2 | 4 | 92.7% / 86.3% | 256 | +0.008 | -0.012 to +0.027 |
| irony | 300 | 0.843 | 0.837 | -0.007 | -0.030 to +0.013 | 7 | 5 | 100.0% / 100.0% | 300 | -0.007 | -0.030 to +0.013 |
| reversed options (3 choice tasks) | 900 | 0.733 | 0.739 | +0.006 | -0.006 to +0.017 | 10 | 15 | 99.6% / 96.7% | 868 | +0.006 | -0.005 to +0.017 |
| suite (all subsets) | 3880 | 0.731 | 0.730 | -0.001 | -0.007 to +0.005 | 68 | 64 | 82.7% / 81.9% | 3171 | -0.002 | -0.009 to +0.004 |

