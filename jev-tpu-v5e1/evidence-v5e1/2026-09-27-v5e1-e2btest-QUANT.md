# 2026-09-27-v5e1-e2btest: 4-bit against bf16, paired on the same records

## e2b-bf16 against ../jev-tpu-31b/results/2026-09-25-followup-e2b-bf16

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.887 | 0.883 | -0.003 | -0.010 to +0.000 | 1 | 0 | 100.0% / 100.0% | 300 | -0.003 | -0.010 to +0.000 |
| ag_news | 300 | 0.300 | 0.307 | +0.007 | +0.000 to +0.017 | 0 | 2 | 93.7% / 94.7% | 280 | +0.007 | +0.000 to +0.018 |
| emotion | 300 | 0.537 | 0.533 | -0.003 | -0.017 to +0.010 | 3 | 2 | 83.0% / 82.3% | 247 | -0.004 | -0.024 to +0.012 |
| irony | 300 | 0.733 | 0.730 | -0.003 | -0.013 to +0.007 | 2 | 1 | 100.0% / 100.0% | 300 | -0.003 | -0.013 to +0.007 |
| reversed options (3 choice tasks) | 900 | 0.566 | 0.570 | +0.004 | +0.001 to +0.009 | 0 | 4 | 99.6% / 99.4% | 894 | +0.004 | +0.001 to +0.009 |
| suite (all subsets) | 3880 | 0.685 | 0.685 | -0.001 | -0.004 to +0.003 | 27 | 25 | 86.5% / 86.5% | 3350 | +0.000 | -0.004 to +0.004 |

