# 2026-09-29-qatw8-v5e1: 4-bit against bf16, paired on the same records

## e2b-qat-w8a8 against results/2026-09-29-e2br-v5e1-e2b-repack

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.893 | 0.880 | -0.013 | -0.027 to -0.003 | 4 | 0 | 100.0% / 100.0% | 300 | -0.013 | -0.027 to -0.003 |
| ag_news | 300 | 0.573 | 0.557 | -0.017 | -0.043 to +0.010 | 12 | 7 | 100.0% / 100.0% | 300 | -0.017 | -0.043 to +0.010 |
| emotion | 300 | 0.460 | 0.450 | -0.010 | -0.037 to +0.017 | 11 | 8 | 100.0% / 100.0% | 300 | -0.010 | -0.037 to +0.017 |
| irony | 300 | 0.787 | 0.803 | +0.017 | -0.007 to +0.040 | 4 | 9 | 100.0% / 100.0% | 300 | +0.017 | -0.007 to +0.040 |
| reversed options (3 choice tasks) | 900 | 0.611 | 0.606 | -0.006 | -0.022 to +0.011 | 31 | 26 | 100.0% / 100.0% | 900 | -0.006 | -0.022 to +0.011 |
| suite (all subsets) | 3880 | 0.678 | 0.686 | +0.007 | +0.000 to +0.015 | 97 | 126 | 96.5% / 96.3% | 3711 | +0.008 | +0.000 to +0.016 |

