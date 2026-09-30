# 2026-09-30-fillB-v5e1: 4-bit against bf16, paired on the same records

## e2b-w8a8-emb4-hf against results/2026-09-29-w8e4-v5e1-e2b-w8a8-emb4

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.883 | 0.883 | +0.000 | -0.017 to +0.020 | 4 | 4 | 100.0% / 100.0% | 300 | +0.000 | -0.017 to +0.020 |
| ag_news | 300 | 0.533 | 0.507 | -0.027 | -0.060 to +0.007 | 17 | 9 | 100.0% / 100.0% | 300 | -0.027 | -0.060 to +0.007 |
| emotion | 300 | 0.493 | 0.460 | -0.033 | -0.070 to +0.003 | 20 | 10 | 100.0% / 100.0% | 300 | -0.033 | -0.070 to +0.003 |
| irony | 300 | 0.780 | 0.803 | +0.023 | -0.010 to +0.057 | 9 | 16 | 100.0% / 100.0% | 300 | +0.023 | -0.010 to +0.057 |
| reversed options (3 choice tasks) | 900 | 0.621 | 0.607 | -0.014 | -0.030 to +0.001 | 32 | 19 | 100.0% / 100.0% | 900 | -0.014 | -0.030 to +0.001 |
| suite (all subsets) | 3880 | 0.677 | 0.673 | -0.004 | -0.011 to +0.004 | 105 | 91 | 95.2% / 94.5% | 3639 | -0.004 | -0.012 to +0.003 |

