# 2026-09-30-fillB-v5e1: 4-bit against bf16, paired on the same records

## e2b-w4a16text against results/2026-09-29-e2br-v5e1-e2b-repack

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.893 | 0.893 | +0.000 | +0.000 to +0.000 | 0 | 0 | 100.0% / 100.0% | 300 | +0.000 | +0.000 to +0.000 |
| ag_news | 300 | 0.573 | 0.573 | +0.000 | +0.000 to +0.000 | 0 | 0 | 100.0% / 100.0% | 300 | +0.000 | +0.000 to +0.000 |
| emotion | 300 | 0.460 | 0.460 | +0.000 | +0.000 to +0.000 | 0 | 0 | 100.0% / 100.0% | 300 | +0.000 | +0.000 to +0.000 |
| irony | 300 | 0.787 | 0.787 | +0.000 | +0.000 to +0.000 | 0 | 0 | 100.0% / 100.0% | 300 | +0.000 | +0.000 to +0.000 |
| reversed options (3 choice tasks) | 900 | 0.611 | 0.611 | +0.000 | +0.000 to +0.000 | 0 | 0 | 100.0% / 100.0% | 900 | +0.000 | +0.000 to +0.000 |
| suite (all subsets) | 3880 | 0.678 | 0.678 | -0.001 | -0.003 to +0.001 | 10 | 7 | 96.5% / 96.5% | 3744 | -0.001 | -0.003 to +0.001 |

