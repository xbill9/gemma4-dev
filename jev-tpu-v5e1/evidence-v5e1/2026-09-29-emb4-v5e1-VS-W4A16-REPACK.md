# 2026-09-29-emb4-v5e1: 4-bit against bf16, paired on the same records

## e2b-emb4 against results/2026-09-29-e2br-v5e1-e2b-repack

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.893 | 0.890 | -0.003 | -0.010 to +0.000 | 1 | 0 | 100.0% / 100.0% | 300 | -0.003 | -0.010 to +0.000 |
| ag_news | 300 | 0.573 | 0.560 | -0.013 | -0.027 to -0.003 | 4 | 0 | 100.0% / 100.0% | 300 | -0.013 | -0.027 to -0.003 |
| emotion | 300 | 0.460 | 0.463 | +0.003 | -0.007 to +0.013 | 1 | 2 | 100.0% / 100.0% | 300 | +0.003 | -0.007 to +0.013 |
| irony | 300 | 0.787 | 0.793 | +0.007 | -0.007 to +0.020 | 1 | 3 | 100.0% / 100.0% | 300 | +0.007 | -0.007 to +0.020 |
| reversed options (3 choice tasks) | 900 | 0.611 | 0.616 | +0.004 | -0.001 to +0.010 | 1 | 5 | 100.0% / 100.0% | 900 | +0.004 | -0.001 to +0.010 |
| suite (all subsets) | 3880 | 0.678 | 0.681 | +0.003 | -0.001 to +0.006 | 17 | 27 | 96.5% / 96.3% | 3734 | +0.003 | -0.001 to +0.006 |

