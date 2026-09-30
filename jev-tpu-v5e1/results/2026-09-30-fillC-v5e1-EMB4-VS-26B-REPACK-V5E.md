# 2026-09-30-fillC-v5e1: 4-bit against bf16, paired on the same records

## 26b-emb4 against results/2026-09-28-lowmem7-v5e1-26b-lowmem

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.950 | 0.947 | -0.003 | -0.010 to +0.000 | 1 | 0 | 100.0% / 100.0% | 300 | -0.003 | -0.010 to +0.000 |
| ag_news | 300 | 0.863 | 0.857 | -0.007 | -0.020 to +0.007 | 3 | 1 | 7.7% / 7.7% | 22 | -0.045 | -0.136 to +0.000 |
| emotion | 300 | 0.603 | 0.603 | +0.000 | -0.010 to +0.010 | 1 | 1 | 29.3% / 28.7% | 83 | +0.000 | +0.000 to +0.000 |
| irony | 300 | 0.903 | 0.907 | +0.003 | +0.000 to +0.010 | 0 | 1 | 100.0% / 100.0% | 300 | +0.003 | +0.000 to +0.010 |
| reversed options (3 choice tasks) | 900 | 0.797 | 0.796 | -0.001 | -0.003 to +0.000 | 1 | 0 | 50.9% / 51.8% | 456 | -0.002 | -0.007 to +0.000 |
| suite (all subsets) | 3880 | 0.754 | 0.755 | +0.001 | -0.003 to +0.004 | 28 | 30 | 69.7% / 70.4% | 2692 | +0.000 | -0.005 to +0.005 |

