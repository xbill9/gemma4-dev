# 2026-09-30-fillC-v5e1: 4-bit against bf16, paired on the same records

## 26b-emb4 against ../jev-tpu-31b/results/2026-09-26-moe2-26b-q4w4

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.950 | 0.947 | -0.003 | -0.010 to +0.000 | 1 | 0 | 100.0% / 100.0% | 300 | -0.003 | -0.010 to +0.000 |
| ag_news | 300 | 0.860 | 0.857 | -0.003 | -0.010 to +0.000 | 1 | 0 | 7.3% / 7.7% | 22 | -0.045 | -0.136 to +0.000 |
| emotion | 300 | 0.600 | 0.603 | +0.003 | +0.000 to +0.010 | 0 | 1 | 30.3% / 28.7% | 84 | +0.000 | +0.000 to +0.000 |
| irony | 300 | 0.907 | 0.907 | +0.000 | -0.010 to +0.010 | 1 | 1 | 100.0% / 100.0% | 300 | +0.000 | -0.010 to +0.010 |
| reversed options (3 choice tasks) | 900 | 0.798 | 0.796 | -0.002 | -0.006 to +0.000 | 2 | 0 | 51.8% / 51.8% | 459 | -0.004 | -0.011 to +0.000 |
| suite (all subsets) | 3880 | 0.753 | 0.755 | +0.001 | -0.003 to +0.005 | 30 | 35 | 69.8% / 70.4% | 2694 | +0.001 | -0.004 to +0.007 |

