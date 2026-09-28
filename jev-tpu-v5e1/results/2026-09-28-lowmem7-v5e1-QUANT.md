# 2026-09-28-lowmem7-v5e1: 4-bit against bf16, paired on the same records

## 26b-lowmem against ../jev-tpu-31b/results/2026-09-26-moe2-26b-q4w4

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.950 | 0.950 | +0.000 | +0.000 to +0.000 | 0 | 0 | 100.0% / 100.0% | 300 | +0.000 | +0.000 to +0.000 |
| ag_news | 300 | 0.860 | 0.863 | +0.003 | -0.007 to +0.013 | 1 | 2 | 7.3% / 7.7% | 22 | +0.000 | +0.000 to +0.000 |
| emotion | 300 | 0.600 | 0.603 | +0.003 | -0.007 to +0.017 | 1 | 2 | 30.3% / 29.3% | 85 | +0.000 | +0.000 to +0.000 |
| irony | 300 | 0.907 | 0.903 | -0.003 | -0.010 to +0.000 | 1 | 0 | 100.0% / 100.0% | 300 | -0.003 | -0.010 to +0.000 |
| reversed options (3 choice tasks) | 900 | 0.798 | 0.797 | -0.001 | -0.004 to +0.002 | 2 | 1 | 51.8% / 50.9% | 457 | -0.002 | -0.009 to +0.004 |
| suite (all subsets) | 3880 | 0.753 | 0.754 | +0.001 | -0.004 to +0.005 | 36 | 39 | 69.8% / 69.7% | 2688 | +0.001 | -0.005 to +0.007 |

