# 2026-09-26-override: 4-bit against bf16, paired on the same records

## 12b-bf16-ovr against ../jev-tpu/results/2026-09-24-v6e1-12b

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.950 | 0.950 | +0.000 | +0.000 to +0.000 | 0 | 0 | 79.7% / 80.7% | 239 | +0.000 | +0.000 to +0.000 |
| ag_news | 300 | 0.863 | 0.863 | +0.000 | +0.000 to +0.000 | 0 | 0 | 78.3% / 77.7% | 232 | +0.000 | +0.000 to +0.000 |
| emotion | 300 | 0.593 | 0.593 | +0.000 | +0.000 to +0.000 | 0 | 0 | 83.7% / 84.0% | 251 | +0.000 | +0.000 to +0.000 |
| irony | 300 | 0.847 | 0.847 | +0.000 | +0.000 to +0.000 | 0 | 0 | 100.0% / 100.0% | 300 | +0.000 | +0.000 to +0.000 |
| reversed options (3 choice tasks) | 900 | 0.796 | 0.796 | +0.000 | -0.003 to +0.003 | 1 | 1 | 79.1% / 80.0% | 711 | +0.000 | -0.004 to +0.004 |
| suite (all subsets) | 3880 | 0.762 | 0.760 | -0.003 | -0.006 to +0.001 | 28 | 17 | - / 78.0% | - | - | - |

