# 2026-09-26-kvbf16: 4-bit against bf16, paired on the same records

## 26b-q4w4-kvbf16 against results/2026-09-26-moe2-26b-q4w4

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.950 | 0.950 | +0.000 | +0.000 to +0.000 | 0 | 0 | 100.0% / 100.0% | 300 | +0.000 | +0.000 to +0.000 |
| ag_news | 300 | 0.860 | 0.860 | +0.000 | +0.000 to +0.000 | 0 | 0 | 7.3% / 7.3% | 22 | +0.000 | +0.000 to +0.000 |
| emotion | 300 | 0.600 | 0.600 | +0.000 | +0.000 to +0.000 | 0 | 0 | 30.3% / 30.3% | 91 | +0.000 | +0.000 to +0.000 |
| irony | 300 | 0.907 | 0.907 | +0.000 | +0.000 to +0.000 | 0 | 0 | 100.0% / 100.0% | 300 | +0.000 | +0.000 to +0.000 |
| reversed options (3 choice tasks) | 900 | 0.798 | 0.798 | +0.000 | +0.000 to +0.000 | 0 | 0 | 51.8% / 51.8% | 466 | +0.000 | +0.000 to +0.000 |
| suite (all subsets) | 3880 | 0.753 | 0.755 | +0.002 | -0.001 to +0.005 | 13 | 21 | 69.8% / 69.8% | 2707 | +0.003 | -0.001 to +0.008 |

