# 2026-09-30-fillF-v5e1: 4-bit against bf16, paired on the same records

## 12b-w8a8-emb4-kvfp8 against results/2026-09-29-12b-v5e1-12b-w8a8-emb4

| group | n | bf16 | 4-bit | difference | 95% range | bf16 right, 4-bit wrong | bf16 wrong, 4-bit right | all labels returned, bf16 / 4-bit | n, both returned all | difference there | 95% range there |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| sst2 | 300 | 0.950 | 0.943 | -0.007 | -0.017 to +0.000 | 2 | 0 | 100.0% / 100.0% | 300 | -0.007 | -0.017 to +0.000 |
| ag_news | 300 | 0.860 | 0.857 | -0.003 | -0.020 to +0.010 | 3 | 2 | 83.0% / 83.7% | 248 | -0.004 | -0.024 to +0.012 |
| emotion | 300 | 0.587 | 0.583 | -0.003 | -0.020 to +0.013 | 4 | 3 | 99.3% / 99.0% | 296 | -0.003 | -0.020 to +0.014 |
| irony | 300 | 0.857 | 0.873 | +0.017 | -0.003 to +0.040 | 3 | 8 | 100.0% / 100.0% | 300 | +0.017 | -0.003 to +0.040 |
| reversed options (3 choice tasks) | 900 | 0.799 | 0.801 | +0.002 | -0.007 to +0.011 | 7 | 9 | 91.0% / 95.9% | 818 | +0.002 | -0.007 to +0.012 |
| suite (all subsets) | 3880 | 0.761 | 0.754 | -0.007 | -0.013 to -0.001 | 91 | 64 | 85.7% / 86.4% | 3274 | -0.008 | -0.014 to -0.001 |

