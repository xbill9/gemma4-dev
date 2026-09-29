gpu before pass 1 (google first): 46, 300 MHz, 2.56 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        340.09 ± 0.76 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         73.63 ± 0.10 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        339.86 ± 0.59 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         80.45 ± 0.03 |

build: f95b0d9 (318)
gpu after pass 1: 55, 1875 MHz, 17.31 W

gpu before pass 2 (exact first): 50, 300 MHz, 4.40 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        339.75 ± 0.86 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         80.57 ± 0.06 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        338.69 ± 0.50 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         73.22 ± 0.09 |

build: f95b0d9 (318)
gpu after pass 2: 57, 1860 MHz, 17.37 W

gpu before pass 3 (exact first): 50, 300 MHz, 2.40 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        339.83 ± 0.66 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         80.47 ± 0.09 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        337.59 ± 0.50 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         73.14 ± 0.03 |

build: f95b0d9 (318)
gpu after pass 3: 57, 1860 MHz, 17.92 W

gpu before pass 4 (google first): 49, 300 MHz, 4.37 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        338.72 ± 0.74 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         73.54 ± 0.05 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        339.94 ± 0.66 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         80.44 ± 0.07 |

build: f95b0d9 (318)
gpu after pass 4: 56, 1860 MHz, 17.83 W

