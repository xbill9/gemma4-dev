gpu before pass 1 (order G,X): 44, 1035 MHz, 11.57 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   4.79 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |        170.63 ± 0.26 |
| gemma4 E4B Q4_0                |   4.79 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         37.56 ± 0.07 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |        172.11 ± 0.16 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         39.75 ± 0.02 |

build: f95b0d9 (318)
gpu after: 56, 1860 MHz, 17.93 W

gpu before pass 2 (order X,G): 49, 300 MHz, 2.14 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |        173.45 ± 0.23 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         39.94 ± 0.01 |
| gemma4 E4B Q4_0                |   4.79 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |        170.31 ± 0.18 |
| gemma4 E4B Q4_0                |   4.79 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         37.58 ± 0.02 |

build: f95b0d9 (318)
gpu after: 54, 1860 MHz, 17.57 W

gpu before pass 3 (order X,G): 50, 300 MHz, 2.19 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |        173.46 ± 0.24 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         39.92 ± 0.02 |
| gemma4 E4B Q4_0                |   4.79 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |        169.01 ± 0.18 |
| gemma4 E4B Q4_0                |   4.79 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         37.56 ± 0.02 |

build: f95b0d9 (318)
gpu after: 55, 1860 MHz, 16.94 W

gpu before pass 4 (order G,X): 50, 300 MHz, 2.19 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   4.79 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |        170.34 ± 0.23 |
| gemma4 E4B Q4_0                |   4.79 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         37.55 ± 0.04 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |        172.18 ± 0.16 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         39.75 ± 0.02 |

build: f95b0d9 (318)
gpu after: 57, 1860 MHz, 17.29 W

