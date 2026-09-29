pkg temp before pass 1: 76 C
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       8 |           pp512 |         94.27 ± 4.55 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       8 |           tg128 |         20.30 ± 0.52 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CPU        |       8 |           pp512 |         95.96 ± 3.37 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CPU        |       8 |           tg128 |         21.70 ± 0.65 |

build: fc07d781e (11243)
pkg temp after pass 1: 80 C
pkg temp before pass 2: 53 C
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CPU        |       8 |           pp512 |       145.00 ± 16.39 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CPU        |       8 |           tg128 |         24.45 ± 0.40 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       8 |           pp512 |        111.12 ± 4.72 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       8 |           tg128 |         22.76 ± 0.25 |

build: fc07d781e (11243)
pkg temp after pass 2: 67 C
