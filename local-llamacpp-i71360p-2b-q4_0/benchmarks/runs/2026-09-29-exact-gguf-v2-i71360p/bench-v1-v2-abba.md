pass 1 (v1 first), pkg 53 C
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CPU        |       8 |           pp512 |       146.60 ± 16.66 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CPU        |       8 |           tg128 |         24.14 ± 0.18 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CPU        |       8 |           pp512 |        113.30 ± 3.17 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CPU        |       8 |           tg128 |         24.07 ± 0.41 |

build: fc07d781e (11243)
after: 66 C
pass 2 (v2 first), pkg 54 C
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CPU        |       8 |           pp512 |       140.54 ± 14.22 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CPU        |       8 |           tg128 |         24.16 ± 0.38 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CPU        |       8 |           pp512 |        102.40 ± 9.37 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CPU        |       8 |           tg128 |         23.83 ± 0.76 |

build: fc07d781e (11243)
after: 69 C
