### -t 4 -C 0x55  (4-distinct-P-no-SMT)
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       4 | 0x55       |          1 |   1 |           pp512 |        135.52 ± 2.69 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       4 | 0x55       |          1 |   1 |           tg128 |         24.89 ± 0.59 |
### -t 4 -C 0x0F  (2-P-both-SMT-siblings)
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       4 | 0x0F       |          1 |   1 |           pp512 |         73.50 ± 2.89 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       4 | 0x0F       |          1 |   1 |           tg128 |         21.66 ± 0.83 |
### -t 8 -C 0xFF00  (8-E-cores-only)
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       8 | 0xFF00     |          1 |   1 |           pp512 |         86.61 ± 3.75 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       8 | 0xFF00     |          1 |   1 |           tg128 |         19.17 ± 0.63 |
### -t 12 -C 0xFF55  (4-distinct-P-plus-8-E)
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |      12 | 0xFF55     |          1 |   1 |           pp512 |        116.77 ± 2.18 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |      12 | 0xFF55     |          1 |   1 |           tg128 |         24.96 ± 1.37 |
### -t 8 -C 0xFF  (4-P-cores-SMT-doubled — the clean SMT control)
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       8 | 0xFF       |          1 |   1 |           pp512 |        139.77 ± 5.71 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CPU        |       8 | 0xFF       |          1 |   1 |           tg128 |         23.49 ± 0.39 |
