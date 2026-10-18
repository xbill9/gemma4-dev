ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7891 MiB):
  Device 0: NVIDIA GeForce RTX 3050 OEM, compute capability 8.6, VMM: yes, VRAM: 7891 MiB
| model                          |       size |     params | backend    | ngl | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --: | --------------: | -------------------: |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  22 |       4 |   1 |           pp512 |        153.73 ± 1.55 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  22 |       4 |   1 |            tg64 |          2.09 ± 0.02 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  22 |       6 |   1 |           pp512 |        154.38 ± 0.86 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  22 |       6 |   1 |            tg64 |          2.62 ± 0.00 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  22 |       8 |   1 |           pp512 |        154.12 ± 0.89 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  22 |       8 |   1 |            tg64 |          2.66 ± 0.00 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  22 |      12 |   1 |           pp512 |        153.43 ± 2.16 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  22 |      12 |   1 |            tg64 |          2.63 ± 0.03 |

build: 7182ef71e (11699)
