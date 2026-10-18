ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7891 MiB):
  Device 0: NVIDIA GeForce RTX 3050 OEM, compute capability 8.6, VMM: yes, VRAM: 7891 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  12 |   1 |           pp512 |        133.38 ± 1.25 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  12 |   1 |            tg64 |          2.13 ± 0.01 |

build: 7182ef71e (11699)
