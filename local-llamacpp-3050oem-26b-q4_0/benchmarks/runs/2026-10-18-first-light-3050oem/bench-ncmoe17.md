ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7891 MiB):
  Device 0: NVIDIA GeForce RTX 3050 OEM, compute capability 8.6, VMM: yes, VRAM: 7891 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --: | --------------: | -------------------: |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |   1 |           pp512 |        408.16 ± 9.22 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |   1 |           tg128 |         33.31 ± 0.06 |

build: 7182ef71e (11699)
