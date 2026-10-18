ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7891 MiB):
  Device 0: NVIDIA GeForce RTX 3050 OEM, compute capability 8.6, VMM: yes, VRAM: 7891 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --: | --------------: | -------------------: |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         18 |   1 |           pp512 |        393.72 ± 8.66 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         18 |   1 |           tg128 |         32.27 ± 0.08 |

build: 7182ef71e (11699)
