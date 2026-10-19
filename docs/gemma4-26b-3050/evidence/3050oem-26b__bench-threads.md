ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7891 MiB):
  Device 0: NVIDIA GeForce RTX 3050 OEM, compute capability 8.6, VMM: yes, VRAM: 7891 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------: | --: | --------------: | -------------------: |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |       4 |   1 |           pp512 |        408.20 ± 9.25 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |       4 |   1 |           tg128 |         28.49 ± 0.16 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |       5 |   1 |           pp512 |        400.99 ± 3.89 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |       5 |   1 |           tg128 |         31.92 ± 0.25 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |       6 |   1 |           pp512 |        403.01 ± 8.81 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |       6 |   1 |           tg128 |         33.03 ± 0.12 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |       8 |   1 |           pp512 |        402.21 ± 7.86 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |       8 |   1 |           tg128 |         33.97 ± 0.05 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |      12 |   1 |           pp512 |        399.97 ± 5.84 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | CUDA       |  99 |         17 |      12 |   1 |           tg128 |         34.65 ± 2.98 |

build: 7182ef71e (11699)
