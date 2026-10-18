ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7891 MiB):
  Device 0: NVIDIA GeForce RTX 3050 OEM, compute capability 8.6, VMM: yes, VRAM: 7891 MiB
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------: | -------------------: |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  12 |       8 |     2048 |   1 |          pp2048 |        238.29 ± 1.18 |

build: 7182ef71e (11699)
