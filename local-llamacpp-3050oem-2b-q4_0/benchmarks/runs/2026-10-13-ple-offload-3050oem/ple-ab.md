== host: nvidia-smi 1510 MiB
I load_tensors:   CPU_Mapped model buffer size =  1260.00 MiB
I load_tensors:   CPU_Mapped model buffer size =   216.00 MiB
I load_tensors:        CUDA0 model buffer size =  1223.91 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    48.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    12.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   128.02 MiB
I sched_reserve:  CUDA_Host compute buffer size =    38.53 MiB
== card: nvidia-smi 2770 MiB
I load_tensors:   CPU_Mapped model buffer size =   216.00 MiB
I load_tensors:        CUDA0 model buffer size =  2483.91 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    48.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    12.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   128.02 MiB
I sched_reserve:  CUDA_Host compute buffer size =    21.03 MiB

gpu before pass 1 (host): 34, 1927 MHz, 32.93 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |     3856.99 ± 112.44 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |          pp2048 |      3767.65 ± 12.95 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |        116.01 ± 0.21 |

build: 77c4540ce (11603)
gpu after: 40, 1897 MHz, 80.48 W

gpu before pass 2 (card): 40, 1897 MHz, 80.48 W
| model                          |       size |     params | backend    | ngl |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------------- | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           pp512 |     3986.60 ± 120.44 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |          pp2048 |      3853.13 ± 16.67 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           tg128 |        116.08 ± 0.24 |

build: 77c4540ce (11603)
gpu after: 43, 1897 MHz, 80.31 W

gpu before pass 3 (card): 38, 210 MHz, 15.81 W
| model                          |       size |     params | backend    | ngl |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------------- | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           pp512 |     3978.74 ± 119.79 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |          pp2048 |      3843.71 ± 14.64 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           tg128 |        115.89 ± 0.21 |

build: 77c4540ce (11603)
gpu after: 45, 1890 MHz, 81.42 W

gpu before pass 4 (host): 40, 210 MHz, 17.17 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |     3833.36 ± 108.89 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |          pp2048 |      3751.08 ± 17.70 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |        115.82 ± 0.21 |

build: 77c4540ce (11603)
gpu after: 47, 1890 MHz, 83.10 W

gpu before pass 5 (host-ub2048): 40, 210 MHz, 15.90 W
| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |     2048 |   1 |           pp512 |     3836.26 ± 109.24 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |     2048 |   1 |          pp2048 |      3747.86 ± 15.92 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |     2048 |   1 |           tg128 |        115.70 ± 0.23 |

build: 77c4540ce (11603)
gpu after: 47, 1890 MHz, 84.73 W

gpu before pass 6 (card-ub2048): 40, 210 MHz, 15.83 W
| model                          |       size |     params | backend    | ngl | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------------- | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |     2048 |   1 | per_layer_token_embd\.weight=CUDA0 |           pp512 |     3974.02 ± 117.66 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |     2048 |   1 | per_layer_token_embd\.weight=CUDA0 |          pp2048 |      3882.90 ± 16.14 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |     2048 |   1 | per_layer_token_embd\.weight=CUDA0 |           tg128 |        115.67 ± 0.23 |

build: 77c4540ce (11603)
gpu after: 47, 1890 MHz, 80.39 W

