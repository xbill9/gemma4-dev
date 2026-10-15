== host: nvidia-smi 4938 MiB
I print_info: n_ctx_train           = 131072
I load_tensors:   CPU_Mapped model buffer size =  1512.00 MiB
I load_tensors:   CPU_Mapped model buffer size =   360.00 MiB
I load_tensors:        CUDA0 model buffer size =  2493.32 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =  2048.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    40.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   256.02 MiB
I sched_reserve:  CUDA_Host compute buffer size =   170.03 MiB
== card: nvidia-smi 6430 MiB
I print_info: n_ctx_train           = 131072
I load_tensors:   CPU_Mapped model buffer size =   360.00 MiB
I load_tensors:        CUDA0 model buffer size =  4005.32 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =  2048.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    40.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   236.02 MiB
I sched_reserve:  CUDA_Host compute buffer size =   149.03 MiB
== card-draft: nvidia-smi 6662 MiB
I print_info: n_ctx_train           = 131072
I load_tensors:   CPU_Mapped model buffer size =   360.00 MiB
I load_tensors:        CUDA0 model buffer size =  4005.32 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =  2048.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    40.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   236.02 MiB
I sched_reserve:  CUDA_Host compute buffer size =   149.03 MiB
I print_info: n_ctx_train           = 131072
I load_tensors:   CPU_Mapped model buffer size =    68.00 MiB
I load_tensors:        CUDA0 model buffer size =    79.05 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   149.00 MiB
I sched_reserve:  CUDA_Host compute buffer size =   139.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   149.00 MiB
I sched_reserve:  CUDA_Host compute buffer size =   139.00 MiB

gpu before pass 1 (host): 34, 1927 MHz, 32.19 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |      2201.97 ± 46.16 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |          pp2048 |       2177.44 ± 7.65 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         60.45 ± 0.08 |

build: 77c4540ce (11603)
gpu after: 44, 1897 MHz, 84.57 W

gpu before pass 2 (card): 39, 210 MHz, 15.73 W
| model                          |       size |     params | backend    | ngl |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------------- | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           pp512 |      2245.78 ± 45.31 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |          pp2048 |       2202.28 ± 7.61 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           tg128 |         60.46 ± 0.09 |

build: 77c4540ce (11603)
gpu after: 48, 1890 MHz, 92.57 W

gpu before pass 3 (card): 40, 210 MHz, 15.57 W
| model                          |       size |     params | backend    | ngl |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------------- | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           pp512 |      2245.29 ± 48.20 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |          pp2048 |      2214.74 ± 15.27 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           tg128 |         60.61 ± 0.08 |

build: 77c4540ce (11603)
gpu after: 50, 1912 MHz, 92.86 W

gpu before pass 4 (host): 40, 210 MHz, 15.54 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |      2190.32 ± 45.08 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |          pp2048 |      2179.16 ± 14.85 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         60.57 ± 0.09 |

build: 77c4540ce (11603)
gpu after: 50, 1912 MHz, 90.61 W

depth sweep, gpu before: 40, 210 MHz, 15.62 W
| model                          |       size |     params | backend    | ngl |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------------- | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           pp512 |      2230.50 ± 61.26 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           tg128 |         60.41 ± 0.11 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |   pp512 @ d8192 |      1860.37 ± 24.83 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |   tg128 @ d8192 |         56.00 ± 0.11 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |  pp512 @ d32768 |      1263.10 ± 11.18 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |  tg128 @ d32768 |         47.05 ± 0.08 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |  pp512 @ d65536 |        883.36 ± 5.69 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |  tg128 @ d65536 |         38.80 ± 0.05 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 | pp512 @ d120000 |        591.45 ± 2.24 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 | tg128 @ d120000 |         30.04 ± 0.01 |

build: 77c4540ce (11603)
gpu after: 59, 1882 MHz, 82.74 W
