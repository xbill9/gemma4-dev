== host: nvidia-smi 1480 MiB
I load_tensors:   CPU_Mapped model buffer size =  1476.00 MiB
I load_tensors:        CUDA0 model buffer size =  1223.91 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    48.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    12.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   122.52 MiB
I sched_reserve:  CUDA_Host compute buffer size =    32.52 MiB
== card: nvidia-smi 2732 MiB
I load_tensors:   CPU_Mapped model buffer size =   216.00 MiB
I load_tensors:        CUDA0 model buffer size =  2483.91 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    48.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    12.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   115.52 MiB
I sched_reserve:  CUDA_Host compute buffer size =    15.02 MiB

gpu before pass 1 (host): 44, 1890 MHz, 16.62 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        347.96 ± 1.11 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |          pp2048 |        319.43 ± 0.59 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         82.17 ± 0.03 |

build: f95b0d9 (318)
gpu after: 54, 1875 MHz, 17.65 W

gpu before pass 2 (card): 50, 300 MHz, 2.34 W
| model                          |       size |     params | backend    | ngl |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------------- | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           pp512 |        345.68 ± 0.72 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |          pp2048 |        319.57 ± 0.13 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           tg128 |         82.16 ± 0.06 |

build: f95b0d9 (318)
gpu after: 55, 1875 MHz, 18.09 W

gpu before pass 3 (card): 49, 300 MHz, 2.07 W
| model                          |       size |     params | backend    | ngl |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------------- | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           pp512 |        345.84 ± 0.70 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |          pp2048 |        319.68 ± 0.11 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           tg128 |         82.05 ± 0.03 |

build: f95b0d9 (318)
gpu after: 57, 1875 MHz, 17.34 W

gpu before pass 4 (host): 50, 300 MHz, 2.15 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        345.00 ± 1.14 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |          pp2048 |        319.36 ± 0.11 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         81.96 ± 0.07 |

build: f95b0d9 (318)
gpu after: 57, 1860 MHz, 17.66 W

