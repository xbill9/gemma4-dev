== -c 8192: nvidia-smi 2770 MiB
I load_tensors:        CUDA0 model buffer size =  2483.91 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    48.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    12.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   128.02 MiB
== -c 32768: nvidia-smi 2938 MiB
I load_tensors:        CUDA0 model buffer size =  2483.91 MiB
I llama_kv_cache:      CUDA0 KV buffer size =   192.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    12.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   152.02 MiB
== -c 131072: nvidia-smi 3610 MiB
I load_tensors:        CUDA0 model buffer size =  2483.91 MiB
I llama_kv_cache:      CUDA0 KV buffer size =   768.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =    12.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   248.02 MiB

| model                          |       size |     params | backend    | ngl |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------------- | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           pp512 |     3959.63 ± 154.81 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |           tg128 |        116.02 ± 0.29 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |   pp512 @ d8192 |      2898.48 ± 46.24 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |   tg128 @ d8192 |        107.06 ± 0.31 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |  pp512 @ d32768 |      1690.19 ± 19.03 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |  tg128 @ d32768 |         90.68 ± 0.23 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |  pp512 @ d65536 |       1079.92 ± 6.25 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 |  tg128 @ d65536 |         75.02 ± 0.15 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 | pp512 @ d120000 |        669.25 ± 2.33 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 | per_layer_token_embd\.weight=CUDA0 | tg128 @ d120000 |         58.14 ± 0.04 |

build: 77c4540ce (11603)
61, 1882 MHz, 85.28 W
