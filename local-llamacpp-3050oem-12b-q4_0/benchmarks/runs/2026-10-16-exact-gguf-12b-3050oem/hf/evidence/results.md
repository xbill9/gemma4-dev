llama.cpp 7182ef71e
== google-32k (-c 32768, f16): DID NOT LOAD
0.01.053.789 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
/home/xbill/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:109: CUDA error
0.17.145.248 E CUDA error: out of memory
== exact-32k (-c 32768, f16): nvidia-smi 7636 MiB
I load_tensors:   CPU_Mapped model buffer size =   540.00 MiB
I load_tensors:        CUDA0 model buffer size =  6390.19 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =   512.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =   480.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   153.52 MiB
I sched_reserve:  CUDA_Host compute buffer size =    63.53 MiB
== exact-49152-f16 (-c 49152, f16): DID NOT LOAD
0.00.926.890 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
0.02.912.028 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 169.52 MiB on device 0: cudaMalloc failed: out of memory
0.02.912.041 E ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 177758336
== exact-65536-q8 (-c 65536, q8_0): nvidia-smi 7550 MiB
I load_tensors:   CPU_Mapped model buffer size =   540.00 MiB
I load_tensors:        CUDA0 model buffer size =  6390.19 MiB
I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =   544.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =   255.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   256.80 MiB
I sched_reserve:  CUDA_Host compute buffer size =    96.81 MiB
== exact-98304-q8 (-c 98304, q8_0): DID NOT LOAD
0.00.949.345 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
0.02.938.314 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 352.80 MiB on device 0: cudaMalloc failed: out of memory
0.02.938.319 E ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 369942656
== exact-32768-f16-draft (-c 32768, f16): DID NOT LOAD
0.01.578.278 E llama_init_from_model: failed to initialize the context: Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)
0.01.652.418 W operator(): failed to measure the memory of the extra model, fitting without it: failed to create llama_context from model
0.01.652.443 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort

gpu before pass 1 (google): 39, 1905 MHz, 35.10 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 ?B Q4_0                 |   6.48 GiB |    11.91 B | CUDA       |  99 |   1 |           pp512 |        981.57 ± 8.12 |
| gemma4 ?B Q4_0                 |   6.48 GiB |    11.91 B | CUDA       |  99 |   1 |           tg128 |         27.68 ± 0.02 |

build: 7182ef71e (11699)
gpu after: 57, 1905 MHz, 116.70 W

gpu before pass 2 (exact): 40, 210 MHz, 16.06 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |           pp512 |        987.49 ± 9.98 |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |           tg128 |         28.64 ± 0.02 |

build: 7182ef71e (11699)
gpu after: 60, 1897 MHz, 112.92 W

gpu before pass 3 (exact): 40, 210 MHz, 15.98 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |           pp512 |        986.97 ± 9.98 |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |           tg128 |         28.64 ± 0.02 |

build: 7182ef71e (11699)
gpu after: 61, 1897 MHz, 116.67 W

gpu before pass 4 (google): 40, 210 MHz, 15.93 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 ?B Q4_0                 |   6.48 GiB |    11.91 B | CUDA       |  99 |   1 |           pp512 |        977.83 ± 8.00 |
| gemma4 ?B Q4_0                 |   6.48 GiB |    11.91 B | CUDA       |  99 |   1 |           tg128 |         27.68 ± 0.01 |

build: 7182ef71e (11699)
gpu after: 61, 1897 MHz, 118.53 W

depth sweep, exact, gpu before: 40, 210 MHz, 16.05 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |           pp512 |       981.09 ± 10.18 |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |           tg128 |         28.63 ± 0.03 |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |   pp512 @ d8192 |        801.58 ± 4.46 |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |   tg128 @ d8192 |         26.81 ± 0.02 |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |  pp512 @ d16384 |        692.32 ± 3.00 |
| gemma4 ?B Q4_0                 |   6.24 GiB |    11.91 B | CUDA       |  99 |   1 |  tg128 @ d16384 |         26.33 ± 0.02 |
gpu after: 64, 1875 MHz, 109.41 W

gpu before pass 5 (e4b-exact): 40, 210 MHz, 15.83 W
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           pp512 |      2240.65 ± 47.79 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | CUDA       |  99 |   1 |           tg128 |         61.19 ± 0.15 |

build: 7182ef71e (11699)
gpu after: 49, 1912 MHz, 91.59 W

