ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7891 MiB):
  Device 0: NVIDIA GeForce RTX 3050 OEM, compute capability 8.6, VMM: yes, VRAM: 7891 MiB
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------: | -------------------: |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | CUDA       |  20 |       8 |      512 |   1 |          pp2048 |        145.30 ± 0.18 |
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1108.01 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 1161834496
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
llama_bench: error: failed to create context with model '/home/xbill/models/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf'
