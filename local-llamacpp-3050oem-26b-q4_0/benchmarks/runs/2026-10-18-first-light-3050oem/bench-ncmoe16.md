ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7891 MiB):
  Device 0: NVIDIA GeForce RTX 3050 OEM, compute capability 8.6, VMM: yes, VRAM: 7891 MiB
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 544.14 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 570572800
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
llama_bench: error: failed to create context with model '/home/xbill/models/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf'
| model                          |       size |     params | backend    | ngl |  n_cpu_moe |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --: | --------------: | -------------------: |
