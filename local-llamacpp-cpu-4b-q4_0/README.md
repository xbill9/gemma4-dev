# local-llamacpp-cpu-4b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), CPU-only build,
serving the exact Q4_0 GGUF of **Gemma 4 E4B-it**
([`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf))
on the i7-10750H of the machine under the desk. The CPU arm of a paired control with
[`local-llamacpp-1650ti-4b-q4_0`](../local-llamacpp-1650ti-4b-q4_0/); forked 2026-09-30 from
[`local-llamacpp-cpu-2b-q4_0`](../local-llamacpp-cpu-2b-q4_0/), with slot 4 the only difference.

| | |
| --- | --- |
| Platform | `local` — no control plane |
| Runtime | `llamacpp` — CPU-only build, `-ngl 0`, CUDA hidden |
| Hardware | `cpu` — i7-10750H, 6 cores, AVX2, 15 GiB |
| Model | `4b` — `google/gemma-4-E4B-it` (4.5B effective, 8.0B total) |
| Encoding | `q4_0` — exact rebuild from the QAT unquantized checkpoint |

```bash
make download    # fetch the GGUF and check its SHA-256 against tpu.env
make serve       # port 8080 — the GPU arm and the E2B rigs use it too; stop them first
make test lint
```
