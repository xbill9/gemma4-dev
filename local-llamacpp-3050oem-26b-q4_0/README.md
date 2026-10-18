# local-llamacpp-3050oem-26b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving Google's QAT Q4_0 GGUF of Gemma 4 **26B A4B**-it
(`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`) off one **NVIDIA RTX 3050 OEM** with the
experts of 21 of its 30 layers in system RAM, and the exact Q4_0 build of Google's 26B
drafter for speculative decoding. Text only.

**Status 2026-10-18: first light.** The 14.4 GB model on an 8 GB card: **131,072-token
context** with a q4_0 cache and the drafter, **7,850 MiB**. Code generates at 40.6 tok/s with
nothing in context and 23.6 at 98k; prose at 27.9 and 16.6
(`benchmarks/runs/2026-10-18-first-light-3050oem`). Without the drafter and with 22 expert
layers in RAM, the full 262,144 tokens fit.

| | |
| --- | --- |
| Platform | `local` — no control plane; the card is in this machine |
| Runtime | `llamacpp` — llama.cpp `7182ef71e` |
| Hardware | `3050oem` — GA106, Ampere, compute capability 8.6, 8192 MiB; host i7-5820K, 46 GiB |
| Model | `26b` — `google/gemma-4-26B-A4B-it` |
| Encoding | `q4_0` — every weight matrix Q4_0 except `token_embd` (Q6_K) |

The 26B sister of [`local-llamacpp-3050oem-12b-q4_0`](../local-llamacpp-3050oem-12b-q4_0/),
[`-4b-q4_0`](../local-llamacpp-3050oem-4b-q4_0/) and [`-2b-q4_0`](../local-llamacpp-3050oem-2b-q4_0/):
same card. All use port 8080 and only one fits on the card, so run one at a time.

## Quick start

```bash
make install                          # pip install -r requirements.txt into the system python3
export PATH=/usr/local/cuda/bin:$PATH # nvcc is not on PATH on this host
make build                            # llama.cpp with CUDA for sm_86 (shared with the sisters)
make download                         # the 26B GGUF to MODEL_PATH, sha256-checked
make draft                            # the exact Q4_0 26B drafter, sha256-checked
make serve                            # llama-server in the foreground on 0.0.0.0:8080
make query                            # one chat completion
```

`tpu.env` is the source of truth; `N_CPU_MOE` sets how many layers' experts stay in RAM.
Results: `benchmarks/runs/`.
