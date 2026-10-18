# local-llamacpp-3050oem-12b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving the exact Q4_0 GGUF of Gemma 4 **12B**-it
(`xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf`) off one **NVIDIA RTX 3050 OEM**, with
the exact Q4_0 build of Google's 12B drafter for speculative decoding. Text only.

**Status 2026-10-18: serving.** **81,920-token context** with a q4_0 context cache and
the drafter: about **55 tok/s** generation with nothing in context, 38 at 32k and 31 at 65k
tokens, **7,853 MiB of an 8,192 MiB card**, answers level with the f16 cache. Without the
drafter the full 131,072 tokens fit at about half the speed
(`benchmarks/runs/2026-10-17-kvcache-12b-3050oem`).

| | |
| --- | --- |
| Platform | `local` — no control plane; the card is in this machine |
| Runtime | `llamacpp` — llama.cpp `7182ef71e` |
| Hardware | `3050oem` — GA106, Ampere, compute capability 8.6, 8192 MiB; host i7-5820K |
| Model | `12b` — `google/gemma-4-12B-it` |
| Encoding | `q4_0` — every weight matrix Q4_0 |

The 12B sister of [`local-llamacpp-3050oem-4b-q4_0`](../local-llamacpp-3050oem-4b-q4_0/)
and [`local-llamacpp-3050oem-2b-q4_0`](../local-llamacpp-3050oem-2b-q4_0/): same card.
All three use port 8080 and only one fits on the card, so run one at a time.

## Quick start

```bash
make install                          # pip install -r requirements.txt into the system python3
export PATH=/usr/local/cuda/bin:$PATH # nvcc is not on PATH on this host
make build                            # llama.cpp with CUDA for sm_86 (shared with the E2B/E4B rigs)
make download                         # the 12B GGUF to MODEL_PATH, sha256-checked
make draft                            # the exact Q4_0 12B drafter, sha256-checked
make serve                            # llama-server in the foreground on 0.0.0.0:8080
make query                            # one chat completion
```

`tpu.env` is the source of truth. Results: `benchmarks/runs/`.
