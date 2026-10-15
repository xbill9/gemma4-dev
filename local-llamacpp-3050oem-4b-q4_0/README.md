# local-llamacpp-3050oem-4b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving the exact Q4_0 GGUF of Gemma 4 **E4B**-it
(`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`) off one **NVIDIA RTX 3050 OEM**, with
Google's E4B drafter for speculative decoding.

**Status 2026-10-13: serving.** About **100 tok/s** generation with the drafter
(58 without), 2,246 t/s prompt reading, full 131,072-token context, **6,662 MiB of
an 8,192 MiB card**. Greedy output with the drafter matches greedy output without it for short prompts; with long text ahead it differs.

| | |
| --- | --- |
| Platform | `local` — no control plane; the card is in this machine |
| Runtime | `llamacpp` — llama.cpp `77c4540ce` |
| Hardware | `3050oem` — GA106, Ampere, compute capability 8.6, 8192 MiB; host i7-5820K |
| Model | `4b` — `google/gemma-4-E4B-it` |
| Encoding | `q4_0` — every weight matrix Q4_0 |

The E4B sister of [`local-llamacpp-3050oem-2b-q4_0`](../local-llamacpp-3050oem-2b-q4_0/):
same card, binary and flags. Both use port 8080 and they don't fit on the card
together, so run one at a time.

## Quick start

```bash
make install                          # pip install -r requirements.txt into the system python3
export PATH=/usr/local/cuda/bin:$PATH # nvcc is not on PATH on this host
make build                            # llama.cpp with CUDA for sm_86 (shared with the E2B rig)
make download                         # the E4B GGUF to MODEL_PATH, sha256-checked
make draft                            # the exact Q4_0 E4B drafter, sha256-checked
make serve                            # llama-server in the foreground on 0.0.0.0:8080
make query                            # one chat completion
```

`tpu.env` is the source of truth. Results: `benchmarks/runs/`.
