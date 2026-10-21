# local-llamacpp-1650ti-12b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving the exact Q4_0 GGUF of Gemma 4 **12B**-it
(`xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf`) off one **NVIDIA GTX 1650 Ti Max-Q** (4 GiB),
with the exact Q4_0 build of Google's 12B drafter for speculative decoding. Text only.

**Status 2026-10-21: serves, too slowly for interactive chat.** The 6.24 GiB of weights do not
fit the card, so 20 of 48 layers go on it and the rest run on the CPU. With the drafter a short
question generates at **7–10 tok/s**; prompt reading is about **60 tok/s**, so a 4k-token document
takes 100 s before the first word and a 16k one 5 minutes. 32,768-token context, q4_0 cache,
3,625 MiB of 4,096 (`benchmarks/runs/2026-10-21-first-light-1650ti`). On this card E4B is the
interactive model; the 12B belongs on the RTX 3050
([`local-llamacpp-3050oem-12b-q4_0`](../local-llamacpp-3050oem-12b-q4_0/)).

| | |
| --- | --- |
| Platform | `local` — no control plane; the card is in this machine |
| Runtime | `llamacpp` — llama.cpp `7182ef71e` |
| Hardware | `1650ti` — TU117, Turing, compute capability 7.5, 4096 MiB; laptop i7-10750H, 15 GiB |
| Model | `12b` — `google/gemma-4-12B-it` |
| Encoding | `q4_0` — every weight matrix Q4_0 |

The 12B sister of [`local-llamacpp-1650ti-4b-q4_0`](../local-llamacpp-1650ti-4b-q4_0/)
and [`local-llamacpp-1650ti-2b-q4_0`](../local-llamacpp-1650ti-2b-q4_0/): same card.
All three use port 8080 and only one fits on the card, so run one at a time.

## Quick start

```bash
make install     # pip install -r requirements.txt into the system python3
make build       # llama.cpp with CUDA for sm_75 (shared with the E2B/E4B rigs)
make download    # the 12B GGUF to MODEL_PATH, sha256-checked
make draft       # the exact Q4_0 12B drafter, sha256-checked
make serve       # llama-server in the foreground on 127.0.0.1:8080
make query       # one chat completion
```

`tpu.env` is the source of truth. Results: `benchmarks/runs/`.
