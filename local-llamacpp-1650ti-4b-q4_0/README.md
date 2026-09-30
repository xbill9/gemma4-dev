# local-llamacpp-1650ti-4b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), driven directly,
serving an **exact Q4_0 GGUF of Gemma 4 E4B-it** off one **NVIDIA GTX 1650 Ti (Max-Q)**, 4096 MiB.
The E4B parallel of [`local-llamacpp-1650ti-2b-q4_0`](../local-llamacpp-1650ti-2b-q4_0/); slot 4 is the
only difference.

**Status 2026-09-30: serves.** 2848 MiB of 4096 at `-ngl 99 -c 8192`, **~39.8 tok/s decode, ~173 t/s
prefill** (llama-bench). The exact rebuild is 1.06x Google's E4B GGUF on decode and 204 MiB smaller in
VRAM. It fits because the 1.59 GB per-layer embedding table stays in the mmap. See
`benchmarks/runs/2026-09-30-exact-gguf-e4b-1650ti/REPORT.md`.

| | |
| --- | --- |
| Platform | `local` — no control plane; the card is in this machine |
| Runtime | `llamacpp` — one process, one GGUF named on the command line |
| Hardware | `1650ti` — TU117, compute capability 7.5, 4096 MiB VRAM |
| Model | `4b` — `google/gemma-4-E4B-it` (4.5B effective, 8.0B total) |
| Encoding | `q4_0` — exact rebuild from the QAT unquantized checkpoint |

## Quick start

```bash
make install     # pip install -r requirements.txt into the system python3
make info        # resident-vs-lazy split, read off the GGUF
make serve       # llama-server in the foreground on 127.0.0.1:8080 — stop the E2B rig first
make query       # one chat completion
make test lint
```

The E2B rig serves on the same port and the two do not fit on the card together. `tpu.env` is the
source of truth; `CLAUDE.md` says what was measured here and what was carried over from E2B.
