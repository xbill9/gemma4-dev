# local-llamacpp-3050oem-2b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving the exact Q4_0 GGUF of Gemma 4 E2B-it
(`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`) off one **NVIDIA RTX 3050 OEM**
in a desktop workstation.

**Status 2026-10-13: serving.** First light: **116.14 tok/s decode and 3,860 t/s
prefill** (llama-bench tg128 / pp512), **1510 MiB of an 8192 MiB card**. Write-up in
`benchmarks/runs/2026-10-13-first-light-3050oem/REPORT.md`.

| | |
| --- | --- |
| Platform | `local` — no control plane; the card is in this machine |
| Runtime | `llamacpp` — one process, one GGUF named on the command line, llama.cpp `77c4540ce` |
| Hardware | `3050oem` — GA106, Ampere, compute capability 8.6, 8192 MiB; host i7-5820K, 46 GiB |
| Model | `2b` — `google/gemma-4-E2B-it` |
| Encoding | `q4_0` — every weight matrix Q4_0, embeddings included |

Forked from [`local-llamacpp-1650ti-2b-q4_0`](../local-llamacpp-1650ti-2b-q4_0/):
same runtime, same file by SHA-256, same serving flags, different card and host.

## Quick start

```bash
make install                          # pip install -r requirements.txt into the system python3
export PATH=/usr/local/cuda/bin:$PATH # nvcc is not on PATH on this host
make build                            # llama.cpp with CUDA for sm_86, -j10
make download                         # the GGUF to MODEL_PATH, sha256-checked
make draft                            # the exact Q4_0 speculative-decoding drafter, sha256-checked
make info                             # resident-vs-lazy memory split, read off the artifact
make serve                            # llama-server in the foreground; Ctrl-C is a full teardown
make status                           # is it up?
make query                            # one chat completion against 127.0.0.1:8080
```

`tpu.env` is the source of truth for all of it. The directory name is
documentation — never copy a slot value into a flag.

## Memory

The file is 2.62 GB. llama.cpp would leave `per_layer_token_embd` (1.321 GB, 51%
of the file) in the mmap, but with 8 GiB to spare `tpu.env` puts it on the card
(`OVERRIDE_TENSORS`, passed as `-ot`): 3.6% faster prefill, same decode, 2770 MiB
in use, 3610 MiB with the full 131,072-token context the rig serves.

## Speculative decoding

`make serve` runs Google's drafter for this checkpoint
(`google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant`), stored as exact Q4_0 on its
QAT grid and fetched from `xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf` by
`make draft`, as a 2-token draft-mtp speculator: about **1.39x faster generation** (1.20x on
stories, 1.85x on code), 3840 MiB in all. Greedy replies differ slightly from
serving without it; empty `SPEC_DRAFT_MODEL` in `tpu.env` to turn it off. **Never pass `--no-mmap`**: `token_embd`'s input copy still reads from it.

## Layout

```
tpu.env            source of truth — model, hardware, endpoint, serving flags
server.py          MCP server (`MCPServer`): GPU/model info, start/stop, status, query, attest
attest.py          reads which binary is serving port 8080 out of /proc
inspect_gguf.py    re-derives the resident-vs-lazy split from the artifact
sweep.py           context × concurrency serving sweep
Makefile           build / download / serve / status / query / info / test / lint
tests/             offline unittest suite
benchmarks/        schema + README synced from the monorepo root; runs/ holds this card's runs
```

There are **no provisioning tools and none is owed** — no `find_*`, no queued
resources, no teardown to remember.

## See also

`../MODELS.md`, `../HARDWARE.md`, `../NAMING.md`, `CLAUDE.md`.
