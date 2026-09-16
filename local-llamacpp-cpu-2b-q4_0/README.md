# local-llamacpp-cpu-2b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving `google/gemma-4-E2B-it-qat-q4_0-gguf` on the **CPU only**
of a local workstation.

**Status 2026-09-16: serving.** Built, downloaded and smoke-tested at 24.8 tok/s on
a single completion. Thread and affinity levers swept with `llama-bench`
(`benchmarks/runs/2026-09-16-thread-sweep-cpu`); no serving benchmark yet.

| | |
| --- | --- |
| Platform | `local` — no control plane; the hardware is in this machine |
| Runtime | `llamacpp` — one process, one GGUF named on the command line |
| Hardware | `cpu` — no accelerator; i7-1360P, AVX2 + AVX-VNNI, 15 GiB RAM |
| Model | `2b` — `google/gemma-4-E2B-it` |
| Encoding | `q4_0` — the QAT GGUF export |

## Quick start

```bash
make install     # pip install -r requirements.txt into the system python3
pip install cmake huggingface_hub
make build       # CPU-only llama.cpp (GGML_CUDA=OFF) at ~/llama.cpp/build-cpu
make download    # the GGUF, into MODEL_PATH's directory — public repo, no login
make info        # resident-vs-lazy split, read off the artifact
make serve       # llama-server in the foreground, -ngl 0, no CUDA device visible
make query       # one chat completion against 127.0.0.1:8080 — slow on a CPU
```

`tpu.env` is the source of truth. The directory name is documentation — never copy
a slot value into a flag.

## CPU only, enforced

`-ngl 0` and an empty `CUDA_VISIBLE_DEVICES` are hardcoded rather than configured,
and `start_model_server` refuses a llama-server binary built with a GPU backend.
Tests assert all three.

## Nothing tuned on the GPU carries over

The thread, KV-cache and flash-attention settings the 1650 Ti sibling measured were
CUDA results. Here they are starting guesses (`-t 4 -tb 8 -ctk f16 -fa 1`) waiting
for a sweep. See `CLAUDE.md`.

## Layout

```
tpu.env            source of truth — model, host, endpoint, serving flags
server.py          MCP server (`MCPServer`): CPU/model info, start/stop, status, query
inspect_gguf.py    re-derives the resident-vs-lazy split from the artifact
sweep.py           context x concurrency benchmark harness
Makefile           build / download / serve / status / query / info / test / lint
tests/             offline unittest suite
benchmarks/        schema + README synced from the monorepo root
```

## See also

`../MODELS.md` (E2B structure, KV cost), `../NAMING.md` (the `local` platform and
slot 3), `../local-llamacpp-1650ti-2b-q4_0/` (the same artifact and runtime on the
GPU, with measurements), `CLAUDE.md` (working notes for this rig).
