# local-llamacpp-sc7180p-2b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf` on the **CPU
only** of a Samsung Galaxy Book Go (Snapdragon 7c Gen 2).

**Status 2026-10-11: serving.** Built at `f2cc7282c`, downloaded (SHA-256
verified), first-light thread/affinity cells measured with `llama-bench`
(`benchmarks/runs/2026-10-11-first-light-sc7180p`), and smoke-tested end to end at
7.78 tok/s decode. No serving sweep yet.

| | |
| --- | --- |
| Platform | `local` — no control plane; the hardware is in this machine |
| Runtime | `llamacpp` — one process, one GGUF named on the command line |
| Hardware | `sc7180p` — Snapdragon 7c Gen 2, 2 Kryo 468 Gold + 6 Silver, aarch64 with dot product, 3.4 GiB RAM, fanless |
| Model | `2b` — `google/gemma-4-E2B-it` |
| Encoding | `q4_0` — every weight matrix Q4_0, embeddings included |

## Quick start

```bash
sudo apt install cmake ninja-build python3-huggingface-hub   # no pip on this host
make build BUILD_JOBS=4   # CPU-only llama.cpp at ~/llama.cpp/build-cpu; keep -j low on 3.4 GiB
make download    # the GGUF, into MODEL_PATH's directory — public repo, no login
make info        # resident-vs-lazy split, read off the artifact
make serve       # llama-server in the foreground, -t 2 -tb 8, no GPU
make query       # one chat completion against 127.0.0.1:8080
```

`tpu.env` is the source of truth. The directory name is documentation — never copy
a slot value into a flag.

## Measured 2026-10-11

| | pp512 tok/s | tg128 tok/s |
| --- | ---: | ---: |
| 2 Gold (serving decode) | 36.36 | **8.33** |
| all 8 (serving prefill) | **46.68** | 5.17 |

Dot product lets llama.cpp repack Q4_0 for SDOT, which is why this 2-big-core
part outruns the 4-big-core Snapdragon 850 sibling — and why it holds 1.44 GB of
anonymous memory on a 3.4 GiB machine. See `CLAUDE.md`.

## Layout

```
tpu.env            source of truth — model, host, endpoint, serving flags
server.py          MCP server (`MCPServer`): CPU/model info, start/stop, status, query
inspect_gguf.py    re-derives the resident-vs-lazy split from the artifact
sweep.py           context x concurrency benchmark harness
Makefile           build / download / serve / status / query / info / test / lint
tests/             offline unittest suite
benchmarks/        schema + README synced from the monorepo root; runs/ is this host's
```

## See also

`../local-llamacpp-sdm850-2b-q4_0/` (same artifact and runtime on a Snapdragon 850,
no dot product), `../MODELS.md` (E2B structure, KV cost), `../NAMING.md` (the
`local` platform and slot 3), `CLAUDE.md` (working notes for this rig).
