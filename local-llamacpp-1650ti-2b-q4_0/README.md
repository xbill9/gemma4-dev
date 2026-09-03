# local-llamacpp-1650ti-2b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving `google/gemma-4-E2B-it-qat-q4_0-gguf` off one **NVIDIA
GTX 1650 Ti (Max-Q)** in the machine under the desk.

**Status 2026-09-03: serving.** First light on 2026-09-03 — **73.75 tok/s single-stream decode,
277.61 tok/s aggregate at 64-way concurrency, ~355 t/s prefill throughout, in 1618 MiB of a
4096 MiB card.** Full write-up in `benchmarks/runs/2026-09-03-first-light-1650ti/REPORT.md`.

| | |
| --- | --- |
| Platform | `local` — no control plane; the card is in this machine |
| Runtime | `llamacpp` — one process, one GGUF named on the command line |
| Hardware | `1650ti` — TU117, compute capability 7.5, 4096 MiB VRAM |
| Model | `2b` — `google/gemma-4-E2B-it` |
| Encoding | `q4_0` — the QAT GGUF export |

This is the **first `local` rig in the tree**. The platform value was added to
slot 1 of [`NAMING.md`](../NAMING.md) for it on 2026-09-03.

## Quick start

```bash
make install     # pip install -r requirements.txt into the system python3
make info        # resident-vs-lazy memory split, read off the artifact
make serve       # llama-server in the foreground; Ctrl-C is a full teardown
make status      # is it up?
make query       # one chat completion against 127.0.0.1:8080
```

`tpu.env` is the source of truth for all of it. The directory name is
documentation — never copy a slot value into a flag.

## The one thing to know before tuning it

The file is 3.35 GB on disk and the card has 3.63 GiB free. Those two numbers
side by side say "barely fits, lower `-ngl`, cap the context." **That is wrong.**

```
$ make info
per_layer_token_embd.weight  [8960, 262144]  Q6_K  1926.8 MB  <- LAZY, host-resident
token_embd.weight            [1536, 262144]  Q6_K   330.3 MB
...
lazy (never on GPU):   1.927 GB  (58% of file)
must be resident:      1.407 GB = 1.31 GiB
```

`per_layer_token_embd` is 58% of the artifact and none of it belongs on the GPU.
llama.cpp creates it with `TENSOR_READ_LAZY` in `src/models/gemma4.cpp` — "read
rows on demand instead of loading whole tensor; requires mmap for now" — and
serves it with `GGML_OP_GET_ROWS` out of the mapped file. It is the `E` in E2B
made concrete: `vocab_size_per_layer_input=262144` × `hidden_size_per_layer_input=256`
× 35 layers, and 256 × 35 = 8960 is the leading dimension above.

Full offload therefore fits with roughly **2.3 GiB to spare**, and this is **measured, not
derived**: `llama-server` at `-ngl 99 -c 8192` occupies **1618 MiB** of the 4096 MiB card, against
a prediction of 1616 MiB. Had the PLE tensor been resident the figure would be ~3.5 GB and the
model would not have loaded at all.

That also corrects the sibling rig `gpu-llamacpp-g5g-2b-q4_0`, which derives *Resident: 3.35 GB*
for the same artifact — see `CLAUDE.md`.

Two rules follow:

- **Never pass `--no-mmap`.** It defeats the mechanism and forces the 1.93 GB
  tensor to be materialised — a comfortable fit becomes an OOM.
- **Don't lower `N_GPU_LAYERS` "to be safe."** It moves real matmul weights to
  the CPU and buys nothing.

## Two things this rig is not

**It is not a q4_0 datapoint, quite.** 67.7% of the artifact is **Q6_K** (both
embedding tensors); only the ~1.05 GB transformer body is Q4_0. Slot 5 says
`q4_0` because that is what `MODEL_NAME` says, per `NAMING.md`. Record the
per-tensor split in any report from here, not just the slot token.

**It is not comparable to the T4 rigs at equal compute capability.** `gpu-vllm-g4dn-2b`
and `gpu-vllm-g5g-2b` are also sm_75, but the T4 is TU104 and has tensor cores
while the GTX 16-series (TU116/TU117) has none. Same compute capability,
different silicon — which is why the hardware slot is `1650ti` and not
`turing75`.

## Layout

```
tpu.env            source of truth — model, hardware, endpoint, serving flags
server.py          FastMCP server: GPU/model info, start/stop, status, query
inspect_gguf.py    re-derives the resident-vs-lazy split from the artifact
Makefile           serve / status / query / info / test / lint
tests/             offline unittest suite (11 tests)
benchmarks/        schema + README synced from the monorepo root; runs/ is empty
```

There are **no provisioning tools and none is owed** — no `find_*`, no queued
resources, no zone-status cache, no teardown to remember. A `tests/` case asserts
that, because a `local` rig that grows capacity-finding machinery has the wrong
name.

## See also

`../MODELS.md` (E2B structure, KV cost, weight footprints), `../HARDWARE.md`
(accelerator properties), `../NAMING.md` (the naming scheme and the `local`
section), `CLAUDE.md` (working notes for this rig).
