# local-pytorch-cpu-2b

Serve **`google/gemma-4-E2B-it`** with **PyTorch + transformers on the local
CPU** — no accelerator, no cloud, no control plane.

> **STATUS 2026-09-04: it serves.** 4.853 tok/s, peak RSS 5.71 GB, load 3.1 s.
> Run: `benchmarks/runs/2026-09-04-first-light-pytorch-cpu/`.

## Why it exists

It is the **reference implementation** — stock `transformers`, no engine of our
own — and therefore the thing other rigs get checked against. It is also the
runtime control for `local-jax-cpu-2b`: same host, same checkpoint, no chip.

## Quick start

```bash
make capacity      # can this host hold the weights? run first
make install       # torch, transformers, accelerate (system python3)
make first-light   # load + generate 128 tok, writes run/first_light.json
make query         # one generation at the full 1024-token budget
```

## The headline: mmap gives it the lazy PLE for free

Peak RSS is **5.71 GB for a 10.25 GB checkpoint**, and the missing 4.7 GB is
exactly one tensor:

```
safetensors file             10.25 GB
minus embed_tokens_per_layer -4.70 GB
= everything else             5.55 GB
MEASURED peak RSS             5.71 GB   (+0.16)
```

Safetensors mmaps; the PLE is an indexed gather, so only the rows for tokens
actually seen are paged in. The same tensor is what llama.cpp keeps off a 4 GiB
GPU with `TENSOR_READ_LAZY` — and what vLLM materialises eagerly at `__init__`
and dies on.

**Size this rig from the file, never from RSS.** RSS right after load reads
1.20 GB and would tell you a host that cannot hold the model is fine.

## Measured

| | |
| :--- | ---: |
| load (warm cache) | 3.1 s |
| params | 5.104 B |
| 128 tok | 26.4 s = **4.853 tok/s** |
| repeat | 4.587 tok/s |
| peak RSS | 5.71 GB |

One end-to-end figure — `generate()` gives no prefill/decode split, so this is
**not** comparable to a sibling's decode-only rate. It is slow because bf16 is
emulated on an AVX2-only CPU; bf16 is chosen for memory, since fp32 would be
20.5 GB and does not fit at all.

## Layout

```
server.py             MCP server — capacity, load, generate. No provisioning.
bench_first_light.py  the reproducible run; reads tpu.env
tpu.env               source of truth, committed
tests/                offline unittest suite
```

Read `CLAUDE.md` before changing anything — including why the GPU in this box is
deliberately unused.
