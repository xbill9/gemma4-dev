---
title: "A 4 GB Laptop GPU Beats a 12-Core CPU by 4.3x on Gemma 4"
published: false
description: "Serving Gemma 4 E2B q4_0 through llama.cpp on one laptop, twice: CPU-only and on a 2021-era 4 GB GTX 1650 Ti. Same GGUF, same binary, same prompts, one flag apart. The card wins decode by 4.3x, and getting the comparison honest took more work than running it."
tags: machinelearning, gpu, benchmarking, python
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/cpu-vs-1650ti/devto-cover.a3124dd6.jpg
---

This article compares two ways of serving the same small language model on the
same laptop: CPU-only, and on the 4 GB GTX 1650 Ti sitting in the same chassis.
The payload is byte-identical on both arms and the command lines differ by a
single flag. The card takes decode by **4.3x**.

The repository is at https://github.com/xbill9/gemma4-dev

The interesting part is not the number. It is that the first three attempts at
this comparison would have produced a plausible, wrong one, and nothing in any
log would have said so.

## What Is Being Compared?

One machine, a 13th Gen Intel Core i7-1360P laptop with a GTX 1650 Ti (Max-Q)
in it. Two arms:

| | CPU arm | GPU arm |
| --- | --- | --- |
| Device | i7-1360P, 12 cores / 16 threads | GTX 1650 Ti Max-Q, 4096 MiB |
| Topology | 4 SMT P-cores (0-7) + 8 E-cores (8-15) | TU117, compute capability 7.5, **no tensor cores** |
| SIMD / math | `avx2`, `avx_vnni`, **no AVX-512** | CUDA |
| llama.cpp | `c6824a9`, `GGML_CUDA=OFF` build | `c6824a9`, CUDA build |
| Flag that differs | `-ngl 0` | `-ngl 99` |

Everything else matches, and that is the whole exercise:

```
-m gemma-4-E2B_q4_0-it.gguf --host 127.0.0.1 --port 8080 \
  -ngl {0|99} -c 8192 -ctk f16 -ctv f16 -fa 1 -t 4 -tb 8 --parallel 1 --metrics
```

The model is `google/gemma-4-E2B-it-qat-q4_0-gguf`, a 3.35 GB quantization-aware
GGUF. The two arms were run alternately against the same endpoint, CPU first,
with a fixed 120 second cooldown between them.

## The Result

Eight cells: four prompt lengths by two output lengths, three repeats each,
concurrency 1. Decode is client-side inter-token rate measured off the SSE
stream, which is the only decode statistic both arms can produce.

| in tok | out tok | CPU decode | GPU decode | 🥇 | CPU TTFT ms | GPU TTFT ms | 🥇 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 94 | 32 | 17.61 | 71.22 | **4.04x** | 1143 | 385 | **2.97x** |
| 94 | 128 | 17.22 | 71.20 | **4.13x** | 1219 | 385 | **3.17x** |
| 516 | 32 | 16.16 | 69.93 | **4.33x** | 5978 | 1610 | **3.71x** |
| 516 | 128 | 16.39 | 69.49 | **4.24x** | 5995 | 1612 | **3.72x** |
| 998 | 32 | 15.81 | 68.60 | **4.34x** | 11748 | 3244 | **3.62x** |
| 998 | 128 | 16.10 | 68.52 | **4.26x** | 11532 | 3242 | **3.56x** |
| 1959 | 32 | 15.66 | 67.61 | **4.32x** | 23926 | 6522 | **3.67x** |
| 1959 | 128 | 15.80 | 67.64 | **4.28x** | 23773 | 6522 | **3.65x** |

Medians over the eight cells: decode **4.27x**, prefill **3.63x**, end-to-end
**3.81x**.

## Is That Difference Real?

Within-cell spread over three repeats was **6.29%** worst case on the CPU arm and
**0.84%** on the GPU arm. The decode ratio spans 4.04x to 4.34x across every cell.
The effect is an order of magnitude clear of the noise, which is the only reason
it is worth reporting at all.

## Decode Is Flat, Prefill Is Linear

Look down the decode columns rather than across them.

CPU decode moves from 17.61 to 15.66 tok/s. GPU decode moves from 71.22 to 67.61.
Across a **21x range of prompt length**, on both arms, decode barely moves. TTFT
over the same range goes from 1143 ms to 23926 ms on the CPU and 385 ms to
6522 ms on the GPU — linear in the prompt.

That is the expected shape stated twice. Decode reads the whole model per token
and is bandwidth-bound; prefill multiplies through the prompt and is
compute-bound. An accelerator helps both, but it helps them for different
reasons, and a benchmark that quotes one number hides that.

The decode *ratio* also climbs with context — 4.04x at 94 tokens, 4.34x at 998.
The CPU arm loses ground as the KV cache grows and the GPU arm barely does.

## How Does 3.35 GB Fit in a 4 GB Card?

It does not have to. With the model loaded and serving, the card holds **1598
MiB**.

The artifact is only 32% Q4_0. Both embedding tensors are Q6_K and account for
2.257 GB of the 3.334 GB of tensor bytes, and the largest of them —
`per_layer_token_embd`, at 1.93 GB, 58% of the file — is created with
`TENSOR_READ_LAZY` in llama.cpp's `src/models/gemma4.cpp` and served by
`GGML_OP_GET_ROWS` straight out of the mmap. A few rows are touched per token.
It never goes to the card.

What is resident is the ~1.08 GB Q4_0 transformer body, which is what decode
reads every token, plus about 60 MiB of KV cache and the compute buffers.

The same mechanism costs something on the CPU arm instead of saving something.
`RssAnon` there is 1,202,152 kB, because llama.cpp repacks Q4_0 weights into an
interleaved layout for its AVX2 kernels, copying the body out of the mmap into
anonymous memory.

So the headline works because of the checkpoint's shape, not in spite of the
card's size. A 4 GB GPU from 2021 is not too small for this model. It is roughly
2.5x larger than it needs to be.

## Why This Took Three Tries

The two rigs were built as siblings months apart, and they had drifted in ways
that only matter when you subtract one from the other.

**The binaries were different.** The CPU rig recorded llama.cpp `82324fc50` and
had no CPU build on disk at all; the GPU rig's binary was `95ef7fc`. Differencing
those two puts an upstream delta inside what is supposed to be a device delta.
Both were rebuilt from `c6824a9` for this run.

**The flags were different.** The GPU rig had no `THREADS_BATCH` setting and
therefore passed no `-tb`, while the CPU rig passed `-tb 8`. Prefill runs on the
card, so the effect is probably small — but a control does not get to assume
which of its differences are the harmless ones.

**The prompts were different.** Each rig's sweep harness built its filler text
out of a description of its own hardware. So `--contexts 512` produced two
different prompts that tokenized to two different lengths, and the paired cells
were not the same measurement. The filler is now device-neutral and the harness
is byte-identical in both rigs. All eight cells in the table above paired on
exactly equal input lengths — 94, 516, 998, 1959 — which is how you can tell.

## The Failure That Does Not Announce Itself

Both arms serve the same model on the same port. That is deliberate: the
endpoint, the harness and the prompts stay fixed while the device changes
underneath them.

It also means **nothing in an HTTP response says which device produced it.** The
sweep harness took its device label from a command-line argument. The status tool
reported `✅ Serving` for whatever held port 8080. Restart into the other arm and
forget, and you get a complete, plausible, correctly-formatted run labelled with
the wrong hardware, and no error anywhere.

So the arm is now read off the running process rather than asserted:

| Source | Answers |
| --- | --- |
| `/proc/<pid>/exe` | which binary is executing, not what the config says |
| `/proc/<pid>/maps` | which ggml backends it has actually loaded |
| `/proc/<pid>/cmdline` | the real `-ngl`, whatever any env file claims |
| `/proc/<pid>/environ` | `CUDA_VISIBLE_DEVICES` as the child received it |

`maps` rather than `ldd` matters: llama.cpp `dlopen`s its backends, so a CUDA
backend can be absent from `ldd` output and present in the running process.

The verdict requires two signals to agree — a GPU backend mapped in **and**
layers assigned to it. Disagreement reports as `mixed` rather than rounding to
either, because a CUDA build running `-ngl 0` computes on the CPU but is not a
clean CPU arm: the device is initialised and llama.cpp can still move large
prefill batches onto it. The CPU arm hides the device from the child entirely
rather than trusting `-ngl 0`.

Every report now carries the attestation, including the SHA-256 of the binary
that ran. A commit is what you meant to build. The hash is what executed.

## One Bug Found by Running It

The CPU server died during model load every time it was started through the
management tool, and came up fine under a plain `make serve`.

The tool spawned it with `asyncio.create_subprocess_exec` and
`start_new_session=True`. Asyncio's subprocess transport kills a live child when
it is torn down — `BaseSubprocessTransport.__del__` calls `close()`, which calls
`_proc.kill()` on a child that has not exited — and a new session does not
prevent it. A spawned `sleep 60` is dead within a second of the interpreter
exiting.

That is very likely why both rigs documented "no pid file" as the *normal* state:
the only code path that writes one could not leave a server behind.
`subprocess.Popen` only warns when collected with a live child.

## What Was Not Controlled

Read the 4x as real and anything under about 7% as nothing.

- **Order and thermals.** The CPU arm ran first and saturated 12 cores. An
  i7-1360P and a Max-Q card share one thermal envelope, so the GPU arm started on
  a warm package. The 120 second cooldown was sized, not measured. If this biases
  anything it understates the GPU arm.
- **Page cache.** The GGUF was hot for both arms. This says nothing about cold
  start, and the lazy-embedding claim above is inferred from the source and the
  resident-memory figures rather than from a dropped-cache experiment.
- **CPU affinity.** Nothing was pinned. A separate `llama-bench` sweep on this
  die found affinity worth **1.61x on prefill**, because llama.cpp splits work
  evenly and every barrier waits on the slowest thread, so adding the eight
  E-cores to the four P-cores makes prefill *slower*. The CPU arm here is
  therefore not at its best, and **3.63x is an upper bound on the prefill gap**.
  Decode is unaffected — thread count is a weak lever there — so the 4.27x
  stands.
- **Concurrency.** Single stream throughout, `--parallel 1`.

## Summary

The goal of this article was to measure what a 4 GB laptop GPU is worth against a
modern 12-core CPU for serving a small quantized model. The key to the solution
was making the two arms genuinely identical — one binary, one prompt set, one
flag apart — and then verifying from `/proc` that the process answering was the
one being claimed. The results were:

- GPU decode is **4.27x** the CPU arm, 4.04x to 4.34x across every cell
- GPU prefill is **3.63x**, an upper bound, since the CPU arm was left unpinned
- End-to-end is **3.81x**
- Decode is flat in context on both arms; TTFT is linear in it on both
- The model needs **1598 MiB** of the card, because 58% of the file is a lazily
  read embedding table that never leaves host memory

Scope: one laptop, one GGUF, llama.cpp `c6824a9` built twice from the same
commit, eight paired cells, three repeats per cell, concurrency 1, CPU arm first
with a 120 second cooldown. CPU affinity was not set on either arm and the page
cache was warm for both.

The strategy for using MCP for local accelerator comparison was validated with an
incremental step by step approach.
