# gemma4-dev

A monorepo of **accelerator rigs** for serving [Gemma 4](https://ai.google.dev/gemma) on Google Cloud TPU,
AWS Inferentia2, and NVIDIA GPUs.

Each rig is a self-contained project that serves one Gemma 4 checkpoint on one hardware shape through one
runtime. A serving rig ships the same shape of thing: a single-file [MCP](https://modelcontextprotocol.io)
server (`server.py`, built on FastMCP) exposing a devops agent that provisions capacity, starts a model
server on it, and does SRE diagnostics against the running endpoint. A handful of **artifact rigs** carry
only measurements — see the second table below.

**Each rig's MCP server is named after the rig directory** — `tpu-jax-v5e1-2b`, `tpu-vllm-v5e1-2b`, and so
on. That name is the key the server is registered under, so it prefixes every tool as
`mcp__<rig>__find_tpu`. The rigs previously all registered as `tpu-devops`, which made a tool call
ambiguous whenever more than one was loaded. Override per rig with `MCP_SERVER_NAME` (or
`project-setup.sh --server-name`) only when a client has already registered one under a different key.

The rigs are siblings, not layers — they share ancestry and diverge. Nothing is imported across rig
boundaries.

## The variants

**Serving rigs** — each ships a `server.py`, an MCP server, and a deployment path:

| Rig | Runtime | Hardware | Model | Notes |
| --- | --- | --- | --- | --- |
| [`tpu-vllm-v5e1-2b`](tpu-vllm-v5e1-2b/) | vLLM in Docker | v5e-1 | `gemma-4-E2B-it` | Flex-start Queued Resource; the live-demo rig |
| [`tpu-vllm-v5e1-2b-q4_0`](tpu-vllm-v5e1-2b-q4_0/) | vLLM in Docker | v5e-1 | `gemma-4-E2B-it-qat-q4_0-unquantized` | QAT-at-q4_0 weights shipped as bf16; the 4-bit load path is unsupported on this stack — see the rig README |
| [`tpu-vllm-v5e1-2b-w4a16`](tpu-vllm-v5e1-2b-w4a16/) | vLLM in Docker | v5e-1 | `gemma-4-E2B-it-qat-w4a16-ct` | Real 4-bit compressed-tensors weights; **expected to fail** at `compressed_tensors.py:149` until a `wNa16` scheme lands — the rig exists to record that |
| [`tpu-vllm-v6e1-2b`](tpu-vllm-v6e1-2b/) | vLLM in Docker | v6e-1 | `gemma-4-E2B-it` | Fork of the v5e-1 rig retargeted to Trillium; provisions in `us-east5-b` |
| [`tpu-jax-v5e1-2b`](tpu-jax-v5e1-2b/) | pure JAX | v5e-1 | `gemma-4-E2B-it-qat-w4a16-ct` | Hand-rolled engine + OpenAI-compatible server; no Docker, no HF token |
| [`tpu-jax-v6e1-2b`](tpu-jax-v6e1-2b/) | pure JAX | v6e-1 | `gemma-4-E2B-it-qat-q4_0-unquantized` | Fork of the v5e-1 JAX rig retargeted to Trillium and **migrated off the Cloud TPU API onto Compute Engine** — no queued-resource path at all |
| [`gce-jaxrust-v6e1-2b`](gce-jaxrust-v6e1-2b/) | Rust-driven XLA (`rlx` → HLO → libtpu PJRT) | v6e-1 | `gemma-4-E2B-it` | Forked from `tpu-jax-v6e1-2b` 2026-08-28 to move the engine off CPython — **no Python in the serving path**. The workspace builds and clippy is clean; **nothing has run on a chip**, and `xla-probe` exists to be the first thing that does. `rlx-gemma` is GPL-3.0-only, so the model feature is opt-in — see `rust/NOTICE.md` |
| [`tpu-pytorch-v5e1-12b`](tpu-pytorch-v5e1-12b/) | PyTorch / `torch_xla` | v5e-1 | `gemma-4-12B-it-qat-w4a16-ct` | Static-shape decode server |
| [`tpu-pytorch-v5e1-2b`](tpu-pytorch-v5e1-2b/) | PyTorch / `torch_xla` | v5e-1 | 2B | Near-identical fork of the 12B rig |
| [`tpu-pytorch-v6e1-2b`](tpu-pytorch-v6e1-2b/) | PyTorch / `torch_xla` | v6e-1 | 2B | |
| [`tpu-pytorch-inf2-2b`](tpu-pytorch-inf2-2b/) | PyTorch / `torch_neuronx` | inf2 | 2B | **AWS Inferentia2, not GCP** — EC2 + Neuron DLAMI, driven over Systems Manager |
| [`gpu-jax-g4dn-2b`](gpu-jax-g4dn-2b/) | pure JAX | g4dn (Intel x86_64 + NVIDIA T4, Turing SM 7.5) | `gemma-4-E2B-it` | Forked from `gpu-jax-g5g-2b` 2026-08-28. **Nothing measured.** The A/B control that separates Turing from Graviton2 — same engine, same chip generation, x86_64 host. Also the cheaper box ($0.3678 spot avg vs $0.3996) |
| [`gpu-jax-g6-2b`](gpu-jax-g6-2b/) | pure JAX | g6 (x86_64 + NVIDIA L4, Ada SM 8.9) | `gemma-4-E2B-it` | Forked from `gpu-jax-g5g-2b` 2026-08-28. **MEASURED the same day: 48.3-48.5 tok/s, 3.7x the T4G at an identical config.** The dtype-tax control, and it answered: conversion 54.0% -> **0.0%**, and the rig now sits at its bandwidth roofline where the T4G sat at 26%. TensorCore stayed 0.0% |
| [`gpu-vllm-g6-2b`](gpu-vllm-g6-2b/) | vLLM | g6 (x86_64 + NVIDIA L4, Ada SM 8.9) | `gemma-4-E2B-it` | Forked from `gpu-vllm-g5g-2b` 2026-08-28. **Nothing measured.** The runtime control for `gpu-jax-g6-2b` on identical silicon — the only clean runtime A/B here, since the T4G pair's vLLM side used reduced Triton tiles. The fork deletes the sibling's 67-minute build: SM 8.9 is in the published amd64 image |
| [`gpu-vllm-g4dn-2b`](gpu-vllm-g4dn-2b/) | vLLM | g4dn (Intel x86_64 + NVIDIA T4, Turing SM 7.5) | `gemma-4-E2B-it` | Forked from `gpu-vllm-g6-2b` 2026-08-29 into a directory that was a stale copy of `gpu-jax-g4dn-2b`. **Nothing measured.** It **isolates one of the two problems** that make `gpu-vllm-g5g-2b` hard: x86_64 pulls the amd64 manifest, which carries SM 7.5, so the 67-minute build is gone — but Turing's 64 KiB shared memory still blocks Gemma 4's 512-wide Triton tile (98,304 > 65,536, arithmetic rather than a margin). So it is the first rig here to deliver that clamp by **patching the published image** and building a derived tag in seconds, instead of compiling vLLM. Also the runtime control for `gpu-jax-g4dn-2b` |
| [`gpu-pytorch-g5g-2b`](gpu-pytorch-g5g-2b/) | PyTorch + transformers | g5g (Graviton2 + NVIDIA T4G, Turing SM 7.5) | `gemma-4-E2B-it` | Forked from `gpu-jax-g5g-2b` 2026-08-28. **Nothing measured.** Slot 2 is the only thing that moves, so it is the runtime A/B against the JAX rig on identical silicon. torch ships **inside** the ARM64 PyTorch DLAMI already built with `sm_75`, so unlike the JAX rig there is no pip CUDA at all |
| [`gpu-pytorch-g4dn-2b`](gpu-pytorch-g4dn-2b/) | PyTorch + transformers | g4dn (Intel x86_64 + NVIDIA T4, Turing SM 7.5) | `gemma-4-E2B-it` | Forked from `gpu-pytorch-g5g-2b` 2026-08-29. **Nothing measured.** The fourth corner of the {jax, pytorch} x {g5g, g4dn} square. Its A/B partner `gpu-jax-g4dn-2b` has now closed the host axis (13.1 vs 13.10 tok/s, host contributes nothing), so this is the **runtime A/B on identical hardware** and the question it answers is whether the 86.9% dtype-plus-fp32-GEMV tax is Turing or XLA. Torch comes from the **x86_64 PyTorch DLAMI** (2.13 / Ubuntu 26.04), never from pip |
| [`gpu-llamacpp-g5g-2b-q4_0`](gpu-llamacpp-g5g-2b-q4_0/) | llama.cpp (`llama-server`) | g5g (Graviton2 + NVIDIA T4G, Turing SM 7.5) | `gemma-4-E2B-it-qat-q4_0-gguf` | Scaffolded 2026-09-02 from `gpu-pytorch-g5g-2b`. **Nothing measured.** **The only rig here serving 4-bit weights on a GPU** — vLLM 0.26.0 has no `gguf` module on any platform, JAX has no GGUF reader, and transformers dequantizes to fp32 while silently dropping 35 `layer_scalar` tensors. Streams 1.407 GB/step against the PyTorch rig's measured 4.514 and frees ~6.9 GB of a 14.07 GB chip — **residency, not speed**: decode here is launch-bound, and a 3.5 GB weight cut on the JAX sibling moved throughput 0.0%. It **compiles**, because llama.cpp ships no prebuilt Linux aarch64 CUDA binary |
| [`gpu-ollama-g5g-2b-q4_0`](gpu-ollama-g5g-2b-q4_0/) | Ollama (links llama.cpp) | g5g (Graviton2 + NVIDIA T4G, Turing SM 7.5) | `gemma4:e2b-it-qat` | Created 2026-09-02. **Nothing measured.** Same silicon, same weights and **the same engine** as the llama.cpp rig — Ollama links `libllama.so`, so slot 2 names the front end, not the decoder. The pair exists for the four differences a benchmark can still see, chief among them that **Ollama ships an aarch64 CUDA bundle with native sm_75 SASS where llama.cpp publishes none**, so this rig has no build step. Its artifact is Ollama's re-containered blob (3,349,514,112 B vs Google's 3,349,516,256, different digest), so `-q4_0` is a weaker claim here. Pins `OLLAMA_LLM_LIBRARY=cuda_v12`: the v13 bundle is **PTX-only for every arch** and JITs at load |
| [`tpu-jax-inf2-2b`](tpu-jax-inf2-2b/) | pure JAX (`jax-neuronx`) | inf2 | `gemma-4-E2B-it` | Created 2026-08-28 from `tpu-pytorch-inf2-2b` provisioning + the `gpu-jax` engine. **The serving path is NOT wired — `server.py` still deploys the vLLM-Neuron container.** Parity already measured the engine token-exact on Neuron at ~14 tok/s for the dense fp16/bf16-KV config; `inf2.xlarge` is $0.1417/hr spot avg |
| [`local-llamacpp-1650ti-2b-q4_0`](local-llamacpp-1650ti-2b-q4_0/) | llama.cpp (`llama-server`) | **1650ti — the workstation GPU** (TU117, Turing SM 7.5, 4096 MiB) | `gemma-4-E2B-it-qat-q4_0-gguf` | Added 2026-09-03, **the first `local` rig** — no control plane, nothing provisioned, Ctrl-C is a complete teardown. **MEASURED 2026-09-03:** 73.75 tok/s single-stream decode and 1618 MiB of 4096, because `per_layer_token_embd` (58% of the file) is `TENSOR_READ_LAZY` and never reaches VRAM. End-to-end serving caps at ~45-48 tok/s: prefill does not batch, so TTFT doubles with every doubling of concurrency |
| [`local-ollama-1650ti-2b-q4_0`](local-ollama-1650ti-2b-q4_0/) | Ollama (links llama.cpp) | **1650ti — the workstation GPU** (TU117, Turing SM 7.5, 4096 MiB) | `gemma4:e2b-it-qat` | Added 2026-09-04. **Differs from the llama.cpp rig in slot 2 and nothing else** — same card, same weights, same engine — so it is the sharpest `llamacpp`-vs-`ollama` test here. **MEASURED 2026-09-04:** end-to-end ties the sibling within 1% in all ten context cells and decode is ~3% behind. The stock tag holds **2762 MiB against 1618** — the daemon loads the 986 MB vision/audio projector unconditionally and `ollama ps` does not count it — so this rig serves `gemma4:e2b-it-qat-text`, the same blob with that layer dropped, at **1612 MiB**. That also removes a 2048-token per-slot minimum that applies only to multimodal models, letting it reach the sibling's exact 32-slot geometry, where the two agree within 4% through c=8 |
| [`local-jax-cpu-2b`](local-jax-cpu-2b/) | pure JAX on the local **CPU** | none (`JAX_PLATFORMS=cpu`) | `gemma-4-E2B-it` | Forked from `gpu-jax-g4dn-2b` 2026-08-29. **Nothing measured** — every throughput figure in its inherited prose was measured on a T4G |
| [`local-pytorch-cpu-2b`](local-pytorch-cpu-2b/) | PyTorch + transformers on the local **CPU** | none — no accelerator | `gemma-4-E2B-it` | Retargeted 2026-09-04 from what was a verbatim copy of `gpu-jax-g4dn-2b` (it contained no PyTorch at all), and **served the same day: 4.853 tok/s, load 3.1 s**. The reference implementation, and the runtime control for `local-jax-cpu-2b`. **Peak RSS is 5.71 GB of a 10.25 GB checkpoint** — safetensors mmaps and the PLE is a gather, so transformers gets llama.cpp's lazy-PLE behaviour for free from the page cache. Size it from the file, never from RSS |
| [`local-vllm-cpu-2b`](local-vllm-cpu-2b/) | vLLM, **CPU backend** | none — no accelerator | `gemma-4-E2B-it` | Retargeted 2026-09-04 from what was a verbatim copy of `gpu-jax-g4dn-2b`. **Has served nothing; the budget is close rather than closed.** The w4a16 checkpoint is 8.32 GB (MEASURED), needing 10.46 GB against ~9.3 GB free on a 16.42 GB machine — short by headroom, not by hardware. Three pessimistic claims in the first version were corrected: no fp32 upcast on x86, AVX2 is a first-class vLLM build target, and w4a16 saves 19% not 75%. The serve tool still refuses while over budget, because exceeding host RAM is accepted and paid for in swap rather than raising. It is the runtime control for `local-jax-cpu-2b`, which does fit |

**Artifact rigs** — measurements and findings only. No `server.py`, no MCP server, no deployment
path. They exist so results measured on hardware no serving rig covers are filed under the hardware
they came from. See [`NAMING.md`](NAMING.md#artifact-rigs--a-rig-that-serves-nothing).

| Rig | Runtime | Hardware | Model | Notes |
| --- | --- | --- | --- | --- |
| [`tpu-jax-v6e1-12b-w4a16`](tpu-jax-v6e1-12b-w4a16/) | pure JAX | v6e-1 | `gemma-4-12B-it-qat-w4a16-ct` | 100% token parity vs the HF reference; ~29.5 ms/step flat from 1K to 8K |
| [`tpu-jax-v6e1-26b-q4_0`](tpu-jax-v6e1-26b-q4_0/) | pure JAX | v6e-1 *(target)* | `gemma-4-26B-A4B-it-qat-q4_0-unquantized` | The only **sparse** checkpoint and the only size with no `-w4a16-ct`. **Verified on CPU; never run on a TPU** |
| [`tpu-jax-v6e1-31b-w4a16`](tpu-jax-v6e1-31b-w4a16/) | pure JAX | v6e-1 | `gemma-4-31B-it-qat-w4a16-ct` | Measured W4A16 error, massive activations, sink reachability, XLA memory cliffs |
| [`gpu-vllm-l4-2b-w4a16`](gpu-vllm-l4-2b-w4a16/) | vLLM | NVIDIA L4 | `gemma-4-E2B-it-qat-w4a16-ct` | 2D concurrency grid, GCE |
| [`gpu-vllm-l4-4b-w4a16`](gpu-vllm-l4-4b-w4a16/) | vLLM | NVIDIA L4 | `gemma-4-E4B-it-qat-w4a16-ct` | 2D concurrency grid, GCE |
| [`gpu-vllm-l4-12b-w4a16`](gpu-vllm-l4-12b-w4a16/) | vLLM | NVIDIA L4 | `gemma-4-12B-it-qat-w4a16-ct` | Four grids — Cloud Run, GCE, EC2, and an MTP build |
| [`gpu-vllm-l4-26b-w4a16`](gpu-vllm-l4-26b-w4a16/) | vLLM | NVIDIA L4 | `gemma-4-26B-A4B-it-qat-w4a16-ct` | **That Hub id does not exist** — the reports name a local mount; see the rig |
| [`gpu-vllm-l4-31b-w4a16`](gpu-vllm-l4-31b-w4a16/) | vLLM | NVIDIA L4 | `gemma-4-31B-it-qat-w4a16-ct` | At bf16 the 31B leaves **0 GB** for KV on an L4 |
| [`gpu-jax-g5g-2b`](gpu-jax-g5g-2b/) | pure JAX | g5g (Graviton2 + NVIDIA T4G, Turing SM 7.5) | `gemma-4-E2B-it` | Same silicon as the vLLM G5g rig, different runtime: pip supplies CUDA, so no build, no toolkit, no Triton patch. Serves the dense reference build — the fused W4A16 Pallas kernel cannot fit Turing shared memory and is refused at startup |
| [`gpu-jaxrust-g5g-2b`](gpu-jaxrust-g5g-2b/) | Rust-driven XLA | g5g (Graviton2 + NVIDIA T4G, Turing SM 7.5) | `gemma-4-E2B-it` | Forked from `gpu-jax-g5g-2b` 2026-08-28 to move the engine off CPython. **The runtime is chosen and the engine builds; nothing has run.** `rlx` + `rlx-gemma` on `Device::Cuda` — verified against the crate source, so the survey's first-draft claim that no Rust JAX reaches SM 7.5 on aarch64 is **withdrawn**; aarch64 itself is still unverified. `rlx-gemma` is GPL-3.0-only, so the model feature is opt-in — see `docs/rust-jax-runtime-survey.md` |

The five `gpu-vllm-l4-*` rigs were migrated from `~/gemma4-tips` on 2026-08-07. That tree duplicated
its artifacts heavily — **82 report files, 20 unique**, with directory names that misattribute models
— so only the 10 reports that self-identify came across. Each rig's `CLAUDE.md` carries the full
warning; do not go back to that tree and read a model off a directory name.

Directory names follow a four-slot scheme — `<platform>-<runtime>-<hardware>-<model>` — plus an
optional fifth slot naming the weight **encoding** when it isn't the reference build.
**[`NAMING.md`](NAMING.md) is the spec**: it defines the permitted value for every slot, the rules for adding
or renaming a rig, and the separate date-first scheme used for benchmark artifacts. Read it before naming
anything.

The name is documentation, not configuration. A rig's authoritative values live in its `tpu.env`
(`MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`) and siblings spell them inconsistently — v5e is
`v5litepod-1` to gcloud in one rig and `v5e-1` in another. **Never copy a slot value into a CLI flag.**

## Installing a rig as a Claude Code plugin

Three of the rigs ship their skill and their MCP server as an installable plugin:

```
/plugin marketplace add xbill9/gemma4-dev
```

then install `tpu-jax-v5e1-2b`, `tpu-pytorch-v5e1-12b`, or `tpu-pytorch-v5e1-2b`.
`tpu-vllm-v5e1-2b` is not packaged as a plugin — use it directly.

Everything a rig installs into a shared namespace is named after the rig, for the same reason: the plugin,
the MCP server (`tpu-jax-v5e1-2b`), and the skill (`tpu-jax-v5e1-2b-management`). The skills previously were
all called `tpu-management`, and `make skill-install` `rm -rf`s its destination — so installing a second rig
silently replaced the first. See `NAMING.md` for the full table and the override variables.

## Working in a rig

Everything is per-rig; there is no top-level build. `cd` into a rig first — each one carries its own
`CLAUDE.md` documenting its commands, conventions, and gotchas.

```bash
cd tpu-jax-v5e1-2b
make install     # pip install -r requirements.txt
make test        # unittest, not pytest
make lint        # ruff
```

Common to every rig:

- **Python 3.13**, plain `pip`, no lockfile. **No virtualenvs** — use the system `python3`; if dependencies
  are missing, report the `pip install -r requirements.txt` command rather than creating a venv.
- **`unittest`, never pytest.** Keep unit tests offline: mock the cloud, subprocess, and network boundaries.
- **`ruff`** as both linter and formatter. No black.
- Auth needs **both** `gcloud auth login` (for the `gcloud` subprocess calls) and
  `gcloud auth application-default login` (ADC, for the Secret Manager client).
- `set_env.sh` must be **sourced**, not executed. `init.sh` is a one-time bootstrap that blocks on `read` in
  its error path — don't run it non-interactively.

## Two things that will mislead you

**Benchmark numbers are not from the rig they sit in.** A report's hardware suffix records the hardware
*measured*, not the rig hosting the file. All four rigs carry a `v6e-1` report despite being v5e-1 rigs —
copies travelled with the forks. Never infer a rig's hardware from a benchmark filename.

**Several files are generated; edit the source, not the snapshot.** `make skill` / `refresh_skill.py`
regenerate the skill snapshots under `.claude/skills/` and `skills/`, and `make tools` regenerates
`GemmaTools.md` from the `@mcp.tool()` decorators in `server.py`. Hand-edits to a generated file are lost on
the next refresh.

## License

**Apache-2.0** — see [`LICENSE`](LICENSE). That covers every rig in this repo, with one
carve-out.

**`gpu-jaxrust-g5g-2b/rust/` is different when built with the `gemma` feature.** It links
`rlx-gemma`, which is **GPL-3.0-only** at 0.2.11 and has no later release — `rlx` itself
relicensed to MIT/Apache-2.0 at 0.2.14 but the model crate did not follow. GPLv3 accepts
Apache-2.0 code; the reverse does not hold, so a **binary** built with `--features gemma` is a
GPLv3 combined work and must ship its corresponding source under GPLv3.

Three things that follow, and the first two are why nothing is owed today:

- **Publishing this source is not distributing a combined work.** `rlx-gemma` is fetched by
  cargo on the builder's machine; it is not vendored here.
- **GPLv3 has no network clause** — that is AGPL. Serving the endpoint publicly triggers
  nothing. The obligation attaches when you hand someone a **binary**: a release artifact, a
  container image, an AMI.
- **The `gemma` cargo feature is the license boundary**, not just a build switch. Built
  `--no-default-features --features cuda`, the binary links no GPL code — it starts, reports the
  device, and refuses generation naming the flag.

Everything else in `rust/` — `gemma4-engine`'s own code and `gemma4-geometry` — is Apache-2.0
like the rest of the repo.
