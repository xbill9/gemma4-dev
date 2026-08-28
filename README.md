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
