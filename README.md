# gemma4-dev

A monorepo of **accelerator rigs** for serving [Gemma 4](https://ai.google.dev/gemma) on Google Cloud TPU.

Each rig is a self-contained project that serves one Gemma 4 checkpoint on one hardware shape through one
runtime. Every rig ships the same shape of thing: a single-file [MCP](https://modelcontextprotocol.io) server
(`server.py`, built on FastMCP) exposing a `tpu-devops` agent that provisions TPU capacity, starts a model
server on it, and does SRE diagnostics against the running endpoint.

The rigs are siblings, not layers — they share ancestry and diverge. Nothing is imported across rig
boundaries.

## The variants

| Rig | Runtime | Hardware | Model | Notes |
| --- | --- | --- | --- | --- |
| [`tpu-vllm-v5e1-2b`](tpu-vllm-v5e1-2b/) | vLLM in Docker | v5e-1 | `gemma-4-E2B-it` | Flex-start Queued Resource; the live-demo rig |
| [`tpu-vllm-v6e1-2b`](tpu-vllm-v6e1-2b/) | vLLM in Docker | v6e-1 | `gemma-4-E2B-it` | Fork of the v5e-1 rig retargeted to Trillium; provisions in `us-east5-b` |
| [`tpu-jax-v5e1-2b`](tpu-jax-v5e1-2b/) | pure JAX | v5e-1 | `gemma-4-E2B-it-qat-w4a16-ct` | Hand-rolled engine + OpenAI-compatible server; no Docker, no HF token |
| [`tpu-pytorch-v5e1-12b`](tpu-pytorch-v5e1-12b/) | PyTorch / `torch_xla` | v5e-1 | `gemma-4-12B-it-qat-w4a16-ct` | Static-shape decode server |
| [`tpu-pytorch-v5e1-2b`](tpu-pytorch-v5e1-2b/) | PyTorch / `torch_xla` | v5e-1 | 2B | Near-identical fork of the 12B rig |

Directory names follow a four-slot scheme — `<platform>-<runtime>-<hardware>-<model>`.
**[`NAMING.md`](NAMING.md) is the spec**: it defines the permitted value for every slot, the rules for adding
or renaming a rig, and the separate date-first scheme used for benchmark artifacts. Read it before naming
anything.

The name is documentation, not configuration. A rig's authoritative values live in its `tpu.env`
(`MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`) and siblings spell them inconsistently — v5e is
`v5litepod-1` to gcloud in one rig and `v5e-1` in another. **Never copy a slot value into a CLI flag.**

## Installing a rig as a Claude Code plugin

Three of the rigs ship their `tpu-management` skill and `tpu-devops` MCP server as an installable plugin:

```
/plugin marketplace add xbill9/gemma4-dev
```

then install `tpu-jax-v5e1-2b`, `tpu-pytorch-v5e1-12b`, or `tpu-pytorch-v5e1-2b`. The plugin is named after
the rig it comes from, since all three provide a skill called `tpu-management` and would otherwise collide.
`tpu-vllm-v5e1-2b` is not packaged as a plugin — use it directly.

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

Apache-2.0.
