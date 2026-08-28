# `gce-jaxrust-v6e1-2b` — Gemma 4 on a single TPU v6e-1, driven from Rust

An inference path for Gemma 4 E2B on one Cloud TPU chip where **the serving process
contains no Python**. The forward pass is built in [rlx](https://github.com/MIT-RLX/rlx)'s
JAX-shaped tensor IR, lowered to HLO, and executed through libtpu's PJRT plugin. An
OpenAI-compatible HTTP/SSE server sits in front of it.

The rig also provisions the hardware it runs on: an MCP server
(`gce-jaxrust-v6e1-2b`) and a Claude Code skill that find capacity, create a flex-start
Compute Engine instance, build the engine on it, and run SRE diagnostics against the
endpoint.

**Forked from [`tpu-jax-v6e1-2b`](../tpu-jax-v6e1-2b) on 2026-08-28** to move the engine
off CPython onto Rust, alongside [`gpu-jaxrust-g5g-2b`](../gpu-jaxrust-g5g-2b) doing the
same on Turing.

## Status: nothing has run on a chip yet

This is a new rig and it has **provisioned nothing and measured nothing**. Read the claims
below in exactly the terms they are written.

**Verified on a workstation, 2026-08-28:**

- The `rust/` workspace compiles — `xla-probe` against `pjrt` 0.2.0, `gemma4-engine`
  against `rlx` / `rlx-gemma` 0.2.11 — `clippy -D warnings` clean.
- `rlx-tpu` resolves into the dependency graph, so `Device::Tpu` is a reachable backend.
- The `jaxrust` startup script renders and passes `bash -n`; the 149-test suite passes.

**Not verified:**

- That libtpu loads, creates a client, or executes anything on a v6e-1 from Rust.
  `xla-probe` exists to answer exactly that, and has never run on a chip.
- That `rlx-gemma` produces correct Gemma 4 E2B output on TPU. Its own backend feature list
  does not include `tpu`, and rlx's published TPU evidence is MiniLM-L6 — an encoder, not a
  decoder LLM. TPU + Gemma 4 is an untested intersection of two tested things.
- Any throughput, latency or HBM number. There are none here.

**The measured v6e-1 numbers you may be looking for are in
[`tpu-jax-v6e1-2b`](../tpu-jax-v6e1-2b)**, not here. They were taken on the Python JAX
engine through the Cloud TPU API. This rig's `benchmarks/` still holds inherited copies of
them; they describe the parent and are pending deletion (see `CLAUDE.md`, "Fork debris").

## The layering

```
gemma4-engine  ── rlx-gemma ──┐
                              ├── rlx (JAX-shaped IR) ── rlx-tpu ──┐
                              ┘                                     ├── libtpu.so (PJRT) ── v6e-1
xla-probe ────────────────── pjrt (PJRT C API) ─────────────────────┘
```

`xla-probe` bypasses rlx and talks to the PJRT C API directly. That is what makes it
useful: when the engine fails, the probe says whether the problem is above or below rlx,
in about a second.

## Quick start

```bash
./init.sh                                  # register the MCP server (writes .mcp.json)
make rust                                  # build both crates locally (needs protoc + clang)
```

Then, through the MCP server or the skill:

```
create_tpu_vm_instance(workload="jaxrust")   # RUNNING is not ready
wait_for_jaxrust_ready                       # Rust + libtpu installed. Nothing built yet.
deploy_jaxrust_engine                        # uploads rust/, builds release, runs the probe
manage_jaxrust_server(action="start", model_path="/path/to/gemma-4-E2B-it")
```

Each step answers a strictly smaller question than the next, so when serving fails, the
step that still passes tells you where to look.

## Why the probe checks a number

`dlopen` on `libtpu.so` succeeds on a host with no chip attached, exactly as `import jax`
succeeds with no TPU backend. So `xla-probe` does not stop at "a device is listed": it
compiles a StableHLO matmul, executes it, and checks that every element of the result is
what arithmetic says it should be. A plugin that loads, a client that creates, and a device
that lists are three different facts, and none of them is evidence that the MXU computed
anything.

It also prints each device's `largest_free_block_bytes` alongside the HBM limit — the
number that actually decides whether a weight tensor can be placed, and the one whose
absence has made a load look feasible right up until it failed on fragmentation.

## Licensing

This rig is Apache-2.0. The `gemma` cargo feature pulls in `rlx-gemma`, which is
**GPL-3.0-only**, making a binary built with it a GPLv3 combined work. Nothing is
distributed as a binary today — the source is uploaded and built on the VM — but the
default is a choice worth making deliberately. `rust/NOTICE.md` has the full table;
`xla-probe` never links it.

## Layout

| Path | What |
| --- | --- |
| `rust/` | The engine: `xla-probe` and `gemma4-engine`. See `rust/README.md`. |
| `server.py` | The MCP server — provisioning, deployment, diagnostics |
| `.claude/skills/gce-jaxrust-v6e1-2b-management/` | The skill, with the four startup templates |
| `jax_engine.py`, `jax_openai_server.py`, `ports/gemma4/` | The Python JAX engine, kept as the parity oracle |
| `docs/gemma4-quirks.md` | The model's architecture and serving path, verified against the reference |
| `tests/` | 149 offline unittest tests |

The Python engine is not legacy being kept out of sentiment. It is the only implementation
here whose numbers have been checked against the reference module, and it no longer shares
a language, a graph builder or an author with the Rust one — which makes a disagreement
between them worth much more than a disagreement between two JAX code paths was.
