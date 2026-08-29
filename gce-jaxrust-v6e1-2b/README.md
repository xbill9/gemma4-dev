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

## Status: Rust runs on the chip; the model does not yet

**MEASURED on a v6e-1 (`ct6e-standard-1t`, `europe-west4-a`, spot, 2026-08-28):** rlx
compiled a 256×256 f32 matmul for `Device::Tpu`, executed it through libtpu's PJRT plugin,
and returned 256.0 in all 65,536 elements. The whole lifecycle ran through this rig's own
MCP tools, and `gemma4-engine` builds on the VM with `--features tpu,gemma`.

Checked three ways that it was not a silent CPU fallback: `/tmp/tpu_logs` grew on the TPU
run and not the CPU run; libtpu logged `XLA::TPU running hlo passes for 6 instructions,
modules: selftest`; and a second process was refused with `The TPU is already in use by
process with pid …`.

**What is still unverified: Gemma 4 E2B itself.** No checkpoint has been loaded. rlx's TPU
backend compiles and executes *a* graph correctly; that is not evidence it does so for
E2B's PLE, alternating local/global attention and KV sharing. There are no throughput,
latency or HBM numbers.

**Three defects the deployment found**, all fixed or recorded:

- `xla-probe` **cannot create a PJRT client** against current libtpu — `pjrt-sys` 0.2.0
  builds `PJRT_Client_Create_Args` at 72 bytes, libtpu 0.75 demands 88. `rlx-tpu` carries
  the right layout, so the engine works where the probe cannot. The rig now gates on
  `tpu-selftest`, which shares the engine's bindings.
- **rlx SIGSEGVs on teardown** after printing every success line — a marker-scanning check
  would call that healthy. Only the exit code separates it from a clean run.
- `libopenblas-dev` was missing from the startup script, and it fails at the *final link*
  after ~150 crates compile.

## The layering

```
gemma4-engine  ── rlx-gemma ──┐
                              ├── rlx (JAX-shaped IR) ── rlx-tpu ──┐
                              ┘                                     ├── libtpu.so (PJRT) ── v6e-1
xla-probe ────────────────── pjrt (PJRT C API) ─────────────────────┘
```

`tpu-selftest` takes the engine's own path minus the model — same rlx, same `rlx-tpu`, same
libtpu — which is what makes it predictive. `xla-probe` bypasses rlx and talks to the PJRT
C API directly, and that independence is exactly why it *cannot* be the gate: it fails on
its own schedule (today, on an ABI mismatch the engine does not have).

## Quick start

```bash
./init.sh                                  # register the MCP server (writes .mcp.json)
make rust                                  # build locally (needs protoc, clang, libopenblas-dev)
```

Then, through the MCP server or the skill:

```
create_tpu_vm_instance(workload="jaxrust")   # RUNNING is not ready
wait_for_jaxrust_ready                       # Rust + libtpu installed. Nothing built yet.
deploy_jaxrust_engine                        # uploads rust/, builds release, runs the self-test
manage_jaxrust_server(action="start", model_path="/path/to/gemma-4-E2B-it")
```

Each step answers a smaller question than the next, so when serving fails, the step that
still passes tells you where to look. That only holds while the steps share a stack — see
`rust/README.md` for how `xla-probe` violated it.

## Why the gate checks a number

`dlopen` on `libtpu.so` succeeds on a host with no chip attached, exactly as `import jax`
succeeds with no TPU backend. So `tpu-selftest` does not stop at "a device is listed": it
compiles a matmul, executes it, and checks that every element of the result is what
arithmetic says it should be. A plugin that loads, a client that creates and a device that
lists are three different facts, and none of them is evidence that the MXU computed
anything.

And the rig asserts on the **exit code**, never on the marker — because on this stack the
success line is printed and *then* the process segfaults in teardown. That is not a
hypothetical: it is what happens today, and a marker-scanning check calls it healthy.

## Licensing

This rig is Apache-2.0. The `gemma` cargo feature pulls in `rlx-gemma`, which is
**GPL-3.0-only**, making a binary built with it a GPLv3 combined work. Nothing is
distributed as a binary today — the source is uploaded and built on the VM — but the
default is a choice worth making deliberately. `rust/NOTICE.md` has the full table;
`xla-probe` never links it.

## Layout

| Path | What |
| --- | --- |
| `rust/` | The engine: `gemma4-engine` (server + `tpu-selftest`) and `xla-probe`. See `rust/README.md`. |
| `server.py` | The MCP server — provisioning, deployment, diagnostics |
| `.claude/skills/gce-jaxrust-v6e1-2b-management/` | The skill, with the four startup templates |
| `jax_engine.py`, `jax_openai_server.py`, `ports/gemma4/` | The Python JAX engine, kept as the parity oracle |
| `docs/gemma4-quirks.md` | The model's architecture and serving path, verified against the reference |
| `tests/` | 150 offline unittest tests |

The Python engine is not legacy being kept out of sentiment. It is the only implementation
here whose numbers have been checked against the reference module, and it no longer shares
a language, a graph builder or an author with the Rust one — which makes a disagreement
between them worth much more than a disagreement between two JAX code paths was.
