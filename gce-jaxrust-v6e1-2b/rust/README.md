# `rust/` — the engine

Two crates, three binaries. `tpu-selftest` is the one to run first.

| Binary | Crate | What it is |
| --- | --- | --- |
| `tpu-selftest` | `gemma4-engine` | **The gate.** Builds a matmul in rlx's IR, compiles it for the device, runs it, checks every element. |
| `gemma4-engine` | `gemma4-engine` | OpenAI-compatible server: rlx's JAX-shaped IR → HLO → PJRT → chip. |
| `xla-probe` | `xla-probe` | Low-level PJRT diagnostic. **Cannot create a client against current libtpu** — see below. |

**`xla-probe` is not the gate, and the reason is worth the paragraph.** It was written to
answer "a strictly smaller question" than the engine — but it does not, because it does not
share the engine's binding layer. It reaches PJRT through the `pjrt` crate; the engine
reaches it through `rlx-tpu`'s own hand-written FFI. Those are different ABIs of different
vintages, and on 2026-08-28 on a live v6e-1 they disagreed:

```
api version : Version { major_version: 0, minor_version: 75 }
Unexpected PJRT_Client_Create_Args size: expected 88, got 72.
```

`pjrt-sys` 0.2.0 builds that struct with 9 fields (72 bytes); libtpu 0.75 requires the
11-field, 88-byte layout that PJRT API 0.59 added for `kKeyValueTryGetCallback`. `rlx-tpu`
already carries the 88-byte version — with a source comment naming that exact API level —
so the engine works where the probe cannot even open a client.

**The lesson generalizes past this bug:** a gate that does not share the binding layer of
the thing it gates is not a smaller question, it is a different one, and it fails
independently in both directions. `tpu-selftest` lives inside `gemma4-engine` precisely so
it cannot drift from what it is vouching for.

## The layering, and which layer to file a bug against

```
gemma4-engine  ── rlx-gemma ──┐
                              ├── rlx (JAX-shaped IR, HIR→MIR→LIR) ── rlx-tpu ──┐
                              ┘                                                  ├── libtpu.so (PJRT plugin) ── v6e-1
xla-probe ─────────────────────────────────── pjrt (PJRT C API) ─────────────────┘
```

`xla-probe` deliberately bypasses rlx entirely and talks to the PJRT C API directly. That
independence was meant to isolate faults — and it is exactly why it cannot serve as the
gate: a stack nothing else uses fails on its own schedule. `tpu-selftest` takes the
engine's path minus the model, which is the property that actually makes a check
predictive.

## Why `pjrt` and not the `xla` crate

`gpu-jaxrust-g5g-2b/docs/rust-jax-runtime-survey.md` names `xla` (xla-rs) as the viable
lower-level option, and on a CUDA rig it is: it links a prebuilt `xla_extension`, and
`elixir-nx/xla` publishes aarch64+CUDA builds of exactly that.

**There is no equivalent argument on TPU.** The chip is reached through libtpu's PJRT
plugin, which is a `.so` implementing the PJRT C API and exporting `GetPjrtApi` — so the
right binding is the one that `dlopen`s a plugin by path, which is what `pjrt` does and
what `rlx-tpu` does underneath. Going through `xla_extension` would mean carrying a
second XLA build to reach a runtime that is already installed on the VM.

That is a TPU-specific conclusion and it does not transfer to the G5g rig. The two rigs
agree on `rlx` at the framework layer and differ below it, which is what "same migration,
different silicon" should look like.

## Building

Needs `protoc`, `clang` and `libopenblas-dev` on the build host. `pjrt-sys` runs
`prost-build` and `bindgen` in its build script, so without `protoc` the build dies with
``Could not find `protoc` `` several minutes in; and rlx-cpu links `-lopenblas` (pulled in
transitively by `rlx-runtime/tpu`), so without it the entire workspace compiles and only
the final link of the engine fails. The `jaxrust` startup script
installs both; on a workstation:

```bash
sudo apt-get install -y protobuf-compiler clang build-essential libopenblas-dev
cargo build --release                     # both crates, tpu + gemma features
cargo build --release -p gemma4-engine --no-default-features --features cpu,gemma
```

The second form is how correctness work happens off the chip: same engine, same
graph, CPU backend. Read `NOTICE.md` before building with `gemma` — that feature
pulls in GPL-3.0-only code.

Nothing here needs a TPU at build time. `libtpu.so` is `dlopen`ed at runtime, so
the whole workspace compiles on any x86-64 Linux host, which is the only reason
a build failure and a capacity failure can be kept apart.

## Running

```bash
# 1. Can rlx compile and execute on this chip at all? (the gate)
./target/release/tpu-selftest tpu     # `cpu` runs the identical graph off the chip

# 2. Serve.
./target/release/gemma4-engine \
  --weights /path/to/gemma-4-E2B-it \
  --max-model-len 8192 --device tpu --port 8000
```

The probe resolves the plugin from `LIBTPU_PATH`, then `TPU_LIBRARY_PATH`, then
`PJRT_PLUGIN_LIBRARY_PATH`, then a bounded search of the usual site-packages
roots; `argv[1]` overrides all of it. `LIBTPU_PATH` comes first because that is
the name `rlx-tpu` reads, and a rig that disagrees with its own engine about
which plugin it loaded is a bug that only surfaces on the chip.

## Things that are true of this engine and will surprise you

**rlx SIGSEGVs on teardown, after computing the right answer.** Measured on a v6e-1
2026-08-28: rlx-tpu 0.2.11 + libtpu 0.75 prints every success line and then dies with 139
while dropping the compiled graph — empty stderr, no panic. `std::process::exit`, which
skips destructors, exits 0 on the identical binary and inputs, so the fault is in PJRT
client teardown rather than in compile or execute. `tpu-selftest` exits that way by
default; `SELFTEST_RUN_DROP=1` reproduces the crash. **Never gate on the success marker
alone** — it is printed before the crash, so only the exit status tells the two apart.

**The runner is `!Send`.** PJRT's `Client` is `Rc`-based, so the compiled graph
and everything holding it are single-threaded. `engine.rs` puts the model on one
dedicated thread and talks to it over a channel; requests queue. On one v6e-1
chip that is not a limitation being worked around, it is the shape of the
hardware.

**Sampling is process-wide.** `GemmaRunner` takes its `SampleOpts` at build time
and keeps the field private, so a per-request `temperature` would mean rebuilding
the runner — which here means recompiling the graph. The HTTP layer accepts
OpenAI's `temperature`/`top_p` and logs that it dropped them, rather than
pretending they applied.

**Multi-turn chat is flattened.** `encode_chat_prompt_auto` takes one system
string and one user string, so earlier turns are rendered with role labels into
the final user turn. That is lossier than the real template's turn markers, and
it is the first place to look for a multi-turn quality regression.

**`--max-model-len` is a compile-time commitment**, not a runtime cap. The graphs
are statically shaped, so raising it costs KV cache HBM whether or not a request
ever uses the length.

## The parity oracle

`jax_openai_server.py`, `jax_engine.py` and `ports/gemma4/` stay in this rig on
purpose. They are the only implementation here whose numbers have been checked
against the reference module, so when this engine and that one disagree, the
evidence is about this engine. Build with `--features cpu,gemma` and compare on
the same host before spending a flex-start window on it.
