# `rust/` — the jaxrust engine

Two crates. `gemma4-engine` is the thing that is meant to serve; `gemma4-geometry` is the
arithmetic it and the rig reason about.

```bash
make rust-test         # 13 tests, no GPU required
make rust-lint         # cargo fmt --check + clippy -D warnings
make rust-check-cuda   # typechecks the CUDA path on any box, no toolkit needed
make rust-build        # the real binary: --no-default-features --features cuda,gemma
```

| Crate | Role |
| --- | --- |
| [`gemma4-engine`](gemma4-engine/) | OpenAI-compatible server on `rlx` + `rlx-gemma`, `Device::Cuda` |
| [`gemma4-geometry`](gemma4-geometry/) | E2B geometry, KV budgets, the padding invariant, dtype conversion |

## Running it on the instance

Nothing here is wired into `deploy_jax_server` yet — that ships the **Python** payload, and
swapping it is a deliberate step that should follow a run that actually works, not precede it.
Build and start it by hand first:

```bash
# On the G5g box, after the checkpoint is present:
cd rust && cargo build --release -p gemma4-engine \
    --no-default-features --features cuda,gemma

./target/release/gemma4-engine \
    --weights /opt/jax-g5g/models/gemma-4-E2B-it \
    --config  /opt/jax-g5g/models/gemma-4-E2B-it/config.json \
    --device cuda --max-model-len 4096
# waits for the graph to compile, then prints:
#   JAXRUST-SERVER: listening on 0.0.0.0:8000
```

It does **not** bind the port until the weights are loaded and the graph compiled, so a successful
connect is a real readiness signal rather than the "RUNNING is not READY" trap. Compare against the
Python engine on the same prompt before believing any number: `--device cpu` runs the same engine
off the chip for parity work.

## Status

**Compiles and tests green. Has never run.** No weights loaded, no token generated, nothing
measured, and **aarch64 is unverified** — everything here was checked on x86_64. See
`../docs/rust-jax-runtime-survey.md` for what that leaves open, and `../CLAUDE.md` for the
decisions the engine encodes.

## Two things to know before touching it

**`rlx-gemma` is GPL-3.0-only** and stops at 0.2.11 — `rlx` relicensed to MIT/Apache at 0.2.14 but
the model crate did not follow. Linking it makes the *binary* a GPLv3 combined work. The `gemma`
cargo feature is the gate; without it the server starts, reports the device, and refuses
generation.

**The default feature set is `cpu`, not `cuda`.** That is deliberate: the crate has to build on a
box with no driver, because that is where parity against the Python port gets done. The rig builds
with `make rust-build`.
