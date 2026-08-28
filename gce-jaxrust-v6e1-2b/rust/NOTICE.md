# Third-party licenses in the engine

This rig is Apache-2.0. Two of the crates it builds against are not, and the
difference matters in exactly one direction — outbound, when a **binary** built
here is handed to someone else. Nothing is distributed as a binary today: the
source is uploaded to the VM and built there by `deploy_jaxrust_engine`. The
moment that changes, so does what has to ship alongside it.

| Crate | Version | License | Reached by |
| --- | --- | --- | --- |
| `pjrt`, `pjrt-sys` | 0.2.0 | MIT OR Apache-2.0 | `xla-probe` |
| `rlx`, `rlx-runtime`, `rlx-tpu`, `rlx-ir`, … | 0.2.11 | MIT OR Apache-2.0 | `gemma4-engine`, always |
| **`rlx-gemma`** and the `rlx-models` family it pulls in (`rlx-qwen3`, `rlx-qwen35`, `rlx-cli`, `rlx-models-core`, …) | 0.2.11 | **GPL-3.0-only** | `gemma4-engine`, **only with `--features gemma`** |

## What that means

`rlx-gemma` carries the Gemma 4 model definition — the config parsing, the
per-layer attention dispatch, the QAT loader, the prefill/decode flow. It is the
reason this rig can serve Gemma 4 from Rust at all without re-porting the 1,570
lines of `ports/gemma4/jax_e_model.py`. It is also GPL-3.0-only, so a binary
linking it is a GPLv3 combined work: distributing that binary means offering its
complete corresponding source under GPLv3, and the Apache-2.0 notice on this
rig does not override that.

The engine is therefore built with the model behind a cargo feature:

```bash
cargo build --release -p gemma4-engine --no-default-features --features tpu        # permissive only
cargo build --release -p gemma4-engine --no-default-features --features tpu,gemma  # + GPLv3
```

`JAXRUST_CARGO_FEATURES` (default `tpu,gemma`) is what `deploy_jaxrust_engine`
passes. Without `gemma` the server still starts, still reports the device, and
refuses generation with a message naming the flag — which is enough to run the
whole capacity and probe path, and the capacity path is the expensive half to
verify.

`xla-probe` never links `rlx-gemma`. It is permissive throughout, which is
deliberate: the artefact that proves the chip works should not be the one with a
licensing question attached to it.

## Two other things worth knowing about these crates

**The model crates trail the framework.** `rlx` publishes 0.2.14 but
`rlx-gemma` 0.2.11 pins `rlx-runtime =0.2.11` exactly, so the whole workspace is
held at 0.2.11. Bumping `rlx` alone does not resolve; the pin has to move first.

**`rlx-gemma`'s crates.io blurb says "Gemma / Gemma 2", and that is stale.** The
source carries `gemma_e2b.rs`, `gemma4_e2b_mm.rs`, `gemma4_vision.rs`,
`gemma4_audio.rs`, `qat.rs` and `qat_loader.rs`, and its own module docs say
"Gemma / Gemma 2 / Gemma 3 / Gemma 4" with "no separate Gemma-4 code path" —
the per-layer accessors cover it. Do not conclude from the registry description
that Gemma 4 is unsupported.
