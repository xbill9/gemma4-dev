# Can this rig's engine be Rust? — a survey

**SURVEYED 2026-08-28. Desk research only: no crate here has been built, linked, or run on a T4G.**

> **REVISED the same day, and the first version of this survey was wrong in its headline.** It
> concluded that `xla-rs` was "the only live option". It had missed [`rlx`](https://crates.io/crates/rlx)
> — a JAX-shaped Rust ML compiler with a CUDA backend, actively released, **and already in use one
> directory over** in `gce-jaxrust-v6e1-2b`. Searching for "Rust JAX" surfaces the crates *named*
> after JAX; it does not surface the one that is actually shaped like it. **Check the sibling rigs
> before the registry.**
Every figure below is quoted from crates.io or the project's own repository on that date, and each is
the kind of fact that goes stale — re-check before acting on it.

This is `RIG-ANALYSIS.md` **step 7** ("is the software path reachable?") applied to a runtime swap
rather than a model. Step 7 is arithmetic and costs nothing; it comes before step 9, which costs a
provisioning cycle. The whole point of doing it first is that the answer turned out to be *no*, and
finding that out on paper cost an afternoon instead of a G5g launch.

## The question

`gpu-jax-g5g-2b` serves `google/gemma-4-E2B-it` at **13.1 tok/s** through a hand-written JAX port
under CPython. Can the *same rig* be driven from Rust — and if so, against what?

## The requirement, stated before looking at any crate

Writing the requirement down first is what makes the survey falsifiable. A replacement runtime must:

1. **Reach SM 7.5 on aarch64.** The GPU is a T4G on a Graviton2 host. This is the rare axis: not
   x86+CUDA, not arm+CPU, but **arm64 *and* CUDA together**.
2. **Compute in float16.** Turing has no bf16 and no fp8 datapath (`HARDWARE.md`). A runtime that
   only offers bf16 will not error — it will emulate through fp32, which is how this rig lost 86.8%
   of decode to a one-line dtype mismatch.
3. **Express the model's four irregularities.** E2B is not a stock transformer. Any port must carry
   two attention geometries (sliding `head_dim=256`, global **512**), 8:1 MQA, a KV-share map that
   collapses 35 layers onto 15 caches, and a 512-slot sliding ring — plus the padding-eviction
   invariant that `docs/padding-window-eviction.md` cost a week to find.
4. **Load a 9.5 GB safetensors checkpoint** and quantize it host-side without doubling resident
   memory, on a box whose device budget is 14.07 GB at **0.661 fragmentation**.

Requirement 1 is the cheapest to check and it is where two of three candidates die.

## The candidates

| Crate | Latest | Published | Downloads | GPU backend | Verdict |
| --- | --- | --- | ---: | --- | --- |
| [`rlx`](https://crates.io/crates/rlx) | 0.2.14 | **2026-08-12** | 1,276 | **CUDA** (cuBLAS/cuDNN/NVRTC) | **the candidate** |
| [`xla`](https://crates.io/crates/xla) (xla-rs) | 0.4.4 | **2026-08-22** | 7,210 | via XLA/PJRT | viable, lower level |
| [`nox`](https://crates.io/crates/nox) | 0.4.0 | **2024-08-23** | 2,971 | via XLA | abandoned |
| [`jax-rs`](https://crates.io/crates/jax-rs) | 0.5.1 | 2025-12-28 | **90** | **WebGPU only** | dead end |

### `rlx` — JAX-shaped, and the monorepo is already using it

Self-described as a "small ML compiler + runtime for transformer inference and training.
**JAX-shaped IR**, autodiff, vmap, on top of CPU / Apple Silicon / **NVIDIA** / AMD / TPU / wgpu
backends" ([MIT-RLX/rlx](https://github.com/MIT-RLX/rlx)). That is a far closer match to "the Rust
version of JAX" than anything named after JAX, and it is the only candidate that ships **model
implementations** rather than an op API.

**`gce-jaxrust-v6e1-2b` already builds against it** — `rlx`, `rlx-runtime` and `rlx-gemma` pinned at
`=0.2.11`, behind `tpu`/`cpu` cargo features, with an axum server. Read that rig before starting
here; it is the same migration one chip over, and it is further along.

**VERIFIED 2026-08-28 by reading the crate source and building against it.** The three things this
rig had to check that the v6e-1 rig did not:

- **`Device::Cuda` exists** — "NVIDIA GPU via native CUDA (cuBLAS, cuDNN)" in
  `rlx-driver/src/device.rs`, alongside `Cpu`, `Metal`, `Mlx`, `Ane`, `Rocm`, `OneApi`, `Tpu`,
  `Hexagon`, `Gpu`, `Vulkan`. `rlx-runtime`'s `cuda` feature is `["cpu", "dep:rlx-cuda"]`.
- **The CUDA path needs no toolkit AT BUILD TIME.** `rlx-cuda` goes through `cudarc` with
  `libloading`, i.e. it **dlopens CUDA at runtime**. `cargo check --features cuda,gemma` therefore
  succeeds on an x86_64 box with no driver and no toolkit, which is how the whole engine here was
  validated before any G5g was launched. Runtime still needs libcuda/cuBLAS/cuDNN present — that
  part of the "nothing would supply CUDA any more" caveat stands, but it is a *runtime* packaging
  question, not a build dependency, and it is much smaller than it looked.
- **`rlx-gemma` supports Gemma 4 E2B, and the crates.io description is STALE.** The registry says
  "Gemma / Gemma 2 causal LMs for RLX", which reads as a hard no for this rig. The source says
  otherwise: `gemma_e2b.rs`, `gemma4_e2b_mm.rs`, `gemma4_vision.rs`, `gemma4_audio.rs`,
  `GemmaArch::Gemma4`, `global_head_dim`, `sliding_window`, `per_layer_inputs` (PLE), plus
  `qat.rs`/`qat_loader.rs`. **This is the same trap as `jax-rs`' README, inverted** — there the
  metadata oversold a prototype, here it undersells a working crate. Read the source either way.

**Still unverified and the next thing to settle:** aarch64. The README documents ARMv7E-M for
microcontrollers and says nothing about aarch64 for the main pipeline, and everything above was
checked on x86_64. Graviton2 + CUDA remains the rare axis. Also unverified: whether the Gemma
builder assumes **bf16**, which Turing does not have — it will not error, it will emulate.

**A licensing correction, and it matters for the sibling.** `rlx` relicensed to MIT OR Apache-2.0 at
**0.2.14**, but **`rlx-gemma` never published 0.2.14 — it stops at 0.2.11 and is GPL-3.0-only.** So
the permissive relicense does *not* reach the model crate, and `gce-jaxrust-v6e1-2b`'s careful note
that linking `rlx-gemma` makes the binary a GPLv3 combined work **still stands in full**. An earlier
draft of this file guessed the constraint might be gone; it is not.

### `jax-rs` — the name is the best thing about it

It is the first hit for "Rust JAX" and its README claims "100% feature parity" and "production
ready". The repository is **13 stars and 7 commits**, and the crate has **90 downloads in total**.
Its own comparison table lists its GPU support as "✅ (WebGPU)" against JAX's "✅ (CUDA/ROCm)" —
so it fails requirement 1 by its own documentation, before maturity is even an argument.

**Do not let the README's confidence survive contact with the download count.** This is the clearest
case in the survey of marketing prose that reads exactly like a finished project.

### `nox` — architecturally right, two years cold

"Tensor library that compiles to XLA (like JAX, but for Rust)" is precisely the correct description
of what this rig would want. But **both published versions are dated 2024-08-23** and it is a
component of the [Elodin](https://github.com/elodin-sys/elodin) aerospace-simulation monorepo rather
than a standalone serving runtime. Two years without a release, against a `jax`/`jaxlib` line that
ships every few weeks, is not a dependency this rig can take.

### `xla` (xla-rs) — viable, but a level below `rlx`

Bindings to the **XLA C++ library** — the same compiler `jax` lowers to — by Laurent Mazare, who also
wrote `candle`. v0.4.4 landed **2026-08-22**, six days before this survey. Architecturally this is
exactly right: it would reuse the PJRT path whose cubins already carry `sm_75` (verified 2026-08-18),
rather than reimplementing kernels.

Two things temper it, and the second is a genuine blocker:

- **It is raw bindings, not a framework.** The author's own repository description is
  "*Experimentation* using the xla compiler from rust." There is no `nn` layer, no Flax equivalent,
  no loader. Everything above raw XLA ops is yours to write.
- **The aarch64 + CUDA binary question is RESOLVED, and the answer is yes.** This was flagged as the
  survey's one blocking unknown and then settled the same day against the
  [`elixir-nx/xla`](https://github.com/elixir-nx/xla) release assets (v0.10.0), which publish
  **nine** `xla_extension` builds. Four are aarch64, and two of those are CUDA:

  ```
  xla_extension-0.10.0-aarch64-linux-gnu-cuda12.tar.gz
  xla_extension-0.10.0-aarch64-linux-gnu-cuda13.tar.gz     <- matches this rig's stack
  ```

  **`cuda13` is exactly what this rig already runs** (`jax[cuda13]`, `2026-08-21-cuda13-py314-g5g`),
  and glibc clears with room: the binaries need 2.31+, the Ubuntu 26.04 base measured **2.43** on
  2026-08-27. So there is **no from-source XLA build on Graviton2**, and the rig's founding advantage
  — no 67-minute compile, per `docs/turing-aarch64-gap.md` — survives the move to Rust.

  Two things this does *not* settle, and the second is new work rather than a risk:

  - **Version alignment is unverified.** `xla` 0.4.4 against `xla_extension` 0.10.0 is an ABI pairing
    nobody here has tested. Check which extension version the crate's build script expects before
    assuming the newest asset is the right one.
  - **Nothing would supply CUDA any more.** This rig's install works because *pip* ships the CUDA
    libraries and the DLAMI supplies only the driver — that is the first bullet of "Why this rig
    exists next to `gpu-vllm-g5g-2b`". **A Rust binary has no pip.** CUDA 13 runtime, cuDNN ≥9.12,
    and NCCL ≥2.27 would have to come from apt, the DLAMI, or be vendored, which reintroduces a
    toolkit-shaped dependency the Python rig deliberately does not have. Not fatal, and not a build
    — but it is the one place where the "no toolkit" claim stops transferring.

## What the port would actually cost

Not a dependency swap. The Python that would have to be rewritten:

| File | Lines | What it holds |
| --- | ---: | --- |
| `ports/gemma4/jax_e_model.py` | 2,030 | the model, both attention geometries, the ring cache, quantization |
| `jax_openai_server.py` | 817 | OpenAI-compatible FastAPI, SSE streaming, metrics |
| `jax_engine.py` | 686 | dtype policy, bucket ladder, generation loop |
| `ports/gemma4/jax_e_loader.py` | 231 | safetensors → parameter tree |
| **total** | **3,764** | |

And it would be rewritten *downward*, from JAX's array API to raw XLA op-building — the abstraction
level of writing HLO by hand. The 2,030-line model file is the hard part, because it encodes findings
that were expensive to obtain and are invisible in the code's shape:

- **The padding-eviction invariant.** "A cache index is an absolute real position, and padding never
  occupies an index a real position uses." A fresh port that right-pads into the ring reproduces the
  original bug exactly, and the failure mode is a **`status="success"` token loop**, not a crash.
- **The dtype policy is read from the live device**, not from config, and `bfloat16` is deliberately
  excluded from the unsupported list because it emulates rather than failing.
- **`release_source` is opt-in**, because deleting unconditionally invalidates the caller's array —
  caught by a CPU test in seconds, and the sort of thing a rewrite silently drops.

`tests/` (136 passing) pins these against the Python implementation. **None of that carries to a Rust
port automatically**, and a Rust rig that passes no equivalent is not a port, it is a rewrite with the
same name.

## Verdict

**The migration is done as far as it can be done off the hardware.** `rust/gemma4-engine` is a
working OpenAI-compatible server built on `rlx` + `rlx-gemma` with `Device::Cuda`: it compiles, its
tests pass, and `cargo check --features cuda,gemma` succeeds. **The 3,764-line rewrite the first
draft of this survey predicted was not necessary** — `rlx-gemma` already implements Gemma 4 E2B
including PLE and QAT, so the work was an engine and an HTTP surface, not a model port.

**What is NOT established, and no amount of desk work will establish it: it has never run.** No
weights have been loaded, no token generated, nothing measured, and aarch64 is unverified. The
remaining risk is concentrated in three places — aarch64 support, whether the Gemma builder assumes
bf16 on a chip that lacks it, and runtime CUDA library packaging on the DLAMI.

**The honest one-line verdict: the engine exists and typechecks against the real API; whether it
serves a token is a G5g launch away, and that is the only thing that will settle it.**

The recommendation is therefore **staged, and the first stage is cheap**:

1. **Read `gce-jaxrust-v6e1-2b/rust/` first.** It is the same migration one chip over, already
   building, with the licensing and feature-gating worked out. Do not start from a blank crate.
2. **Settle `rlx` on this hardware, cheaply and in this order:** does it build for aarch64; does the
   `cuda` feature link against a Turing-era toolkit; does `rlx-gemma` (or `rlx-models`) carry a
   Gemma 4 builder, and does it assume bf16. Each is a build, not a launch.
   ~~Settle the arm64+CUDA `libxla_extension` question.~~ **Done — the binary exists** (above), which
   keeps `xla-rs` alive as the fallback.
3. **Port bottom-up against a parity harness, not top-down.** Load the checkpoint in
   Rust and assert tensor-for-tensor equality against the Python loader first; then one attention
   layer, both geometries, against `jax_e_parity_test.py`. The KV ring is the step to fear, not the
   matmuls.
4. **Keep `gpu-jax-g5g-2b` serving throughout.** It is the only thing in this family with a measured
   number, and it is the A/B control the `jaxrust` rig would be judged against.

**If the goal is Rust *serving* rather than Rust *JAX*, the answer is different and much better.**
[`candle`](https://github.com/huggingface/candle) — same author as `xla-rs` — has real CUDA support
and existing Gemma implementations. It would serve on this T4G far sooner than any XLA-from-Rust
route. But it is PyTorch-shaped, not JAX-shaped, so under `NAMING.md` slot 2 that rig is honestly
`candle`, **not `jaxrust`** — a different rig with a different name, not this one.

## Sources

- https://crates.io/crates/rlx · https://github.com/MIT-RLX/rlx
- https://crates.io/crates/jax-rs · https://github.com/cryptopatrick/jax-rs
- https://crates.io/crates/nox · https://github.com/elodin-sys/elodin
- https://crates.io/crates/xla · https://github.com/LaurentMazare/xla-rs
- https://github.com/elixir-nx/xla
