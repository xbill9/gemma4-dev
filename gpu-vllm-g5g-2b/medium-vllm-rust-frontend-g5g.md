---
title: "Installing Rust for vLLM on Graviton: a G5g walk-through 🦀"
published: false
description: "Step-by-step: getting vLLM's Rust frontend built and running on an aarch64 EC2 G5g box. rustup, setuptools-rust, protoc, the release flag, and how to prove the Rust frontend is actually the one answering."
tags: rust, vllm, aws, cuda
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-g5g-2b/images/header-vllm-rust-frontend-g5g.jpg
---
![](images/header-vllm-rust-frontend-g5g.jpg)


This tutorial walks through installing and setting up the **Rust toolchain for vLLM** on an
AWS EC2 **G5g** instance — Graviton2 (aarch64) with an NVIDIA T4G GPU — and getting vLLM's
Rust frontend (`vllm-rs`) built, running, and *verified*.

If you build vLLM from source, some of this applies to you on any architecture: since
v0.27.2rc0 the build has a hard Rust dependency. The aarch64 + Turing box is just where every
sharp edge shows up at once.

Everything below was run on the box. 🦀

---

#### Wait, vLLM has Rust in it?

You betcha. Since [PR #40848](https://github.com/vllm-project/vllm/pull/40848) (merged
2026-05-21), vLLM vendors a **14-crate Rust workspace**:

```
bench  chat  cmd  engine-core-client  llm  managed-engine  metrics
mock-engine  parser  parser/python  server  text  tokenizer  tracing
```

Edition 2024, resolver 3. Straight from the vendored `rust/Cargo.toml`:

| Crate | Version | Job |
| --- | --- | --- |
| `axum` | 0.8.8 | the HTTP server |
| `tokio` | 1.47.1 | async runtime |
| `zeromq` | 0.6.0 | talks to the Python engine |
| `rmp-serde` / `rmpv` | 1.3.1 | msgpack on the wire |
| `minijinja` | 2.22 | chat templates |
| `tonic` / `prost` | 0.14.6 / 0.14.3 | gRPC — **remember this one** |

It's a drop-in replacement for the Python FastAPI server. Two artifacts get built:

- 🦀 **`vllm-rs`** — the axum frontend binary
- 🐍 **`vllm._rust_tool_parser`** — a PyO3 extension module

---

#### Rust is a build requirement now

That's the headline, and it's reason enough on its own: **you cannot build vLLM from source at
v0.27.2rc0 without Rust in the picture.** `setup.py` imports it at module scope, line 21,
unguarded:

```python
from setuptools_rust.build import build_rust
```

No `try`, no feature flag, no opt-out. Metadata generation doesn't happen without it.

And this isn't a quirk of one release. vLLM's Rust surface is **14 crates** covering the HTTP
frontend, the tool parser, the tokenizer and the benchmark client, and it has been growing
since it landed. If you build inference infrastructure from source, a Rust toolchain is
becoming table stakes — so it's worth knowing how to drive it properly rather than working
around it.

Three things do get conflated, though, and they have different scopes:

| Component | Needed to build vLLM? | Needed to serve? |
| --- | --- | --- |
| `setuptools_rust` (Python pkg) | **yes, always** | no |
| `cargo` / `rustc` toolchain | for working Rust artifacts | no |
| `protoc` | for `vllm-rs` specifically | no |

#### Then why doesn't `pip install vllm` need this?

Because normally pip installs it for you. `pyproject.toml` declares it:

```toml
[build-system]
requires = [
    "cmake>=3.26.1", "ninja", "packaging>=24.2",
    "setuptools>=77.0.3,<81.0.0", "setuptools-scm>=8.0",
    "setuptools-rust>=1.9.0",          # <- pip grabs this automatically
    "torch == 2.13.0",                 # <- ...and this. Which is the problem.
    "wheel", "jinja2",
]
```

Under normal **build isolation**, pip creates a clean env, installs that list, and builds.
You never see `setuptools_rust` because you never had to think about it.

But look at the `torch` pin. Building in isolation means pip installs **torch 2.13.0 from
PyPI** — and the PyPI aarch64 wheels are built for sm_80 and up. **No `sm_75`.** Which
destroys the entire reason for building from source on a T4G.

So on this box you must build against the DLAMI's own torch, and that means:

```bash
python use_existing_torch.py
pip install -e . --no-build-isolation
```

**`--no-build-isolation` turns off the automatic install of everything in that `requires`
list.** From that moment on, every build dependency is yours to supply by hand — including
`setuptools_rust`, which is why it turns up as a bare `ModuleNotFoundError` minutes into a
build that has nothing visibly to do with Rust.

So the toolchain was always required; isolation was just hiding it. Building this way means
you own the dependency list, which is the rest of this walk-through. ⚡

---

#### What the DLAMI gives you, and what it doesn't

The AWS Deep Learning ARM64 AMI ships a **runtime**, not a build environment. On a fresh box:

| Thing | Present? |
| --- | --- |
| PyTorch 2.12 with `sm_75` | ✅ |
| NVIDIA driver | ✅ |
| `nvcc` / CUDA toolkit | ❌ |
| Rust toolchain | ❌ |
| `setuptools_rust` | ❌ |
| **`protoc`** | ❌ |

Four of those six are on you. Let's install them.

---

#### Step 1 — Rust itself

Standard rustup, nothing aarch64-specific about it:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
```

```
stable-aarch64-unknown-linux-gnu installed - rustc 1.97.1 (8bab26f4f 2026-07-14)

Rust is installed now. Great!
```

Note the triple: `stable-aarch64-unknown-linux-gnu`. Rust's aarch64 support is a **complete
non-event**, which is a lovely change of pace on this hardware. ⚡

---

#### Step 2 — setuptools-rust

```bash
python3 -m pip install setuptools_rust
```

Per the section above: `--no-build-isolation` means pip won't do this for you. Do it **early**
— the failure lands during metadata generation, minutes into a build, as a bare
`ModuleNotFoundError: No module named 'setuptools_rust'` nowhere near anything that looks
like Rust.

⚠️ Install it into the **same interpreter you'll build with**. On the DLAMI that's
`/opt/pytorch/bin/python3`, not the system `python3` — they're different, and the one that
matters is whichever owns the torch you're building against.

---

#### Step 3 — protoc 🔎

This is the one nobody documents:

```bash
apt-get install -y protobuf-compiler
protoc --version
```

```
libprotoc 3.21.12
```

**Why:** `vllm-rs` depends on the `vllm-server` crate, `vllm-server` builds gRPC stubs with
`tonic`/`prost`, and `prost-build` shells out to `protoc`. Skip it and the frontend binary
does not get built — see the summary at the end for how loudly that *doesn't* fail.

The tool parser has no protobuf dependency, which is why it builds either way.

---

#### Step 4 — the CUDA toolkit, while you're here

Not Rust, but the same class of problem, and you need it for vLLM's kernels:

```bash
# NVIDIA's **sbsa** repo — not the x86 one, easy reflex to get wrong on Arm
apt-get install -y cuda-toolkit-13-2
```

---

#### Step 5 — build the Rust artifacts

```bash
cd /opt/vllm-src
python tools/build_rust.py --release
```

⚠️ **Do not omit `--release`.** setuptools-rust builds inplace targets in debug by default,
and `pip install -e .` is an inplace build. The difference is not subtle:

| Artifact | Debug | Release |
| --- | --- | --- |
| `_rust_tool_parser.abi3.so` | 100,913,216 B | **1,009,080 B** |

**100x.** The debug artifact is four times the size of *every CUDA kernel in vLLM combined*.

Timing on a `g5g.xlarge` (4 vCPU), cold:

```
real    9m1.746s
user    25m9.199s
sys     1m35.023s
```

**501 crates. Zero warnings. Exit 0.** 🟢

Rust's aarch64 support does not put up a fight here — which is a pleasant contrast with the
CUDA side of this box, where SM 7.5 on Graviton needs a custom arch list and a patched
kernel.

---

#### Step 6 — check what you got

```bash
ls -la vllm/vllm-rs vllm/_rust_tool_parser.abi3.so
```

```
-rwxr-xr-x 1 root root 50039024 vllm/vllm-rs
-rwxr-xr-x 1 root root  1009080 vllm/_rust_tool_parser.abi3.so
```

```bash
file vllm/vllm-rs
```

```
ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV),
dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, not stripped
```

```bash
vllm/vllm-rs --help
```

```
Rust frontend and managed-engine CLI for vLLM.

Commands:
  frontend  Run the Rust OpenAI frontend as a Python-supervised worker
  serve     Launch a managed Python headless engine, then run the Rust OpenAI frontend
  bench     Run vLLM benchmarks
  render    Run engine-free request rendering and preprocessing
```

If `vllm/vllm-rs` isn't there, go back to **Step 3**.

---

#### Step 7 — run it, and mind the entrypoint ⚠️

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve google/gemma-4-E2B-it \
  --dtype float16 \
  --kv-cache-dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 --port 8000
```

**It must be `vllm serve`.** If you launch the module directly —

```bash
# ❌ VLLM_USE_RUST_FRONTEND is IGNORED here
python -m vllm.entrypoints.openai.api_server --model … --host 0.0.0.0 --port 8000
```

— the variable does nothing. No warning, no `Unknown vLLM environment variable` line. The
server comes up healthy and serves happily on the Python frontend, and a benchmark run
against it looks entirely normal.

The flag is read in exactly two places:

```
vllm/entrypoints/cli/serve.py:62        envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None
vllm/entrypoints/openai/dp_supervisor.py:261   if envs.VLLM_USE_RUST_FRONTEND and envs.VLLM_RUST_FRONTEND_PATH:
```

`api_server.py` never mentions it.

---

#### How do I know it's actually Rust? 🔎

Three checks. Do all three the first time.

**1. The `server:` header:**

```bash
curl -si localhost:8000/health | grep -i '^server:'
```

| Frontend | Response |
| --- | --- |
| 🐍 Python | `server: uvicorn` |
| 🦀 Rust | *(no `server:` header at all)* |

**2. The process:**

```bash
pgrep -af vllm-rs
```

```
26588 /opt/vllm-src/vllm/vllm-rs frontend --listen-fd 17
  --input-address  ipc:///tmp/f60f3962-d45b-4bcd-9026-c0dc32736028
  --output-address ipc:///tmp/5f75411d-2787-43bb-b4fc-14bf504a1cce
  --engine-start-index 0 --engine-count 1 --data-parallel-size 1
```

**3. The log prefix** — `(RustFrontend pid=…)` instead of `(APIServer pid=…)`:

```
INFO [utils.py:392] Launching Rust frontend: /opt/vllm-src/vllm/vllm-rs frontend --listen-fd 17 …
```

---

#### So where does Rust actually sit?

In **two** places, and they're quite different. One is a separate process; the other is a
shared object loaded *inside* the Python process. Here's the whole VM:

```
┌─ EC2 g5g.4xlarge ── Graviton2, aarch64 ─────────────────────────────────────┐
│                                                                             │
│  Deep Learning ARM64 AMI · Ubuntu 24.04 · NVIDIA driver 595.71.05           │
│  you add > cuda-toolkit-13-2 (sbsa) · rustup 1.97.1 · protobuf-compiler     │
│                                                                             │
│      HTTP :8000                                                             │
│          |                                                                  │
│          v                                                                  │
│  ┌───────────────────────────────┐                                          │
│  │ [RUST] vllm-rs                │  50 MB aarch64 ELF, its OWN process      │
│  │        axum 0.8.8 · tokio     │  built from the vendored rust/ workspace │
│  │        minijinja · fastokens  │  <- Step 5                               │
│  └────────┬─────────────▲────────┘                                          │
│           |             |                                                   │
│  ipc://   | ROUTER      | PULL     msgpack (rmp-serde / rmpv)               │
│           v             |                                                   │
│  ┌────────┴─────────────┴────────┐                                          │
│  │ [PY]   vLLM supervisor        │  `vllm serve` opens the socket, then     │
│  │                               │  hands listen-fd 17 down to vllm-rs      │
│  └────────┬──────────────────────┘                                          │
│           | spawns                                                          │
│           v                                                                 │
│  ┌───────────────────────────────┐                                          │
│  │ [PY]   EngineCore             │  torch 2.12.0+cu132, arch list has sm_75 │
│  │  ┌─────────────────────────┐  │                                          │
│  │  │ [RUST] _rust_tool_parser│  │  PyO3 .so LOADED INTO the Python         │
│  │  │        1.0 MB release   │  │  process — not a process of its own      │
│  │  └─────────────────────────┘  │                                          │
│  └────────┬──────────────────────┘                                          │
│           | CUDA                                                            │
│           v                                                                 │
│  ┌───────────────────────────────┐                                          │
│  │ NVIDIA T4G · SM 7.5           │  15,360 MiB GDDR6 · 277 GB/s measured    │
│  │ TRITON_ATTN kernels           │  weights 9.94 GiB · KV 2.95 GiB          │
│  └───────────────────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

Two things worth pulling out of that picture:

- **`vllm-rs` is not a sidecar you point at a port.** The Python side opens the listening
  socket and passes the *file descriptor* down. It's a worker the supervisor forks and feeds.
- **`_rust_tool_parser` is Rust living inside Python.** It's the one that always builds
  (no `protoc` needed), which is why a broken install still leaves Rust on the box — just not
  the Rust you wanted.

And note where the GPU sits relative to all of this: at the bottom, behind everything. That's
the reason the benchmark below comes out the way it does.

---

#### Bonus: there's a Rust benchmark client too

```bash
VLLM_USE_RUST_BENCH=1 vllm bench serve …
```

Same binary, `bench` subcommand. Requires `VLLM_RUST_FRONTEND_PATH` to resolve, so it needs
the same Step 3 → Step 5 you just did.

---

#### Two warnings you should not scroll past 🔴

The server came up healthy. These went by in the startup log anyway.

**Gemma 4 defeats the fast tokenizer:**

```
INFO    [hf.rs:200] loading tokenizer with fastokens
WARNING [hf.rs:221] failed to load tokenizer with fastokens; falling back to
        HuggingFace tokenizers
        error=tokenizer error: normalizer error: unsupported normalizer type: Replace
```

`fastokens` 0.2.1 doesn't implement the `Replace` normalizer that Gemma 4's `tokenizer.json`
uses, so it falls back to the same HuggingFace `tokenizers` the Python path uses. Note the
fallback is graceful and correct — you just don't get the fast path on *this* model yet. It's
a coverage gap in a young crate, and one normalizer away from closing.

**Multimodal isn't wired up for this model:**

```
WARNING [multimodal.rs:446] multimodal model spec is not registered; disabling
        image/video support   model_id="google/gemma-4-E2B-it" model_type="gemma4"
```

Gemma 4 E2B is a **vision** model, and `gemma4` isn't in the Rust multimodal spec table yet.
Text requests behave identically and the endpoint is healthy, so nothing in a normal check
reveals it. Also a registration gap rather than a design problem — but check it for your model
before you switch, because a healthy endpoint won't tell you.

---

#### Is it faster?

On a T4G, no. Output token throughput, same engine config, client on the box against
localhost:

| Concurrency | 🐍 Python | 🦀 Rust |
| --- | --- | --- |
| 1 | 28.65 | 29.30 |
| 4 | 97.48 | 97.26 |
| 8 | 168.33 | 169.39 |
| 16 | 169.96 | 170.19 |
| 32 | 170.99 | 170.34 |

Median TTFT tracks just as tightly — 14305 ms against 14311 ms at concurrency 32.

That's the expected result, and worth saying plainly: decode on this card is
**bandwidth-bound** at a measured 277 GB/s, and the engine saturates at `--max-num-seqs 8`.
A frontend rewrite targets CPU-side per-request overhead. Here that overhead hides behind the
GPU, so swapping it can't move a bottleneck-limited number. **If you want the Rust frontend
to buy you tokens per second on a small GPU, it won't.**

One signal does appear, in median inter-token latency at high concurrency:

| Concurrency | 🐍 Python | 🦀 Rust | Δ |
| --- | --- | --- | --- |
| 16 | 38.55 | **36.18** | **−6.4%** |
| 32 | 38.41 | **36.23** | **−5.9%** |

Mean TPOT barely moves, so this is the middle of the distribution tightening rather than
everything speeding up — the shape you'd expect from a frontend scheduling streaming work
more evenly once many streams are in flight. Worth knowing if you serve at concurrency; not
worth switching for on its own. 📊

---

#### If a plain `pip install -e .` already ran

A from-source vLLM install done **without** the steps above succeeds, exits 0, and leaves you
with a 96 MB debug tool parser and no frontend binary. Four defaults stack up to make that
silent:

| Symptom | Cause | Fix |
| --- | --- | --- |
| No `vllm/vllm-rs` after a clean build | `protoc` absent ⇒ `vllm-server` fails with code 101 | Step 3 |
| `pip install` exits 0 anyway | `optional=not should_require_rust_frontend()` — setuptools-rust swallows it | `VLLM_REQUIRE_RUST_FRONTEND=1` |
| `_rust_tool_parser.abi3.so` is ~96 MB | editable ⇒ inplace ⇒ debug profile | `--release` |
| `FileNotFoundError: … vllm-rs was not found` | the above, discovered at import time | Steps 3 + 5 |
| Healthy server, but `server: uvicorn` | flag set on the `api_server` module, which never reads it | `vllm serve` |

`VLLM_REQUIRE_RUST_FRONTEND=1` turns the second row into a hard build failure, which is what
you want on any machine you plan to serve from.

---

#### Cheat sheet

```bash
# toolchain — you supply these by hand because the sm_75 requirement
# forces --no-build-isolation, which disables pip's automatic build deps
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
/opt/pytorch/bin/python3 -m pip install setuptools_rust   # the BUILD interpreter
apt-get install -y protobuf-compiler cuda-toolkit-13-2

# build against the DLAMI's torch, not a PyPI one (PyPI aarch64 has no sm_75)
cd /path/to/vllm
python use_existing_torch.py
TORCH_CUDA_ARCH_LIST=7.5 VLLM_REQUIRE_RUST_FRONTEND=1 \
  pip install -e . --no-build-isolation

# Rust artifacts, release profile (editable installs default to debug: 100x bigger)
VLLM_REQUIRE_RUST_FRONTEND=1 python tools/build_rust.py --release

# confirm
ls -la vllm/vllm-rs && vllm/vllm-rs --help

# run — `vllm serve`, NOT the api_server module
VLLM_USE_RUST_FRONTEND=1 vllm serve <model> --host 0.0.0.0 --port 8000

# verify it's really Rust
curl -si localhost:8000/health | grep -i '^server:'   # Rust sends none
pgrep -af vllm-rs
```

---

*Run on EC2 `g5g.xlarge` and `g5g.4xlarge`, `us-east-1a`, NVIDIA T4G (SM 7.5). vLLM
`0.27.2rc1.dev0+g7f7a32cfe`, rustc 1.97.1, setuptools-rust 1.13.0, libprotoc 3.21.12,
torch 2.12.0+cu132. Benchmarks are one run per cell for Rust and two for Python; treat the
TPOT delta as suggestive.*
