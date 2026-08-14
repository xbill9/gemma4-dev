# 2026-08-14 — vLLM's Rust frontend on G5g: why the source build silently omits it

**Finding: a from-source vLLM install on this rig produces a 96 MB debug-profile Rust
tool parser and no Rust frontend binary at all — and reports success while doing it.**
The cause is a three-way interaction between an absent `protoc`, `optional=True` on the
Rust extensions, and setuptools-rust's inplace-means-debug default. All of it was executed
on hardware; the `protoc` link is established by a controlled A/B, not inferred.

## What vLLM v0.27.2rc0 ships in Rust

The tag this rig builds carries a 14-crate Rust workspace at `rust/` — `bench`, `chat`,
`cmd`, `engine-core-client`, `llm`, `managed-engine`, `metrics`, `mock-engine`, `parser`,
`parser/python`, `server`, `text`, `tokenizer`, `tracing` — on edition 2024, resolver 3,
with axum 0.8.8 and hyper 1.10.1. It landed via
[PR #40848](https://github.com/vllm-project/vllm/pull/40848), merged 2026-05-21.

`tools/build_rust.py` registers exactly two artifacts:

| Target | setuptools-rust binding | Needs `protoc`? |
| --- | --- | --- |
| `vllm.vllm-rs` — the axum OpenAI frontend binary | `Binding.Exec` | **yes**, via `vllm-server` → `prost-build` |
| `vllm._rust_tool_parser` | `Binding.PyO3`, abi3-py38 | no |

`vllm-rs --help` (built here) reports four subcommands: `frontend`, `serve`, `bench`,
`render`. Runtime switches are `VLLM_USE_RUST_FRONTEND=1`, `VLLM_USE_RUST_BENCH=1` and
`VLLM_RUST_FRONTEND_PATH` (default `auto`).

## The failure, as shipped

`ami-0b44b90b3d02430ee` was built 2026-08-12 with an **editable** install — confirmed on the
image:

```
direct_url: {"dir_info": {"editable": true}, "url": "file:///opt/vllm-src"}
setuptools_rust 1.13.0
```

What that install left behind in `/opt/vllm-src/vllm/`:

| Artifact | Size |
| --- | --- |
| `_C_stable_libtorch.abi3.so` — all of vLLM's CUDA kernels | 16,355,424 B |
| `_moe_C_stable_libtorch.abi3.so` | 8,759,728 B |
| **`_rust_tool_parser.abi3.so`** | **100,913,216 B** |
| `cumem_allocator.abi3.so` | 73,344 B |
| `fs_io_C.abi3.so` | 73,592 B |
| `spinloop.abi3.so` | 71,096 B |
| **`vllm-rs`** | **absent** |

The debug-profile Rust tool parser is **4.0x the size of every CUDA kernel in vLLM
combined**, on an image whose entire reason for existing is `sm_75` kernels.

Asking for the frontend fails at import time:

```
FileNotFoundError: VLLM_RUST_FRONTEND_PATH=auto but the vllm-rs binary was not found
at /opt/vllm-src/vllm/vllm-rs. Build with setuptools-rust or set the path explicitly.
```

## Why — three causes, one silent outcome

**1. `protoc` is absent from the DLAMI.** Established by controlled A/B on two instances
launched from the same AMI. On `g5g.4xlarge` `i-010083a8387e9198e`, image as shipped,
`which protoc` → absent, and the release build fails:

```
error: failed to run custom build command for `vllm-server v0.1.0 (/opt/vllm-src/rust/src/server)`
  Error: Custom { kind: NotFound, error: "Could not find `protoc`. If `protoc` is installed,
  try setting the `PROTOC` environment variable to the path of the `protoc` binary. ...
  For more information: https://docs.rs/prost-build/#sourcing-protoc" }
error: `cargo build --manifest-path rust/src/cmd/Cargo.toml --message-format=json-render-diagnostics
  --release --features native-tls-vendored --bin vllm-rs` failed with code 101
```

`vllm-rs` depends on the `vllm-server` crate; `vllm-server`'s build script needs `protoc`.
`_rust_tool_parser` does not depend on `vllm-server`, which is exactly why one artifact
exists on the image and the other does not.

After `apt-get install protobuf-compiler` (libprotoc 3.21.12) on `g5g.xlarge`
`i-037eb7384d0aa7c92`, the same command succeeds — see the next section. That is the A/B:
same image, same command, `protoc` the only variable.

**2. The extensions are optional by default.** `setup.py` line 1264:

```python
rust_extensions = rust_build.rust_extensions(
    optional=not should_require_rust_frontend()
)
```

and `should_require_rust_frontend()` is false unless `VLLM_REQUIRE_RUST_FRONTEND` is set.
So setuptools-rust swallows the code-101 failure and **`pip install` exits 0**. The build
does not warn in a way that survives to the installed tree; nothing on the image records
that a component was dropped.

**3. Editable install means debug profile.** setuptools-rust builds inplace targets in
debug unless told otherwise, which is why the parser is 96 MB rather than 1 MB.

## Building it properly — measured

`python tools/build_rust.py --release`, with `protoc` present:

| | `g5g.xlarge` (4 vCPU), cold | `g5g.4xlarge` (16 vCPU), warm |
| --- | --- | --- |
| Wall time | **9 m 01.7 s** | 1 m 32.9 s |
| CPU time | 25 m 09.2 s user + 1 m 35.0 s sys | 13 m 35.4 s user + 27.4 s sys |
| Crates compiled | 501 | — |
| Compiler warnings | **0** | 0 |
| Exit | 0 | 0 |

**The two columns are not comparable.** The `4xlarge` figure is a warm rebuild: the
protoc-less control run on that host had already compiled most of the dependency graph
before failing at `vllm-server`, so cargo reused it. Only the `xlarge` number is a cold
release build. Treat 9 m 01.7 s as the honest figure for "build the Rust frontend from
nothing", and note it was taken on a host that was also carrying 16 GiB of swap.

Artifacts, release profile:

| Artifact | Debug (as shipped) | Release | Ratio |
| --- | --- | --- | --- |
| `_rust_tool_parser.abi3.so` | 100,913,216 B | **1,009,080 B** | **100.0x smaller** |
| `vllm-rs` | *never built* | **50,039,024 B** | — |

```
/opt/vllm-src/vllm/vllm-rs: ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV),
dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, for GNU/Linux 3.7.0, not stripped
```

**The Rust frontend compiles clean on aarch64 with zero warnings.** Unlike the CUDA side of
this rig, there is no architecture problem here at all — Rust's aarch64 support is a
non-event. The only thing standing between this image and a working `vllm-rs` was a missing
protobuf compiler and an `optional=True`.

## Frontend comparison — method

Both frontends run the **same engine config** on the **same instance**, back to back, and
the benchmark client runs **on the box against `localhost`**, so no network sits between the
client and the frontend under test. The only variable between the two runs is
`VLLM_USE_RUST_FRONTEND=1` in `/opt/serve-rust.sh`; every serving flag is identical.

```
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions
  --dataset-name random --random-input-len 512 --random-output-len 128
  --num-prompts $((C*4)) --max-concurrency C --ignore-eos
```

**The engine saturates at `--max-num-seqs 8`.** Above that, throughput is flat and extra
concurrency converts directly into queueing delay, so the c=16 and c=32 cells compare
queueing behaviour rather than throughput. That is a property of the engine, not the
frontend, and it applies equally to both runs.

### Python frontend (baseline)

| Concurrency | Output tok/s | TTFT median (ms) | TTFT P99 (ms) | TPOT median (ms) | Duration (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 28.65 | 409.12 | 764.51 | 31.44 | 17.87 |
| 4 | 97.48 | 1433.11 | 1455.64 | 32.71 | 21.01 |
| 8 | 168.33 | 382.36 | 2859.61 | 35.96 | 24.33 |
| 16 | 169.96 | 4969.23 | 9866.46 | 38.55 | 48.20 |
| 32 | 170.99 | 14305.55 | 24272.78 | 38.41 | 95.82 |

Engine start to healthy endpoint: **225 s** (warm volume). KV cache 329,579 tokens, matching
the 2026-08-12 first-serve run exactly.

### Setting `VLLM_USE_RUST_FRONTEND=1` on the wrong entrypoint does nothing

**This cost a whole benchmark run and is worth recording as a finding in its own right.**

The first attempt set `VLLM_USE_RUST_FRONTEND=1` on the rig's normal serving command —

```
python3 -m vllm.entrypoints.openai.api_server --model … --host 0.0.0.0 --port 8000
```

— which is what `/opt/serve.sh` on the AMI runs, and what `server.py`'s `_user_data` renders
into the launch script. The server came up healthy in 266 s and benchmarked fine. It was
still the Python frontend:

```
$ curl -si localhost:8000/health | grep ^server:
server: uvicorn

$ pgrep -af vllm-rs
(nothing)
```

The variable is consumed **only** on the `vllm serve` CLI path. From the source on the image:

```
vllm/entrypoints/cli/serve.py:61   rust_frontend_path = (
vllm/entrypoints/cli/serve.py:62       envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None
vllm/entrypoints/openai/dp_supervisor.py:261   if envs.VLLM_USE_RUST_FRONTEND and envs.VLLM_RUST_FRONTEND_PATH:
```

`vllm/entrypoints/openai/api_server.py` never mentions it. There is no warning, no
`Unknown vLLM environment variable` line — the flag is simply inert, and the only way to
notice is to check the `server:` response header or look for the process.

**Implication for this rig:** `/opt/serve.sh` and `server.py`'s user-data template both use
the module entrypoint, so neither can ever reach the Rust frontend regardless of environment.
Switching to `vllm serve` is a prerequisite for using it at all.

That makes four independent silent failures in one feature: the missing `protoc`, the
`optional=True` swallow, the debug-profile default, and now an env var that only one of two
equivalent-looking entrypoints reads.

### Python frontend, repeat run

The mislabelled run is retained as `bench-python-repeat/` — same config, same instance,
second execution — because it gives the run-to-run variance the 2026-08-12 report lacked.

### Rust frontend — confirmed engaged

Under `vllm serve` with `VLLM_USE_RUST_FRONTEND=1`, the supervisor launches the binary:

```
INFO [utils.py:392] Launching Rust frontend: /opt/vllm-src/vllm/vllm-rs frontend
  --listen-fd 17
  --input-address  ipc:///tmp/f60f3962-d45b-4bcd-9026-c0dc32736028
  --output-address ipc:///tmp/5f75411d-2787-43bb-b4fc-14bf504a1cce
  --engine-start-index 0 --engine-count 1 --data-parallel-size 1
  --args-json {"dtype": "float16", "max_model_len": 16384, "max_num_seqs": 8, …}
```

Transport is **`ipc://` Unix sockets**, not TCP, for a single-node deployment. Three
independent confirmations that it is the frontend answering:

| Check | Python | Rust |
| --- | --- | --- |
| `curl -si /health \| grep ^server:` | `server: uvicorn` | **no `server:` header** |
| `pgrep -af vllm-rs` | nothing | `vllm-rs frontend --listen-fd 17 …` |
| Log prefix | `(APIServer pid=…)` | `(RustFrontend pid=…)` |

#### Two functional regressions, both silent-ish

**1. Gemma 4's tokenizer defeats `fastokens`.** The Rust frontend's fast tokenizer cannot
load this model and falls back to the same HuggingFace `tokenizers` the Python path uses:

```
INFO    [hf.rs:200] loading tokenizer with fastokens
WARNING [hf.rs:221] failed to load tokenizer with fastokens; falling back to
        HuggingFace tokenizers
        error=tokenizer error: failed to load tokenizer:
              normalizer error: unsupported normalizer type: Replace
```

Gemma 4's `tokenizer.json` uses a `Replace` normalizer that `fastokens` 0.2.1 does not
implement. Tokenization is a large part of what a frontend does, so **on this model the Rust
frontend gives up one of its main advantages before serving a single request.** It is a
warning, not an error, and nothing downstream reports it.

**2. Multimodal support is silently disabled.**

```
WARNING [multimodal.rs:446] multimodal model spec is not registered; disabling
        image/video support for this model  model_id="google/gemma-4-E2B-it" model_type="gemma4"
WARNING [multimodal.rs:413] no multimodal modality resolved; disabling multimodal
        support for this model
```

**Gemma 4 E2B is a vision model** — the Python path loads a SigLIP image processor for it,
and this rig's engine profiles an encoder cache of 2,496 tokens. Under the Rust frontend that
capability is switched off, with a warning and a healthy endpoint. Text-only requests behave
identically, so nothing in a normal health check would reveal it.

This is a genuine capability regression, not a packaging problem, and it is the strongest
argument against using the Rust frontend for Gemma 4 today regardless of what the throughput
numbers say.

#### Startup

Healthy in **85 s**, against 266 s for the module-entrypoint Python run earlier in the same
session. **Do not read this as a Rust win.** The two are not controlled: by the 18:23 run the
9.54 GiB checkpoint was in a 30 GiB page cache and the vLLM compile cache was warm, whereas
the 18:14 run was still paying first-touch EBS costs. Startup was not the object of this
experiment and this number should not be quoted.

#### Sweep

Three runs on one instance, same engine config throughout. **Py-1** and **Py-2** are two
executions of the Python frontend — Py-2 is the mislabelled run, kept because two identical
runs are what makes the third interpretable. **Rust** is the confirmed Rust frontend.

**Output token throughput (tok/s)** — no difference:

| Concurrency | Py-1 | Py-2 | Rust |
| --- | --- | --- | --- |
| 1 | 28.65 | 22.70 | 29.30 |
| 4 | 97.48 | 97.26 | 97.26 |
| 8 | 168.33 | 169.22 | 169.39 |
| 16 | 169.96 | 169.60 | 170.19 |
| 32 | 170.99 | 170.61 | 170.34 |

**Median TTFT (ms)** — no difference:

| Concurrency | Py-1 | Py-2 | Rust |
| --- | --- | --- | --- |
| 1 | 409.12 | 409.10 | 425.80 |
| 4 | 1433.11 | 1446.18 | 1446.09 |
| 8 | 382.36 | 320.97 | 330.96 |
| 16 | 4969.23 | 4987.82 | 4969.34 |
| 32 | 14305.55 | 14342.69 | 14311.22 |

**Median TPOT (ms)** — the one place a signal appears:

| Concurrency | Py-1 | Py-2 | Rust | Rust vs Python |
| --- | --- | --- | --- | --- |
| 1 | 31.44 | 31.68 | 31.20 | −1.0% |
| 4 | 32.71 | 32.75 | 32.67 | −0.2% |
| 8 | 35.96 | 36.04 | 35.75 | −0.7% |
| 16 | 38.55 | 38.71 | **36.18** | **−6.4%** |
| 32 | 38.41 | 38.60 | **36.23** | **−5.9%** |

#### Reading it

**Throughput and TTFT are unchanged, and that is the expected result.** Decode on a T4G is
GPU-bandwidth-bound — 277 GB/s measured on this card — and the engine saturates at
`--max-num-seqs 8`. The frontend was never the bottleneck, so replacing it cannot move the
bottleneck-limited numbers. A frontend rewrite is aimed at CPU-side per-request overhead, and
on this hardware that overhead is invisible behind the GPU.

**Median TPOT is ~6% lower under Rust at c=16 and c=32.** This is the one result worth
keeping. The two Python runs agree with each other to 0.4% at those cells, so a 6% gap sits
well outside run-to-run noise, and it appears independently at both high-concurrency levels.
Mean TPOT moves much less (c=32: 40.22 vs 40.81/40.89, −1.5%), so the effect is a tightening
of the middle of the inter-token distribution rather than a shift of the whole thing —
consistent with a frontend that schedules streaming work more evenly once many streams are
in flight.

**Caveat that limits it:** one Rust run, no repeats, so there is no Rust-side variance figure.
Two Python runs bound the noise; one Rust run does not confirm the effect is stable. Treat
the 6% as suggestive and reproducible-in-principle, not established.

**And it is bought at a price on this model.** `fastokens` falls back to HuggingFace
tokenizers for Gemma 4, and multimodal support is disabled outright. A 6% median-TPOT
improvement does not pay for losing vision on a vision model.

#### Verdict for this rig

Do not switch. The throughput case is nil on bandwidth-bound hardware, the latency case is
small and unconfirmed, the tokenizer fast path does not support this model, and the frontend
silently disables a capability the checkpoint has. Revisit if `fastokens` gains the `Replace`
normalizer and `gemma4` is registered in the Rust multimodal spec table.

## Incidental finding: first boot from this AMI is ~18 minutes, not ~3

`CLAUDE.md` records "boot to SSM ~50–70 s, then engine init **177–184 s** on `g5g.xlarge`".
On a **freshly launched** instance that is not what happens, on either host tested today.

Both `i-037eb7384d0aa7c92` (`xlarge`) and `i-010083a8387e9198e` (`4xlarge`) sat at

```
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
```

for 15–20 minutes. The cause is **EBS snapshot lazy loading**, not the engine, not swap and
not the GPU. Measured on the `4xlarge` mid-load, with 26 GiB of free RAM and no swap
pressure:

```
sectors read in 10s: 197696  =>  9 MB/s
iowait 6%,  EngineCore in state Dl
```

**9 MB/s** against a gp3 volume's nominal 125 MB/s. Every block of the 9.54 GiB checkpoint
is being faulted in from S3 on first touch. 9.54 GiB at 9 MB/s is ~18 minutes, which is what
both hosts took.

The `xlarge` case was initially misread as swap thrash — it had 16 GiB of swap and 56–71%
iowait, which looks exactly like thrashing. The `4xlarge` control rules that out: 30 GiB of
RAM, no swapfile, same stall, same 9 MB/s.

So the 177–184 s figure is a **warm-volume** number. It is reachable on a second start, or
after `fio`-style volume initialisation, but it is not what a first launch from
`ami-0b44b90b3d02430ee` costs. Anyone using this AMI to "turn a multi-hour provision into
~4 minutes" should budget ~20 minutes for the first serve.

## Environment

| | |
| --- | --- |
| Instances | `g5g.xlarge` spot `us-east-1a` (`i-037eb7384d0aa7c92`, terminated); `g5g.4xlarge` on-demand `us-east-1a` (`i-010083a8387e9198e`) |
| AMI | `ami-0b44b90b3d02430ee` — `gpu-vllm-g5g-2b-sm75-vllm0272rc0-80g-v2` |
| GPU | NVIDIA T4G, compute capability 7.5, 15,360 MiB |
| vLLM | `0.27.2rc1.dev0+g7f7a32cfe.d20260812`, editable at `/opt/vllm-src` |
| Rust | rustc 1.97.1 (8bab26f4f 2026-07-14), cargo 1.97.1 |
| setuptools-rust | 1.13.0 |
| protoc | absent as shipped; libprotoc 3.21.12 installed for the build |

`g5g.4xlarge` was on-demand rather than spot because G5g spot capacity was exhausted across
all four `us-east-1` AZs at every size on 2026-08-14 — `InsufficientInstanceCapacity` for
`xlarge`/`2xlarge`/`4xlarge`/`8xlarge` in `1a`/`1b`/`1c`/`1d`, with only a single `xlarge`
obtainable earlier in the day.

## Caveats

Single build per host, no repeats. Build wall time on the `xlarge` was measured while that
instance had 16 GiB of swap and had just been serving; treat it as indicative. The 501-crate
count is `Compiling` lines from one log and includes both targets.

## Still open

- Whether `_rust_tool_parser` being 96 MB of debug code has any runtime cost, or is purely
  disk. Not measured.
- Whether `VLLM_REQUIRE_RUST_FRONTEND=1` turns the silent skip into a hard failure as the
  name implies. Not tested.
