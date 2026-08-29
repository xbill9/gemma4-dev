# GEMINI.md — `gce-jaxrust-v6e1-2b`

Mirror of `CLAUDE.md`, kept in step by hand — there is no generator, and these three files
have drifted before. **`CLAUDE.md` is authoritative where they disagree.**

Guidance for Claude Code when working in this repository.

## Project Overview

This rig packages one Claude Code skill (`gce-jaxrust-v6e1-2b-management`) and one
**Model Context Protocol (MCP) server** (`gce-jaxrust-v6e1-2b`, a FastMCP app in
`server.py`). Together they:

1. **Operate TPU infrastructure:** find, probe, provision, and destroy Google Cloud TPU
   capacity — **through Compute Engine only**. See "Control plane" below; there is no
   queued-resource path in this rig.
2. **Serve Gemma 4 from Rust — the default path:** provision flex-start VMs with
   `workload="jaxrust"`. The startup script installs a pinned Rust toolchain, the build
   dependencies `pjrt-sys` needs, and libtpu; `deploy_jaxrust_engine` builds `rust/` on
   the VM and runs the probe; `manage_jaxrust_server` starts the OpenAI-compatible
   server. No docker, no HF token, **and no Python in the serving path**.
3. **Serve with the Python JAX engine — the parity oracle:** `workload="jax"` installs
   `jax[tpu]` and `jax_openai_server.py` serves the QAT checkpoint. This is not a legacy
   path being kept around out of sentiment: it is the only implementation here whose
   numbers have been checked against the reference module, so it is what the Rust engine
   gets differenced against.
4. **Serve with vLLM — the alternative path:** `workload="vllm"` auto-starts the vLLM TPU
   docker image on a multi-chip shape. Requires an `hf-token` Secret Manager secret.
5. **Do SRE diagnostics:** verify model health, fetch engine/system/Cloud Logging logs,
   and use the self-hosted Gemma 4 model to triage them (`analyze_cloud_logging`).

**Forked from `tpu-jax-v6e1-2b` on 2026-08-28** to move the engine off CPython onto Rust,
in parallel with `gpu-jaxrust-g5g-2b` doing the same on Turing. **This rig has provisioned
nothing and measured nothing on the Rust path.** Everything below about the Rust engine is
"it compiles and its API contracts are real"; nothing is "it ran on a chip."

## What is actually proven, and what is not

The distinction matters more here than in a settled rig, so it is at the top rather than
buried in a benchmarks README.

**MEASURED on a real v6e-1 (`ct6e-standard-1t`, `europe-west4-a`, spot, 2026-08-28):**

- **Rust executes on the TPU.** `tpu-selftest` built a 256×256 f32 matmul in rlx's IR,
  compiled it for `Device::Tpu`, ran it through libtpu's PJRT plugin and got 256.0 in all
  65,536 elements. The full lifecycle ran through this rig's own tools —
  `create_tpu_vm_instance(workload="jaxrust")` → `wait_for_jaxrust_ready` →
  `deploy_jaxrust_engine` → `verify_rust_tpu`.
- **It is not a silent CPU fallback**, checked three ways rather than trusting the device
  name: `/tmp/tpu_logs` grew on the TPU run and not on the CPU run; libtpu logged
  `XLA::TPU running hlo passes for 6 instructions, modules: selftest`; and a second
  process was refused with `The TPU is already in use by process with pid …`.
- **The `gemma4-engine` binary builds on the VM** with `--features tpu,gemma`, linking
  `rlx-gemma`.

**Still not verified:**

- **That Gemma 4 E2B produces correct output on TPU.** No checkpoint has been loaded. What
  is proven is that rlx's TPU backend compiles and executes *a* graph correctly — not that
  it does so for E2B's PLE, alternating local/global attention and KV sharing. Do not
  promote the selftest into a claim about the model.
- Any throughput, latency or HBM figure. There are none.

**Three defects found by running it, all now fixed or recorded:**

1. **`xla-probe` cannot create a PJRT client at all** — see "The Rust engine" below. It is
   no longer what the rig gates on.
2. **rlx-tpu SIGSEGVs on teardown.** rlx-tpu 0.2.11 against libtpu 0.75 computes correctly,
   prints every success line, and then dies with 139 while dropping the compiled graph.
   stderr is empty and there is no Rust panic. Leaving via `std::process::exit`, which
   skips destructors, exits 0 on the identical binary. **A marker-scanning supervisor would
   have called that healthy** — which is why `verify_rust_tpu` asserts on the exit code and
   never on the marker. `SELFTEST_RUN_DROP=1` reproduces the crash against a newer stack.
3. **`libopenblas-dev` was missing from the startup script**, and it fails later than
   `protoc` does: rlx-cpu links `-lopenblas`, `rlx-runtime/tpu` pulls `cpu` in
   transitively, and the entire ~150-crate workspace compiles before the final link of the
   engine fails. `xla-probe` linked fine because it has no rlx in it, which is the only
   reason the cause was legible.

## Control plane: Compute Engine, and only Compute Engine

This rig provisions with `gcloud compute instances create --machine-type=ct6e-standard-1t`.
It carries **no queued-resource tools at all** — `create_tpu_queued_resource`,
`manage_queued_resource`, `destroy_queued_resource`, `list_queued_resources`,
`describe_queued_resource` and `find_tpu` were removed, and `find_tpu_vm` is now the only
capacity sweep. The Cloud TPU API is no longer under active development and TPU7x and
later are Compute Engine or GKE only, so the old path has a shelf life.

**The naming caveat is resolved, in the other direction.** The parent rig kept the name
`tpu-jax-v6e1-2b` by an explicit decision on 2026-08-18 even though its slot 1 was wrong
— it had already migrated to Compute Engine. This fork takes the name the spec asks for:
`gce` because Compute Engine provisions it, `jaxrust` because the forward pass is defined
in Rust and lowered to XLA rather than written in Python. Both slots are now claims the
code actually backs. See `NAMING.md` slots 1 and 2; `jaxrust` is one slot value and is
never spelled `jax-rust`.

The flag mapping, the four provisioning models, the two quota metrics and the failure
modes that do not announce themselves all live in the skill —
`.claude/skills/gce-jaxrust-v6e1-2b-management/SKILL.md`. **Read it before touching
provisioning logic.** The three that cost the most time:

- **RUNNING is not ready.** An instance is RUNNING the moment the VM boots, before the
  startup script has done anything, and a dead boot reports RUNNING forever. Only the
  serial log or the port tells you the difference.
- **PENDING is either no quota or no capacity**, and a create never says which.
  `probe_zone_capacity` fires a throwaway SPOT create — spot does not queue, so it fails
  fast and names the reason — and settles it in seconds.
- **Flex-start spends PREEMPTIBLE quota**, falling back to the family quota, and the two
  metrics carry opposite defaults (family absent = 0, preemptible absent = 1536).
  `gcloud compute regions describe` shows neither; it only carries the v5-era metrics.

**v5e cannot be reached from here.** `ct5lp-*` machine types exist in the catalog in 26
zones and a create is refused with `This user agent is not allowed to use the machine
type`, which is neither a quota nor a does-not-exist error. `_gce_machine_type` omits them
and `_unsupported_accelerator_message` explains the refusal, because a generic
"unsupported" invites a pointless retry in another zone.

## The Rust engine

`rust/` is a two-crate workspace. `rust/README.md` has the layering diagram and the
runtime surprises; `rust/NOTICE.md` has the licensing, which is not optional reading.

| Crate | What it is | Licenses reached |
| --- | --- | --- |
| `xla-probe` | Loads the PJRT plugin, lists devices with HBM stats, compiles a StableHLO matmul, runs it, checks the numbers | permissive only |
| `gemma4-engine` | OpenAI-compatible server: rlx's JAX-shaped IR → HLO → PJRT → chip | permissive, **plus GPL-3.0-only with `--features gemma`** |

**`rlx-gemma` is GPL-3.0-only and this rig is Apache-2.0.** Linking it makes a built
binary a GPLv3 combined work. **Reviewed and kept on 2026-08-28**: the default is
`JAXRUST_CARGO_FEATURES=tpu,gemma`, because GPL obligations attach to *conveying a binary*
and this rig conveys none — the source is uploaded to the VM and built there, for the
operator's own use. The condition to watch is therefore a single one: **if a compiled
engine ever ships** — a container image, a `dist/` artifact, a VM image handed to someone
else — that binary is GPLv3 and its complete corresponding source has to go with it.

Note what this is *not* about. The **checkpoint's** license is irrelevant here either way:
GPL attaches to code that is linked, not to data a program loads at runtime, so the weights
neither pick up the binary's license nor change it. The GPL code is `rlx-gemma` itself —
the Rust implementation of the architecture from `MIT-RLX/rlx-models`.

Drop `gemma` and the server still starts, still reports the device, and refuses generation
with a message naming the flag; `xla-probe` never links it at all.

**Four things about this workspace that will cost time if rediscovered:**

- **`protoc` and `clang` are mandatory build dependencies.** `pjrt-sys` runs `prost-build`
  and `bindgen` in its build script and dies with ``Could not find `protoc` `` minutes into
  a cargo build, nowhere near the flag that caused it. The startup script installs both and
  `make rust` refuses to start without them.
- **The model crates trail the framework.** `rlx` publishes 0.2.14, but `rlx-gemma`
  0.2.11 pins `rlx-runtime =0.2.11` exactly, so the workspace is held at 0.2.11 throughout.
  Bumping `rlx` alone does not resolve — cargo reports it as a version-selection conflict.
- **The runner is `!Send`.** PJRT's `Client` is `Rc`-based, so the compiled graph and
  everything holding it are single-threaded. `engine.rs` runs the model on one dedicated
  thread and queues requests over a channel. On one v6e-1 chip that is the shape of the
  hardware, not a workaround.
- **`rlx-gemma`'s crates.io description is stale.** It says "Gemma / Gemma 2"; the source
  carries `gemma_e2b.rs`, `gemma4_e2b_mm.rs`, `gemma4_vision.rs`, `gemma4_audio.rs` and a
  QAT loader, and its module docs say Gemma 4 needs no separate code path. Do not conclude
  from the registry blurb that the model is unsupported.

## Repository Layout & the Snapshot Sync Model

The repo-root files are authoritative; the skill is distributed as generated copies.
**Edit the sources, then run `make skill` — never edit a snapshot directly.**

- **Sources (root):** `server.py` (the MCP server), `rust/` (the engine), `project-setup.sh`,
  `requirements.txt`, `tpu.md` (TPU getting started guide; gitignored, private). The guide's
  prose is kept vendor-neutral — no Google branding or TPU Builders Program references; only
  functional `gcloud` commands, API endpoints, and doc/console URLs may contain "google".
- **Hand-maintained (not regenerated):**
  `.claude/skills/gce-jaxrust-v6e1-2b-management/SKILL.md` and the four startup templates
  beside it — `startup_script_jaxrust_template.sh`, `startup_script_jax_template.sh`,
  `startup_script_template.sh`, `startup_script_cpu_template.sh`.
- **Generated by `make skill` (`refresh_skill.py`):** the `mcp/{server.py,project-setup.sh,
  requirements.txt}` copies, `references/tpu-guide.md`, and the whole
  `skills/gce-jaxrust-v6e1-2b-management/` plugin copy.
  **`refresh_skill.py` does not touch `.agents/skills/`** — that third snapshot exists and
  has to be synced by hand.
- **Generated, gitignored:** `.mcp.json` (embeds the GCP project id), `rust/target/`.
  `rust/Cargo.lock` **is** committed: these are binaries, and the rlx pin above is exactly
  the kind of thing that should not float.
- **Plugin marketplace:** the marketplace `/plugin` reads is the **monorepo root**
  `../.claude-plugin/marketplace.json`; the copy here only matters if this rig is published
  standalone. Keep both in sync. Validate with `claude plugin validate .`

## Common Commands

```bash
make skill         # Refresh skill snapshots + plugin copy from the root sources
make skill-install # Refresh + copy the skill to ~/.claude/skills (all projects)
make skill-package # Refresh + rebuild dist/gce-jaxrust-v6e1-2b-management-skill.zip
make test          # 149 unittest tests, offline
make lint          # ruff on the Python sources + bash -n on the shell scripts
make rust          # cargo build --release, both crates
make rust-lint     # cargo fmt --check + clippy -D warnings
claude plugin validate .
```

`make rust` is deliberately **not** wired into `make test` / `make lint`: the root Makefile
fans those out across every rig, and a cold build of the rlx workspace is minutes.

The server's config comes from env vars: `GOOGLE_CLOUD_PROJECT`, `MODEL_NAME`,
`ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`, `INSTANCE_NAME`, `PROVISIONING_MODEL`,
`REQUEST_VALID_FOR` / `MAX_RUN_DURATION`, `BOOT_DISK_SIZE_GB`, `IMAGE_FAMILY` /
`IMAGE_PROJECT`, `GCE_QUOTA_ID` / `GCE_SPOT_QUOTA_ID` / `GCE_TPU_FAMILY`, the Rust knobs
`RUST_TOOLCHAIN` / `LIBTPU_SPEC` / `JAXRUST_CARGO_FEATURES` / `JAXRUST_REMOTE_DIR`, and the
Python-path knobs `JAX_PYTHON_VERSION` / `JAX_PIP_SPEC` / `JAX_PIP_EXTRAS`. `get_help`
prints the live values.

**Default deployment target: a single-chip v6e-1 flex-start VM on Compute Engine** —
`ACCELERATOR_TYPE=v6e-1` (`ct6e-standard-1t`, 32 GB HBM), `TENSOR_PARALLEL_SIZE=1`,
`GOOGLE_CLOUD_ZONE=europe-west4-a`, `MODEL_NAME=google/gemma-4-E2B-it`,
`PROVISIONING_MODEL=flex-start`. `ACCELERATOR_TYPE` is **documentation on this path**: it
is the Cloud TPU API's spelling, kept so the rig name and the benchmark reports line up,
and `gcloud compute instances create` would reject it outright. What Compute Engine
consumes is the machine type derived from it.

## The standard lifecycle on the Rust path

```
create_tpu_vm_instance(workload="jaxrust")   # RUNNING is not ready
wait_for_jaxrust_ready                       # Rust + libtpu installed. Nothing built yet.
deploy_jaxrust_engine                        # uploads rust/, builds release, RUNS THE PROBE
verify_rust_tpu                              # re-runs the probe on demand
manage_jaxrust_server(action="start", model_path=...)
```

Each step answers a strictly smaller question than the next one down, which is the point:
when serving fails, the step that still passes tells you where to look. `deploy_jaxrust_engine`
running the probe as its last act is deliberate — a build that succeeds proves the toolchain
works and says nothing about the chip.

**The probe asserts on a computed value, not on a device list.** `dlopen` on `libtpu.so`
succeeds on a host with no chip attached, exactly as `import jax` succeeds with no TPU
backend. `verify_jax_tpu` handled that by asserting on `jax.devices()`; `xla-probe` goes
further and checks the numbers coming out of a StableHLO matmul, because this repo's
standing rule is that a thing being accepted is not evidence it did anything.

## Fork debris — not yet cleared

Inherited from `tpu-jax-v6e1-2b` and **still describing that rig, not this one**. Left in
place rather than rewritten, because rewriting them would falsify them:

- `benchmarks/runs/**` and `benchmarks/reports/*.json` — four runs and one report, measured
  on a v6e-1 through the **Cloud TPU API** on the **Python JAX** engine. They are
  byte-identical to the copies still in `tpu-jax-v6e1-2b`. `benchmarks/rollup.py` globs
  `*/benchmarks/` and will attribute them to this rig, which is exactly the failure the v5p
  rig avoided by deleting its inherited artifacts. **These should be deleted here**; that
  deletion was blocked in the session that set this rig up and is left as an explicit
  follow-up rather than done silently.
- `devto-jax-gemma4-e2b.md`, `devto-gemma4-qat-jax-v6e1.md` — articles about the Python JAX
  rig on v6e-1. Restored verbatim after an over-broad rename touched them.
- `huggingface/gemma4-e2b-tpu-v6e-benchmarks/README.md` — a dataset card for the same
  measurements.
- `benchmarks/queued/kernel_gap_suite.py` — a JAX profiling harness, tied to `jax_engine.py`.

Everything else in the tree has been retargeted: `server.py`, the four registration points,
the skill (all three snapshots), the plugin manifests, the Makefile and the tests.

## Coding Standards

- **Python: no virtualenvs.** Use the system `python3`; if dependencies are missing, warn
  with the `pip install -r requirements.txt` command instead of creating a venv.
- **Every subprocess call goes through `run_command(cmd: list[str])`** using
  `asyncio.create_subprocess_exec`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown strings with emoji status prefixes.
- Existing code uses `Optional[str]`, not `X | None` — match the surrounding file.
- Don't assume `pandas` is installed; prefer stdlib `csv`/`json`.
- **Startup templates go through `str.format()`**, so a stray literal `{` or `}` raises at
  render time — i.e. at VM-creation time, not in CI. The templates avoid braces entirely
  rather than escaping them, and `TemplateBraceHygieneTests` enforces it.
- **Rust: `cargo fmt` and `clippy -D warnings` are the bar**, and both are clean today.

## Related Documentation

- `rust/README.md` — the engine: layering, build, the four runtime surprises
- `rust/NOTICE.md` — third-party licenses; read before building a binary with `gemma`
- `.claude/skills/gce-jaxrust-v6e1-2b-management/SKILL.md` — lifecycle, tool catalog, field
  notes, cautions (read before touching provisioning logic)
- `references/tpu-guide.md` (inside the skill) — flex-start zones per TPU family, quota
  metrics and request procedure, troubleshooting
- `docs/gemma4-quirks.md` — the architecture and the serving path, verified against the
  reference module. Part II includes two open wrong-output bugs the Python test suite
  passes straight through. **Still the best description of what the model does**, whichever
  language implements it.

## The parity oracle, and why the Python engine stays

`jax_openai_server.py`, `jax_engine.py`, `ports/gemma4/` and `tests/` are kept, and
`workload="jax"` still provisions a VM for them. Nearly every test in this repo is a parity
assertion between two of our own code paths, so an assumption both paths share is invisible
to all of them — and that was already true when both paths were Python. With a Rust engine
alongside, the two implementations no longer share a language, a graph builder or an author,
which makes a disagreement between them worth considerably more than a disagreement between
two JAX code paths was.

Build with `--no-default-features --features cpu,gemma` and compare on a workstation before
spending a flex-start window on it. The arithmetic steps are free; the ones that cost a
capacity cycle come last (`RIG-ANALYSIS.md`).

**Measurement rule earned the hard way:** a config flag being accepted is not evidence it
did anything, and an A/B can be internally valid while its baseline is wrong. Cross-check
against an absolute physical bound — bytes moved per second against calibrated bandwidth —
not just against another configuration. See
`benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md` for the corrections that rule produced
(a Python-path artifact; see "Fork debris" above).

## Rust, XLA and JAX references

- **[`pjrt` crate docs](https://docs.rs/pjrt)** — the PJRT C API bindings `xla-probe` uses.
  `pjrt::plugin(path).load()`, `Client::builder(&api).build()`, `Program::new(MLIR, code)`,
  `LoadedExecutable::builder(&client, &program).build()`. The README example still shows an
  older `load_plugin` free function; the builder is what 0.2.0 exports.
- **[rlx](https://github.com/MIT-RLX/rlx)** — the JAX-shaped IR, its HIR→MIR→LIR pipeline,
  and the TPU backend that lowers to HLO and drives libtpu.
- **[`docs/gemma4-quirks.md`](docs/gemma4-quirks.md)** — read before touching the model
  path in either language.
- **[How to think in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)** —
  still the right mental model, and its lessons transfer: arrays are immutable (which is why
  a KV-cache write rebuilds the cache without buffer donation), static vs traced values, and
  why dynamic shapes will not compile — hence the statically shaped, bucket-padded caches
  and masks, and hence `--max-model-len` being a compile-time commitment here too.
- **[Gemma 4 QAT checkpoints](https://ai.google.dev/gemma/docs/core#qat)** — the authority
  on which checkpoint variant to load. The suffixes are not interchangeable: `-w4a16-ct`
  is what the Python engine loads by default, `-q4_0-unquantized` ships half-precision QAT
  weights, and `-gguf` / `-mobile-ct` target other runtimes. rlx reads safetensors and GGUF
  both, so the constraint on the Rust path is different from the Python one — do not carry
  `deploy.md`'s checkpoint conclusions across without rechecking them.
