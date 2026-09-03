# CLAUDE.md — `local-jax-cpu-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **pure JAX on the local CPU**. No accelerator,
no cloud, no provisioning.

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`. It
is not one of the artifact rigs.

"Pure JAX" is literal: no PyTorch, no torch_xla, no vLLM. The engine is this monorepo's own
Gemma 4 port (`ports/gemma4/`) driven by `jax_engine.py` behind an OpenAI-compatible FastAPI
server (`jax_openai_server.py`), run as an **ordinary background process owned by the invoking
user** — not systemd, not docker. Read logs with `get_jax_logs`, which tails one file.

## Read this first: the provenance rule

**Forked from `gpu-jax-g4dn-2b` on 2026-08-29, which was forked from `gpu-jax-g5g-2b` a day
earlier. Every throughput number in the inherited prose was measured on a Turing GPU.** The
figures you will see quoted — 12.4, 12.8, 13.10 tok/s, 54% dtype conversion, 0.0% TensorCore —
are `gpu-jax-g5g-2b`'s, on a T4G. **None of them is this rig's**, and differencing one against
a number from here does not give you a CPU-vs-GPU result until this rig has recorded a run of
its own.

What DOES carry, because it describes the checkpoint or the engine rather than a chip: the
weight-footprint table, the KV arithmetic, the padding-eviction bug and its fix, the quantizer
memory bugs, and the observability machinery. Those are marked as inherited where they appear.

**Facts measured on THIS host on 2026-08-29** are the swap incident, the host inventory, the
anonymous-checkpoint verification, and the XLA:CPU trace lane names. Each says so.

## Why this rig exists

**It is the zero point.** Every other rig here measures an accelerator, and none of them can
separate the chip from the engine — there is no run without a chip. This is the only place a
number for this JAX port is attributable entirely to the model code and the XLA CPU backend.

It is also the only rig with **no hourly cost and no capacity risk**. That is not just
convenient; it changes what the rig is *for*. `docs/padding-window-eviction.md` describes a
silent correctness bug in the shared port that was found on a GPU and **verified on CPU**,
because nothing in the mechanism is hardware-specific. That verification could have happened
here first, for free, and the next one should.

## There is no control plane, and the deletion is the design

`server.py` lost the EC2 launch, AMI resolution, spot handling, SSM Run Command, Secrets
Manager, the S3 compilation-cache sync, the systemd unit, cloud-init, and `boto3`. There is no
`deploy_jax_server`: the payload runs in place.

`tests/test_server.py::NoCloudControlPlaneTests` asserts the **absence** of that vocabulary,
and it is the most load-bearing class in the suite. A fork of a cloud rig keeps passing its own
tests while describing hardware that does not exist — that is exactly how `gpu-jax-g4dn-2b`
shipped with four registration files naming `gpu-jax-g5g-2b` and a skill path that was never
there. Dead cloud code here is worse than unused: it reads as live configuration.

Those tests strip comments and docstrings before searching (`code_only()` in the test module),
and the reason is worth keeping: the most valuable prose in this rig is precisely the prose
that **names** what was removed. A naive substring search fails on the module docstring
explaining there is no spot reclamation, and the obvious fix — deleting the explanation — makes
the codebase worse.

What replaced the control plane is the same *shape* of problem:

| Cloud rig | Here |
| --- | --- |
| `check_g4dn_quotas` | `check_host_capacity` |
| `verify_gpu_arch` | `verify_cpu_backend` |
| `create_*_instance` / `terminate_*` | `start_jax_server` / `stop_jax_server` |
| `get_install_progress` (cloud-init) | `check_dependencies` |
| `get_jax_logs` (systemd journal) | `get_jax_logs` (one logfile) |
| AMI + instance-type selection | `fetch_checkpoint` |

**Nothing is approval-gated in `.codex/config.toml`, deliberately.** On the cloud rigs the
gates are on the tools that cost money or destroy capacity. Nothing here does either, and a
prompt on an action with no consequence trains the operator to click through the ones that
have. `test_nothing_is_approval_gated` pins it so it is not "fixed" by analogy with a sibling.

## Memory is the binding constraint, and it does not raise

It will be slow. That is expected and it is not the interesting part. What stops a serve is
RAM, and the failure mode is **worse than a cloud rig's**: exceeding a quota is refused at the
API; exceeding host RAM is *accepted* and paid for in swap. A thrashing serve is
indistinguishable from a loading one.

That asymmetry is why `check_host_capacity` exists as a tool rather than a note in the README,
and why `start_jax_server` **refuses** rather than starting — the one place this rig takes a
decision away from the operator.

Weight footprints, INHERITED from `gpu-jax-g5g-2b`'s 2026-08-26 measurement. They are
properties of the checkpoint and the levers, not of the device, so they carry — and every
prediction in that run landed within 1%:

| Config | Weights | vs dense |
| --- | ---: | ---: |
| `ple0` (dense) | 9.257 GB | — |
| `ple8` | 6.927 GB | −2.330 |
| **`ple4` — the default here** | **5.752 GB** | **−3.505** |
| `ple0 + int8head` | 9.660 GB | +0.403 |
| `ple4 + int8head` | 6.155 GB | −3.102 |

Plus ~1.6 GB of prefill transient at a bucket below 4K — **flat** in that range, not per-token,
which is why lowering `MAX_MODEL_LEN` below 4096 buys latency rather than memory.

### `INT8_LM_HEAD` is OFF here, inverting every GPU sibling

This is the one default that is deliberately reversed, and the reasoning is the whole point of
the rig having its own config. `int8_lm_head` **adds 0.403 GB** — an int8 copy of the tied
embedding placed alongside the original — to buy +2.3% throughput by halving the bytes *read*
per decode step. That is a memory-bandwidth trade. It is the wrong trade on a host where the
constraint is resident bytes and the surplus goes to swap: you would pay 0.4 GB of paging to
save bandwidth you are not short of.

It is also not what its name suggests on **any** rig here — `jax_e_model.py` dequantizes the
int8 table to the compute dtype in full (0.75 GiB) on every decode step and then runs the same
matmul. Turn it on only after `check_host_capacity` reports real headroom.

`PLE_BITS=4` carries over unconditionally: a pure memory win, and decode is *identical* across
`ple0`/`ple8`/`ple4` on the parent because the per-layer embedding table is a gather, never a
matmul, so decode never streams it.

### Never size this rig's context from KV arithmetic

INHERITED reasoning, verified against `MODELS.md`. E2B has 35 layers but only 15 own a KV cache
(`first_kv_shared_layer_idx=15`); with `window_kv` on — it auto-resolves True whenever
`max_model_len > sliding_window`, so always here — the 12 sliding ones are capped at a 512-slot
ring and only 3 full-attention layers hold the full context. At 2048 tokens, float16, B=1, one
KV head, that is about **18 MiB**.

**The prefill transient is ~1.6 GB — roughly ninety times the entire KV cache.** Raising the
context does not trade against KV in any meaningful way.

## bfloat16 is forced by memory, not chosen for speed

**The device decides, not `tpu.env`.** `ports/gemma4/jax_e_model.py` reads the live device;
`DTYPE` is the override and `JAX_E_COMPUTE_DTYPE` is the escape hatch. On CPU there is no
compute capability to read, so `IS_PRE_AMPERE` is False and the port resolves `bfloat16`.

That happens to be the right answer, for a reason that has nothing to do with speed — and this
is where the rig's reasoning **inverts** a sibling's rather than extending it:

- **XLA:CPU has no bf16 datapath.** It upconverts to fp32 in front of every use — precisely
  the emulation that is 54% of decode time on the Turing siblings, and the reason they resolve
  `float16` instead.
- **float16 is not an escape here.** Turing has fp16 tensor cores to escape *into*. A CPU has
  no 16-bit float datapath of any kind, so fp16 storage is upconverted identically. The escape
  the GPU rigs have does not exist.
- **float32 storage would avoid the conversion and does not fit.** 18.5 GB of weights against
  a typical host's RAM. Arithmetic, not an attempt.

So the dtype tax is **unavoidable rather than a bug**, and what makes it unavoidable is memory.
`docs/bf16-weights-on-turing.md` is an investigation into removing it, with three placements
tried and measured. **Do not port its open work item here** — the option space is smaller, not
larger. `profile_decode.py` says so in as many words when conversion exceeds 15% of decode, so
the same ticket does not get opened twice.

## The one silent failure this rig has that the GPU siblings do not

**Pallas has no CPU backend, so the fused W4A16 kernel auto-enables INTERPRET MODE.**
`JAX_E_PALLAS_INTERPRET` defaults to `1` whenever the platform is neither TPU nor GPU.

On the GPU siblings that kernel is *refused at startup* by
`check_w4a16_fits_scoped_memory()`, with the arithmetic attached, because Turing's 64 KiB
shared-memory ceiling cannot hold its 550 KiB – 1.1 MiB tiles. Here it **runs**, in a
simulator, producing correct numbers at a speed that means nothing.

A refusal is a better failure than a simulator. The only warning is the startup banner:

```
jax_e_model device policy: platform=cpu compute_capability=n/a
compute_dtype=bfloat16 pallas_interpret=True
```

Read it before believing any w4a16 number from here. That banner is emitted at **import** of
`jax_e_model`, which is imported via `jax_engine`, so `logging.basicConfig(force=True)` must
precede that import in `jax_openai_server.py` or the line is dropped —
`test_root_logging_is_configured_before_the_engine_import` pins the ordering, and it matters
more on this rig than on the one it was written for.

## Swap, and why it took two goes

**MEASURED HERE 2026-08-29.** The development host had **no swap** and 3.3 GB of `MemAvailable`
against 5.75 GB of weights, so a serve could not start at all.

`fallocate` + `mkswap` + `swapon` — the remedy the parent rig renders into its cloud-init —
**fails on this host**:

```
swapon: /swapfile: swapon failed: Invalid argument
```

The actual reason reaches only `dmesg`:

```
BTRFS warning (device vdb): swapfile must not be copy-on-write
```

The root filesystem is btrfs, where a `fallocate`d file is copy-on-write and the kernel will
not host a swapfile on one. The remedy is btrfs-progs' own helper, which sets NOCOW and
disables compression on the file it creates:

```bash
sudo btrfs filesystem mkswapfile --size 16G /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Worth keeping for the same reason the parent kept its `mkswap -q` incident: **the error names
the syscall and not the cause**, so the failure reads as a broken script rather than as a
filesystem that will not host that file. `check_host_capacity` prints this remedy when it finds
no swap configured.

## The host is not part of the rig

Unlike every sibling, whose hardware slot names a SKU the rig provisions, **`cpu` names
whatever machine the rig is checked out on.** Two runs of this rig on two machines are not
comparable, and the rig name cannot carry the difference.

So a benchmark report from here **must record the host in its own fields**. `tune_loop.py`
writes the CPU model, core count, RAM and swap into every `summary.json` for exactly this
reason, and `test_the_hardware_slot_is_honest` asserts `tpu.env` says so out loud.

The host this rig was built on, MEASURED 2026-08-29:

| | |
| --- | --- |
| CPU | 13th Gen Intel Core i7-1360P, 16 logical cores |
| RAM | 14.3 GiB (`MemTotal` 14,682,148 kB) |
| Swap | none originally; 16 GiB added, see above |
| Root fs | btrfs on `/dev/vdb` |

`_host_facts()` reads **`MemAvailable`, never `MemFree`** — free memory on a warm machine is
near zero because the page cache holds the rest, and quoting `MemFree` is how you conclude a
15 GB host has 0.5 GB. `test_available_not_free_is_what_is_read` pins it.

## Profiling works here, in a different lane

**MEASURED HERE 2026-08-29** by tracing a jitted matmul and reading the lane names back.

There is no device compute stream in an XLA:CPU trace, so `profile_decode.py`'s original GPU
lane filter (`"Stream" in lane and "Compute" in lane`) matches **nothing** — the unadapted
script would have reported "no events found" and exited 1. A profiler that silently declines to
profile.

XLA:CPU does still emit per-op events, in a lane named `tf_XLAPjRtCpuClient/<id>`, and with the
**same op names the GPU traces carry** — `wrapped_convert` appears by name. So the analysis
carries and only the lane does not. `device_ops()` tries the GPU stream first and falls back,
and reports which it used, because a table you cannot attribute to a lane is not evidence.

Three traps came out of that check, all handled:

- **`tf_XLAEigen/<id>` worker lanes hold thread-pool bookkeeping, not ops.**
- **Every op also emits an `end: <name>` marker**, which double-counts if kept.
- **`SlinkyThreadPool::Await` is a WAIT and is frequently the largest single entry.** Counting
  it as compute inverts whatever conclusion you were about to draw.

The CPU lane total is **wall time inside the executable, not core-seconds** — it does not
decompose into per-core work and must not be compared against a GPU rig's device-lane total.
The script says so when it uses that lane.

**xprof's structured rollups are UNVERIFIED here.** Every xprof finding in the inherited prose
came off a CUDA trace, and `kernel_stats` has no obvious analogue in an XLA:CPU xplane. Prefer
`profile_decode.py`'s own table until someone checks. `summarize_xprof()` in `tune_loop.py`
will return `{}` rather than lie.

**`profile_prefill.py` is the tool that gains the most from running here**, and the reason is
that it never needed a device: `memory_analysis()` reports buffer assignment for a shape that
cannot run, so the whole analysis is an ahead-of-time compile. A prefill sizing question about
a GPU rig can be answered from this rig for free — as long as you remember the *answer* is the
transient in bytes, a property of the graph, and not the verdict about whether it fits, which
is a property of the device you did not use.

## The compilation cache is not ephemeral here

On every cloud sibling the XLA cache lives on a root volume that dies with the instance, which
is why they carry an S3 sync, a systemd timer and an operator-supplied bucket URI. **All of
that is deleted rather than ported** — there is nothing to reclaim.

It also fixes by construction the bug written up in the parent's `CLAUDE.md`:
`ports/gemma4/jax_e_model.py` sets `jax_compilation_cache_dir` **unconditionally at import**,
and `jax_openai_server.py` resolves the same variable at line 72 and then imports the port at
line 80 — so the later import wins and the configured directory stayed empty forever while
`~/.cache/jax_compilation_cache` filled up (MEASURED on G5g 2026-08-27: 447 files in the wrong
place, 0 in the right one). **This rig's default IS the port's fallback path**, so the two
agree and the knob is honest. `test_both_modules_resolve_the_cache_dir_the_same_way` stays,
because "they agree today" is not "they cannot disagree".

`server.py` calls `expanduser` on the **resolved** value, not just the default: `tpu.env`
spells it `~/.cache/...` and dotenv does not expand a tilde the way a shell would, so a literal
`~` would otherwise become a directory of that name in the working directory.

## The build id still means something, and it means something different

There is no deploy, so a build id cannot be *stale* in the parent's sense. It answers a
different question that is just as real: **which copy of the payload is this process running?**

`_payload_root()` resolves the skill snapshot at `.claude/skills/<skill>/mcp/` **before** the
rig root, and the MCP server itself runs from that snapshot. So an MCP-launched serve runs the
snapshot's payload, not the working tree you are editing. `verify_model_health` compares the
served id against this tree's digest and reports **`DIFFERENT PAYLOAD`**, with the remedy.

**Always `make skill` before starting a serve you intend to measure.** `tune_loop.py` does it
for you and then asserts the two ids agree.

The parent's other stale-code hazard is *worse* here, not better: an already-running process is
serving the code as it was when it started, and editing the model port does not disturb it.
There is no deploy step to remind you. That is why `tune_loop.py` restarts by default.

## Engineering rules

- Every subprocess call takes a **list**, via `asyncio.create_subprocess_exec`. **Never
  `shell=True`** — `test_serve_argv_is_a_list_never_a_shell_string` checks the code, not the
  comments.
- **Binds loopback by default.** Every cloud sibling binds `0.0.0.0` because reaching the box
  from elsewhere is the point there. This server has **no authentication of any kind** and this
  host is not behind a cloud security group, so `JAX_HOST=0.0.0.0` puts unauthenticated
  inference on the network. Set it deliberately or not at all.
- **The pid is verified against `/proc/<pid>/cmdline`** before it is believed. A recycled pid is
  how a process manager calls a dead service healthy, and pid reuse on a developer box is fast.
- **`start_new_session=True`**, or a Ctrl-C delivered to the agent's process group takes the
  model down after a ten-minute load.
- **Stopping is cheap.** Nothing was built and the warm compilations are on disk. Do not import
  the EC2 rigs' "weigh stop against terminate" reasoning.
- **Never hardcode an endpoint.** `_endpoint_base()` also normalises a `0.0.0.0` bind to a
  loopback dial — a bind address is not a destination, and that is the one way a local endpoint
  actually goes wrong.
- **`verify_model_health` uses `/v1/chat/completions`**, because raw `/v1/completions` skips the
  chat template and is unreliable on `-it` models. And it does **not** pass on "the reply was
  non-empty" — it reads `tpu_jax_degenerate_responses_total` either side of its own probe, so
  the verdict is the server's judgement of the full text. A token loop is reported as
  `status="success"`.
- **The HF token, if one is needed, goes to `~/.cache/huggingface/token` at mode 0600.**
  `google/gemma-4-E2B-it` is readable **anonymously** — VERIFIED HERE 2026-08-29 — so most
  operators need none.

## Metrics keep the `tpu_jax_` prefix on a rig with no TPU

Deliberate. Both existing benchmark reports compare on `tpu_jax_decode_tokens_per_second` **by
name**, and the `rig` label is what separates rigs; renaming the prefix would break continuity
with them for no gain. `RIG_NAME` reaches the serving process through `_serve_env()`.

**`tpu_jax_hbm_used_bytes` is ABSENT here rather than zero.** On an accelerator that series
reading 0 means "the allocator holds nothing", which is a symptom; here it would mean "there is
no allocator", which is not. An absent series is the honest rendering of a question the
platform cannot answer, and it is also what keeps `tpu_jax_host_rss_bytes` from looking like a
redundant duplicate. `memory_stats()` reports `has_device_allocator` so the exporter can tell
the two apart — and it handles both shapes of "no": the accessor being absent (CPU) and the
accessor returning `None`.

**RSS is read from `/proc/self/statm`, not psutil.** A serving dependency that exists to read
one number is a dependency that can be missing at exactly the moment you are trying to find out
why memory ran out.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v` (123 tests,
all passing as of 2026-08-29). Fully offline — no network, no accelerator, no cloud account.

`make lint` runs `ruff check` over a **hardcoded file list**, then `bash -n` on **three** shell
scripts (`project-setup.sh`, `init.sh`, `set_env.sh` — `save-aws-creds.sh` is deleted).
`ports/` is excluded on purpose: ruff's UP006/UP045 would rewrite its `Dict`/`Optional`
annotations, which the monorepo `CLAUDE.md` forbids and which would drift it from the copy
`tpu-jax-v5e1-2b` shares.

**A new top-level module is silently unlinted until it is added to that list.**
`test_every_top_level_module_is_on_the_ruff_line` now checks the **ruff line specifically**,
not the whole Makefile — the old version searched the entire file, so `make-medium.py` passed
vacuously on the strength of appearing in the `medium` recipe while never being linted. It had
four real errors when first checked, on 2026-08-29.

`make skill` regenerates the snapshots under `.claude/skills/` and `skills/`. **Eight files are
generated**, not just the MCP control plane: `server.py`, `project-setup.sh`, both requirements
files, **and the whole serving payload** — because an installed copy under `~/.claude/skills`
still has to be able to start a serve.

`SKILL.md` sits in the same tree and is a hand-written **source**: `refresh_skill.py` will not
recreate it, so `rm -rf .claude/skills` destroys it permanently — which is what happened during
the parent's t4g→g5g rename. `test_skill_is_complete_in_both_copies` guards both copies and
also fails if any of the eight generated files is stale.

`make serve` runs the server in the foreground, and **shells out to `_serve_argv()`** rather
than copying the argv, because a documented command that drifts from the tool it documents is
how a manual reproduction fails to reproduce. The parent's `get_deployment_config` printed
`VolumeSize=200` while its launch tool created 100. `test_the_documented_command_is_the_command_that_runs`
and `test_make_serve_shells_out_rather_than_copying_the_argv` pin both halves.

There is no `make deploy` recipe. The target exists and says there is nothing to deploy.

## MCP registration lives in four places

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and
`.claude/settings.local.json`'s `enabledMcpjsonServers`. All four must name the server
`local-jax-cpu-2b`, which prefixes every tool as `mcp__local-jax-cpu-2b__…`.

**Only `.mcp.json` is generated.** `project-setup.sh` writes it (`--server-name` sets both the
registered key and what the server advertises) and does not touch
`.claude/settings.local.json`. Those two are gitignored; the other two are committed.

`test_registration_files_all_name_this_rig` and `test_the_registered_entry_point_exists` cover
all four — the second because the parent fork's actual failure was a **path**, not a name:
`plugin.json` pointed at a skill directory that did not exist, and the rig was *unregisterable*
rather than merely misregistered. `server.py` was right throughout, which is why the breakage
was invisible to its tests.

`SKILL_STEM` in `project-setup.sh` is **derived** from the rig directory, never a literal. A
literal is what silently survives a rename.

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree**, and a convention change has to be applied
to all three by hand.

## Naming

`local-jax-cpu-2b` — platform `local`, runtime `jax`, hardware `cpu`, model `2b`.

**`local` was added to `NAMING.md` on 2026-08-29** as a sixth platform value. The spec's own
rule is that a value is earned by a genuinely different execution target, and this is the one
target with **no control plane at all**. `gce` was the alternative — the spec already said it
covers CPU-only rigs — and it was rejected because it would claim a Compute Engine control
plane this rig does not have, which is the exact class of error the platform slot exists to
prevent.

The bare four-slot name is a positive claim: this is the **dense reference checkpoint**.
`PLE_BITS=4` is a runtime lever, not a weight encoding, so it earns no fifth slot — the same
reason `kv_cache_dtype` does not.

**`tpu.env` is still called `tpu.env`**, on a rig with no TPU, like every other rig here. It is
the name the loaders, Makefiles and tests agree on, and renaming it would buy accuracy in one
filename at the cost of the one thing that makes a monorepo of near-identical rigs navigable.

## Measurement

**This rig has measured nothing yet.** `benchmarks/runs/` is empty on purpose and
`benchmarks/reports/` with it.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them, so edit the root originals.

Run directories take the `-cpu` suffix, matching the hardware slot: `benchmarks/runs/<date>-<what>-cpu/`.

Three numbers you will be tempted to reuse, and must not:

- **13.10 tok/s** — `gpu-jax-g5g-2b` on a T4G, 2026-08-26. Same engine, same checkpoint,
  different silicon by a wide margin.
- **43.1 / 44.24 tok/s** — the vLLM sibling on the same T4G silicon, obtained *with reduced
  Triton tiles*.
  **CORRECTED 2026-08-30 — neither figure is a benchmark, do not compare against either.**
  `43.1` is one sample from the 2026-08-12 first-serve run, whose own report says "single-run,
  single-stream, no repeats and no variance figure", taken with a 19-token prompt. `44.24` has
  **no benchmark artifact anywhere in the tree** — it survives only in `gpu-vllm-g5g-2b/server.py`'s
  swap comment and `tests/test_server.py`, where it was measured 2026-08-13 to show that
  `g5g.xlarge` + a 16 GiB swapfile reaches a healthy endpoint at all. The tile-clamp caveat is real
  but does not distinguish them: it applies to every vLLM-on-T4G number, the good ones included.
  **Compare against `gpu-vllm-g5g-2b/benchmarks/runs/2026-08-14-rust-frontend-g5g/`** — `vllm bench
  serve`, three runs, one `g5g.4xlarge`: c=1 TPOT 31.44 ms (~31.8 tok/s decode), c=4 ~97 tok/s,
  c=8 168.33 tok/s.
- **Anything from `~/gemma4-tips`** — that tree duplicated its own artifacts and its directory
  names misattribute both model and chip.

A config flag being accepted is not evidence it did anything. Cross-check against a physical
bound, and **warm up at the shape you intend to measure** — `max_new_tokens` is a
`static_argnames` entry, so `(bucket, max_tokens)` *is* the compiled shape. On the parent that
was a 4x error (3.4 tok/s warmed at 32 and measured at 48, against 13.5 for the same config
warmed correctly). It is worse here: XLA compiles the shape on the same CPU that then has to
run it.
