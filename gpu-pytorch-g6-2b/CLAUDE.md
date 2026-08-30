# CLAUDE.md — `gpu-pytorch-g6-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **stock PyTorch + HF transformers** on
**AWS EC2 G6** — an **x86_64** host with an **NVIDIA L4** GPU (Ada, SM 8.9, **23034 MiB**
measured, not the nominal 24 GB).

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`.
It is **not** one of the `gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform
slot and the actual silicon with them.

"Stock PyTorch" is literal and is the whole point: **no custom model port, no XLA, no
compiled static shapes, no from-source build, no vLLM.** `AutoModelForCausalLM` and an HF
KV cache, behind an OpenAI-compatible FastAPI server (`torch_openai_server.py`), under
systemd — **not docker**.

> **PROVENANCE.** Created 2026-08-29 by retargeting `gpu-pytorch-g5g-2b` (the same runtime
> on a T4G) from G5g/Graviton2/Turing to G6/x86_64/Ada. It was **not** converted from
> `gpu-jax-g6-2b`, whose directory it briefly occupied as a byte-identical copy.
> It has **one measurement of its own** — `benchmarks/runs/2026-08-29-first-serve-g6/` —
> and everything below is either that run, or explicitly labelled as a sibling's.

## This rig has served

**One run, its own**: `2026-08-29-first-serve-g6` on `g6.2xlarge` spot in `us-east-1d`.
First serve, first sweep, teardown the same hour.

**20.93 tok/s median decode**, 8/8 cells, 0 degenerate, 0 failed of 64 requests, weights
**10.209 GB** at **bfloat16**. Decode spans 20.75–21.44 — a **3.3% spread over a 27x
context range**, which is the signature of a cost proportional to the *weights* rather than
the context. **KV is not what sets decode speed here.**

Four things that carry from the lineage and are easy to lose:

- **Concurrency is not an axis.** `MAX_NUM_SEQS=1`; the server decodes one sequence at a
  time. Sweep context and output length. This is also the single most interesting thing
  *not* yet tried — see the roofline section.
- **Installability is not a served token.** Wheels resolving for a platform was never the
  same claim as "it serves". `verify_gpu_arch` converts an install into evidence by running
  a real matmul on the device.
- **Warm up at the shape you measure.** Nothing is compiled here, but cuBLAS autotune and
  allocator growth are per-shape. `tpu_jax_cold_requests_total` counted 9 cold shapes across
  the sweep; `sweep.py` warms every cell before timing it.
- **`make skill` before `deploy_torch_server`.** `server.py` resolves the payload next to
  itself, so deploying from the MCP snapshot ships the *previous* `make skill` output with
  no warning. `verify_model_health` compares served vs local build id and says `STALE
  DEPLOY`. This run reported `a4be4ec1edb6` on both sides.

## The result, and the two comparisons

**Full analysis in the run's `REPORT.md`.** The summary:

| | weights | tok/s | ms/step | roofline | % of it | overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **this rig** — PyTorch, L4 | 10.209 GB | **20.93** | 47.78 | 34.02 | **71.2%** | **13.75 ms** |
| `gpu-jax-g6-2b` — JAX, L4 | 6.155 GB | 48.40 | 20.66 | 20.51 | 99.3% | 0.15 ms |
| `gpu-pytorch-g5g-2b` — PyTorch, T4G | 10.209 GB | 10.88 | 91.95 | 31.90 | 34.7% | 60.05 ms |

**The clean comparison is the T4G one, and only that one.** `gpu-pytorch-g5g-2b` runs the
identical runtime on the identical dense checkpoint with **byte-identical**
`tpu_jax_weight_bytes` (10,208,595,008). Only the GPU differs: **1.92x**.

**The JAX comparison is NOT clean and must never be quoted as a runtime result on its own.**
`gpu-jax-g6-2b` is the same chip but serves **6.155 GB** (`ple4 + int8_lm_head`) against this
rig's **10.209 GB**, and decode is bandwidth-bound on exactly those bytes. Normalising the
footprint away predicts ~29.2 tok/s here, leaving JAX ahead by **~1.66x**. *That* residual is
the runtime difference; the raw 2.31x is mostly quantization.

### The overhead is eager mode, and batching is the untried lever

This rig sits at **71% of its bandwidth roofline** where the JAX rig sits at **99%**. The
missing **13.75 ms/step** is not weight streaming. HF transformers decodes eagerly: hundreds
of small kernel launches per step, driven from Python, with no fusion and no graph capture.

Two consequences worth acting on, in order:

1. **Batching is the obvious win and is currently closed.** At `B=1` decode is a
   matrix-*vector* product, so the launch overhead is paid per token with nothing to amortise
   it against. This is the same shape as the JAX sibling's conclusion that "the route to the
   ceiling is a GEMM, not a dtype" — but here the fix is ordinary (`MAX_NUM_SEQS>1`), not
   blocked by a `NotImplementedError`.
2. **`torch.compile` / CUDA graphs would attack the launch overhead directly.** Untried.
   Note it trades against load time and adds recompilation on shape change.

**Neither has been measured. The 13.75 ms is arithmetic against the roofline, not a kernel
table** — `torch.profiler` would attribute it directly, and no profile was taken.

## Ada changed which lessons transfer

The T4G siblings spent days establishing that **87% of decode on Turing was dtype tax**.
None of that applies here, and the mechanism is worth keeping straight:

- **bfloat16 is native on Ada.** `resolve_compute_dtype` reads the live compute capability
  and picks bfloat16 at SM >= 8.0. VERIFIED on hardware — the first line the process emits is
  `torch device policy: name=NVIDIA L4 compute_capability=8.9 pre_ampere=False
  compute_dtype=bfloat16`, and `torch.cuda.is_bf16_supported()` is True.
- **Keep the guard anyway.** The failure it prevents is silent in exactly one direction:
  bfloat16 on a pre-Ampere GPU **does not raise** — CUDA emulates through fp32, output stays
  correct, and most of decode disappears. `DTYPE` in `tpu.env` is only an override.
- **The T4G's 34.7%-of-roofline is that tax**; this rig's 71% is a different, smaller problem
  (launch overhead), and the two must not be conflated.

**fp8 is newly available and is probably not the win it looks like.** Ada has an fp8
datapath, but KV is not the binding constraint on this engine and decode is bandwidth-bound
on the **weights**, which fp8 KV does not touch. If you want a memory win here it is weight
encoding — and this rig has no quantization path at all.

## What this rig deliberately does not have

**No quantization, anywhere.** No PLE table, no int8 LM head, no W4A16, no `--quant-mode`.
`torch_openai_server.py` defines exactly `--model`, `--host`, `--port`, `--seq`.

`tpu.env` still carries inert keys (`QUANT_MODE`, `PLE_BITS`, `INT8_LM_HEAD`,
`PREFILL_CHUNK_SIZE`) because it was forked wholesale. **They do nothing and must not be
re-plumbed into the serving command on the strength of existing** — argparse rejects an
unknown flag with exit code 2, so the unit would crash-loop under `Restart=on-failure` with
the reason only in `journalctl`. `_serve_argv`'s docstring says this at the call site.

One visible symptom: `/metrics` reports `quant_mode="fp16"` and `ple_bits="0"`. Those labels
are inert passengers, not a description of anything the process did.

**The XLA compilation-cache feature was REMOVED at this fork**, not carried across. torch
compiles nothing on this path, so the S3 restore would sync an empty prefix and the timer
would upload one, both reporting success forever. The expensive thing to re-fetch here is the
**10.2 GB checkpoint**; caching the HF hub directory would be a real win and is deliberately
not implemented on the strength of that reasoning alone.

## The arch check was wrong, and it failed a healthy GPU

**MEASURED 2026-08-29, and it cost this run its automated bootstrap.**

The install aborted on `assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()`. The
DLAMI's `torch 2.13.0+cu130` carries `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` —
**no `sm_89`** — and the L4 is `sm_89`.

**The GPU was fine; the assertion was wrong.** CUDA cubins run on any device of the **same
major** version with a **minor >= their own**, so `sm_86` covers `sm_89`. Proved on the box
before changing anything: fp16, bf16 and fp32 matmuls all ran correctly and
`is_bf16_supported()` was True.

Three things worth keeping:

- **The exact-match test was inherited from the Turing sibling, where `sm_75` IS in the list
  and it happened to hold.** A test that passes for the wrong reason survives a fork.
- **It fails in the expensive direction** — it rejects working hardware, which reads as "this
  chip is unsupported" rather than "this check is too strict".
- **The arch list is a claim; a launched kernel is the evidence.** Both the bootstrap and
  `verify_gpu_arch` now select the dtype the rig will actually serve (bf16 here, not fp16)
  and run the matmul in it. `CubinCompatibilityTests` pins all of it.

**Consequence for this run's provenance:** the install was completed by hand on the same
instance after the fix, so **the corrected bootstrap has not itself been launched from
scratch.** The next launch is the one that verifies it.

## Instance sizing

**`g6.2xlarge` is the default and the only size measured.** 8 vCPU, **32 GiB** host, 1 L4.

**G6 has twice the host RAM of G5g at the same suffix**, and the size table is the thing most
likely to be got wrong by habit. VERIFIED 2026-08-29 against `ec2 describe-instance-types`:

| size | GPUs | host RAM | | size | GPUs | host RAM |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `g6.xlarge` | 1 | 16 GiB | | `g6.12xlarge` | **4** | 192 GiB |
| `g6.2xlarge` | 1 | 32 GiB | | `g6.16xlarge` | **1** | 256 GiB |
| `g6.4xlarge` | 1 | 64 GiB | | `g6.24xlarge` | **4** | 384 GiB |
| `g6.8xlarge` | 1 | 128 GiB | | `g6.48xlarge` | **8** | 768 GiB |

- **`g6.16xlarge` is SINGLE-GPU despite the suffix.** On G5g, 16xlarge was multi-GPU. Never
  infer GPU count from the size name.
- **There is no `g6.metal`.**
- **Nothing shards.** The serving path is single-device, so a bigger instance buys host RAM
  and vCPUs, **not** device memory.

**Every size is supported.** `g6.xlarge` (16 GiB) is the one size at or below the inclusive
swap gate and the only one that renders a swapfile — and **it has never been launched here**,
so per the G5g lesson (a code path that only renders for a size nobody launches is untested
code) treat that block as unexercised. `g6.2xlarge` at 32 GiB is above the gate, so this run
exercised no swap at all.

The G5g swap evidence does **not** transfer by size name: its 2xlarge OOM was a JAX
`quantize_ple_table` upcast, and this rig has no PLE path. What does carry is the *mapping*
failure — without swap the kernel refuses to mmap the 10.2 GB checkpoint at all — because
that is a property of the checkpoint and the kernel, not the runtime.

## AMI resolution

`DLAMI_SSM_PARAMETER` pins `x86_64/oss-nvidia-driver-gpu-pytorch-2.13-ubuntu-26.04`.
**Never hardcode an AMI id.**

Three traps, all hit during this fork:

- **The arch is NOT in an x86_64 DLAMI's name.** Only ARM64 images announce theirs. The G5g
  filter with `arm64`→`x86_64` substituted becomes `"Deep Learning x86_64 AMI OSS ..."`,
  which matched **ZERO images** — verified against `describe-images`. A fallback that matches
  nothing is not a loud failure. The real name is
  `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.13 (Ubuntu 26.04)`, and the architecture
  is pinned by an explicit `architecture: x86_64` filter on the call instead.
- **`/latest/` in a parameter path is only the newest build within one PyTorch+Ubuntu line**,
  and AWS stops rebuilding old lines. 2.13/26.04 is live — the resolved AMI was built the
  same day it was pinned.
- **Changing the SSM path requires changing `DLAMI_NAME` in the same edit**, or the fallback
  quietly resolves a different image. Both are covered by
  `test_tpu_env_agrees_with_server_defaults`.

**Ubuntu 26.04 ships Python 3.14 as the system interpreter**, so deadsnakes leaves the
critical path — CONFIRMED, the `python-3.14` stage took **0s**. Total install: **43s**.

**But the interpreter is PROBED, never hardcoded, and that is load-bearing.** The DLAMI ships
torch in a venv, and on this image that venv is **Python 3.13**, not the 3.14 the system
carries. `TORCH_PYTHON_VERSION` only seeds candidate names; the bootstrap finds the
interpreter that can already `import torch` and installs into **that**, recording it in
`/opt/torch-g6/PYTHON_BIN` and rewriting `ExecStart` to match. Installing transformers into
`/usr/bin/python3.14` would give `ModuleNotFoundError: torch` *after* the install reported
success.

**torch comes from the DLAMI — but on Ada that is a preference, not a requirement.** On
Turing it was hard: upstream PyPI wheels omit `sm_75`, so pip torch served on CPU. Upstream
wheels carry `sm_89`. The DLAMI is used to keep a multi-GB download off a spot host's
critical path. **Do not re-derive the Turing rationale for it.**

## The bootstrap is two-stage, on purpose

Cloud-init installs the **runtime only** and then waits. The serving payload is this rig's
own source, shipped separately over SSM as a gzipped tarball (13 KiB of base64) — **user data
could not carry it: the limit is 16 KB.**

```
create_g6_instance → get_install_progress → verify_gpu_arch → deploy_torch_server
                   → get_torch_logs → verify_model_health
```

Install progress goes to `/var/log/torch-install.log`; `{APP_DIR}/INSTALL_DONE` appears only
after torch **imports and sees the GPU**. Stage timings: `grep -F '[stage]'`. The unit is
`torch-g6.service` — read it with `journalctl`, not `docker logs`.

Measured stages on this run: apt-base 7s, python-3.14 0s, torch-interpreter 14s,
pip-bootstrap 2s, torch-deps 13s, serving-deps 3s, profiling-deps 4s — **43s total**. Model
load was a further 27.3s (10.209 GB) plus 4.3s warmup.

**PEP 668 applies from Ubuntu 23.04 on**, so `--break-system-packages` is load-bearing rather
than defensive. Harmless inside the DLAMI venv, which is not marked.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Do not create broad
  network or IAM policy. This run used a dedicated `gpu-pytorch-g6-2b-sg` scoped to the
  operator's IP on 8000 — deliberately **not** the G5g rig's SG, which carries a
  `0.0.0.0/0` demo rule.
- Scope instance discovery to `ManagedBy=gpu-pytorch-g6-2b`.
- Hugging Face tokens live in Secrets Manager and are fetched at boot into a root-only
  `EnvironmentFile`. **Never** in user data — instance metadata is readable by anything on
  the box. `set +x` wraps the fetch because the script runs under `set -x` and bash traces
  assignments *with their values*. Tests assert both.
- Launches default to spot. Surface capacity errors rather than silently retrying.
- **Termination is cheap here**: no built image is lost, only a pip install and the model
  cache. Do not import the vLLM sibling's "weigh stop against terminate" reasoning.
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.
- `verify_model_health` uses `/v1/chat/completions`. **Do not health-check by testing for a
  non-empty response** — a sibling returned `': ok: ok: ok…'`, which passes that test. It
  reads `tpu_jax_degenerate_responses_total` either side of its own probe instead.

## Spot capacity

G6 spot in `us-east-1` was priced at **~96–100% of on-demand** on 2026-08-29
(`g6.2xlarge` spot $0.9408–0.9776 against $0.9776 list). Spot is capped at on-demand, so a
price at the cap signals **tight capacity, not a bargain** — expect little saving and real
interruption risk. This launch nevertheless got capacity in `us-east-1d` on the first try.

**Do not read cheapness as availability**; the G5g sibling measured the cheapest AZ being the
only one *with* capacity on one day and the most expensive on another. Retry across AZs.

Reclamation is highly variable — the G5g rig saw 21 minutes once and 19.2 hours another time
on the same instance type in the same region. **Quote the range, and checkpoint
continuously**: `sweep.py` writes per cell for exactly this reason, and `make_report.py` is
deliberately separate so a report can be shaped after the instance is gone.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v`
(**75 tests, all passing** as of 2026-08-29). Fully offline — no AWS, no network, no GPU.

`tests/test_engine.py` was **rewritten at this fork**. The inherited version imported
`ports.gemma4.jax_e_model`, which does not exist in a PyTorch rig, caught its own ImportError
and **skipped every test — silently, forever.** A test file that can only skip reports green.
It now stubs the compute capability and exercises the real `resolve_compute_dtype`.

`make lint` runs `ruff check server.py refresh_skill.py torch_generate.py
torch_openai_server.py sweep.py make_report.py tests`, then `bash -n` on four shell scripts.
**A new top-level module is silently unlinted until it is added to that list.**

`make skill` regenerates both snapshots. **`SKILL.md` is a hand-written source** —
`refresh_skill.py` will not recreate it, so `rm -rf .claude/skills` destroys it permanently.
`test_skill_is_complete_in_both_copies` guards both copies and fails if any generated file is
stale; it caught exactly that during this fork.

## MCP registration lives in four places

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and
`.claude/settings.local.json`'s `enabledMcpjsonServers`. All four must name the server
`gpu-pytorch-g6-2b`, which prefixes every tool as `mcp__gpu-pytorch-g6-2b__…`.

**Two of the four arrived from the fork wrong, and one was dangerous.** `.codex/config.toml`
was still the **JAX** rig's: it registered `gpu-jax-g5g-2b`, pointed at a `skills/` path that
does not exist, and gated `create_g5g_instance` / `terminate_g5g_instance` /
`stop_g5g_instance` — tools this rig does not have.

**A gate on a tool name that does not exist fails open and says nothing.** A rename silently
converts a safety control into a no-op while it still reads as configured. `plugin.json` was
correct in name but advertised "G5g (Graviton2 + NVIDIA T4G)" and pure-JAX. Ground truth for
tool names is always `grep -n "^@mcp.tool" server.py`. `deploy_torch_server` is gated too.

**Only `.mcp.json` is generated** (`project-setup.sh`, which derives `SKILL_STEM` from the
directory — never reintroduce a literal). `.claude/settings.local.json` is hand-written; both
are gitignored. The other two are committed and **nothing tests them against the directory
name**, which is why they keep surviving forks wrong.

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. **`CLAUDE.md` is
authoritative where they disagree**; there is no generator.

## Measurement

**This rig has ONE measurement of its own**, in `benchmarks/runs/2026-08-29-first-serve-g6/`.
The `-g6` suffix is the hardware **measured**, and it is load-bearing.

Quote the **`tpu_jax_decode_tokens_per_second` gauge**, not end-to-end: end-to-end carries
prefill and the HTTP round trip, so it falls with context (19.93 → 16.33) while decode does
not move (21.37 → 20.75). They disagree by design. The `tpu_jax_` prefix is an **identifier,
not a description** — it is kept so the sibling reports' comparisons stay valid by name;
series carry a `rig` label to separate them.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them, so edit the root originals.

Numbers you will be tempted to reuse, and must not:

- **10.66–10.98 tok/s** — `gpu-pytorch-g5g-2b`, same runtime on a **T4G**. This is the
  *control* this rig is measured against, not a baseline it inherits. Always name the chip.
- **48.3–48.5 tok/s** — `gpu-jax-g6-2b`, same chip, **different runtime AND 40% fewer weight
  bytes**. Never quote the raw ratio as a runtime result.
- **43–44 tok/s** — the vLLM sibling on **T4G**, obtained with reduced Triton tiles.
- **Anything from `~/gemma4-tips`** — that tree duplicated its own artifacts and its directory
  names misattribute both model and chip. **This applies with special force to the five
  `gpu-vllm-l4-*` artifact rigs, which are the same GPU as this rig** — same chip is not same
  measurement, and their provenance is the weakest in the tree.

A config flag being accepted is not evidence it did anything. Cross-check against an absolute
physical bound — **~300 GB/s of GDDR6 and 23034 MiB is the whole envelope here** — not
against another config. This run peaked at **14142 MiB of 23034**, so `MAX_MODEL_LEN=4096` is
an inherited conservative value with real headroom behind it, **not a measured ceiling**.
Finding the actual context ceiling is the cheapest open question.
