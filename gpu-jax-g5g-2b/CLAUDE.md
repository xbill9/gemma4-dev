# CLAUDE.md — `gpu-jax-g5g-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **pure JAX** on **AWS EC2 G5g** — a Graviton2
(aarch64) host with an **NVIDIA T4G** GPU (Turing, SM 7.5, **15360 MiB** measured, not the
nominal 16 GB).

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`.
It is **not** one of the `gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform
slot with them.

"Pure JAX" is literal: no PyTorch, no torch_xla, no vLLM. The engine is this repo's own
Gemma 4 port (`ports/gemma4/`) driven by `jax_engine.py` behind an OpenAI-compatible FastAPI
server (`jax_openai_server.py`), run under systemd — **not docker**.

## This rig has served

**Five runs, all its own**, on `g5g.2xlarge`. `2026-08-19-first-serve-g5g` is the first
serve and `2026-08-21-cuda13-py314-g5g` repeats it on CUDA 13 / Python 3.14;
`2026-08-25-context-sweep-g5g` is the first sweep and the first xprof profile;
`2026-08-26-config-sweep-g5g` and `2026-08-26-quant-levers-fixed-g5g` break and then fix the
quantization levers. The headline is **13.1 tok/s single-stream** at the current default,
**12.8** at the one the first three runs measured. See **Measurement** for the table.

Two things that still hold and are easy to lose:

- **The grid is still thin in one axis and closed in another.** Context × output length is
  swept (12 cells, 2026-08-25) and serving config is swept (5 configs, 2026-08-26), but
  **concurrency is not an axis and cannot be**: `MAX_NUM_SEQS=1`, `Gemma4EModelJAX` raises
  `NotImplementedError` for `B > 1`, and the decode step donates its KV buffers. `tpu.env`
  still marks values MEASURED or PREDICTED; respect the split and do not quote a PREDICTED
  number as a result.
- **Installability is not a served token.** `jaxlib` and the CUDA plugins publishing
  `manylinux_2_27_aarch64` wheels, and the PJRT arch tables carrying `sm_75` with an SM 6.0
  floor (verified 2026-08-18), were never the same claim as "it serves".

`verify_gpu_arch` remains the cheapest way to convert the install into evidence — it runs a
real fp16 matmul on the device rather than checking that a flag was accepted.

**Warm up before recording anything.** Cold prefill measured **56x** slower than warm on
2026-08-19; a harness that skips warm-up understates this rig by more than 2x on decode.

## Why the hardware slot is `g5g` and not `t4g`

Settled 2026-08-12; `NAMING.md` has the carve-out. Do not "correct" this back.

The GPU really is an NVIDIA **T4G**, and slot 3's normal rule is the GPU SKU, so `t4g` looks
right. Two facts outweigh it:

- **`t4g` is already an EC2 instance family** — `t4g.nano`…`t4g.2xlarge`, Graviton2
  burstable **CPU** boxes with no GPU. In an AWS context the string reads as a cheap CPU
  instance far more often than as a GPU.
- **G5g is the only Graviton+GPU family AWS ships**, and no Graviton3 or Graviton4 GPU
  instance exists. So `g5g` is not a lossy stand-in for the chip the way `ec2` or `cloudrun`
  would be — it names the Graviton2+T4G pairing exactly, and that pairing, not the GPU alone,
  is what makes this hardware hard.

The chip is still called T4G everywhere it is the chip being discussed — in this file, in
`HARDWARE.md`, and in `tpu.env`. Only the slot is `g5g`.

## Why this rig exists next to `gpu-vllm-g5g-2b`

Identical hardware, different runtime, and **the runtime is the whole point.**

The vLLM path gets to a served token, but only through a ~67-minute from-source build for
SM 7.5, a CUDA toolkit the DLAMI does not ship, a Rust toolchain, and an **unlanded patch to
Triton's attention kernel** that lives on one instance and has to be reapplied after every
upgrade. That last one is not a packaging problem: Gemma 4's heterogeneous head dims
(sliding 256, global **512**) force `TRITON_ATTN`, whose 512-wide tile wants ~96 KiB of
shared memory per block against Turing's 64 KiB ceiling.

JAX sidesteps all four:

- **No build.** pip supplies CUDA; the DLAMI only has to supply the driver.
- **No toolkit and no Rust.** Nothing is compiled on the instance.
- **No arch gap.** The plugin's cubins cover SM 7.5 already.
- **No patch to carry.** Attention here is ordinary XLA, not a hand-tiled Triton kernel, so
  there is no per-block shared-memory ceiling in the attention path to hit.

`docs/turing-aarch64-gap.md` is the vLLM-side write-up of all of this and is **measured** —
it is the sibling's evidence, kept here because it is the reason this rig was built.

## What JAX does *not* sidestep

The same 64 KiB ceiling bites in a different place. The fused **W4A16 Pallas kernel** is
tiled for TPU VMEM (16 MB per core) and needs **550 KiB – 1.1 MiB** per block at this model's
shapes — three orders of magnitude past what Turing allows. On GPU, Pallas lowers through
Triton and those tiles become shared memory, so the kernel cannot run.

`check_w4a16_fits_scoped_memory()` in `ports/gemma4/jax_e_model.py` computes the requirement
and **raises at startup with the arithmetic attached**, rather than dying as an
`OutOfResources` at the first token — which is exactly the failure mode that made the vLLM
path hard to diagnose.

So this rig serves the **dense reference checkpoint** (`MODEL_NAME=google/gemma-4-E2B-it`,
`QUANT_MODE=fp16`), deliberately **not** the `-qat-w4a16-ct` export the TPU JAX rig serves.
Two reasons, and the second is load-bearing:

1. The dense model fits — 9.5 GiB of weights in 15360 MiB of device memory.
2. Serving w4a16 here would silently fall back to dequantize-then-matmul: **4x the weight
   traffic of the fused path *and* the dense model's memory**, the worst of both.

Because the reference build is what it serves, the rig name carries **no encoding slot**.

## Turing is not L4, and it is not v6e

The five `gpu-vllm-l4-*` rigs were written for SM 8.9; the model port came from a TPU rig.
**Turing has no bf16 and no fp8.** Do not copy a flag from either lineage.

| | TPU v6e-1 | L4 siblings (SM 8.9) | this rig (SM 7.5) |
| --- | --- | --- | --- |
| compute dtype | `bfloat16` | `bfloat16` | **`float16`** |
| KV cache dtype | `bf16`/`fp8` | `fp8` | **`auto` → float16**; `int8` to halve it |
| scoped memory | 16 MB VMEM | — | **64 KiB/block, opt-in** |
| matmul precision | `jax_default_matmul_precision=bfloat16` | — | **left at JAX's default** |
| fused W4A16 Pallas | yes | — | **refused at startup** |

Three things about the dtype policy that are easy to get wrong:

- **The device decides, not `tpu.env`.** `ports/gemma4/jax_e_model.py` reads the live
  compute capability and picks `float16` below SM 8.0. `DTYPE=float16` in `tpu.env` is the
  *override*, and `JAX_E_COMPUTE_DTYPE` is the escape hatch.
- **bfloat16 does not fail here, it emulates.** XLA runs it through fp32 conversions, so the
  numbers come out right and every matmul quietly pays. That is worse than an error, and it
  is why `_PRE_AMPERE_UNSUPPORTED` in `jax_engine.py` deliberately excludes `bfloat16` —
  it gets a warning. **fp8 is refused outright**: `resolve_cache_dtype()` raises rather than
  silently downgrading.
- **`jax_default_matmul_precision` is set only on TPU.** Setting `"bfloat16"` tells XLA it
  may demote fp32 matmul inputs to a format Turing has no unit for. `jax_openai_server.py`
  used to set it unconditionally; do not reintroduce that.

`fp8_e5m2` is in the cache-dtype table and is **DEGRADED** — same capacity as int8, two
mantissa bits, visibly truncates output. Kept for comparison; never serve with it.

## The model port is vendored, not this rig's own

`ports/gemma4/` is a clean-room JAX port shared with `tpu-jax-v5e1-2b`. Consequences:

- **`make lint` deliberately excludes it.** ruff's UP006/UP045 would rewrite its
  `Dict`/`Optional` annotations, which the monorepo `CLAUDE.md` forbids and which would drift
  it away from the sibling copy.
- **`detect_hardware_profile()` falls back to the TPU profile off-accelerator.** A `HARDWARE`
  read on a CPU host reports `tpu-v6e-1`, not a T4G. The tests exercise the Turing branch by
  overriding the detected platform under `JAX_PLATFORMS=cpu` — that tests the *policy*, not
  the hardware.
- Fixes that describe the **model** belong in the root `MODELS.md`; fixes that describe the
  **chip** belong in `HARDWARE.md`. Only a measurement stays in this rig.

## Instance sizing, and the swapfile

`g5g.2xlarge` is the default. **Every size is supported** — `_validate_instance_type` only
enforces the size list.

`_user_data` provisions a swapfile at or below `_SWAP_AT_OR_BELOW_HOST_RAM_GB = 16`, which
covers **`g5g.xlarge` and the default `g5g.2xlarge`**. Two distinct pressures, both measured,
and the larger one sets the threshold:

- **`g5g.xlarge` (8 GiB) cannot even mmap the checkpoint** (below).
- **`g5g.2xlarge` (16 GiB) mmaps fine and then dies in `quantize_ple_table`**, which upcasts
  the 4.70 GB PLE table to float32 while the full tree is resident. MEASURED 2026-08-26:
  OOM-killed five times at 14.3 GB anon-RSS under `Restart=on-failure`. The gate was
  `< 16` and 2xlarge has *exactly* 16, so the rig provisioned swap for the one size that did
  not need it for this and skipped the one that did. **The threshold was the bug, not the
  remedy** — the `fallocate`/`mkswap`/`swapon` block already existed.

**Making it inclusive immediately exposed a second bug**, which is worth keeping as a pattern:
the swap block had used `mkswap -q`, a **busybox** flag that util-linux (Ubuntu 22.04) rejects
with `invalid option -- 'q'`. Under `set -e` that killed cloud-init *before install.sh was
written*, so the instance booted with an empty `/opt/jax-g5g` and no install log at all. It
was latent for as long as only `g5g.xlarge` rendered the block and nobody launched one.
Two lessons: **a code path that only renders for a size nobody launches is untested code**,
and the reason this cost a launch rather than a minute is covered under **Tracing a
deployment** below.

The mmap failure, measured on the vLLM sibling 2026-08-13 with the same checkpoint and host,
so it carries: without swap the kernel refuses to **mmap** the 10.2 GB checkpoint at all —

```
RuntimeError: unable to mmap 10246621918 bytes from model.safetensors:
Cannot allocate memory (12)
```

— and systemd crash-loops on it. The failure is the *mapping*, not residency; 16 GiB of swap
took the same instance to a healthy endpoint. **This rig once rejected `g5g.xlarge` outright
on the theory that 8 GiB "cannot stage 9.5 GiB of weights". The conclusion was right and the
reason was wrong**, and the remedy is swap rather than a bigger instance — the same fix
`tpu-pytorch-inf2-2b` applies for its neff load.

`g5g.16xlarge` and `g5g.metal` carry two T4Gs. **Nothing shards across them.** The engine is
single-device (`jax.devices()[0]`), `_serve_argv` emits no tensor-parallel flag, and the
second GPU idles. `_tensor_parallel_size` reports the GPU count and nothing acts on it yet.

## AMI resolution

Two requirements, and they are separate:

1. **arm64** — the `ami-012ba162b9cd2729c` the legacy tips-tree rigs hardcode is x86_64 and
   cannot boot on Graviton2 at all.
2. **The NVIDIA driver** — AWS also ships ARM64 DLAMIs built for Graviton *CPU* inference.
   Those boot perfectly well on a G5g and simply have no GPU, which reads as a broken runtime
   rather than a wrong AMI.

`DLAMI_SSM_PARAMETER` pins both and is single-valued, so it is preferred; the `DLAMI_NAME`
describe-images filter is the fallback and deliberately requires the driver in the name.
**Never hardcode an AMI id** — resolve it at launch.

### `/latest/` in a DLAMI parameter path does not mean latest

**This is the trap, and the rig was in it.** A DLAMI parameter path names a
PyTorch-version *and* an Ubuntu-version line, and `/latest/` is only the newest build **within
that line** — which AWS eventually stops rebuilding. The rig pinned
`oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04`, which resolved to an image built
**2026-05-02** and will never move again: 22.04 stops at PyTorch 2.7, and 2.8–2.12 are 24.04
only. It read as "track latest" and was a pin to a dead line. VERIFIED 2026-08-27 by
enumerating the 26 `arm64` DLAMI parameters.

**Changed 2026-08-27 to `base-oss-nvidia-driver-gpu-ubuntu-26.04`.** Two independent reasons:

- **Base, not PyTorch.** This rig never installs into the DLAMI's PyTorch — it ships its own
  CUDA libraries and jax brings its own — so a PyTorch DLAMI is GBs of image whose entire
  content is deliberately unused.
- **26.04 ships Python 3.14 as the *system* interpreter** (3.14.3, verified in the Ubuntu
  archive), and 3.14 is exactly `JAX_PYTHON_VERSION`. That takes the deadsnakes PPA, an
  `add-apt-repository` and a second full `apt-get update` off the critical path. 24.04 ships
  3.12 and would still need them. deadsnakes publishes `python3.14` for jammy and noble only —
  it does not need to for resolute.

**Not yet launched on.** Three things were true of the old base and must be re-checked on the
first launch, in this order of risk: the **driver version** (was 580.126.09 reporting CUDA
13.0, and `jax[cuda13]` needs 580+ — this is the one that could actually break); whether the
**aws CLI** is present, since the base image is smaller; and the **PEP 668** handling. Falling
back is one env var — the 24.04 base still works via deadsnakes.

**Changing the SSM path required changing `DLAMI_NAME` in the same commit.** The old filter
required `Deep Learning ARM64 AMI` *contiguously*, and the base images are named
`Deep Learning ARM64 Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04)` — so it matched none of
them, and the fallback would have quietly resolved the **old PyTorch image**. A revert that
reports success. The two keys are now both covered by
`test_tpu_env_agrees_with_server_defaults`, which had been checking neither.

**jax and Python were already current** and needed no change: `jax[cuda13]` carries no version
so pip resolves latest (0.11.1, and `cuda13` is the newest CUDA extra jax publishes — there is
no `cuda14`), and 3.14 is the newest stable CPython. **3.15 is deliberately not used**:
`jaxlib` publishes cp315 aarch64 wheels, but that is jaxlib building against a pre-release, not
evidence that 3.15 has shipped. `tpu.env` records that constraint.

**There is no prebuilt AMI here and none is owed.** The vLLM sibling needs one because it
carries a 67-minute build; this install is `pip install`, so a stock DLAMI is the right base.
Do not copy that rig's `ami-0b44b90b3d02430ee` into anything here — it is a vLLM image with
a Triton patch and no JAX. (The separate question of baking an AMI to survive *spot
reclamation* is open — see **Tracing a deployment**.)

## The bootstrap is two-stage, on purpose

Cloud-init installs the **runtime only** and then waits. The serving payload is this rig's own
source — there is no published artifact for "our JAX Gemma 4 port", and cloning the monorepo
would need credentials on the box — so it ships separately over SSM as a gzipped tarball
(~30 KB of base64). **User data could not carry it: the limit is 16 KB.**

The tarball is built deterministically (mtime and uid/gid zeroed), which is what makes
`deploy_jax_server` idempotent and lets an unchanged redeploy be detectable.

Order of operations:

```
create_g5g_instance → get_install_progress → verify_gpu_arch → deploy_jax_server
                    → get_jax_logs → verify_model_health
```

Install progress goes to `/var/log/jax-install.log`; `{APP_DIR}/INSTALL_DONE` appears only
after JAX **imports and sees the GPU**, so "INSTALL COMPLETE" is an assertion, not a guess.
The unit is `jax-g5g.service` — read it with `journalctl`, not `docker logs`.

`jax >= 0.11` needs Python >= 3.12 and the rig runs **3.14**, the newest stable CPython. On
the Ubuntu 26.04 base that is the *system* interpreter, so the bootstrap uses it directly; on
an older base (reachable by overriding `DLAMI_SSM_PARAMETER`) it falls back to installing 3.14
from deadsnakes. The branch is `command -v python3.14` — **do not make it unconditional in
either direction.**

It deliberately does **not** install into the DLAMI's own PyTorch environment: that ships its
own CUDA libraries and `jax[cuda13]` brings its own. That is also why the AMI is the **base**
image and not a PyTorch one — see **AMI resolution**.

**PEP 668 applies from Ubuntu 23.04 on.** The system interpreter is marked
externally-managed, so a system-wide `pip install` fails outright with
`error: externally-managed-environment`. The bootstrap passes `--break-system-packages`
(including to `get-pip.py`, which runs before the wrapper variable exists). This is a
single-purpose serving box installing into the interpreter systemd will run, and the monorepo
forbids virtualenvs, so the override is the honest answer rather than a workaround.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Do not create broad
  network or IAM policy. (The legacy sample this was scaffolded from auto-creates a security
  group open to `0.0.0.0/0` — that was not carried over.)
- Scope instance discovery to `ManagedBy=gpu-jax-g5g-2b`. Unlike the inf2 rig, which keeps a
  legacy tag to avoid orphaning instances, this rig is new and uses its own name.
- Hugging Face tokens live in Secrets Manager and are fetched at boot into a root-only
  `EnvironmentFile`. **Never** in user data — instance metadata is readable by anything on
  the box. `set +x` wraps the fetch because the script runs under `set -x` and bash traces
  assignments *with their values*. Tests assert both.
- Launches default to spot. Surface capacity errors rather than silently retrying.
- **Termination is cheap here**, unlike on the vLLM sibling: there is no built image to lose
  with the root volume, only a pip install and the model cache. Do not import that rig's
  "weigh stop against terminate" reasoning.
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.
- `verify_model_health` uses `/v1/chat/completions`, because raw `/v1/completions` skips the
  chat template and is unreliable on `-it` models. On the vLLM sibling it was measured
  returning `': ok: ok: ok…'` — degenerate repetition, not the empty body the monorepo
  `CLAUDE.md` documents for the TPU rigs. Either way: **do not health-check by testing for a
  non-empty response**, or you will call a body full of garbage fine.

## AWS credentials

`server.py` uses the standard boto3 provider chain, so whatever `aws sts get-caller-identity`
resolves is what the rig gets. **When credentials expire, refresh them with
`./save-aws-creds.sh`**, which re-exports the active credentials to `.aws_creds` at mode 0600.

Three things about it that are easy to get wrong:

- **It snapshots credentials, it does not mint them.** `aws configure export-credentials`
  fails outright on an expired session, so re-authenticate first and then run the script. Its
  error message says this; the failure otherwise reads as a broken script rather than an
  expired login.
- **It refuses to write anywhere inside a git work tree that is not gitignored.** `.aws_creds`
  is in this rig's `.gitignore` for exactly that reason. Never remove that line and never
  reach for `FORCE=1` — the guard is the thing keeping live keys out of a commit.
- **Nothing in this rig reads `.aws_creds` automatically.** The script's closing message
  ("the Makefile will now use these") is inherited from the legacy `~/gemma4-tips-aws` tree,
  whose Makefile loaded the file; this rig's does not. The snapshot is for exporting into a
  shell or handing to a container. For `server.py` itself the provider chain is enough, and
  `AWS_PROFILE` is the supported way to pick a profile.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v` (122 tests,
all passing as of 2026-08-27). They are fully offline — no AWS, no network, no GPU — and pin
the facts above: the Turing dtype constraints, the arm64+driver AMI filter, the host-RAM
floor, the shared-memory ceiling, that the token never reaches user data, that `tpu.env` and
`server.py` still agree, and that no `VLLM_*`/`TORCH_CUDA*` key survived the fork.

`make lint` runs `ruff check server.py refresh_skill.py jax_engine.py jax_openai_server.py
profile_decode.py tests`, then `bash -n` on **four** shell scripts (`project-setup.sh`, `init.sh`,
`set_env.sh`, `save-aws-creds.sh`). **A new top-level module is silently unlinted until it is
added to that list** — `profile_decode.py` sat outside it and was red for a day. `ports/` is excluded on purpose — see above.

**`deploy_jax_server` ships the SKILL SNAPSHOT, not the working tree.** `server.py` resolves
the payload next to itself, and the MCP server runs from `.claude/skills/…/mcp/`, so editing
`ports/gemma4/jax_e_model.py` and deploying ships the *previous* `make skill` output with no
warning — the deploy reports success and the instance runs stale code. Cost one full
measure-and-conclude cycle on 2026-08-24 before the md5s were compared. **Always `make skill`
before `deploy_jax_server`.**

`make skill` regenerates the snapshots under `.claude/skills/` and `skills/`. **Eight files
are generated**, not just the MCP control plane: `server.py`, `project-setup.sh`, both
requirements files, **and the whole serving payload** (`jax_openai_server.py`,
`jax_engine.py`, `ports/gemma4/jax_e_{loader,model}.py`) — because an installed copy under
`~/.claude/skills` still has to be able to run `deploy_jax_server`, and `server.py` resolves
the payload next to itself first.

`SKILL.md` sits in the same tree and is a hand-written **source**: `refresh_skill.py` will not
recreate it. So `rm -rf .claude/skills` destroys it permanently, which is what happened during
the t4g→g5g rename. `test_skill_is_complete_in_both_copies` now guards both copies, and also
fails if any of the eight generated files is stale.

There is no `make deploy` recipe on purpose: provisioning resolves an arm64 AMI at launch
time, and a Makefile would have to hardcode one. The target exists and prints that.

## MCP registration lives in four places

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and
`.claude/settings.local.json`'s `enabledMcpjsonServers`. All four must name the server
`gpu-jax-g5g-2b`, which prefixes every tool as `mcp__gpu-jax-g5g-2b__…`. All four agree as of
2026-08-18. A mismatch makes `/mcp` and the tool prefix disagree about what this rig is.

**Only `.mcp.json` is generated.** `project-setup.sh` writes it (`--server-name` sets both the
registered key and what the server advertises) and does **not** touch
`.claude/settings.local.json` — despite what the old text here claimed. That file is
hand-written; both are gitignored. The other two are committed.

**The fork left all four wrong, and the failure was not cosmetic.** `plugin.json` and
`.codex/config.toml` both named `gpu-vllm-g5g-2b` and pointed at
`skills/gpu-vllm-g5g-2b-management/mcp/server.py`, a path that does not exist. Worse,
`project-setup.sh` carried a **hardcoded** `SKILL_STEM="gpu-vllm-g5g-2b-management"`, so it
could not find the skill at all and died with `cannot locate the bundled skill` — the rig was
unregisterable, not merely misregistered. `SKILL_STEM` is now **derived** from the rig
directory, matching what the `Makefile` and `refresh_skill.py` already did. Never reintroduce
a literal: the Makefile, `refresh_skill.py`, and this script must agree on one name, and a
literal is what silently survives a rename.

`server.py` was right throughout (`RIG_NAME = "gpu-jax-g5g-2b"`, asserted by
`test_rig_name_matches_directory`) — which is why the breakage was invisible to the tests.

Editing any of the four generated-or-copied files means re-running `make skill`:
`project-setup.sh` is one of the eight files snapshotted into both skill copies, and
`test_skill_is_complete_in_both_copies` fails until you do.

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree**, and a convention change has to be
applied to all three by hand. Both are currently **still the vLLM rig's copies** and describe
a runtime this rig does not use.

This rig has no `.claude-plugin/marketplace.json` of its own, which only matters if it is ever
published standalone. The marketplace `/plugin` actually reads is the **monorepo root** copy,
and it gained a `gpu-jax-g5g-2b` entry on 2026-08-18.

## How large a model this rig will serve

**`docs/larger-models-on-t4g.md` — measured 2026-08-23. E2B is the ceiling today.**

**Its DENSE prefill bracket is superseded.** That document bracketed the dense checkpoint at
(115, 2015] tokens; the `logits_at` fix lifted it, the 2026-08-25 sweep cleared 4,105, and
2026-08-26 located the ceiling between **4,105 and 5,120**. See the `MAX_MODEL_LEN` section.
The model-size table below is unaffected — those blockers are load-time and per-request
residency, not prefill context.

| Model | Loads? | Serves? | Blocker |
| --- | --- | --- | --- |
| E2B QAT + `ple_bits=4` | yes | **yes** | — 3.05 GB, 13.5 tok/s |
| E4B QAT | **no** | — | OOM 5.25 GiB *during load* |
| 12B QAT | yes (8.15 GB) | **no** | OOM 12.61 GiB *per request* |
| 26B A4B | — | — | **no w4a16 export exists (404)**; 15.27 GiB > budget |
| 31B | — | — | ~15.5 GB int4 > budget |

Three things worth keeping:

- **The budget is 14.07 GB on every G5g size.** The engine is single-device and the payload
  contains no sharding primitives at all, so the second T4G on `16xlarge`/`metal` idles.
  A bigger instance buys host RAM, not device memory.
- **E4B and 12B fail on TRANSIENT allocations, not resident weights.** Both fit comfortably.
  That is a tractable class of problem, unlike 26B/31B which are hard-blocked on residency.
  The transients (4.52 / 5.25 / 12.61 GiB) scale with model size and are **unidentified** —
  finding them is what stands between this rig and both larger models.
- **`MODELS.md`'s int4 column under-predicts by 19%** (E2B measured 3.054 GB vs 2.58 GB) —
  it quarters everything, but `embed_tokens` stays bf16 and the scales cost extra. Its bf16
  column over-predicts by 9%.

**`max_tokens` is part of the compiled shape** (`max_new_tokens` is a `static_argnames`
entry), so warming up at a different `max_tokens` than you measure leaves the measured request
cold. Measured here as a 4x error: 3.4 tok/s warmed at 32 and measured at 48, against 13.5
tok/s for the same config warmed at the shape it was measured at.

## A silent correctness bug in the shared port

**`docs/padding-window-eviction.md` — FIXED 2026-08-24, verified on CPU and on a T4G.**
Nothing in the mechanism is Turing-specific, which is what made a CPU reproduction possible.
`tpu-jax-v5e1-2b` is **still unfixed**: that copy of `jax_e_model.py` has diverged (1,570 lines
against 1,842 here, so they are NOT byte-identical and "shared therefore affected" is not a
valid argument), every ingredient is present in it, and nothing here touched it.

Right-padding to a power-of-two bucket writes pad K/V into the sliding layers' 512-slot KV
ring, **evicting the real tokens**. At `pad_len >= 512` the ring holds only padding, 28 of
E2B's 35 layers attend to an entirely masked window, and the model emits a token loop that
the server records as `status="success"`.

Three things that are easy to get wrong about it:

- **It is not a long-context bug.** A 1,415-token prompt fails and a 4,055-token prompt
  succeeds. Padding is the variable; length only makes large padding likely. Predicting
  `pad_len >= 512` scored 14/14 across two buckets.
- **It is not numerical.** `bfloat16` (emulated through fp32 on Turing, so strictly more
  headroom) reproduces the failure table byte-for-byte.
- **The existing guard in `make_ring_decode_mask` does not cover it.** That docstring
  documents the pad gap and correctly stops the model *attending to* pad K/V. It does not
  stop pad K/V *evicting* real K/V from a ring shorter than the padding.

- **It is decided at the first decode step, not progressively.** Generated tokens refill the
  ring, so pad=407 stays coherent through 600 generated tokens while pad=2035 loops from the
  first token. Guaranteeing `pad_len < 512` therefore *prevents* the failure rather than
  postponing it, which is why the bucket ladder is worth having as well.

**The fix is an invariant: a cache index is an absolute real position, and padding never
occupies an index a real position uses.** Three changes carry it — `_ring_store_one` takes
`real_len` and gathers only real positions into the ring, `cache_valid` is threaded through
`Gemma4EModelJAX.__call__` so prefill can supply it, and decode writes at `prompt_len + t`
rather than `bucket + t`. `make_ring_decode_mask` and `make_decode_mask` are **unchanged** and
stay correct under it.

Two things about that worth keeping:

- **Gating the prefill write on `prompt_valid` alone would NOT have fixed it**, despite the
  write-up originally saying so. It removes pad K/V from the ring's *contents* but leaves the
  pad indices in the cache's *coordinate space*, and the mask then rejects those slots anyway.
  Masking cannot repair a layout problem; the gap had to be removed, not skipped.
- **`B > 1` now raises `NotImplementedError`** rather than silently reverting to a shared
  bucket slot. A row's real length only coincides with the bucket at `B == 1`, and both engines
  here serve `MAX_NUM_SEQS=1`.

`static_sequence_buckets` also changed, as defence in depth: `(64, 128, 256)` plus 128-steps to
16384, so worst-case padding is **127 tokens** instead of `B/2`. Costs one compile per newly
seen bucket, amortised by the persistent compilation cache.

**Verified as padding invariance, not as "does not loop".** `tests/test_engine.py` builds a
four-layer random model on CPU (three sliding layers at `window=8`, one full-attention) and
asserts the generated tokens are identical at pad 0, 4, 8 and 28. Against the pre-fix port that
test reproduces the reported signature exactly: every pad at or above the window returns the
*same* degenerate sequence, with a token repeated four times running.

**Confirmed on a T4G 2026-08-24** by forcing the OLD power-of-two ladder back in, so
`pad_len >= 512` is reproduced rather than avoided: 1,515 tokens at pad 533 and 3,515 at pad
581 — both of which looped on 2026-08-23 — now return coherent continuations. That is what
establishes the store fix rather than the ladder as the remedy. TPU remains untested.

`tpu_jax_degenerate_responses_total` still counts occurrences; it is observational, changes
neither the response nor the status code, and is kept because it does not depend on eviction
being the only cause.

## bf16 weights are the transient nobody could name

**`docs/bf16-weights-on-turing.md` — measured 2026-08-24. Root cause confirmed, NOT fixed.**

The unexplained per-request transients in `larger-models-on-t4g.md` are **dtype conversions**,
not dense-materialised quantised weights. The loader stores every float parameter as
**bfloat16** while `COMPUTE_DTYPE` here is **float16**, so XLA converts in front of every use
and the converted copy is a transient the size of the weight. The prefill HLO names it:
`f32[262144,1536] wrapped_convert` — 1.50 GiB, the LM-head weight, where 1536 is E2B's
`hidden_size`, not a sequence length. `embed_tokens_per_layer` at bf16[262144,8960] = 4.375 GiB
accounts for the "4.52 GiB" figure.

Two things make this diagnosable rather than a guess, and both are worth keeping:

- **The transient is FLAT in the prompt bucket** — 1.504 GiB at 512 and at 1,536, 1.742 GiB at
  4,096. Quadratic would be attention scores, linear an activation. `profile_prefill.py
  --sweep` exists to make exactly that distinction, off `compiled.memory_analysis()` and the
  optimized HLO, and it never has to let the allocation succeed.
- **It is the same conversion `profile_decode.py` measured as 55% of decode time** the day
  before. One cause, two symptoms, two tools.

**Do not "just" change the loader default to `COMPUTE_DTYPE`.** It is one line, it is correct,
and all three placements of the resulting cast were tried on hardware and are worse than the
convert: on-device OOMs (source and destination resident together), host-side at shard load is
unusably slow (`ml_dtypes` casts are not vectorised — E2B's 4.7 GB table did not finish in 10
minutes on Graviton2), and building the tree on the host under `jax.default_device(cpu)` then
placing it cannot find a contiguous 4.38 GiB block even ordered largest-first. The untried
direction is a `view(uint16)` bit-twiddle, which is what bf16→f16 actually is.

**`ml_dtypes.bfloat16` is an extension dtype**: `dtype.kind` is `'V'` and
`np.issubdtype(bf16, np.floating)` is **False**. A `kind == "f"` guard converts float16 and
float32, which need nothing, and silently skips bf16, which is the only dtype that needs it.

**What did land:** `prefill_with_kv_cache` selects the last real token *before* the LM head
(`logits_at`) instead of computing `[B, S, vocab]` and slicing one row. Confirmed in the HLO —
no sequence-sized logits tensor at bucket 4,096 — and it lifted the dense ceiling: the dense
checkpoint now serves 1,515 and 3,515 tokens, where `larger-models-on-t4g.md` bracketed it at
(115, 2015].

## The rig could not explain its own failures

**Added and CONFIRMED ON A T4G 2026-08-25** on `i-0bd73466d5a07a578`
(`g5g.2xlarge`, AMI `ami-077792d0bb6a000b8`, jax 0.11.1 / Python 3.14), after an
end-to-end CPU pass against the real FastAPI app.

**It had to be on-demand, and that is a finding about the hardware, not the
code.** G5g spot was exhausted across all four `us-east-1` AZs and all three
sizes tried. Polling won capacity in ~9 minutes and AWS reclaimed the instance
**21 minutes later** (`instance-terminated-no-capacity`, launched 16:06:43Z,
terminated 16:27:41Z) — before the wheel install finished. A verification cycle
here needs ~30-45 minutes (install, deploy, a 9.5 GB load, warm-up, queries), so
**spot in this family cannot currently sustain one.** Budget on-demand for
anything that has to run to completion.

That reclamation was itself diagnosed by this work: the failure surfaced as
`SSM Failed (command-id d48800b7-…)` where it would previously have read
`SSM Failed:` with an empty body, and the id yielded `ResponseCode: -1`, which
pointed at the agent vanishing rather than a broken command.

Every incident already written up above cost more than it should have because
the evidence was being destroyed as it was produced. The specific mechanisms:

- **Every `logger.info` in the serving payload was discarded.** Measured, not
  inferred: `jax_openai_server.py` never called `logging.basicConfig`, and
  `uvicorn.run()` configures only its own `uvicorn*` loggers — it never adds a
  root handler. Root kept **zero handlers before and after** uvicorn's
  `dictConfig`, so `logging.lastResort` handled records at WARNING and above and
  the module loggers' effective level was 30. That silently dropped the
  `jax_e_model` **device-policy banner** — platform, compute capability, resolved
  compute dtype — on the one rig whose entire premise is which dtype the device
  picked, plus `quant_mode=auto resolved to ...` and the non-text-tower line. The
  warnings that did get through printed bare, with no level or logger name,
  because `lastResort` has no formatter.

  **The placement of the fix is load-bearing, not the call itself.** The banner
  fires at *import* of `jax_e_model`, which is imported via `jax_engine`, so
  `basicConfig` must precede that import. `force=True` so a dependency that
  configures first cannot win. `test_root_logging_is_configured_before_the_engine_import`
  pins the ordering and `test_uvicorn_does_not_configure_the_root_logger` pins
  the premise, so if uvicorn ever starts adding a root handler the test says so.

- **A 500 left nothing in the journal.** Both handlers raised
  `HTTPException(detail=str(exc))`, which discards the traceback, and logged
  nothing. A per-request JAX OOM — the whole subject of
  `larger-models-on-t4g.md` — was invisible to `get_jax_logs`. Worse, the SSE
  generator body runs *after* the handler returns, so a **streaming** failure was
  outside the handler's `try` entirely: not counted, not logged, just a short
  answer.

- **`req_id` reached nothing.** It existed only inside the response body. There
  is now one flat `key=value` log line per request and an `X-Request-Id` header,
  so a report of "request `chatcmpl-jax-…` was wrong" is resolvable.

- **`pad_len` — the variable that decided the eviction bug — was computed and
  thrown away.** `bucket_s` was used only for the chunk-divisibility test. It is
  now on `GenerationStats`, in `usage`, in the log line, and on
  `tpu_jax_last_pad_tokens` / `tpu_jax_max_pad_tokens`. That is what turns
  `tpu_jax_degenerate_responses_total` from a smoke alarm with no address into a
  diagnosis, and padding at or past the sliding window now warns by name.

- **Nothing identified which payload a process was running.** `_payload_tar_b64`
  was already deterministic, so `_payload_digest()` hashes the payload **file
  contents** (not the tarball — the digest rides inside it, which would be
  circular), ships a `PAYLOAD_SHA` stamp, and the server reports it on `/health`,
  on `X-Build-Id`, and as a label. `deploy_jax_server` now also prints the
  `_payload_root()` it resolved, which silently picks between the working tree
  and the skill snapshot — the 2026-08-24 stale deploy in one line of output.
  **`verify_model_health` compares the two and says `STALE DEPLOY`.**

- **`verify_model_health` broke the rule it exists to enforce.** It gated on
  `text.strip()` — health-checking by testing for a non-empty response, which the
  engineering rules call out by name. It now reads
  `tpu_jax_degenerate_responses_total` either side of its own probe, so the
  verdict is the server's judgement of the full text rather than "not empty".

- **Two silent fallbacks now warn.** `prefill_chunk_size` reverts to one-shot
  prefill when the chunk does not divide the bucket, and `max_new_tokens` is
  clamped against `max_model_len`. Both left no trace; the clamp additionally
  **changes the compiled shape**, since `max_new_tokens` is a `static_argnames`
  entry.

- **SSM discarded its `CommandId` on every failure path**, including timeout —
  where the command is still *running* on the box. It is now logged at issue time
  and carried in both error messages. Output truncation is **detected** against
  the documented 24,000-character cap rather than returned as if complete:
  `get_jax_logs` at `tail=5000` will exceed it, and reading a partial journal is
  how you conclude an error is not there.

- **`_error()` swallowed the traceback across 18 tool bodies.** One
  `logger.exception` inside it covers every call site.

**What the T4G run established**, beyond the CPU pass:

- **The device-policy banner is the first line the process emits**, and it had
  never once appeared in a journal on this rig:
  `INFO ports.gemma4.jax_e_model: jax_e_model device policy: platform=gpu
  compute_capability=7.5 compute_dtype=float16 pallas_interpret=False`.
  Both halves matter — `float16` is the device choosing Turing's only real
  16-bit datapath, and `pallas_interpret=False` is the difference between
  serving and silently running a simulator.
- **Load is now staged**: download 87.7s, read_shards 73.5s (1 shard, 600
  tensors, 0.95 GB of non-text towers skipped), convert_params 3.4s, device_put
  0.0s — 164.7s and 9.26 GB total. A hang is now attributable to a stage.
- **`window_kv=auto resolved to True (sliding_window=512, max_model_len=8192)`**
  — the flag implicated in the eviction bug, stated rather than inferred.
- **The whole resolved configuration is one greppable line**: `READY
  build_id=6852f5680f43 … compute_dtype=float16 kv_cache_dtype=float16
  kv_cache_requested=auto pre_ampere=True quant_mode=fp16 window_kv=True`.
- **Build id matched end to end** — deploy reported `6852f5680f43` and the
  payload root it resolved, `PAYLOAD_SHA` landed in `/opt/jax-g5g/app`, and
  `verify_model_health` confirmed the served id against the local digest.
- **SSM truncation detection fires for real**: a deliberate 117 KB command came
  back cut at 24,000 characters with the notice and the command-id recovery
  line attached.
- **The instrumentation is performance-neutral.**
  `tpu_jax_decode_tokens_per_second` read **12.30**, against the 12.4/12.5 of the
  two recorded runs. HBM 9.30 GB used of the 14.07 GB limit, weights 9.257 GB.

**Cold/warm reproduced on the real chip**: the same prompt at the same
`max_tokens` took **18.77s cold, then 4.35s and 4.33s warm** — matching the
18.06s/4.50s measured 2026-08-21. `tpu_jax_cold_requests_total` counted the two
cold shapes, and `get_metrics` correctly refused to let the 5.45 tok/s
cumulative figure pass as a result while they were in it.

Two things worth keeping from the CPU verification:

- **`cold_shape` is worth its own field.** The same request shape measured
  **17.7 tok/s cold and 439.6 tok/s warm** — a 25x gap that was previously an
  unexplained outlier averaged into the cumulative mean.
- **The clamp really does recompile.** A request clamped from 9,999 to 236
  came back `clamped=True cold=True` — the connection between the silent clamp
  and a mystery slow request, observed rather than reasoned about.

Metrics gained a **`rig` label** rather than a renamed prefix: two rigs serving
the same checkpoint previously emitted byte-identical series names *and* label
sets, but both benchmark reports compare on `tpu_jax_decode_tokens_per_second`
**by name**, so renaming would break continuity with them. `RIG_NAME` reaches the
serving process through the systemd `EnvironmentFile`. `tpu_jax_decode_seconds_total`
also lands, so cumulative decode is a real rate instead of the lower bound
`get_metrics` used to apologise for.

`tests/test_server.py::ObservabilityTests` pins all of it.

## 87% of decode is dtype tax, and nothing else is close

**`benchmarks/runs/2026-08-25-context-sweep-g5g/` — MEASURED, with the first xprof profile
taken on this rig.** This is the single most important number here and it subsumes most
optimization ideas you might arrive at independently.

| Kernel class | Time | Share |
| --- | ---: | ---: |
| dtype conversion (`wrapped_convert_*`) | 811.4 ms | **54.4%** |
| fp32 `gemvx` | 486.2 ms | **32.6%** |
| reduce fusions | 167.2 ms | 11.2% |
| everything else | 27.2 ms | 1.8% |

The cause is already written up in `docs/bf16-weights-on-turing.md` and this confirms it from
the other side: the loader stores all 540 float parameters as **bfloat16** while
`COMPUTE_DTYPE` is **float16**, so XLA converts in front of every use — and the matmuls that
remain run as **fp32 GEMV**.

Three corroborating facts, each of which kills a plausible-sounding alternative theory:

- **Decode does not degrade with context.** 12.80 tok/s mean, **3.4% total spread** across a
  100× context range (41 → 4,105 tokens) and a 4× output range. A cost proportional to the
  *weights* rather than the context produces exactly that. If KV were binding, decode would
  fall as context grew. It does not, so **the KV cache is not what sets decode speed here.**
- **No kernel used a TensorCore** — `Kernel uses TensorCore` is `False` for **100.0%** of
  kernel time, on a chip with 65.1 TFLOP/s of fp16 tensor-core throughput. Do **not** size an
  expected win off that peak: at `B=1` decode is a matrix-*vector* product, bandwidth-bound by
  nature, and tensor cores need matrix-matrix work to pay.
- **Prefill is linear in the PADDED bucket**, not the real prompt: `prefill_ms = 1.478 ×
  bucket − 101`, **R² = 0.997**. TTFT is a bucket property, which is why the bucket ladder is
  a latency knob and not just a correctness one.

**The untried fix is a bit-shift, and it is the highest-value work available here.**
`docs/bf16-weights-on-turing.md` records three placements tried and rejected on hardware. The
direction it flags as untried is straightforward: **bf16 → float32 is a 16-bit left shift**
(`u16.astype(uint32) << 16` viewed as `float32`), pure NumPy and fully vectorised, and
float32 → float16 is a native vectorised cast. That sidesteps `ml_dtypes`' unvectorised
element-wise path entirely — the thing that made E2B's 4.7 GB table not finish in 10 minutes
on Graviton2. Do it per shard at load, so no bf16 source outlives its converted copy.

**Fragmentation, not peak, is the memory constraint.** xprof `GPU_0_bfc` at peak: 10.171 GiB
in use, 2.937 GiB free, **fragmentation 0.661**. The free two-thirds is not contiguous, which
is the same condition behind the `device_put` failures in `docs/bf16-weights-on-turing.md`.
**Any capacity claim on this rig should quote the largest contiguous block, not free bytes** —
two of the three quantization bugs below failed with GBs nominally free.

## The quantization levers: broken three ways, then fixed and measured

**`2026-08-26-config-sweep-g5g` found that none of the three levers could load;
`2026-08-26-quant-levers-fixed-g5g` fixed all three and measured them. Both are MEASURED.**

| Config | Weights | vs base | Load s | Decode tok/s (128 / 1024 / 4096) |
| --- | ---: | ---: | ---: | --- |
| `ple0` — the old default | 9.257 GB | — | 184.3 | 12.80 / 12.77 / 12.60 |
| `ple8` | 6.927 GB | −2.330 | 80.5 | 12.80 / 12.70 / 12.60 |
| `ple4` | 5.752 GB | −3.505 | 126.8 | 12.80 / 12.70 / 12.60 |
| `ple0+int8head` | 9.660 GB | +0.403 | 94.8 | 13.10 / 13.00 / 12.80 |
| **`ple4+int8head` — the current default** | **6.155 GB** | **−3.102** | 95.3 | **13.10 / 13.00 / 12.80** |

**`tpu.env` changed on 2026-08-26 to `PLE_BITS=4` / `INT8_LM_HEAD=1`.** It is strictly better
than the old default on both axes — 33% less memory and +2.3% throughput — but
**`INT8_LM_HEAD` is not numerics-preserving (~0.8% logit error)**, so it is a deliberate trade
recorded as such, not a free win.

Four things worth keeping:

- **PLE is memory-only, exactly as the port's comments predicted.** Decode is identical across
  `ple0`/`ple8`/`ple4`: the table is a gather, never a matmul, so decode never streams it.
  Every prediction in `jax_e_model.py` landed — `ple_bits=8` → −2.330 GB against −2.35
  predicted (0.8%), `ple_bits=4` → −3.505 against −3.51 (0.1%), `int8_lm_head` +0.403 GB
  exact.
- **`int8_lm_head` does not do an int8 matmul.** xprof says the conversion kernel shrank 11%
  rather than disappearing, because `jax_e_model.py` dequantizes the int8 table to fp16 **in
  full — 0.75 GiB — on every decode step** and then runs the same matmul. It halves the bytes
  *read* and pays a full-table convert regardless. That is the entire +2.3%. **Turing has int8
  tensor cores (~130 TOPS) and this path never touches them** — a genuine int8 matmul is the
  unexploited win, not a larger PLE. Same shape as the W4A16 result: what is labelled
  quantized execution is really **dequantize-then-matmul**.
- **PLE quantization makes loads FASTER** — 80–127 s against 184 s baseline. Less to place on
  the device outweighs the host-side quantization cost.
- **Two of the three bugs were one pattern**: allocate the destination before releasing the
  source, on a device whose free memory is 66% fragmented. `quantize_lm_head` upcast
  `[262144, 1536]` to float32 **on device** (1.50 GiB, exact) when the correct host-side
  chunked pattern was 1,200 lines up in the same file; `quantize_ple_table` placed the int8
  copy (2.19 GiB, exact) while the 4.38 GiB bf16 original was still resident. The third was
  the swap gate. **`release_source` is opt-in deliberately** — the first version deleted
  unconditionally and a CPU test caught it in seconds, because `.delete()` invalidates the
  *caller's* array. A test asserts `load()` actually opts in, or the fix would sit in the tree
  doing nothing.

## `MAX_MODEL_LEN` is 4096 because 8192 was not reachable

**MEASURED 2026-08-26.** Lowered from 8192, and the old value was not an honest number — a
little over half of it was reachable.

| prompt tokens | status | failed allocation |
| ---: | --- | ---: |
| 4,105 | ok | — |
| 5,120 | **infeasible** | 2.59 GiB |
| 6,144 | **infeasible** | 3.48 GiB |
| 7,800 | **infeasible** | 5.11 GiB |

**The prefill transient has a flat term AND a linear one, and both prior measurements were
right.** `docs/bf16-weights-on-turing.md` measured it as FLAT in the bucket (1.504 GiB at 512
and at 1,536, 1.742 GiB at 4,096) — that measurement was taken entirely inside the flat
region. Above ~4K a linear term at roughly **0.9 MiB/token** takes over. Do not treat either
as the whole story.

**The documented remedy is structurally unreachable.** `PREFILL_CHUNK_SIZE` exists precisely
to bound prefill temporaries, and `jax_e_model.py` raises `prefill_chunk_size requires
window_kv=False`. `window_kv` auto-resolves to **True** whenever `max_model_len >
sliding_window` (8192 > 512, and 4096 > 512), so setting `PREFILL_CHUNK_SIZE` raises at
startup. The only route to chunked prefill is `window_kv=off`, which is **untested here**.
The one mitigation for the ceiling is gated behind an untested flag.

## Tracing a deployment

**Added 2026-08-27, after the `mkswap` incident. Verified offline; not yet exercised on a
launch.**

The provisioning path had two blind spots, and the first one is why a one-character flag cost
a launch rather than a minute.

- **`get_install_progress` could not tell a dead bootstrap from a slow one.** Cloud-init
  writes `install.sh` and *backgrounds* it, so anything that kills cloud-init before that
  point leaves no install log — and the tool rendered that as `INSTALL IN PROGRESS` +
  `no install log yet`, indefinitely, which is also exactly what a healthy slow install looks
  like. It now reports `cloud-init status --long`, tails `cloud-init-output.log` when the
  install log is absent, and returns a verdict separating **cloud-init error** /
  **done-but-never-started** / **genuinely-still-booting**. The failing states say `NOT a slow
  install` in as many words, because that inference is the one that was not being made.
- **The install was the longest phase of a deploy and the only untimed one.** The model load
  reports four stages; this reported none, so the 2026-08-25 spot reclamation at 21 minutes
  left no record of which step was running. `install.sh` now emits `[stage] <name> +Ns` around
  apt, the interpreter, pip bootstrap, the jax wheels, the serving deps and the GPU assertion.
  Grep with `grep -F '[stage]' /var/log/jax-install.log`.

Two optimizations landed with them:

- **The root volume was left at gp3's 125 MiB/s default and the load sat on it.** MEASURED
  2026-08-25: `read_shards` moved the checkpoint in 73.5 s (~139 MB/s) and the download took
  87.7 s (~116 MB/s). **Two unrelated stages landing on one number is a volume ceiling**, not
  CPU and not network. Now 500 MiB/s / 6000 IOPS — ~4× baseline, still under `g5g.2xlarge`'s
  own EBS cap ("up to" 4.75 Gbps ≈ 593 MB/s) so the smaller sizes stay instance-bound, and
  satisfying gp3's `throughput <= IOPS × 0.25` rule, which is enforced at run-instances time
  and so fails a *launch* rather than slowing a disk. **UNTESTED as a remedy** — the load
  stages are already timed, so one launch settles it.
- **The XLA compilation cache can now survive a relaunch**, via `JAX_CACHE_S3_URI`. It lives
  on the ephemeral root volume, so every relaunch recompiles every shape — and the 128-step
  bucket ladder means far more distinct shapes than the old power-of-two one, at ~14 s each
  (18.77 s cold against 4.35 s warm). **Empty by default**, so the default rendering is
  byte-identical to what the rig shipped before; opt in with an operator-supplied URI, in the
  same spirit as the required subnet/security-group/instance-profile ids. Upload is on a
  **timer, not `ExecStopPost`**: spot gives ~2 minutes and does not reliably run shutdown
  hooks. The restore carries `|| true` because it runs under `set -e` and the first launch
  syncs a prefix that does not exist yet — the `mkswap` failure exactly.

**`get_deployment_config` printed `VolumeSize=200` while `create_g5g_instance` launched 100.**
Both now render from one set of constants. A copy-pasteable repro command that provisions a
different volume from the tool it documents is how a manual reproduction fails to reproduce.

**Still open: there is no prebuilt AMI, and the argument against one has weakened.**
`CLAUDE.md` has always argued a stock DLAMI is right here because this install is `pip
install`, not the vLLM sibling's 67-minute build. That reasoning is sound *against that
comparison*. But the 2026-08-25 reclamation killed an instance at 21 minutes, **before the
wheels finished** — so on spot this rig currently cannot reliably survive its own install.
Baking the install into an AMI is what would make spot usable at all. That is a decision, not
a cleanup, and it has not been taken.

## Measurement

**This rig has five measurements**, all its own, in `benchmarks/runs/<date>-<what>-g5g/`
where `<hw-short>` equals the hardware slot:

| Run | Decode | What it added |
| --- | ---: | --- |
| `2026-08-19-first-serve-g5g` | 12.5 tok/s | First serve. CUDA 12 / Python 3.12. |
| `2026-08-21-cuda13-py314-g5g` | 12.4 tok/s | CUDA 13 / Python 3.14, same AMI. |
| `2026-08-25-context-sweep-g5g` | 12.80 tok/s | 12 cells, context × output. First xprof. |
| `2026-08-26-config-sweep-g5g` | 12.8 tok/s | Config sweep: all three levers fail. Locates the prefill ceiling. |
| `2026-08-26-quant-levers-fixed-g5g` | **13.10 tok/s** | Levers fixed. 5/5 configs, 15/15 cells. Sets the current default. |

**The 12.4/12.5/12.8 figures are all the same configuration** (`ple0`, no int8 head) on
successive stacks, and they are within noise of each other. **13.10 is a different
configuration**, not an improvement to the old one — see the quantization-lever section.

**The CUDA 13 / 3.14 bump is performance-neutral** — it buys currency, not speed. Compare the
two on the `tpu_jax_decode_tokens_per_second` gauge, not end-to-end tok/s: the same prompt
returned 64 completion tokens in one run and 53 in the other, and end-to-end wall includes
prefill and HTTP, so the token count moves it.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them, so edit the root originals, never
these. `reports/` and `runs/` stay in the rig.

Three numbers you will be tempted to reuse, and must not:

- **43.1 / 44.24 tok/s** — the vLLM sibling on `g5g.4xlarge` / `g5g.xlarge`, 2026-08-12/13.
  Same silicon, different runtime, and the figure was obtained *with reduced Triton tiles*.
  It is the number this rig exists to beat, not a baseline it inherits.
- **~44 tok/s on one Inferentia core** from `~/gemma4-tips-aws` — different harness, different
  silicon.
- **Anything from `~/gemma4-tips`** — that tree duplicated its own artifacts and its directory
  names misattribute both model and chip. Never read a model or a chip off one.

A config flag being accepted is not evidence it did anything. Cross-check against an absolute
physical bound — 320 GB/s of GDDR6 and 15360 MiB is the whole envelope here — not against
another config.

## Fork debris — cleared, and why the list stays

This rig was forked from `gpu-vllm-g5g-2b` and the code was rewritten before the prose was.
**The backlog is empty as of 2026-08-25**: registration files repaired 2026-08-18, and
`caf89a2` retitled `README.md`, rewrote `AGENTS.md` and `GEMINI.md` (they are no longer
byte-identical vLLM copies), and gave the monorepo `NAMING.md` and `README.md` their
`gpu-jax-g5g-2b` entries.

The list stays because the *class* of error keeps recurring and the recurrences are what
matter — TPU-rig or vLLM-rig prose describing precision this chip cannot run, in a rig whose
whole premise is which precision the device picked. Cleared 2026-08-23:

- `jax_openai_server.py`'s module docstring claimed "TPU v6e-1", the `-qat-w4a16-ct`
  checkpoint and BF16 activations. It now points at `/health` and the
  `tpu_jax_precision_info` series instead of naming a precision at all.
- The load banner printed **"Loading W4A16 QAT weights"** unconditionally while loading the
  dense fp16 checkpoint — so `get_jax_logs` told an operator the box had done the one thing
  this rig refuses to do.
- `/health` reported `weights="bf16"` and a hardcoded `activations="bfloat16"` on a chip with
  no bf16 datapath, and echoed the *requested* KV dtype (`auto`) rather than what it resolved
  to. All three now come off `ENGINE.precision_info()`.
- `make lint`'s `B023` in `tests/test_engine.py:42` is fixed; the gate is green.
