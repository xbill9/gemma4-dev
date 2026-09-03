# CLAUDE.md — `gpu-jax-g4dn-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **pure JAX** on **AWS EC2 G4dn** — an x86_64
Intel Cascade Lake host with an **NVIDIA T4** GPU (Turing, SM 7.5, **15360 MiB measured**, not
the nominal 16 GB `describe_instance_types` reports).

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`.
It is **not** one of the `gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform
slot with them.

"Pure JAX" is literal: no PyTorch, no torch_xla, no vLLM. The engine is this repo's own
Gemma 4 port (`ports/gemma4/`) driven by `jax_engine.py` behind an OpenAI-compatible FastAPI
server (`jax_openai_server.py`), run under systemd — **not docker**.

## This rig has served, and it was forked from a rig that had served more

**Forked from `gpu-jax-g5g-2b` on 2026-08-28. FIRST SERVE 2026-08-29**, on spot
`i-02eb43cb99429a8f7` (`g4dn.xlarge`, `us-east-1b`), launched → deployed → measured →
terminated in one ~25-minute cycle. One run so far:
`benchmarks/runs/2026-08-29-first-serve-g4dn/`, driven end to end by `tune_loop.py`.

**READ THE PROVENANCE RULE BEFORE ANY NUMBER IN THIS FILE.** Most of the findings below were
measured by the **G5g sibling on a T4G**, not here, and this file inherited them written as
though they were this rig's. Until 2026-08-29 `server.py`'s own module docstring asserted
"THIS RIG HAS SERVED, AND THE NUMBERS ARE ITS OWN: 12.4-12.5 tok/s decode on g4dn.xlarge …
across the two runs under benchmarks/runs/" while `benchmarks/runs/` was **empty**. That is
exactly the hazard the monorepo `CLAUDE.md` names — "benchmark JSON travelled with the forks"
— and it was corrected the same day. The split to hold onto:

- **Chip-level findings carry, and are now corroborated here.** T4 and T4G are both Turing
  SM 7.5 with the same 15360 MiB and the same roofline.
- **Host-level ones do not automatically.** Install time, load staging and host RSS depend on
  a CPU this rig does not share with the sibling.

### The A/B came back clean

This rig exists to change exactly one variable against the sibling — same engine, same
checkpoint, same dtype policy, same SM 7.5, **different host ISA**. The result:

| | `gpu-jax-g5g-2b` (Graviton2 + T4G) | **this rig (x86_64 + T4)** |
| --- | ---: | ---: |
| decode, gauge | 13.10 tok/s | **13.1 / 13.2 / 13.1** |
| `tpu_jax_weight_bytes` | 6.155 GB | **6.155 GB** — exact |
| dtype conversion | 54.4% / 54.0% | **54.4%** |
| fp32 GEMV | 32.6% / 32.8% | **32.8%** |
| TensorCore utilisation | 0.0% | **0.0%** |
| peak FLOP / HBM / ridge | 65126.4 / 298.083 / 203.479 | **identical to 3 d.p.** |
| device memory (`nvidia-smi`) | 15360 MiB | **15360 MiB** |

**The host contributes nothing measurable to decode.** That converts the sibling's central
result — 86.9% of decode in dtype conversion plus fp32 GEMV — from "a finding about a
Graviton2+T4G box" into **a finding about Turing**, which is the whole reason this rig was
built. Everything in *High-value optimizations* below therefore applies here unchanged,
including the 2026-08-28 falsification: at `B=1` decode is a matrix-*vector* product, cuBLAS
dispatches an all-fp32 `gemvx` kernel with **no half-precision path**, and no storage dtype
removes it. The route to the ceiling is a GEMM — batching — which `MAX_NUM_SEQS=1` closes.

**One inherited prediction was falsified.** `README.md` claimed the T4 reports a full
16,384 MiB against the T4G's 15,360 — "a real 1 GiB difference … the kind of thing that
decides whether a configuration fits". That came from `describe_instance_types`, which
reports the nominal figure. `nvidia-smi` on the device reports `Tesla T4, 7.5, 15360 MiB`.
**There is no capacity advantage.** The device is what allocates, so the device is what
counts.

Two things that still hold and are easy to lose:

- **The grid is thin in one axis and closed in another.** This rig has swept context only
  (3 cells: 41 / 521 / 2,057 input tokens at 64 output). **Concurrency is not an axis and
  cannot be**: `MAX_NUM_SEQS=1`, `Gemma4EModelJAX` raises `NotImplementedError` for `B > 1`,
  and the decode step donates its KV buffers. The report records that cell as `infeasible`
  rather than omitting it. `tpu.env` marks values MEASURED or PREDICTED; respect the split.
- **Installability is not a served token** — and on this rig the installability argument is
  much weaker than it was on the sibling. `jaxlib` and the CUDA plugins publishing wheels
  mattered on **aarch64**, where a missing wheel was a real hazard. x86_64 + CUDA is the axis
  every one of these projects publishes for first. **The load-bearing claim was always
  `sm_75`**, not the architecture.

`verify_gpu_arch` remains the cheapest way to convert the install into evidence — it runs a
real fp16 matmul on the device rather than checking that a flag was accepted. It reported
`Tesla T4, 7.5, 15360 MiB … fp16 matmul ok: True` on 2026-08-29.

**Warm up before recording anything.** Cold prefill measured **56x** slower than warm on the
sibling 2026-08-19; a harness that skips warm-up understates this rig by more than 2x on
decode. `tune_loop.py` warms at the shape it measures, because `max_new_tokens` is a
`static_argnames` entry and warming at a different one was previously a 4x error.

## Why the hardware slot is `g4dn`, and why the sibling's is not `t4g`

Slot 3's normal rule is the GPU SKU. Here the two agree closely enough that **no carve-out is
needed**: the GPU is an NVIDIA **T4**, and `t4` is not an EC2 instance family. The family name
`g4dn` is still the better slot value, because it also pins the x86_64 host.

**Do not generalise that to the sibling.** `gpu-jax-g5g-2b` *needs* its carve-out (settled
2026-08-12, recorded in `NAMING.md`) for a reason specific to it: its GPU is a **T4G**, and
`t4g` is already an EC2 instance family — `t4g.nano`…`t4g.2xlarge`, Graviton2 burstable **CPU**
boxes with no GPU — so in an AWS context that string reads as a cheap CPU instance far more
often than as a GPU.

The chip is called a T4 everywhere it is the chip being discussed — in this file, in
`HARDWARE.md`, and in `tpu.env`. Only the slot is `g4dn`.

## Why this rig exists next to `gpu-jax-g5g-2b` and `gpu-vllm-g5g-2b`

Two different comparisons, and they are not the same argument.

**Against `gpu-jax-g5g-2b`: same runtime, different host.** That is the A/B above, and it is
this rig's actual purpose. It came back clean, so the sibling's numbers are chip numbers.

**Against `gpu-vllm-g5g-2b`: same chip generation, different runtime — and the runtime is the
whole point.**

The vLLM path gets to a served token, but only through a ~67-minute from-source build for
SM 7.5, a CUDA toolkit the DLAMI does not ship, a Rust toolchain, and an **unlanded patch to
Triton's attention kernel** that lives on one instance and has to be reapplied after every
upgrade. That last one is not a packaging problem: Gemma 4's heterogeneous head dims
(sliding 256, global **512**) force `TRITON_ATTN`, whose 512-wide tile wants ~96 KiB of
shared memory per block against Turing's 64 KiB ceiling.

JAX sidesteps all four:

- **No build.** pip supplies CUDA; the DLAMI only has to supply the driver. Measured here
  2026-08-29: **76 seconds** from boot to a GPU-verified install, xprof and tensorboard
  included.
- **No toolkit and no Rust.** Nothing is compiled on the instance.
- **No arch gap.** The plugin's cubins cover SM 7.5 already.
- **No patch to carry.** Attention here is ordinary XLA, not a hand-tiled Triton kernel, so
  there is no per-block shared-memory ceiling in the attention path to hit.

**`../gpu-jax-g5g-2b/docs/turing-aarch64-gap.md`** is the vLLM-side write-up of all of this
and is **measured**. It is **not in this rig** — it was deliberately not inherited, because
only half of it applies: the Turing shared-memory analysis carries, the aarch64 packaging half
does not. Read it in the parent, and read only that half.

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
  read on a CPU host reports `tpu-v6e-1`, not a T4. The tests exercise the Turing branch by
  overriding the detected platform under `JAX_PLATFORMS=cpu` — that tests the *policy*, not
  the hardware.
- Fixes that describe the **model** belong in the root `MODELS.md`; fixes that describe the
  **chip** belong in `HARDWARE.md`. Only a measurement stays in this rig.

## Instance sizing, and the swapfile

`g4dn.xlarge` is the default. **Every size is supported** — `_validate_instance_type` only
enforces the size list.

The size ladder is **not monotonic in GPU count** and must be read from
`describe_instance_types`, never inferred from the suffix: `g4dn.12xlarge` has **four** T4s
and `g4dn.16xlarge` has **one**. `_G4DN_SIZES` in `server.py` is the table.

`_user_data` provisions a swapfile at or below `_SWAP_AT_OR_BELOW_HOST_RAM_GB = 16`. On
**g4dn that is the default `g4dn.xlarge` alone** — every larger size has at least 32 GiB. That
differs from the G5g sibling, where the same threshold covered two sizes, so do not carry over
"the two smallest hosts" as a description here.

**The threshold is inherited from the sibling, not re-measured on this hardware**, and it is
worth knowing why it is where it is. Two distinct host-RAM pressures, both measured there
against the same checkpoint, neither of them GPU-specific:

- **8 GiB cannot even mmap the checkpoint** (below).
- **16 GiB mmaps fine and then dies in `quantize_ple_table`**, which upcasts the 4.70 GB PLE
  table to float32 while the full tree is resident. MEASURED 2026-08-26: OOM-killed five times
  at 14.3 GB anon-RSS under `Restart=on-failure`. The gate was `< 16` and that host had
  *exactly* 16, so the rig provisioned swap for the one size that did not need it for this and
  skipped the one that did. **The threshold was the bug, not the remedy** — the
  `fallocate`/`mkswap`/`swapon` block already existed.

`g4dn.xlarge` has exactly 16 GiB and so lands in the second case. CONFIRMED 2026-08-29: it
loaded cleanly with the swapfile in place, READY 151 s after the service started.

**Making the threshold inclusive immediately exposed a second bug**, which is worth keeping as
a pattern: the swap block had used `mkswap -q`, a **busybox** flag that util-linux rejects
with `invalid option -- 'q'`. Under `set -e` that killed cloud-init *before install.sh was
written*, so the instance booted with an empty `APP_DIR` and no install log at all. It was
latent for as long as only the smallest size rendered the block and nobody launched one.
Two lessons: **a code path that only renders for a size nobody launches is untested code**,
and the reason this cost a launch rather than a minute is covered under **Tracing a
deployment** below. `test_swap_block_uses_only_portable_flags` now pins it.

The mmap failure, measured on the vLLM sibling 2026-08-13 with the same checkpoint, so it is
a checkpoint-and-RAM property rather than a host-ISA one: without swap the kernel refuses to
**mmap** the 10.2 GB checkpoint at all —

```
RuntimeError: unable to mmap 10246621918 bytes from model.safetensors:
Cannot allocate memory (12)
```

— and systemd crash-loops on it. The failure is the *mapping*, not residency; 16 GiB of swap
took the same instance to a healthy endpoint. **The sibling once rejected its 8 GiB size
outright on the theory that it "cannot stage 9.5 GiB of weights". The conclusion was right and
the reason was wrong**, and the remedy is swap rather than a bigger instance — the same fix
`tpu-pytorch-inf2-2b` applies for its neff load.

`g4dn.12xlarge` carries four T4s and `g4dn.metal` eight. **Nothing shards across them.** The
engine is single-device (`jax.devices()[0]`), `_serve_argv` emits no tensor-parallel flag, and
the payload contains no sharding primitives at all, so the extra GPUs idle.
`_tensor_parallel_size` reports the GPU count and nothing acts on it yet. **A bigger instance
buys host RAM and network, not device memory** — the budget is the same 15360 MiB on every
size.

## AMI resolution

Two requirements, and they are separate:

1. **x86_64** — pinned by the `/x86_64/` segment of the SSM path and by the `Architecture`
   filter in `_resolve_ami`'s describe-images fallback.
2. **The NVIDIA driver** — AWS ships plenty of DLAMIs without one. Such an image boots
   perfectly well on a G4dn and simply has no GPU visible to JAX, which reads as a broken
   runtime rather than a wrong AMI. The bootstrap's `verify_gpu` step is what turns that into
   an install-time failure instead of a first-token one.

`DLAMI_SSM_PARAMETER` pins both and is single-valued, so it is preferred; the `DLAMI_NAME`
describe-images filter is the fallback and deliberately requires the driver in the name.

**Never hardcode an AMI id** — resolve it at launch. The architecture half of that rule is
weaker here than on the G5g sibling, where a hardcoded x86_64 id could not boot at all; the
half that still bites is **rot**, which the next section is about. Resolved
`ami-0216c4aa131462acf` on 2026-08-29.

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

**VERIFIED on hardware 2026-08-27** (`i-021f15b2b45e13793`,
`benchmarks/runs/2026-08-27-ubuntu2604-base-g5g/`). All three risks settled: driver
**595.91.07**, well past `jax[cuda13]`'s 580+ floor; **aws-cli 2.36.30 present** and the
HF token written; and **PEP 668 genuinely in force** —
`/usr/lib/python3.14/EXTERNALLY-MANAGED` exists, so `--break-system-packages` was
load-bearing, not defensive. glibc is **2.43** against 2.35, so xprof's `manylinux_2_35`
floor now has headroom instead of sitting exactly on it. Falling back is still one env var —
the 24.04 base works via deadsnakes.

**The install went from >21 minutes to ~80 seconds**, and `python-3.14` is **1 second** of it
because the base image already carries the interpreter. Decode measured **12.80 tok/s**,
inside the 12.4–13.1 band of every prior run: this buys currency and install time, **not**
speed.

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
create_g4dn_instance → get_install_progress → verify_gpu_arch → deploy_jax_server
                     → get_jax_logs → verify_model_health
```

Install progress goes to `/var/log/jax-install.log`; `{APP_DIR}/INSTALL_DONE` appears only
after JAX **imports and sees the GPU**, so "INSTALL COMPLETE" is an assertion, not a guess.
The unit is `jax-g4dn.service` and `APP_DIR` is `/opt/jax-g4dn` — read logs with
`journalctl`, not `docker logs`.

**Stage timings, MEASURED 2026-08-29** (`grep -F '[stage]' /var/log/jax-install.log`):
apt-base 6s, python-3.14 **1s** (already the system interpreter, so deadsnakes was skipped),
pip-bootstrap 6s, jax-wheels 42s, serving-deps 13s, **profiling-deps 5s**, gpu-verify 3s,
unit-rewrite 0s — **76 seconds total**. The load then took 151 s from service start to READY.

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
  network or IAM policy — with **one deliberate exception, recorded below**. (The legacy
  sample this was scaffolded from auto-creates a security
  group open to `0.0.0.0/0` — that was not carried over.)
- Scope instance discovery to `ManagedBy=gpu-jax-g4dn-2b`. Unlike the inf2 rig, which keeps a
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
- **The shared S3 policy is deliberately LOOSE, and must stay that way.** The instance role
  (`gpu-jax-g5g-2b-instance-role`, shared across the GPU JAX rigs) carries an inline policy
  `jax-compilation-cache-rw` granting `s3:{GetObject,PutObject,DeleteObject}` on
  `vllm-models-bucket/jax-cache/*` and `vllm-models-bucket/benchmarks/*` — **the rig segment
  is a wildcard, on purpose**.

  It used to enumerate rigs (`jax-cache/gpu-jax-g5g-2b/*`, and likewise for `g6`), which meant
  **every fork was denied its own cache and results until someone edited IAM**. That is
  exactly what happened here on 2026-08-29: `tune_loop.py`'s artifact upload — the supported
  way to get a profile back, because SSM truncates at 24,000 characters — failed for a rig
  that was otherwise complete. Operator decision, same day: widen it rather than extend it.

  **Do not "tighten" this back to per-rig prefixes.** It reads like a hardening win and it is
  a regression: it reintroduces an IAM edit into the path of standing up a rig, and the
  failure it produces (a denied upload, mid-benchmark) is far from the cause. The bucket
  boundary and the two-prefix scope are the actual controls and neither moved.
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

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v` (**140
tests**, all passing as of 2026-08-29). They are fully offline — no AWS, no network, no GPU —
and pin the facts above: the Turing dtype constraints, the x86_64+driver AMI filter, the
host-RAM floor, the shared-memory ceiling, that the token never reaches user data, that
`tpu.env` and `server.py` still agree, that both requirements mirror files match their tuples
in `server.py`, that the profiler installs but never on the serving list, and that no
`VLLM_*`/`TORCH_CUDA*` key survived the fork.

`make lint` runs `ruff check server.py refresh_skill.py jax_engine.py jax_openai_server.py
profile_decode.py profile_prefill.py tune_loop.py tests`, then `bash -n` on **four** shell
scripts (`project-setup.sh`, `init.sh`, `set_env.sh`, `save-aws-creds.sh`). **A new top-level
module is silently unlinted until it is added to that list** — `profile_decode.py` sat outside
it and was red for a day. `ports/` is excluded on purpose — see above.

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
recreate it. So `rm -rf .claude/skills` destroys it permanently, which is what happened on the
sibling during its `t4g`→`g5g` rename. `test_skill_is_complete_in_both_copies` now guards both
copies, and also fails if any of the eight generated files is stale.

There is no `make deploy` recipe on purpose: provisioning resolves an x86_64 AMI at launch
time, and a Makefile would have to hardcode one. The target exists and prints that.

## MCP registration lives in four places

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and
`.claude/settings.local.json`'s `enabledMcpjsonServers`. All four must name the server
`gpu-jax-g4dn-2b`, which prefixes every tool as `mcp__gpu-jax-g4dn-2b__…`. A mismatch makes
`/mcp` and the tool prefix disagree about what this rig is.

**The g4dn fork left `.codex/config.toml` entirely unconverted, and it was invisible.** It
named `gpu-jax-g5g-2b` as the server, pointed at
`.claude/skills/gpu-jax-g5g-2b-management/mcp/server.py` — a path that does not exist here —
set `INSTANCE_TYPE=g5g.2xlarge`, and gated three tool names this rig does not have
(`create_g5g_instance`, `terminate_g5g_instance`, `stop_g5g_instance`). `plugin.json` was
right on names and paths but its description and keywords still said Graviton2/T4G/aarch64.
Both fixed 2026-08-29; all four now agree. **Nothing in the test suite caught this**, for the
same reason as the fork before it: `server.py` was correct throughout.

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

`server.py` was right throughout (`RIG_NAME = "gpu-jax-g4dn-2b"`, asserted by
`test_rig_name_matches_directory`) — which is why the breakage was invisible to the tests.
The same blind spot recurred on the g4dn fork; see above.

Editing any of the four generated-or-copied files means re-running `make skill`:
`project-setup.sh` is one of the eight files snapshotted into both skill copies, and
`test_skill_is_complete_in_both_copies` fails until you do.

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree**, and a convention change has to be
applied to all three by hand. Both were rewritten for this rig on 2026-08-29 — they had been
the G5g rig's copies, describing Graviton2, a T4G and an arm64 AMI requirement.

This rig has no `.claude-plugin/marketplace.json` of its own, which only matters if it is ever
published standalone. The marketplace `/plugin` actually reads is the **monorepo root** copy,
and it gained a `gpu-jax-g4dn-2b` entry when the rig was added.

## How large a model this rig will serve

**`../gpu-jax-g5g-2b/docs/larger-models-on-t4g.md` — measured 2026-08-23. E2B is the ceiling
today.** That file is **in the parent, not here**: the fork deleted it on the theory that this
rig's T4 has a 16,384 MiB budget against the T4G's 15,360, which would have shifted every
threshold in it. **MEASURED 2026-08-29: the T4 reports 15,360 MiB too**, so its arithmetic
transfers unchanged and the deletion was the wrong call. Everything below applies here as
written.

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
  contains no sharding primitives at all, so the extra T4s on `12xlarge`/`metal` idle.
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

**Added 2026-08-27, after the `mkswap` incident. Exercised on a launch the same day —
`benchmarks/runs/2026-08-27-ubuntu2604-base-g5g/`.** The cloud-init reporting and the stage
markers both worked; the *failure* verdicts are still only pinned by CPU tests, because
nothing failed. The `JAX_CACHE_S3_URI` path was exercised separately, and is written up
below — it worked, and it found a bug that had made it pointless.

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
  CPU and not network. Now 500 MiB/s / 6000 IOPS — ~4× baseline, still under `g4dn.xlarge`'s
  own EBS cap ("up to" 4.75 Gbps ≈ 593 MB/s) so the smaller sizes stay instance-bound, and
  satisfying gp3's `throughput <= IOPS × 0.25` rule, which is enforced at run-instances time
  and so fails a *launch* rather than slowing a disk. **CONFIRMED 2026-08-27**: `read_shards`
  went 73.5s → **24.7s**, a clean 3.0x on the same checkpoint and the same read, which happens
  *before* quantization so the config change cannot explain it. The hypothesis was right —
  two unrelated stages had been sitting on one number because the volume was the ceiling.
- **The XLA compilation cache can now survive a relaunch**, via `JAX_CACHE_S3_URI`. It lives
  on the ephemeral root volume, so every relaunch recompiles every shape. **Empty by default**,
  so the default rendering is byte-identical to what the rig shipped before; opt in with an
  operator-supplied URI, in the same spirit as the required
  subnet/security-group/instance-profile ids. Upload is on a **timer, not `ExecStopPost`**:
  spot gives ~2 minutes and does not reliably run shutdown hooks. The restore carries
  `|| true` because it runs under `set -e` and the first launch syncs a prefix that does not
  exist yet — the `mkswap` failure exactly.

  **The timer is CONFIRMED firing unattended, 2026-08-28.** Left alone overnight it took the
  prefix from 413 objects / 2.0 MiB to **805 objects / 9.10 MB**, and AWS then reclaimed the
  instance — precisely the case an `ExecStopPost` hook would have missed.

  **Cross-instance restore CONFIRMED 2026-08-28**: a fresh instance restored **805 files / 12 MB
  in 6 s** (`[stage] cache-restore +6s`) from what the *terminated* 2026-08-27 box had compiled,
  and the timer came up `enabled` unattended. Every leg of the loop — compile, upload, reclaim,
  restore — is now measured.

  **MEASURED 2026-08-27, three A/B pairs: first request 25.62 s → 13.89 s, a 1.8x speedup**
  (ratios 1.7–2.0, against a first-request spread of 18–26 s, so the effect is well clear of
  the noise). Two things it does **not** buy, both worth knowing before quoting it:
  **time-to-ready is unchanged** (it moved both directions across the three pairs; load is
  dominated by the 9.5 GB read and host-side quantization), and **`cold_shape` stays `True`**
  — that flag tracks whether *this process* has seen the shape, not whether XLA compiled, so
  `tpu_jax_cold_requests_total` cannot tell you whether the cache helped. Exercising this is
  what uncovered the cache-dir bug below; **cross-instance restore is still untested.**

**`get_deployment_config` printed `VolumeSize=200` while `create_g5g_instance` launched 100.**
Both now render from one set of constants. A copy-pasteable repro command that provisions a
different volume from the tool it documents is how a manual reproduction fails to reproduce.

**The AMI-bake argument has now WEAKENED AGAIN, in the other direction.** An 80-second
install is not worth baking an image for; the reason a verification cycle could not finish on
spot was the install length, and that is gone. What did NOT change is capacity —
`InsufficientInstanceCapacity` in us-east-1b, 1c and 1d on 2026-08-27, with only 1a
available, and **1a was the most expensive AZ by spot price**, so price is not a usable proxy
for capacity here. Retry across AZs; do not read cheapness as availability.

**Reclamation is highly variable, not reliably fast.** The 2026-08-25 instance was taken at
**21 minutes**; the 2026-08-27 one ran **19.2 hours** before `instance-terminated-no-capacity`
(launched 16:40Z, terminated 2026-08-28 11:51Z), on the same instance type in the same region.
**Neither figure is typical** — quote the range, and checkpoint continuously rather than
sizing work to an assumed lifetime. `sweep.py`, `config_sweep.py` and `tune_loop.py` already
write results per request or per cell for exactly this reason.

**Still open, for the record: there is no prebuilt AMI.**
`CLAUDE.md` has always argued a stock DLAMI is right here because this install is `pip
install`, not the vLLM sibling's 67-minute build. That reasoning is sound *against that
comparison*. But the 2026-08-25 reclamation killed an instance at 21 minutes, **before the
wheels finished** — so on spot this rig currently cannot reliably survive its own install.
Baking the install into an AMI is what would make spot usable at all. That is a decision, not
a cleanup, and it has not been taken.

## The compilation cache was configured, and configured nothing

**FIXED and MEASURED 2026-08-27 on `i-021f15b2b45e13793`.** Found only because the
`JAX_CACHE_S3_URI` sync was actually exercised — nothing about it ever failed.

`ports/gemma4/jax_e_model.py` set the cache directory **unconditionally at import**:

```python
_cache_dir = os.path.expanduser("~/.cache/jax_compilation_cache")
jax.config.update("jax_compilation_cache_dir", _cache_dir)
```

`jax_openai_server.py` resolves `JAX_COMPILATION_CACHE_DIR` and calls the same update at
line 72 — then imports `jax_engine` at line 80, which imports this module. **The later import
wins**, so the port silently overwrote the server's choice on every single start.

The evidence is unambiguous: the systemd unit set `JAX_COMPILATION_CACHE_DIR=/opt/jax-cache`,
`/proc/PID/environ` confirmed the process had it, and `/opt/jax-cache` held **0 files** while
**447 files and 5.1 MB** accumulated under the port's hardcoded path. After the fix, one
request inverts it completely: **413 files / 3.4 MB** in `/opt/jax-cache`, **0** in the
fallback.

Three things worth keeping:

- **`tpu.env` advertised this variable as a working knob since the fork**, with "skips
  recompilation on restart" as its effect. The directory it named was never written to. That
  comment is now corrected *and* carries its own measurement rather than the TPU rig's.
- **It would have made the S3 sync a perfect no-op.** Restore and upload both target the
  configured directory, so the feature would have backed up an empty directory forever and
  reported success on every timer tick. **Both halves working correctly against a path nothing
  writes to** — the same silent-success class as the 2026-08-24 stale deploy, and the reason
  "the flag was accepted" is never evidence here.
- **The siblings are NOT affected, checked rather than assumed.** `tpu-jax-v5e1-2b` and
  `tpu-jax-v6e1-2b` carry the identical three lines, but **neither sets
  `JAX_COMPILATION_CACHE_DIR` anywhere**, so the hardcoded path is simply their effective one.
  They have no knob to ignore.

**Under systemd `HOME` is unset**, so `expanduser("~")` resolves via `pwd` to `/root`. That is
why the files were somewhere non-obvious rather than absent, and why "the cache is not working"
was the wrong first hypothesis — it was working, in the wrong place.

`test_both_modules_resolve_the_cache_dir_the_same_way` pins it. They run in one process and
the later import wins, so agreeing is the only thing that makes the result independent of
import order.

## The tuning loop

**`tune_loop.py` — added 2026-08-27.** One command runs a full iteration against a live
instance: `make skill` + deploy, wait for READY, assert the served build id equals the local
payload digest, sweep, profile, pull artifacts back, and diff against a previous run.

```bash
python3 tune_loop.py --instance i-0123 --label baseline
# ... change the model port ...
python3 tune_loop.py --instance i-0123 --label candidate \
    --compare benchmarks/runs/2026-08-27-baseline-g5g
```

**Every ingredient already existed and none of them composed.** `sweep.py` lived *inside* a
run directory and was copy-pasted per run, so each iteration re-derived its own harness;
`profile_decode.py` sat at the rig root and was driven by hand; the xprof extraction was prose
in `docs/profiling-recipes.md`. Three independent sources of drift between two numbers that
are supposed to be comparable.

Four things it is opinionated about, each because it has cost a measurement here:

- **Deploy what you measure**, then assert the served build id equals the local digest. The
  2026-08-24 stale deploy is one line of output now.
- **Warm at the shape you measure.** `max_new_tokens` is a `static_argnames` entry, so
  `(bucket, max_tokens)` *is* the compiled shape — previously a 4x error.
- **Artifacts return through S3, not SSM.** SSM truncates at 24,000 characters silently, and
  a kernel table exceeds it; a partial JSON is how you conclude a finding is not there.
- **Median, not mean**, over repeats — a spot host gives the occasional slow request and the
  cold/warm gap here is 4x.

The sweep runs **before** the profiler, because `profile_decode.py` needs the GPU to itself
and the service must stop for it.

**Baseline through the loop, 2026-08-27** (`benchmarks/runs/2026-08-27-baseline-g5g/`): 12.8
and 13.0 tok/s gauge, weights 6.155 GB, 0 degenerate, and **convert 53.9% / gemv 32.8%** — an
independent reproduction of the 54.4% / 32.6% dtype tax from a fresh harness.

**`--xprof` pulls the structured rollups back too** (`2026-08-27-baseline-xprof-g5g/`), and
they reproduce every prior xprof finding automatically:

| | 2026-08-25, by hand | 2026-08-27, through the loop |
| --- | ---: | ---: |
| kernel time using a TensorCore | 0.0% | **0.0%** |
| dtype conversion | 54.4% | **54.0%** |
| fp32 gemv | 32.6% | **32.8%** |
| peak FLOP / HBM / ridge | 65,126 / 298.1 / 203.5 | **65,126.4 / 298.083 / 203.479** |

`summarize_xprof()` reads xprof's `{cols, rows}` grid by column name rather than scraping
formatted text, so `is_kernel_using_tensor_core` is read as a flag rather than inferred from a
kernel name.

**Two things broke getting there, both silent, both now fixed in the loop:**

- **`requirements-profiling.txt` was never on the instance.** It is excluded from the deploy
  payload, and nothing else shipped it — so `docs/profiling-recipes.md` told operators to
  `pip install -r <APP_DIR>/requirements-profiling.txt`, **a path that has never existed.**
  xprof "installed" with `Could not open requirements file` and the extraction died on
  `ModuleNotFoundError`, both in logs nobody reads. The loop ships it; the recipe is corrected.

  **SUPERSEDED 2026-08-29 — the profiler now installs at boot.** See *The profiler ships with
  the image* below. `tune_loop.py` still ships the file, which is now belt-and-braces rather
  than the only path.
- **`xspace_to_tool_data` returns `bytes`, not `str`.** Passing it through `json.dumps` raises
  `Object of type bytes is not JSON serializable`, the handler catches it, and you get a
  **0-byte file plus a FAILED line** — a profile that looks captured and is empty.

Current xprof also maps the old `kernel_stats^` names (`Received old tool format`), so the
`^` suffix in `docs/profiling-recipes.md` still works but is no longer what the tool reports.

### First iteration: the bf16 blocker is gone, but the fix is not this one

`docs/bf16-weights-on-turing.md` records `astype(float16)` on `ml_dtypes` bfloat16 as unusably
slow — E2B's 4.70 GB PLE table "did not finish in 10 minutes" on Graviton2, 2026-08-24 — and
names a `view(uint16)` bit-shift as the only untried direction. **Both halves are now false,
re-measured on the same chip** (`benchmarks/runs/2026-08-27-baseline-g5g/bf16_convert_bench.txt`,
numpy 2.5.2 / ml_dtypes 0.6.0):

| path | throughput | 4.70 GB table | bit-identical |
| --- | ---: | ---: | --- |
| direct `astype(float16)` | 1.15 GB/s | 4.1 s | yes |
| **via float32** | **1.91 GB/s** | **2.5 s** | yes |
| `view(uint16)` bit-shift | 1.27 GB/s | 3.7 s | yes |

All three are bit-identical, so it is purely a speed question. **The blocker was a stale
`ml_dtypes`, not a property of Graviton2**, and the "untried" bit-shift is not even the fastest
of the three. Cross-checked on x86_64 the same day, same conclusion.

**Converting at shard load nonetheless FAILED on hardware, and the loop caught it in one
iteration.** Casting every tensor to `COMPUTE_DTYPE` as its shard lands, plus pointing the
loader's default at `COMPUTE_DTYPE`, produced two regressions: `read_shards` went **24.7 s →
79.0 s** (an order of magnitude worse than the 5 s the microbenchmark predicts for 9.26 GB, so
the in-situ cost is not the cast itself), and `quantize_ple_table` then **OOMed allocating
4.38 GiB on device**. Reverted; the tree serves `51bc52c9e2e9`.

**DIAGNOSED 2026-08-28. The 54 s was a device→host round-trip, and none of the three
candidates above was right.**

`jax_engine.load()` uses `from safetensors.flax import load_file`, which returns
`jaxlib._jax.ArrayImpl` **already resident on `CudaDevice(id=0)`** (`memory_kind=device`,
verified on hardware). **The weights never touch the host on this path**, so every line of
`docs/bf16-weights-on-turing.md` reasoning about *where on the host* to cast was answering the
wrong question. The 2026-08-27 code called `np.asarray()` on device arrays; measured D2H here
is **1.25 GB/s**, so the 9.26 GB tree costs ~7.4 s of pure transfer before any conversion.
Removing the host cast returns `read_shards` to **25.6 s** against the 24.7 s baseline —
**the whole regression was the round-trip.**

**The one-line loader change alone is correct and still not sufficient.** Pointing `get_arr`'s
default at `COMPUTE_DTYPE` with no host cast makes it an ordinary on-device `astype`, and
`read_shards` is then normal — but it OOMs on **384 MiB**, because `raw_weights` holds every
bf16 source for the whole conversion while the f16 tree is built beside it. bf16 and float16
are both 2 bytes, so **the finished tree is the same size**; the problem is only that both
exist at once, 9.26 + 9.26 GB against 14.07.

**The remedy is to release each source as its copy is made** — peak becomes
`finished tree + one tensor` instead of `2x tree`. That is the `release_source=True` pattern
`quantize_ple_table` already uses. Two cautions: `.delete()` invalidates the **caller's**
array, so release must be opt-in with a test asserting the caller opts in; and peak is
`9.26 GB + largest tensor`, where the largest is `embed_tokens_per_layer` at **4.375 GiB**
(~13.6 GB against 14.07), so **convert that one early**, while little else is resident.

The dtype tax remains 86.9% of decode and the largest available win.

## High-value optimizations, ranked against the roofline

**Derived 2026-08-28 from `benchmarks/runs/2026-08-27-baseline-xprof-g5g/`.** Ranked by
measured upside, not by plausibility. The whole point of the ordering is that **items 1 and 2
are the same bug**, and everything below them is rounding error until that bug is gone.

The physical envelope first, because every claim below is checked against it. xprof reports
peak HBM **298.1 GiB/s = 320 GB/s**, and `tpu_jax_weight_bytes` is **6.155 GB**. A decode step
that streams the weights once therefore cannot beat:

```
6.155 GB / 320 GB/s = 19.2 ms/step  ->  52 tok/s      (a FLOOR, not a promise)
```

Measured is **73.35 ms/step → 13.6 tok/s**, i.e. **26% of the bandwidth-bound ceiling**. The
other 74% is not physics.

| # | Item | Cost now | Upside | Confidence |
| --- | --- | ---: | --- | --- |
| 1 | ~~bf16→f16 weight conversion~~ | 39.60 ms/step (54.0%) | **FALSIFIED 2026-08-28 — 0.0%** | tested on hardware |
| 2 | fp32 GEMV has **no half path** at `B=1` | 24.08 ms/step (32.8%) | needs a GEMM, not a dtype | measured |
| 3 | 854 fusion launches/step | 8.98 ms/step (12.2%) | ≤ 12% | launch-bound, not measured as fixable |
| 4 | bucket ladder below 512 | ~39% of a short prompt's prefill | TTFT only, not decode | measured 2026-08-25 |
| 5 | a real int8 LM-head matmul | — | +2.3% is all `int8_lm_head` buys today | measured 2026-08-26 |

**1 and 2 were believed to be one fix. They are not, and 1 is not a fix at all.**

**FALSIFIED ON HARDWARE 2026-08-28** (`benchmarks/runs/2026-08-28-f16-weights-g5g/`). The whole
parameter tree was converted to float16 in the loader and re-profiled. Convert and GEMV kernels
came back **identical to the microsecond** — 19.94 vs 19.95, 9.98 vs 9.98, 9.32 vs 9.32,
gemvx 11.90 vs 11.92 — and throughput moved **+0.0%** (12.9 / 13.0 tok/s, unchanged).

**The kernel signature says why.** At `B=1` decode is a matrix-*vector* product and cuBLAS
dispatches

```
gemvx::kernel<int, int, float, float, float, float, ...>   is_op_tensor_core_eligible = False
```

— every template parameter **fp32**. **A GEMV has no half-precision path**, so weights are
promoted to fp32 whatever they are stored as. The `wrapped_convert` kernels are that promotion
**to fp32**; they were never a bf16→float16 fixup.

Three consequences worth carrying:

- **The storage dtype is not a lever here.** bf16 and float16 cost exactly the same.
- **`0.0% TensorCore` is structural, not a misconfiguration.** cuBLAS declares a GEMV
  ineligible; no flag changes that.
- **The route to the 86.9% is a GEMM, not a dtype** — batching so the matmul stops being a
  matrix-vector product. That is currently closed: `MAX_NUM_SEQS=1`, `B > 1` raises
  `NotImplementedError`, and the decode step donates its KV buffers. **That, not the dtype, is
  the thing standing between this rig and its ceiling.**

Three kernels carry the promotion:

```
wrapped_convert_1    19.90 ms/step  27.1%   40 calls/step   (per-layer)
wrapped_convert_3     9.96 ms/step  13.6%   20 calls/step
wrapped_convert_61    9.31 ms/step  12.7%    1 call/step    (the LM head)
```

**The arithmetic still closes, but it now describes a different fix.** 33.75 ms → 29.6 tok/s
and 21.7 ms → 46 tok/s remain the right numbers for *removing the fp32 promotion*; what changed
is that no dtype choice removes it. Reaching them needs the matmul to become a GEMM. The
21.7 ms figure lands 2.5 ms above the 19.2 ms bandwidth floor, which is still the reason to
believe the ceiling is real rather than wishful — and it would put this rig at the vLLM
sibling's 43–44 tok/s.

**Do not spend more effort on the weight dtype.** Three attempts, all measured: a host-side
cast (a device→host round-trip, `read_shards` 24.7 → 79.0 s), a loader-only on-device cast
(OOM at `2x tree`), and a released-source cast that loads cleanly and buys **+0.0%**.

**Do not size any of this off the 65 TFLOP/s peak.** At `B=1` decode is a matrix-*vector*
product and **0.0% of kernel time uses a TensorCore** — correctly, since tensor cores need
matrix-matrix work. The win is removing waste and halving weight traffic, not lighting up
units that cannot help.

**Item 3 is real but small and differently shaped.** 854 fusion launches per step at ~10.5 µs
each is launch-bound rather than compute-bound on a chip whose launch overhead is 5–10 µs.
Worth revisiting only after 1 and 2, when it would be ~40% of a much smaller total.

## The model's quirks, and where they land on Turing

E2B is irregular in four ways that all touch memory or attention, and **the port handles all four
correctly** — verified by reading `ports/gemma4/` against `MODELS.md` on 2026-08-28. This section
records *how*, because the correctness is not obvious from the code and the arithmetic it produces is
the reason this rig's constraint is prefill rather than KV.

### Two attention geometries, not one

**28 sliding layers at `head_dim=256`; 7 full-attention layers at `global_head_dim=512`.** Both keep
8 query heads and **1 KV head**. `global_head_dim` is real and applies to Q, K, V *and* the norms —
not a Q-only field.

The port never assumes one head_dim. `Gemma4EAttentionJAX` picks per layer:

```python
self.head_dim   = config.head_dim if self.is_sliding else config.global_head_dim
self.num_kv_heads = config.num_key_value_heads if self.is_sliding else config.num_global_key_value_heads
```

and carries **two RoPE tables** (`inv_freq_sliding` at `rope_theta`, `inv_freq_global` at
`global_rope_theta`), because the two geometries have different widths *and* different theta.

### Three head mismatches

| Mismatch | Value | How the port avoids it |
| :--- | :--- | :--- |
| Q:KV is **8:1** — full MQA, not GQA | `num_attention_heads=8`, `num_key_value_heads=1` | K/V reshaped to `num_kv_heads`, broadcast at the score |
| Heads **do not tile** hidden_size | `8 x 256 = 2048` vs `hidden_size=1536` | head_dim read from config, never derived |
| `global_head_dim` **applies to K/V** | 512 vs 256 | per-layer selection above |

**Mismatch 2 is the one that bites**, because deriving `head_dim = hidden_size / num_heads` yields
**192** and is wrong everywhere.

**The dataclass defaults are NOT E2B's, and one of them is that exact mistake.**
`Gemma4EConfig` defaults to `hidden_size=2048` — which is `num_attention_heads x head_dim`, the Q
projection's *output* width — against E2B's real **1536**; and to `num_key_value_heads=4` against E2B's
real **1**. Every shape-critical field is `pick()`ed from the checkpoint's `config.json`, so a complete
config is correct. But **a field missing from a config yields a wrong-shaped model rather than an
error.** Do not read these defaults as documentation of E2B.

### KV sharing: 20 of 35 layers, resolving to exactly two caches

`first_kv_shared_layer_idx = 35 - 20 = 15`. Layers 0-14 own caches; 15-34 read an earlier layer's.
`kv_share_map()` implements "the last preceding layer of the **same attention type**", and with the
period-5 pattern that collapses to **two sources: layer 13 for all 16 shared sliding layers, layer 14
for the 4 shared full ones.** Not a rolling window, not paired layers. `create_kv_cache` allocates over
`range(first_kv_shared_layer_idx)` only, so **15 caches exist, not 35.**

### The sliding ring caps 12 of those 15 at the window

With `window_kv` on — it auto-resolves True whenever `max_model_len > sliding_window`, so **always
here** — sliding layers allocate `min(max_seq_len, sliding_window)` = **512 slots**, indexed
`position % buf_len`. Full-attention layers allocate the full context.

**The four quirks compound, and the result is that KV is free on this rig:**

| | sliding (12 layers, ring) | full (3 layers) | total | of 14.07 GB |
| :--- | ---: | ---: | ---: | ---: |
| ctx 4096, `window_kv=True` | 6.0 MiB | 24.0 MiB | **30.0 MiB** | **0.22%** |
| ctx 4096, `window_kv=False` | 48.0 MiB | 24.0 MiB | 72.0 MiB | 0.54% |
| ctx 8192, `window_kv=True` | 6.0 MiB | 48.0 MiB | 54.0 MiB | 0.40% |

At float16, B=1, one KV head. The per-token figure if every cached layer grew is
`12x256x2x2 + 3x512x2x2` = **18,432 B = 18 KiB/token**, reproducing `MODELS.md` exactly.

**So the entire KV cache at 4K is 30 MiB, and the prefill transient this rig actually OOMs on is
~5.2 GiB — 174x larger.** That is the arithmetic behind "KV is NOT the binding constraint, prefill
is", and it is why raising `MAX_MODEL_LEN` does not trade against KV in any meaningful way. **Never
size this rig's context from KV arithmetic.**

### Where Turing's 64 KiB actually bites — and where it does not

**The 512-wide full-attention head is exactly what breaks the vLLM path on this chip.** Gemma 4's
heterogeneous head dims force `TRITON_ATTN`, whose 512-wide tile wants ~96 KiB of shared memory per
block against Turing's 64 KiB ceiling — `../gpu-jax-g5g-2b/docs/turing-aarch64-gap.md` is the
measured write-up, and it
is the reason this rig exists.

**JAX does not care, and the reason is specific:** attention here is ordinary XLA ops, not a hand-tiled
kernel, so there is no per-block shared-memory budget in the attention path at all. A 512 head_dim is
just a wider matmul.

**The ceiling still bites — one step over, in quantization rather than attention.** The fused W4A16
Pallas kernel is tiled for TPU VMEM and needs **550 KiB - 1.1 MiB** per block at these shapes.
`check_w4a16_fits_scoped_memory()` computes the requirement and **raises at startup with the arithmetic
attached** rather than dying as a Triton `OutOfResources` at the first token — which is precisely the
failure mode that made the vLLM path hard to diagnose.

**The distinction is worth keeping: attention escaped the ceiling because XLA does not tile by hand;
the quantized matmul did not, because Pallas does.**

### One quirk E2B does not have

`attention_k_eq_v` — full-attention layers shipping no `v_proj`, one projection feeding both K and V —
is **`false` on E2B and E4B, and `true` on 12B, 26B and 31B.** The port reads it and aliases V to K.
Treat it as the family default **above** E4B: a larger sibling that reports missing tensors on load
should be checked against this first, and a loader that tolerates `None` there produces a silently
broken model that still emits fluent text.

## The profiler ships with the image

**CHANGED 2026-08-29. xprof and tensorboard are installed by cloud-init**, as their own stage,
and `INSTALL_PROFILING=0` restores the previous serving-only image.

They used to be on-demand, and the reasoning was sound on its face — "a serving image should
not carry a profiler", and xprof drags in gcsfs plus google-cloud-storage behind a 39 MB
wheel. **In practice it meant they were never installed at all.** The documented command
pointed at a `requirements-profiling.txt` inside `APP_DIR` that the deploy payload
deliberately excludes, so `pip install -r` failed with `Could not open requirements file` and
the xprof extraction then died on `ModuleNotFoundError` — both in logs nobody reads.
`tune_loop.py` worked around it by shipping the file itself. Installing at boot removes the
workaround and the failure mode together.

**MEASURED 2026-08-29: the stage costs 5 seconds of a 76-second install** (xprof 2.23.1,
tensorboard 2.21.0). The "GBs of image" objection was never the right size.

Three things about the shape of the change:

- **It is its own stage, and deliberately NON-FATAL.** The serving deps run under `set -e` and
  *should* kill the install if they fail; a broken profiler wheel must not cost a box that can
  serve. The `$PIP xprof tensorboard || echo 'WARNING: …'` line is that distinction, and a
  test asserts the `||` is there — without it this is the `mkswap` incident again.
- **`_PROFILING_REQUIREMENTS` is the source of truth**, mirrored by
  `requirements-profiling.txt`, with a test asserting they agree — the same arrangement as
  `_SERVING_REQUIREMENTS`. A separate test asserts neither package is *on* the serving list.
- **xprof's wheel is `manylinux_2_35`.** Ubuntu 26.04 gives glibc 2.43, so there is headroom —
  but the old 22.04 base sat exactly on that floor. That matters more now than it did
  on-demand: this is the default path, so losing the wheel fails a stage rather than a command
  nobody ran.

**TensorBoard will not render these profiles on its own.** xprof is the successor to
`tensorboard-plugin-profile` and serves its own UI: `xprof --logdir /tmp/jaxtrace --port 6006`,
tunnelled — this rig opens no inbound port for it and must not. TensorBoard is there for
everything else, and because having both present removes "is it just not installed?" from a
debugging session.

## Measurement

**This rig has ONE measurement of its own**, in `benchmarks/runs/<date>-<what>-g4dn/` where
`<hw-short>` equals the hardware slot:

| Run | Decode | What it added |
| --- | ---: | --- |
| `2026-08-29-first-serve-g4dn` | **13.1 tok/s** | First serve. 3 context cells + xprof. Settles the Turing-vs-Graviton2 A/B. |

Report: `benchmarks/reports/2026-08-29-first-serve-g4dn.json`, schema 1.1, validating.

**The eight runs under `gpu-jax-g5g-2b/benchmarks/runs/` are the SIBLING's** and are quoted
throughout this file as the source of most findings here. They were measured on a T4G behind a
Graviton2, and their `<hw-short>` is `g5g` for exactly that reason. Do not move one into this
rig, and do not difference a number from one against a number from the other without saying
which is which:

| Sibling run | Decode | What it established |
| --- | ---: | --- |
| `2026-08-19-first-serve-g5g` | 12.5 tok/s | First serve. CUDA 12 / Python 3.12. |
| `2026-08-21-cuda13-py314-g5g` | 12.4 tok/s | CUDA 13 / Python 3.14, same AMI. |
| `2026-08-25-context-sweep-g5g` | 12.80 tok/s | 12 cells, context × output. First xprof. |
| `2026-08-26-config-sweep-g5g` | 12.8 tok/s | Config sweep: all three levers fail. Locates the prefill ceiling. |
| `2026-08-26-quant-levers-fixed-g5g` | **13.10 tok/s** | Levers fixed. Sets the current default (`ple4+int8head`). |
| `2026-08-27-ubuntu2604-base-g5g` | 12.80 tok/s | Ubuntu 26.04 base. 80s install, read_shards 3x faster. |
| `2026-08-27-baseline-g5g` + `-xprof` | 12.80 tok/s | First runs through `tune_loop.py`. |
| `2026-08-28-full-run-cached-g5g` | 12.9 tok/s | Fresh instance restoring the cache from S3. |

**Its 12.4/12.5/12.8 figures are all the same configuration** (`ple0`, no int8 head) on
successive stacks, and within noise of each other. **13.10 is a different configuration**, not
an improvement to the old one — see the quantization-lever section. This rig's 13.1 is at that
same `ple4+int8head` default, which is what makes it comparable.

**The CUDA 13 / 3.14 bump is performance-neutral** — it buys currency, not speed. Compare on
the `tpu_jax_decode_tokens_per_second` gauge, not end-to-end tok/s: the same prompt returned 64
completion tokens in one run and 53 in the other, and end-to-end wall includes prefill and
HTTP, so the token count moves it. **The same caution applies to this rig's own three cells**:
end-to-end falls 12.67 → 11.65 → 8.75 across the context sweep while the gauge holds at
13.1/13.2/13.1. **Decode does not degrade with context; TTFT does.** Quote the gauge.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them, so edit the root originals, never
these. `reports/` and `runs/` stay in the rig. `make benchmarks-rollup` regenerates
`benchmarks/INDEX.md`; do not hand-edit it.

**`tune_loop.py` hardcoded its `<hw-short>` as `g5g`**, so this rig's first run initially
landed in `benchmarks/runs/2026-08-29-first-serve-g5g` — a directory name asserting hardware
the run did not happen on, inside the rig whose whole point is that the hardware differs.
`HW_SHORT` is now derived from slot 3 of the rig directory. **A literal is what silently
survives a fork**; the same lesson as `SKILL_STEM` in `project-setup.sh`.

Three numbers you will be tempted to reuse, and must not:

- **43.1 / 44.24 tok/s** — the vLLM sibling on `g5g.4xlarge` / `g5g.xlarge`, 2026-08-12/13.
  Different runtime, and different silicon (T4G, not T4), and the figure was obtained *with
  reduced Triton tiles*. It is the number this rig exists to beat, not a baseline it inherits.
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
- **~44 tok/s on one Inferentia core** from `~/gemma4-tips-aws` — different harness, different
  silicon.
- **Anything from `~/gemma4-tips`** — that tree duplicated its own artifacts and its directory
  names misattribute both model and chip. Never read a model or a chip off one.

A config flag being accepted is not evidence it did anything. Cross-check against an absolute
physical bound — 320 GB/s of GDDR6 and 15360 MiB is the whole envelope here, and xprof puts
peak HBM at 298.083 GiB/s on this device — not against another config.

**Spot is the capacity constraint, and price does not predict it.** `us-east-1a` refused with
`InsufficientInstanceCapacity` on 2026-08-29 while being one of the *cheapest* AZs;
`us-east-1b` took the request immediately at $0.3724/hr. Retry across AZs; never read
cheapness as availability. The sibling recorded the same asymmetry with the AZs reversed,
which is the point — it moves.

## Fork debris — the class of error, and this rig's own instance of it

**This rig was forked from `gpu-jax-g5g-2b` on 2026-08-28 and the code was converted before
the prose was** — the same sequence, one generation on, that the sibling went through when it
forked from `gpu-vllm-g5g-2b`. The list stays because the *class* of error keeps recurring:
inherited prose describing hardware the rig does not have, in a rig whose whole premise is
that the hardware differs.

**Cleared 2026-08-29** (this rig's fork):

- **`server.py`'s module docstring claimed this rig had served and that the numbers were its
  own** — "12.4-12.5 tok/s decode on g4dn.xlarge across the two runs under `benchmarks/runs/`"
  — while `benchmarks/runs/` was **empty**. Those were the sibling's T4G numbers relabelled.
  This is the single worst kind of fork debris, because it is indistinguishable from a result.
- **`.codex/config.toml` was 100% unconverted**: server name, skill path, `INSTANCE_TYPE`, and
  three gated tool names this rig does not have. The rig was misregistered for Codex and
  nothing caught it, because `server.py` was right.
- **`plugin.json`'s description and keywords** still said Graviton2 / T4G / aarch64.
- **`AGENTS.md` and `GEMINI.md` were the G5g rig's copies**, describing a Graviton2 host, a
  T4G, and an arm64 AMI requirement.
- **`tune_loop.py` hardcoded `<hw-short>` as `g5g`**, so this rig's first run was written to a
  directory naming hardware it did not run on. Now derived from the rig directory.
- **`README.md` predicted a 16,384 MiB device budget** off `describe_instance_types`. The
  device reports **15,360 MiB**, same as the T4G. Falsified by the first launch.
- Throughout `server.py`, `tpu.env`, `Makefile` and the three requirements files: ARM64 DLAMI
  reasoning, Graviton-CPU driverless-image warnings, and "aarch64 wheels exist" packaging
  arguments — all inherited, none of them true here, and the last of them is close to vacuous
  on x86_64.

**Cleared earlier, on the sibling, and kept because the class recurs** — TPU-rig or vLLM-rig
prose describing precision this chip cannot run:

- `jax_openai_server.py`'s module docstring claimed "TPU v6e-1", the `-qat-w4a16-ct`
  checkpoint and BF16 activations. It now points at `/health` and the
  `tpu_jax_precision_info` series instead of naming a precision at all.
- The load banner printed **"Loading W4A16 QAT weights"** unconditionally while loading the
  dense fp16 checkpoint — so `get_jax_logs` told an operator the box had done the one thing
  this rig refuses to do.
- `/health` reported `weights="bf16"` and a hardcoded `activations="bfloat16"` on a chip with
  no bf16 datapath, and echoed the *requested* KV dtype (`auto`) rather than what it resolved
  to. All three now come off `ENGINE.precision_info()`.

**The pattern worth extracting:** every one of these was invisible to the test suite, and in
each case for the same reason — `server.py`'s `RIG_NAME` was correct, so
`test_rig_name_matches_directory` passed while the files *around* it named another rig. Tests
that check a rig agrees with itself cannot catch a rig that agrees with itself about the wrong
hardware. Only a launch settles that, which is what 2026-08-29 was for.
