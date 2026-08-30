# CLAUDE.md — `gpu-pytorch-g4dn-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **PyTorch + `transformers`** on **AWS EC2
G4dn** — an x86_64 Intel host with an **NVIDIA T4** GPU (Turing, SM 7.5, **15360 MiB
measured**, not the 16384 `describe_instance_types` implies).

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`.
It is **not** one of the `gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform
slot with them.

"PyTorch" here means **plain CUDA PyTorch**, not `torch_xla`: the model is
`AutoModelForCausalLM` from `transformers`, driven by `torch_openai_server.py` behind an
OpenAI-compatible FastAPI server, run under systemd — **not docker**. There is no vendored
model port, no XLA, and nothing is compiled on the instance.

## THIS RIG HAS SERVED NOTHING

Forked from `gpu-pytorch-g5g-2b` on **2026-08-29** and retargeted from Graviton2 (aarch64) to
x86_64. `benchmarks/` is empty. No instance has been launched from this tree.

**There are no MEASURED values anywhere in this rig.** Every number in the code, in
`tpu.env`, or in this file is one of three things, and each is labelled where it appears:

- read from an AWS API (the instance-type table, the AMI parameters — both 2026-08-29),
- a vendor spec, or a figure the JAX rig on this exact instance type measured (the T4's
  15360 MiB / 14.07 GiB usable and 320 GB/s), or
- **inherited from another rig**, in which case `docs/INHERITED.md` says whether it carries.

`docs/INHERITED.md` is the most important file here. Read it before quoting anything.

## Where this rig sits: the 2×2

Four rigs, two axes. The GPU is the same Turing generation in both columns — T4G on G5g, T4
on G4dn, both SM 7.5, both **15360 MiB** — so **the column is the host and the row is the
runtime**:

| | G5g (Graviton2, aarch64) | G4dn (x86_64) |
| --- | --- | --- |
| pure JAX | `gpu-jax-g5g-2b` ✅ measured | `gpu-jax-g4dn-2b` ✅ measured |
| PyTorch | `gpu-pytorch-g5g-2b` | **`gpu-pytorch-g4dn-2b`** |

Reading **down a column** isolates the runtime. Reading **across a row** isolates the host.

**The row is already answered, and that is what makes this rig's column the interesting
one.** `gpu-jax-g4dn-2b` first served on **2026-08-29** and landed on top of its G5g sibling:
decode **13.1 tok/s against 13.10**, `tpu_jax_weight_bytes` **6.155 GB against 6.155 GB**,
and an xprof profile reproducing **54.4% dtype conversion / 32.8% fp32 GEMV / 0.0%
TensorCore** with roofline peaks identical to three decimals. Its conclusion, in its own
words: *the host architecture contributes nothing measurable to decode.*

So the **86.9% dtype-plus-fp32-GEMV tax is a Turing property**, not a Graviton2 one. And the
JAX rigs have already **falsified** the obvious fix: converting the whole parameter tree to
float16 moved throughput **+0.0%**, because at `B=1` cuBLAS dispatches
`gemvx::kernel<int,int,float,float,float,float,...>` — an all-fp32 GEMV with **no half
path**. The promotion is not a storage-dtype bug, no flag removes it, and the route to the
ceiling is a **GEMM** (batching), which `MAX_NUM_SEQS=1` closes on both engines.

**What is still open is the column: is that tax the CHIP, or is it XLA?** `gpu-jax-g4dn-2b`
is the same host, the same chip, the same checkpoint and the same dtype policy as this rig,
which makes this **the cleanest runtime A/B in the tree** — cleaner than the G5g pair, whose
vLLM side used hand-reduced Triton tiles, and cleaner than any cross-host comparison.

**The baseline to beat or match**, from that report (median of 3, warmed at the measured
shape, 64 output tokens):

| input tokens | 41 | 521 | 2,057 |
| --- | ---: | ---: | ---: |
| `tpu_jax_decode_tokens_per_second` | 13.1 | 13.2 | 13.1 |
| end-to-end `output_tok_per_s` | 12.671 | 11.654 | 8.748 |

Compare on the **gauge**, not end-to-end: the gauge is flat in context and the end-to-end
figure falls because it carries prefill and HTTP. Same instance type, same region, same
checkpoint, so no correction is needed — which is exactly the point of the grid.

One number from that run is a **correction rather than a baseline**: the T4 reports
**15360 MiB** via nvidia-smi, identical to the T4G, with **14.07 GiB usable** — not the
16384 MiB `describe_instance_types` implies. That rig predicted a real 1 GiB advantage on the
API's number and did not get one. **Size from 14.07 GiB.**

## Two forks, two axes, and provenance has a direction

```
gpu-jax-g5g-2b  --(runtime: JAX -> PyTorch, 2026-08-28)-->  gpu-pytorch-g5g-2b
                                                                    |
                                              (host: aarch64 -> x86_64, 2026-08-29)
                                                                    v
                                                            gpu-pytorch-g4dn-2b
```

An inherited claim has to be checked against **which axis it belongs to**: a claim about the
**GPU** carries down both arrows; a claim about the **host** dies at the second; a claim about
the **runtime** died at the first. The recurring failure in this monorepo is not a wrong
number, it is a **right number attributed to the wrong rig**, and this rig is two hops from
where most of its inherited prose was written.

Three inherited claims are worth naming here because each is stated flatly somewhere in the
tree and each is **wrong or unproven for this rig**:

- **"Upstream PyPI wheels omit `sm_75`."** Measured for **aarch64**. Upstream x86_64 CUDA
  wheels have carried Turing for years. Taking torch from the AMI is still right — vendor
  build on a vendor driver — but that is not the reason, and `verify_gpu_arch` is what
  settles it on a given image.
- **"`torch_xla` cannot be installed here."** `torch_xla` 2.9.0 publishes
  `manylinux_2_28_x86_64`, which is exactly this platform. The argument against it now rests
  entirely on the CUDA backend being **retired** (it warns on initializing the deprecated
  XLA:CUDA device; the nightly CUDA builds are gone) and on plain CUDA being the better
  experiment — XLA vs not-XLA rather than two frontends over one compiler.
- **The fused W4A16 Pallas ceiling.** True of the JAX rigs, irrelevant here — see below.

## Torch comes from the AMI, and that inverts the JAX rigs

The one structural difference from every JAX rig in this tree.

| | JAX rigs | this rig |
| --- | --- | --- |
| what the AMI supplies | driver only | driver **and torch** |
| what pip supplies | CUDA + jax | `transformers accelerate` |
| AMI parameter | `base-oss-nvidia-driver-gpu-*` | `oss-nvidia-driver-gpu-pytorch-*` |
| interpreter the unit runs | any python3.x | **the DLAMI's own venv**, probed |

`install.sh` **probes** for the interpreter that can already `import torch` and writes the
resolved path to `{APP_DIR}/PYTHON_BIN`; the unit's `ExecStart` is then rewritten to it. The
venv path moves between DLAMI releases, so a hardcoded interpreter is only visible as a
failure at the first token. Installing `transformers` into `/usr/bin/python3` and pointing
the unit there yields `ModuleNotFoundError: No module named 'torch'` **after the install has
reported success**.

**A box with no torch is a WRONG AMI, not a missing `pip install torch`.** A base driver-only
DLAMI satisfies architecture and driver, boots perfectly well, and fails at the
torch-interpreter stage. Adding a pip torch to "fix" it would replace a vendor build with an
upstream wheel — a change nobody asked for and one this rig has not tested.

## AMI resolution — three requirements, not two

The JAX rigs need two things of an image (architecture, driver). This one needs three, and
the third is what a base DLAMI fails:

1. **x86_64.** A Graviton image cannot boot here.
2. **The NVIDIA driver.**
3. **torch itself.**

`DLAMI_SSM_PARAMETER` pins all three and is single-valued, so it is preferred;
`DLAMI_NAME` + `describe-images` is the fallback. **Never hardcode an AMI id.**

### `/latest/` in a DLAMI parameter path does not mean latest

Inherited and still true. A DLAMI parameter path names a **PyTorch-version and an
Ubuntu-version line**, and `/latest/` is only the newest build *within that line* — which AWS
eventually stops rebuilding. The version in the path is a **real pin** that has to be
revisited; it reads as "track latest" and is not.

**VERIFIED 2026-08-29** by enumerating the 60 `x86_64` DLAMI parameters:
`oss-nvidia-driver-gpu-pytorch-2.13-ubuntu-26.04` is the newest line, it resolved to
`ami-0d7cd40a7956dd2c4` (*Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.13 (Ubuntu 26.04)
20260829*), and its SSM entry had been rewritten **that morning** — a live line, not a frozen
one.

**Two things here are newer than the G5g sibling because x86_64 has them and arm64 does
not:** PyTorch 2.13 against its 2.12, and Ubuntu 26.04 against its 24.04. 26.04 is also what
the JAX rigs settled on.

### The name filter is architecture-specific, and this cost the sibling nothing only by luck

**AWS names the two architectures' images in different word order:**

```
arm64    Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch 2.12 (Ubuntu 24.04)
x86_64   Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.13 (Ubuntu 26.04)
```

**VERIFIED 2026-08-29:** the G5g rig's pattern, run against `x86_64`, matches **zero**
images. Carried over unchanged it would have raised only when SSM was *also* unavailable —
which is precisely when the fallback is load-bearing. `DLAMI_NAME` and `DLAMI_SSM_PARAMETER`
must change together; `test_tpu_env_agrees_with_server_defaults` covers both keys.

### The interpreter version moves with the AMI

`TORCH_PYTHON_VERSION=3.14` here against the sibling's `3.12`, and this is **forced, not
chosen**. deadsnakes publishes `python3.14` for jammy and noble **only** — not for 26.04. So
pinning 3.12 against a 26.04 image would miss the system interpreter, take the deadsnakes
branch, and fail under `set -e` on a PPA with nothing to offer it.

It does **not** decide what the service runs — that is the DLAMI's venv, found by the probe.
It only decides whether that bootstrap stage is a no-op or a PPA round trip.
`test_interpreter_version_tracks_the_ami_line` pins the pair.

## Instance sizing, and two things that read as typos

`g4dn.xlarge` is the default, and **every g4dn size is supported.** Unlike the G5g sibling,
which rejects its 8 GiB size, this family starts at 16 GiB — so `project-setup.sh` rejects no
size at all and `_validate_instance_type` only enforces the list.

Read from `describe_instance_types` on 2026-08-29:

| size | GPUs | GPU mem | vCPU | RAM |
| --- | ---: | ---: | ---: | ---: |
| `g4dn.xlarge` | 1 | 16 GB | 4 | 16 GB ← default |
| `g4dn.2xlarge` | 1 | 16 GB | 8 | 32 GB |
| `g4dn.4xlarge` | 1 | 16 GB | 16 | 64 GB |
| `g4dn.8xlarge` | 1 | 16 GB | 32 | 128 GB |
| `g4dn.12xlarge` | **4** | 64 GB | 48 | 192 GB |
| `g4dn.16xlarge` | **1** | 16 GB | 64 | 256 GB |
| `g4dn.metal` | 8 | 128 GB | 96 | 384 GB |

- **The size suffix does not give the GPU count.** `16xlarge` carries ONE T4 and `12xlarge`
  carries four. Anything deriving GPU count from the number in the name is wrong here, which
  is why `_G4DN_SIZES` is data rather than arithmetic.
- **vCPU is RAM/4, not RAM/2.** The sibling computed vCPUs as `host_ram_gb // 2`, which holds
  on every G5g size and on **no** g4dn size. The G-family quota is counted in vCPUs, so a
  derived figure is wrong in the one place the number is used — `check_g4dn_quotas`. vCPUs
  are now carried in the table, and `test_vcpus_are_carried_not_derived` asserts the derived
  form would be wrong for every size.

**Nothing shards across multiple GPUs.** The engine is single-device, `_serve_argv` emits no
tensor-parallel flag, and `_tensor_parallel_size` reports the count with nothing acting on
it. On `12xlarge` three T4s idle; on `metal`, seven. **A bigger instance buys host RAM and
vCPUs, never device memory** — and the waste is larger here than on G5g, whose biggest sizes
carry two GPUs rather than eight.

### The swapfile, and why the threshold outlived both its reasons

`_SWAP_AT_OR_BELOW_HOST_RAM_GB = 16`, inclusive, which on this family selects **exactly one
size: `g4dn.xlarge`, the default.**

**Neither documented cause can occur here.** The G5g rig's 8 GiB size could not `mmap` the
10.2 GB checkpoint at all — this family has no 8 GiB size. Its 16 GiB size was OOM-killed
five times at 14.3 GB anon-RSS inside the JAX loader's PLE-table quantiser — that code is not
in this rig. What remains is generic and untested: 16 GiB staging a 10.2 GB checkpoint
through transformers' loader is thin, and the failure mode is a kernel kill under
`Restart=on-failure`, which reads as a crash-loop rather than as memory pressure. Cheap
insurance. **Delete it only with a measurement.**

**The `mkswap -q` fix is inherited and matters more here.** That is a busybox flag; util-linux
rejects it with `invalid option -- 'q'`, and under `set -e` it kills cloud-init *before
install.sh is written*, leaving an empty `APP_DIR` and no install log. On the G5g rig this
stayed latent for as long as the block only rendered for a size nobody launched. **Here it
renders for the DEFAULT size**, so the same bug would break the very first launch. A code
path that only renders for a size nobody launches is untested code — and this one is not.

## Turing decides the dtype, not the config

**bfloat16 on this chip does not raise — CUDA emulates it through fp32.** Correct numbers,
quiet slowdown, which is worse than an error. There is no fp8 either.

`resolve_compute_dtype()` reads the live compute capability and returns `float16` below
SM 8.0, logging what it resolved:

```
torch device policy: name=Tesla T4 compute_capability=7.5 pre_ampere=True compute_dtype=float16
```

`DTYPE=float16` in `tpu.env` is a **record of that policy, not the input to it.**

The guard is duplicated in `torch_openai_server.py` and `torch_generate.py` on purpose —
either can be run alone, and `torch_generate.py` exists precisely for when the server module
cannot be imported. `tests/test_engine.py` asserts the two agree, that the boundary is SM 8.0
rather than a chip name, that a CPU device never claims a 16-bit path, and that the policy is
**logged** rather than merely applied. That last one is not ceremony: the JAX rig's
device-policy banner went to a root logger with no handler and was discarded on every single
run, on the one rig whose entire premise is which dtype the device picked.

`test_bfloat16_is_never_hardcoded_as_the_compute_dtype` scans both payload files for a literal
`dtype=torch.bfloat16`, because that substitution would sail past every other test — those
test the resolver, not its callers.

## Where Turing's 64 KiB bites, and where it does not

Turing allows **64 KiB of shared memory per block**; Ampere and later allow 100–227 KiB. That
one number is what makes Gemma 4 hard on this chip.

- **In vLLM it hits attention.** Gemma 4's head dims are heterogeneous — sliding 256, global
  **512** — only FA4 or Triton handle that, FA4 is unavailable, so `TRITON_ATTN` is *forced*,
  and at `head_size=512` its tile wants ~96 KiB. Hence `gpu-vllm-g5g-2b`'s unlanded Triton
  patch, a ~67-minute from-source build, a CUDA toolkit and a Rust toolchain.
- **In the JAX rigs it moved to quantisation.** Attention is ordinary XLA so there is nothing
  to patch, but the fused W4A16 Pallas kernel lowers through Triton on GPU and needs
  **550 KiB – 1.1 MiB** per block. `check_w4a16_fits_scoped_memory()` raises at startup with
  the arithmetic attached.
- **Here it is in neither place.** Attention is transformers' SDPA — not a hand-tiled kernel,
  so there is no per-block shared-memory budget in the attention path. And there is no Pallas
  and no fused quantised path at all.

**⚠️ That third row is reasoning, not a measurement.** Whether transformers' attention
actually handles Gemma 4's 512-wide global head on SM 7.5 is **the open risk of this rig**.
SDPA *should* avoid the ceiling; nothing has proven it. It is the second thing to check on a
real run, after the device-policy line.

## Why the dense checkpoint — and NOT for the JAX rigs' reason

This rig serves `google/gemma-4-E2B-it` (`QUANT_MODE=fp16`), deliberately not the
`-qat-w4a16-ct` export, so **the rig name carries no encoding slot.**

The JAX rigs reach that conclusion through the Pallas ceiling above. **Do not repeat that
argument here.** There is no Pallas in this rig and no fused path: `AutoModelForCausalLM`
simply has nowhere to put w4a16 weights without `bitsandbytes` or `torchao`, neither of which
is installed. Same outcome, entirely different mechanism — and the difference matters,
because the JAX argument implies a ceiling that would lift on Ampere, while this one implies
a dependency that could be added on any chip.

The dense model fits: ~9.5 GiB of weights in 14.07 GiB of usable device memory. That is
less headroom than the EC2 API's 16384 MiB suggests — see the 2×2 section.

## Nothing compiles here, so the cache machinery was removed

There is no `torch.compile` on this path, by an argument `torch_openai_server.py`'s own
docstring makes: on CUDA nothing recompiles when the sequence dimension changes, so a
compiled static-shape buffer only means every decoded token pays a full SEQ-length forward.
That is XLA discipline applied where it does not pay.

**The fork carried an XLA compilation cache into a rig with no compiler.**
`gpu-pytorch-g5g-2b` still has `JAX_COMPILATION_CACHE_DIR`, `JAX_CACHE_S3_URI`,
`XLA_PYTHON_CLIENT_MEM_FRACTION`, and a systemd timer pushing `/opt/jax-cache` to S3 every
ten minutes — writing two JAX variables into the unit's environment for a process that reads
neither. The directory stays empty and the timer reports a **successful sync of nothing,
forever.**

That is the same silent-success shape as the JAX rig's own cache bug, where both halves
worked correctly against a path nothing wrote to. **Removed here**, along with the inert
`PLE_BITS` / `INT8_LM_HEAD` / `PREFILL_CHUNK_SIZE` knobs, and
`tests/test_server.py::NoCompilationCacheTests` asserts none of it comes back. If
`torch.compile` is ever adopted, the knob is `TORCHINDUCTOR_CACHE_DIR`.

**The rule the removals share: a config key that nothing reads is worse than a missing one,
because "the flag was accepted" then looks like evidence.**
`test_tpu_env_carries_no_key_without_a_reader` enforces it in the other direction too — every
live key in `tpu.env` must appear in `server.py`.

Warm-up still matters. The first call pays autotune and allocator growth, and the inherited
rule holds: **warm up at the shape you intend to measure.**

## `MAX_MODEL_LEN` is inherited and probably too low

`MAX_MODEL_LEN=4096`, passed as `--seq`, is this rig's **one real serving knob** — the only
value in `tpu.env` that reaches the process as a parameter rather than as a record.

**Its reasoning does not transfer.** The JAX rig lowered 8192 → 4096 because a 5,120-token
prompt failed allocating 2.59 GiB of prefill temporaries in its hand-written prefill, and the
documented mitigation (`PREFILL_CHUNK_SIZE`) was structurally unreachable behind
`window_kv=False`. **None of that code is here** — transformers prefills through SDPA and
owns its own cache.

So 4096 is a **conservative placeholder carried across a fork**, not a measured ceiling. It
may well be low. Raise it with a measurement, not with an argument.

## The bootstrap is two-stage, on purpose

Cloud-init installs the **runtime only** and then waits. The serving payload is this rig's own
source — there is no published artifact for "our OpenAI-compatible transformers server", and
cloning the monorepo would need credentials on the box — so it ships separately over SSM as a
gzipped tarball. **User data could not carry it: the limit is 16 KB.**

The tarball is built deterministically (mtime and uid/gid zeroed), which is what makes
`deploy_torch_server` idempotent and lets an unchanged redeploy be detectable.

```
create_g4dn_instance → get_install_progress → verify_gpu_arch → deploy_torch_server
                     → get_torch_logs → verify_model_health
```

Install progress goes to `/var/log/torch-install.log`; `{APP_DIR}/INSTALL_DONE` appears only
after **torch imports and sees the GPU**, so "INSTALL COMPLETE" is an assertion, not a guess.
`install.sh` emits `[stage] <name> +Ns` markers — grep with
`grep -F '[stage]' /var/log/torch-install.log` — so a hang is attributable to a step rather
than to "the install".

The unit is `torch-g4dn.service`; read it with `journalctl`, **not** `docker logs`.

**PEP 668 applies from Ubuntu 23.04 on**, so the bootstrap passes `--break-system-packages`
(including to `get-pip.py`, which runs before the wrapper variable exists). It is harmless and
ignored inside the DLAMI venv, which is not marked. This is a single-purpose serving box
installing into the interpreter systemd will run, and the monorepo forbids virtualenvs.

### The rendered script is 78% comments, and it has a hard 16 KB budget

**EC2 caps user data at 16 KB RAW**, before its own base64 encoding, and rejects
`run-instances` above it — so an overrun breaks the *launch*, not the install, and the error
names a size rather than a cause.

**Neither this rig nor the two it descends from tested that.** Measured 2026-08-29 while
adding the g4dn retarget: the rendered script had reached **15,076 B, 92.0% of the limit**,
with over half of it comment text — so a single added paragraph would have spent the
remainder. `gpu-pytorch-g5g-2b` renders 14,253 B, i.e. it is in the same position and does not
know it.

Trimming the shipped script brought it to **12,849 B (78.4%)**, and
`test_user_data_fits_the_ec2_limit_with_margin` now fails above 92% with a message saying to
trim `_user_data` rather than raise the bound. The long-form reasoning those comments carried
lives here instead; the script keeps the one-line version. **When you add to `_user_data`,
spend bytes on what an operator reads in `/var/log/cloud-init-output.log`, and put the history
in this file.**

The serving payload is deliberately not in that budget — it goes over SSM for exactly this
reason, and `test_payload_fits_one_ssm_run_command` guards it separately.

### `get_install_progress` distinguishes dead from slow

Cloud-init writes `install.sh` and **backgrounds** it, so anything that kills cloud-init
before that point leaves no install log — which used to render as `INSTALL IN PROGRESS` +
`no install log yet`, indefinitely, exactly like a healthy slow install. It now reports
`cloud-init status --long`, tails `cloud-init-output.log` when the install log is absent, and
returns a verdict separating **cloud-init error** / **done-but-never-started** /
**genuinely-still-booting**.

**SSM truncates at 24,000 characters.** Truncation is detected and announced rather than
returned as if complete, and the `CommandId` is logged at issue time and carried in every
error message — including timeout, where the command is still *running* on the box.

## Engineering rules

- boto3 and the standard AWS credential provider chain — **never shell out to the AWS CLI.**
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Do not create broad
  network or IAM policy. (The legacy sample this lineage was scaffolded from auto-creates a
  security group open to `0.0.0.0/0` — that was not carried over.)
- Scope instance discovery to `ManagedBy=gpu-pytorch-g4dn-2b`. Unlike the inf2 rig, which
  keeps a legacy tag to avoid orphaning instances, this rig is new and uses its own name.
- Hugging Face tokens live in Secrets Manager and are fetched at boot into a root-only
  `EnvironmentFile`. **Never** in user data — instance metadata is readable by anything on
  the box. `set +x` wraps the fetch because the script runs under `set -x` and bash traces
  assignments *with their values*. Tests assert both.
- Launches default to spot. Surface capacity errors rather than silently retrying.
- **Termination is cheap here**: there is no built image to lose with the root volume, only an
  install and the model cache. Do not import the vLLM sibling's "weigh stop against
  terminate" reasoning — that rig carries a 67-minute build.
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.
- `verify_model_health` uses `/v1/chat/completions`, because raw `/v1/completions` skips the
  chat template and is unreliable on `-it` models. On the vLLM sibling it was measured
  returning `': ok: ok: ok…'` — degenerate repetition, not the empty body the monorepo
  `CLAUDE.md` documents for the TPU rigs. Either way: **do not health-check by testing for a
  non-empty response**, or you will call a body full of garbage fine.
- `init.sh` blocks on `read` in its error path — never run it non-interactively.

## Spot capacity is not a given

Inherited from the G5g rigs and **not** re-measured on this family, but the shape of the
problem is regional rather than architectural, so plan for it:

- G5g spot was **exhausted across all four `us-east-1` AZs** on 2026-08-25.
- **Reclamation is highly variable, not reliably fast** — 21 minutes in one case, 19.2 hours
  in another, same instance type, same region. **Neither figure is typical**; quote the range.
- **Price is not a proxy for capacity.** On 2026-08-27 only `us-east-1a` had capacity, and it
  was the most *expensive* AZ by spot price.

**Checkpoint continuously rather than sizing work to an assumed lifetime.** g4dn is a much
older and much more widely deployed family than g5g, so capacity is plausibly better here —
but that is a guess, not a finding.

## AWS credentials

`server.py` uses the standard boto3 provider chain, so whatever `aws sts get-caller-identity`
resolves is what the rig gets. **When credentials expire, refresh them with
`./save-aws-creds.sh`**, which re-exports the active credentials to `.aws_creds` at mode 0600.

Three things about it that are easy to get wrong:

- **It snapshots credentials, it does not mint them.** `aws configure export-credentials`
  fails outright on an expired session, so re-authenticate first and then run the script.
- **It refuses to write anywhere inside a git work tree that is not gitignored.** `.aws_creds`
  is in this rig's `.gitignore` for exactly that reason. Never remove that line and never
  reach for `FORCE=1` — the guard is what keeps live keys out of a commit.
- **Nothing in this rig reads `.aws_creds` automatically.** The script's closing message is
  inherited from the legacy `~/gemma4-tips-aws` tree, whose Makefile loaded the file; this
  rig's does not. For `server.py` the provider chain is enough, and `AWS_PROFILE` is the
  supported way to pick a profile.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v` — **81
tests, all passing and none skipped** as of 2026-08-29. They are fully offline (no AWS, no
network, no GPU) and pin the facts above: the Turing dtype boundary, the x86_64 AMI parameter
and its architecture-specific name filter, the interpreter/AMI pairing, the instance table
including both of its traps, that the token never reaches user data, that `tpu.env` and
`server.py` still agree, that no JAX or XLA key survived the fork, and that no `VLLM_*` /
`TORCH_CUDA*` key survived the one before it.

**No test is skipped, and that is deliberate.** The inherited `tests/test_engine.py` had 22
tests importing `ports.gemma4.jax_e_model` — a package this rig does not vendor — so every one
of them skipped on every run, forever. **A suite that always skips reads as coverage in the
summary line and asserts nothing.** It was rewritten against the torch device policy.

`make lint` runs `ruff check server.py refresh_skill.py torch_generate.py
torch_openai_server.py tests`, then `bash -n` on **four** shell scripts (`project-setup.sh`,
`init.sh`, `set_env.sh`, `save-aws-creds.sh`). **A new top-level module is silently unlinted
until it is added to that list.** There is no `ports/` exclusion here — the JAX rigs need one
because ruff would rewrite annotations in a shared vendored port; nothing here is shared that
way, so everything at the top level is linted.

**`deploy_torch_server` ships the SKILL SNAPSHOT, not the working tree.** `server.py` resolves
the payload next to itself, and the MCP server runs from `.claude/skills/…/mcp/`, so editing
`torch_openai_server.py` and deploying ships the *previous* `make skill` output with no
warning — the deploy reports success and the instance runs stale code. That cost the JAX rig a
full measure-and-conclude cycle before the md5s were compared. **Always `make skill` before
`deploy_torch_server`.**

`make skill` regenerates the snapshots under `.claude/skills/` and `skills/`. **Six files are
generated** — `server.py`, `project-setup.sh`, both requirements files, **and the serving
payload** (`torch_openai_server.py`, `torch_generate.py`) — because an installed copy under
`~/.claude/skills` still has to be able to run `deploy_torch_server`.

`SKILL.md` sits in the same tree and is a hand-written **source**: `refresh_skill.py` will not
recreate it, so `rm -rf .claude/skills` destroys it permanently.
`test_skill_is_complete_in_both_copies` guards both copies and also fails if any generated
file is stale.

There is no `make deploy` recipe on purpose: provisioning resolves an AMI from SSM at launch
time, and a Makefile would have to hardcode one. The target exists and prints that. There is
no `make medium` either — this rig carries no `medium-*.md` sources, and the target was
dropped rather than left pointing at nothing.

## MCP registration lives in four places

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and
`.claude/settings.local.json`'s `enabledMcpjsonServers`. All four must name the server
`gpu-pytorch-g4dn-2b`, which prefixes every tool as `mcp__gpu-pytorch-g4dn-2b__…`. All four
agree as of 2026-08-29. A mismatch makes `/mcp` and the tool prefix disagree about what this
rig is.

**Only `.mcp.json` is generated.** `project-setup.sh` writes it (`--server-name` sets both the
registered key and what the server advertises) and does **not** touch
`.claude/settings.local.json`. That file is hand-written; both are gitignored. The other two
are committed.

**The fork left `.codex/config.toml` naming `gpu-jax-g5g-2b`** — two rigs and one runtime
away — and pointing at a skill path that does not exist. `SKILL_STEM` in `project-setup.sh` is
**derived** from the rig directory rather than hardcoded, which is what the `Makefile` and
`refresh_skill.py` already did. Never reintroduce a literal: a literal is what silently
survives a rename.

`server.py` carries `RIG_NAME = "gpu-pytorch-g4dn-2b"`, asserted by
`test_rig_name_matches_directory` — which is why a registration breakage of this kind is
invisible to the tests and has to be checked by reading the four files.

Editing any of the generated-or-copied files means re-running `make skill`:
`project-setup.sh` is one of the six snapshotted into both skill copies, and
`test_skill_is_complete_in_both_copies` fails until you do.

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree**, and a convention change has to be
applied to all three by hand.

This rig has no `.claude-plugin/marketplace.json` of its own, which only matters if it is ever
published standalone. The marketplace `/plugin` actually reads is the **monorepo root** copy.

## Measurement

**`benchmarks/` is empty and must stay empty until this rig serves something.** Runs belong in
`benchmarks/runs/<date>-<what>-g4dn/`, where `<hw-short>` equals the hardware slot.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them, so edit the root originals, never
these. `reports/` and `runs/` stay in the rig.

Four numbers you will be tempted to reuse, and must not:

- **12.4 – 13.1 tok/s** — `gpu-jax-g5g-2b`. Different runtime **and** different host, i.e.
  diagonally opposite this rig in the 2×2. It is the number this rig exists to be compared
  against, not a baseline it inherits.
- **43.1 / 44.24 tok/s** — `gpu-vllm-g5g-2b`, and obtained **with reduced Triton tiles**.
- **~44 tok/s on one Inferentia core** from `~/gemma4-tips-aws` — different harness, different
  silicon.
- **Anything from `~/gemma4-tips`** — that tree duplicated its own artifacts and its directory
  names misattribute both model and chip. Never read a model or a chip off one.

When this rig does measure something: quote the server's `tpu_jax_decode_tokens_per_second`
gauge rather than an end-to-end rate, which also carries prefill and the HTTP round trip. The
`tpu_jax_*` metric prefix is **deliberately retained** across all these rigs — the benchmark
reports compare on those series *by name*, so renaming would break continuity; the rig is
distinguished by a `rig` label instead.

**A config flag being accepted is not evidence it did anything.** Cross-check against an
absolute physical bound — the T4's 320 GB/s and 14.07 GiB usable is the whole envelope — not against
another config. Most tests here are parity assertions between two of our own code paths, so an
assumption both paths share is invisible to all of them.

## Fork debris — what was cleared, and what the class of error looks like

Cleared on 2026-08-29, when the tree was re-forked from `gpu-pytorch-g5g-2b`. The list stays
because **the class of error keeps recurring**: prose describing a runtime the rig does not
run, or a host it does not run on, in a rig whose whole premise is which of those it is.

- `server.py`'s module docstring described the JAX path and claimed *"THIS RIG HAS SERVED,
  AND THE NUMBERS ARE ITS OWN: 12.4-12.5 tok/s"* — on a rig that had served nothing.
- `get_help` rendered a table of XLA knobs and a paragraph beginning *"Serving with **pure
  JAX** on AWS Graviton2"*, on a PyTorch rig.
- `.codex/config.toml` registered the server as `gpu-jax-g5g-2b`.
- `.claude-plugin/plugin.json` described *"pure-JAX"* serving on *"Graviton2 + NVIDIA T4G"*.
- The `Makefile` help text advertised `jax[cuda12]` aarch64 wheels and carried a `medium`
  target with no sources.
- `tests/test_engine.py` skipped all 22 of its tests on every run.
- `verify_gpu_arch`'s docstring cited an aarch64 wheel measurement as though it constrained
  x86_64.

Every one of those was inherited verbatim through a fork that changed the code first and the
prose later — which is how a rig ends up asserting the opposite of what it does.
