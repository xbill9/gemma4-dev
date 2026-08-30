---
name: gpu-pytorch-g4dn-2b-management
description: Manage AWS EC2 G4dn capacity (x86_64 + NVIDIA T4) and Gemma 4 E2B serving under PyTorch/transformers. Use when the user asks about provisioning, launching, listing, or terminating G4dn instances, installing or debugging PyTorch on T4 / Turing / SM 7.5, G-family quotas, DLAMI selection, or the gpu-pytorch-g4dn-2b devops MCP agent. Triggers include "G4dn", "T4", "Turing", "SM 7.5", "PyTorch on GPU", "transformers serving", "PyTorch DLAMI".
---

# gpu-pytorch-g4dn-2b management

Provision and operate **EC2 G4dn** (x86_64 Intel host + NVIDIA **T4** GPU) serving
`google/gemma-4-E2B-it` under **PyTorch + transformers**, through the
`gpu-pytorch-g4dn-2b` MCP server.

## This rig has served nothing

Forked from `gpu-pytorch-g5g-2b` on 2026-08-29 and retargeted from Graviton2 to x86_64.
`benchmarks/` is empty. **Do not quote any number as this rig's own**, and be specific about
where a number came from — `docs/INHERITED.md` is the list of what carries and what does not.

Numbers that will be offered to you and are not this rig's:

- **13.1 / 13.2 / 13.1 tok/s** — `gpu-jax-g4dn-2b`, 2026-08-29. Same host, same chip, same
  checkpoint, **different runtime**. This is the comparison this rig exists to make, so use
  it as a *baseline to report against*, never as this rig's own result.
- **12.4–13.10 tok/s** — `gpu-jax-g5g-2b`. Different runtime *and* different host.
- **43–44 tok/s** — `gpu-vllm-g5g-2b`, and obtained with hand-reduced Triton tiles.
- **Anything from `~/gemma4-tips`** — that tree duplicated its artifacts and its directory
  names misattribute both model and chip.

## Where this rig sits

Four rigs form a 2x2 over {runtime} × {host}. The GPU is the same Turing generation in both
columns — T4G on G5g, T4 on G4dn, both SM 7.5, both **15360 MiB** — so the column is the **host** and
the row is the **runtime**:

| | G5g (Graviton2, aarch64) | G4dn (x86_64) |
| --- | --- | --- |
| pure JAX | `gpu-jax-g5g-2b` | `gpu-jax-g4dn-2b` |
| PyTorch | `gpu-pytorch-g5g-2b` | **`gpu-pytorch-g4dn-2b`** |

Reading down a column isolates the runtime; reading across a row isolates the host.

**The row is answered.** `gpu-jax-g4dn-2b` first served 2026-08-29 at **13.1 tok/s against the
G5g rig's 13.10**, with weight bytes equal to the byte and an xprof profile reproducing 54.4%
dtype conversion / 32.8% fp32 GEMV / 0.0% TensorCore. The host contributes nothing measurable,
so the 86.9% tax is a **Turing** property.

**The column is what this rig is for: chip, or XLA?** Its A/B partner is `gpu-jax-g4dn-2b` —
identical hardware, so no correction is needed. Baseline: **13.1 / 13.2 / 13.1 tok/s** at
41 / 521 / 2,057 input tokens, 64 output, median of 3, warmed at the measured shape. Compare
on the `tpu_jax_decode_tokens_per_second` gauge, not end-to-end.

## Start here, every time

**Run `verify_gpu_arch` before anything else on a new instance.** It reports nvidia-smi's
view, `torch.__version__`, `torch.cuda.get_arch_list()`, the device capability, the dtype the
rig selects from it, and the result of **one real fp16 matmul**.

It asks *torch*, not jax, and that inverts the JAX rigs for a concrete reason: **torch comes
from the AMI here**, so "does this GPU have kernels" is a question about the image that
booted rather than about anything this rig installed.

**It is also the only thing that settles arch coverage on this rig.** The G5g sibling states
flatly that upstream PyPI wheels omit `sm_75` — that was measured for **aarch64** and is not
established for x86_64, where upstream CUDA wheels have long carried Turing. That claim was
deliberately not carried across the fork. Do not repeat it; run the tool.

## Torch comes from the AMI, never from pip

The single structural difference from the JAX siblings, and the reason the AMI is a
**PyTorch** DLAMI rather than the base driver image:

| | JAX rigs | this rig |
| --- | --- | --- |
| what the AMI supplies | driver only | driver **and torch** |
| what pip supplies | CUDA + jax | `transformers accelerate` |
| AMI parameter | `base-oss-nvidia-driver-gpu-*` | `oss-nvidia-driver-gpu-pytorch-*` |
| interpreter the unit runs | any python3.x | **the DLAMI's own venv**, probed |

`install.sh` probes for the interpreter that can already `import torch` and writes the
resolved path to `APP_DIR/PYTHON_BIN`, because the venv path moves between DLAMI releases.
Installing `transformers` into `/usr/bin/python3` and pointing the unit there yields
`ModuleNotFoundError: No module named 'torch'` **after the install reports success**.

If the box comes up with no torch at all, that is a **wrong AMI** — a base driver-only image
boots perfectly well and fails at the torch-interpreter stage. It is not a missing
`pip install torch`, and adding one would replace a vendor build on a vendor driver.

## Turing rules

The T4 is Turing (SM 7.5), **not** the Ada L4 the `gpu-vllm-l4-*` rigs target and not a TPU.

- **`float16`, not `bfloat16`.** bf16 does not fail on Turing — CUDA *emulates* it through
  fp32. Correct numbers, quiet slowdown, which is worse than an error.
  `resolve_compute_dtype()` reads the live compute capability and picks float16 below SM 8.0;
  `DTYPE` in `tpu.env` is a record of that policy, not the input to it.
- **No fp8.** There is no fp8 datapath, and no cache-dtype resolution step to refuse one:
  transformers owns the KV cache and allocates it in the model dtype.
- **The dense reference checkpoint**, not a `-qat-w4a16-ct` export — but **not for the JAX
  rigs' reason**. Theirs is that the fused W4A16 Pallas kernel needs 550 KiB – 1.1 MiB of
  shared memory per block against Turing's 64 KiB ceiling. There is no Pallas here and no
  fused path at all; `AutoModelForCausalLM` simply has nowhere to put w4a16 weights without
  bitsandbytes or torchao, neither of which is installed. Same outcome, different mechanism.
- **No tensor-parallel flag.** The engine is single-device.

Copying a flag set from a `gpu-vllm-l4-*` rig, from `~/gemma4-tips-aws`, or from a TPU rig
produces a config that is accepted and then wrong.

## Instance sizing

`g4dn.xlarge` is the default, and **every g4dn size is supported** — unlike the G5g sibling,
which rejects its 8 GiB size, this family starts at 16 GiB.

Two things about this family that read as typos and are not:

- **The size suffix does not give the GPU count.** `g4dn.16xlarge` carries **one** T4;
  `g4dn.12xlarge` carries **four**. Read `_G4DN_SIZES`, never the name.
- **vCPU is RAM/4, not RAM/2.** A `g4dn.xlarge` has 16 GiB and **4** vCPUs. The G-family
  quota is counted in vCPUs, so a figure derived the G5g way is wrong where it matters.

Only `g4dn.xlarge` gets a swapfile (the threshold is 16 GiB, inclusive). That is the size the
rig launches, so unlike the sibling — where the block stayed latent behind a size nobody
launched — it is on the critical path from the first launch.

**A bigger instance buys host RAM and vCPUs, never device memory.** On `12xlarge` three T4s
idle; on `metal`, seven.

## AMI resolution — three requirements, not two

`_resolve_ami` prefers the AWS public SSM parameter. All three must hold and a name filter
pins none of them reliably:

- **x86_64** — a Graviton image cannot boot here.
- **NVIDIA driver.**
- **torch itself** — the requirement a base DLAMI satisfies on the first two and fails on.

`/latest/` in a DLAMI parameter path is only the newest build **within** that
PyTorch-version + Ubuntu-version line, and AWS freezes lines it stops rebuilding. The version
in the path is a **real pin** that has to be revisited; it does not track.

**The `describe-images` fallback is architecture-specific.** AWS names the two architectures'
images in different word order — `Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch …`
against `Deep Learning OSS Nvidia Driver AMI GPU PyTorch …` — so the G5g rig's pattern
matches **zero** x86_64 images. Carried over unchanged it would have failed only when SSM was
also unavailable, which is exactly when the fallback is load-bearing.

`TORCH_PYTHON_VERSION` and `DLAMI_SSM_PARAMETER` move together: deadsnakes publishes
`python3.14` for jammy and noble only, so a 26.04 image needs 3.14 and an older one must not
ask for it.

**Never hardcode an AMI id.** There is no prebuilt AMI for this rig and none is needed —
nothing is built on the instance.

## Order of operations

`create_g4dn_instance` → `get_install_progress` → `verify_gpu_arch` → `deploy_torch_server`
→ `get_torch_logs` → `verify_model_health`

Cloud-init installs the **runtime only**. The serving payload is this rig's own source
(`torch_openai_server.py`, `torch_generate.py`), so `deploy_torch_server` ships it over SSM
as a gzipped tarball — there is no published artifact to pull, and user data could not carry
it at a 16 KB limit.

**`deploy_torch_server` ships the skill snapshot, not the working tree.** Run `make skill`
first, or you deploy the previous snapshot and the tool reports success either way.

## Nothing compiles here

There is no `torch.compile` on this path, so there is no compilation cache to warm, persist
or sync. The JAX rigs' `JAX_COMPILATION_CACHE_DIR` / `JAX_CACHE_S3_URI` machinery — including
a systemd timer pushing an empty directory to S3 every ten minutes and reporting success —
was **removed** in this fork rather than carried inert. If `torch.compile` is ever adopted,
the knob is `TORCHINDUCTOR_CACHE_DIR`.

Warm-up still matters: the first call pays autotune and allocator growth. **Warm up at the
shape you intend to measure.**

## Lifecycle guardrails

- Launches default to **spot**. Surface capacity errors; do not silently retry. G5g spot was
  exhausted across four AZs on 2026-08-25 and reclamation there ranged from 21 minutes to
  19 hours — quote the range, and checkpoint continuously rather than sizing work to an
  assumed lifetime.
- Terminating is permanent but cheap here: no built image to lose, only an install and the
  model cache. One-time spot instances cannot be stopped, only terminated.
- Provisioning requires explicit **subnet, security-group, and instance-profile ids**. Do not
  create broad network or IAM policy to make a launch succeed.
- The HF token comes from Secrets Manager at boot (`save_hf_token`). **Never** put it in user
  data — instance metadata is readable by anything on the box. The bootstrap disables
  `xtrace` around the fetch, because `set -x` traces assignments with their values.
- Instance discovery is scoped to `ManagedBy=gpu-pytorch-g4dn-2b`. Never operate outside it.

## Diagnostics

- `verify_model_health` uses `/v1/chat/completions`. Raw `/v1/completions` skips the chat
  template and is unreliable on `-it` models — on the vLLM sibling it returned degenerate
  repetition, not an empty body, so **do not health-check by testing for a non-empty
  response**.
- `get_endpoint` resolves the address from the instance. Never hardcode an endpoint.
- `get_torch_logs` reads the **systemd journal**, not docker: nothing here is containerized.
- `get_install_progress` reports INSTALL COMPLETE only once torch imported *and* saw the GPU,
  and separates **cloud-init error** / **done-but-never-started** / **still-booting** — a
  dead bootstrap and a slow one used to render identically.
- SSM truncates output at 24,000 characters. Truncation is detected and announced; a partial
  journal is how you conclude an error is not there.

## Measurement discipline

A config flag being accepted is not evidence it did anything. Cross-check against an absolute
physical bound — the T4's 320 GB/s and 14.07 GiB usable is the whole envelope — not against another
config. Record results in `benchmarks/runs/<date>-<what>-g4dn/`.
