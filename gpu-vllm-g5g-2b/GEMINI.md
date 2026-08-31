# GEMINI.md — `gpu-vllm-g5g-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **vLLM** on **AWS EC2 G5g** — a Graviton2
(aarch64) host with an **NVIDIA T4G** GPU (Turing, SM 7.5, 15360 MiB measured).

**Status: serving.** ~43 tok/s, 2026-08-12 — `benchmarks/runs/2026-08-12-first-serve-g5g/`.

**`CLAUDE.md` is authoritative where this file disagrees with it.** There is no generator;
a convention change has to land in `CLAUDE.md`, `AGENTS.md`, and `GEMINI.md` by hand.

## The two obstacles, in order

**1. Packaging — real, but AWS already solved it.** G5g needs aarch64 and SM 7.5 together.
Upstream `vllm/vllm-openai` arm64 is compiled `8.0 8.7 8.9 9.0 10.0 11.0 12.0` while the
amd64 image of the same tag carries 7.5, and the Dockerfile sets no `+PTX`. But **the AWS
ARM64 GPU DLAMI ships PyTorch with `sm_75`** — measured on both 2.7.0+cu128 and 2.12.0+cu132.
So PyTorch never needs building; only vLLM's kernels do, and CMake accepts
`CUDA target architectures: 7.5`. Two gaps in the DLAMI: **no `nvcc`** (install
`cuda-toolkit-13-2` from the sbsa repo) and **no Rust** (vLLM's `vllm-rs` needs
`setuptools_rust`).

**2. The actual blocker — Turing shared memory.** With the build working, the server still
would not start:

```
triton.runtime.errors.OutOfResources: shared memory, Required: 98304, Hardware limit: 65536
Gemma4 model has heterogeneous head dimensions
{'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN.
```

Gemma 4's global-attention layers are **512-wide**; only FA4 or Triton support heterogeneous
head dims; FA4 is unavailable so Triton is **forced** and cannot be overridden. Triton at
`head_size=512` wants ~96 KiB of shared memory per block. **Turing allows 64 KiB per block at
most** — and only if the kernel opts in via the dynamic shared-memory attribute; the default
static limit is 48 KiB, which is what `shared_memory_per_block` reports. Ampere and later have
164 KiB+. This is
the intersection of this model and this chip — not a packaging problem.

**A patch to `vllm/v1/attention/ops/triton_unified_attention.py` clamps the KV tile on
pre-Ampere devices and makes it work.** It is **not upstream and not in this repo** — it
lives only on the instance that was built, and any rebuild must reapply it. It is the
obvious contribution back to vLLM. `docs/turing-aarch64-gap.md` has the diff.

**vLLM must be ≥ v0.27.2rc0.** v0.26.0 dies on Gemma 4 against current `transformers` with
`AmbiguousGlobalPerLayerAttributeError`; the `per_layer_config` fix landed in v0.27.2rc0.

**Run `verify_gpu_arch` first** on any new instance. It costs minutes and tells you which
side of these problems you are on.

## Turing is not L4 — do not copy flags from a sibling

The five `gpu-vllm-l4-*` rigs and the legacy `~/gemma4-tips-aws` tree were all written for
SM 8.9. **Turing has no bf16 *datapath* and no fp8** — but see the corrections below; bf16
is emulated rather than refused.

| | L4 siblings (SM 8.9) | this rig (SM 7.5), measured |
| --- | --- | --- |
| `--dtype` | `bfloat16` | **`float16`** — see the correction below |
| `--kv-cache-dtype` | `fp8` | **`auto`** — no fp8 datapath |
| attention | FlashAttention | **`TRITON_ATTN`, forced by vLLM** — not selectable |
| `--quantization` | `compressed-tensors` (w4a16) | unused, but **not ruled out** |

Four corrections to what this file originally asserted, all from the 2026-08-12 run:

- **bfloat16 is not a hard failure.** PyTorch upconverts on Turing and a bf16 matmul runs;
  vLLM logs `Casting torch.bfloat16 to torch.float16` and proceeds. `float16` is still right
  because it is what executes — but a wrong reason invites someone to test torch, watch it
  pass, and delete the guard.
- **The backend is `TRITON_ATTN`, not XFORMERS**, and vLLM forces it for Gemma 4's
  heterogeneous heads. `VLLM_ATTENTION_BACKEND` is **not a recognized variable** in v0.27 —
  it is silently ignored, so it has been removed from `tpu.env`.
- **w4a16 is not blocked by Marlin.** The build compiled
  `sm75_kernel_float16_u4b8_float16.cu.o`; vLLM ships Turing-specific Marlin kernels.
  Untested here, but the old claim was wrong.
- **The GPU is 15360 MiB, not 16 GB.** Serving E2B used 13501 MiB, leaving a **2.95 GiB KV
  pool = 329,579 tokens**, 20.12x concurrency at 16k context.

This rig serves the **reference bf16 checkpoint**, so its name carries no encoding slot —
vLLM casts it to fp16 at load.

## Sizing and AMI

`g5g.xlarge` is **rejected at validation** on the grounds that 8 GiB of host RAM cannot
stage 9.5 GiB of weights. **That premise is untested** — safetensors loading is mmap-backed,
so peak resident memory is plausibly far under the checkpoint size. The *build* genuinely
needs more (it ran `MAX_JOBS=12` on 16 vCPU / 30 GiB); serving-only on xlarge is untried and
is the obvious next measurement. `g5g.2xlarge` is the default. `g5g.16xlarge` / `g5g.metal`
carry two T4Gs and get `--tensor-parallel-size 2`.

`_resolve_ami` prefers the AWS public SSM parameter for the ARM64 **GPU** DLAMI, which pins
architecture *and* NVIDIA driver. A name filter alone also matches driverless ARM64 DLAMIs,
which boot fine on a G5g with no GPU. **Never hardcode an AMI id.**

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Create no broad network
  or IAM policy.
- Scope instance discovery to `ManagedBy=gpu-vllm-g5g-2b`.
- Hugging Face tokens live in Secrets Manager, fetched at boot. **Never** in user data — a
  test asserts this.
- Launches default to spot. Surface capacity errors rather than silently retrying.
- Termination is permanent and destroys the locally built SM 7.5 image with the root volume.
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.
- `verify_model_health` uses `/v1/chat/completions`; raw `/v1/completions` skips the chat
  template and is unreliable on `-it` models. Measured here: it returns degenerate repetition
  (`': ok: ok: ok...'`), **not** the empty body the monorepo doc describes — so never
  health-check by testing for an empty response.

## There is a prebuilt AMI — use it, do not rebuild

**`ami-0b44b90b3d02430ee`** (us-east-1, 80 GiB). Carries the ~67-minute from-source build, so
a launch is **~24 minutes** instead of hours: CUDA 13.2, vLLM v0.27.2rc0 for `sm_75`, **the
Turing shared-memory patch**, the E2B model cache (`HF_HUB_OFFLINE=1`), `vllm.service` and
`vllm-swap.service`. (This read "~4 minutes" until 2026-08-31; see the startup bullet.)

- The Turing patch is **not upstream** — reapply on any vLLM upgrade or the engine won't start.
- Swap is created at boot, not baked into the image.
- ~$2/month (EBS bills written blocks, not the 80 GiB nominal). Not the Archive tier: restore
  takes 24–72 h. Not a stopped instance: that's ~$6.40/month and still pays full engine init.
- Startup, **MEASURED over 3 cold boots on `g5g.2xlarge` 2026-08-31**: launch → health 200 is
  **1417.8 s = 23m 38s** (median; 1346.8–1525.2, 12.6% spread). **Weight loading alone is 546 s**
  — a 9.54 GiB checkpoint against 11.19 GiB available RAM, so it thrashes page cache even with
  no swapfile. Engine init is 207 s; warm `systemctl restart` → health is 264.3 s. First
  completion is fast, 0.5 s cold. The PyTorch sibling reaches a serving endpoint in **195 s**
  while *downloading* 9.5 GB, so this rig is 7.3x slower from an image that downloads nothing.
  The old text here ("~4 minutes", "`g5g.2xlarge` needs no swapfile and buys that time back")
  was wrong by ~6x. See `benchmarks/runs/2026-08-31-crossrig-vllm-g5g/REPORT.md`.
- Cloning it smaller needs `sgdisk --partition-guid=` as well as the filesystem UUID and
  label: the initramfs boots by `PARTUUID`, and a fresh one leaves the AMI unbootable.

## AWS credentials

`server.py` uses the standard boto3 provider chain. **When credentials expire, refresh with
`./save-aws-creds.sh`**, which re-exports the active credentials to `.aws_creds` (mode 0600).

- It snapshots, it does not mint. `aws configure export-credentials` fails on an expired SSO
  session — re-authenticate (`aws sso login`) first, or the failure looks like a broken script.
- It refuses to write to a non-gitignored path inside a work tree. `.aws_creds` is gitignored
  here for that reason; never remove that line, never use `FORCE=1`.
- Nothing in this rig reads `.aws_creds` automatically — the script's "the Makefile will now
  use these" message is inherited from `~/gemma4-tips-aws` and does not apply. Use
  `AWS_PROFILE` to select a profile for `server.py`.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v`. Fully
offline — no AWS, no network, no GPU.

`make lint` runs `ruff check server.py refresh_skill.py tests` then `bash -n` on the three
shell scripts; a new top-level module is silently unlinted until added to that list.

`make skill` regenerates **only** the three `mcp/` files under `.claude/skills/` and
`skills/`. `SKILL.md` lives in the same tree but is a hand-written source — `rm -rf` on that
directory destroys it and `make skill` will not bring it back.

No `make deploy` recipe on purpose: provisioning resolves an arm64 AMI at launch time.

## Measurement

**One measurement, its own:** `benchmarks/runs/2026-08-12-first-serve-g5g/`. Single run,
single stream, no repeats, no variance — do not quote 43 tok/s as a characterisation of the
hardware, and note it was taken with reduced Triton tiles. `benchmarks/` otherwise holds
synced copies of the root schema and README — edit the root originals, never these. Runs go
in `benchmarks/runs/<date>-<what>-g5g/`.

The ~44 tok/s that `~/gemma4-tips-aws` records for E2B on one Inferentia core is **not** a
comparison: different harness, different silicon.
