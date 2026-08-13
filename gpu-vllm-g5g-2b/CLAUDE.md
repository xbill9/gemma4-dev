# CLAUDE.md — `gpu-vllm-g5g-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **vLLM** on **AWS EC2 G5g** — a Graviton2
(aarch64) host with an **NVIDIA T4G** GPU (Turing, SM 7.5, 16 GB).

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`.
It is **not** one of the `gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform
slot with them.

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
  is what makes this rig hard. The whole build problem is aarch64 **and** SM 7.5 together.

The chip is still called T4G everywhere it is the chip being discussed — in this file, in
`HARDWARE.md`, and in `tpu.env`. Only the slot is `g5g`.

## This rig has now served, and the docs below are measured

**2026-08-12: Gemma 4 E2B serves on a T4G at ~43 tok/s.** Full run, environment, and the
patch it required: `benchmarks/runs/2026-08-12-first-serve-g5g/REPORT.md`. Everything in
this file that used to be a prediction has been replaced with what the hardware did — several
of the original claims were wrong and are corrected below.

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
SM 8.9. **Turing has no bf16 and no fp8.**

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

## Instance sizing

`g5g.xlarge` is **rejected by `_validate_instance_type`** on the grounds that 8 GiB of host
RAM cannot stage E2B's 9.5 GiB of weights. **That premise is still untested** — it was never
measured, and safetensors loading is mmap-backed, so peak resident memory is plausibly well
under the checkpoint size. Treat the guard as unvalidated.

What *is* known: the **build** needs far more than xlarge offers — the 2026-08-12 build ran
`MAX_JOBS=12` on 16 vCPU / 30 GiB. A serving-only xlarge is the interesting case (spot
$0.128/hr against $0.350 for 2xlarge, same 15360 MiB GPU on both) and is the obvious next
measurement. `g5g.2xlarge` is the current default.

`g5g.16xlarge` and `g5g.metal` carry two T4Gs; `_tensor_parallel_size` derives TP from the
GPU count, so those get `--tensor-parallel-size 2`. Every other size gets 1.

## AMI resolution

`_resolve_ami` filters on `architecture=arm64`. This is load-bearing. The legacy tips-tree
rigs hardcode `ami-012ba162b9cd2729c`, an **x86_64** DLAMI that cannot boot on Graviton2.
Never hardcode an AMI id here; resolve it at launch.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Do not create broad
  network or IAM policy. (The legacy sample this was scaffolded from auto-creates a security
  group open to `0.0.0.0/0` on 22 and 8080 — that was not carried over.)
- Scope instance discovery to `ManagedBy=gpu-vllm-g5g-2b`. Unlike the inf2 rig, which keeps
  a legacy tag to avoid orphaning instances, this rig is new and uses its own name.
- Hugging Face tokens live in Secrets Manager and are fetched at boot. **Never** in user
  data — instance metadata is readable by anything on the box. A test asserts this.
- Launches default to spot. Surface capacity errors rather than silently retrying.
- Termination is permanent, and here it also destroys the locally built SM 7.5 image with
  the root volume — the next launch rebuilds from source. Weigh stop against terminate.
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.
- `verify_model_health` uses `/v1/chat/completions`, because raw `/v1/completions` skips the
  chat template and is unreliable on `-it` models. **Measured here 2026-08-12, and it does
  not match the symptom the monorepo `CLAUDE.md` describes:** raw completions returned
  `': ok: ok: ok: ok: ok: ok: ok: ok'` — degenerate repetition, 16 tokens — not the empty
  body documented for the TPU rigs. Either way it is not evidence about deploy health, but
  do not health-check by testing for an empty response: on this rig you would get a
  non-empty body full of garbage and call it fine.

## AWS credentials

`server.py` uses the standard boto3 provider chain, so whatever `aws sts get-caller-identity`
resolves is what the rig gets. **When credentials expire, refresh them with
`./save-aws-creds.sh`**, which re-exports the active credentials to `.aws_creds` at mode 0600.

Three things about it that are easy to get wrong:

- **It snapshots credentials, it does not mint them.** `aws configure export-credentials`
  fails outright on an expired SSO session, so re-authenticate first (`aws sso login`) and
  then run the script. Its error message says this; the failure otherwise reads as a broken
  script rather than an expired login.
- **It refuses to write anywhere inside a git work tree that is not gitignored.** `.aws_creds`
  is in this rig's `.gitignore` for exactly that reason. Never remove that line and never
  reach for `FORCE=1` — the guard is the thing keeping live keys out of a commit.
- **Nothing in this rig reads `.aws_creds` automatically.** The script's closing message
  ("the Makefile will now use these") is inherited from the legacy `~/gemma4-tips-aws` tree,
  whose Makefile loaded the file; this rig's does not. The snapshot is for exporting into a
  shell or handing to a container. For `server.py` itself the provider chain is enough, and
  `AWS_PROFILE` is the supported way to pick a profile.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v`. They are
fully offline — no AWS, no network, no GPU — and pin the facts above, including that
`tpu.env` and `server.py` still agree.

`make lint` runs `ruff check server.py refresh_skill.py tests` then `bash -n` on the three
shell scripts. **A new top-level module is silently unlinted until it is added to that
list.**

`make skill` regenerates the snapshots under `.claude/skills/` and `skills/`. **Only the
three `mcp/` files are generated** — `server.py`, `project-setup.sh`, `requirements.txt`.
`SKILL.md` sits in the same tree and is a hand-written **source**: `refresh_skill.py` will
not recreate it. So `rm -rf .claude/skills` destroys it permanently, which is what happened
during the t4g→g5g rename. `test_skill_is_complete_in_both_copies` now guards it.

## MCP registration lives in four places

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and
`.claude/settings.local.json`'s `enabledMcpjsonServers`. All four name the server
`gpu-vllm-g5g-2b`, which prefixes every tool as `mcp__gpu-vllm-g5g-2b__…`. `.mcp.json` and
`settings.local.json` are gitignored and generated by `project-setup.sh`; the other two are
committed. Keep them agreeing — a mismatch makes `/mcp` and the tool prefix disagree about
what this rig is.

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree**, and a convention change has to be
applied to all three by hand.

There is no `make deploy` recipe on purpose: provisioning resolves an arm64 AMI at launch
time, and a Makefile would have to hardcode one.

## Measurement

**This rig has exactly one measurement**, and it is its own:
`benchmarks/runs/2026-08-12-first-serve-g5g/` — the first successful serve, on
`g5g.4xlarge` spot in `us-east-1a`. Single run, single stream, no repeats, no variance
figure. One sample per cell; do not quote 43 tok/s as a characterisation of the hardware.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them, so edit the root originals,
never these. `reports/` and `runs/` stay in the rig. The L4 4B artifacts that arrived with
the directory it was scaffolded from were deleted rather than left to be counted against it
by `benchmarks/rollup.py` — the same call made for `tpu-vllm-v5p1-2b`.

Naming is `benchmarks/runs/<date>-<what>-g5g/` — `<hw-short>` equals the hardware slot.

The ~44 tok/s that `~/gemma4-tips-aws` records for E2B on one Inferentia core is a tempting
comparison and **is not one**: different harness, different silicon, and this figure was
measured with reduced Triton tiles. Do not put them in the same table.
