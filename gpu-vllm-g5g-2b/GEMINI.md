# GEMINI.md — `gpu-vllm-g5g-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **vLLM** on **AWS EC2 G5g** — a Graviton2
(aarch64) host with an **NVIDIA T4G** GPU (Turing, SM 7.5, 16 GB).

**`CLAUDE.md` is authoritative where this file disagrees with it.** There is no generator;
a convention change has to land in `CLAUDE.md`, `AGENTS.md`, and `GEMINI.md` by hand.

## The one thing to know before touching anything here

**G5g needs aarch64 and SM 7.5 together, and no prebuilt CUDA artifact provides both.**
Read from the published `vllm/vllm-openai:v0.27.1` image config on 2026-08-12:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? |
| --- | --- | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | **no** |

The one arch this rig needs is the only one the two images disagree about, and the
Dockerfile sets **no `+PTX`**, so nothing JIT-compiles to cover the gap.
`docs/turing-aarch64-gap.md` has the reproduction and what is still unverified.

- `serving='build'` (default) compiles vLLM on the instance with
  `--build-arg torch_cuda_arch_list=7.5`. **Hours** on a Graviton2. Do not simplify it back
  to a plain `docker run` of the published image.
- `serving='stock'` runs the published image unchanged and is **expected to fail**. It is
  apparatus for reproducing the gap, not a fallback.
- **Run `verify_gpu_arch` first.** It settles in minutes what the build path takes hours to
  discover. A config flag being accepted is not evidence it did anything.

## Turing is not L4 — do not copy flags from a sibling

The `gpu-vllm-l4-*` rigs and `~/gemma4-tips-aws` were written for SM 8.9. **Turing has no
bf16 and no fp8.**

| | L4 siblings (SM 8.9) | this rig (SM 7.5) |
| --- | --- | --- |
| `--dtype` | `bfloat16` | **`float16`** — bfloat16 is a hard failure here |
| `--kv-cache-dtype` | `fp8` | **`auto`** — no fp8 datapath |
| attention | FlashAttention | **`XFORMERS`** — FA needs SM 8.0+ |

This rig serves the reference bf16 checkpoint, so its name carries no encoding slot: E2B is
9.5 GiB against 16 GB, which leaves room for a real KV pool at 18 KiB/token.

## Sizing and AMI

`g5g.xlarge` is **rejected at validation** — 8 GiB of host RAM cannot stage 9.5 GiB of
weights. `g5g.2xlarge` is the floor and default. `g5g.16xlarge` / `g5g.metal` carry two
T4Gs and get `--tensor-parallel-size 2`.

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
- `verify_model_health` uses `/v1/chat/completions` — raw `/v1/completions` returns an empty
  completion on `-it` models.

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

**This rig has no measurements of its own, and none may be attributed to it.** `benchmarks/`
holds synced copies of the root schema and README — edit the root originals, never these.
First run goes in `benchmarks/runs/<date>-<what>-g5g/`, and the first thing worth recording
is the `verify_gpu_arch` output.
