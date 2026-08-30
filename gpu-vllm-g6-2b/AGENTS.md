# AGENTS.md — `gpu-vllm-g6-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **vLLM** on **AWS EC2 G6** — an x86_64 host
with an **NVIDIA L4** GPU (Ada, SM 8.9, 23034 MiB measured on the same silicon by the JAX
sibling).

**Status: SERVED 2026-08-30** on `g6.2xlarge` spot in `us-east-1d`, torn down the same
session. `benchmarks/runs/2026-08-30-first-serve-g6/` and
`benchmarks/reports/2026-08-30-gemma4-e2b-g6.json` are its own. **46.09 tok/s single-stream,
360.17 tok/s at concurrency 8.** Both fork premises held.

**`CLAUDE.md` is authoritative where this file disagrees with it.** There is no generator;
a convention change has to land in `CLAUDE.md`, `AGENTS.md`, and `GEMINI.md` by hand.

## Why this rig exists

It is the runtime control for `gpu-jax-g6-2b`, which MEASURED **48.3–48.5 tok/s** on this
exact chip under pure JAX on 2026-08-28. Same silicon, same checkpoint, different runtime.
The T4G pair was never a clean comparison, because the vLLM side there ran with hand-reduced
Triton tiles.

## The fork removed the sibling's hard part

G5g needs **aarch64 and SM 7.5 together**, and no published CUDA artifact has both: the
`vllm/vllm-openai` arm64 manifest is compiled `8.0 8.7 8.9 9.0 10.0 11.0 12.0` and only the
amd64 manifest carries 7.5, with no `+PTX` to JIT from. That costs the sibling a ~67-minute
from-source build, a CUDA toolkit, a Rust toolchain and an unlanded Triton patch.

**G6 is x86_64 and SM 8.9 — both covered by the published amd64 image.** No build, no
toolkit, no Rust, no prebuilt AMI, and no `serving=` mode.

**Do not copy `ami-0b44b90b3d02430ee`** into anything here: that is the sibling's arm64
SM 7.5 image and cannot boot a G6.

## What might still bite

Gemma 4's heterogeneous head dims (sliding 256, global **512**) force `TRITON_ATTN`, whose
tile wants **~96 KiB of shared memory per block**. Turing caps a block at 64 KiB; **Ada
allows ~99 KiB**. **SETTLED 2026-08-30: it fits unpatched** — no `OutOfResources`. Do not
port the sibling's Triton patch. The ~3 KiB margin is still narrow, so this holds for *this*
tile at *this* head size, not Ada generally.

`VLLM_ATTENTION_BACKEND` is left unpinned — pinning is how the sibling ended up with a patch.
Measured there: vLLM v0.27 does not recognize the variable at all and forces `TRITON_ATTN`
for this model regardless.

## Ada is not Turing

| | T4G sibling | **this rig** |
| --- | --- | --- |
| compute dtype | `float16` | **`bfloat16`** |
| KV cache | `auto` → float16 | **`auto` → bfloat16**; fp8 reachable, unused |
| device memory | 15360 MiB | **23034 MiB** |
| per-block shared memory | 64 KiB | **~99 KiB** |

`DTYPE=bfloat16` because **the checkpoint is bf16** — float16 would make vLLM convert every
weight on load, a mismatch the JAX rig measured at 54% of decode on Turing.

## Traps carried by the fork

- **`VLLM_IMAGE` is `v0.28.0`, and `v0.27.2rc0` IS NOT A PUBLISHED TAG.** MEASURED
  2026-08-30: the tag this rig shipped with does not resolve, and cloud-init died at
  `failed to resolve reference ... not found`. It is the sibling's **`VLLM_REF`** — a git ref
  it compiled from source — copied into an image-tag field. Releases go `v0.27.1` →
  `v0.28.0`; there is no `v0.27.2`. The floor is on the **fix** (v0.26.0 dies with
  `AmbiguousGlobalPerLayerAttributeError`, Gemma 4's `head_dim` being per-layer), not on that
  string. The guard test is now an **allowlist of tags verified to resolve** — a blocklist
  passes anything it has not heard of.
- **Host RAM doubled at every size suffix.** `g6.xlarge` has 16 GiB where `g5g.xlarge` had 8,
  so the sibling's rejection of xlarge does not carry.
- **`g6.16xlarge` is single-GPU** where `g5g.16xlarge` had two. GPU count is not monotonic in
  the size name.
- **G6 is 4 GiB/vCPU; G5g was 2.** Any inherited `RAM // 2` vCPU shortcut doubles.
- **`mkswap -q` is a busybox flag util-linux rejects.** Under `set -e` it killed cloud-init
  before anything logged on the sibling. The swap block is dead code here (no G6 size trips
  the gate) and therefore untested; a test guards the flag.
- **Changing `DLAMI_SSM_PARAMETER` requires changing `DLAMI_NAME` in the same commit**, or
  the fallback silently resolves a different image and reports success.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command; no inbound SSH rule, no private key.
- Require explicit subnet, security-group and instance-profile ids. Do not create broad
  network or IAM policy.
- Scope discovery to `ManagedBy=gpu-vllm-g6-2b`.
- HF tokens live in Secrets Manager, fetched at boot behind `set +x`. **Never** in user data.
- Never hardcode an AMI id or an endpoint.
- Launches default to spot; surface capacity errors rather than retrying silently.
- **Termination is cheap here** — nothing is built. Do not import the sibling's stop-vs-
  terminate or AMI-maintenance reasoning.
- **Do not health-check by testing for a non-empty response** — a broken deploy in this
  lineage answered `': ok: ok: ok…'`.

## Commands

`python3 -m unittest discover -s tests -v` — 38 offline tests, no AWS/network/GPU.
`make lint`, `make skill`. There is no `make deploy`: the AMI is resolved at launch.

## Measurement

**None.** Naming will be `benchmarks/runs/<date>-<what>-g6/`, where `<hw-short>` is the
hardware **measured**. Do not reuse the JAX rig's 48.3–48.5 (different runtime — that is the
comparison), the T4G sibling's 43.1/44.24 (different silicon — and **corrected 2026-08-30: neither is a
benchmark.** 43.1 is a single first-serve sample, 44.24 has no artifact and was a swap smoke
test. The measured T4G figures are c=1 TPOT 31.44 ms / ~31.8 tok/s and c=8 168.33 tok/s, from
`gpu-vllm-g5g-2b/benchmarks/runs/2026-08-14-rust-frontend-g5g/`), or anything
from the `gpu-vllm-l4-*` artifact rigs (**same GPU and runtime, weakest provenance in the
tree** — same chip is not same measurement).
