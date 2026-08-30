# AGENTS.md — `gpu-vllm-g4dn-2b`

Guidance for coding agents working in this rig. **`CLAUDE.md` is authoritative where these
disagree**; there is no generator, so a convention change has to be applied to all three of
`CLAUDE.md`, `AGENTS.md` and `GEMINI.md` by hand.

## What this rig is

Serves **`google/gemma-4-E2B-it`** under **vLLM** on **AWS EC2 G4dn** — x86_64 (Intel) host,
**NVIDIA T4** GPU (Turing, SM 7.5, 16 GB nominal / 15360 MiB measured on the T4G sibling).

**It has served nothing.** The directory was a stale copy of `gpu-jax-g4dn-2b`; the vLLM side
was forked from `gpu-vllm-g6-2b` on 2026-08-29. Every number is arithmetic or inherited.

## The one fact that explains the whole rig

`gpu-vllm-g5g-2b` hits **two independent problems**. This rig keeps one and deletes the other.

| | SM 7.5 in the published image? | Triton tile vs Turing's 64 KiB |
| --- | --- | --- |
| `gpu-vllm-g5g-2b` — aarch64, SM 7.5 | **NO** → ~67-min build | **NO** → tile clamp |
| `gpu-vllm-g6-2b` — x86_64, SM 8.9 | yes | yes (~99 KiB), unverified |
| **this rig** — x86_64, SM 7.5 | **YES** | **NO** → tile clamp |

The `linux/amd64` manifest of `vllm/vllm-openai` is compiled for
`7.5 8.0 8.6 8.9 9.0 10.0 12.0`; `linux/arm64` for `8.0 8.7 8.9 9.0 10.0 11.0 12.0`. G4dn is
Intel, so it gets the one with 7.5. **No build, no CUDA toolkit, no Rust, no AMI to bake.**

The Turing half is untouched: Gemma 4's 512-wide global heads force a Triton tile wanting
98,304 B per block against Turing's 65,536. `docs/turing-shared-memory.md` is the write-up.

## Things that will bite you

- **Do not copy the fork parent's dtype.** `gpu-vllm-g6-2b` is Ada and defaults to
  `bfloat16`. Turing has no bf16 and no fp8 datapath. bfloat16 **does not error** — PyTorch
  upconverts and vLLM logs `Casting torch.bfloat16 to torch.float16` — so the wrong value is
  silent.
- **Do not quote the JAX rigs' "54% of decode" dtype tax.** That was a JAX loader converting
  at every *use*, per step. vLLM converts once at load.
- **`VLLM_ATTENTION_BACKEND` is not a knob.** vLLM v0.27 does not recognize it and forces
  `TRITON_ATTN` for this model regardless. The tile size inside that kernel is the knob.
- **GPU count is not monotonic in the instance size.** `g4dn.12xlarge` has 4, `16xlarge` has
  1, `metal` has 8. Never infer it from the suffix.
- **G4dn is 4 GiB of RAM per vCPU; G5g was 2.** Any inherited `RAM // 2` shortcut doubles.
- **The swap gate deliberately differs from `gpu-jax-g4dn-2b`** on the same instance type:
  strictly-below-16 here, at-or-below-16 there. That rig OOMs at 16 GiB inside its own JAX
  loader (`quantize_ple_table`); vLLM has no such step. Do not harmonise them.
- **`make skill` generates FOUR files**, including `patch_triton_turing.py`. An installed
  skill copy without it cannot launch an instance at all. `SKILL.md` is a hand-written source
  and is **not** regenerated.
- **A Codex approval gate on a tool name that does not exist fails open silently.** All tools
  are `*_g4dn_*` here; a test checks the gates.
- **The image tag this rig inherited did not exist.** `gpu-vllm-g6-2b` pins
  `vllm/vllm-openai:v0.27.2rc0`; it is a 404 on Docker Hub and not a git tag. Now `v0.28.0`.
  It survived because the version-floor test asserted "not v0.27.1, not v0.26" — trivially
  true of a tag nobody published. **A floor test that never checks the artifact exists is
  "an accepted flag is not evidence" one level up.**
- **Do not anchor the Triton patch on the kernel launch.** Upstream copies the tile constants
  into `tile_size` and `launch_num_stages` into `launch_kwargs` before launching, so a clamp
  there rewrites variables nothing reads — and every check this rig has would still pass.
  The insertion point is derived: after the last tile assignment, before the first read.
  VERIFIED against real v0.28.0 source 2026-08-29.

## Commands

- Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v`. Fully
  offline — no AWS, no network, no GPU, no docker.
- `make lint` runs `ruff check server.py patch_triton_turing.py refresh_skill.py tests` then
  `bash -n` on four shell scripts. **A new top-level module is silently unlinted until it is
  added to that list.**
- There is no `make deploy`: provisioning resolves an x86_64 AMI at launch time.

## Order of operations

```
check_g4dn_quotas → create_g4dn_instance → get_install_progress
                  → verify_gpu_arch → verify_triton_patch → verify_model_health
```

`verify_gpu_arch` passing says **nothing** about `verify_triton_patch`. The two problems are
independent, and only the first is gone on this hardware.
