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

**Twice, and both runs are its own.** `benchmarks/runs/2026-08-19-first-serve-g5g/` is the
first serve; `benchmarks/runs/2026-08-21-cuda13-py314-g5g/` repeats it on the CUDA 13 /
Python 3.14 stack. The headline is **~12 tok/s single-stream** on `g5g.2xlarge`.

Two things that still hold and are easy to lose:

- **Most numbers you might want are still absent or belong to a sibling.** Two prompts, one
  concurrency, one context. `tpu.env` still marks values MEASURED or PREDICTED; respect the
  split and do not quote a PREDICTED number as a result. The KV-ceiling estimate in
  particular remains PREDICTED — the serving runs did not test it.
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

`g5g.xlarge` (8 GiB) needs a swapfile, and `_user_data` provisions one automatically below
`_SWAP_BELOW_HOST_RAM_GB = 16`. Measured on the vLLM sibling 2026-08-13, same checkpoint and
same host, so it carries: without swap the kernel refuses to **mmap** the 10.2 GB checkpoint
at all —

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

**There is no prebuilt AMI here and none is owed.** The vLLM sibling needs one because it
carries a 67-minute build; this install is `pip install`, so a stock DLAMI is the right base.
Do not copy that rig's `ami-0b44b90b3d02430ee` into anything here — it is a vLLM image with
a Triton patch and no JAX.

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

`jax >= 0.11` needs Python 3.12 and the Ubuntu 22.04 DLAMI base ships 3.10, so the bootstrap
installs a 3.12 interpreter from deadsnakes. It deliberately does **not** install into the
DLAMI's own PyTorch environment: that ships its own CUDA libraries and `jax[cuda12]` brings
its own.

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

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v` (76 tests,
all passing as of 2026-08-24). They are fully offline — no AWS, no network, no GPU — and pin
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

## Measurement

**This rig has two measurements**, both its own, in `benchmarks/runs/<date>-<what>-g5g/`
where `<hw-short>` equals the hardware slot:

| Run | Decode | Note |
| --- | ---: | --- |
| `2026-08-19-first-serve-g5g` | 12.5 tok/s | First serve. CUDA 12 / Python 3.12. |
| `2026-08-21-cuda13-py314-g5g` | 12.4 tok/s | CUDA 13 / Python 3.14, same AMI. |

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

## Fork debris still to clean up

This rig was forked from `gpu-vllm-g5g-2b` and the code was rewritten before the prose was.
The registration files were repaired on 2026-08-18. Still stale:

- **`README.md`** — titled `gpu-vllm-g5g-2b` and describes the vLLM runtime throughout.
- **`AGENTS.md`, `GEMINI.md`** — same, byte-identical vLLM copies.
- **Monorepo `NAMING.md` and `README.md`** — have no entry for `gpu-jax-g5g-2b` at all, and
  `NAMING.md`'s `g5g` carve-out names only the vLLM rig. (The root `marketplace.json` does
  have one.)

Cleared 2026-08-23, listed here because the *class* of error keeps recurring — TPU-rig prose
describing precision this chip cannot run:

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
