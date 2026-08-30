# GEMINI.md — `gpu-pytorch-g4dn-2b`

Guidance for coding agents working in this rig. **`CLAUDE.md` is authoritative where these
disagree**; there is no generator, so a convention change has to be applied to all three of
`CLAUDE.md`, `AGENTS.md` and `GEMINI.md` by hand.

## What this rig is

Serves **`google/gemma-4-E2B-it`** under **PyTorch + `transformers`** on **AWS EC2 G4dn** —
an x86_64 Intel host with an **NVIDIA T4** GPU (Turing, SM 7.5, **15360 MiB measured**,
14.07 GiB usable — not the 16384 the EC2 API implies).

Plain CUDA PyTorch, not `torch_xla`: the model is `AutoModelForCausalLM`, driven by
`torch_openai_server.py` (with `torch_generate.py` as the out-of-band smoke test), under
**systemd, not docker**. Read logs with `journalctl`, never `docker logs`.

**No vendored model port, no XLA, nothing compiled on the instance.**

## THIS RIG HAS SERVED NOTHING

Forked from `gpu-pytorch-g5g-2b` on 2026-08-29 and retargeted from Graviton2 to x86_64.
`benchmarks/` is empty. **There are no MEASURED values in this rig** — every number is read
from an AWS API, taken from a vendor spec, or inherited and labelled.

**Read `docs/INHERITED.md` before quoting anything.** This rig is two forks from where most of
its inherited prose was written, and the two forks changed different axes.

## Where it sits, and why that decides what carries

|  | G5g (Graviton2, aarch64) | G4dn (x86_64) |
| --- | --- | --- |
| pure JAX | `gpu-jax-g5g-2b` | `gpu-jax-g4dn-2b` |
| PyTorch | `gpu-pytorch-g5g-2b` | **this rig** |

Same Turing generation in both columns (T4G / T4, both SM 7.5, both **15360 MiB**), so the column is
the **host** and the row is the **runtime**. A claim about the **GPU** carries; a claim about
the **host** or the **runtime** must be checked against which fork it predates.

## Hard constraints

- **Turing has no bf16 and no fp8.** The device decides the compute dtype, not `tpu.env`:
  `resolve_compute_dtype()` reads the live compute capability and picks `float16` below
  SM 8.0. **bfloat16 does not raise — CUDA emulates it through fp32**, which is worse than an
  error. `DTYPE=float16` in `tpu.env` is a record of that policy, not the input to it.
- **The guard is duplicated** in `torch_openai_server.py` and `torch_generate.py` on purpose;
  either can be run alone. `tests/test_engine.py` asserts they agree.
- **Never hardcode `dtype=torch.bfloat16`** anywhere in the payload. A test scans for it,
  because that substitution sails past every test of the resolver itself.
- **Torch comes from the AMI, never from pip.** The bootstrap installs INTO the DLAMI's own
  PyTorch venv, found by probing for an interpreter that can `import torch`. A box with no
  torch is a **wrong AMI** (a base driver-only image), not a missing `pip install torch`.
- **Nothing compiles.** No `torch.compile`, so no compilation cache. The JAX cache knobs
  (`JAX_COMPILATION_CACHE_DIR`, `JAX_CACHE_S3_URI`, `XLA_PYTHON_CLIENT_MEM_FRACTION`) and the
  inert JAX-engine knobs (`PLE_BITS`, `INT8_LM_HEAD`, `PREFILL_CHUNK_SIZE`) were **removed**,
  and tests assert they stay removed. If `torch.compile` is adopted, use
  `TORCHINDUCTOR_CACHE_DIR`.
- **Dense reference checkpoint only**, so the rig name carries no encoding slot — but **not
  for the JAX rigs' reason.** There is no Pallas here; `AutoModelForCausalLM` has nowhere to
  put w4a16 weights without bitsandbytes or torchao. Do not repeat the Pallas argument.

## Things that look like the sibling and are not

- **The AMI name filter is architecture-specific.** AWS orders the words differently:
  `Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch …` vs
  `Deep Learning OSS Nvidia Driver AMI GPU PyTorch …`. The sibling's pattern matches **zero**
  x86_64 images. `DLAMI_NAME` and `DLAMI_SSM_PARAMETER` change together.
- **`TORCH_PYTHON_VERSION=3.14`**, not 3.12, and it is forced by the AMI: deadsnakes publishes
  `python3.14` for jammy and noble only, so a 26.04 image must not ask for 3.12.
- **The size suffix does not give the GPU count.** `g4dn.16xlarge` has ONE T4;
  `g4dn.12xlarge` has four.
- **vCPU is RAM/4, not RAM/2.** The sibling's derivation is wrong on every g4dn size, and the
  G-family quota is counted in vCPUs.
- **Every g4dn size is supported** — this family starts at 16 GiB, so nothing is rejected.
  Only `g4dn.xlarge` gets a swapfile, and that is the default, so the swap path is exercised
  from the first launch rather than latent.

## Engineering rules

- boto3 and the standard credential chain — **never shell out to the AWS CLI.**
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Do not create broad
  network or IAM policy to make a launch succeed.
- Scope instance discovery to `ManagedBy=gpu-pytorch-g4dn-2b`.
- HF tokens come from Secrets Manager at boot into a root-only `EnvironmentFile`. **Never** in
  user data. `set +x` wraps the fetch because `set -x` traces assignments with their values.
- Never hardcode an endpoint or an AMI id.
- `verify_model_health` uses `/v1/chat/completions`. **Do not health-check by testing for a
  non-empty response** — the vLLM sibling returned `': ok: ok: ok…'`, which is not empty and
  not healthy.
- `init.sh` blocks on `read` in its error path — never run it non-interactively.

## Commands

- Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v`.
  **81 tests, none skipped.** Keep it that way — the inherited `test_engine.py` skipped all 22
  of its tests forever, and a suite that always skips reads as coverage while asserting
  nothing.
- `make lint` — ruff over `server.py refresh_skill.py torch_generate.py torch_openai_server.py
  tests`, then `bash -n` on four shell scripts. **A new top-level module is silently unlinted
  until it is added to that list.**
- `make skill` regenerates both snapshot copies. **Always run it before
  `deploy_torch_server`**, which ships the snapshot rather than the working tree and reports
  success either way.
- `SKILL.md` is a hand-written source that `refresh_skill.py` will not recreate.
- No `make deploy` and no `make medium`, both deliberately.

## Order of operations

```
create_g4dn_instance → get_install_progress → verify_gpu_arch → deploy_torch_server
                     → get_torch_logs → verify_model_health
```

**Run `verify_gpu_arch` first on any new instance.** It prints `torch.cuda.get_arch_list()`
next to the measured capability and runs a real fp16 matmul. On this rig it is also the only
thing that settles what the image's torch build covers — the sibling's "upstream wheels omit
sm_75" was measured for aarch64 and is **not** established for x86_64.

## Measurement

`benchmarks/` is empty and stays empty until this rig serves something. Runs go in
`benchmarks/runs/<date>-<what>-g4dn/`.

Never quote as this rig's own: **12.4–13.1 tok/s** (`gpu-jax-g5g-2b`), **43–44 tok/s**
(`gpu-vllm-g5g-2b`, and with reduced Triton tiles), **~44 tok/s** (Inferentia, different
harness), or anything from `~/gemma4-tips`.

A config flag being accepted is not evidence it did anything. Cross-check against a physical
bound — 320 GB/s and 14.07 GiB usable is the whole envelope.
