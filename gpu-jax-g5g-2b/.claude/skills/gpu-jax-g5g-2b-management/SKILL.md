---
name: gpu-jax-g5g-2b-management
description: Manage AWS EC2 G5g capacity (Graviton2 + NVIDIA T4G) and Gemma 4 E2B serving under pure JAX. Use when the user asks about provisioning, launching, listing, or terminating G5g instances, installing or debugging JAX on T4G / Turing / aarch64, G-family quotas, or the gpu-jax-g5g-2b devops MCP agent. Triggers include "G5g", "T4G", "Graviton", "Turing", "SM 7.5", "arm64 GPU", "JAX on GPU", "jax[cuda12]".
---

# gpu-jax-g5g-2b management

Provision and operate **EC2 G5g** (AWS Graviton2 host + NVIDIA **T4G** GPU) serving
`google/gemma-4-E2B-it` under **pure JAX**, through the `gpu-jax-g5g-2b` MCP server.

This is the JAX sibling of `gpu-vllm-g5g-2b`. Same hardware, different runtime, and the
runtime is the whole point.

## What has been measured

This rig **has served, and the numbers below are its own**: 12.4-12.5 tok/s decode on
`g5g.2xlarge` in the two runs recorded under `benchmarks/runs/` (an unrecorded spot-check
on 2026-08-23 read 12.3). Quote the server's own
`tpu_jax_decode_tokens_per_second` gauge (`get_metrics`) rather than an end-to-end rate,
which also carries prefill and the HTTP round trip. **Warm up before recording anything** —
cold decode measures several times slower, and cold prefill far worse again.

Still do not attribute a sibling's numbers to this rig. And note the vLLM sibling's
"~43-44 tok/s" is **not a benchmark** (corrected 2026-08-30) — 43.1 was a single first-serve
sample and 44.24 a swapfile smoke test with no artifact. Its measured figures are c=1 TPOT
31.44 ms (~31.8 tok/s decode) and c=8 168.33 tok/s, in
`gpu-vllm-g5g-2b/benchmarks/runs/2026-08-14-rust-frontend-g5g/`.

## Start here, every time

**Run `verify_gpu_arch` before anything else on a new instance.** It reports nvidia-smi's
view, JAX's device list and compute capability, and the result of one real fp16 matmul. It
asks *JAX*, not torch — the DLAMI's torch already carries sm_75, which says nothing about
jaxlib.

## Why this rig exists

The vLLM path on identical hardware works, but only through a ~67-minute from-source build,
a CUDA toolkit, Rust, and **a patch to Triton's attention kernel that is not upstream**. JAX
needs none of them:

| | vLLM path | this rig |
| --- | --- | --- |
| aarch64 + SM 7.5 wheels | absent — build from source | **published** |
| CUDA toolkit on the host | required (`cuda-toolkit-13-2`) | **not needed** — pip ships CUDA |
| Rust | required (`vllm-rs`) | not needed |
| attention | `TRITON_ATTN`, forced, needs a patch | **ordinary XLA** |
| provision time | hours (or a prebuilt AMI) | a pip install |

Verified 2026-08-18: `jaxlib`, `jax-cuda12-plugin` and `jax-cuda12-pjrt` all publish
`manylinux_2_27_aarch64` wheels; the plugin's arch tables contain `sm_75` and its floor is
SM 6.0; and every CUDA dependency (cublas, cudnn, cuda-runtime, cusolver, nvcc) publishes
aarch64 wheels, so the DLAMI only has to supply the driver.

## The ceiling that still bites

Turing allows **64 KiB of shared memory per block**. Ampere and later have 100–227 KiB. That
single number is what makes Gemma 4 hard on this chip, and it did not go away — it moved:

- **In vLLM** it hits attention. Gemma 4's head dims are heterogeneous (sliding 256, global
  512), only FA4 or Triton handle that, FA4 is unavailable, so Triton is *forced* — and at
  `head_size=512` it wants ~96 KiB. Hence the patch.
- **In JAX** attention is plain XLA, so there is nothing to patch. But the fused **W4A16
  Pallas kernel** lowers through Triton on GPU and its tiles become shared memory: it needs
  **550 KiB – 1.1 MiB** per block at this model's shapes, 8.6x to 17.7x over budget.

So this rig serves the **dense reference checkpoint at float16**, and
`check_w4a16_fits_scoped_memory()` refuses the fused kernel at startup with the arithmetic
attached rather than dying on the first token.

## Turing flag rules

T4G is Turing (SM 7.5), **not** the Ada L4 every sibling GPU rig here targets.

- **`float16`, not `bfloat16`.** bf16 does not fail on Turing — XLA *emulates* it through
  fp32 conversions. Correct numbers, quiet slowdown, which is worse than an error. The port
  picks the dtype from the live compute capability; `DTYPE` in `tpu.env` is only an override.
- **`--kv-cache-dtype auto`**, which resolves to float16 here. **Never `fp8`** — no datapath,
  and `resolve_cache_dtype` raises rather than downgrading silently. Use `int8` to halve KV;
  it carries a per-row scale.
- **`--quant-mode` matches the checkpoint, not the chip.** `fp16` for the dense reference
  build, `w4a16` only for a `-w4a16-` export. `auto` reads it off the checkpoint name.
  Getting this wrong loads garbage rather than failing.
- **No tensor-parallel flag.** The JAX engine is single-device. On `g5g.16xlarge` and
  `g5g.metal` the second T4G idles.

Copying a flag set from a `gpu-vllm-l4-*` rig, from `~/gemma4-tips-aws`, or from the TPU JAX
rig will produce a config that is accepted and then wrong.

## Order of operations

`create_g5g_instance` → `get_install_progress` → `verify_gpu_arch` → `deploy_jax_server`
→ `get_jax_logs` → `verify_model_health`

Cloud-init installs the **runtime only**. The serving payload is this rig's own source
(`jax_openai_server.py`, `jax_engine.py`, `ports/gemma4/*.py`), so `deploy_jax_server` ships
it over SSM as a gzipped tarball — there is no published artifact to pull, and user data
could not carry it at a 16 KB limit.

Python **3.12** is installed from deadsnakes because jax ≥ 0.11 requires it and the Ubuntu
22.04 DLAMI base ships 3.10.

## Instance sizing

`g5g.2xlarge` is the default. `g5g.xlarge` is **supported** — it gets a 16 GiB swapfile,
without which the kernel refuses to mmap the 10.2 GB checkpoint against 7.5 GiB of RAM. That
was measured on the vLLM sibling; the checkpoint and host are the same here.

## AMI resolution — two requirements, not one

`_resolve_ami` prefers the AWS public SSM parameter for the ARM64 **GPU** DLAMI. Both halves
matter and a name filter only pins one:

- **arm64** — the x86_64 DLAMI ids hardcoded in the legacy tips tree cannot boot on Graviton2.
- **NVIDIA driver** — AWS also ships ARM64 DLAMIs built for Graviton CPU inference. Those
  boot perfectly well on a G5g and simply have no GPU, which presents as JAX silently
  falling back to CPU rather than as a wrong AMI.

**Never hardcode an AMI id.** There is no prebuilt AMI for this rig and none is needed.

## Lifecycle guardrails

- Launches default to **spot**. Surface capacity errors; do not silently retry.
- Terminating is permanent but cheap to redo here: there is no built image to lose, only a
  pip install and the model cache. One-time spot instances cannot be stopped, only terminated.
- Provisioning requires explicit **subnet, security-group, and instance-profile ids**. Do not
  create broad network or IAM policy to make a launch succeed.
- The HF token comes from Secrets Manager at boot (`save_hf_token`). **Never** put it in user
  data — instance metadata is readable by anything on the box. The bootstrap also disables
  `xtrace` around the fetch, because `set -x` traces assignments with their values.
- Instance discovery is scoped to `ManagedBy=gpu-jax-g5g-2b`. Never operate outside that tag.

## Diagnostics

- `verify_model_health` uses `/v1/chat/completions`. Raw `/v1/completions` skips the chat
  template and is unreliable on `-it` models — on the vLLM sibling it returned degenerate
  repetition, not an empty body, so **do not health-check by testing for a non-empty
  response**.
- `get_endpoint` resolves the address from the instance. Never hardcode an endpoint.
- `get_jax_logs` reads the **systemd journal**, not docker: nothing here is containerized.
- `get_install_progress` reports INSTALL COMPLETE only once JAX imported *and* saw the GPU.

## Measurement discipline

A config flag being accepted is not evidence it did anything. Cross-check against a physical
bound, and record results in `benchmarks/runs/<date>-<what>-g5g/`.
