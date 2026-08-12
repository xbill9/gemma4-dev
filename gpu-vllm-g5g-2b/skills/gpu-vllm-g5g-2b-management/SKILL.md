---
name: gpu-vllm-g5g-2b-management
description: Manage AWS EC2 G5g capacity (Graviton2 + NVIDIA T4G) and Gemma 4 E2B vLLM serving. Use when the user asks about provisioning, launching, listing, or terminating G5g instances, building or debugging vLLM on T4G / Turing / aarch64, G-family quotas, or the gpu-vllm-g5g-2b devops MCP agent. Triggers include "G5g", "T4G", "Graviton", "Turing", "SM 7.5", "arm64 GPU", "vLLM on T4G".
---

# gpu-vllm-g5g-2b management

Provision and operate **EC2 G5g** (AWS Graviton2 host + NVIDIA **T4G** GPU) serving
`google/gemma-4-E2B-it` under vLLM, through the `gpu-vllm-g5g-2b` MCP server.

## Start here, every time

**Run `verify_gpu_arch` before anything else on a new instance.** This rig exists in a
packaging gap, and that tool is the only cheap way to know which side of it you are on.

## The constraint that shapes everything

G5g needs **aarch64 and SM 7.5 together**. No published CUDA artifact provides both.
`vllm/vllm-openai:v0.27.1`, read from the published image config on 2026-08-12:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? |
| --- | --- | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | **no** |

vLLM's Dockerfile deliberately sets no `+PTX`, so there is no JIT fallback — an unsupported
device fails with `no kernel image is available for execution on the device`.

Two serving modes follow:

- **`serving='build'` (default)** — rebuilds the image on the instance with
  `--build-arg torch_cuda_arch_list=7.5`. Takes **hours** on a Graviton2. Follow it with
  `get_build_progress`; cloud-init detaches the build so the boot does not block.
- **`serving='stock'`** — runs the published arm64 image unchanged. **Expected to fail.**
  It exists to reproduce the gap on real hardware, not as a fallback.

One layer is still unverified: PyTorch's own aarch64 CUDA wheels appear to be built for
9.0/10.0/12.0, also without 7.5. If so, `build` is necessary but not sufficient and a
from-source PyTorch is needed too. `verify_gpu_arch` settles it. See
`docs/turing-aarch64-gap.md`.

## Turing flag rules

T4G is Turing (SM 7.5), **not** the Ada L4 every sibling GPU rig here targets.

- `--dtype float16`. **Never `bfloat16`** — bf16 arrives with Ampere; on T4G it fails.
- `--kv-cache-dtype auto`. **Never `fp8`** — no fp8 datapath.
- `VLLM_ATTENTION_BACKEND=XFORMERS`. FlashAttention needs SM 8.0+.
- No `--quantization`. The reference bf16 E2B checkpoint is 9.5 GiB against 16 GB of GPU
  memory, so it fits; w4a16 would want Marlin kernels that need SM 8.0+ anyway.

Copying a flag set from a `gpu-vllm-l4-*` rig or from `~/gemma4-tips-aws` will produce a
config that is accepted and then fails on the device.

## Instance sizing

`g5g.2xlarge` is the default and the floor. **`g5g.xlarge` is rejected**: 8 GiB of host RAM
cannot stage E2B's 9.5 GiB of weights. `g5g.16xlarge` and `g5g.metal` have two T4Gs and get
`--tensor-parallel-size 2`; everything else gets 1.

## AMI resolution — two requirements, not one

`_resolve_ami` prefers the AWS public SSM parameter for the ARM64 **GPU** DLAMI. Both halves
matter and a name filter only pins one:

- **arm64** — the x86_64 DLAMI ids hardcoded in the legacy tips tree cannot boot on Graviton2.
- **NVIDIA driver** — AWS also ships ARM64 DLAMIs built for Graviton CPU inference. Those
  boot perfectly well on a G5g and simply have no GPU, which presents as a broken container
  rather than a wrong AMI.

**Never hardcode an AMI id.**

## Lifecycle guardrails

- Launches default to **spot**. Surface capacity errors; do not silently retry.
- **Terminating destroys the locally built SM 7.5 image** along with the root volume, and the
  next launch rebuilds from source — hours. Prefer `stop_g5g_instance` when the instance will
  be needed again. One-time spot instances cannot be stopped, only terminated.
- Provisioning requires explicit **subnet, security-group, and instance-profile ids**. Do not
  create broad network or IAM policy to make a launch succeed.
- The HF token comes from Secrets Manager at boot (`save_hf_token`). Never put it in user
  data — instance metadata is readable by anything on the box.
- Instance discovery is scoped to `ManagedBy=gpu-vllm-g5g-2b`. Never operate on instances
  outside that tag.

## Diagnostics

- `verify_model_health` uses `/v1/chat/completions`. Raw `/v1/completions` returns an empty
  completion on `-it` models, so an empty body there is not a broken deploy.
- `get_endpoint` resolves the address from the instance. Never hardcode an endpoint.
- `get_vllm_logs` and `get_build_progress` both run over SSM — no SSH.

## Measurement discipline

A config flag being accepted is not evidence it did anything. Cross-check against a physical
bound, and record `verify_gpu_arch` output in `benchmarks/runs/<date>-<what>-g5g/` — it is
the finding this rig was built to establish. This rig currently has **no measurements of its
own**; do not attribute any sibling's numbers to T4G hardware.
