---
name: gpu-vllm-g6-2b-management
description: Manage AWS EC2 G6 capacity (x86_64 + NVIDIA L4, Ada SM 8.9) and Gemma 4 E2B vLLM serving. Use when the user asks about provisioning, launching, listing, or terminating G6 instances, running or debugging vLLM on L4 / Ada / SM 8.9, G-family quotas, or the gpu-vllm-g6-2b devops MCP agent. Triggers include "G6", "L4", "Ada", "SM 8.9", "vLLM on L4", "g6.xlarge".
---

# gpu-vllm-g6-2b management

Provision and operate **EC2 G6** (x86_64 host + NVIDIA **L4** GPU, Ada SM 8.9) serving
`google/gemma-4-E2B-it` under vLLM, through the `gpu-vllm-g6-2b` MCP server.

> **THIS RIG HAS SERVED NOTHING.** Forked from `gpu-vllm-g5g-2b` on 2026-08-28. Everything
> below is arithmetic or inherited from a sibling. `benchmarks/` is deliberately empty.

## Start here, every time

**Run `verify_gpu_arch` before anything else on a new instance.** On this rig it is expected
to **pass**, which is the opposite of the sibling, where the same tool exists to confirm an
absence. That inversion is the fork's entire premise and it has not been checked once.

## The constraint that shaped the sibling — and does not exist here

G5g needs **aarch64 and SM 7.5 together**, and no published CUDA artifact provides both.
Read from the published image config on 2026-08-12:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? | SM 8.9? |
| --- | --- | :---: | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** | **yes** |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | no | yes |

**G6 is x86_64 and SM 8.9, so it wants the amd64 manifest and that manifest carries 8.9.**
Both axes are covered by the published image. There is therefore:

- **no from-source build** (the sibling spends ~67 minutes),
- **no CUDA toolkit** to install (the DLAMI does not ship one),
- **no Rust toolchain**,
- **no Triton attention patch** to reapply after every upgrade, and
- **no `serving=` mode to choose** — the sibling's `build` / `stock` split is gone, because
  `build` would have nothing to do and `stock` nothing to fail at.

## What is still open, and it is the interesting part

Gemma 4's head dims are heterogeneous — sliding **256**, global **512** — and vLLM forces
`TRITON_ATTN` for the model regardless of what you ask for. That tile wants **~96 KiB of
shared memory per block**. Turing caps a block at **64 KiB**, which is why the sibling carries
an unlanded patch shrinking the tile. **Ada allows ~99 KiB, so it should fit unpatched.**

**Unverified.** `VLLM_ATTENTION_BACKEND` is deliberately left unpinned so vLLM dispatches for
the real part — pinning a backend is how the sibling ended up carrying a patch.

Note also, MEASURED on the sibling: vLLM v0.27 does not recognize `VLLM_ATTENTION_BACKEND`
at all and logs `Unknown vLLM environment variable detected`. Setting it did nothing there.

## Dtype: Ada has bf16 and fp8, Turing had neither

`DTYPE=bfloat16`, and not merely because the part supports it — **the checkpoint is bf16**, so
float16 would make vLLM convert every weight on load. `gpu-jax-g6-2b` MEASURED that exact
mismatch costing **54% of decode** on Turing, and **0.0%** once the dtypes matched.

fp8 KV is newly reachable and **is not enabled**: KV is ~18 KiB/token, so the whole cache at
16K context is ~288 MiB against 23034 MiB. It is not the binding constraint, and nothing here
has measured its accuracy cost.

## The image tag is not the sibling's

The sibling's `VLLM_STOCK_IMAGE` was `v0.27.1`, but it used that tag **only to reproduce the
SM 7.5 failure** — it never served from it, and built its real image from `v0.27.2rc0`.

**MEASURED: v0.26.0 dies** with `AmbiguousGlobalPerLayerAttributeError` against current
transformers, because Gemma 4's `head_dim` is per-layer; the fix landed in **v0.27.2rc0**.
That is a constraint of the **model**, not the chip, so it carries across the fork unchanged.
**Do not pin below it** — and note `v0.27.1` is exactly the wrong tag to inherit by mistake.

## Operating order

```
check_g6_quotas → create_g6_instance → get_install_progress
                → verify_gpu_arch → verify_model_health → query_model
```

- **Quota is not capacity.** MEASURED 2026-08-28 on the JAX sibling: `g6.xlarge` spot was
  exhausted in **all five** us-east-1 AZs with quota to spare. Use
  `aws ec2 get-spot-placement-scores` to pick a size and AZ instead of launching in a loop.
- **`get_install_progress` separates a dead bootstrap from a slow one.** Cloud-init can die
  before it writes anything, and the naive rendering of that is `IN PROGRESS` forever — which
  is also what a healthy slow launch looks like.
- **The container starting is not the model being ready.** vLLM still has to download and load
  the checkpoint.
- **Termination is cheap here.** There is no built image to lose with the root volume, only an
  image pull and the model cache — unlike the sibling, whose reasoning about weighing stop
  against terminate does not carry.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group and instance-profile ids. Do not create broad
  network or IAM policy.
- Scope instance discovery to `ManagedBy=gpu-vllm-g6-2b`.
- Hugging Face tokens live in Secrets Manager and are fetched at boot. **Never** in user data —
  instance metadata is readable by anything on the box, and `set +x` wraps the fetch because
  bash traces assignments with their values.
- Never hardcode an AMI id, and never hardcode an endpoint.
- **Do not health-check by testing for a non-empty response.** On this rig's lineage a broken
  deploy answered `': ok: ok: ok…'` — degenerate repetition that passes an emptiness check.
