---
name: gpu-vllm-g4dn-2b-management
description: Manage AWS EC2 G4dn capacity (x86_64 + NVIDIA T4, Turing SM 7.5) and Gemma 4 E2B vLLM serving. Use when the user asks about provisioning, launching, listing, or terminating G4dn instances, running or debugging vLLM on T4 / Turing / SM 7.5, the Triton shared-memory ceiling, G-family quotas, or the gpu-vllm-g4dn-2b devops MCP agent. Triggers include "G4dn", "T4", "Turing", "SM 7.5", "vLLM on T4", "g4dn.xlarge", "OutOfResources shared memory".
---

# gpu-vllm-g4dn-2b management

Provision and operate **EC2 G4dn** (x86_64 host + NVIDIA **T4** GPU, Turing SM 7.5) serving
`google/gemma-4-E2B-it` under vLLM, through the `gpu-vllm-g4dn-2b` MCP server.

> **THIS RIG HAS SERVED NOTHING.** Forked from `gpu-vllm-g6-2b` on 2026-08-29 into a directory
> that was a stale copy of `gpu-jax-g4dn-2b`. Everything below is arithmetic or inherited.
> `benchmarks/` is deliberately empty.

## The one thing to understand first

This rig **isolates one of the two problems** that make `gpu-vllm-g5g-2b` hard:

| | SM 7.5 in the published image? | Triton tile vs shared memory |
| --- | --- | --- |
| `gpu-vllm-g5g-2b` — aarch64, SM 7.5 | **NO** → ~67-min from-source build | **NO** → tile clamp |
| `gpu-vllm-g6-2b` — x86_64, SM 8.9 | yes | yes (~99 KiB), unverified |
| **this rig** — x86_64, SM 7.5 | **YES** | **NO** → tile clamp, mandatory |

`vllm/vllm-openai`'s `linux/amd64` manifest is compiled for `7.5 8.0 8.6 8.9 9.0 10.0 12.0`;
its `linux/arm64` manifest for `8.0 8.7 8.9 9.0 10.0 11.0 12.0`. **G4dn is Intel, so it pulls
the one that carries 7.5.** No build, no CUDA toolkit, no Rust, no AMI to bake, no `serving=`
mode.

**The Turing half is untouched, and it is arithmetic rather than a margin:** Gemma 4's
512-wide global heads force a Triton tile wanting **98,304 B** per block against Turing's
**65,536**.

## Start here, every time

```
check_g4dn_quotas → create_g4dn_instance → get_install_progress
                  → verify_gpu_arch → verify_triton_patch → verify_model_health → query_model
```

- **`verify_gpu_arch` is expected to PASS**, the opposite of the G5g rig where the same tool
  confirms an absence. It probes with **float16**, not bfloat16 — a bf16 probe would pass by
  upconversion and tell you nothing about what executes.
- **Passing it says NOTHING about the Triton ceiling.** The two problems are independent.
  **`verify_triton_patch` is the check that matters here.**
- **`get_install_progress` separates a dead bootstrap from a slow one**, and calls out a
  refused patch by name. Cloud-init can die before it writes anything, and the naive rendering
  of that is `IN PROGRESS` forever — which is also what a healthy slow launch looks like.
- **The container starting is not the model being ready.** vLLM still has to download and load
  the checkpoint.
- **Termination is cheap.** Nothing is compiled — a relaunch costs an image pull, a
  seconds-long derived build and the model download. Do not import the G5g rig's "weigh stop
  against terminate" reasoning.

## The patch, and how it fails

The bootstrap pulls the published image, resolves the module path **inside** it, patches
`triton_unified_attention.py`, builds a derived tag with a single `COPY`, verifies the clamp
is present **in the built image**, and serves that tag.

`patch_triton_turing.py` **refuses rather than no-ops** — exit 2 with the surrounding source —
when an identifier is missing, the launch-site anchor is gone, or the pipeline-stage variable
is ambiguous. A refusal kills cloud-init on purpose: serving unpatched behind a patched tag
reports success for ten minutes and then dies with

```
triton.runtime.errors.OutOfResources: shared memory, Required: 98304, Hardware limit: 65536
```

**VERIFIED against real upstream source 2026-08-29** (`v0.28.0`, byte-identical to `main`).
The patch applies cleanly and lands at line 997 of `unified_attention`. Two findings worth
carrying:

- **`_get_tile_size` has no shared-memory awareness** — upstream has not fixed this, so the
  clamp is still required.
- **The launch-site anchor was wrong and would have failed silently.** Upstream copies the
  tile constants into `tile_size` and `launch_num_stages` into `launch_kwargs` *before* the
  launch, so a clamp there rewrites variables nothing reads — marker present, verification
  green, kernel still asking for 98,304 bytes. The insertion point is now derived from the
  code, after the last tile assignment and before the first read.

**Source-level verification is not a served token.** No instance has been launched.

`verify_triton_patch` distinguishes the four ways this ends up wrong: the image was never
built, the `COPY` landed somewhere the module is not imported from, the container is running
the **stock** tag, or the box carries a **stale** patch script.

## Dtype: Turing has neither bf16 nor fp8

`DTYPE=float16`. The fork parent `gpu-vllm-g6-2b` runs Ada and defaults to **bfloat16**, which
is the single most likely thing to inherit by mistake.

**bfloat16 does not fail here — it upconverts.** MEASURED on the G5g rig: PyTorch runs bf16 on
Turing by upconverting and vLLM logs `Casting torch.bfloat16 to torch.float16`. float16 is
right because it is what **executes**, not because bf16 errors. A silent cost, not an error.

**fp8 has no datapath at all**, unlike on the G6 rig where it exists and is merely unused.

**Do not quote the JAX rigs' "54% of decode" dtype-tax figure here.** That was a JAX loader
converting at every *use*, per step. vLLM converts once at load.

## Attention backend: not a knob

MEASURED on the G5g rig 2026-08-12: vLLM v0.27 does not recognize `VLLM_ATTENTION_BACKEND` at
all (`Unknown vLLM environment variable detected`), and forces `TRITON_ATTN` for Gemma 4
regardless, because the head dims are heterogeneous (sliding 256 / global 512) and only FA4 or
Triton handle that. **The tile size inside the forced kernel is the knob**, and that is what
the patch changes. `HARDWARE.md` still says the Turing backend is `XFORMERS`; that line
predates this measurement.

## The image tag floor

**`VLLM_IMAGE=vllm/vllm-openai:v0.28.0`.** The tag this rig inherited from `gpu-vllm-g6-2b`,
`v0.27.2rc0`, **does not exist** — 404 on Docker Hub and no such git tag. CHECKED 2026-08-29.
Cloud-init would have died at `docker pull`. It survived because the floor test asserted "not
v0.27.1, not v0.26", which a nonexistent tag passes trivially.

The floor itself still holds: **v0.26.0 dies** with `AmbiguousGlobalPerLayerAttributeError`
because Gemma 4's `head_dim` is per-layer. A constraint of the **model**, not the chip.
`v0.28.0` clears it outright.

**Not `nightly`** — it is a moving tag and this rig renders deterministic user data. Checked:
`main` is byte-identical to `v0.28.0` for the file being patched, so nightly buys nothing.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group and instance-profile ids. Do not create broad
  network or IAM policy.
- Scope instance discovery to `ManagedBy=gpu-vllm-g4dn-2b`.
- Hugging Face tokens live in Secrets Manager and are fetched at boot. **Never** in user data —
  instance metadata is readable by anything on the box, and `set +x` wraps the fetch because
  bash traces assignments with their values.
- Never hardcode an AMI id, and never hardcode an endpoint. Note the legacy tips-tree
  `ami-012ba162b9cd2729c` **is** x86_64 and so would boot here — a worse trap than on the G5g
  rig, where it fails immediately.
- **Do not health-check by testing for a non-empty response.** On this lineage a broken deploy
  answered `': ok: ok: ok…'` — degenerate repetition that passes an emptiness check.
