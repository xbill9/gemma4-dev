# gpu-vllm-g4dn-2b

Serve **`google/gemma-4-E2B-it`** with **vLLM** on **AWS EC2 G4dn** — an x86_64 (Intel) host
paired with an **NVIDIA T4 Tensor Core** GPU (Turing, SM 7.5).

The rig ships a single-file FastMCP server exposing a devops agent that provisions G4dn
capacity with boto3, brings up the model server, and does SRE diagnostics against the
endpoint.

> **This rig has served nothing.** The directory was a stale copy of
> [`gpu-jax-g4dn-2b`](../gpu-jax-g4dn-2b); the vLLM side was forked from
> [`gpu-vllm-g6-2b`](../gpu-vllm-g6-2b) on 2026-08-29. Everything below is arithmetic or
> inherited from a sibling.

| | |
| --- | --- |
| Model | `google/gemma-4-E2B-it` (reference bf16 release — no encoding slot in the name) |
| Runtime | vLLM, OpenAI-compatible API on `:8000` |
| Host | x86_64 (Intel Cascade Lake) |
| GPU | NVIDIA T4 — Turing, **SM 7.5**, 16 GB nominal / 15360 MiB measured on the T4G sibling |
| Default size | `g4dn.xlarge` (1 GPU, 4 vCPU, 16 GiB RAM) |
| Region | `us-east-1` |

Authoritative values live in [`tpu.env`](tpu.env). The directory name describes; the env file
decides.

## Why this rig exists

**It isolates one of the two problems that make [`gpu-vllm-g5g-2b`](../gpu-vllm-g5g-2b) the
hardest rig in this tree.** That rig hits both at once and has to solve both to serve a single
token:

| | SM 7.5 in the published image? | Triton tile vs Turing's 64 KiB |
| --- | --- | --- |
| `gpu-vllm-g5g-2b` — aarch64, SM 7.5 | **no** → ~67-minute from-source build | **no** → tile clamp |
| `gpu-vllm-g6-2b` — x86_64, SM 8.9 | yes | yes (~99 KiB), unverified |
| **this rig** — x86_64, SM 7.5 | **yes** | **no** → tile clamp |

`vllm/vllm-openai` publishes one manifest list with two platforms:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? |
| --- | --- | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | no |

G4dn is Intel, so it pulls the amd64 manifest — the one carrying 7.5. **Same image, same tag,
different answer purely because of the host architecture.** So there is no build, no CUDA
toolkit, no Rust, and no AMI to bake.

**The Turing half survives untouched**, and unlike on Ada it is not a margin to check:
Gemma 4's 512-wide global attention heads force a Triton tile wanting **98,304 B** of shared
memory per block, against Turing's **65,536**. That is arithmetic, and the same silicon has
already produced the failure on the G5g sibling.

**It is also the runtime control for [`gpu-jax-g4dn-2b`](../gpu-jax-g4dn-2b)** — same chip,
same host, same checkpoint, different runtime.

## What is new here: the patch is delivered, not compiled

The G5g rig can only get its Triton clamp in by compiling vLLM from source, because its image
has no SM 7.5 kernels at all. Here the kernels are already present and exactly one pure-Python
file is wrong, so the bootstrap patches the published image and builds a derived tag:

```
docker pull → resolve the module path INSIDE the image → patch → FROM <stock>; COPY
            → verify the clamp is in the BUILT image → serve the derived tag
```

Seconds, not an hour. [`patch_triton_turing.py`](patch_triton_turing.py) **refuses loudly**
rather than no-op'ing, because an unpatched file behind a patched tag reports success for ten
minutes and then dies at engine start.

## Quick start

```bash
./project-setup.sh                      # install the skill, register the MCP server
python3 -m unittest discover -s tests   # offline: no AWS, no network, no GPU, no docker
```

Then, through the MCP server:

```
check_g4dn_quotas → create_g4dn_instance → get_install_progress
                  → verify_gpu_arch → verify_triton_patch → verify_model_health
```

**`verify_gpu_arch` passing says nothing about `verify_triton_patch`.** The two problems are
independent, and only the first one is gone on this hardware. Run both.

**Quota is not capacity.** G-family spot in `us-east-1` has been exhausted in every AZ but one
with quota to spare, and the one AZ with capacity was the most expensive. Use
`aws ec2 get-spot-placement-scores` rather than launching in a loop.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — authoritative. The fork's deltas, the dtype policy, the patch
  mechanism, and the numbers you must not reuse.
- [`docs/turing-shared-memory.md`](docs/turing-shared-memory.md) — why the clamp is needed,
  why x86_64 does not help, and why the patch script refuses.
- [`tpu.env`](tpu.env) — every setting, with the measurement or the arithmetic behind it.
- [`docs/INHERITED.md`](docs/INHERITED.md) — what was deliberately *not* copied.
