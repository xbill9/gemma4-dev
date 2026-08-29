# gpu-vllm-g6-2b

Serve **`google/gemma-4-E2B-it`** with **vLLM** on **AWS EC2 G6** — an x86_64 host paired
with an **NVIDIA L4 Tensor Core** GPU (Ada, SM 8.9).

The rig ships a single-file FastMCP server exposing a devops agent that provisions G6
capacity with boto3, brings up the model server, and does SRE diagnostics against the
endpoint.

> **This rig has served nothing.** Forked from [`gpu-vllm-g5g-2b`](../gpu-vllm-g5g-2b) on
> 2026-08-28. Everything below is arithmetic or inherited from a sibling.

| | |
| --- | --- |
| Model | `google/gemma-4-E2B-it` (reference bf16 release — no encoding slot in the name) |
| Runtime | vLLM, OpenAI-compatible API on `:8000` |
| Host | x86_64 |
| GPU | NVIDIA L4 — Ada, **SM 8.9**, 24 GB nominal / 23034 MiB measured |
| Default size | `g6.xlarge` (1 GPU, 4 vCPU, 16 GiB RAM) |
| Region | `us-east-1` |

Authoritative values live in [`tpu.env`](tpu.env). The directory name describes; the env
file decides.

## Why this rig exists

**It is the runtime control for [`gpu-jax-g6-2b`](../gpu-jax-g6-2b)**, which measured
**48.3–48.5 tok/s** on this exact silicon under pure JAX on 2026-08-28. Same chip, same
checkpoint, different runtime — the only clean runtime comparison in this tree.

The T4G pair was never clean: the vLLM side there ran with hand-reduced Triton tiles, so its
number and the JAX rig's were not measuring the same thing.

## What the fork removed

`gpu-vllm-g5g-2b` is hard because **G5g needs aarch64 and SM 7.5 together, and no published
CUDA artifact provides both**:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? | SM 8.9? |
| --- | --- | :---: | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** | **yes** |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | no | yes |

vLLM's Dockerfile sets no `+PTX`, so nothing JITs.

**G6 is x86_64 and SM 8.9, so it wants the amd64 manifest and that manifest carries 8.9.**
No build, no CUDA toolkit, no Rust, no prebuilt AMI, and no `serving=` mode — this rig runs
`vllm/vllm-openai` as published.

## What might still bite

Gemma 4's head dims are heterogeneous (sliding 256, global **512**), which forces
`TRITON_ATTN`. That tile wants **~96 KiB of shared memory per block**. Turing caps a block at
64 KiB — the sibling's actual blocker, worked around with an unlanded patch. **Ada allows
~99 KiB**, so it should fit unpatched.

**That margin is narrow and unverified.** It is the first thing to check.

## Quick start

```bash
./project-setup.sh                      # install the skill, register the MCP server
python3 -m unittest discover -s tests   # 38 offline tests: no AWS, no network, no GPU
```

Then, through the MCP server:

```
check_g6_quotas → create_g6_instance → get_install_progress
                → verify_gpu_arch → verify_model_health → query_model
```

**Quota is not capacity.** Measured 2026-08-28: `g6.xlarge` spot was exhausted in all five
`us-east-1` AZs with quota to spare. Use `aws ec2 get-spot-placement-scores` to pick a size
and AZ rather than launching in a loop.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — authoritative. The fork's deltas, the dtype policy, the Triton
  margin, and the numbers you must not reuse.
- [`tpu.env`](tpu.env) — every setting, with the measurement or the arithmetic behind it.
- [`docs/INHERITED.md`](docs/INHERITED.md) — what was deliberately *not* copied at the fork.
