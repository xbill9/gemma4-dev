# gpu-jax-g4dn-2b

Serve **`google/gemma-4-E2B-it`** with **pure JAX** on **AWS EC2 G4dn** — an x86_64 Intel host
paired with an **NVIDIA T4** (Turing, SM 7.5).

> **Status: this rig has served nothing.** Forked from
> [`gpu-jax-g5g-2b`](../gpu-jax-g5g-2b/) on 2026-08-28. The engine, the server and the MCP
> devops agent are inherited unchanged; what moved is the **host architecture**, from Graviton2
> `aarch64` to `x86_64`. 137 tests pass offline. No instance has been launched, no weights
> loaded, no token generated, and `benchmarks/` is deliberately empty.

## Why this exists

**It is the A/B control that separates Turing from Graviton2.**

`gpu-jax-g5g-2b` measured **13.10 tok/s**, with **54.0% of decode in dtype conversion** and
**0.0% TensorCore utilisation**. Nothing in that run can tell you how much of it is the T4G and
how much is the Graviton2 host, because both changed at once relative to every other rig.

This rig changes exactly one variable. Same engine, same model, same dtype policy, same Turing
SM 7.5 — different host CPU and different instruction set. If the numbers land on top of the
G5g's, the host is irrelevant and the finding is purely Turing. If they diverge, the divergence
is the Graviton2 contribution, isolated.

| | `gpu-jax-g5g-2b` | **this rig** |
| --- | --- | --- |
| host | AWS Graviton2, `aarch64` | **Intel/AMD, `x86_64`** |
| GPU | NVIDIA T4G | **NVIDIA T4** |
| architecture | Turing, SM 7.5 | Turing, SM 7.5 — *same* |
| VRAM reported | 15,360 MiB | **16,384 MiB** (`describe_instance_types`) |
| default size | `g5g.2xlarge` | **`g4dn.xlarge`** |
| compute dtype | `float16` (device-selected) | `float16` (device-selected) |

**The T4 reports a full 16,384 MiB where the T4G reports 15,360.** That is a real 1 GiB
difference on the same nominal 16 GB, and it is the kind of thing that decides whether a
configuration fits.

## What is cheaper here, and what is not

`g4dn.xlarge` is **$0.5260/hr on-demand, $0.3678 spot average** (us-east-1, 7-day history,
2026-08-28) against `g5g.2xlarge`'s $0.5560 / $0.3996. So the control is also the cheaper box.

What it does **not** buy is any escape from Turing. No bf16, no fp8, the same 64 KiB
shared-memory ceiling, and the fused W4A16 Pallas kernel is refused at startup for the same
arithmetic. If you want out of those constraints, that is [`gpu-jax-g6-2b`](../gpu-jax-g6-2b/),
not this rig.

## What changed in the fork

- **AMI resolution** — the SSM parameter moved from `/arm64/` to `/x86_64/`, and the name
  fallback is narrower: `Deep Learning Base OSS Nvidia Driver GPU*Ubuntu*`. The parent's pattern
  also accepted the frozen PyTorch line; there is no reason to inherit that.
- **Instance shapes** — read from `describe_instance_types` on 2026-08-28, not a product page.
  Note the ladder is not monotonic: `g4dn.12xlarge` has **four** T4s and `g4dn.16xlarge` has
  **one**.
- **Tool names** follow the rig's hardware slot: `create_g4dn_instance`, `check_g4dn_quotas`, …
- **The swap threshold is inherited, not re-measured.** 16 GiB inclusive, from the parent's
  2026-08-26 OOM-kill run. `g4dn.xlarge` has exactly 16 GiB, so it is the boundary case.

## Quickstart

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v   # 137 tests, fully offline
```

Then through the MCP tools (`mcp__gpu-jax-g4dn-2b__…`):

```
create_g4dn_instance → get_install_progress → verify_gpu_arch → deploy_jax_server
                   → get_jax_logs → verify_model_health → query_model → get_metrics
```

**Always `make skill` before `deploy_jax_server`** — the deploy ships the skill snapshot, not
the working tree.

## Inherited findings — read them against the parent, not this rig

The engine-level documentation lives in [`gpu-jax-g5g-2b/docs/`](../gpu-jax-g5g-2b/docs/) and
every number in it was measured on G5g. `docs/turing-aarch64-gap.md` is **half** applicable here:
the Turing shared-memory analysis carries, the aarch64 packaging half does not — x86_64 + CUDA is
the well-trodden axis, which is precisely why this rig should be cheaper to stand up.

## License

Apache-2.0 — see [`../LICENSE`](../LICENSE).
