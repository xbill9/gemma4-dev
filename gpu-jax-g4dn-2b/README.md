# gpu-jax-g4dn-2b

Serve **`google/gemma-4-E2B-it`** with **pure JAX** on **AWS EC2 G4dn** — an x86_64 Intel host
paired with an **NVIDIA T4** (Turing, SM 7.5).

> **Status: this rig has served.** Forked from
> [`gpu-jax-g5g-2b`](../gpu-jax-g5g-2b/) on 2026-08-28; **first serve 2026-08-29**, on a spot
> `g4dn.xlarge` launched, deployed, measured and terminated in one ~25-minute cycle
> (`benchmarks/runs/2026-08-29-first-serve-g4dn/`). The engine, the server and the MCP devops
> agent are inherited unchanged; what moved is the **host architecture**, from Graviton2
> `aarch64` to `x86_64`. 140 tests pass offline.
>
> Until that run, `server.py` and this file both carried the **sibling's numbers relabelled as
> this rig's own** — the exact fork hazard the monorepo `CLAUDE.md` warns about. Every figure
> below is now either measured here or explicitly attributed.

## Why this exists

**It is the A/B control that separates Turing from Graviton2 — and the control came back
clean.**

`gpu-jax-g5g-2b` measured **13.10 tok/s**, with **54.0% of decode in dtype conversion** and
**0.0% TensorCore utilisation**. Nothing in that run could tell you how much of it was the T4G
and how much the Graviton2 host, because both changed at once relative to every other rig.

This rig changes exactly one variable. Same engine, same model, same dtype policy, same Turing
SM 7.5 — different host CPU and different instruction set.

**The numbers landed on top of the G5g's, so the host is irrelevant and the finding is purely
Turing.**

| | `gpu-jax-g5g-2b` (T4G) | **this rig (T4)** |
| --- | ---: | ---: |
| decode, gauge | 13.10 tok/s | **13.1 / 13.2 / 13.1** |
| `tpu_jax_weight_bytes` | 6.155 GB | **6.155 GB** — exact |
| dtype conversion | 54.4% / 54.0% | **54.4%** |
| fp32 GEMV | 32.6% / 32.8% | **32.8%** |
| TensorCore utilisation | 0.0% | **0.0%** |
| peak FLOP / HBM / ridge | 65126.4 / 298.083 / 203.479 | **identical to 3 d.p.** |

That settles the sibling's central result as a **chip** finding rather than a Graviton2 one:
the 86.9% spent in dtype conversion plus fp32 GEMV is a Turing property. Its 2026-08-28
falsification carries here unchanged too — at `B=1` decode is a matrix-*vector* product,
cuBLAS dispatches an all-fp32 `gemvx` kernel that has **no half-precision path**, and no
storage dtype changes that. The route to the ceiling is a GEMM, i.e. batching, which
`MAX_NUM_SEQS=1` closes.

| | `gpu-jax-g5g-2b` | **this rig** |
| --- | --- | --- |
| host | AWS Graviton2, `aarch64` | **Intel/AMD, `x86_64`** |
| GPU | NVIDIA T4G | **NVIDIA T4** |
| architecture | Turing, SM 7.5 | Turing, SM 7.5 — *same* |
| VRAM reported | 15,360 MiB | **15,360 MiB** (`nvidia-smi`, measured) |
| default size | `g5g.2xlarge` | **`g4dn.xlarge`** |
| compute dtype | `float16` (device-selected) | `float16` (device-selected, confirmed) |

**Correction, measured 2026-08-29: the T4 reports 15,360 MiB, exactly as the T4G does.** This
file previously claimed a full 16,384 MiB here, and called it "a real 1 GiB difference … the
kind of thing that decides whether a configuration fits". That number came from
`describe_instance_types`, which reports the nominal 16 GB; `nvidia-smi` on the actual device
reports `Tesla T4, 7.5, 15360 MiB`. **There is no capacity advantage.** The device is what
allocates, so the device is what counts — the same rule as "a config flag being accepted is
not evidence it did anything".

## What is cheaper here, and what is not

`g4dn.xlarge` is **$0.5260/hr on-demand, $0.3724 spot** (us-east-1b, paid 2026-08-29) against
`g5g.2xlarge`'s $0.5560 / $0.3996. So the control is also the cheaper box: the whole
first-serve cycle cost roughly **$0.16**.

**Spot capacity is the real constraint, and price does not predict it.**
`us-east-1a` refused with `InsufficientInstanceCapacity` while being one of the *cheapest*
AZs; `us-east-1b` took the request immediately. Retry across AZs rather than reading
cheapness as availability.

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
  2026-08-26 OOM-kill run. `g4dn.xlarge` has exactly 16 GiB, so it is the boundary case — and
  it did load cleanly with the swapfile in place on 2026-08-29.
- **xprof and tensorboard install at boot**, as their own non-fatal stage costing **5 seconds
  of a 76-second install**. On the parent they were on-demand via a documented path inside
  `APP_DIR` that the deploy payload excludes, so they were in practice never installed at all.
  `INSTALL_PROFILING=0` restores a serving-only image.

## Quickstart

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v   # 140 tests, fully offline
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
every number in it was measured on G5g. `turing-aarch64-gap.md` there is **half** applicable here:
the Turing shared-memory analysis carries, the aarch64 packaging half does not — x86_64 + CUDA is
the well-trodden axis.

The 2026-08-29 run is what licenses reading the rest of that documentation as applying here:
decode, weight residency and the kernel breakdown reproduced within noise, so the parent's
**chip-level** conclusions transfer. Its **host-level** ones do not automatically — install
time, load staging and host RSS all depend on a CPU this rig does not share. Note that the
packaging argument is close to vacuous on x86_64, and should not be repeated here as though it
were evidence; the load-bearing claim was always `sm_75`.

## License

Apache-2.0 — see [`../LICENSE`](../LICENSE).
