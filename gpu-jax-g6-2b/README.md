# gpu-jax-g6-2b

Serve **`google/gemma-4-E2B-it`** with **pure JAX** on **AWS EC2 G6** — an x86_64 host paired
with an **NVIDIA L4** (Ada Lovelace, SM 8.9).

> **Status: this rig has served nothing.** Forked from
> [`gpu-jax-g5g-2b`](../gpu-jax-g5g-2b/) on 2026-08-28. 137 tests pass offline. No instance has
> been launched, no weights loaded, no token generated, and `benchmarks/` is deliberately empty.

## Why this exists — it is the dtype-tax control

The parent rig's profile is the reason for this one:

| share of decode on T4G | |
| ---: | --- |
| 54.0% | **dtype conversion** |
| 32.9% | fp32 `gemv` |
| 12.2% | fusion |
| **0.0%** | **TensorCore** |

The obvious explanation was bf16 weights being converted on a chip with no bf16 datapath. On
2026-08-28 the parent's checkpoint was converted to float16 host-side — parameter dtypes read
`{'float16': 541, ...}` — and **conversion stayed at 54.0%**. The obvious explanation is wrong
and the real one is unknown.

Ada removes the pressure at the hardware level: **a native bfloat16 datapath and fp8**, neither
of which Turing has. And it needs no code change to exercise — `jax_e_model._compute_dtype()`
reads the live compute capability and returns `bfloat16` at SM ≥ 8.0, where it returned
`float16` below it. Same engine, same weights, conversion pressure removed.

**If conversion collapses and TensorCore utilisation rises above zero, that closes the open
question. If 54% survives onto a bf16-native chip, the cause was never dtype at all** — and that
is the more interesting result, because it would mean the parent's leading hypothesis has been
wrong twice.

| | `gpu-jax-g5g-2b` | **this rig** |
| --- | --- | --- |
| GPU | NVIDIA T4G, Turing | **NVIDIA L4, Ada** |
| compute capability | SM 7.5 | **SM 8.9** |
| bf16 datapath | **none** — emulates via fp32 | **native** |
| fp8 | none — refused | present (unused so far) |
| compute dtype | `float16` | **`bfloat16`** (device-selected) |
| VRAM reported | 15,360 MiB | **22,888 MiB** (not the nominal 24 GB) |
| default size | `g5g.2xlarge` | **`g6.xlarge`** |

## What Ada does *not* fix

**The fused W4A16 Pallas kernel is still refused at startup.** It is tiled for TPU VMEM and
needs **550 KiB – 1.1 MiB per block**; Ada raises the per-block shared-memory ceiling above
Turing's 64 KiB but nowhere near that. So this rig also serves the **dense reference
checkpoint**, and the rig name carries no encoding slot. `check_w4a16_fits_scoped_memory()`
still computes the requirement and raises with the arithmetic attached.

## Cost — this is the expensive sibling

`g6.xlarge` is **$0.8048/hr on-demand and $0.7854 spot average** (us-east-1, 7-day, 2026-08-28).
Note the ~2% spot discount: full L4s barely move, unlike `g5g` (28%) or `g6f` (50%). Budget
close to on-demand and do not plan around spot savings that are not there.

`g6f.4xlarge` — **half** an L4 at 12 GB — is $0.9500 on-demand but $0.4723 spot average, so it is
the cheaper way onto Ada *if and only if* you are on spot and 12 GB is enough. It is not this
rig; it would be a separate one.

## The 22,888 MiB matters

24 GB is the nominal number and **22,888 MiB is what `describe_instance_types` reports** — the
same measured-vs-nominal trap as the T4G's 15,360. At 6.155 GB of weights and ~1.6 GB of decode
transients there is room to spare here, which is new: this is the first rig in the `gpu-jax`
line where memory is not the binding constraint. That headroom is what would make a 12B
checkpoint reachable later.

## Quickstart

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v   # 137 tests, fully offline
```

Then through the MCP tools (`mcp__gpu-jax-g6-2b__…`):

```
create_g6_instance → get_install_progress → verify_gpu_arch → deploy_jax_server
                   → get_jax_logs → verify_model_health → query_model → get_metrics
```

**Always `make skill` before `deploy_jax_server`.**

## What to watch on the first run

The device-policy banner is the first line the process emits, and on this rig it should read
`compute_capability=8.9 compute_dtype=bfloat16 pre_ampere=False`. **If it says `float16`, the
detection failed and the whole point of the rig is void** — check it before believing any
number that follows.

## License

Apache-2.0 — see [`../LICENSE`](../LICENSE).
