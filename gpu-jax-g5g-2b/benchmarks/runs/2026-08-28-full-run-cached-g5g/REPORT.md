# 2026-08-28 — full loop on a fresh instance, with the cache restored from S3

**`g5g.2xlarge` spot (`i-0e629977053de0e57`), `us-east-1a`, AMI `ami-025a6e5b3b786cf61`
(Deep Learning ARM64 Base OSS Nvidia Driver GPU, **Ubuntu 26.04.1 LTS**), driver 595.91.07,
jax 0.11.1 / Python 3.14, build id `51bc52c9e2e9`.**

First end-to-end run of `tune_loop.py --xprof` against an instance that restored its
compilation cache at boot. It closes the last open item from 2026-08-27 and reproduces the
prior profile almost exactly on different hardware.

## Cross-instance cache restore — CONFIRMED

The item `2026-08-27-ubuntu2604-base-g5g/REPORT.md` listed as not established.

`install.sh` restored **805 files / 12 MB** into `/opt/jax-cache` in **6 seconds**
(`[stage] cache-restore +6s`), on a **fresh instance**, from what the **now-terminated**
2026-08-27 box had compiled. `jax-g5g-cache.timer` came up `enabled` without intervention.

That is the full loop the feature exists for: compile → timer uploads → instance reclaimed →
new instance restores. Every leg is now measured.

## Install: 117 s, with the restore included

| Stage | Delta | Total |
| --- | ---: | ---: |
| `jax-wheels` | 43s | 84s |
| `serving-deps` | 14s | 98s |
| `gpu-verify` | 13s | 111s |
| **`cache-restore`** | **6s** | 117s |
| `unit-rewrite` | 0s | 117s |

Against 80 s on 2026-08-27 without the restore — so the cache costs ~6 s to fetch and the
rest is ordinary variance. `cloud-init status: done`, no errors.

**The AMI id changed** — `ami-025a6e5b3b786cf61` here against `ami-0bff4343bfd56a20e`
yesterday, now Ubuntu 26.04**.1**. That is the `/latest/` SSM parameter behaving as it should
on a line AWS still rebuilds, and the contrast with the dead `pytorch-2.7-ubuntu-22.04` line
(frozen at a 2026-05-02 image) is the whole reason for the move.

## Sweep — decode is flat across a 50x context range

| in tok | out | gauge tok/s | end-to-end tok/s |
| ---: | ---: | ---: | ---: |
| 41 | 64 | 12.9 | 12.43 |
| 521 | 64 | 13.0 | 11.28 |
| 2,057 | 64 | 12.9 | 8.22 |

**Decode does not move**: 12.9 / 13.0 / 12.9 across 41 → 2,057 input tokens. A third context
point for the 2026-08-25 finding, on different hardware.

**End-to-end does move** (12.43 → 8.22) and that is prefill, not decode — prefill is linear in
the padded bucket, so a longer prompt costs more wall time while the decode rate is unchanged.
**Quote the gauge, not end-to-end.**

`tpu_jax_weight_bytes` = **6,155,450,950**, byte-identical to 2026-08-26 and 2026-08-27.
0 degenerate, 0 failed.

## xprof — the profile reproduces to 0.07%

| | 2026-08-27 | **2026-08-28** |
| --- | ---: | ---: |
| total kernel time | 1467.1 ms | **1466.0 ms** |
| kernels | 108 | **108** |
| dtype conversion | 54.0% | **54.1%** |
| fp32 gemv | 32.8% | **32.8%** |
| fusion | 12.2% | **12.1%** |
| **TensorCore** | **0.0%** | **0.0%** |

Different instance, different AMI build, restored cache — and the kernel time lands **1.1 ms
apart on 1467**. That is the strongest available evidence that the loop measures the rig
rather than the run.

**The dtype tax is unchanged at 86.9%**, as expected: nothing in this run touched it.

## Both profilers installed

`xprof 2.23.1` and `tensorboard 2.21.0`, both binaries on `PATH`. **Only xprof renders these
traces** — tensorboard has no profile plugin and xprof is the successor to
`tensorboard-plugin-profile`. tensorboard is present for everything else and so that "is it
just not installed?" never costs a debugging session.

The **raw trace came back**: 14 MB of `jaxtrace/`, including the `*.xplane.pb` and per-op
`hlo_proto.pb`. So this profile can be reopened with `xprof --logdir <run>/jaxtrace` on any
machine, after the instance is gone — which yesterday's could not, and yesterday's instance is
gone.

## Settings reviewed, unchanged

All at their measured-best values; nothing needed correcting.

```
MODEL_NAME  google/gemma-4-E2B-it     DTYPE           float16   (device-selected)
QUANT_MODE  fp16                      KV_CACHE_DTYPE  auto -> float16
PLE_BITS    4                         INT8_LM_HEAD    True      (6.155 GB, the measured best)
MAX_MODEL_LEN 4096                    MAX_NUM_SEQS    1         (B>1 raises)
PREFILL_CHUNK_SIZE ''                 root volume     100 GB gp3 500 MiB/s / 6000 IOPS
```

One change made for this run: **`JAX_CACHE_S3_URI` was set at launch**, which is what produced
the restore above. It remains empty in `tpu.env` as the shipped default.

## Not established

- **The dtype fix is still unwritten.** 86.9% of decode, unchanged. The 2026-08-27 attempt
  regressed `read_shards` 24.7 s → 79.0 s and OOMed in `quantize_ple_table`; where those 54 s
  went is still unknown and is the next thing to profile.
- **`get_install_progress`'s failure verdicts** remain CPU-tested only — nothing failed again.
