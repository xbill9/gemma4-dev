# 2026-08-27 — Ubuntu 26.04 base image, and an 80-second install

**`g5g.2xlarge` spot (`i-021f15b2b45e13793`), `us-east-1a`, AMI `ami-0bff4343bfd56a20e`
(Deep Learning ARM64 Base OSS Nvidia Driver GPU, Ubuntu 26.04, build 20260825),
jax 0.11.1 / Python 3.14.4, build id `6442508c3817`.**

First launch on the base image. It settles the three risks recorded as UNVERIFIED when the
AMI parameter changed, and measures the deployment-path work from the same day.

## The three risks, all settled

| Risk | Was | Measured |
| --- | --- | --- |
| Driver too old for `jax[cuda13]` (needs 580+) | 580.126.09 | **595.91.07** |
| aws CLI absent from the smaller base image | present on PyTorch DLAMI | **aws-cli/2.36.30 present**; `HF_TOKEN` written |
| PEP 668 blocks a system-wide pip install | not applicable on deadsnakes | **`/usr/lib/python3.14/EXTERNALLY-MANAGED` EXISTS** |

The third is the one worth keeping: PEP 668 is genuinely in force on this base, so
`--break-system-packages` was load-bearing rather than defensive. Without it the install
would have failed outright at the first `pip install`.

glibc is **2.43**, against 2.35 on the old base — xprof's `manylinux_2_35` wheel floor now
has real headroom instead of sitting exactly on it.

## Install: ~80 seconds, from >21 minutes

`[stage]` markers, added the same day, from `/var/log/jax-install.log`:

| Stage | Delta | Total |
| --- | ---: | ---: |
| `apt-base` | 8s | 8s |
| `python-3.14` | **1s** | 9s |
| `pip-bootstrap` | 9s | 18s |
| `jax-wheels` | 45s | 63s |
| `serving-deps` | 13s | 76s |
| `gpu-verify` | 4s | ~80s |

**`python-3.14` costs 1 second** because the base image already has it:
`python3.14 is already present at /usr/bin/python3.14; skipping deadsnakes`. On the old base
that stage was a PPA add plus a second full `apt-get update`.

For scale: on 2026-08-25 AWS reclaimed a spot instance **21 minutes in, before the wheel
install finished**. An 80-second install changes what spot is good for here — though see
below, because capacity is still the binding constraint.

## Load: read_shards 3x faster

| Stage | 2026-08-25 (`ple0`) | Here (`ple4+int8head`) |
| --- | ---: | ---: |
| download | 87.7s | **24.9s** |
| read_shards | 73.5s | **24.7s** |
| convert_params | 3.4s | 24.8s |
| device_put | 0.0s | 0.0s |

**`read_shards` is the honest comparison** — same checkpoint, same read, and it happens
*before* quantization, so the config change does not touch it. 73.5s -> 24.7s is **3.0x**, and
the root volume moving from gp3's 125 MiB/s default to **500 MiB/s / 6000 IOPS** (verified on
the volume itself: `size=100 GB type=gp3 throughput=500 iops=6000`) is the only thing that
changed about it. That confirms the hypothesis the volume was the ceiling: two unrelated
stages had been sitting on one number.

**`convert_params` is slower for a different reason and is not a regression.** The 3.4s
baseline was `ple0`; the default is now `ple4+int8head`, whose host-side chunked quantization
IS convert work. Total load here was 57.4s with the download cached, against 95.3s recorded
for the same config on 2026-08-26.

## Serving — performance-neutral, which is the point

`tpu_jax_decode_tokens_per_second` = **12.80**, inside the 12.4-13.1 band of every prior run.
The base-image change buys currency and install time, **not** speed; do not cite it as a
performance change.

`tpu_jax_weight_bytes` = **6,155,450,950** — 6.155 GB, matching the 2026-08-26 `ple4+int8head`
figure exactly, so the shipped default is what actually loaded. HBM 6.197 GB of 14.07 GB.

Cold/warm reproduced at a fixed shape (64 output tokens, warmed at the shape measured):
**18.29s cold, then 5.67s and 5.44s warm.** `tpu_jax_cold_requests_total` counted 2, and
`get_metrics` refused to let the 6.80 tok/s cumulative figure stand while they were in it.

0 degenerate responses, 0 failed requests, `max_pad_tokens` 16 — far under the 512 sliding
window, so nothing here went near the eviction bug.

## Other things this run established

- **`get_install_progress`'s cloud-init reporting works**: `status: done`, `errors: []`
  throughout, and the `Installing` / `Runtime installed` verdicts both rendered. The failure
  verdicts were not exercised — nothing failed.
- **The swapfile is not just insurance.** `swapon --show` reports the 16 GiB file with
  **850.6 MB in use** on a `g5g.2xlarge`, so the inclusive threshold is doing real work for
  `ple4`'s host-side quantization. The `mkswap` fix is confirmed on the image whose util-linux
  rejects `-q`.
- **No stale deploy.** `deploy_jax_server` reported payload root
  `/home/xbill/gemma4-dev/gpu-jax-g5g-2b` (the working tree, not the skill snapshot) and build
  id `6442508c3817`; `verify_model_health` confirmed the server serves that same id.
- **Spot capacity remains the binding constraint.** `InsufficientInstanceCapacity` in
  us-east-1b, then 1c, then 1d; only **1a** had capacity — and 1a was the *most expensive* AZ
  by spot price ($0.4463 vs 1b's $0.3831), so price is not a usable proxy for capacity here.
- `NRestarts=0`, no OOM kills, no errors in the journal.

## Not established

- **Whether the AMI change survives a long run.** This instance served minutes, not hours.
- **The `JAX_CACHE_S3_URI` compilation-cache path was NOT exercised** — it is empty by
  default and no bucket was configured, so it rendered nothing. Untested.
- **`get_install_progress`'s failure verdicts are untested on hardware** — CPU tests pin them.
- Nothing here re-measures the dtype tax, which remains 87% of decode and unaddressed.

## Artifacts

| File | What |
| --- | --- |
| `install.log` | full `/var/log/jax-install.log` with `[stage]` markers and cloud-init status |
| `journal.log` | serving journal: device policy banner, load stages, READY line |
| `gpu_probe.txt` | `verify_gpu_arch` — T4G, SM 7.5, `fp16 matmul ok: True` |
| `metrics.txt` | `get_metrics` after warm-up |
| `system.txt` | OS, glibc, python, driver, aws CLI, EXTERNALLY-MANAGED, swap, disk, pip versions |
