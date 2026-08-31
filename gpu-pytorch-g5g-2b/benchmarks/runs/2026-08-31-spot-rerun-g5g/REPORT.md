# 2026-08-31 — spot re-run, `gpu-pytorch-g5g-2b`

**A repeat of the 2026-08-29 profile-and-fixes sweep on the same build, to see what the rig's
run-to-run noise actually is.** `g5g.2xlarge` spot in **`us-east-1a`** (`i-0a3295d189ba6fc28`,
AMI `ami-07a66fa2acbcfea88`, $0.4416/hr), torch 2.12.0+cu132 on Python 3.13, build id
`060a572aeb55` — **byte-identical payload to run 2**. Driven by `sweep.py`; artifacts are
`sweep.json`, `sweep.log`, `metrics.prom` and `REPORT.json` (schema 1.1, validated).

## Result

**Decode 11.02 tok/s (median of 10 cells), 10/12 cells ok, 2 infeasible, 0 failed, 0 degenerate.**

| input tok | output | decode tok/s | end-to-end tok/s | warm-up tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 92 | 32 | 10.67 | 10.14 | 10.64 |
| 92 | 78 | 10.69 | 10.47 | 10.72 |
| 633 | 32 | **11.33** | 10.06 | 11.24 |
| 633 | 87 | 11.00 | 10.52 | 11.25 |
| 1259 | 32 | 11.09 | 8.96 | 10.85 |
| 1259 | 87 | 11.05 | 10.15 | 11.19 |
| 2501 | 32 | 10.97 | 7.34 | 10.82 |
| 2501 | 89 | 11.17 | 9.46 | 11.13 |
| 3746 | 32 | 11.20 | 6.00 | 11.20 |
| 3746 | 89 | 10.95 | 8.39 | 11.16 |

Cumulative over the whole run: 2,365 completion tokens in 214.88 s of decode = **11.01 tok/s**,
agreeing with the per-cell median to 0.1%. Weights **10.209 GB**, unchanged.

**The two 3,800-context cells are `infeasible`, not failures.** The prompt tokenises to 4,630
against a `--seq` of 4096, and the server refuses it with a clean 400 naming both numbers. Run 2
hit the identical wall with the identical message, so this is a reproducible bound, not a
regression — and it is the reason the sweep tops out at 3,746 input tokens rather than 4,096.

## The point of this run: the noise floor is ~2%, so the 15% headline survives

Same build id, same AMI, same instance type, **different host and different AZ**:

| | run 2 (2026-08-29) | **run 3 (this)** | delta |
| --- | ---: | ---: | ---: |
| build id | `060a572aeb55` | `060a572aeb55` | identical |
| AZ | us-east-1d | us-east-1a | — |
| decode median | 10.840 | **11.021** | **+1.7%** |
| decode min | 10.272 | 10.669 | +3.9% |
| decode max | 10.913 | 11.327 | +3.8% |
| cells ok | 10/12 | 10/12 | — |
| degenerate | 0 | 0 | — |

**+1.7% on the median across two hosts running the same bytes.** That is the number to hold
against every comparison this rig is used for, and it settles one that mattered: the
JAX-versus-PyTorch headline is **15%** (12.80 against 10.88), which sits far outside a ~2% band.
The framework difference is real and is not host lottery.

It also puts a floor under what is worth chasing. The `ple0+int8head` lever on the JAX sibling
moved decode **+2.3%** — inside, or barely outside, this band. A single-run +2% on this rig
should not be reported as an improvement without a repeat.

## Flat decode, reproduced a third time

Decode spans 10.669 → 11.327 tok/s over inputs from 92 to 3,746 tokens — a **6.2% spread across
a 41x context range, with no monotonic trend** (the maximum is at 633 tokens, the minimum at 92).
This is the third independent reproduction of the finding, after run 1 (3.9% over 27x) and run 2
(6.2% over 41x), and it agrees with the JAX sibling's 3.4% over 100x from a different framework
and a different harness.

**End-to-end fell 10.14 → 6.00 tok/s over the same rows where decode was flat** — a 41% spread,
against decode's 6.2%. Same lesson as run 1, larger: quote
`tpu_jax_decode_tokens_per_second`, never the end-to-end rate.

## Operational notes

- **`apt-daily`/`apt-daily-upgrade` were masked over SSM before the checkpoint download**, which
  is the open action item run 1 raised after that timer restarted the serving unit mid-load and
  cost two full 9.5 GB downloads. It is still not in the bootstrap; it was done by hand here, and
  the load completed on the first attempt. **Boot to healthy endpoint was under 2 minutes.**
- **g5g spot capacity was exhausted region-wide for ~11 minutes.** All four AZs that offer the
  family rejected `g5g.2xlarge`, and `g5g.4xlarge` was refused too, so it was not a size or an AZ
  problem. A retry loop cycling the four AZs on a 60 s backoff landed on round 7 in `us-east-1a`.
  Note AWS's error text names the *other* AZs as available in every case — that is on-demand
  boilerplate and is worthless for spot.
- The instance was terminated as soon as the artifacts were captured.
