# What the whole exercise cost — derivation

**This is arithmetic, not a bill.** AWS drops terminated instances from
`describe-instances` after about an hour, so exact billed seconds could not be recovered for
most of these. Every duration below is a *lower bound*: it is the measured launch-to-healthy
time, and each instance actually lived until its artifacts were captured and it was terminated.

## Instances launched, 2026-08-31 / 09-01

| # | purpose | id | type | market |
|---|---|---|---|---|
| 1 | spot re-run, PyTorch | `i-0a3295d189ba6fc28` | 2xlarge | spot |
| 2 | cross-rig, JAX | `i-0ffdb819663bacc77` | 2xlarge | spot |
| 3 | cross-rig, vLLM | `i-04a83e4a41be52ffa` | 2xlarge | spot |
| 4 | cross-rig, PyTorch | `i-02e79988a6cbeecbf` | 2xlarge | spot |
| 5-7 | boot campaign v1 (discarded, coarse polling) | `i-04c8e270624c853e8`, `i-050e851d43605e5e2`, `i-0f71498e346bd1d60` | 2xlarge | spot |
| 8-16 | boot campaign v2, 9 cold boots + warm legs | see `boot_results.json` | 2xlarge | 6 spot, 2 on-demand, 1 backfilled spot |
| 17 | 32 GiB RAM test | `i-0785db75689508c2a` | **4xlarge** | spot |
| 18 | prefetch A/B, reclaimed 10 min in | `i-06fa8e3ba5dfa2a55` | 2xlarge | spot |
| 19 | prefetch A/B, re-run | `i-02109c6c38368dfec` | 2xlarge | on-demand |

**19 instances**, not the "roughly 15" an earlier draft claimed.

## Hours

| source | hours | basis |
|---|---:|---|
| boot campaign, 9 cold boots | 1.57 | `boot_results.json`, health times |
| boot campaign, 9 warm legs | 0.30 | same |
| prefetch A/B (cold + 3 restarts) | 0.56 | `2026-09-01-prefetch-ab-g5g/result.json` |
| serving runs + discarded v1 + reclaimed attempt | 1.63 | session log launch-to-terminate |
| **subtotal, `g5g.2xlarge`** | **4.06** | |
| 32 GiB test, `g5g.4xlarge` | 0.38 | `2026-08-31-boot-32gib-g5g/result.json` |

## Bounds

`g5g.2xlarge` is $0.556/hr on-demand (AWS Pricing API, us-east-1, verified 2026-08-31) against
measured spot of $0.3813-$0.4416. `g5g.4xlarge` is twice the vCPU and roughly twice the rate.

| scenario | cost |
|---|---:|
| everything at the cheapest measured spot | **$1.84** |
| everything at on-demand | **$2.68** |
| the actual mix (mostly spot, three on-demand) | **~$2.30** |

**So the defensible claim is "under $3", not "under $2".** The earlier figure came from an
estimate of 3.5-4 instance-hours that was never reconstructed from the artifacts; at 4 hours
and the on-demand rate it was already $2.22, i.e. false at the top of its own range.

None of this changes the argument. The point is that discarding a finished campaign over a
poll-interval bug cost twenty minutes and small change, and that holds at $1.84 or $2.68.
