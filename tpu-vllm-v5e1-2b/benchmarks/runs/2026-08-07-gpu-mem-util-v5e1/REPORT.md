# `--gpu-memory-utilization 0.95` on E2B, v5e-1 — KV math exact, engine does not boot

**Result: the KV pool sizes exactly as predicted and the engine still dies.** `gpu_memory_utilization`
does not reserve room for compiled program images; they are allocated out of whatever the cap leaves
behind. At 0.95 that remainder is too small and XLA fails loading `jit_structured_decode_fn`.

**Recommendation: stay at 0.92.** The hypothesis that 0.95 was a cheap +8% of KV is falsified — not
because the arithmetic was wrong (it was right to 0.03%) but because the headroom it consumes is
load-bearing.

- Date: 2026-08-07
- Host: `tpu-2B-v5e1-devops-agent`, `v5litepod-1`, us-west4-a, `aisprint-491218`
- Image, pinned by ID: `sha256:2a4a1f82793f748e02af54d77a62e590d34d2c9c68e833a8bb00d26a878a684c`
  (`vllm/vllm-tpu:nightly`), vLLM `0.26.1rc1.dev125+ga7a204cc6`
- Model: `google/gemma-4-E2B-it`, flags identical to the live baseline except the one under test
- Script: `swap_util.sh` (this directory) — stop-and-rename discipline, baseline never deleted
- Full container log: `logs/util095.log` (1,168 lines, complete)

## Prediction, registered before the run

0.95 x 15.75 = 14.96 GiB cap, minus 8.97 GiB weights = 5.99 GiB KV, at the 18 KiB/token figure from
`MODELS.md` = ~349,000 tokens / ~10,900 blocks.

## Measured

| | baseline 0.92 | arm 0.95 | predicted 0.95 |
| :--- | ---: | ---: | ---: |
| `total_hbm_limit_cap_gb` | 14.49 | **14.96** | 14.96 |
| `total_hbm_used_gb` (weights) | 8.97 | **8.97** | 8.97 |
| `total_hbm_avail_gb` (KV) | 5.52 | **5.99** | 5.99 |
| KV cache size, tokens | 321,376 | **348,896** | ~349,000 |
| `num_blocks` (per layer, x15) | 10,043 | **10,903** | ~10,900 |

Every cell lands. The token count is within **0.03%** of prediction.

## This independently confirms 18 KiB/token, and discriminates against 15

The *difference* between the two arms cancels the weights term entirely, so it tests the KV cost model
on its own:

```
tokens:  348,896 - 321,376 =  27,520          (= 860 blocks x 32 tokens, exactly)
memory:      5.99 - 5.52   =   0.47 GiB
                             0.47 GiB / 27,520 tokens = 17.9 KiB/token
```

At the retracted 15 KiB/token figure the same 0.47 GiB would have bought 32,850 tokens — **19% more
than measured**. So this run is a clean discriminator, and `MODELS.md`'s corrected 18 KiB/token is
confirmed on a second, independent configuration. The `2026-08-06-vllm-sweep-v5e1` REPORT.md memory
table still carries the old 15 KiB/token derivation and should be read with that in mind.

## Why it dies anyway

KV allocates successfully and the engine reports:

```
Init kv-cache | num_total_layers=15 | num_blocks=[10903 x 15] | regular_attn_dtype=bfloat16
              | hbm=[(14.97, 15.75)]Gb
```

Then, **13 minutes later**, during program loading:

```
RESOURCE_EXHAUSTED: E0101: RuntimeProgramAllocationFailure:
Error loading program 'jit_structured_decode_fn': Attempting to reserve 384.11M at the bottom of
memory. That was not possible. There are 346.77M free, 0B reserved, and 346.77M reservable.
```

**Short by 37.34 MB.** The mechanism: after KV init, 14.97 of 15.75 GiB is resident, leaving ~799 MB.
Programs loaded before this one consumed ~452 MB of it, so total program demand is **at least ~836 MB**
— against the ~1,290 MB that 0.92 leaves free. `gpu_memory_utilization` governs weights + KV only;
compiled programs come out of the remainder and are invisible to the knob.

Note the cost of finding out: the failure lands **~17 min in**, after the full compile, not at
allocation. Unlike the qwix arms — which failed in 2.5-4 min — probing this knob is not cheap.

## What this does and does not establish

**Does:**

- 0.95 is not viable on this build, model and chip. Reproducible: the failure is deterministic
  allocation, not a race.
- `gpu_memory_utilization` excludes compiled program images. This is the general lesson and it applies
  to any attempt to tighten the cap.
- 18 KiB/token for E2B, confirmed by difference on an independent configuration.

**Does not:**

- Rule out 0.93 or 0.94. Estimating from the ~836 MB program demand: 0.94 leaves ~963 MB (margin
  ~127 MB) and 0.93 leaves ~1,127 MB (margin ~291 MB). Both are arithmetic, not measurements, and
  program demand may itself shift with block count.
- Justify chasing them. The upside is +5.8% KV at 0.94 and +3.0% at 0.93, each costing ~17 min of
  downtime to test, with 0.94's margin only 3.4x the size of the shortfall that just killed 0.95.
- Say anything about other engine builds — program demand is a property of what XLA compiled.

## Rig state

**Restored to the 0.92 baseline.** The original container was stopped and renamed
`vllm-gemma4-park-092`, never deleted, then started back as `vllm-gemma4` with its compile cache warm.
The failed arm remains as the exited `vllm-gemma4-util095` with its full log captured here — the qwix
run lost an arm's log to a reused container name, so this script names containers per configuration.

## Reproducing

`./swap_util.sh forward 0.95` on the host, then `./swap_util.sh back`. Three pre-flights run before
anything stops — Secret Manager reachable, image present by ID, park name free — so a failure cannot
strand the rig with no serving container.

**`gcloud compute tpus tpu-vm ssh` does not work from the dev sandbox**, crashing with
`ConnectionResetError` on its own internal API call; this also breaks the MCP tools that shell out
through it (`get_vllm_docker_logs`, `manage_vllm_docker`, `get_tpu_system_logs`, `run_vllm_benchmark`).
Direct `ssh -i ~/.ssh/google_compute_engine xbill@<ip>` works. Plain gcloud API calls are unaffected.
