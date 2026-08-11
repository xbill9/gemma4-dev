# 2026-08-10 — Compute Engine vs Cloud TPU API on v6e-1: are they different?

**Short answer: not where it serves. Substantially, where it provisions.**

Same chip, same checkpoint, same engine build, same serving flags, same benchmark harness —
reached through two control planes. 10 throughput cells plus an allocation check against the
sibling `tpu-vllm-v6e1-2b` run `2026-08-10-config-validation-v6e1`.

Provenance and the caveats on it are in `README.md`. The one that matters most: **the baseline's
zone and provisioning model are unrecorded** — the JSON fields that appear to record them are
hardcoded literals in the harness. The engine build *is* pinned and *is* identical.

---

## 1. The allocation is bit-identical

| | Compute Engine | Cloud TPU API |
| :--- | ---: | ---: |
| KV cache tokens | **1,151,744** | **1,151,744** |
| weights resident | 8.97 GiB | 8.97 GiB |
| HBM available | 19.77 GiB | 19.77 GiB |
| `max_model_len` | 32,768 | 32,768 |
| derived `block_size` | 64 | 64 |
| KV cache tensors | 15 | 15 |

Not "within tolerance" — the same integers. **The control plane does not change what the chip
gives you.** All five of `verify_allocation.py`'s checks pass on this path exactly as they did on
the other, including the arithmetic one: 1,151,744 tokens against 19.77 GiB is the bf16 model to
0.01% and the fp8 model to 50%.

**The false fp8 signal reproduces here too.** The engine logs `Automatically using fp8_e5m2 for
FP8 KV cache on TPU v6e` on this path as well, and allocates bf16 regardless. That is now the
seventh recorded instance, and the first on a second *control plane* rather than a second chip.
It is a property of the engine, not of how the hardware was requested — as `@../QUANTIZATION.md`
would predict, and worth stating because it is exactly the kind of claim a migration is tempted
to re-attribute.

## 2. Throughput is the same to within the noise that matters

| cell | role | GCE tok/s | TPU-API tok/s | ratio |
| :--- | :--- | ---: | ---: | ---: |
| 128 × 1 | control | 202.97 | 202.80 | **1.001** |
| 128 × 8 | control | 1127.27 | 1124.38 | **1.003** |
| 1024 × 16 | control | 1553.53 | 1544.30 | **1.006** |
| 32000 × 16 | long_ctx | 229.91 | 249.14 | 0.923 |
| 32000 × 32 | long_ctx | 230.70 | 232.78 | 0.991 |
| 16000 × 64 | v6e_only | 468.49 | 469.20 | 0.998 |
| 16000 × 32 | v6e_only | 408.44 | 444.00 | 0.920 |
| 8192 × 64 | v6e_only | 969.59 | 989.55 | 0.980 |
| 4096 × 64 | bandwidth | 1605.43 | 1643.60 | 0.977 |
| 8192 × 32 | bandwidth | 921.50 | 938.59 | 0.982 |

Mean ratio **0.978**, median **0.986**. By role: control **1.003**, bandwidth 0.979,
v6e_only 0.966, long_ctx 0.957.

**The three control cells agree to within 0.6%, and they are the cells to trust.** They are the
smallest, they run first against a freshly-started server in both arms, and they carry almost no
KV state — so they are the least sensitive to history. Where the two paths are most comparable,
they are indistinguishable.

The larger cells sit 2–8% low on Compute Engine. **This report does not claim that gap is real**,
for a reason established below.

## 3. What the noise floor actually is — and why it doesn't license a 2% claim

Three back-to-back repeats of each control cell on the live node:

| cell | rep 1 | rep 2 | rep 3 | spread |
| :--- | ---: | ---: | ---: | ---: |
| 128 × 1 | 209.26 | 209.25 | 209.22 | 0.02% |
| 128 × 8 | 1224.65 | 1224.47 | 1224.65 | 0.01% |
| 1024 × 16 | 1809.48 | 1810.31 | 1811.97 | 0.14% |

Repeatability is **excellent** — 0.14% worst case. Naively that makes a 2% gap significant.

**It doesn't, and the same table is why.** Compare those repeats to the sweep's own numbers for
the identical cells: 202.97 → 209.2 (**+3.1%**), 1127.27 → 1224.6 (**+8.6%**), 1553.53 → 1810
(**+16.5%**). Nothing changed but the cache being warm. So this benchmark has a noise floor near
zero and a **history sensitivity up to 16%** — the dominant term is not measurement error, it is
what the server saw beforehand, and that is not controlled between the two arms beyond running
the cells in the same order.

The cells showing the largest deficits are precisely the ones the harness's own docstring flags
as prefix-cache contaminated at seed 0: `16000×32` follows `16000×64`, `8192×32` follows
`8192×64`, `32000×32` follows `32000×16`. **A 2–8% difference in those cells is not
attributable.** The clean cells differ by 0.3%.

TTFT tells the same story: median ratio **1.007**, with the three outliers (1.59, 1.28, 1.23)
landing on exactly those contaminated cells.

**Verdict: no measured serving difference between the control planes.** To claim one would need
repeats with a cold cache per cell and varied seeds, in both arms.

## 4. Where they genuinely differ: everything before the model loads

This is the real answer, and none of it shows up in a throughput table.

| | Cloud TPU API | Compute Engine |
| :--- | :--- | :--- |
| **Docker** | preinstalled in `v2-alpha-tpuv6e` | **absent** — the boot fails at the first pull |
| **quota pool** | 512 v6e chips, us-east5 | CT6E=32, eight regions, **none in us-east5** |
| **readiness** | QR `ACTIVE` implies the node is up | `RUNNING` means the VM booted, nothing more |
| **discovery** | `tpus tpu-vm list` | `compute instances list` — the other returns nothing |
| **SSH** | `tpus tpu-vm ssh` | `compute ssh` — the other cannot reach the instance |
| **flex-start** | `--provisioning-model=flex-start` | `FLEX_START`, and available from the Makefile |
| **auto-stop** | flex-start only | any model, `--max-run-duration` + termination action |

**The failure modes are the migration cost, not the flags.** Every one of these failed *quietly*
on the first attempt: the missing Docker failed the startup script while the instance reported
`RUNNING` for the next half hour; the quota mismatch would have rejected a create in the zone the
sibling provisions in daily; four MCP tools shelled to `tpu-vm ssh` and returned not-found against
a VM that was plainly up. Flag-mapping is the part that is easy and the part that gcloud checks
for you.

Cost was identical where it could be compared: flex-start v6e is **$1.35/chip-hr** in both
regions. Spot is *dearer* than flex-start in both (europe-west4 $1.782, us-east5 $1.403).

## 5. What this run does not establish

- **That the 2% is or isn't real.** Section 3.
- **That flex-start behaves the same under contention.** Capacity was granted immediately here
  and `--request-valid-for-duration=2h` never engaged. Nothing was learned about DWS queueing.
- **That the template fix boots from scratch.** This instance's Docker was installed by hand and
  the startup script re-triggered. `startup_script_template.sh` now does it, and a test pins the
  ordering, but no instance has yet booted clean from it.
- **Anything about zone effects.** The baseline's zone is unrecorded.
