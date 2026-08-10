# 2026-08-10 — config validation on v6e-1

**The first measurement this rig has ever taken on its own hardware.** Everything else under
`benchmarks/` came in with the fork and was measured on v5e-1, or on a v6e-era fork that is a
different rig and a different vLLM. See the rig `CLAUDE.md`.

## What is being exercised

Two changes that interact, and neither has been measured here:

1. **The chip.** v6e-1 against the v5e-1 the rig was forked from. `@HARDWARE.md` is blunt that
   this is not a throughput upgrade — 2.25x the price for **1.907x** the bandwidth (the units
   trap: Google quotes v5e in GiBps and v6e in GBps) — but it is a **3.6x** capacity upgrade,
   1,151,744 KV tokens against 321,376.
2. **The setting.** `MAX_MODEL_LEN` 16384 → 32768, set in `tpu.env` on 2026-08-10. 32768 was
   measured equal-or-better on v5e at a quarter of the KV pool; here it should be free.

## Order of operations, and why

`run_all.sh` verifies the **allocation** before it runs a single throughput cell, and stops if
verification fails. This is not ceremony. `@QUANTIZATION.md` records a case where the fp8 KV
flag was accepted at the CLI, echoed in `non-default args`, praised in an engine log line,
reported in `/metrics`, and allocated a genuinely fp8 tensor — **five independent signals it had
worked** — while changing nothing. A throughput number gathered against a config that did not
take effect is worse than no number: it looks like evidence.

So `verify_allocation.py` reads the boot log and checks it against arithmetic derived from
`@MODELS.md`, never against the flags:

| check | what would falsify it |
| :--- | :--- |
| `max_model_len` is 32768 | the fork's 16384 default still in force — `tpu.env` not read |
| `block_size` derived to 64 | someone pinned `--block-size`, which breaks long-context scaling |
| KV pool matches **bf16** arithmetic | fp8 actually engaged (it never has on this stack) |
| weights resident ≈ 8.97 GiB | a weight-quantization route engaged |
| `num_kv_cache_groups` = 1 | the upstream sliding-window `TODO` landed — a **2.9x** win |

The fp8 check is the sharp one: at 18,432 B/token the pool arithmetic closes to 0.10%, while
the fp8 model would be **50% off**. The two hypotheses are not close together, so this is a real
discriminator rather than a tolerance argument.

## Cells

`OUTPUT_LEN` is 128 throughout. Roles are sized against the **v6e** pool, not v5e's:

| role | meaning |
| :--- | :--- |
| `control` | trivially fits both chips — isolates raw bandwidth/compute |
| `bandwidth` | fits both, but KV is a large share of bytes moved per step |
| `v6e_only` | **exceeds v5e's pool, fits v6e's** — v5e had to evict and recompute |
| `long_ctx` | needs `max_model_len` > 16384 — impossible on the old config *and* on v5e |

The two `long_ctx` cells (32000 tokens at concurrency 16 and 32) are the point of the exercise:
they cannot run under the pre-retarget 16384, and their 514,048 / 1,028,096 KV tokens exceed
v5e's entire pool at any setting.

## Reading the result

**The asymmetric prediction is what makes this informative.** If v6e were simply "faster", every
cell would improve by roughly the same factor. The claim in `@HARDWARE.md` is narrower:

- `control` cells should move roughly **with bandwidth (~1.9x) and no more**. A large gain here
  means something other than memory changed, and the run needs re-reading.
- `v6e_only` cells should gain **more**, because v5e was thrashing there and v6e is not.
- A flat `long_ctx` result would be the interesting negative — it would say the extra context is
  allocable but not usable at speed.

v5e reference numbers are inlined per cell from `../2026-08-07-kv-quant-v5e1/results/cells_bf16.json`
so the comparison is automatic rather than remembered. **They are not a controlled A/B** —
different chip, zone, engine build, and provisioning model. Shape, not delta.

## Files

| | |
| :--- | :--- |
| `verify_allocation.py` | reads the boot log, checks it against derived arithmetic. Runs first |
| `run_cells.py` | the throughput sweep; runs on the VM so the load generator is local |
| `run_all.sh` | driver — waits for vLLM, captures the log, verifies, then benchmarks |
| `logs/`, `results/` | captured output; logs under `benchmarks/runs/**` are committed deliberately |

## Provenance

v6e-1, `us-east5-b`, **spot** (`--provisioning-model=spot`), `vllm/vllm-tpu:nightly`,
`google/gemma-4-E2B-it`, TP=1. Spot is *dearer* than flex-start on v6e in us-east5
($1.4033 vs $1.35/chip-hr) but is not bounded by `--max-run-duration`, so a long sweep cannot be
cut off at 4h. It is preemptible with ~30s notice; a partial result set is an expected outcome.
