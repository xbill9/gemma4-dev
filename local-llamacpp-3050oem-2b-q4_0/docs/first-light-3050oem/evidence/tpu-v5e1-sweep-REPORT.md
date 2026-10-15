# Serving Sweep — Gemma 4 E2B on TPU v5e-1, vLLM on spot capacity

**Run:** 2026-08-06 · `tpu-2B-v5e1-devops-agent` (v5litepod-1, 1x1, **spot**, us-west4-a, `aisprint-491218`)
**Engine:** vLLM `0.26.1rc1.dev125+ga7a204cc6`, `vllm/vllm-tpu:nightly`, tpu-inference (JAX, flax_nnx), TP=1, `max_model_len` 16384
**Matrix:** 4 concurrency levels {1, 4, 16, 64} × 4 context lengths {128, 1024, 8192, 32768}, output 128 tok/cell, `--ignore-eos`
**Coverage: 12 measured cells, 0 failed, 4 infeasible-by-config** (recorded, not silently skipped).

Full matrices in [`tables.md`](tables.md), machine-readable cells in
[`results/summary.json`](results/summary.json), raw `vllm bench serve` dumps in `results/*.json`,
per-cell logs in `logs/`. Report: [`../../reports/2026-08-06-gemma4-e2b-v5e1.json`](../../reports/2026-08-06-gemma4-e2b-v5e1.json)
(schema 1.1). Regenerate everything with `python3 aggregate.py`.

**The 32768 row was not attempted.** It is recorded `infeasible` because 32768 + 128 output exceeds
the configured `max_model_len` of 16384 — determined from the config, not from a failed run. That is
a weaker claim than a measured failure and is labelled as such here and in the report's `error` field.

## Headline

| | value |
| :--- | ---: |
| Peak aggregate output | **1,382 tok/s** (ctx 128, c=64) |
| Single-stream output | 120 tok/s @ 15.9 ms median TTFT, 8.05 ms TPOT |
| Cost per M output tokens | **$0.12** at saturation · $1.34 single-stream |
| Spot rate | $0.5779 / chip-hour (us-west4, live catalog) |
| Time to healthy (cold) | 986 s, of which 738 s is XLA/JAX compile |

## Cold start is compilation, not weights

Engine init took 814 s of the 986 s to healthy, and **compilation alone was 738 s — 91% of init**.
Weight download was 9.8 s for a 9.54 GiB checkpoint. On spot capacity this matters more than it
looks: a preemption repays the full ~16 minutes, so the effective cost of an interruption is far
above the per-hour rate difference between spot and on-demand.

## Concurrency helps until KV binds, then reverses

Aggregate output tok/s:

| ctx \ users | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| 128 | 120 | 419 | 1,022 | **1,382** |
| 1024 | 120 | 411 | 909 | 1,140 |
| 8192 | 94 | 346 | 378 | 346 |

At ctx 128 the chip returns 11.5× single-stream throughput at 64 users. At ctx 8192 the curve
**peaks at 16 users and falls back at 64** (378 → 346), and per-stream throughput collapses from
119 to 10 tok/s. The engine reported 321,376 resident KV tokens, so 64 streams at 8192 context want
~532K tokens of KV — roughly 1.7× what is resident. Past that point added users buy queueing, not
throughput.

Tail latency shows the same wall more sharply than the mean: median TTFT at ctx 8192 goes
290 ms → 72 ms → 541 ms → **10.2 s** across c=1→64, with p99 at 17.5 s. Anything past c=16 at long
context is not interactive.

## Memory anatomy

| | GiB |
| :--- | ---: |
| Total HBM | 15.75 |
| Usable (engine cap) | 14.49 |
| Weights | 8.97 |
| KV cache | **5.52** |

KV is allocated for **15 layers**, at **18 KiB/token** in bf16, giving 321,376 resident tokens.
The figure is read from `total_hbm_avail_gb=5.52GiB` in the boot log, and it closes the budget
exactly: 8.97 + 5.52 = 14.49.

> **Corrected 2026-08-07.** This table previously said **4.60 GiB and 15 KiB/token**, derived from the
> block count as `10,043 × 32 × 15 layers`. That derivation took a boot log line describing **one layer
> group** as if it described the whole model. E2B is hybrid — 28 sliding layers at 256-dim and 7
> full-attention layers at 512-dim (`MODELS.md`) — so a single 256-dim head does not characterize it.
> The old number left 17% of the KV budget unexplained.
>
> 18 KiB/token is confirmed two independent ways: the budget closes to the digit above, and the
> `--gpu-memory-utilization 0.95` arm in [`../2026-08-07-gpu-mem-util-v5e1/`](../2026-08-07-gpu-mem-util-v5e1/REPORT.md)
> measured **27,520 extra tokens for 0.47 extra GiB = 17.9 KiB/token** — a difference measurement that
> cancels the weights term entirely. At 15 KiB/token that same 0.47 GiB would have bought 32,850
> tokens, 19% more than observed.
>
> **No measured quantity in this run changes** — throughput, TTFT and cost are untouched. The corrected
> fields in `../../reports/2026-08-06-gemma4-e2b-v5e1.json` are `memory.kv_cache_gib`,
> `memory.kv_bytes_per_token`, and the two `notes` strings.

## Comparing this against the v6e-1 report

The root `benchmarks/ROLLUP.md` puts this beside `2026-07-21-gemma4-e2b-v6e1` (peak 2,215 tok/s,
$0.17/M). **The two are not a controlled comparison** and the rollup does not claim they are:

| | v6e-1 report | this run |
| :--- | :--- | :--- |
| Sweep shape | 1-D (5 concurrency points) | 2-D (4 ctx × 4 conc) |
| `max_model_len` | 65536 | 16384 |
| Engine | 0.23.1rc1.dev1076 | 0.26.1rc1.dev125 |
| Provisioning | flex-start | spot |
| KV cache dtype | fp8_e5m2 | bf16 (default) |

The cost figures are the most nearly comparable pair, and even there the rate basis differs
(flex-start DWS vs spot). Peak throughput is *not* comparable across them: a 1-D sweep's peak is at
whatever single context it used.
