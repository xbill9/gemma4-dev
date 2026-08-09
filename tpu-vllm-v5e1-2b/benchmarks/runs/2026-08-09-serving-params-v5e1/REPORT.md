# Serving-parameter validation — Gemma 4 E2B on TPU v5e-1, vLLM

**Run:** 2026-08-09 · `tpu-vllm-v5e1-2b` (v5litepod-1, 1x1, **spot**, us-west4-a, `aisprint-491218`)
**Engine:** vLLM `0.26.1rc1.dev125+ga7a204cc6`, image `vllm/vllm-tpu:nightly`
@ `sha256:5b63034a1d04e6f9f3232f0920b81462da2e2d3d721b33cc41033b9eb38f712f`
**Purpose:** decide the rig's serving flags by measurement rather than derivation, and validate the
resulting configuration end to end.

Five arms. Raw cells in [`results/cells-armE.jsonl`](results/cells-armE.jsonl) (36 records),
arm inventory in [`results/arms-summary.md`](results/arms-summary.md), scripts alongside.

## Headline

| | value |
| :--- | ---: |
| Peak aggregate output | **1,496.5 tok/s** ±0.8% (ctx 128, 64 clients) |
| Single-stream output | 123.7 tok/s @ 15.9 ms TTFT, 8.02 ms TPOT |
| Cost per M output tokens | **$0.107** at peak (spot) · $1.30 single-stream |
| Cold boot | **857 s** |
| Warm boot, compile cache mounted | **497 s** (−42%) |
| KV cache | 5.52 GiB = 321,344 tokens |

## The recommended configuration

```
--dtype bfloat16 --kv-cache-dtype auto --max-model-len 32768
--max-num-batched-tokens 4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.92
--enable-prefix-caching --disable-chunked-mm-input --limit-mm-per-prompt '{"image":4,"audio":1}'
--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```
plus `-v ~/.cache/vllm:/root/.cache/vllm` on the `docker run`.

**One change from the rig's previous production config: `max-model-len` 16384 → 32768.** Everything
else is either an explicit restatement of a resolved default, or unchanged.

`gpu_memory_utilization` and `kv_cache_dtype` **do not appear in the engine's `non-default args`**
when passed at these values — vLLM itself confirms they are no-ops. They are pinned for auditability,
because these defaults are computed several layers from where they appear to be declared (below).

## Measured matrix — 12 cells x 3 reps (arm E)

Output tok/s (± is coefficient of variation across 3 reps):

| ctx \ clients | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| 128 | 123.7 ±0.0% | 433.9 ±1.6% | 1152.5 ±3.4% | **1496.5 ±0.8%** |
| 1024 | 120.4 ±0.0% | 415.4 ±0.2% | 991.0 ±0.0% | 1258.7 ±2.4% |
| 8192 | 94.2 ±0.1% | 254.4 ±0.5% | 399.6 ±1.0% | 324.3 ±1.0% |

Median TTFT (ms):

| ctx \ clients | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| 128 | 15.9 | 31.8 | 98.8 *(±56%)* | 247.3 |
| 1024 | 40.3 | 51.5 | 170.1 | 421.7 |
| 8192 | 289.4 | 304.2 | 594.4 | 11733.7 |

Median TPOT (ms):

| ctx \ clients | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| 128 | 8.02 | 8.96 | 13.18 | 40.72 |
| 1024 | 8.05 | 9.17 | 14.46 | 47.45 |
| 8192 | 8.42 | 14.13 | 36.93 | 104.82 |

Throughput cells are stable (cv ≤ 3.4%, most ≤ 1%). **The 128/16 TTFT cell is not** — cv 56% across
reps, so no point value should be quoted for it.

> **Comparison with `2026-08-06-vllm-sweep-v5e1` is valid but not identical.** Arm A reproduced that
> run's configuration on this node and matched it to the digit (TPOT 8.05/8.08/8.43 ms, KV 5.52 GiB,
> 10,043 blocks, ITL p99 173.9 vs 172.8). The arm-E numbers above are higher because the configuration
> differs, not because the earlier run was wrong. One archived cell does look anomalous: 8192/4 reported
> 346 tok/s at **71.8 ms** median TTFT, below its own single-stream TTFT of 290 ms. Today's 8192/4 is
> 254.4 tok/s at 304.2 ms, consistent with its c=1 neighbour. Prefer today's.

## What each arm established

### Arm B — the multimodal floor is 2496, and the default cannot boot

Removing `--max_num_batched_tokens` entirely does **not** fall back to something safe:

```
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
is larger than max_num_batched_tokens (2048). Please increase max_num_batched_tokens.
```

Two facts in one line: the per-item multimodal ceiling for this model at
`{"image":4,"audio":1}` is **2496 tokens**, and the resolved default is **2048**.

**Where 2048 and 256 come from, since it is not where you would look.** `SchedulerConfig`
declares `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` / `DEFAULT_MAX_NUM_SEQS = 128`, but
`EngineArgs.get_batch_defaults()` overrides from a dict keyed by usage context and gated on device
memory. Both gates fail on this hardware:

- `current_platform.get_device_total_memory()` raises `NotImplementedError` on tpu-inference and is
  swallowed by a bare `except`, so `device_memory` reads **0** — the generic non-H100 branch wins,
  giving `max_num_seqs = 256`.
- The TPU-specific override that would set `max_num_batched_tokens` per chip (V6E 1024 / **V5E 512** /
  V5P 256) tests `chip_name == "V5E"`, and `get_device_name()` returns **`'TPU V5E'`**. The string
  never matches, so **vLLM's per-chip TPU tuning is dead code on this part.**

Verified on the running engine: `device_name: 'TPU V5E'`, and `get_device_total_memory()` raising.

### Arm A — control; the archive is trustworthy

Production config reproduced `2026-08-06` exactly: `total_hbm_limit_cap_gb=14.49`,
`total_hbm_used_gb=8.97`, `total_hbm_avail_gb=5.52`, `num_blocks=10043`, `num_kv_cache_groups=1`.
Boot 1,089 s.

### Arm C — the proposed config was worse on every cell

`max-model-len 32768` + `max-num-seqs 64` + `max-num-batched-tokens 2496`: **−24.7% throughput** at
8192/64, +22% TPOT, +27% TTFT, and boot 1,701 s. Registered predictions and their outcomes:

| prediction | outcome |
| :--- | :--- |
| ITL p99 tail falls 122 → ~75 ms | **falsified** — 173.9 → 174.2 ms, unchanged |
| (revised) tail unchanged because 2496 pads to the 4096 bucket | **confirmed** |
| c=1 TPOT improves from capping max_num_seqs | **falsified** — 8.05 → 8.32 ms, worse |
| KV pool unchanged | confirmed (321k tokens both) |

**Mechanism: token buckets are powers of two.** Both 2496 and 4096 compile the identical ladder
`16,32,…,4096`. A 2496-token chunk therefore executes in a 4096-shaped kernel — same cost, 61% of the
work. Since the multimodal floor is 2496 and the next bucket down is 2048, **every value in
(2048, 4096] is equivalent in cost and 4096 does the most work: 4096 is optimal in that interval.**

### Arm D — `max-model-len` was innocent

Production config with **only** `max-model-len 32768`: equal or better than arm A on every cell
(+4.4% at 1024/16, −5.5% TPOT at 8192/64). So arm C's regression belonged entirely to the two flags
added alongside.

**Block size scales with context to hold blocks-per-request constant**, which is why longer context
is free:

| arm | max_model_len | block_size | blocks/request | KV blocks | KV tokens | c=1 TPOT |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 16,384 | 32 | 512 | 10,043 | 321,376 | 8.05 ms |
| D/E | 32,768 | 64 | 512 | 5,021 | 321,344 | 8.02 ms |
| C | 32,768 | **16** | **2048** | 20,086 | 321,376 | 8.32 ms |

**Do not pin `--block-size`.** The backend derives it, and the derivation is what keeps long context
free. Arm C's 2048 blocks/request cost 3.4% of decode.

> Arm D's boot time was **not captured** — no polling loop was set, and the uvicorn
> `Application startup complete` line carries no timestamp. Recorded as unmeasured.

### Arm E — the recommended config, validated end to end

Boot **857 s** — *faster* than arm A's 1,089 s despite double the context, which retires the concern
that 32768 costs compile time. Compile phase 19:22:10 → 19:33:35 = **685 s, 80% of boot.**

Functional tests, all passing:

| test | result |
| :--- | :--- |
| chat completion | ✅ |
| **tool calling** | ✅ `{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}` |
| **multimodal image** | ✅ correctly described a synthetic gradient PNG |
| **long context** | ✅ **26,016 prompt tokens** accepted (>16,384, so 32768 is genuinely active) |

### Compile cache — measured, and worth mounting

The JAX compile cache is `/root/.cache/vllm` (197 MB) and is **container-local by default**, destroyed
on `docker rm`. Mounting it to the host:

| | seconds |
| :--- | ---: |
| cold (empty cache) | 857 |
| warm (197 MB cache retained across `docker rm`) | **497** |
| saving | **360 s, −42%** |

Not a total elimination of the 685 s compile — 497 s still covers weight load, KV allocation and
warmup compilation that is not cached — but 6 minutes per restart, on a preemptible node.

## Recommended client counts

| workload | context | clients | measured |
| :--- | ---: | ---: | :--- |
| Interactive chat / agent turns | ≤1K | **16** | 991–1,152 tok/s, 99–170 ms TTFT, ~14 ms TPOT |
| Max throughput | ≤1K | **64** | 1,259–1,496 tok/s, 247–422 ms TTFT |
| RAG / long documents | 8K | **16** | 399.6 tok/s, 594 ms TTFT |
| Long-context interactive | 8K | **≤4** | 254 tok/s, ~304 ms TTFT |

**Keep `clients × context` under ~250,000 tokens** — about 78% of the 321,344-token pool. Above it,
tail latency degrades faster than throughput improves: at 8192/64 (524K wanted) median TTFT is
**11.7 s**.

## Cost

Live Cloud Billing Catalog rates for `us-west4`, per chip-hour: 3-yr commit **0.5400**, spot
**0.5779**, flex-start (`DWS Defined Duration V5e`) **0.6000**, 1-yr commit / Reserved **0.8400**,
on-demand **1.2000**.

$ per 1M output tokens at measured throughput:

| workload | tok/s | spot | flex | on-demand |
| :--- | ---: | ---: | ---: | ---: |
| 128 ctx, 64 clients | 1,496.5 | **$0.107** | $0.111 | $0.223 |
| 1K ctx, 64 clients | 1,258.7 | $0.128 | $0.132 | $0.265 |
| 1K ctx, 16 clients | 991.0 | $0.162 | $0.168 | $0.336 |
| 8K ctx, 16 clients | 399.6 | $0.402 | $0.417 | $0.834 |
| single client | 123.7 | $1.298 | $1.347 | $2.695 |

24/7 for a month (730 h): spot **$422**, flex **$438**, on-demand **$876**.

**Spot vs flex-start.** Flex-start costs $0.0221/h more (3.8%). A preemption costs a full rebuild —
$0.1376 cold, $0.0798 warm. So spot is cheaper only while preemptions are **less frequent than every
6.2 h (no cache mount) or every 3.6 h (cache mounted)**. Flex-start additionally self-terminates via
`--max-run-duration`; spot and on-demand bill until deleted.

**Batching dominates the pricing decision.** 1 → 64 clients at short context is a **12.1x** reduction
in cost per token, against 2.08x between on-demand and spot.

## Verification arms (2026-08-09, second session)

Five further arms run specifically to remove cited-but-unmeasured claims.

| # | check | result |
| :--- | :--- | :--- |
| F | `--kv-cache-dtype fp8_e4m3` re-verify | **1.000x** — 5,021 blocks, 5.52 GiB, 321,344 tokens, identical to bf16. Shape `(64,1,2,256)` → `(64,1,4,256)`, dtype really `float8_e4m3fn`. Reproduces the earlier result at a *different* page size (64 vs 32), so word alignment is not a block-size artifact |
| G | `--kv-cache-dtype int8` | **Claim retracted.** Rejected at the CLI enum: `invalid choice: 'int8'`. It never reaches the scale-hardcoding path. Full accepted set is 16 values (see `@QUANTIZATION.md`) |
| H | `--gpu-memory-utilization 0.95` | **Fails, confirmed.** cap 14.96 / weights 8.97 / KV 5.99 GiB, `num_blocks=5451` = 348,864 tokens (+8.6%), then `RuntimeProgramAllocationFailure` at **691 s**: wants 384.11 M, 347.33 M free. Second independent reproduction (earlier: 384.11 M / 346.77 M at page size 32) |
| I | prefix caching | **Works.** 4,813-token prefix re-sent → **4,800 hits (99.7%)**; latency 0.244 s → 0.100 s → 0.033 s (**7.4x**); a different prefix produced **zero** new hits |
| J | flex-start provisioning | **Command verified.** QR reached `ACTIVE` with `FLEX_START` / `maxRunDuration: 14400s` / `v5litepod-1`, then deleted cleanly (~10 min, ~$0.10) |

Warm boot was measured twice: **497 s and 500 s** (cold 857 s).

> **A broken instrument nearly produced a false negative.** The first prefix-caching script was mangled
> by nested shell quoting; all four requests failed in ~15 ms and would have read as "no benefit". Only
> `vllm:prefix_cache_queries_total = 0` revealed that no inference had run. Rewritten as a standalone
> file. Same lesson as fp8: **a clean-looking number is not evidence the thing was measured.**

## Claims still cited rather than measured here

- **2.8x sliding-window KV** — `num_kv_cache_groups=1` is measured on every arm; the 2.8x is arithmetic
  from `@MODELS.md` layer geometry.
- **~3.15 GiB moved per decode step / "49% of peak bandwidth"** — derived from layer geometry, not an
  instrument reading. The 8.02 ms TPOT it is compared against *is* measured.
- **Preemption dynamics** — rebuild costs use measured boot times, but this node was never preempted.
- **Upstream's failing correctness test for E2B** — read from tpu-inference's support table, not
  independently reproduced. Local quality probes were clean.

## What this run cost

The node was a **spot** `v5litepod-1` in `us-west4-a`, alive **5.8 h** end to end, at the live
catalog rate of $0.5779/chip-hr:

| item | |
| :--- | ---: |
| node, 5.8 h at spot | **$3.38** |
| flex-start verification QR (~10 min, reached ACTIVE then deleted) | $0.10 |
| **total** | **~$3.48** |

The same 5.8 h would have been $3.51 on flex-start and **$7.01 on-demand**. Nine arms, 36 benchmark
cell-runs, four functional tests and five verification probes for under four dollars — which is the
argument for measuring rather than deriving, given three of the derived recommendations turned out to
be wrong.

## Still unexploited: 2.8x KV capacity

`Hybrid KV cache layout: num_kv_cache_groups=1` on every arm — all 15 cached layers get full-length
allocation, though 12 are sliding-attention layers windowed at 512 tokens. Windowing would be worth
**2.8x** the KV capacity at 16K context at zero quality cost. It is unreachable: tpu-inference sets
`disable_sliding_window = len(head_size_set) > 1`, and Gemma 4 has two head dims (256 sliding /
512 full), with a source `TODO: enable sliding windows once mixed dims support`. Re-check on image
bumps.

## Not measured here

`max-model-len 65536`, `max-num-batched-tokens 8192`, `VLLM_TPU_BUCKET_PADDING_GAP=128`,
`ATTN_BUCKETIZED_NUM_REQS`, `SLICE_ROPE_CACHE`, `NUM_PRECOMPILE_WORKERS`, n-gram speculative decoding,
and `gpu-memory-utilization` 0.93/0.94. `--disable-chunked-mm-input` was treated as a fixed product
requirement and never questioned — it is what creates the 2496 floor, and dropping it is the only
route to the 2048 bucket.
