> **STALE — describes the Compute Engine instance path this rig no longer has.**
> `server.py` provisions GKE node pools as of 2026-08-25; `startup_script_template.sh` and every
> `gcloud compute instances` / `tpus tpu-vm` command below are gone from the rig. `CLAUDE.md` and
> `gke/README.md` are correct. Rewrite or delete this file; do not follow it.

# Serving parameters for this rig — the decision, and why

**Status: ported 2026-08-10 from the v5e rig, not yet measured here.** This file states what the rig
should run and, for every line, whether the reason behind it is a fact about the *model*, the *stack*,
or the *chip* — because only the first two crossed the fork.

> **The evidence is v5e-1 evidence.** It lives in the sibling rig at
> [`../tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-09-serving-params-v5e1/REPORT.md`](../tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-09-serving-params-v5e1/REPORT.md)
> — nine arms on a spot `v5litepod-1` in us-west4-a, 36 benchmark cell-runs. **Nothing in this rig's
> `benchmarks/` was measured on v6e-1** (see `CLAUDE.md`). Rows below are tagged:
>
> - **carries** — the reason is a property of the checkpoint or of vLLM/tpu-inference, so the chip is
>   not part of the argument.
> - **v5e-only** — the reason involves v5e's memory or timing. Restated for provenance, *not* asserted
>   here. Re-measure before relying on it.
> - **v6e** — grounded in a v6e-1 measurement (`../HARDWARE.md`, `gemma4-e2b-v6e1-demo.html`).

## The configuration

```
vllm serve google/gemma-4-E2B-it \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --disable-chunked-mm-input \
  --limit-mm-per-prompt '{"image":4,"audio":1}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

plus `-v ~/.cache/vllm:/root/.cache/vllm` on the `docker run`.

**One change from what this rig ran before: `MAX_MODEL_LEN` 16384 → 32768**, now set in `tpu.env`.
`server.py:119` supplies the old default and both deploy paths read the same value through
`_vllm_serve_flags()`, so the env file is the only place to change it.

Note `--gpu-memory-utilization` is **absent** from the list above, deliberately — see the table.

| flag | value | basis | why |
| :--- | :--- | :--- | :--- |
| `--max-model-len` | **32768** | v5e-only → safe here | measured equal-or-better than 16384 on every v5e cell (+4.4% at 1K/16) while costing no KV capacity, on a chip with **a quarter** of this one's KV pool. **65536 is separately known-good on v6e-1** — see below |
| `--max-num-batched-tokens` | **4096** | carries | bounded on both sides by the model's multimodal item size and vLLM's power-of-two buckets; 4096 is forced. See §1 |
| `--max-num-seqs` | **unset (256)** | carries | capping to 64 measured worse; the cap never binds below 64 offered load. The default resolution is a stack property (§3) |
| `--kv-cache-dtype` | **auto** | **v6e — confirmed** | fp8 KV bought **nothing on this chip**: the recorded v6e allocation is bf16 to 0.10% despite the engine logging fp8. See the rescan below. Never set it |
| `--dtype` | **bfloat16** | carries | no weight-quantization route boots (`../QUANTIZATION.md`) |
| `--gpu-memory-utilization` | **unset** | v5e-only | 0.92 was chosen because **0.95 failed after a 691 s compile on v5e-1's 16 GB**. That failure is a v5e memory result and does not transfer to 32 GB. Left underived here rather than porting a number whose only justification is another chip's ceiling |
| `--tensor-parallel-size` | **1** | carries | v6e-1 is one chip, and E2B's `num_key_value_heads=1` cannot shard — more chips would multiply its KV cost, not divide it (`../MODELS.md`) |
| `--enable-prefix-caching` | **on** | carries | 99.7% hit on a repeated 4,813-token prefix, 7.4x faster prefill |
| `--block-size` | **never set** | carries | the backend derives it from `max_model_len`; that derivation is what keeps long context free (§2) |

## What v6e-1 actually measured

One allocation is recorded on this chip and checkpoint, at **65,536 context** — the source
`../HARDWARE.md` cites for its v6e-1 row, captured in `gemma4-e2b-v6e1-demo.html`:

| | GiB |
| :--- | ---: |
| Total HBM | 31.24 |
| E2B weights, resident | 8.97 |
| KV cache pool | **19.79** |
| Reserved headroom | 2.48 |

**1,151,744 KV tokens**, against ~321,350 measured on v5e-1 — 3.6x. The arithmetic closes
independently: 19.79 GiB ÷ 18,432 B/token (`../MODELS.md`) = 1,153,434, within 0.15% of the measured
figure, and the same division reproduces the v5e number to 0.07%.

**This is why `max-model-len 65536` is not a leap here.** It is the configuration the recorded v6e
allocation came from. 32768 is set only because it is the value with a measured *throughput sweep*
behind it; 65536 has a measured *allocation* and no sweep. Raising it is one line in `tpu.env`.

**The capacity rule of thumb does not port — and on v6e there is no knee to port it to.**
**Measured 2026-08-10 and this falsifies the derivation that stood here.** This file previously scaled
v5e's "keep `clients × context` under ~250,000 tokens" (78% of its pool) to **~900,000 tokens** on v6e,
flagged as a derivation. It was tested directly, at fixed 16,000 context with only concurrency varying,
and there is **no knee anywhere in 56–90% of the pool**:

| concurrency | KV needed | % of pool | tok/s | median TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 645,120 | 56% | 452.7 | 2,088 ms |
| 46 | 741,888 | 64% | 465.9 | 3,659 ms |
| 52 | 838,656 | 73% | 470.6 | 5,273 ms |
| 56 | 903,168 | **78%** | 467.7 | 6,309 ms |
| 60 | 967,680 | 84% | 471.6 | 7,373 ms |
| 64 | 1,032,192 | 90% | 469.2 | 8,451 ms |

`TTFT = -8519 + 265 × concurrency`, **R² = 1.0000**, slope flat at 259–270 ms across every interval
including the predicted 78% point. Throughput varies 4.2% end to end. This is ordinary queueing — each
extra concurrent request adds a fixed increment — not a memory cliff.

**Why the earlier sweep looked like it had a 3.4x cliff, and why that was wrong.** Two artifacts
compounded. Nothing was sampled between 46% and 89% of pool, and the low anchor (`16000×32`, 2,461 ms)
was **~100% prefix-cache hits** — 1,539,904 hit tokens against 1,536,000 input tokens, 0.25% apart —
because it ran after `16000×64` with the same `--seed 0` and drew a subset of the same prompts. Two
points far apart with the lower one artificially fast reads as a cliff.

> **`--seed 0` plus `enable_prefix_caching=True` (the vLLM default) silently couples cells at the same
> `input_len`.** Any later cell re-uses the earlier cell's prompts. Vary the seed per cell, or accept
> that same-context cells are not independent. This is a benchmarking trap in `run_cells.py`, not a
> property of the hardware.

**And crossing the pool boundary is not an event either.** Four further cells at 101%, 112%, 134% and
157% of pool: throughput 458–474 tok/s (unchanged), TTFT still on the same line, and
**`num_preemptions_total` = 0 in every single cell**. The full fit over 56%→157% of pool is
`TTFT = -8542 + 265 × concurrency`, **R² = 1.00000** across 10 points — a line fitted entirely *below*
100% predicts 157% occupancy to within 0.13%.

The mechanism is that the scheduler **admission-controls rather than evicts**: it admits what fits and
queues the rest, so the working set never thrashes. Occupancy alone costs nothing; *eviction* would,
and it never engaged. That is why there is no threshold to find.

**What this means operationally: size by the latency you will accept, not by a pool fraction.** Pick a
concurrency from the line above — every added concurrent request buys **265 ms of TTFT and nothing
else**, and throughput saturates near concurrency 40 regardless. The pool bounds what can be
*resident*; it is not a performance threshold at any occupancy tested.

Full write-up and the remaining gaps in
[`benchmarks/runs/2026-08-10-config-validation-v6e1/REPORT.md`](benchmarks/runs/2026-08-10-config-validation-v6e1/REPORT.md).

## Low-precision rescan against the v6e profile

Re-run 2026-08-10 against `../HARDWARE.md` and `../QUANTIZATION.md`. **Conclusion: bf16 is still the
only thing that works, but one v5e blocker is plausibly lifted by this chip.**

### fp8 — dead on v6e, and now *confirmed* rather than predicted

`../HARDWARE.md`: v6e has **no native fp8**, same as v5e. So fp8 could only ever buy footprint and
bandwidth here, never FLOPS. But it does not even buy footprint:

- **KV.** v5e measured `--kv-cache-dtype fp8_e4m3` at a **1.000x** capacity ratio — identical 321,376
  tokens and 10,043 blocks — because the KV block layout is **word-aligned**: as the element width
  halves the shape goes `(32,1,2,256)` → `(32,1,4,256)` and the byte count never moves. All 8
  throughput cells lost 1.8–5.6%.
- **This reproduces on v6e.** The recorded v6e run logged `Automatically using fp8_e5m2 for FP8 KV
  cache` and allocated **bf16 anyway**: 1,151,744 tokens × 18,432 B = 19.77 GiB against the 19.79 GiB
  measured pool — **0.10% off**. Had fp8 engaged, the same pool would hold ~2.3M tokens; the fp8 model
  is **50% off**. The word-alignment mechanism is a tpu_inference property, and it just showed up on a
  second generation.
- **Weights.** compressed-tensors fp8 w8a8 exists on the JAX path but needs a pre-quantized checkpoint,
  which does not exist for Gemma 4; the qwix fp8 route hits the same two boot failures as int8.

### int8 — the only format with a compute win, still not bootable, but worth retrying here

int8 is the one low-precision format with a real 2x MXU win on v6e — **verified at the source
2026-08-10**: Google's v6e page publishes `bf16: 918 TFLOPs` and `Int8: 1836 TOPs`, exactly 2x, and
**no fp8 row at all**. Two separate things to keep apart:

- **int8 KV is not reachable and never was.** Plain `int8` is **rejected by vLLM's CLI enum** before
  tpu_inference sees it (`invalid choice: 'int8'`). `int8_per_token_head` *is* accepted and then
  **kills the server at boot** with `TypeError: data type not understood`. Passing CLI validation
  proves nothing on this path.
- **int8 weights via qwix is the live question, and the v6e HBM budget changes it.** On v5e-1 the
  concrete (default) path died at
  `RESOURCE_EXHAUSTED: HLO temporaries (16.23G) exceeds available HBM (15.75G)` — **short by 0.48 G**.
  **v6e-1 has 31.24 GiB, roughly 15 GiB of headroom over that same 16.23 G temporary peak.** The
  failure was an HBM ceiling, and this chip doubles it.

> **This is the one actionable item the rescan produced.** `../QUANTIZATION.md` records qwix as "does
> not boot" on the strength of two v5e failures. The *second* one — `use_abstract_model: true` raising
> `ValueError: no module or parameter named …down_proj.weight` — is a code defect and more HBM will not
> touch it. But the abstract path exists only for models whose bf16 weights do not fit, and **E2B fits
> at bf16 on both chips**, so on v6e the concrete path is the one that matters and its only recorded
> blocker is memory. Cheap to test: both arms failed in 2.5–4 min, well before the ~738 s compile.
>
> Invocation (mind the `str.format()` brace-escaping rule in `startup_script_template.sh`):
> ```
> --additional-config '{"quantization":{"qwix":{"rules":[{"module_path":".*","weight_qtype":"int8"}]}}}'
> ```
> **Verify from the allocation log, not the flag:** E2B's bf16 figure is `total_hbm_used_gb` **8.97
> GiB**. If that number does not drop, nothing downstream matters. If the line never prints, the engine
> died before it — also an answer.

### KV cache options, rescanned

**No dtype flag reduces 18 KiB/token, on either chip.** The accepted-vs-working gap is the trap:

| | |
| :--- | :--- |
| CLI accepts (this build) | 16 values |
| tpu_inference actually maps | **4** — `fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp4` |
| of those, passable | 3 — `fp4` is mapped but absent from vLLM's enum, so it fails validation first |
| everything else | falls through to `jnp.dtype(...)`; `int8_per_token_head`, `turboquant_*`, `nvfp4`, `fp8_inc`, `fp8_ds_mla` **kill the server at boot** |
| `--calculate-kv-scales` | **no-op** for Gemma 4 — honored only on the DeepSeek MLA path |

Keep `auto`. **`auto` never reaches `to_jax_dtype` at all**, so an explicit dtype — even `bfloat16`,
which resolves to the dtype the model already uses — takes a *different code path*, one that enters
`gemma4.py`'s write-side KV quantization with its hardcoded `_k_scale`/`_v_scale = 1.0`. Identical
bytes, extra risk, no gain.

**4-bit KV is predicted to be a third 1.000x** — extrapolating the layout pattern (2 B → dim 2, 1 B →
dim 4, both 32,768 B/block/layer), 4-bit lands at dim 8 and the same byte count. Prediction, not
measurement, but it is the default expectation after fp8.

**The real KV prize is allocation, not dtype, and it is 2.9x.** tpu_inference switches sliding windows
off for every Gemma 4 size — `disable_sliding_window = len(head_size_set) > 1`, and Gemma 4's
256/512 head-dim split trips it family-wide — so all 12 sliding layers get full-length allocation
though they never read past 512 tokens. At 32,768 context that is 576.0 MiB/seq allocated against
198.0 MiB/seq windowed, **2.91x forgone**. Gated on a named upstream `TODO`; re-check on every image
bump. It is worth more than every dtype in this section combined.

### The other caches

Distinct from KV, and neither is a quantization question:

- **Compile cache (on disk).** `/root/.cache/vllm`, ~197 MB, container-local and destroyed on
  `docker rm`. Mounting it is still the highest-value operational change here — see Operational below.
- **Prefix cache (in HBM, `--enable-prefix-caching`).** On, and it shares the KV pool — so this chip's
  3.6x larger pool is also 3.6x more room for cached prefixes. Measured 99.7% hit on a repeated
  4,813-token prefix, 7.4x faster prefill, on v5e.
- **CPU swap (`--swap-space`)** is untouched and unmeasured on this stack; nothing here has exercised it.

## The three things worth understanding

**1. `max-num-batched-tokens` is bounded on both sides, and 4096 is forced.** *(carries — model + stack)*
`--disable-chunked-mm-input` imposes a hard floor: one multimodal item must fit in a single batch, and
for this model at `{"image":4,"audio":1}` that is **2496 tokens** — below it the server refuses to
start. Token buckets are powers of two, so 2496 and 4096 compile the *identical* ladder and a
2496-token chunk runs in a 4096-shaped kernel: same cost, 61% of the work. Measured penalty for trying
it: **−24.7%** throughput. Every value in (2048, 4096] costs the same and 4096 does the most work.
Reaching the 2048 bucket requires dropping `--disable-chunked-mm-input`, which has never been tested.

**2. Longer context is free because block size scales with it.** *(carries — derivation logic)*
`block_size` is derived to hold blocks-per-request at 512: 16384 → 32, 32768 → 64, 65536 → 128. Decode
speed tracks blocks-per-request, not context length, which is why `--block-size` must stay underived.
The *pool* the blocks are cut from is chip-dependent; the derivation is not.

**3. The defaults are not where they appear to be.** *(carries, but re-verify the chip gate)*
`SchedulerConfig` declares `DEFAULT_MAX_NUM_SEQS = 128` and it is dead code on the serve path.
`EngineArgs.get_batch_defaults()` overrides from a usage-context dict gated on device memory — and on
v5e both gates failed: `get_device_total_memory()` raises `NotImplementedError` (swallowed by a bare
`except`, so memory reads 0), and the per-chip TPU table tests `chip_name == "V5E"` while
`get_device_name()` returns `'TPU V5E'`. The rig silently received generic defaults (2048 / 256).
**On v6e the memory gate should fail identically, and the string gate cannot match at all** — the table
has no v6e row to hit. Same outcome, one more reason. Worth confirming against the actual boot log
rather than assumed, since it is the only §-3 claim whose chip-independence is inferred.

## Operational

**Mount the compile cache.** `/root/.cache/vllm` (197 MB) is container-local and destroyed on
`docker rm`. On v5e, compile was 685 s of an 857 s cold boot and mounting it cut a restart to **497 s
(−42%)**, reproduced twice. The *mechanism* carries — the cache is a compilation artifact, not a chip
property — but **the timings are v5e's**; v6e recompiles its own kernels and the boot profile is
unmeasured. `startup_script_template.sh` still does not do this; adding it remains the
highest-value operational change available.

**Provisioning — the v5e ranking is reversed here, do not carry it.** On v5e, flex-start cost 3.8%
more than spot. In **us-east5 on v6e, flex-start lists at $1.35/chip-hr and spot at $1.4033 — spot is
dearer** (`CLAUDE.md`). So the interactive-workload argument for flex-start no longer trades against
price at all: it is both cheaper *and* preemption-free, and it self-terminates via
`--max-run-duration`, which spot and on-demand do not. Read the ranking out of
`estimate_deployment_cost` rather than assuming; it is a fact about one chip in one region, and a
published rate is not an offer of capacity.

## Known-unexploited, and untested

Ranked by expected value, after the 2026-08-10 rescan:

**1. Retry qwix int8 W8A8 — the v5e blocker was an HBM ceiling this chip doubles.** The highest-value
*reachable* experiment, and the only one the new hardware profile actually unblocks. Details and the
verification rule in the rescan section above. Fails fast (2.5–4 min) if it still doesn't boot.

**2. 2.9x KV capacity is being left on the table** — the sliding-window over-allocation, worth more
than every dtype flag combined, but gated on an upstream `TODO` and not settable from here. Re-check
on every image bump. See the rescan section and `../QUANTIZATION.md`.

**3. Mount the compile cache** — see Operational; mechanism carries, timings are v5e's.

**Never tested on this rig:** everything, strictly speaking — no sweep has run on v6e-1. The specific
gaps beyond re-confirming the v5e results are `max-num-batched-tokens 8192`,
`VLLM_TPU_BUCKET_PADDING_GAP=128` (which would create a ~2560 bucket and change the calculus in §1),
`ATTN_BUCKETIZED_NUM_REQS`, `SLICE_ROPE_CACHE`, `NUM_PRECOMPILE_WORKERS`, n-gram speculative decoding
(needs no draft checkpoint and is marked passing on TPU), `--swap-space`, and — newly relevant on
32 GB — where `gpu-memory-utilization` actually fails, since v5e's 0.95 ceiling was a 16 GB result.

**This rig's copy of the fp8 KV evidence is incomplete.** `benchmarks/runs/2026-08-07-kv-quant-v5e1/`
carries only the `bf16` arm (`cells_bf16.json`, `quality_bf16.json`); the fork left the `fp8` arm
behind. The complete set — `cells_fp8.json`, `cells_fp8.log`, `quality_fp8.json`, `cells_bf16.log` —
is in the v5e rig at the same path, and that is where the 1.8–5.6% figures above were read from.

**The first sweep here should re-run the v5e matrix, not assume it.** The two chips differ by 3.6x in
KV pool and ~1.9x in bandwidth, and v6e is *not* the way to buy decode throughput — it buys context.
Several v5e conclusions were themselves reversals of source-reading, which is the reason this file
tags its provenance line by line.

**Caveat on the checkpoint:** *(carries)* tpu-inference's own support table marks `gemma-4-E2B-it`
✅ unit / ❌ correctness / ❓ performance, while the 26B and 31B pass all three. Local quality probes on
v5e were clean (8/9 byte-identical, 3/3 needles), but the upstream flag is not ours to dismiss.
