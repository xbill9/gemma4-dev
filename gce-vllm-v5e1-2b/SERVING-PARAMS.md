# Serving parameters for this rig — the decision, and why

> **Inherited from `tpu-vllm-v5e1-2b`, and correctly so.** Everything below is a property of the *chip, the
> checkpoint and the serving stack* — none of it is a property of the control plane, so it carries to this rig
> unchanged. What does **not** carry is the evidence: nothing here has been measured on a Compute Engine
> instance, because nothing here has been provisioned. The run directory this file cites lives in the twin
> (`../tpu-vllm-v5e1-2b/benchmarks/runs/`), not here — this rig's `benchmarks/` is empty on purpose. See
> `CLAUDE.md`.
>
> One caveat with teeth: the config below sets `--max-model-len 32768` while this rig's `tpu.env` ships
> **16384**, which is what the twin actually serves. That gap predates the fork and is unresolved in the twin
> too. Do not close it here unilaterally — the two rigs have to stay in step or the A/B comparison stops being
> one.

**Status: measured 2026-08-09** on the TPU-API twin. The evidence lives in that rig's
`benchmarks/runs/2026-08-09-serving-params-v5e1/REPORT.md` — nine arms on a spot `v5litepod-1`, 36 benchmark
cell-runs, and the functional tests. Read that for numbers; read this for the decision.

> **This file was rewritten on 2026-08-09.** Its first version was derived from the archive and source
> reading, and **three of its recommendations were wrong** — measurement reversed them. The falsified
> versions are recorded in the report rather than deleted, because the reversals are the useful part.

## The configuration

```
vllm serve google/gemma-4-E2B-it \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --disable-chunked-mm-input \
  --limit-mm-per-prompt '{"image":4,"audio":1}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

plus `-v ~/.cache/vllm:/root/.cache/vllm` on the `docker run`.

**Exactly one change from what the rig ran before: `MAX_MODEL_LEN` 16384 → 32768.** Everything else
either restates a resolved default or is unchanged. To adopt it, set `MAX_MODEL_LEN=32768` in
`tpu.env`; `server.py:112` supplies the old default and both deploy paths read the same value through
`_vllm_serve_flags()`.

| flag | value | why |
| :--- | :--- | :--- |
| `--max-model-len` | **32768** | measured equal-or-better than 16384 on every cell (+4.4% at 1K/16), costs **no** KV capacity, and boots *faster* (857 s vs 1,089 s) |
| `--max-num-batched-tokens` | **4096** | provably optimal in (2048, 4096] — see below |
| `--max-num-seqs` | **unset (256)** | capping it to 64 was measured **worse**; the cap never binds below 64 offered load |
| `--kv-cache-dtype` | **auto** | fp8 measured 1.000x capacity and ~2% slower; `auto` is fully determined once `--dtype` is pinned |
| `--dtype` | **bfloat16** | no weight-quantization route boots (`@QUANTIZATION.md`) |
| `--gpu-memory-utilization` | **0.92** | 0.95 fails after a 691 s compile; measured twice |
| `--tensor-parallel-size` | **1** | one chip; E2B's single KV head cannot shard and would be replicated |
| `--enable-prefix-caching` | **on** | measured 99.7% hit on a repeated 4,813-token prefix, 7.4x faster prefill |
| `--block-size` | **never set** | the backend derives it from `max_model_len`; that derivation is what keeps long context free |

Three of these restate the resolved default. **vLLM confirms they are no-ops** — `gpu_memory_utilization`
and `kv_cache_dtype` do not appear in the engine's `non-default args` at these values. They are written
down anyway, because the real defaults are computed several layers from where they look declared.

## The three things worth understanding

**1. `max-num-batched-tokens` is bounded on both sides, and 4096 is forced.**
`--disable-chunked-mm-input` imposes a hard floor: one multimodal item must fit in a single batch, and
for this model at `{"image":4,"audio":1}` that is **2496 tokens** — below it the server refuses to
start. Token buckets are powers of two, so 2496 and 4096 compile the *identical* ladder and a
2496-token chunk runs in a 4096-shaped kernel: same cost, 61% of the work. Measured penalty for trying
it: **−24.7%** throughput. Every value in (2048, 4096] costs the same and 4096 does the most work.
Reaching the 2048 bucket requires dropping `--disable-chunked-mm-input`, which has never been tested
here.

**2. Longer context is free because block size scales with it.**
`block_size` is derived to hold blocks-per-request at 512: 16384 → 32, 32768 → 64, both landing on
~321,350 KV tokens. Decode speed tracks blocks-per-request, not context length. This is why
`--block-size` must stay underived.

**3. The defaults are not where they appear to be.**
`SchedulerConfig` declares `DEFAULT_MAX_NUM_SEQS = 128` and it is dead code on the serve path.
`EngineArgs.get_batch_defaults()` overrides from a usage-context dict gated on device memory — and both
gates fail here: `get_device_total_memory()` raises `NotImplementedError` (swallowed by a bare
`except`, so memory reads 0), and the per-chip TPU table tests `chip_name == "V5E"` while
`get_device_name()` returns `'TPU V5E'`. **vLLM's v5e-specific tuning is dead code on this part**, so
the rig silently receives generic defaults (2048 / 256).

## Operational

**Mount the compile cache.** `/root/.cache/vllm` (197 MB) is container-local and destroyed on
`docker rm`. Compile is 685 s of an 857 s cold boot; mounting it cuts a restart to **497 s (−42%)**,
reproduced twice. `startup_script_template.sh` does not do this — adding it is the single highest-value
operational change available.

**Provisioning.** Flex-start costs 3.8% more than spot ($0.6000 vs $0.5779/chip-hr, us-west4) and is
worth it for anything interactive: no mid-run preemption, and it self-terminates via
`--max-run-duration`, which spot and on-demand do not. Spot wins on pure cost only while preemptions
are less frequent than every 6.2 h (no cache mount) or 3.6 h (with it).

**Capacity rule of thumb: keep `clients × context` under ~250,000 tokens**, about 78% of the pool.
Above it, tail latency degrades faster than throughput improves — at 8192×64 (524K wanted) median TTFT
is 11.7 s.

## Known-unexploited, and untested

**2.8x KV capacity is being left on the table.** All 15 cached layers get full-length allocation
(`num_kv_cache_groups=1`, measured on every arm) though 12 are sliding-attention layers windowed at
512 tokens. Unreachable: tpu-inference sets `disable_sliding_window = len(head_size_set) > 1` and
Gemma 4 has two head dims, with an upstream `TODO`. Re-check on image bumps — it is worth more than
every flag in this file combined. See `@QUANTIZATION.md`.

**Never tested here:** `max-model-len 65536` (block_size would go to 128 at the same 512
blocks/request, so it may also be free), `max-num-batched-tokens 8192`, `VLLM_TPU_BUCKET_PADDING_GAP=128`
(which would create a ~2560 bucket and change the calculus in §1), `ATTN_BUCKETIZED_NUM_REQS`,
`SLICE_ROPE_CACHE`, `NUM_PRECOMPILE_WORKERS`, n-gram speculative decoding (needs no draft checkpoint and
is marked passing on TPU), and `gpu-memory-utilization` 0.93/0.94.

**Caveat on the checkpoint:** tpu-inference's own support table marks `gemma-4-E2B-it` ✅ unit /
❌ correctness / ❓ performance, while the 26B and 31B pass all three. Local quality probes were clean
(8/9 byte-identical, 3/3 needles), but the upstream flag is not ours to dismiss.
