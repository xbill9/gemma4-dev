# 2026-08-25 — first measurement on this rig: TP=4 allocation, and a Zimbres 2026 partial replication

**This rig had provisioned nothing and measured nothing before this run.** Every capacity figure in
`tpu.env`, `SERVING-PARAMS.md` and `CLAUDE.md` was arithmetic. This is the run that changes that, and it
changes it for **TP=4 only**.

## What ran

| | |
| :--- | :--- |
| checkpoint | `google/gemma-4-31B-it`, bf16 |
| hardware | **`ct6e-standard-4t`** — 4 v6e chips, NOT the 8 this rig is named for |
| zone / model | `europe-west4-a`, FLEX_START, granted immediately |
| TP | **4** |
| serving | `vllm/vllm-tpu:nightly` (floating — see caveats), `--max-model-len 32768` |
| duration / cost | ~57 min, ~$5.13 at $1.35/chip-hr |

**The machine type is not this rig's.** `ct6e-standard-8t` stocked out in five zones (see below), so this is
a 31B-on-**v6e-4** measurement. Do not let `rollup.py` or a reader take it for a v6e-8 number.

## Result 1 — the KV allocation, and Equation 1 holds at TP=4

```
total_hbm_limit_gb=124.97GiB | total_hbm_limit_cap_gb=114.97GiB
total_hbm_used_gb=58.4GiB    | total_hbm_avail_gb=56.57GiB
GPU KV cache size: 67,392 tokens | block_size=64 | 1053 blocks x 60 layers
regular_attn_dtype=bfloat16
```

```
56.57 GiB / 67,392 tokens = 880.2 KiB/token
```

`@../MODELS.md`'s ideal decomposition — `50 x 16 x 256 + 10 x 4 x 512`, x2 planes x2 B — is **880
KiB/token**. Measured 880.2. **At TP=4 there is no padding**: the minimum KV head count (4) equals TP, every
layer shards fully, exactly as Zimbres 2026 Eq. 1 predicts. A padded layout would have given 960 KiB/token
and ~62,500 tokens; it gave 67,392.

Two of the paper's incidentals also confirmed: the memory fraction is **0.920** (114.97/124.97) and usable
HBM is **31.24 GiB/chip**, both as `@../HARDWARE.md` states.

**Weights are 58.4 GiB resident** (`Init model | hbm=[(14.6, 31.24) x4]`). That sits with Zimbres' 58.25 GiB
and the staged tar's 58.28 GiB, and NOT with `@../MODELS.md`'s 57.7 GiB — which is parameter bytes
(31.0B x 2), a different quantity. Both figures are right about different things; do not reconcile them.

### What this does and does not say about TP=8

It measures the **unpadded side** of the crossing. Zimbres measured the padded side: their 187,136 tokens,
recomputed with the 58.4 GiB weights measured here, give **961.1 KiB/token**. So both sides now agree with
the mechanism — but the 960 KiB/token figure in this rig's `tpu.env` **remains arithmetic**, because
nothing here has run at TP=8.

## Result 2 — the fp8 banner lies, reproduced on a newer stack

```
INFO tpu_platform.py:211  Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e.   (x4)
...
INFO kv_cache_manager.py:986  regular_attn_dtype=bfloat16
```

Zimbres §6.3 documented exactly this on `vllm-tpu 0.26.0`: a startup banner announcing an fp8 KV cache over
an allocation that delivered bf16, and the banner naming **e5m2** specifically. **It reproduces here on
`:nightly`**, months later and several versions on. The automatic fp8 path is still inert, and the
allocation line is still the only thing that says so. This is independent confirmation of the paper's most
repeated methodological claim, and it is why `verify_model_health` and a flag echo are not evidence.

## Result 3 — throughput is 2.34x off the paper, and the log says why

`vllm bench serve`, batch 256, 32-token prompts, `--random-output-len 128 --ignore-eos`, greedy:

| | Zimbres TP=4 (heuristic) | this run |
| :--- | ---: | ---: |
| output tok/s | 4,752 | **2,031** |
| ITL | 51.7 ms | **120.0 ms** |

Throughput ratio 2.34x, ITL ratio 2.32x — consistent, so it is a per-decode-step cost, not scheduling.
The cause is a backend cascade this stack falls into and the paper's did not:

```
config.py:242  Gemma4 model has heterogeneous head dimensions
               {'sliding_attention': 256, 'full_attention': 512}.
               FA4 not available, forcing TRITON_ATTN backend.
importing.py:53  Triton is installed but 0 active driver(s) found. Disabling Triton...
vllm.py:680      Model Runner V2 requires Triton; using the V1 model runner instead.
```

vLLM forces `TRITON_ATTN` because of Gemma 4's heterogeneous head dims, then disables Triton because there
is no driver on a TPU, then drops to the V1 model runner. **The ragged paged attention Pallas kernel that
the whole Zimbres series tunes is not in play at all here.**

That has a sharp consequence: **their kernel results are unreachable on this stack, not merely
unimplemented.** There is no decode block size to override, so the 69%/54% bkv-128 gains cannot be
reproduced or refuted from this rig as it stands. It also means their `heuristic (bkv 2048)` row is NOT the
right baseline for our numbers — that row is still Pallas, just badly blocked.

## Result 4 — the GCS checkpoint staging works

```
Restoring Hugging Face cache from gs://...-hf-cache-europe-west4/... into /dev/shm
Cache restored in 337s. Serving offline; no Hugging Face download needed.
```

**5.6 minutes in-region**, against the 60-90 minute Hugging Face pull the boot budget was sized for. First
exercise of the restore leg; it worked on the first real boot. Engine init after that was 427.9 s
(compilation 399.6 s), which is now the dominant term in time-to-serve.

## Capacity: the finding nobody asked for

Five provisioning attempts, and the pattern is itself a result:

| ask | zone | model | outcome |
| :--- | :--- | :--- | :--- |
| 8 chips | europe-west4-a | flex-start | queued 26 min, expired: **not enough resources** |
| 8 chips | us-east1-b, us-east5-a, us-east5-b, asia-northeast1-b | spot | **ZONE_RESOURCE_POOL_EXHAUSTED** |
| 8 chips | us-central1-b | spot | granted, **preempted at ~6 min**, never served |
| **4 chips** | **europe-west4-a** | **flex-start** | **granted instantly** |

Eight chips stocked out in europe-west4-a and four were granted in the same zone under an hour later.
**For CAPACITY, the size of the ask is the variable.** Note this does not contradict the rule in
`CLAUDE.md` that grants track the region and not the number — that rule is about **quota requests**, which
are a different mechanism. Both are true and they are about different things.

Also confirmed the live price gap: europe-west4 flex-start **$1.3500**/chip-hr vs spot **$1.7820** — spot is
32% dearer, and was also the model that got preempted.

## What was NOT captured, and why

**No xprof / TensorBoard trace.** `VLLM_TORCH_PROFILER_DIR` **does not exist in this nightly** — the only
profiler variables it knows are `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`,
`VLLM_CUSTOM_SCOPES_FOR_PROFILING`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`,
`VLLM_NVTX_SCOPES_FOR_PROFILING`, `VLLM_TRACE_FUNCTION`. Setting it produces
`WARNING: Unknown vLLM environment variable detected`, and `/start_profile` returns **404** with no profile
route in the OpenAPI document. `tpu_inference` is importable, so `jax.profiler` could trace from inside the
engine process, but that needs code injection into a running container, not an env var.

**Therefore the paper's centerpiece — the per-layer 1.92x vs 1.26x split and the +58.1% -> +4.3% premium
collapse — is untouched by this run.** It needs both TP arms and a working profiler, and this run has one
arm and no profiler.

## Files

- `vllm-startup.log` — boot, including the GCS restore line
- `vllm-container.log` — engine init, allocation line, fp8 banner, backend cascade
