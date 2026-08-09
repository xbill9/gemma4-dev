---
title: "Self-hosting a lite agent backend on one TPU: Gemma 4 E2B + vLLM on a v5e-1"
published: false
tags: tpu, vllm, llm, gcp
---

# Self-hosting a lite agent backend on one TPU chip

A single Google Cloud TPU v5e chip — 16 GB of HBM, about $0.58/hour on spot — will serve `google/gemma-4-E2B-it` under vLLM at **1,496 output tokens/sec** aggregate, with **8.02 ms** per-token latency at single stream and native tool-calling. That is enough to back a fleet of 8–16 concurrent "lite" agents for roughly **$0.107 per million output tokens**.

This is a build log with numbers. Everything here was measured on the hardware, and the sections that say "I was wrong about this" are the ones worth your time — four of my confident predictions were falsified by the benchmark, and each falsification was more useful than the guess.

**Setup under test:** `v5litepod-1` (one v5e chip), `us-west4-a`, `vllm/vllm-tpu:nightly`, vLLM `0.26.1rc1.dev125+ga7a204cc6`, tpu-inference JAX backend, `google/gemma-4-E2B-it` at bf16.

---

## Part 1 — Scaffold and run

### 1.1 Prerequisites

```bash
gcloud auth login                     # for gcloud subprocess calls
gcloud auth application-default login # ADC, for the Secret Manager client
```

Put your Hugging Face token in Secret Manager rather than in a script or an env file — the TPU VM's startup script is stored as instance metadata, and anything you bake in is readable from the instance:

```bash
printf '%s' "hf_xxxxxxxxxxxx" | gcloud secrets create hf-token --data-file=- --project=YOUR_PROJECT
```

**Zone constraint that will waste your afternoon if you miss it:** flex-start `v5litepod-1` is only accepted in `us-west4-a`. `europe-west4-a` and `-b` reject it at the API with `FLEX_START provisioning model is not supported for accelerator type "v5litepod-1"`, *regardless of quota*. Non-zero quota in a zone tells you nothing — the provisioning model is the blocker.

### 1.2 Provision the chip

Three provisioning models, three different commands. Note `v5e` is spelled **`v5litepod`** to gcloud — "v5e-1" is fine in prose and is never valid in a CLI argument.

```bash
# Spot — cheapest, preempted with ~30s notice, NO run limit (bills until you delete it)
gcloud alpha compute tpus tpu-vm create gemma4-v5e \
  --zone=us-west4-a --type=v5litepod --topology=1x1 \
  --provisioning-model=spot --version=v2-alpha-tpuv5-lite

# On-demand — full price, no preemption, also unbounded
gcloud alpha compute tpus tpu-vm create gemma4-v5e \
  --zone=us-west4-a --type=v5litepod --topology=1x1 \
  --version=v2-alpha-tpuv5-lite
```

Flex-start goes through the Queued Resource API instead, and is the only model that accepts `--max-run-duration`, i.e. the only one that stops billing on its own:

```bash
gcloud alpha compute tpus queued-resources create gemma4-qr \
  --node-id=gemma4-qr-node --zone=us-west4-a \
  --accelerator-type=v5litepod-1 --runtime-version=v2-alpha-tpuv5-lite \
  --provisioning-model=flex-start --max-run-duration=4h
```

> **Verified 2026-08-09:** this exact command was run — the QR reached `ACTIVE` with `provisioningModel: FLEX_START` and `maxRunDuration: 14400s`, then deleted cleanly. Note there is no dry-run: a create either queues or provisions, and `PROVISIONING` state cannot be deleted, so you will pay for at least a few minutes if capacity is immediately available.

> **Spot and on-demand have no automatic stop.** They bill until preempted or deleted. Set a calendar reminder, or use flex-start. See the cost section — flex-start is only 3.8% more than spot.

Spot draws on a **separate quota** (`TPUV5sPreemptibleLitepodPerProjectPerZoneForTPUAPI`), not the standard TPU quota. A zone with plenty of on-demand quota can still refuse spot.

### 1.3 Get the token onto the node and start the server

`gcloud compute tpus tpu-vm ssh` crashes with `ConnectionResetError` from some sandboxed environments (it fails inside its own internal API call, while plain gcloud API calls work fine). Direct SSH always works:

```bash
IP=$(gcloud compute tpus tpu-vm describe gemma4-v5e --zone=us-west4-a \
      --format='value(networkEndpoints[0].accessConfig.externalIp)')

# Pipe the secret straight in — never through a shell variable or a log line
gcloud secrets versions access latest --secret=hf-token \
  | ssh -i ~/.ssh/google_compute_engine xbill@$IP 'umask 077; cat > ~/.hf_token'

ssh -i ~/.ssh/google_compute_engine xbill@$IP 'sudo docker pull vllm/vllm-tpu:nightly'
```

Then start it. This is the configuration the rest of the article defends:

```bash
sudo docker run -d --name vllm-gemma4 --privileged --net=host \
  -v /dev/shm:/dev/shm --shm-size 10gb \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -e HF_HOME=/dev/shm -e HF_TOKEN="$(cat ~/.hf_token)" \
  vllm/vllm-tpu:nightly \
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

**That `-v ~/.cache/vllm:/root/.cache/vllm` line is the highest-value thing in this article.** The JAX compile cache lives there (197 MB measured) and is otherwise *container-local*, destroyed on every `docker rm`. Compilation is **685 s of the 857 s** cold start. Measured: mounting it cuts a restart to **497 s, a 42% saving**. Without the mount, every restart, flag change and spot preemption repays the full compile.

### 1.4 Verify

Cold start is **857 s** (14 min) and **80% of it is XLA compilation, not weight loading** — the 9.54 GiB checkpoint downloads in about 10 seconds. Be patient, and watch the log rather than the clock:

```bash
sudo docker logs -f vllm-gemma4 2>&1 | grep -E "Memory statistics|Init kv-cache|startup complete"
```

You want to see this, which is the whole memory budget in one line:

```
Memory statistics | total_hbm_limit_gb=15.75GiB | total_hbm_limit_cap_gb=14.49GiB
                  | total_hbm_used_gb=8.97GiB   | total_hbm_avail_gb=5.52GiB
```

Then smoke-test it. **Use `/v1/chat/completions`, not `/v1/completions`** — raw completions return an empty string on `-it` models, which looks exactly like a broken deploy and isn't:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"google/gemma-4-E2B-it",
  "messages":[{"role":"user","content":"Say hi in five words."}]}' | jq -r '.choices[0].message.content'
```

### 1.5 Tear down

```bash
gcloud compute tpus tpu-vm delete gemma4-v5e --zone=us-west4-a --quiet
```

---

## Part 2 — The flags, and what they actually do

### The mental model: one HBM budget, and a part of it nobody governs

Everything else follows from this:

```
 15.75 GiB   total HBM on the chip
×  0.92      --gpu-memory-utilization
─────────
 14.49 GiB   the cap the engine will allocate inside
−  8.97 GiB  model weights (bf16)
─────────
  5.52 GiB   KV cache  →  321,376 tokens at 18 KiB/token

  1.26 GiB   what is left OUTSIDE the cap — compiled XLA programs live here,
             and gpu_memory_utilization does not govern them
```

That last line is the trap. I tested `--gpu-memory-utilization 0.95` expecting a free +8% of KV. The KV pool sized *exactly* as arithmetic predicted — cap 14.96 GiB, KV 5.99 GiB, **348,864 tokens, +8.6%** — and then XLA died **691 s in**, loading `jit_structured_decode_fn`:

```
RuntimeProgramAllocationFailure: Attempting to reserve 384.11M at the bottom of memory.
That was not possible. There are 347.33M free, 0B reserved, and 347.33M reservable.
```

The knob governs weights + KV only; compiled program images come out of the remainder it leaves behind. I have now measured this failure twice, months apart, at different page sizes (32 and 64) — 384.11 M wanted against 346.77 M and 347.33 M free. It is deterministic, not a race.

**0.92 is a ceiling, not a conservative default.** And note the shape of that failure: it costs a full compile to discover — 691 s before it tells you.

### `--gpu-memory-utilization 0.92`

Fraction of total HBM the engine may allocate for **weights + KV cache**. Not activations, not compiled programs. 0.95 does not boot on this model/chip/build. 0.93 and 0.94 are untested; the estimated margins are ~291 MB and ~127 MB against a failure that was 37 MB short, so the risk/reward is poor.

There is also `--kv-cache-memory` (config field `kv_cache_memory_bytes`), which pins the pool in **bytes** instead of as a fraction, and skips memory profiling on later boots. Measured-good value on this setup: `5923602432`.

### `--max-model-len 32768`

The maximum context. The surprising part is that **it costs no KV capacity**:

| `max-model-len` | `block_size` | blocks/request | KV blocks | **KV tokens** |
|---:|---:|---:|---:|---:|
| 16,384 | 32 | 512 | 10,043 | **321,376** |
| 32,768 | 64 | 512 | 5,021 | **321,344** |

The Pallas backend scales `block_size` *with* `max_model_len` to hold blocks-per-request constant at
512. Doubling the context doubles the page size, halves the block count, and lands on the same token
capacity. Measured, both arms.

And blocks-per-request turns out to predict decode speed:

| blocks/request | c=1 TPOT |
|---:|---:|
| 512 (`max-model-len` 16384) | 8.05 ms |
| 512 (`max-model-len` 32768) | **8.02 ms** |
| 2048 (a misconfigured arm) | 8.32 ms |

**Corollary: do not set `--block-size`.** The backend picks it, and pinning 32 would fight the scaling that keeps long context free.

### `--max-num-batched-tokens 4096`

The token budget per scheduler step — the chunk size for chunked prefill. Smaller favours inter-token latency (a big prefill chunk stalls every in-flight decode); larger favours time-to-first- token. This is the single most misunderstood flag on TPU, for two reasons.

**Reason 1: the buckets are powers of two.** TPU needs a compiled graph per tensor shape, so vLLM precompiles a ladder — `16, 32, 64, … 4096` — and rounds your value **up** to the next bucket. Setting 2496 compiles *exactly the same ladder* as 4096. Each chunk then costs a 4096-shaped kernel while carrying 2496 tokens of work. I tried it: **−24.7% throughput** at 8k/64 with the ITL tail completely unchanged (173.9 → 174.2 ms).

**Reason 2: multimodal sets a hard floor.** With `--disable-chunked-mm-input`, one multimodal item must fit in a single batch. Go under it and the server refuses to start:

```
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
is larger than max_num_batched_tokens (2048). Please increase max_num_batched_tokens.
```

So for this model with `{"image":4,"audio":1}`: **floor 2496, next bucket 4096**. Every value in (2048, 4096] compiles identically, so the largest one wins. **4096 is optimal in that interval** — not by taste, by construction.

### `--max-num-seqs` (leave it alone)

Maximum concurrent sequences. The default is **256**, and where that comes from is worth knowing, because it is not where you'd look:

- `SchedulerConfig.DEFAULT_MAX_NUM_SEQS = 128` is **dead code on the serve path.**
- `EngineArgs.get_batch_defaults()` overrides it from a dict keyed by usage context and gated on device memory: `>= 70 GiB` → 1024, otherwise → **256**.
- vLLM *ships* per-chip TPU tuning (V6E 1024 / **V5E 512** / V5P 256 for `max_num_batched_tokens`)… and it never fires, because `get_device_name()` returns `'TPU V5E'` while the code tests `chip_name == "V5E"`. It also calls `get_device_total_memory()`, which raises `NotImplementedError` on tpu-inference and is swallowed by a bare `except`, so the memory gate reads 0.

So on a v5e you silently get **generic non-GPU defaults**. I tried capping it to 64 on the theory that 256 over-admits (256 × 16384 = 4.19M KV tokens against 321,376 resident). **It made things worse**, and in any case my benchmark's peak offered load was 64, so the cap never bound. Leave it at 256 unless you are actually seeing preemption in the logs.

### `--kv-cache-dtype auto`

Leave this alone, and be suspicious of anyone who tells you otherwise. `--kv-cache-dtype fp8_e4m3` gives a **1.000x** capacity ratio on this stack. The KV block layout is word-aligned:

```
bf16: regular_attn_shape=(num_blocks, (32, 1, 2, 256))  →  32,768 bytes/block/layer
fp8:  regular_attn_shape=(num_blocks, (32, 1, 4, 256))  →  32,768 bytes/block/layer
```

The third dimension doubles exactly as the element width halves. Narrowing the element buys padding, not capacity — and costs ~2% throughput. The flag is accepted at the CLI, echoed in `non-default args`, praised by a log line, reported in `/metrics`, and allocates a genuinely `float8_e4m3fn` tensor. **Five independent signals that it worked, and it did nothing.**

I expected `--kv-cache-dtype int8` to be worse still — reachable, and silently rounding the cache to whole integers because the model hardcodes its K/V scales to `1.0`. **I tested it and that is wrong on this build:** `int8` is not in vLLM's CLI enum and is rejected before it reaches any of that (`invalid choice: 'int8'`). The 16 accepted values are `auto, bfloat16, float16, fp8, fp8_ds_mla, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_per_token_head, int4_per_token_head, int8_per_token_head, nvfp4` and four `turboquant_*`. The ones that *are* accepted but unsupported (`int8_per_token_head`, `turboquant_*`, `nvfp4`, `fp8_inc`, `fp8_ds_mla`) fail loudly at boot rather than quietly — so on this build the honest summary is: **fp8 is the trap, because it is the one that appears to work.**

`auto` is not vague here — it means "inherit the model dtype", which `--dtype bfloat16` pins.

### `--tensor-parallel-size 1`

One chip, so this is forced. But it's worth knowing *why* more chips wouldn't help this model: E2B has `num_key_value_heads=1` — full MQA. A single KV head **cannot be sharded**; runtimes pad it up to a multiple of the TP size, so at TP=4 you pay 4× the KV memory to store the same head replicated. For this model, more chips multiply KV cost rather than dividing it.

### `--disable-chunked-mm-input` and `--limit-mm-per-prompt`

These are the multimodal contract, and **together they set the 2496-token floor** discussed above. If you don't need images and audio, dropping them lets `max-num-batched-tokens` reach the 2048 bucket, which is the only route to a lower latency floor on this stack. That is a real trade I'd evaluate before copying this config into a text-only deployment.

### `--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4`

What makes this an *agent* backend rather than a text box: OpenAI-compatible tool calling, parsed natively. Worth knowing that these pull in structured-decoding machinery — the program that ran out of memory in the 0.95 experiment was `jit_structured_decode_fn` — so they are not free against that ungoverned 1.26 GiB remainder.

### Environment variables worth knowing

Set with `-e` on `docker run`; these come from tpu-inference, not vLLM:

| var | default | what it does |
|---|---|---|
| `ATTN_BUCKETIZED_NUM_REQS` | `False` | precompile attention at power-of-two *request* buckets instead of a single shape at `max_num_seqs` |
| `SLICE_ROPE_CACHE` | `False` | slice the rotary cache to `max_model_len` at load — free HBM |
| `NUM_PRECOMPILE_WORKERS` | `1` | parallel XLA precompilation; compile is 685 s of an 857 s boot |
| `VLLM_TPU_BUCKET_PADDING_GAP` | `0` | linear bucket increments (use 128) instead of powers of two — the fix for the 2496-pads-to-4096 problem |
| `VLLM_XLA_CHECK_RECOMPILATION` | `False` | error on a runtime recompile; turn on for one validation boot |

---

## Part 3 — Results: how many clients, and for what

### Throughput and latency, measured

> **Provenance.** All 12 cells were measured on the configuration above, **3 repetitions each** (36 runs). Throughput is stable — coefficient of variation ≤3.4%, most ≤1%. The one exception is marked: 128-ctx/16-client **TTFT** has cv 56% across reps, so no point value should be trusted there. An earlier config was re-run as a control and reproduced a previous sweep to the digit (TPOT 8.05/8.08/8.43 ms, KV 5.52 GiB, 10,043 blocks), so the rig itself is stable.

Aggregate output tokens/sec:

| context ↓ / clients → | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| 128 | 123.7 | 433.9 | 1,152.5 | **1,496.5** |
| 1,024 | 120.4 | 415.4 | 991.0 | 1,258.7 |
| 8,192 | 94.2 | 254.4 | 399.6 | 324.3 |

Median time-to-first-token (ms):

| context ↓ / clients → | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| 128 | 15.9 | 31.8 | 98.8* | 247.3 |
| 1,024 | 40.3 | 51.5 | 170.1 | 421.7 |
| 8,192 | 289.4 | 304.2 | 594.4 | **11,734** |

Per-stream tokens/sec (what one user feels):

| context ↓ / clients → | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| 128 | 125 | 112 | 76 | 25 |
| 1,024 | 124 | 109 | 69 | 21 |
| 8,192 | 119 | 71 | 27 | 10 |

### The three regimes

**Short context (≤1K) scales cleanly to 64 clients.** 12.1× the single-stream throughput. This is the regime most agent traffic lives in.

**Long context (8K) peaks at 16 clients and then goes backwards.** 400 → 324 tok/s from 16 to 64, with median TTFT blowing out to 11.7 seconds. The reason is arithmetic: 64 streams × 8,192 tokens = 524,288 KV tokens wanted against **321,376 resident**. Past the KV wall, more clients buy queueing, not throughput.

**Single stream is bandwidth-limited at about half of peak.** Decode moves ~3.15 GiB per step (the model's 4.38 GiB of per-layer embedding tables are *gathered*, not streamed), which at the v5e's 800 GiBps is a 3.94 ms floor against 8.02 ms measured — **49% of peak**. *(That 3.15 GiB is derived from the model's layer geometry, not an instrument reading; the 8.02 ms is measured.)*. There is roughly 2× of headroom sitting in fixed per-step cost, not in memory bandwidth.

### Recommended client counts

| workload | context | clients | expected |
|---|---:|---:|---|
| Interactive chat / agent turns | ≤1K | **16** | 991–1,152 tok/s, 99–170 ms TTFT, ~14 ms TPOT |
| Max throughput, batch/offline | ≤1K | **64** | 1,259–1,496 tok/s, 247–422 ms TTFT |
| RAG / long documents | 8K | **16** | 399.6 tok/s, 594 ms TTFT |
| Long-context interactive | 8K | **≤4** | 254 tok/s, ~304 ms TTFT |

**Rule of thumb: keep `clients × context < 250,000 tokens`** — about 78% of the 321,344-token KV pool. Cross it and you're paying in tail latency for throughput you don't get.

### Why this fits "lite agents" specifically

Agent workloads have a shape that suits this chip well:

- **Short-to-medium contexts.** A tool-calling turn is a system prompt, a few messages and a schema — usually well under 4K. That is exactly where this config is strongest.
- **Bursty, not saturating.** Agents think, call a tool, wait, resume. The 16-client interactive point (63 tok/s per stream, 230 ms TTFT) comfortably backs far more than 16 *registered* agents.
- **Tool calling is native.** `--tool-call-parser gemma4` gives OpenAI-compatible function calling with no glue.
- **Prefix caching is on by default and it works** — measured: re-sending a 4,813-token prefix scored **4,800 cache hits (99.7%)** and cut prefill latency from 0.244 s to 0.033 s (**7.4x**), while a different prefix correctly produced zero new hits. Agents re-send near-identical system prompts constantly, so this is close to free throughput.
- **Multimodal for free** — 4 images + 1 audio clip per prompt, within the same 16 GB.

What it is *not* good for: long-context batch summarization at high concurrency (the 8K/64 cell is the worst on the chart), or anything needing >321K tokens of live KV.

---

## Part 4 — Cost: spot vs flex-start vs reserved

Live rates from the Cloud Billing Catalog for `us-west4`, per chip-hour:

| model | $/chip-hr | vs spot | stops itself? | preemptible? |
|---|---:|---:|---|---|
| 3-year commitment | 0.5400 | −6.6% | n/a | no |
| **Spot** | **0.5779** | — | **no** | yes, ~30 s notice |
| **Flex-start** (DWS) | **0.6000** | +3.8% | **yes** (`--max-run-duration`) | no, once running |
| 1-year commit / Reserved | 0.8400 | +45% | n/a | no |
| **On-demand** | **1.2000** | **+108%** | no | no |

### The result that surprised me: flex-start is the default choice, not spot

Spot is only **3.8% cheaper** than flex-start, and that discount is fragile. A preemption costs a full cold start — 857 s, of which 685 s is recompilation — which is **$0.138** of wasted spend at the spot rate. Flex-start's premium is **$0.0221/hour**. So:

> **Spot beats flex-start only while preemptions are less frequent than every 6.2 h without the cache mount, or every 3.6 h with it.** Measured rebuild cost: $0.1376 cold, $0.0798 warm.

And flex-start *stops billing on its own* via `--max-run-duration`, while spot and on-demand run until you remember to delete them. One forgotten weekend on a spot node ($0.58 × 60 h ≈ $35) wipes out months of the 3.8% saving.

**Mount the compile cache and the calculus changes again** — a warm cache turns a preemption from ~18 minutes into ~6, which is exactly why that one `-v` flag matters more than any tuning flag here.

### Cost per million output tokens

| workload | tok/s | spot | flex-start | on-demand |
|---|---:|---:|---:|---:|
| 128 ctx, 64 clients | 1,496.5 | **$0.107** | $0.111 | $0.223 |
| 1K ctx, 64 clients | 1,258.7 | $0.128 | $0.132 | $0.265 |
| 1K ctx, 16 clients | 991.0 | $0.162 | $0.168 | $0.336 |
| 8K ctx, 16 clients | 399.6 | $0.402 | $0.417 | $0.834 |
| 1 client (any ctx) | 123.7 | $1.298 | $1.347 | $2.695 |

**Running one chip 24/7:** $422/month on spot, $438 on flex-start, $876 on-demand.

The headline: **batching is worth more than any pricing decision.** Going from 1 client to 64 at short context is a **12.1× cost reduction per token** — far larger than the 2.08× between on-demand and spot. Tune your concurrency before you shop for discounts.

---

## Part 5 — Four things I was confidently wrong about

The falsified predictions were the most valuable output of this exercise.

**1. "fp8 KV cache will double capacity."** It gave 1.000x, because the layout is word-aligned. Five independent signals said the flag had worked. Only the block-shape line revealed the truth. *Lesson: verify quantization from the boot allocation log, never from the flag being accepted.*

**2. "0.95 memory utilization is a free 8% of KV."** The KV math was right to 0.03% and the engine died anyway, 13 minutes later, because compiled programs live outside the knob's control. *Lesson: a memory knob that doesn't govern all memory will lie to you.*

**3. "Lowering `max-num-batched-tokens` to the exact multimodal floor will cut latency."** It cost 24.7% throughput and moved the latency tail by 0.2%, because 2496 and 4096 round to the same compiled bucket. *Lesson: on TPU, the shape you get is not the number you typed.*

**4. "Capping `max-num-seqs` at 64 will speed up decode."** It didn't, and the arm carrying it was worse on every cell. *Lesson: if your benchmark's offered load never reaches the cap, the cap is untested — say so instead of claiming the win.*

There's a fifth I haven't been able to fix. The boot log reports `Hybrid KV cache layout: num_kv_cache_groups=1` — every one of the 15 cached layers gets full-length KV allocation, even though **12 of them are sliding-attention layers windowed at 512 tokens**. Windowing them would be worth **2.8× the KV capacity** at 16K context, at zero quality cost. It's unreachable: tpu-inference disables sliding windows for any model with more than one head dim, and Gemma 4 has two (256 on sliding layers, 512 on full). The source carries a `TODO: enable sliding windows once mixed dims support`. Worth re-checking on every image bump — it's the largest single win still on the table.

---

## Everything above was validated end to end

The final configuration was booted from scratch and exercised, not assembled from winning fragments:

| check | result |
| :--- | :--- |
| cold boot | **857 s** (compile 685 s = 80%) |
| warm boot, compile cache mounted | **497 s** (−42%) |
| memory | 14.49 GiB cap / 8.97 weights / 5.52 KV — matches every other arm |
| KV capacity | `block_size` 64 x 5,021 blocks = **321,344 tokens** |
| chat completion | ✅ |
| **tool calling** | ✅ `{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}` |
| **multimodal image** | ✅ correctly described a synthetic gradient PNG |
| **long context** | ✅ **26,016 prompt tokens** accepted |
| throughput | 12 cells x 3 reps, cv ≤3.4% |

The three flags that restate a default (`--dtype`, `--kv-cache-dtype`, `--gpu-memory-utilization`) are **confirmed no-ops by vLLM itself** — they do not appear in the engine's `non-default args` when passed at these values. They are in the command for auditability, since the real defaults are computed several layers from where they look like they are declared.

**fp8 KV was re-verified on this exact build**, because the original result predated a container rebuild. Same 5,021 blocks, same 5.52 GiB, same 321,344 tokens as bf16 — while the shape goes `(64,1,2,256)` → `(64,1,4,256)` and the dtype really is `float8_e4m3fn`. The first time I measured this the page size was 32; it reproduces at 64, so the word-alignment mechanism is not a block-size artifact.

---

## Appendix: gotchas that cost me real time

- **`/v1/completions` returns an empty string on `-it` models.** Use `/v1/chat/completions`. An empty benchmark result is expected there, not a broken deploy.
- **`v5e` is `v5litepod` to gcloud.** Accelerator type `v5litepod-1`, runtime `v2-alpha-tpuv5-lite`, `--type=v5litepod --topology=1x1`.
- **Don't hardcode the endpoint.** The external IP changes every time the node is recreated.
- **Don't trust an image "ID" as a pull target.** `sha256:2a4a1f82…` from `docker images` is a config-blob ID, not a manifest digest; `docker pull` by it fails with `unexpected media type`. The version string is the better handle — `0.26.1rc1.dev125+g**a7a204cc6**` embeds vLLM's git SHA, so you can read the exact source your build shipped.
- **Upstream flags E2B's correctness tests as failing.** tpu-inference's own support table marks `gemma-4-E2B-it` ✅ unit / ❌ correctness / ❓ performance, while the 26B and 31B pass all three. My own quality probes were clean (8/9 byte-identical outputs, 3/3 needle retrievals at 2K/8K/14K), but you should know the flag exists.
- **The TPU config docs at `docs.vllm.ai/en/v0.11.1/configuration/tpu/` are excellent and partly stale.** Their headline `VLLM_TPU_MOST_MODEL_LEN` recommendation no longer exists in either vLLM or tpu-inference. Check the version before you copy.

---

## What I'd do differently next time

Test the *whole* configuration, not a change to it. My final recommendation is a config that was never run end-to-end — it's the winning arm plus three pins that *should* be no-ops. Every time I assumed something was a no-op in this project, I was eventually proved wrong.

Untested and plausibly better: `max-model-len 65536` (block_size would go to 128, blocks-per-request stays 512, so it may be free too), `max-num-batched-tokens 8192`, `VLLM_TPU_BUCKET_PADDING_GAP=128`, and **n-gram speculative decoding** — which needs no draft checkpoint and is marked fully passing on TPU, against a workload sitting at 49% of memory bandwidth.
