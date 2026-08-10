![](https://raw.githubusercontent.com/xbill9/gemma4-dev/main/tpu-vllm-v6e1-2b/v6e-cover.jpg)

# Serving Gemma 4 E2B on a TPU v6e-1

A Cloud TPU v6e-1 (Trillium) costs **2.25×** a v5e-1 and returns **1.62–1.68×** the throughput on workloads that fit in a v5e, and **2.32–2.77×** on workloads that do not. Per output token that makes v6e **34–39% dearer** in the first regime and **3–19% cheaper** in the second — so the case for the bigger chip is narrower than the memory ratio suggests, and break-even sits at roughly 270,000 KV tokens.

v6e is not a general upgrade over v5e. It is a **memory** upgrade sold at a compute price: 32 GB against 16, a KV pool of **1,151,744 tokens against 321,376 (3.6×)**, for **1.907×** the bandwidth. Where the extra memory does nothing, the workload pays 2.25× for 1.6×.

Two findings drive the rest:

- **There is no capacity knee at any occupancy tested.** `TTFT = −8542 + 265 × concurrency`, R² = **0.999996**, across **56% to 157%** of the KV pool, with `num_preemptions_total = 0` in every cell. A line fitted entirely below 100% occupancy predicts 157% to within 0.13%.
- **Spot is more expensive than flex-start on this chip** — $1.4033 against $1.35/chip-hr in us-east5, reversing the v5e ordering. The cheaper option is also the preemption-free one that stops billing by itself.

**Configuration:** `v6e-1` (`ct6e-standard-1t`, one Trillium chip), `vllm/vllm-tpu:nightly`, vLLM `0.26.1rc1.dev256+gf5bb701fa`, tpu-inference JAX backend, `google/gemma-4-E2B-it` at bf16, TP=1, `max_model_len` 32768, `max_num_batched_tokens` 4096, `kv_cache_dtype=auto`, prefix caching on. `OUTPUT_LEN` 128 throughout. v5e-1 comparison figures are from the same model and engine family on `v5litepod-1` and are not a controlled A/B — read them as shape, not delta.

---

## Part 1 — Getting a v6e-1 at all

### 1.1 The gcloud spelling table

On v5e, `v5e` is spelled **`v5litepod`** to gcloud. On v6e the marketing name and the CLI value coincide — which teaches a habit that breaks on the next chip.

```
Context                     v5e single chip                                     v6e single chip
--------------------------  --------------------------------------------------  -------------------------------------------
Prose, directory names      v5e-1 / v5e1                                        v6e-1 / v6e1
--accelerator-type          v5litepod-1                                         v6e-1
Flex-start runtime version  v2-alpha-tpuv5-lite                                 v2-alpha-tpuv6e
--type / --topology         v5litepod / 1x1                                     v6e / 1x1
TPU API quota id            TPUV5sLitepodPerProjectPerZoneForTPUAPI             TPUV6EPerProjectPerZoneForTPUAPI
Spot quota id               TPUV5sPreemptibleLitepodPerProjectPerZoneForTPUAPI  TPUV6EPreemptiblePerProjectPerZoneForTPUAPI
```

**Nothing in that table survives a retarget by analogy.** The v6e quota ids drop the `Litepod` the v5e ids carry, and a stale quota id fails *quietly* — it matches no rows rather than erroring, producing a confident "no quota anywhere" that is a typo. `v6e1`, the directory spelling without the hyphen, is still not a valid gcloud value even though `v6e` is.

### 1.2 Provisioning clears three independent gates

A creation must pass three separate checks. They fail differently, and the one that is easiest to query carries the least information.

**Gate 1 — does the zone have `v6e-1` hardware?** Of 37 zones reporting quota, only **18** offer the accelerator type:

```bash
gcloud compute tpus accelerator-types list --filter="type=v6e-1"
```

Google's regions-and-zones page names 8, a strict subset of what the API accepts. Read the API. This gate is provisioning-model-independent.

**Gate 2 — does that zone offer that provisioning model for that accelerator type?** Independent of both quota and hardware, and where a published price stops meaning anything. **us-central1-b and us-south1-a have v6e-1 hardware, quota, and a published `DWS Defined Duration V6e` rate for their region, and both reject flex-start at the API:**

```
FLEX_START provisioning model is not supported for accelerator type "v6e-1" in location "us-central1-b"
```

Confirmed accepting flex-start: **us-east5-a, us-east5-b, europe-west4-a**. This is the v6e analogue of the v5e result, where flex-start `v5litepod-1` was accepted in exactly one zone out of 44. Note that europe-west4 **inverts across generations**: it rejected `v5litepod-1` while quoting a rate for it, and accepts `v6e-1`.

**Gate 3 — is there free capacity right now?** Reachable only after the first two pass, and the one gate that is not a property of the zone. Requests in accepting zones sit at `WAITING_FOR_RESOURCES` for tens of minutes to hours before capacity is granted. **That state is not a failure** — it should not be recorded as one, and the request should not be torn down, because flex-start capacity can take up to two hours to come back once dropped.

> A published rate is not an offer of capacity, and not even an offer of the *provisioning model*.
> Quota is the first thing most people check and the last thing that should reassure them.

### 1.3 Provision

```bash
# Flex-start, via the Queued Resource API — the only model that accepts --max-run-duration,
# i.e. the only one that stops billing on its own.
gcloud alpha compute tpus queued-resources create gemma4-v6e \
  --node-id=gemma4-v6e-node --zone=us-east5-b \
  --accelerator-type=v6e-1 --runtime-version=v2-alpha-tpuv6e \
  --provisioning-model=flex-start --max-run-duration=4h

# Spot — preemptible with ~30s notice, no run limit, and on this chip not the cheap option.
gcloud alpha compute tpus tpu-vm create gemma4-v6e \
  --zone=us-east5-b --type=v6e --topology=1x1 \
  --provisioning-model=spot --version=v2-alpha-tpuv6e
```

`--max-run-duration` is flex-start-only. `--valid-until-duration` bounds the *request*, not the run, so it is shared by all three models. Spot and on-demand nodes bill until preempted or deleted.

The Hugging Face token belongs in Secret Manager, not in the startup script — the rendered script is uploaded as **instance metadata**, and anything baked into it is readable from the instance:

```bash
printf '%s' "hf_xxxxxxxxxxxx" | gcloud secrets create hf-token --data-file=- --project=YOUR_PROJECT
```

### 1.4 Serve

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
    --enable-prefix-caching \
    --disable-chunked-mm-input \
    --limit-mm-per-prompt '{"image":4,"audio":1}' \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

Two deliberate differences from the equivalent v5e configuration:

- **`--gpu-memory-utilization` is absent.** On v5e, 0.92 was a ceiling: 0.95 died after a 691-second compile while loading `jit_structured_decode_fn`, because compiled XLA programs live *outside* the knob's control. That is a 16 GB result with no bearing on 32 GB, and v6e's ceiling is unmeasured, so the flag is left underived rather than carrying another chip's limit forward.
- **`--max-model-len` is 32768**, not the 16384 typical of v5e-era configs. §2.3 shows it is free.

Verify against the boot log, not against the flags:

```bash
sudo docker logs -f vllm-gemma4 2>&1 | grep -E "Memory statistics|GPU KV cache size|block_size"
```

```
Memory statistics | total_hbm_limit_gb=31.24GiB | total_hbm_limit_cap_gb=28.74GiB
                  | total_hbm_used_gb=8.97GiB   | total_hbm_avail_gb=19.77GiB
GPU KV cache size: 1,151,744 tokens
```

Smoke-test on **`/v1/chat/completions`**. Raw `/v1/completions` returns an empty string on `-it` models, which looks exactly like a broken deploy and is not.

---

## Part 2 — The chip

### 2.1 On paper

```
Spec (per chip)  v5e                     v6e (Trillium)                                      Ratio
---------------  ----------------------  ----------------------------------  ---------------------
HBM capacity     16 GB                   32 GB                                                2.0x
HBM bandwidth    800 GiBps               1,638 GBps                          1.907x — units differ
Peak bf16        197 TFLOPs              918 TFLOPs                                          4.66x
Peak Int8        393 TOPs                1,836 TOPs                                          4.67x
TensorCores      1 (4 MXUs, 128x128)     1 (MXUs 256x256, count unresolved)
ICI              400 GBps bidi, 4 ports  800 GBps bidi, 4 ports                               2.0x
Machine type     ct5lp-hightpu-1t        ct6e-standard-1t
On-demand list   ~$1.20/chip-hr          ~$2.70/chip-hr                                      2.25x
```

**The units trap costs 7%.** Google quotes v5e HBM bandwidth in **GiBps** and v6e in **GBps** — on the v5e page, in the same table that uses GBps for ICI. Normalised, 800 GiBps = 858.99 GB/s, making the true ratio **1.907×** rather than the 2.047× obtained by dividing the printed figures. The launch-blog claim that Trillium "doubled" HBM bandwidth is the naive reading. For bandwidth-bound work — decode is bandwidth-bound — that 7% separates a ratio that explains the measurement from one that does not.

**The shape trap:** 2.25× the price for 2× memory, ~1.9× bandwidth, and **4.7× the raw FLOPS**. The 4.7× only pays for prefill-heavy or long-context work that burns the matrix units. For pure decode, v5e is priced close to right and v6e is not.

One row not to build on: Google's v6e page states each TensorCore has **2** MXUs, but two 256×256 arrays is exactly 2× v5e's four 128×128, against a published peak of **4.66×** — which would require a 2.33× clock increase on top. Four 256×256 closes it almost exactly (262,144 MACs × 2 flops × 1.75 GHz = 917.5 TFLOPs against a published 918). The **918 figure is sound**, cross-checking against the same page's 234.9 PFLOPs-per-Pod row. Treat peak compute as reliable and the MXU count as unresolved.

### 2.2 The memory budget

```
                                     v5e-1      v6e-1
---------------------  -------------------  ---------
Total HBM visible                15.75 GiB  31.24 GiB
Allocation cap         14.49 GiB (at 0.92)  28.74 GiB
E2B weights, resident             8.97 GiB   8.97 GiB
KV cache pool                     5.52 GiB  19.77 GiB
KV tokens                          321,376  1,151,744
```

**3.58× the KV capacity for 2.25× the price.** The weights are identical — E2B costs 8.97 GiB wherever it runs, consuming **62% of a v5e's usable budget and 31% of a v6e's**. Every byte of the difference goes to KV.

The arithmetic closes independently on both chips, which is what distinguishes a real allocation from a log line: 19.77 GiB ÷ 18,432 B/token = 1,151,686, within **0.005%** of the measured 1,151,744. The same division reproduces the v5e figure to 0.06%.

### 2.3 Longer context is free

```
max_model_len  block_size           KV pool
-------------  ----------  ----------------
       16,384          32  1,151,776 tokens
       32,768          64  1,151,744 tokens
       65,536         128  1,151,744 tokens
```

**A 0.003% spread across a 4× range.** The Pallas backend derives `block_size` to hold blocks-per-request constant at 512, so doubling the context doubles the page size, halves the block count, and arrives at the same token capacity. The same behaviour holds on v5e at a quarter of the pool.

**Never set `--block-size`.** Pinning it fights the derivation that keeps long context free.

### 2.4 Data types

```
format      native in the MXU?  v5e                     v6e
----------  ------------------  ----------------------  --------------------------
bf16        ✅                   baseline                baseline
int8        ✅ 2x bf16           the only compute win    still the only compute win
fp8         ❌                   storage/bandwidth only  ❌ — unchanged
int4 / fp4  ❌                   footprint only          footprint only
```

Trillium does not bring fp8. **Google's v6e page publishes exactly three peak-compute rows — `bf16: 918 TFLOPs`, `Int8: 1836 TOPs`, and a per-Pod bf16 figure — and no fp8 row anywhere.** The int8 figure being *exactly* 2× bf16 is the signature of a native MXU path; the absent fp8 row is the tell in the other direction. **v7/Ironwood is the first TPU with fp8 in the matrix units** — no conclusion in this section carries forward to it.

The consequence is observable at boot. With `--kv-cache-dtype auto` — the flag never passed — the engine logs this **twenty times**:

```
Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e
```

…and allocates `regular_attn_dtype=bfloat16`. The arithmetic is not close: 1,151,744 tokens against 19.77 GiB is the **bf16** model to 0.01%, while the fp8 model is **50%** off. The two hypotheses are far apart, making this a discriminator rather than a tolerance argument.

That is the sixth false fp8 signal recorded on this stack and the first on a second silicon generation. On v5e, `--kv-cache-dtype fp8_e4m3` was accepted at the CLI, echoed in `non-default args`, praised by a log line, reported in `/metrics`, and allocated a genuinely `float8_e4m3fn` tensor — five independent signals of success — for a **1.000×** capacity ratio, because the KV block layout is word-aligned: as the element width halves, the shape goes `(32,1,2,256) → (32,1,4,256)` and the byte count never moves.

> **Verify quantization from the boot allocation arithmetic — never from the flag being accepted, and
> never from engine prose.**

What v6e plausibly does unblock: on v5e, qwix int8 **weight** quantization died at `RESOURCE_EXHAUSTED: HLO temporaries (16.23G) exceeds available HBM (15.75G)`, short by 0.48 G. v6e has 31.24 GiB, roughly 15 GiB of headroom over that same temporary peak. That failure was an HBM ceiling and this chip doubles it. Untested, and it fails fast — 2.5–4 minutes if it still does not boot.

---

## Part 3 — Six properties of Gemma 4 E2B that decide the rest

None of these change with the chip, but their consequences land differently on 32 GB.

```
field                                      value
-----------------------------------------  --------------------------------------------
num_hidden_layers                          35 (28 sliding / 7 full, i % 5 == 4 is full)
num_kv_shared_layers                       20 — only 15 layers own a cache
num_attention_heads / num_key_value_heads  8 / 1
head_dim / global_head_dim                 256 / 512
hidden_size / intermediate_size            1536 / 6144
vocab_size                                 262,144 (tied embeddings)
sliding_window                             512
resident at bf16                           8.97 GiB
```

**1. "E2B" is not a 2B model.** ~2B *effective* against ~5B total, landing at **8.97 GiB** resident. The `E` prefix is load-bearing: reading `E4B` as "a 4B model" understates its weights by roughly 2×, exactly the difference between fitting a 16 GB chip and not. On v6e this matters less for E2B than for what else becomes possible — **E4B fits at bf16 here and does not on v5e.**

**2. There are two attention geometries.** Sliding layers run at `head_dim` 256; the seven full-attention layers run at **512**, applying to K and V, not just Q. Reading a single `head_dim` and applying it to all 35 layers under-counts the full layers by 2× — a 17% KV sizing error. It is also the root cause of the 2.9× capacity tax in Part 5.

**3. Twenty of the thirty-five layers read another layer's cache.** `first_shared = 35 − 20 = 15`, so layers 0–14 own KV and 15–34 share. The rule is "last preceding layer of the same attention type", and within 0–14 that means **all twenty shared layers resolve to two source caches** — layer 13 for the sliding ones, layer 14 for the full ones.

**4. KV costs 18 KiB/token, and the boot log misreports why.**

```
12 sliding cached layers × 1 KV head × 2 (K,V) × 256 × 2 B = 12,288 B
 3 full    cached layers × 1 KV head × 2 (K,V) × 512 × 2 B =  6,144 B
                                                    total  = 18,432 B = 18 KiB/token
```

The line describing the cache, `regular_attn_shape=(num_blocks, (64, 1, 2, 256))`, is a **first-wins sample taken from layer 0**, which is sliding, hence 256. It says nothing about layers 4, 9 and 14: `count` increments for all 15 tensors while `shape` is written once and never updated. The allocation is correct; the line is misleading. **Size KV from the config geometry and check it against `total_hbm_avail_gb`.**

**5. One KV head means more chips make things worse.** `num_key_value_heads = 1` is full MQA, and a single head **cannot be sharded** — runtimes pad `num_kv_heads` up to a multiple of the TP size, so at TP=4 the same head is replicated at **4× the KV memory**. A larger topology multiplies this model's KV cost rather than dividing it. **The answer to "E2B needs more memory" is a bigger chip, not more chips.**

**6. The heads do not tile the hidden size.** `8 × 256 = 2048` against `hidden_size = 1536`, so the Q projection is rectangular. Code computing `head_dim = hidden_size / num_heads` gets **192** and is silently wrong.

One further property explains performance rather than memory: **4.38 GiB of the 8.97 GiB resident is per-layer embedding tables** (262,144 × 256 × 35), which are *gathered per token, not streamed*. Only ~3.15 GiB moves per decode step, which is why an 8.97 GiB model decodes as fast as it does, and it sets the bandwidth floor used in Part 4.

---

## Part 4 — Results

Roles are sized against the v6e pool: `control` fits both chips trivially, `bandwidth` fits both but moves substantial KV per step, `v6e_only` exceeds v5e's entire pool, and `long_ctx` requires `max_model_len > 16384` — impossible on v5e at any setting.

```
   ctx  clients  role       KV needed  v5e tok/s  v6e tok/s  ratio  per-stream  median TTFT
------  -------  ---------  ---------  ---------  ---------  -----  ----------  -----------
   128        1  control          256      123.3      202.9  1.65x       202.9        12 ms
   128        8  control        2,048      738.3    1,195.1  1.62x       149.4        26 ms
 1,024       16  control       18,432      896.1    1,508.0  1.68x        94.2       149 ms
 4,096       64  bandwidth    270,336      585.9    1,360.0  2.32x        21.3       303 ms
 8,192       32  bandwidth    266,240      307.8      758.4  2.46x        23.7       348 ms
 8,192       64  v6e_only     532,480      314.4      870.0  2.77x        13.6     1,006 ms
16,000       32  v6e_only     516,096      166.8      432.6  2.59x        13.5     3,276 ms
16,000       64  v6e_only   1,032,192      166.7      446.0  2.68x         7.0     8,459 ms
32,000       16  long_ctx     514,048                 242.6               15.2     2,233 ms
32,000       32  long_ctx   1,028,096                 229.0                7.2     8,760 ms
```

There is correctly no v5e reference for the `long_ctx` cells: that configuration cannot exist on v5e.

**A caution about the 4,096–8,192 band.** Those three cells are the most sensitive in the matrix to how a sweep is ordered. Run at a shared `--seed` after a longer-context cell, they report **12–19% higher** than they do with a distinct seed per cell, while every cell at 128, 1,024, 16,000 and 32,000 tokens moves by under 6.3% either way. The figures above are the clean-seed ones. Anything quoting this band from a single-seed sweep is quoting the high side.

### 4.1 The asymmetry

```
regime                            cells  mean vs v5e
--------------------------------  -----  -----------
working set < 10% of v5e's pool       3        1.65x
working set >= 83% of v5e's pool      5        2.56x
```

Against a **1.907×** bandwidth ratio and a **2.25×** price ratio.

If v6e were simply faster, every cell would improve by roughly the same factor. Control cells move roughly with bandwidth and no more. Cells where v5e was over its pool — evicting and recomputing — move about 2.6×, because v6e is not doing that work. **On decode throughput alone this chip is a poor deal. It pays for capacity, not speed** — and note that even the memory-bound mean of 2.56× only just clears the 2.25× price ratio.

Single stream is the cleanest bandwidth read: **TPOT 4.72 ms on v6e against 8.02 ms on v5e, 1.70×** on a 1.907× bandwidth ratio. Decode moves ~3.15 GiB per step (derived from layer geometry), which at 1,638 GB/s is a **2.06 ms floor against 4.72 ms measured — 44% of peak**, slightly worse utilisation than v5e's 49%. Roughly 2× of headroom sits in fixed per-step cost on both chips, not in memory bandwidth.

The **1.65×** control figure sits *below* the bandwidth ratio and is unexplained. The MXU geometry change (4×128×128 → 256×256, a 4× larger minimum tile) is a plausible cause but is not demonstrated; separating it from "small batches do not saturate bandwidth" requires a batch-size sweep at fixed short context.

### 4.2 There is no capacity knee

A widely used v5e rule of thumb — keep `clients × context` under ~78% of the pool — scales to ~900,000 tokens on v6e. Tested directly at fixed 16,000 context with only concurrency varying:

```
clients  KV needed  % of pool  tok/s  median TTFT  preemptions
-------  ---------  ---------  -----  -----------  -----------
     40    645,120        56%  452.7     2,088 ms
     46    741,888        64%  465.9     3,659 ms
     52    838,656        73%  470.6     5,273 ms
     56    903,168        78%  467.7     6,309 ms
     60    967,680        84%  471.6     7,373 ms
     64  1,032,192        90%  446.0     8,459 ms            0
     72  1,161,216       101%  458.5    10,544 ms            0
     80  1,290,240       112%  459.1    12,689 ms            0
     96  1,548,288       134%  472.0    16,937 ms            0
    112  1,806,336       157%  465.5    21,229 ms            0
```

**`TTFT = −8542 + 265 × concurrency`, R² = 0.999996 over all ten points.** Throughput is flat at 446.0–472.0 tok/s (**5.8%**) across the range. TPOT sits at 66.2–67.4 ms and does not move. `num_preemptions_total` is **0 in every cell, including 157% occupancy.**

The line is the most reproducible result in the matrix. Re-measured on a separate node, the 90% and 157% points land at **+0.18%** and **+0.22%** of what it predicts, with zero preemptions in both.

There is no knee at 78%, none anywhere in 56–157%, and **crossing 100% of the pool is not an event**. A line fitted entirely below 100% predicts the 157% point to within 0.13%.

**The scheduler admission-controls rather than evicts.** It admits what fits and queues the rest, so the working set never thrashes. Occupancy alone costs nothing; *eviction* would, and it never engaged. The v5e rule was a queueing curve read as a memory cliff.

> **Size by the latency you will accept, not by pool occupancy.** Throughput saturates near
> concurrency 40, and every further concurrent request buys **265 ms of TTFT and nothing else**. The
> pool bounds what is *resident*; it is not a performance threshold.

### 4.3 Two benchmarking traps on this stack

**1. A config can silently fail to take effect, and the obvious check can miss it.** vLLM prints `max_model_len` inside a dict repr (`'max_model_len': 16384`), so a regex written for the bare key matches nothing and returns no result rather than a mismatch — indistinguishable from a pass unless the check distinguishes them. The reliable signal is a *different* derived quantity downstream of the same setting: `block_size` reads 32 at 16384 and 64 at 32768. **Verify a setting through two independent derivations.**

**2. `--seed 0` plus prefix caching silently couples cells.** vLLM defaults to `enable_prefix_caching=True`, and the random dataset is deterministic in the seed, so two cells at the same `input_len` draw overlapping prompts and the later one is served from cache. A `16000×32` cell run after `16000×64` took **1,539,904 cache-hit tokens against 1,536,000 input tokens** — essentially all of it — and its 2,461 ms TTFT is not comparable to anything.

That artifact, combined with an unsampled gap between 46% and 89% of pool, is enough to manufacture an apparent 3.4× cliff: **two points far apart with the lower one artificially fast reads as a cliff.** Vary the seed per cell; the knee and overflow sweeps above measured **0.0% prefix hits throughout**.

---

## Part 5 — Why there is no cliff, and what it costs

E2B caches 15 of 35 layers: 12 sliding-window at head_dim 256 with a 512-token window, 3 full-attention at 512. tpu_inference sees two head dims, sets `disable_sliding_window`, and gives **every** layer full-length blocks — `num_kv_cache_groups=1`, 15 tensors, confirmed in the boot log.

Allocation and reads therefore diverge:

```
                      12 sliding layers               3 full layers
--------------------  ------------------------------  ---------------------
allocated             full context — 12,288 B/token   6,144 B/token
read per decode step  6.29 MB, constant once L > 512  98.3 MB at L = 16,000
```

At 16,000 context, **94% of the KV bytes read come from 3 of the 15 cached layers.** The pool is charged ~18 KiB/token where ~6.4 KiB would suffice.

This does not prevent a cliff — it moves the cliff closer. It is a capacity *tax*, not a latency shield; no cliff appears because nothing is ever evicted.

The tax is worth **2.91×**: at 32,768 context, 576.0 MiB/seq allocated against 198.0 MiB/seq windowed. The trigger is `disable_sliding_window = len(head_size_set) > 1`, and Gemma 4's 256/512 head-dim split trips it **family-wide, at every size**. It is gated on an upstream `TODO` and is not settable from the serving side.

> That single upstream fix is worth more than every available quantization flag combined — 2.91× the
> effective KV capacity at zero quality cost.

---

## Part 6 — Cost

Rates from the Cloud Billing Catalog, per chip-hour:

```
model             v5e (us-west4)  v6e (us-east5)  ratio
----------------  --------------  --------------  -----
Spot                     $0.5779         $1.4033  2.43x
Flex-start (DWS)         $0.6000         $1.3500  2.25x
On-demand                $1.2000         $2.7000  2.25x
```

**On v6e, spot is dearer than flex-start**, inverting the v5e ordering and the advice with it. On v5e, flex-start cost 3.8% more and bought preemption-freedom. On v6e flex-start is **both cheaper and preemption-free**, and it self-terminates via `--max-run-duration`, which the other two do not. There is no trade left to make.

Two catalog naming traps: flex-start is sold as **"DWS Defined Duration"** (Dynamic Workload Scheduler) and drops the `Tpu` prefix (`DWS Defined Duration V6e`), while spot is `usageType: Preemptible` spelled `TpuV6e attached to Spot Preemptible VMs`. The `Reserved …`, `Commitment v1: …` and **`Capacity Optimized TpuV6e …`** SKUs describe the same chip in the same region and three are also `OnDemand`, so anchor the match patterns.

### Cost per million output tokens, flex-start both sides

```
   ctx  clients  KV needed vs v5e pool  v5e @ $0.60  v6e @ $1.35  v6e vs v5e
------  -------  ---------------------  -----------  -----------  ------------------
   128        1                   0.1%       $1.352       $1.848  1.37x dearer
   128        8                   0.6%       $0.226       $0.314  1.39x dearer
 1,024       16                     6%       $0.186       $0.249  1.34x dearer
 4,096       64                    84%       $0.284       $0.276  0.97x — break-even
 8,192       32                    83%       $0.542       $0.494  0.91x — cheaper
16,000       32                   161%       $0.999       $0.867  0.87x — cheaper
16,000       64                   321%       $1.000       $0.841  0.84x — cheaper
 8,192       64                   166%       $0.530       $0.431  0.81x — cheaper
32,000       16                                           $1.545  v6e only
```

**One chip 24/7:** $438/month on v5e flex-start, **$986/month on v6e flex-start**.

### Choosing

The rule follows from the table, and it is a memory question rather than a throughput one:

> **Determine whether the steady-state working set — `clients × context` — exceeds ~270,000 tokens,
> roughly 84% of a v5e's pool. Below that, v5e is 26–29% cheaper per token. Above it, v6e is cheaper,
> but by 3% at the boundary and never by more than 19%**, because v5e can no longer hold the job and
> starts recomputing it.

Two things follow that the memory ratio alone would not predict. **The advantage saturates:** once past the boundary, v6e settles at 0.81–0.87× and does not keep improving with working-set size — 16,000 × 64 runs at 321% of a v5e pool and is still only 16% cheaper. And **break-even is not a comfortable margin.** At 84% of pool the two chips are within 3% of each other, which is inside the run-to-run spread of the band those cells sit in.

This is not "v6e for long context". The `4096 × 64` cell is only 4K of context and reaches the boundary anyway, because 64 clients × 4,096 tokens is 270,336 KV tokens. **Concurrency crosses the line as readily as context does.** The threshold is a product, which is why the single-stream cell is v6e's worst showing at 1.37× dearer.

One effect dominates both columns: moving from a single stream to the best-measured concurrent cell is a **7.4× cost reduction per token on v6e** ($1.848/1M at 128 ctx × 1 client → $0.249/1M at 1,024 ctx × 16). That is larger than the 2.25× between the two chips and larger than any other effect here. **Tune concurrency before shopping for chips.**

---

## Limits

- **Nothing above 157% of pool, and nothing that forces preemption.** Every cell was admission-controlled; the eviction regime — the condition that would actually produce a cliff — is untested.
- **The knee sweep varied concurrency at fixed context only.** A context sweep at fixed concurrency loads the 3 full-attention layers differently and is not covered.
- **The 1.65× control figure is unexplained**, sitting below the bandwidth ratio.
- **Single run per cell, on two independent nodes.** Every cell in the results table was measured twice, on separate v6e-1 nodes in different zones. Seven of ten agreed to within 6.3%; the three in the 4,096–8,192 band diverged 12–19% and are reported at the clean-seed value. No within-node variance figure is established, so a difference under ~6% between any two cells here is not a result.
- **Cold-boot and compile timings are unmeasured on v6e.** The v5e figures — 857 s cold, 685 s of it compilation, 497 s with the compile cache mounted — describe a mechanism that carries, not timings that do. v6e compiles its own kernels.
- **`--gpu-memory-utilization` has no established v6e ceiling.**
- **qwix int8 weights, `max_model_len 65536`, `max-num-batched-tokens 8192`, `VLLM_TPU_BUCKET_PADDING_GAP=128`, and n-gram speculative decoding** are untried on this chip.
- **Upstream flags E2B's correctness tests as failing** — tpu-inference's support table marks `gemma-4-E2B-it` ✅ unit / ❌ correctness / ❓ performance, while the 26B and 31B pass all three. Quality probes on v5e were clean (8/9 byte-identical outputs, 3/3 needle retrievals at 2K/8K/14K).

---

## Further reading


- **"Self-hosting a lite agent backend on one TPU: Gemma 4 E2B + vLLM on a v5e-1"** — the companion piece, and the source of every v5e-1 figure quoted here: the 321,376-token KV pool, the 8.02 ms single-stream TPOT, the `--gpu-memory-utilization 0.95` failure at 691 s, the 1.000× fp8 KV result, and the per-cell throughput used in the ratio and cost tables. Same model, same engine family, same `OUTPUT_LEN`, one chip generation down.
- [Debugging deployments with Gemma 4 4B, TPU v6e-1, MCP and Antigravity CLI](https://xbill999.medium.com/debugging-deployments-with-gemma-4b-tpu-v6e-1-mcp-and-antigravity-cli-c9846231237a) — the earlier v6e-1 write-up, covering the MCP tooling and the deploy path rather than the serving numbers.
- [Cloud TPU v6e](https://docs.cloud.google.com/tpu/docs/v6e) and [Cloud TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e) documentation — the source of both spec columns, and of the GiBps/GBps units mismatch: the v5e page quotes HBM bandwidth in GiBps while the v6e page uses GBps.
- [Dynamic Workload Scheduler pricing](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/dws) — flex-start is billed under DWS, which is why the SKU is named "DWS Defined Duration V6e" rather than anything containing "flex".
- [TPU system architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — the 256×256-versus-128×128 MXU dimensions behind the unresolved MXU-count arithmetic in §2.1.

---

## Appendix: operational traps

- **A published price is not an offer of capacity, or even of the provisioning model.** Three independent gates; quota is the weakest signal and the one most often checked first.
- **Google's regions-and-zones page undercounts v6e zones** — 8 documented against 18 the API accepts.
- **`WAITING_FOR_RESOURCES` is not a failure.** Do not record it as one and do not tear the request down; flex-start capacity can take two hours to return.
- **v6e quota ids drop the `Litepod` that v5e's carry**, and a stale id matches no rows rather than erroring — indistinguishable from "no quota anywhere".
- **`/v1/completions` returns an empty string on `-it` models.** Use `/v1/chat/completions`.
- **Endpoints are not stable.** The external IP changes every time the node is recreated.
- **Google quotes v5e bandwidth in GiBps and v6e in GBps.** Normalise before dividing.
- **`31.24 GiB` and `33.55 GB` are the same number.** XLA prints GiB; `memory_analysis()` returns bytes.
- **XLA compares temporaries alone against the whole chip** and does not subtract resident weights. `available HBM (31.24G)` in an error message is not headroom.

---

## Figures

![v6e-1 returns 1.62–1.68x on workloads that fit a v5e-1 and 2.32–2.77x on those that do not, against a 2.25x price ratio.](https://raw.githubusercontent.com/xbill9/gemma4-dev/main/tpu-vllm-v6e1-2b/docs/assets/v6e-asymmetry.png)
*v6e-1 returns 1.62–1.68x on workloads that fit a v5e-1 and 2.32–2.77x on those that do not, against a 2.25x price ratio.*

![Time to first token stays linear from 56% to 157% of the KV pool, with zero preemptions.](https://raw.githubusercontent.com/xbill9/gemma4-dev/main/tpu-vllm-v6e1-2b/docs/assets/v6e-no-knee.png)
*Time to first token stays linear from 56% to 157% of the KV pool, with zero preemptions.*
