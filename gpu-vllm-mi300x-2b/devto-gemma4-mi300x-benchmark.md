---
title: "Nine Million Tokens of KV Cache, and Prefill Is Still the Wall: Gemma 4 E2B on One MI300X"
published: false
description: "A 12-cell serving sweep of Gemma 4 E2B on one AMD Instinct MI300X under vLLM on ROCm. The card allocates 155 GiB of KV cache and never comes close to filling it — and the first sweep was measuring the prefix cache rather than the card."
tags: amd, vllm, rocm, gemma
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-mi300x-2b/devto-benchmark-cover.2ff88304.jpg
---

This article provides a step by step benchmark of Gemma 4 E2B on one AMD Instinct MI300X hosted GPU
enabled system. A suite of Python MCP tools is built to simplify management of the vLLM hosted
deployment, and the sweep itself runs through those same tools.

https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b

The deployment walk-through that gets to a serving endpoint — droplet lifecycle, vetting the ROCm
image before a 35 GB pull, and verifying every modality the checkpoint claims — is a separate
article in the same directory. This one starts where that one stops: the endpoint is up, and the
question is what it actually does under load.

---

#### What is this project trying to Do?

Measure one card honestly. One AMD Instinct MI300X on a DigitalOcean GPU droplet reached through
AMD Developer Cloud, serving `google/gemma-4-E2B-it` through vLLM in Docker, swept across four
concurrency levels and three context lengths.

Three things came out of it, and only one was the thing I set out to measure:

- **The card allocates 155.04 GiB of KV cache — nine million tokens — and the workload never used
  more than 5.8% of it.** The throughput ceiling at long context is prefill, not memory. That
  inverts the intuition the TPU siblings in this monorepo built up, where the KV pool is the wall.
- **My first two sweeps were partly measuring vLLM's prefix cache.** `vllm bench serve` seeds its
  random prompts at 0, so cells and repeats generated *identical* prompts and read each other's
  cache entries. At 8192-token context that was worth **2.25x**. Nothing warned me.
- **KV costs 18,443 bytes per token on this stack**, which matches the geometry-derived 18,432 B in
  this monorepo's `MODELS.md` to 0.06% — and therefore does not reproduce the roughly-half figure
  an earlier NVIDIA L4 run reported.

---

#### Where do I start?

The same incremental approach as the sibling rigs, one step further along. The endpoint has to be up
and verified before a number off it means anything, so the order is: confirm the droplet, confirm the
container, confirm the engine's own view of its memory, and only then put load on it.

Every step below is cheap and reversible. The only thing that costs money is the droplet, and it is
already running.

---

#### At this point you should have

- A DigitalOcean GPU droplet tagged `gemma`, powered on, with an MI300X visible to ROCm.
- `vllm` running in Docker on it, answering on `127.0.0.1:8000`.
- The rig's MCP server registered, so the tools below are reachable from Claude Code.
- `DIGITALOCEAN_ACCESS_TOKEN` in the rig's `.env`, mode 0600 and gitignored. It is never in
  `tpu.env`, which is committed.

---

#### Confirm the Droplet Before Anything Else

```
list_droplets
```

```
📡 1 droplet(s) tagged `gemma`.

| Name | ID | Status | Size | Region | Public IPv4 |
| --- | --- | --- | --- | --- | --- |
| `debian-gpu-mi300x1-192gb-devcloud-atl1` | 601142018 | active | gpu-mi300x1-192gb-devcloud | atl1 | 165.245.134.217 |
```

Every lookup is filtered by the tag, so the server can only ever see droplets somebody deliberately
tagged for it. An untagged droplet in the same account is invisible, which is the point — a mistyped
id cannot power-cycle an unrelated machine.

There is no `create` tool and no `destroy` tool, on purpose. Both are dollar-per-hour decisions and
they stay a deliberate step in the console. Worth repeating because it catches people: **powering a
droplet off does not stop DigitalOcean billing it.** The resources stay reserved and the meter runs.
Only destroying it stops the charge.

---

#### Verify the GPU Architecture

```
gpu_status debian-gpu-mi300x1-192gb-devcloud-atl1
```

```
✅ GPU reporting on `debian-gpu-mi300x1-192gb-devcloud-atl1`.

| Card | Product | GPU use % | VRAM used % |
| --- | --- | --- | --- |
| card0 | Aqua Vanjaram [Instinct MI300X VF] | 0 | 87 |

📡 1 GPU(s) reported by rocm-smi.
```

The card presents as an SR-IOV **VF**, which reads alarmingly like a partition. It is not: all 304
compute units and the whole 191.7 GiB are there.

This tool was broken when I started and the failure was instructive. It ran `rocm-smi --json`, which
**rocm-smi refuses** — `Cannot print JSON/CSV output for concise output` — because the default
concise table has no JSON form. The fix is to name the fields:

```
rocm-smi --showid --showproductname --showtemp --showuse --showmemuse --json
```

Naming fields is what makes it emit JSON at all. The tool had been reporting *no card* on a perfectly
healthy MI300X, and since `rocm-smi` exits 0 when it fails, nothing upstream noticed.

---

#### Read the Engine's Own Memory Accounting

Before measuring throughput, get the engine to say what it did with the card. This is the single most
useful thing in the boot log and it is printed exactly once.

```
docker logs vllm 2>&1 | grep -E 'Available KV cache memory|GPU KV cache size|Actual usage is'
```

```
Available KV cache memory: 155.04 GiB
GPU KV cache size: 9,026,017 tokens, Maximum concurrency for 32,768 tokens per request: 275.45x
Free memory on device (191.36/191.69 GiB) on startup. Desired GPU memory utilization is (0.9,
172.52 GiB). Actual usage is 11.88 GiB for consumed memory (weights + non-torch), 5.59 GiB for peak
activation, and 3.9 GiB for CUDAGraph memory. Current kv cache memory in use is 155.04 GiB.
```

The whole HBM budget, from the engine rather than from arithmetic:

| Component | GiB | Share of 191.69 GiB |
| --- | ---: | ---: |
| KV cache | 155.04 | 80.9% |
| Weights + non-torch | 11.88 | 6.2% |
| Peak activation | 5.59 | 2.9% |
| CUDA graph pool | 3.90 | 2.0% |
| Unallocated (below `--gpu-memory-utilization 0.90`) | 15.28 | 8.0% |

A 2B-class checkpoint on a 192 GiB card spends four fifths of the card on KV. That is the shape of
the whole result: there is so much KV that nothing in the sweep can exhaust it.

---

#### The KV Number Settles an Open Question in This Repo

Divide the engine's pool by the engine's token count:

```
155.04 GiB x 1073741824 / 9,026,017 tokens = 18,443.7 bytes/token
```

*(That division is arithmetic, not a measurement. Both inputs are the engine's.)*

This monorepo's `MODELS.md` derives **18,432 B/token** for E2B from the checkpoint's layer geometry —
sliding-attention layers 256-wide, full-attention layers 512-wide — and cross-checks it exactly on two
TPU runs. The ROCm figure lands **0.06% off it**.

That matters because `MODELS.md` carries an open discrepancy: an NVIDIA L4 run under stock vLLM
reported 9,622 B/token, a 1.92x gap, and the leading hypothesis was that **vLLM v1 charges
sliding-window layers only their window** rather than the full context. If that were an engine-level
policy it would apply here too, on vLLM 0.29.1 — and it does not. So the hypothesis does not survive
this data point, and whatever produced the L4 figure is narrower than "the vLLM path".

**One card, one run.** It is a third data point on a third stack, not a closed case.

---

#### The Benchmark Tool, and Why It Runs in Its Own Container

The sibling vLLM rigs expose a `run_vllm_benchmark` tool that drives `vllm bench serve`. This rig had
no such tool, so the first job was adding one.

```python
docker run --rm --net host \
  -v /opt/hf-cache:/root/.cache/huggingface -v /dev/shm:/dev/shm --shm-size 8g \
  --entrypoint vllm \
  vllm/vllm-openai-rocm:nightly-rocm100 bench serve ...
```

Two details differ from the serving container and both are deliberate.

**No GPU device is attached.** No `--device /dev/kfd`, no `--device /dev/dri`, no `--group-add`. The
bench client is an HTTP load generator that needs a tokenizer and nothing else. Without a device it
*cannot* touch the card, which is what makes it safe to run beside a live server — it costs host CPU,
not card time.

**`--entrypoint vllm` is set unconditionally.** The serving path in this rig branches on the image,
because `vllm/vllm-openai-rocm` declares `ENTRYPOINT ["vllm","serve"]` while AMD's own `rocm/vllm`
images declare none. For the bench that branch is unnecessary and harmful: overriding the entrypoint
outright works for both, and without it the official image reads `bench` as a model id.

---

#### The Sweep Grid

Four concurrency levels against four context lengths, 128 output tokens throughout, `ignore_eos` so
every request produces exactly 128 tokens rather than stopping early. `num_prompts = max(8, 2 x
concurrency)`. The grid matches `tpu-vllm-v5e1-2b` and `tpu-vllm-v6e1-2b` so the *shape* is
comparable.

```
python3 benchmarking_suite.py --droplet debian-gpu-mi300x1-192gb-devcloud-atl1 \
  --run-id 2026-09-16-vllm-sweep-mi300x --repeat 3 --seed-base 5000
```

```
16 cells, 12 runnable, 4 infeasible at max_model_len 32768
```

The 32768-context row cannot exist: `32768 + 128 > 32768`. Those four cells are recorded
`infeasible` with the reason rather than dropped, because a missing cell is indistinguishable from
one nobody ran. Schema 1.1 has a status field for exactly this.

---

#### The Benchmark Was Measuring the Prefix Cache

This is the part worth carrying to any other engine, and it cost two full sweeps.

`vllm bench serve` derives its random prompts from `--seed`, which defaults to **0**. vLLM's default
config has `enable_prefix_caching=True`. Put those together and two runs that share a seed generate
the *same prompts* — so the second one reads the first one's cache entries and reports a speedup that
is the cache, not the card.

It is worse than per-repeat. Cells at the same context length draw from the same pool, so `c4-in128`
replayed `c1-in128`'s prompts. Nothing errors. Nothing warns. The throughput just looks good.

The size of it, measured within a single sweep — repeat 1 ran seeds a previous sweep had already
served, repeats 2 and 3 did not, and all three ran inside the same minute so card warm-up is not the
explanation:

| Context | Concurrency | Warm tok/s | Cold tok/s | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 340.7 | 340.5 | 1.00x |
| 128 | 64 | 10,849.2 | 10,496.2 | 1.03x |
| 1,024 | 16 | 3,415.4 | 2,891.3 | 1.18x |
| 1,024 | 64 | 8,930.4 | 6,252.7 | 1.43x |
| 8,192 | 16 | 1,961.5 | 870.2 | 🥇 2.25x |
| 8,192 | 64 | 2,095.8 | 948.9 | 2.21x |

The gradient is the tell. A prefix cache saves prefill, prefill's share of the work grows with
context, so the benefit should be nil at 128 tokens and large at 8192. It is exactly that: **1.00x
and 2.25x**. Those two numbers are the control that makes the rest of the table mean something.

The fix is one line — give every cell and every repeat a seed no other run in the sweep uses:

```python
def _seed(base: int, index: int, rep: int) -> int:
    return base + index * 100 + rep
```

What it bought, in the reported run's own spread:

| Sweep | Worst-cell spread (cv) | Cells under 1% cv |
| --- | ---: | ---: |
| Seeds 0,1,2 per cell | 51.1% | 4 of 12 |
| 🥇 Unique seed per cell and repeat | 8.4% | 9 of 12 |

A 51% coefficient of variation is not noise. It is two different experiments averaged together.

**Prefix caching was left on**, because that is how the rig actually serves. The numbers below are
the unique-prompt case, which is the conservative end: real traffic with a shared system prompt will
do better than this, and the table above says roughly how much better.

---

#### Results: Aggregate Output Tokens/sec

Median of 3 repeats per cell, 12 cells, every cell a unique prompt set.

| context ↓ / clients → | 1 | 4 | 16 | 64 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 340.4 | 1,118.1 | 3,583.6 | 🥇 **10,293.6** |
| 1,024 | 305.6 | 967.3 | 2,681.4 | 5,447.6 |
| 8,192 | 193.2 | 449.3 | 711.2 | 725.3 |

#### Median Time to First Token (ms)

| context ↓ / clients → | 1 | 4 | 16 | 64 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 12.3 | 21.1 | 49.7 | 126.4 |
| 1,024 | 30.1 | 68.1 | 197.1 | 568.8 |
| 8,192 | 164.3 | 497.3 | 1,387.0 | **4,640.1** |

#### Per-stream Tokens/sec — What One User Feels

| context ↓ / clients → | 1 | 4 | 16 | 64 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 349.7 | 291.5 | 244.5 | 196.5 |
| 1,024 | 326.8 | 278.6 | 220.8 | 136.4 |
| 8,192 | 255.1 | 198.0 | 84.8 | **19.1** |

---

#### The Three Regimes

**Short context scales almost linearly to 64 clients.** 340.4 to 10,293.6 output tok/s is **30.2x** *(arithmetic)* for
64 times the clients, and a single stream still feels fast at 196.5 tok/s with the card fully loaded.
This is the regime agent traffic lives in, and the card barely notices it.

**Long context stops scaling at 16 clients.** 711.2 to 725.3 tok/s going from 16 to 64 is **2%** *(arithmetic)* for
4 times the clients, while median TTFT goes 1,387 ms to 4,640 ms and p99 reaches 8,746 ms. Past 16 clients at 8K
you are buying latency, not throughput.

**And the reason is not memory.** This is where the MI300X parts company with the TPU siblings, where
the same collapse is a KV wall:

```
64 clients x 8,192 tokens = 524,288 KV tokens
Resident KV pool          = 9,026,017 tokens
Occupancy                 = 5.8%
```

*(Arithmetic; both inputs measured.)* The pool is 17x larger than the workload's peak demand *(arithmetic: 9,026,017 / 524,288)*. Nothing
is being evicted and nothing is queueing for blocks. The limit is prefill throughput: the same cell
moves **47,142 total tok/s** including prompt tokens, against 46,231 at 16 clients — the card is
saturated on prefill work and the extra clients only lengthen the queue.

**So on this part, the sizing question is inverted.** On the TPU rigs the rule is `clients x context
< KV pool`. Here that rule never binds — 275x concurrency at full 32K context, per the engine's own
report — and the operating point is set by how much prefill you are willing to pay for.

---

#### Recommended Client Counts

| workload | context | clients | expected |
| --- | ---: | ---: | --- |
| Interactive chat / agent turns | ≤1K | **64** | 5,448–10,294 tok/s, 126–569 ms TTFT |
| Latency-sensitive interactive | ≤1K | **16** | 2,681–3,584 tok/s, 50–197 ms TTFT |
| RAG / long documents | 8K | **16** | 711 tok/s, 1,387 ms TTFT |
| Long-context interactive | 8K | **≤4** | 449 tok/s, 497 ms TTFT, 198 tok/s per stream |

The 8K/64 cell is in the report and is not in this table on purpose. It is 2% more throughput than
8K/16 for 3.3 times the time to first token *(arithmetic on the TTFT table)*.

---

#### Cost Analysis

The droplet bills at **$1.99/hour**, read from the DigitalOcean v2 API rather than a price page —
`droplet.size.price_hourly` for `gpu-mi300x1-192gb-devcloud`, read 2026-09-16.

Cost per million output tokens is `1.99 / (tok/s x 3600) x 1,000,000`. *(Arithmetic on a measured
rate and a measured price.)*

| Operating point | tok/s | $/M output tokens |
| --- | ---: | ---: |
| 🥇 128 ctx, 64 clients | 10,293.6 | **0.054** |
| 🥈 128 ctx, 16 clients | 3,583.6 | 0.154 |
| 1,024 ctx, 64 clients | 5,447.6 | 0.102 |
| 8,192 ctx, 16 clients | 711.2 | 0.777 |
| 128 ctx, single stream | 340.4 | 1.624 |
| 8,192 ctx, single stream | 193.2 | 2.862 |

**Serving one stream at a time costs 30x more per token than serving 64** *(arithmetic: 1.624 / 0.054)*, on the same card at the same
hourly rate.** Batching is a far bigger lever here than any hardware choice, and the spread is wider
than on the smaller cards precisely because there is so much headroom to fill.

Compute only — no storage, no transfer, and no idle time. A card at $1.99/hour producing nothing
costs the same as one producing 10,000 tok/s.

---

#### Why There Is No Comparison Table

The sibling articles end with a table putting two deployments side by side. This one cannot, and the
reason is worth stating rather than quietly omitting.

**Nothing else in this monorepo serves this checkpoint on AMD.** There is no A/B twin — the TPU rigs
differ in chip, runtime, control plane and cloud all at once, and the CUDA rigs differ in chip,
cloud and instance shape. Differencing a number from here against one of those and reading the
result as a hardware finding would be four confounded variables reported as one.

What can be said honestly is narrower and still useful: this card serves a 2B-class checkpoint at
**$0.054 per million output tokens** at its best measured operating point, and the constraint is
prefill throughput rather than memory.

---

#### What I Was Wrong About

**I assumed a 192 GiB card would be interesting because of the KV pool.** It is not. The pool is so
far oversized for a 2B checkpoint that it never enters the picture; the card would serve this model
with a tenth of it. The interesting property is prefill throughput, which is the thing the headline
HBM number tells you nothing about.

**I assumed repeated runs of a benchmark were independent.** They are not, on any engine with prefix
caching and a seeded prompt generator. This is not an AMD or a vLLM problem, it is a property of
benchmarking a cache-bearing server with deterministic inputs, and it will bite the same way on
CUDA.

**I assumed a config the engine accepted was a config that did something.** `--limit-mm-per-prompt`
with a non-zero audio count is accepted here and cannot work, because no ROCm vLLM image ships the
`vllm[audio]` extras. `audio: 0` in this rig is a fact about the images, not a preference.

**I assumed the first sweep was as good as the second.** It was not, and I still cannot fully explain
it: the first sweep of the day, on a server idle for three hours, read 4,202 tok/s on a cell that
later read 10,294 — a 2.4x gap on a cell where prefix caching is worth 1.03x. One observation, not a
characterised effect, and the console output is archived in the repo rather than written up as a
finding. **Discard your first sweep.**

---

#### Tear Down

There is nothing to tear down, and that is the deliberate part. `stop_vllm` releases the card:

```
stop_vllm debian-gpu-mi300x1-192gb-devcloud-atl1
```

```
✅ `docker rm -f vllm` exited 0.
```

The droplet keeps billing. `stop_droplet` warns about this rather than implying otherwise, and there
is no `destroy_droplet` tool in this rig — destroying the droplet is the only thing that stops the
meter, and it stays a human decision in the DigitalOcean console.

---

#### Summary

The goal of this article was to measure what one AMD Instinct MI300X does serving Gemma 4 E2B under
vLLM on ROCm. The key to the solution was making the load generator independent of the serving
container and giving every cell a prompt set no other cell had seen. The measured results were:

- **10,293.6 output tokens/sec** at 64 concurrent streams and 128-token context, 30.2x the
  single-stream rate, at **$0.054 per million output tokens**.
- **The KV pool is never the constraint.** 9,026,017 resident tokens against a peak workload demand
  of 524,288 — 5.8% occupancy. Throughput at 8K context flattens on prefill, not memory.
- **18,443 bytes/token of KV**, matching this repo's geometry-derived 18,432 B to 0.06%, and
  therefore not reproducing the ~9.6 KiB an earlier NVIDIA L4 run reported.
- **Prefix caching inflated an uncontrolled sweep by up to 2.25x** at 8192 context and 1.00x at 128,
  and dropped the worst-cell spread from 51.1% to 8.4% once seeds were made unique.
- **155.04 GiB of the card's 191.69 GiB goes to KV cache** — 80.9% — for a 2B-class checkpoint.

Scope: one droplet, `debian-gpu-mi300x1-192gb-devcloud-atl1` in DigitalOcean's `atl1` region, one
MI300X at tensor-parallel 1, vLLM `0.29.1rc1.dev187+gaf1c01499.rocm100` on
`vllm/vllm-openai-rocm:nightly-rocm100`, bf16 with `quantization=None` and the `TRITON_ATTN` backend,
`--max-model-len 32768` and `--gpu-memory-utilization 0.90`. Twelve cells, three repeats each,
median reported, worst-cell coefficient of variation 8.4%. The load generator ran on the same host as
the server. No cell was compared against a run on other hardware.

The strategy for using MCP for AMD GPU benchmarking was validated with an incremental step by step
approach.
