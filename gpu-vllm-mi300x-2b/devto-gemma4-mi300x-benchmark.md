---
title: "Gemma 4 E2B on One MI300X: What 155 GiB of KV Cache Buys, and What It Doesn't"
published: false
series: Gemma4
description: "A twelve cell serving sweep of Gemma 4 E2B on one AMD Instinct MI300X under vLLM on ROCm, driven through a tag scoped Python MCP server. The engine allocates nine million tokens of KV cache and the heaviest cell in the grid uses 5.8 percent of it. Two sweeps were thrown away first, because the benchmark was measuring the prefix cache."
tags: amd, vllm, rocm, benchmarking
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-mi300x-2b/devto-benchmark-cover.47634b3a.jpg
---

This article provides a step by step serving benchmark of Gemma 4 E2B on a single AMD Instinct MI300X hosted GPU enabled system. A suite of Python MCP tools is built to simplify management of the vLLM deployment, and the sweep itself is driven through the same server.

https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b

The companion piece on the same droplet inventories the card and measures what its matrix cores execute, and states plainly that it measured no serving throughput. This is that number. The workstation writing it has no AMD GPU and never will; everything that reaches the hardware goes through tag scoped MCP tools or explicit SSH.

Three results came out of the grid, and only the first was the one I set out to measure. The card sustains **10,293.6 output tokens a second** across 64 streams at short context, at **$0.054 per million output tokens**. The KV pool it advertises so loudly is never the constraint — the heaviest cell in the grid wants 5.8 percent of it — so the ceiling at long context is prefill, which inverts the sizing rule every TPU rig in this monorepo runs on. And two full sweeps went in the bin before either number was trustworthy, because `vllm bench serve` seeds its prompts at zero and this deployment caches prefixes.

#### Prerequisites

- An AMD Developer Cloud account with a GPU droplet already created and serving. `devcloud.amd.com`
  is DigitalOcean underneath — same v2 API, same droplet ids — and the token comes from the
  **My AMD Team** account, not a personal DigitalOcean one.
- The droplet tagged. Every lookup is scoped by `tag_name`, so an untagged droplet is invisible to
  the server and a mistyped id cannot power cycle an unrelated machine.
- `DIGITALOCEAN_ACCESS_TOKEN` in the environment or in a mode 0600 `.env`. Never in `tpu.env`,
  which is committed.
- `vllm` already up and answering. Getting there — vetting the ROCm image before a 35 GB pull, and
  the vendor build that cannot load Gemma 4 at all — is the subject of a different article.
- Python 3 with `httpx` and `python-dotenv` in the system interpreter. No virtualenv; `.mcp.json`
  launches the server with a bare `python3`.

#### The Box, Briefly

```
list_droplets
```

```
📡 1 droplet(s) tagged `gemma`.

| Name | ID | Status | Size | Region | Public IPv4 |
| --- | --- | --- | --- | --- | --- |
| `debian-gpu-mi300x1-192gb-devcloud-atl1` | 601142018 | active | gpu-mi300x1-192gb-devcloud | atl1 | 165.245.134.217 |
```

One card, `gfx942`, 191.69 GiB, billing at $1.99 an hour read from `droplet.size.price_hourly` rather than a price page. It presents as an SR-IOV virtual function, which reads alarmingly like a partition and is not one: all 304 compute units and the whole 191.69 GiB are there.

There is no `create` tool and no `destroy` tool in this server, deliberately. Both are dollar per hour decisions and they stay a human step in the console. Worth repeating because it catches people: powering a droplet off does not stop DigitalOcean billing it. Only destroying it stops the meter.

#### Where the 191.69 GiB Actually Goes

vLLM prints its whole memory budget once, at startup, and then never again.

```
docker logs vllm 2>&1 | grep -E 'Available KV cache memory|GPU KV cache size|Actual usage is'
```

```
Available KV cache memory: 155.04 GiB
GPU KV cache size: 9,026,017 tokens, Maximum concurrency for 32,768 tokens per request: 275.45x
Free memory on device (191.36/191.69 GiB) on startup. Desired GPU memory utilization is (0.9,
172.52 GiB). Actual usage is 11.88 GiB for consumed memory (weights + non-torch), 5.59 GiB for
peak activation, and 3.9 GiB for CUDAGraph memory.
```

| Component | GiB | Share |
| --- | ---: | ---: |
| KV cache | 155.04 | 80.9% |
| Weights + non-torch | 11.88 | 6.2% |
| Peak activation | 5.59 | 2.9% |
| CUDA graph pool | 3.90 | 2.0% |
| Unallocated below `--gpu-memory-utilization 0.90` | 15.28 | 8.0% |

A 2B class checkpoint on a 192 GiB card spends four fifths of the card on KV cache. Hold that number; the rest of the article is about how little of it gets used.

#### One Division Settles an Open Question

Divide the engine's own pool by the engine's own token count and you get **18,443.7 bytes per token**. That is arithmetic on two measured inputs rather than a measurement, but both inputs come off the same log line.

This repository's `MODELS.md` derives **18,432 bytes per token** for E2B from the checkpoint's layer geometry — sliding attention layers 256 wide, full attention layers 512 wide — and cross checks it exactly against two TPU runs. The ROCm figure lands **0.06 percent off it**.

That matters because the same file carries an open discrepancy. An NVIDIA L4 under stock vLLM 0.28.0 reported 9,622 bytes per token, a 1.92x gap, and the leading explanation was that vLLM v1 charges sliding window layers only their **window** rather than the full context. If that were an engine level policy it would apply here too, on vLLM 0.29.1, on the same v1 engine. It does not. The hypothesis does not survive a third stack, and whatever produced the L4 number is narrower than "the vLLM path".

It does not explain the L4 figure, and nothing here shows that figure wrong. One card, one run, recorded in `MODELS.md` beside the other two.

#### The Load Generator Must Not Be Able to Touch the Card

The sibling vLLM rigs expose a `run_vllm_benchmark` tool. This one had none, so building it was the first job, and one decision inside it is worth stating.

```
docker run --rm --net host \
  -v /opt/hf-cache:/root/.cache/huggingface -v /dev/shm:/dev/shm --shm-size 8g \
  --entrypoint vllm \
  vllm/vllm-openai-rocm:nightly-rocm100 bench serve ...
```

No `--device /dev/kfd`, no `--device /dev/dri`, no `--group-add`. The bench client is an HTTP load generator that needs a tokenizer and nothing else, so with no device mapped in it **cannot** touch the card, which is what makes it safe to run beside a live server. It costs host CPU, not card time.

`--entrypoint vllm` is set unconditionally, unlike the serving path in this rig, which branches on the image because `vllm/vllm-openai-rocm` declares `ENTRYPOINT ["vllm","serve"]` while AMD's own `rocm/vllm` images declare none. Overriding outright works for both. Without it, the official image reads `bench` as a model id.

#### The Benchmark Was Measuring the Prefix Cache

This is the part worth carrying to any other engine, and it cost two complete sweeps.

`vllm bench serve` derives its random prompts from `--seed`, which defaults to **0**. vLLM's default configuration has `enable_prefix_caching=True`, and this deployment left it on because that is how the rig actually serves. Two runs that share a seed therefore generate the *same prompts*, and the second reads the first's cache entries and reports a speedup that is the cache rather than the card.

It is worse than per repeat. Cells at the same context length draw from the same prompt pool, so `c4-in128` replayed `c1-in128`. Nothing errors, nothing warns, the throughput simply looks good.

Measured inside a single sweep, where repeat 1 used seeds an earlier sweep had already served and repeats 2 and 3 did not, all three inside the same minute so card warm up is not the explanation:

| Context | Clients | Warm tok/s | Cold tok/s | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 340.7 | 340.5 | 1.00x |
| 128 | 64 | 10,849.2 | 10,496.2 | 1.03x |
| 1,024 | 16 | 3,415.4 | 2,891.3 | 1.18x |
| 1,024 | 64 | 8,930.4 | 6,252.7 | 1.43x |
| 8,192 | 16 | 1,961.5 | 870.2 | 🥇 2.25x |
| 8,192 | 64 | 2,095.8 | 948.9 | 2.21x |

The gradient is what makes this believable rather than a story. A prefix cache saves prefill, prefill's share of the work grows with context, so the effect should be nothing at 128 tokens and large at 8192. It is exactly that: **1.00x and 2.25x**. The 1.00x row is the control, and without it the 2.25x would be an anecdote.

The fix is one line — give every cell and every repeat a seed no other run in the sweep uses:

```python
def _seed(base: int, index: int, rep: int) -> int:
    return base + index * 100 + rep
```

| Sweep | Worst cell spread | Cells under 1% spread |
| --- | ---: | ---: |
| Seeds 0, 1, 2 per cell | 51.1% | 4 of 12 |
| 🥇 Unique seed per cell and repeat | 8.4% | 9 of 12 |

A 51 percent coefficient of variation is not noise. It is two different experiments averaged together. The contaminated run is kept in the repository next to the clean one, because the difference between them is itself the measurement.

#### The Grid

Four client counts against four context lengths, 128 output tokens throughout, `ignore_eos` so every request produces exactly 128 tokens instead of stopping early, and `num_prompts = max(8, 2 x clients)`. The shape matches `tpu-vllm-v5e1-2b` and `tpu-vllm-v6e1-2b` so the grid is comparable even though the numbers are not.

```
python3 benchmarking_suite.py --droplet debian-gpu-mi300x1-192gb-devcloud-atl1 \
  --run-id 2026-09-16-vllm-sweep-mi300x --repeat 3 --seed-base 5000
```

```
16 cells, 12 runnable, 4 infeasible at max_model_len 32768
```

The 32768 context row cannot exist, because 32768 + 128 is more than 32768. Those four cells are recorded `infeasible` with the reason rather than dropped — a missing cell is indistinguishable from one nobody ran, and the report schema has a status field for exactly this.

#### The Result

Median of three repeats per cell, every cell a prompt set no other cell saw.

Aggregate output tokens a second:

| context ↓ / clients → | 1 | 4 | 16 | 64 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 340.4 | 1,118.1 | 3,583.6 | 🥇 **10,293.6** |
| 1,024 | 305.6 | 967.3 | 2,681.4 | 5,447.6 |
| 8,192 | 193.2 | 449.3 | 711.2 | 725.3 |

Median time to first token, milliseconds:

| context ↓ / clients → | 1 | 4 | 16 | 64 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 12.3 | 21.1 | 49.7 | 126.4 |
| 1,024 | 30.1 | 68.1 | 197.1 | 568.8 |
| 8,192 | 164.3 | 497.3 | 1,387.0 | **4,640.1** |

Per stream tokens a second, which is what one user feels:

| context ↓ / clients → | 1 | 4 | 16 | 64 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 349.7 | 291.5 | 244.5 | 196.5 |
| 1,024 | 326.8 | 278.6 | 220.8 | 136.4 |
| 8,192 | 255.1 | 198.0 | 84.8 | **19.1** |

#### Is That Difference Real?

Worst cell spread over three repeats is **8.4 percent**, and nine of the twelve cells are under 1 percent. Both loose cells are at 8192 context, which is where a single slow prefill moves a median furthest.

So read the 30x scaling at short context and the flat line at long context as real, and anything under about 9 percent as nothing. The 2 percent between 711.2 and 725.3 tok/s is inside the noise and is being read as "flat", not as a rise.

#### Short Context Scales, Long Context Stops, and Memory Is Not Why

Going from 1 client to 64 at 128 token context is **30.2x the throughput**, and a single stream still feels quick at 196.5 tokens a second with the card fully loaded. That is the regime agent traffic lives in, and the card barely notices it.

At 8192 tokens the same step from 16 clients to 64 is **2 percent**, while median time to first token goes from 1,387 ms to 4,640 ms and p99 reaches 8,746 ms. Past 16 clients at long context you are buying latency, not throughput.

On the TPU rigs in this monorepo that collapse is a KV wall, and the sizing rule is `clients x context < KV pool`. Here that rule never binds:

```
64 clients x 8,192 tokens = 524,288 KV tokens wanted
resident pool             = 9,026,017 tokens
occupancy                 = 5.8%
```

The pool is seventeen times larger than the heaviest cell's peak demand. Nothing is being evicted and nothing is queueing for blocks. What saturates is prefill: that same cell moves **47,142 total tokens a second** counting prompt tokens, against 46,231 at 16 clients — the card is already doing all the prefill work it can, and the extra 48 clients only lengthen the queue.

So on this part the sizing question is inverted. The KV budget is not the thing to plan around, and the operating point is set by how much prefill latency you are willing to pay for.

| workload | context | clients | expected |
| --- | ---: | ---: | --- |
| Interactive chat and agent turns | ≤1K | **64** | 5,448–10,294 tok/s, 126–569 ms TTFT |
| Latency sensitive interactive | ≤1K | **16** | 2,681–3,584 tok/s, 50–197 ms TTFT |
| RAG and long documents | 8K | **16** | 711 tok/s, 1,387 ms TTFT |
| Long context interactive | 8K | **≤4** | 449 tok/s, 497 ms TTFT, 198 tok/s per stream |

The 8192 by 64 cell is in the report and deliberately not in that table. It is 2 percent more throughput for 3.3 times the time to first token.

#### And Price/Performance?

Cost per million output tokens is the hourly rate over the measured rate, which is arithmetic on a measured price and a measured throughput.

| Operating point | tok/s | $/M output tokens |
| --- | ---: | ---: |
| 🥇 128 context, 64 clients | 10,293.6 | **0.054** |
| 🥈 128 context, 16 clients | 3,583.6 | 0.154 |
| 1,024 context, 64 clients | 5,447.6 | 0.102 |
| 8,192 context, 16 clients | 711.2 | 0.777 |
| 128 context, single stream | 340.4 | 1.624 |
| 8,192 context, single stream | 193.2 | 2.862 |

Serving one stream at a time costs **30x more per token** than serving 64, on the same card at the same hourly rate. Batching is a far bigger lever than any hardware choice here, and the spread is wider than on the small cards precisely because there is so much headroom left to fill.

Compute only. No storage, no transfer, and no idle time — a card at $1.99 an hour producing nothing costs exactly what one producing 10,293.6 tokens a second costs.

#### What Was Controlled

| | How it was held |
| --- | --- |
| Engine | one container, up throughout, never restarted between cells |
| Config | one `docker run`, read back from `docker inspect` into the report rather than copied from `tpu.env` |
| Prompts | vLLM's own `random` dataset, `ignore_eos`, exactly 128 output tokens per request |
| Prompt identity | every cell and every repeat given a seed no other run in the sweep used |
| Load generator | same host, own container, **no GPU device mapped in** |
| Repeats | three per cell, median throughput run reported, spread recorded beside it |
| Infeasible cells | recorded with the reason, not dropped |

The prompt identity row is the one worth dwelling on. Everything else on that list is ordinary hygiene. That one is the difference between the numbers above and numbers up to 2.25x higher that would have looked entirely plausible.

#### What This Does Not Cover

**No cross hardware comparison, and that is a deliberate omission rather than an oversight.** Nothing else in this monorepo serves this checkpoint on AMD, so there is no A/B twin. The TPU rigs differ in chip, runtime, control plane and cloud at once; the CUDA rigs differ in chip, cloud and instance shape. Differencing a number here against one of those and reading the result as a hardware finding would be four confounded variables reported as one.

**bf16 throughout.** The engine reports `quantization=None`. fp8 `e4m3fnuz` is the only format on this card faster than bf16 and it is worth 1.77x on a matmul, but that is a GEMM ratio measured in the companion article, not tokens per second, and no fp8 serving arm was run.

**The attention backend was not varied.** The engine selected `TRITON_ATTN` and it was left there. `VLLM_ROCM_USE_AITER=1` is an untested lever.

**Contexts beyond 8192 were not sampled at all.** The 32768 row is infeasible at this `--max-model-len`, and nothing in between was measured, so the prefill ceiling is bounded from one side only.

**The first sweep of the day is unexplained.** On a server idle for three hours it read 4,202 tok/s on a cell that later read 10,294 — a 2.4x gap on a cell where prefix caching is worth 1.03x. One observation, not a characterised effect, and the console output is archived in the repository rather than written up as a finding. The practical version is: discard your first sweep.

#### Summary

The goal of this article was to measure what one AMD Instinct MI300X does serving Gemma 4 E2B under vLLM on ROCm. The key to the solution was making the load generator physically unable to touch the card, and giving every cell a prompt set no other cell had seen. The measured results were:

- **10,293.6 output tokens a second** at 64 clients and 128 token context, 30.2x the single
  stream rate, at **$0.054 per million output tokens**.
- **The KV pool is never the constraint.** 9,026,017 resident tokens against 524,288 wanted by
  the heaviest cell, 5.8 percent. Long context flattens on prefill, not memory, which inverts
  the sizing rule the TPU rigs run on.
- **18,443.7 bytes per token of KV**, 0.06 percent off this repository's geometry derivation,
  which refutes the engine policy explanation offered for an earlier NVIDIA L4 reading.
- **Prefix caching inflated an uncontrolled sweep by up to 2.25x** at 8192 context and 1.00x at
  128, and unique seeds took the worst cell spread from 51.1 percent to 8.4 percent.
- **155.04 GiB of the card's 191.69 GiB is KV cache**, 80.9 percent, for a 2B class checkpoint.

Scope: one droplet, `debian-gpu-mi300x1-192gb-devcloud-atl1` in DigitalOcean's `atl1` region, one MI300X at tensor parallel 1, vLLM `0.29.1rc1.dev187+gaf1c01499.rocm100` on `vllm/vllm-openai-rocm:nightly-rocm100`, bf16 with the `TRITON_ATTN` backend, `--max-model-len 32768` and `--gpu-memory-utilization 0.90`. Twelve cells, three repeats each, median reported, worst cell spread 8.4 percent. The load generator ran on the same host as the server. No cell was compared against a run on other hardware.

The strategy for using MCP for AMD GPU serving benchmarks was validated with an incremental step by step approach.
