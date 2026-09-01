---
title: "Three Runtimes, One T4G: vLLM is 3.2x Faster to Decode and 7x Slower to Boot"
published: false
description: "vLLM, JAX and PyTorch serving the same Gemma 4 checkpoint on the same AWS G5g GPU, on one harness and one statistic. The decode ranking reverses on boot time — and five claims died on the way."
tags: aws, machinelearning, benchmarking, python
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-pytorch-g5g-2b/devto-cover.jpg
---

*Three inference runtimes, one `google/gemma-4-E2B-it` checkpoint, one NVIDIA T4G. Every
earlier comparison in this family used three harnesses computing three different statistics
on three different days, so this one rebuilds the harness first. The decode spread is 3.2x.
The ranking then reverses on boot and revision time. Along the way five claims were
falsified, three of them mine.*

https://github.com/xbill9/gemma4-dev

| | |
|---|---|
| Model | `google/gemma-4-E2B-it` |
| Hardware | `g5g.2xlarge` — Graviton2 + 1x NVIDIA T4G (SM **7.5**, 15,360 MiB) |
| Runtimes | vLLM v0.27.2rc0 · JAX · PyTorch + transformers (torch 2.12.0+cu132) |
| Method | one harness, one statistic, same AZ, same day, 3 repeats per cell |
| Result | decode **32.53 / 12.69 / 10.24** tok/s · cold boot **1417.8 / 242.2 / 195.2** s |

---

Three rigs in this monorepo serve the same 2B checkpoint on the same chip. Only the runtime
changes. That should make them the cleanest possible A/B, and for months it did not, because
each rig measured itself with its own harness and quoted its own number.

This is the run that fixed that. The comparison is the small part; **the interesting part is
what had to be wrong first.**

## The harness could not measure vLLM at all

The sweep script read its throughput figure out of the response body:

```python
"decode_tps": usage.get("decode_tokens_per_second", 0.0),
```

`usage.decode_tokens_per_second` is a field **our own servers invent**. vLLM does not emit
it, and neither does anything else. So the harness could not be pointed at the vLLM rig, and
the three-way comparison had never actually been run — it existed as prose comparing numbers
that came from different tools.

Re-running a rig does not fix that. Only a common statistic does.

## The common statistic is client-side inter-token latency

Every OpenAI-compatible server streams, so the portable measurement is the gap between
tokens on the wire:

```python
ap.add_argument("--decode-source", choices=("auto", "usage", "stream", "both"),
                default="auto")
```

`stream` uses **`vllm bench serve`'s exact TPOT definition** — `(latency - ttft) / (output_len - 1)`
— so a number from this harness is directly comparable to that tool's published figures.
`auto` probes the endpoint once and picks: `both` where the server emits its own gauge,
`stream` where it does not.

## The calibration offset is per-rig and must never be borrowed

`both` runs each repeat twice, once non-streaming and once streamed, which measures the
offset between the two statistics rather than assuming it:

| rig | server gauge | client stream | stream/gauge |
|---|---:|---:|---:|
| JAX | 12.962 | 12.687 | **0.9799** |
| PyTorch | 10.814 | 10.243 | **0.9543** |

**2.0% against 4.6% on the same day, same instance shape.** Converting one rig's number into
the other's statistic with a borrowed ratio would inject a 2.6% error into a comparison whose
smallest interesting gap is 24%. This is why the cross-rig table below is built from `stream`
throughout rather than from each rig's native gauge.

## One rig could not resolve its own experiments

The JAX server emitted its decode gauge like this:

```python
f'tpu_jax_decode_tokens_per_second{{model="{MODEL_ID}"}} {METRICS["last_tokens_per_second"]:.1f}',
```

At ~13 tok/s, one decimal is **0.78% resolution**. Every sweep that rig had ever produced
showed all three repeats of a cell as byte-identical — 12.8, 12.8, 12.8 — which reads as
perfect reproducibility and is really the measurement floor.

That rig had been used to argue about 2% effects it could not see. Two characters fixed it.
The first run afterwards reads **12.962**, where before it would have said 13.0.

## Deploy: three rigs, one AZ, one day

Spot capacity for the whole g5g family was exhausted across all four availability zones
repeatedly, so the launcher cycles them with a 60-second backoff:

```
[12:16:23] round 5 us-east-1c: InsufficientInstanceCapacity
[12:17:19] us-east-1a: Launching i-0ffdb819663bacc77 (g5g.2xlarge, spot, 1x T4G)
```

All three landed in `us-east-1a` within hours of each other. Note that AWS's error text names
the *other* zones as available in every case — that is on-demand boilerplate and tells you
nothing about spot.

## Decode at concurrency 1

```bash
python3 sweep.py --base http://<ip>:8000/v1 --out benchmarks/runs/<run> \
  --contexts 64,512,1024,2048,3072,3800 --outputs 32,128 --repeats 3 \
  --decode-source both
```

| | runtime | decode tok/s | % of ceiling | vs PyTorch | cells |
|---|---|---:|---:|---:|---:|
| 🥇 | vLLM v0.27.2rc0 | **32.53** | 53.0% | **3.18x** | 12/12 |
| 🥈 | JAX | 12.69 | 20.7% | 1.24x | 10/12 |
| 🥉 | PyTorch + transformers | 10.24 | 16.7% | 1.00x | 10/12 |

The ceiling is arithmetic, not a measurement: E2B streams 4.514 GB of weights per decode step
against a measured 277 GB/s, giving 16.30 ms and **61.4 tok/s**. The PLE table is excluded
because it is a gather, never a matmul — quantising it from 9.257 GB to 5.752 GB moved decode
by 0.00 tok/s across three cells.

**All three are far below that ceiling, so none is bandwidth-bound at batch 1.** The PyTorch
profile shows why: ~5,650 kernel launches per step at 1-3 us each, on a chip whose launch
overhead is 5-10 us.

## The number that was almost a headline

Time-to-first-token looked like the story of the run:

| input tok | vLLM | JAX | PyTorch |
|---:|---:|---:|---:|
| 92 | 103 ms | 225 ms | 164 ms |
| 1,259 | 118 ms | 1,615 ms | 657 ms |
| 3,746 | **178 ms** | **5,352 ms** | **2,339 ms** |

A 29x advantage, far bigger than the 3.2x on decode. It is also **impossible**. Prefill at
4,630 tokens is roughly 17 TFLOP against a T4G's realistic 20-30 TFLOP/s, so it cannot finish
in 116 ms.

It did not. vLLM ships `enable_prefix_caching=True`, and its own metrics for that run say so:

```
vllm:prefix_cache_queries_total  102898
vllm:prefix_cache_hits_total      97440
```

**A 94.7% hit rate.** vLLM genuinely prefilled 5.3% of the tokens it was sent, because the
harness reused one prompt for a cell's warm-up and all three repeats. Neither sibling has a
prefix cache, so both paid full prefill every time. The 29x was the harness, not the engine.

The fix is a nonce placed **first** in the prompt — a shared prefix is exactly what the cache
keys on, so a trailing nonce would not have defeated it. That property is now a test, because
it is the kind of thing a later cleanup quietly reverses.

## What the prefill data does support

Strip the contaminated column and a real result remains. Neither of the other two runtimes
caches prefixes, both saw identical prompts, both ran the same harness on the same instance
shape:

| | TTFT slope | at 3,746 tokens |
|---|---:|---:|
| JAX | **1.403 ms/token** | 5,352 ms |
| PyTorch | 0.595 ms/token | 2,339 ms |

**JAX prefills 2.4x slower than PyTorch**, consistent across all five shared context lengths.
On any interactive workload with real context, that dominates the user-visible latency — and
it is larger, in the wrong direction, than the 1.24x decode advantage the same rig enjoys.

## Boot time reverses the ranking

Nine cold boots, three per runtime, plus nine warm reloads. Start line is the moment
`run_instances` returns an id; capacity wait is excluded because it measures AWS, not the rig.

| | runtime | cold boot | spread | warm reload | cold/warm |
|---|---|---:|---:|---:|---:|
| 🥇 | PyTorch | **195.2 s** | 11.8% | **24.5 s** | 8.0x |
| 🥈 | JAX | 242.2 s | 11.5% | 74.1 s | 3.3x |
| 🥉 | vLLM | **1417.8 s** | 12.6% | 264.3 s | 5.4x |

**vLLM takes 23m 38s to serve, from a prebuilt AMI that downloads nothing.** PyTorch installs
its runtime from wheels *and* pulls a 9.5 GB checkpoint over the network, and is still 7.3x
faster.

Boot variance is 11.5% / 11.8% / 12.6% — too consistent across three different runtimes to be
a runtime property. That is about 6x the decode noise floor, which makes a single boot
measurement nearly worthless and is why this section has repeats.

## Health 200 is not "ready to serve"

The harness records two stop lines, and the gap between them is a finding:

| runtime | first completion, cold | warm |
|---|---:|---:|
| vLLM | 0.5 s | 0.2 s |
| PyTorch | 1.0 s | 0.7 s |
| JAX | **22.9 s** | **9.2 s** |

JAX returns health 200 and *then* compiles XLA per shape bucket on the first real request.
Quoting health alone understates its time-to-serving by 22 seconds, and the cost does not
disappear when warm. vLLM is the mirror image: slowest to boot, fastest first token, because
graph capture is paid before the port binds.

## Revision cost is where it inverts hardest

| runtime | to change serving code |
|---|---|
| PyTorch | ship 3 files (16 KiB) over SSM, restart — **25 s** |
| JAX | same mechanism, plus 9.2 s of compile — **83 s** |
| vLLM | **no deploy path exists**: rebuild from source, ~67 min, and reapply an out-of-tree Turing patch |

vLLM's 264 s warm figure is a `systemctl restart`, not a code change, so it flatters the
comparison. **vLLM wins decode 3.2x and loses the iteration loop by 3-100x** depending on what
you are changing.

## Five theories about 546 seconds

vLLM's cold boot is dominated by one number: weight loading, 468-561 s across four
measurements. Explaining it took five attempts, four of which were wrong.

1. *"`g5g.2xlarge` needs no swapfile and buys that time back."* The rig's own docs. Falsified
   by the three-boot campaign: 23m 38s, not the ~4 minutes claimed.
2. *"A bigger host will not fix it."* My replacement text. Retracted the same day — no large
   host had been measured, so the claim had no standing.
3. *"A larger host would very plausibly fix it."* The retraction's own replacement. Falsified
   by one `g5g.4xlarge` boot: available RAM 11.19 -> **26.49 GiB**, weight loading
   546 -> **468 s**, a 14% move, and total boot **-4.7%, inside the noise band.**
4. *"It is the loader — vLLM's log says auto-prefetch is disabled on EXT4."* Falsified by a
   within-box A/B: **76.13 s as shipped against 75.12 s with `--safetensors-load-strategy=prefetch`.**
   That is -1.3%, i.e. nothing.

What that last run *did* find is the useful part:

| | weight load | n |
|---|---:|---|
| cold boot, fresh instance | **468-561 s** | 4 |
| warm restart, same box | **32-76 s** | 3 |

Same volume, same filesystem, same engine. The only difference is that the blocks had been
read once. 9.54 GiB in 468 s is **20.9 MB/s** — absurd for gp3 steady state, entirely ordinary
for first-touch reads against a snapshot-backed volume.

Theory five is **EBS lazily hydrating the volume from the AMI snapshot**, and it is written
down as untested. Given the strike rate, it does not get promoted by reasoning. The test is one
`dd` of the volume before starting the engine.

## A measurement floor that looked like reproducibility

The first boot campaign was discarded and re-run. Two independent instances had reported
**214.4 s and 125.1 s, identical to the tenth** — which reads as extraordinary consistency and
is really a 5-second poll interval quantising two similar boots onto the same tick.

The data was not fake; log wall-clock confirms 215 s and 216 s. But publishing three identical
numbers would have presented the measurement floor as a result. Health polling went to 0.5 s,
and the very next pair of boots came in at 216.90 s and 193.86 s — an **11.9% spread that the
coarse harness had hidden completely.**

## Cost

The whole exercise — roughly 15 instances, ~3.5-4 instance-hours, a serving sweep, three
cross-rig runs, nine boots and three A/B restarts — came to **under $2**.

| | $/hr |
|---|---:|
| `g5g.2xlarge` on-demand | **$0.556** |
| spot, measured across four AZs | $0.3813 - $0.4416 |

That 26-46% premium is the entire spot-versus-on-demand decision on this hardware, which is
to say there is not one: try spot, fall back, keep working. The automatic fallback cost about
$0.24 across a nine-boot campaign.

**That is what made the method possible, not just affordable.** Discarding a completed campaign
over a poll-interval bug cost twenty minutes and pennies. On hardware where a run is expensive,
the same discovery argues for shipping the numbers with a caveat instead — and "214.4 twice"
goes into a report as reproducibility.

## Teardown

Every instance is terminated as soon as its artifacts are captured; there is no built image to
lose, only a pip install and a model cache.

```
$ aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[?starts_with(InstanceType,`g5g`)].InstanceId' --output text
(no output)
```

## What this does not cover

**Everything above is concurrency 1, which makes it a latency comparison and not a serving
one.** Continuous batching is vLLM's whole value proposition and it is untested here. Only
vLLM can serve concurrently at all today: the PyTorch server holds an `asyncio.Lock` with the
comment *"one GPU, one process -> serialize requests"*, and the JAX rig has no batching
machinery whatever.

The engine-level batch sweep suggests what is on the table — batch 8 reaches 84.16 tok/s for
an extra 0.258 GB, with per-step time growing 2.0% across an 8x batch — but that never leaves
the engine.

Three further gaps, stated once: the JAX leg ran its shipped quantised configuration against
two dense runtimes; no output-quality axis was measured at all, on a comparison where one
runtime uses a deliberately lossy LM head; and every TTFT figure was taken before the
prompt-uniqueness fix, so only the JAX-versus-PyTorch half of that table is sound.

## The short version

The goal of this article was to compare three inference runtimes on identical silicon without
the harness being a variable. The key to the solution was a single client-side statistic every
OpenAI-compatible server can produce. The measured results were:

- **Decode at concurrency 1: vLLM 32.53, JAX 12.69, PyTorch 10.24 tok/s** — 53%, 21% and 17%
  of a 61.4 tok/s bandwidth ceiling, so none is bandwidth-bound.
- **The ranking reverses on lifecycle.** Cold boot 195.2 s for PyTorch against 1417.8 s for
  vLLM; revision 25 s against a from-source rebuild.
- **JAX prefills 2.4x slower than PyTorch**, 1.403 against 0.595 ms/token.
- **A 94.7% prefix-cache hit rate** turned a 29x TTFT result into a harness artifact.
- **Calibration offsets are per-rig** — 0.9799 and 0.9543 — and are not transferable.
- **Boot variance is ~12%** on this platform regardless of runtime.

Scope: one `g5g.2xlarge` per runtime in `us-east-1a` on 2026-08-31, three repeats per cell and
three repeats per boot, mixed spot and on-demand with the market recorded per run. The vLLM leg
ran `max_model_len` 16384 against 4096 for the other two, and its quantisation configuration
differed from JAX's; both are named where they matter. Decode is unaffected by the prefix-cache
issue, which changes prefill only.

The strategy for using MCP for multi-runtime comparison was validated with a incremental step
by step approach.
