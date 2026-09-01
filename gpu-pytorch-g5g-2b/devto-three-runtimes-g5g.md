---
title: "Comparing Three Gemma 4 Deployments on One NVIDIA T4G: what the runtime changes and what it doesn't"
published: false
description: "vLLM, JAX and PyTorch serving the same Gemma 4 E2B checkpoint on the same AWS G5g GPU, driven by one harness and one statistic. The decode ranking reverses on boot time, and five claims were falsified on the way."
tags: aws, machinelearning, benchmarking, python
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-pytorch-g5g-2b/devto-cover.jpg
---

This article provides a step by step comparison of three Gemma 4 deployments on a single AWS
hosted GPU enabled system. A suite of Python MCP tools is built to simplify management of each
deployment, and one benchmark harness is shared across all three so that the runtime is the
only variable.

https://github.com/xbill9/gemma4-dev

## Where Do I Start?

Three rigs in this monorepo serve `google/gemma-4-E2B-it` on an AWS G5g instance. One runs
vLLM, one runs a pure JAX port, one runs PyTorch with transformers. The hardware is identical
and only the runtime slot moves, so this should be the cleanest A/B available.

For months it was not, because each rig measured itself with its own harness and quoted its
own number. Three harnesses computing three statistics is not a comparison.

## At This Point You Should Have...

- An AWS account with G-family quota in `us-east-1`. Each `g5g.2xlarge` is 8 vCPU, so 16 vCPU
  of spot quota runs two at once.
- A subnet, a security group opening TCP 8000, and an instance profile carrying
  `AmazonSSMManagedInstanceCore` plus read on the Hugging Face token secret.
- A Hugging Face token in Secrets Manager. It is fetched at boot into a root-only
  `EnvironmentFile` and never appears in user data.
- `boto3` and the standard credential chain. No AWS CLI shell-outs, no inbound SSH rule, and
  no private key anywhere in the flow.

## The Hardware Under All Three

| | |
|---|---|
| Instance | `g5g.2xlarge` — 8 vCPU, 16 GiB host |
| Host CPU | AWS Graviton2, aarch64 |
| GPU | 1x NVIDIA T4G, Turing, SM 7.5 |
| GPU memory | 15,360 MiB per `nvidia-smi`; AWS lists 16,384 nominal |
| Model | `google/gemma-4-E2B-it` |

G5g is the only family AWS ships that puts an NVIDIA GPU behind a Graviton host, which makes
it the only place to get aarch64 and compute capability 7.5 together.

## Why the Comparison Did Not Exist

The sweep script read its throughput figure straight out of the response body:

```python
"decode_tps": usage.get("decode_tokens_per_second", 0.0),
```

`usage.decode_tokens_per_second` is a field our own servers invent. vLLM does not emit it, and
neither does anything else, so the harness could not be pointed at the vLLM rig at all. The
three-way comparison had never actually been run.

Re-running a rig does not fix that. Only a common statistic does.

## One Harness, One Statistic

Every OpenAI-compatible server streams, so the portable measurement is the gap between tokens
on the wire.

```bash
python3 sweep.py --help | grep -A2 decode-source
```

```
  --decode-source {auto,usage,stream,both}
                        where the decode figure comes from; see the module
                        docstring
```

The `stream` path uses `vllm bench serve`'s exact TPOT definition,
`(latency - ttft) / (output_len - 1)`, so a number from this harness is directly comparable to
that tool's published figures. `auto` probes the endpoint once and picks `both` where the
server emits its own gauge, `stream` where it does not.

## Is the Calibration Transferable?

No, and that is worth a measurement rather than an assumption. Running `both` measures each
rig's offset between the two statistics.

| rig | server gauge | client stream | stream/gauge |
|---|---:|---:|---:|
| JAX | 12.962 | 12.687 | 0.9799 |
| PyTorch | 10.814 | 10.243 | 0.9543 |

Two percent against 4.6 percent, on the same day and the same instance shape. Borrowing one
rig's ratio to convert the other's number would inject a 2.6 percent error into a comparison
whose smallest interesting gap is 24 percent. The cross-rig table below is therefore built
from `stream` throughout.

## A Gauge That Could Not See Its Own Experiments

The JAX server emitted its decode gauge with one decimal place:

```python
f'tpu_jax_decode_tokens_per_second{{model="{MODEL_ID}"}} {METRICS["last_tokens_per_second"]:.1f}',
```

At about 13 tok/s, one decimal is 0.78 percent resolution. Every sweep that rig had produced
showed all three repeats of a cell as byte-identical — 12.8, 12.8, 12.8 — which reads as
perfect reproducibility and is really the measurement floor. That rig had been used to argue
about two percent effects it could not see.

Two characters fixed it. The first run afterwards reads 12.962, where before it would have
said 13.0.

## Launching the First Rig

Capacity for the whole G5g family was exhausted across all four availability zones several
times, so the launcher cycles them with a sixty second backoff.

```
[12:16:23] round 5 us-east-1c: ❌ AWS InsufficientInstanceCapacity
[12:17:19] us-east-1a: ✅ Launching `i-0ffdb819663bacc77` (g5g.2xlarge, spot, 1x T4G) in `us-east-1`.
```

All three rigs landed in `us-east-1a` within hours of each other. Note that AWS names the
other zones as available in every one of those errors — that text describes on-demand capacity
and says nothing about spot.

## Does Torch Actually Have Kernels for This GPU?

A config flag being accepted proves nothing, so the probe runs a real matmul on the device.

```
verify_gpu_arch i-02e79988a6cbeecbf
```

```
NVIDIA T4G, 7.5, 15360 MiB
torch: 2.12.0+cu132
arch_list: ['sm_75', 'sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']
capability: (7, 5)
compute_dtype: float16
fp16 matmul ok: True

✅ torch reached the GPU and a real fp16 matmul executed.
```

The DLAMI's torch carries `sm_75`. Upstream PyPI aarch64 wheels do not, so a `pip install
torch` on this box would serve on CPU without saying so.

## Deploying the Serving Code

The payload is the rig's own source, shipped over SSM as a gzipped tarball because user data
caps at 16 KiB.

```
deploy_torch_server i-02e79988a6cbeecbf
```

```
✅ Deployed 3 files (16 KiB base64) to `i-02e79988a6cbeecbf`.

Payload root: `/home/xbill/gemma4-dev/gpu-pytorch-g5g-2b`
Build id: `060a572aeb55` — verify_model_health checks the running server reports this.
```

## Validating the Endpoint

A non-empty reply is not evidence of health. One sibling was once measured answering
`': ok: ok: ok…'`, so the check reads the server's own degenerate-response counter either side
of its probe, and compares the served build id against the local payload.

```
verify_model_health i-02e79988a6cbeecbf
```

```
✅ health=200 tokens=5 reply='ok'

- Degenerate (server's own verdict on the full text): **no**
- Build id served: `060a572aeb55`
- Build id matches the local payload (`060a572aeb55`).
```

## Running the Benchmark Sweep

The same command runs against all three rigs. Only the endpoint changes.

```bash
python3 sweep.py --base http://<ip>:8000/v1 --out benchmarks/runs/<run> \
  --contexts 64,512,1024,2048,3072,3800 --outputs 32,128 --repeats 3 \
  --decode-source both
```

```
decode-source: both -> both
ctx~512 out=32: in=633 out=32 decode=11.33 tok/s  e2e=10.06 tok/s  stream/usage=0.9616
ctx~2048 out=128: in=2501 out=89 decode=11.17 tok/s  e2e=9.46 tok/s  stream/usage=0.9540
ctx~3800 out=32: FAILED HTTP Error 400 {"detail":"prompt is 4630 tokens and the context
  bound is 4096, leaving no room to decode."}
```

Cells that cannot exist on the hardware are recorded as `infeasible` rather than dropped. An
absent cell is indistinguishable from an untried one, which is how a sweep overstates its own
coverage.

## Decode at Concurrency One

| | runtime | decode tok/s | % of ceiling | vs PyTorch | cells |
|---|---|---:|---:|---:|---:|
| 🥇 | vLLM v0.27.2rc0 | 32.53 | 53.0% | 3.18x | 12/12 |
| 🥈 | JAX | 12.69 | 20.7% | 1.24x | 10/12 |
| 🥉 | PyTorch + transformers | 10.24 | 16.7% | 1.00x | 10/12 |

## How Close Is That to the Hardware?

The ceiling is arithmetic, not a measurement. E2B streams 4.514 GB of weights per decode step
against a measured 277 GB/s, giving 16.30 ms per step and **61.4 tok/s**.

The PLE table is excluded from that figure because it is a gather and never a matmul.
Quantising it from 9.257 GB to 5.752 GB moved decode by 0.00 tok/s across three cells, which
is what confirms it never streams.

All three runtimes sit far below the ceiling, so **none of them is bandwidth-bound at batch
one**. The PyTorch profile shows why: about 5,650 kernel launches per step at one to three
microseconds each, on a chip whose launch overhead is five to ten.

## The Number That Was Almost a Headline

Time to first token looked like the result of the run.

| input tok | vLLM | JAX | PyTorch |
|---:|---:|---:|---:|
| 92 | 103 ms | 225 ms | 164 ms |
| 1,259 | 118 ms | 1,615 ms | 657 ms |
| 3,746 | 178 ms | 5,352 ms | 2,339 ms |

A 29x advantage, far larger than the 3.2x on decode. It is also impossible. Prefill at 4,630
tokens is roughly 17 TFLOP against a T4G's realistic 20 to 30 TFLOP/s, so it cannot finish in
116 ms.

It did not. vLLM ships `enable_prefix_caching=True`, and its own metrics say so:

```bash
grep -E "^vllm:prefix_cache_(queries|hits)_total" metrics.prom
```

```
vllm:prefix_cache_queries_total{engine="0"} 102898.0
vllm:prefix_cache_hits_total{engine="0"}     97440.0
```

**A 94.7 percent hit rate.** vLLM genuinely prefilled 5.3 percent of the tokens it was sent,
because the harness reused one prompt for a cell's warm-up and all three repeats. Neither
sibling has a prefix cache, so both paid full prefill every time.

The fix places a nonce first in the prompt, since a shared prefix is exactly what the cache
keys on and a trailing nonce would not have defeated it. That property is now a unit test.

## What the Prefill Data Does Support

Strip the contaminated column and a real result remains. Neither of the other two runtimes
caches prefixes, and both saw identical prompts.

| | TTFT slope | at 3,746 tokens |
|---|---:|---:|
| JAX | 1.403 ms/token | 5,352 ms |
| PyTorch | 0.595 ms/token | 2,339 ms |

**JAX prefills 2.4x slower than PyTorch**, consistently across all five shared context
lengths. On an interactive workload with real context that dominates user-visible latency, and
it runs in the opposite direction to the 1.24x decode advantage the same rig enjoys.

## Boot Time Reverses the Ranking

Nine cold boots, three per runtime, plus nine warm reloads. The start line is the moment
`run_instances` returns an id, because capacity wait measures AWS rather than the rig.

| | runtime | cold boot | spread | warm reload | cold/warm |
|---|---|---:|---:|---:|---:|
| 🥇 | PyTorch | 195.2 s | 11.8% | 24.5 s | 8.0x |
| 🥈 | JAX | 242.2 s | 11.5% | 74.1 s | 3.3x |
| 🥉 | vLLM | 1417.8 s | 12.6% | 264.3 s | 5.4x |

**vLLM takes 23m 38s to serve, from a prebuilt AMI that downloads nothing.** PyTorch installs
its runtime from wheels and pulls a 9.5 GB checkpoint over the network, and is still 7.3x
faster.

Boot variance is 11.5, 11.8 and 12.6 percent — too consistent across three different runtimes
to be a property of any of them. That is roughly six times the decode noise floor, which makes
a single boot measurement nearly worthless.

## Is Health 200 the Same as Ready to Serve?

Not on every runtime, which is why the harness records two stop lines.

| runtime | first completion, cold | warm |
|---|---:|---:|
| vLLM | 0.5 s | 0.2 s |
| PyTorch | 1.0 s | 0.7 s |
| JAX | 22.9 s | 9.2 s |

JAX returns health 200 and then compiles XLA per shape bucket on the first real request.
Quoting health alone understates its time to serving by 22 seconds, and the cost does not
vanish when warm. vLLM is the mirror image: slowest to boot, fastest first token, because
graph capture is paid before the port binds.

## What Does a Code Change Cost?

| runtime | to change serving code |
|---|---|
| PyTorch | ship 3 files over SSM, restart — 25 s |
| JAX | same mechanism, plus 9.2 s of compile — 83 s |
| vLLM | no deploy path exists: rebuild from source, ~67 min, and reapply an out-of-tree Turing patch |

vLLM's 264 s warm figure is a `systemctl restart`, not a code change, so it flatters the
comparison. **vLLM wins decode 3.2x and loses the iteration loop by 3 to 100x.**

## Five Theories About 546 Seconds

vLLM's cold boot is dominated by weight loading: 468 to 561 seconds across four measurements.
Explaining it took five attempts, four of which were wrong.

1. *`g5g.2xlarge` needs no swapfile and buys that time back.* The rig's own documentation.
   Falsified by the three-boot campaign at 23m 38s.
2. *A bigger host will not fix it.* Retracted the same day — no large host had been measured.
3. *A larger host would very plausibly fix it.* Falsified by one `g5g.4xlarge` boot: available
   RAM 11.19 to 26.49 GiB, weight loading 546 to 468 s, and total boot 4.7 percent lower,
   inside the noise band.
4. *It is the loader; vLLM's log says auto-prefetch is disabled on EXT4.* Falsified by a
   within-box A/B — 76.13 s as shipped against 75.12 s with
   `--safetensors-load-strategy=prefetch`, which is 1.3 percent and therefore nothing.

What that last run did find is the useful part.

| | weight load | n |
|---|---:|---|
| cold boot, fresh instance | 468-561 s | 4 |
| warm restart, same box | 32-76 s | 3 |

Same volume, same filesystem, same engine, differing only in whether the blocks had been read
once. 9.54 GiB in 468 s is 20.9 MB/s, which is absurd for gp3 steady state and ordinary for
first-touch reads against a snapshot-backed volume.

Theory five is EBS lazily hydrating the volume from the AMI snapshot, and it is written down
as untested. Given the strike rate it does not get promoted by reasoning.

## A Measurement Floor That Looked Like Reproducibility

The first boot campaign was discarded and re-run. Two independent instances had reported
214.4 s and 125.1 s, identical to the tenth, which reads as extraordinary consistency and is
really a five second poll quantising two similar boots onto the same tick.

The data was not wrong; the campaign log shows 215 s and 216 s of wall clock. It was unusably
coarse. Health polling went to half a second, and the next pair of boots came in at 216.90 s
and 193.86 s — an 11.9 percent spread the old harness could not see.

## And Price/Performance?

| | $/hr |
|---|---:|
| `g5g.2xlarge` on-demand | $0.556 |
| spot, measured across four AZs | $0.3813 - $0.4416 |

The whole exercise — roughly 15 instances, 3.5 to 4 instance-hours, a serving sweep, three
cross-rig runs, nine boots and three A/B restarts — came to under $2.

That 26 to 46 percent premium is the entire spot versus on-demand decision on this hardware,
which is to say there is not one. Try spot, fall back, keep working; the automatic fallback
cost about $0.24 across a nine-boot campaign.

**Cheap hardware is what made the method possible, not merely affordable.** Discarding a
completed campaign over a poll-interval bug cost twenty minutes and pennies. Where a run is
expensive, the same discovery argues for shipping the numbers with a caveat instead.

## Tearing It Down

Every instance is terminated as soon as its artifacts are captured. There is no built image to
lose, only a pip install and a model cache.

```
terminate_g5g_instance i-02e79988a6cbeecbf
```

```
🗑️ Terminating `i-02e79988a6cbeecbf`. Relaunch costs a pip install, not a build.
```

```bash
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[?starts_with(InstanceType,`g5g`)].InstanceId' --output text
```

```
🟢 no g5g instances running
```

## What This Does Not Cover

Everything above is concurrency one, which makes it a latency comparison rather than a serving
one. Continuous batching is vLLM's whole value proposition and it is untested here.

Only vLLM can serve concurrently at all today. The PyTorch server holds an `asyncio.Lock` with
the comment *"one GPU, one process -> serialize requests"*, and the JAX rig has no batching
machinery whatever. The engine-level batch sweep suggests what is on the table — batch eight
reaches 84.16 tok/s for an extra 0.258 GB, with per-step time growing two percent across an
eight-fold batch — but that number never leaves the engine.

Three further gaps: the JAX leg ran its shipped quantised configuration against two dense
runtimes; no output-quality axis was measured at all, on a comparison where one runtime uses a
deliberately lossy LM head; and every TTFT figure predates the prompt-uniqueness fix, so only
the JAX versus PyTorch half of that table is sound.

## The Short Version

The goal of this article was to compare three inference runtimes on identical silicon without
the harness being a variable. The key to the solution was a single client-side statistic that
every OpenAI-compatible server can produce. The measured results were:

- Decode at concurrency one: vLLM 32.53, JAX 12.69, PyTorch 10.24 tok/s — 53, 21 and 17
  percent of a 61.4 tok/s bandwidth ceiling, so none is bandwidth-bound.
- The ranking reverses on lifecycle. Cold boot 195.2 s for PyTorch against 1417.8 s for vLLM;
  a code change costs 25 s against a from-source rebuild.
- JAX prefills 2.4x slower than PyTorch, 1.403 against 0.595 ms/token.
- A 94.7 percent prefix-cache hit rate turned a 29x TTFT result into a harness artifact.
- Calibration offsets are per-rig, 0.9799 and 0.9543, and are not transferable.
- Boot variance is about 12 percent on this platform regardless of runtime.

Scope: the decode and boot numbers are one `g5g.2xlarge` per runtime in `us-east-1a` on
2026-08-31, three repeats per cell and three repeats per boot, mixed spot and on-demand with
the market recorded per run. Three things differed between the legs and are named where they
matter: vLLM ran `max_model_len` 16384 against 4096 for the other two; the JAX leg ran its
shipped quantised configuration against two dense runtimes; and vLLM booted from a prebuilt
AMI carrying its model cache while the other two installed from wheels and downloaded the
checkpoint, which is the point of the boot comparison rather than a flaw in it. Two runs sit
outside that envelope and say so in the text: the RAM test was a single `g5g.4xlarge`, and the
prefetch A/B ran 2026-09-01 on on-demand after a spot reclamation killed the first attempt.
Decode is unaffected by the prefix-cache issue, which changes prefill only.

The strategy for using MCP for multi-runtime comparison was validated with a incremental step
by step approach.
