# 2026-09-01 — concurrency, all three runtimes on one T4G

**The gap every earlier number in this family left open.** Everything before today was
concurrency 1, which measures latency and not serving; continuous batching is vLLM's whole
value proposition and had never been tested here.

One harness (`sweep.py --concurrency`), one shape — 512 in / 128 out, matching
`vllm bench serve`'s `--random-input-len` / `--random-output-len` — three `g5g.2xlarge`
instances in `us-east-1` on 2026-09-01. Artifacts are `concurrency.json` in each rig's run
directory.

## Aggregate throughput

| c | vLLM | JAX | PyTorch |
| ---: | ---: | ---: | ---: |
| 1 | 32.49 | 11.08 | 9.72 |
| 2 | 52.70 | 11.63 | 9.86 |
| 4 | 76.13 | 11.66 | 10.06 |
| 8 | **100.28** | 11.56 | 10.05 |
| 16 | 100.40 | 11.53 | 10.00 |
| 32 | 99.01 | 11.50 | 10.08 |
| **scaling 1→peak** | **3.09x** | 1.05x | 1.04x |

**Only vLLM converts a second client into capacity.** It scales 3.09x to c=8 and then stops
dead — 100.28 / 100.40 / 99.01 across c=8/16/32 — which lands exactly on `--max-num-seqs 8`
and independently reproduces the 2026-08-14 saturation finding on a different instance size
with a different harness. The other two are flat within 5% across a 32x range.

## The c=1 figure understates vLLM by 3x

| | c=1 | c=32 |
| --- | ---: | ---: |
| vLLM vs PyTorch, throughput | 3.3x | **9.8x** |
| vLLM vs PyTorch, TTFT | 0.9x | **10.6x** |

The "3.2x faster" number this project has quoted for weeks is the concurrency-1 result. At
realistic load vLLM is ~10x on both axes at once. **A latency benchmark and a serving
benchmark are different measurements, and only one of them answers a deployment question.**

## Two different ways to not have continuous batching

Flat aggregate is where the similarity ends. PyTorch and JAX fail in opposite directions:

| c=16 | aggregate | TTFT | TPOT | per-stream |
| --- | ---: | ---: | ---: | ---: |
| vLLM | 100.40 | 5.2 s | 48 ms | 21.18 |
| JAX | 11.53 | 14.0 s | **1,205 ms** | 0.83 |
| PyTorch | 10.00 | **60.0 s** | 96 ms | 10.45 |

- **PyTorch queues.** `torch_openai_server.py` holds one `asyncio.Lock` — *"one GPU, one
  process -> serialize requests"*. A request waits its full turn, then decodes at full speed.
  All the cost lands in TTFT: 0.4 s at c=1 to **124.3 s at c=32**, linear in c, while TPOT
  never leaves ~96 ms.
- **JAX interleaves.** Requests are admitted and time-sliced, so first tokens arrive ~4x
  sooner than PyTorch's and then every stream crawls — **0.83 tok/s each at c=16**, 1.2 s per
  token. Past c=16 some queueing joins the time-slicing: TPOT and TTFT both grow ~1.5x rather
  than 2x from c=16 to c=32.

Same ceiling, opposite distribution. A JAX user gets a fast first token then watches it type;
a PyTorch user waits a minute then gets a fast answer.

## What this prices

The PyTorch engine reaches **84.16 tok/s at B=8** (`2026-08-29-profile-and-fixes-g5g`), and
its served path delivers **10.05**. That 8.4x is what continuous batching is worth on this
rig, measured on both sides rather than argued from a roofline.

## Scope, and one number not to quote

Every figure is one instance per runtime, `g5g.2xlarge`, 512/128, unique prompts so no engine
answers from a prefix cache.

**The absolute ceiling is not trustworthy and the ratios are.** This run's vLLM c=8 reads
100.28 tok/s where 2026-08-14 measured 168.33. Three things differ and none was isolated: that
run used a `g5g.4xlarge` (16 vCPU against 8), it used `vllm bench serve --ignore-eos` which
forces exactly 128 output tokens where these stop at EOS, and **its client ran on the box
against localhost while this one drives up to 32 parallel SSE streams over the WAN.** The last
is the likeliest — at high concurrency the client may be the bottleneck rather than the server.
An on-box client is what would settle the ceiling; the shape and the between-rig ratios do not
depend on it, because all three were measured identically.
