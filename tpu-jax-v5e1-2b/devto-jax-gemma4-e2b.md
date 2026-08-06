---
title: "One TPU Chip, Eight Agents: Is Raw JAX a Viable Serving Path for Small Agent Workloads?"
published: false
description: "Gemma 4 E2B QAT can't be loaded by vLLM on TPU, so I built the inference path in pure JAX and worked out what a single v6e chip can actually carry: the memory math for a small agent fleet, what raw JAX gives you, and the three things you'd have to build before it's production."
tags: tpu, llm, jax, agents
---

*Cloud TPU v6e-1 (`ct6e-standard-1t`, one v6e chip, 32 GB HBM), GCE flex-start, europe-west4-a. vLLM baseline measured 2026-07-21.*

## The workload nobody benchmarks

Serving benchmarks optimize for the wrong shape. They report throughput at concurrency 100 with 1,024-token prompts, because that's what a public inference endpoint looks like.

A small agent workload looks nothing like that:

- **Low concurrency, high value per stream.** Two to eight agents, not two hundred. Each one is a person waiting, or a pipeline stage blocking.
- **Latency compounds serially.** An agent turn is *think → call tool → read result → think again*. Ten round trips at 400 ms of model time each is four seconds of wall clock the user feels.

  Worth flagging that this premise is contested. Zimbres' measurement series argues the reverse for agent fleets: agents "exchange messages among themselves, multiplying token volume by an order of magnitude or more per user task, with no human waiting on any individual token. There, per-token latency loses most of its meaning and aggregate throughput becomes the binding constraint" ([DOI 10.5281/zenodo.21212010](https://doi.org/10.5281/zenodo.21212010)). Both are right, for different topologies — a serial agent loop with a human at the end is latency-bound; a parallel machine-to-machine fleet is throughput-bound. The metric that covers both is **goodput**: throughput subject to a per-token latency objective. I use that framing below, but I do not claim a production goodput result from a kernel harness.
- **Context grows monotonically.** Every tool result is appended and resent. A 20-turn agent loop replays a context that started at 500 tokens and ended at 8,000. This costs more than the extra prefill: on a v5e chip, *decode* velocity for Gemma 2B falls from ~8,300 to ~2,900 tok/s as input length approaches the context limit, "governed almost entirely by input length and almost not at all by output length" ([DOI 10.5281/zenodo.21227936](https://doi.org/10.5281/zenodo.21227936)). The generation rate itself sags as history accumulates.
- **Output is structured.** Tool calls are JSON that has to parse. A malformed call isn't a quality regression, it's a crash. Greedy decoding can be a good fit for a narrowly fine-tuned tool model because its output distribution is sharper ([DOI 10.5281/zenodo.21212010](https://doi.org/10.5281/zenodo.21212010)). That does not make greedy decoding lossless for this general-purpose Gemma checkpoint; it only makes deterministic, cacheable behaviour an attractive operating point to test.

That workload fits on one chip — if you can serve the model at all. Which brings us to the problem.

## The forcing function: #3225

Quantization-aware-trained exports of Gemma 4 E2B **do not load in vLLM on TPU**:

| Checkpoint | Backend | Failure | Verdict |
| :--- | :--- | :--- | :---: |
| `-qat-w4a16-ct` | tpu-inference (JAX) | `int4 compressed-tensors` scheme unimplemented for `per_layer_model_projection` | ✕ no load |
| `-qat-q4_0-unquantized` | tpu-inference (JAX) | demands `self_attn.k_norm.weight` for layers 15–34 | ✕ no load |
| `-qat-q4_0-unquantized` | torchax (`MODEL_IMPL_TYPE=vllm`) | identical missing-weights error | ✕ no load |
| `google/gemma-4-E2B-it` (BF16) | tpu-inference (JAX) | — | ✓ serves |

**The checkpoint is right and the loader is wrong.** Comparing safetensors headers: the BF16 export ships `self_attn.k_norm` for all 35 layers; the QAT export ships it only for the 15 non-KV-shared layers. Both configs declare `num_kv_shared_layers: 20`.

Layers 15–34 reuse K/V computed by lower layers. They hold no K-side parameters, so a k-norm for them is meaningless — the QAT export is the architecturally honest one, and the plain checkpoint only loads because it happens to carry the unused tensors along. Filed as [tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225); the fix is to skip instantiating K/V-side parameters for shared layers rather than demanding them unconditionally.

So: serve BF16 and forgo QAT, or write the inference path yourself. I wrote it.

## The engine

`ports/gemma4/jax_e_model.py` is a pure-JAX Gemma 4 E2B — no PyTorch, no `torch_xla`, nowhere in the path. That's not an aspiration: the checkpoint goes safetensors → JAX PyTree directly (via `safetensors.flax`, which handles bfloat16 natively), and `tests/test_jax_engine.py` asserts `torch` never even enters `sys.modules` during a load. If you're used to "JAX" inference that quietly loads weights through `AutoModelForCausalLM` and converts, this isn't that.

Sharing is encoded in the layer definition instead of patched at load time:

```python
@property
def first_kv_shared_layer_idx(self) -> int:
    return self.num_hidden_layers - self.num_kv_shared_layers   # 35 - 20 = 15

def kv_share_map(self) -> List[int]:
    """Maps each layer index to the source layer index for KV state sharing."""
    first = self.first_kv_shared_layer_idx
    last_of_type = {}
    for i in range(first):
        last_of_type[self.layer_types[i]] = i
    return [
        i if i < first else last_of_type[self.layer_types[i]]
        for i in range(self.num_hidden_layers)
    ]
```

Sharing is per attention *type* — E2B interleaves sliding and full attention, and a shared layer inherits from the last non-shared layer of its own kind. Only `range(first_kv_shared_layer_idx)` ever allocates K/V. W4A16 weights (eight int4 values per int32, BF16 scale per 32-element group) are unpacked on chip, read straight from safetensors into a JAX PyTree.

Decoding runs against a static KV cache written with `dynamic_update_slice`, so each step attends to the full prefix. The way you know a decoder is correct is to compare it against the slow thing that obviously is — `tests/test_kv_cache_parity.py` runs greedy generation both ways, cached decode versus re-running the full model over the growing sequence every step:

```
step | max|dlogit| | ref tok  | cached tok | ref top-2 gap
   0 |    0.000000 | [62, 36] | [62, 36]   | [0.009738, 0.118984]
   ...
   7 |    0.000001 | [97, 95] | [97, 95]   | [0.012065, 0.000438]
```

Maximum divergence across every step: **1e-6**, float32 roundoff. (The test asserts logit agreement within tolerance and token agreement only where the top-2 gap exceeds it — near a 0.0004 tie, an exact argmax assertion is flaky by construction.) A companion test proves padding a prompt to a 128-aligned bucket is invisible to the output: RoPE follows logical position, not cache slot.

## Feasibility, part 1: the memory math

Here is the result that actually settles the "can one chip hold my agent fleet" question, and it needs no benchmark at all — just the architecture.

E2B's KV sharing means only 15 of 35 layers hold KV tensors. Twelve sliding-attention
layers use a 256-wide KV head; three full-attention layers use a 512-wide one:

```
12 × 1 × 256 × (K+V) + 3 × 1 × 512 × (K+V)
    = 9,216 KV elements/token
    = 18.00 KiB/token at bf16
```

That 18.00 KiB figure is computed from the checkpoint shapes and reproduced by
`init_kv_cache`. An earlier 15.0 KiB estimate treated every layer as 256-wide and
under-counted the three full-attention layers. So the resident bf16 KV alone costs:

| Fleet | Context each | Resident KV tokens | KV total (bf16) |
| :--- | ---: | ---: | ---: |
| 4 agents | 4,096 | 16,384 | 302 MB |
| 8 agents | 4,096 | 32,768 | 604 MB |
| 8 agents | 8,192 | 65,536 | 1.21 GB |
| 32 agents | 8,192 | 262,144 | 4.83 GB |

The real QAT checkpoint loads as 6.56 GB of weights, so eight 8K contexts plus
weights account for about **7.77 GB** before activations, executable buffers and
allocator overhead. Do not subtract that from 32 GB and call the remainder free:
capacity measurements show non-KV temporaries can consume roughly 40% of HBM at
large batch. The defensible conclusion is narrower: **eight 8K contexts fit
comfortably; the weight-plus-KV arithmetic is not the limiting admission check.**

At ctx 8,192, the donated decode path was bisected at 1,425,408 resident KV tokens
for bf16 and 2,818,048 for int8: B=174 and B=344 respectively. Those are decode
residency ceilings, not promises that the same number of full prompts can be
prefilled together. Prefill has a separate measured rule:
`batch × chunk_size <= 8,192` prompt tokens per pass.

**So the chip isn't full. The problem is that my server only runs one request at
a time.**

Here is the entire proof. One request decodes at 140.5 tok/s. Eight requests at
once decode at 143.3 tok/s *combined* — eight times the work for 2% more output.
Each user just waits eight times longer, 497 ms to 1,775 ms. Nothing ran out of
memory and no request failed. The chip wasn't full and it wasn't slow. It was
doing one request at a time while the other seven queued, which is exactly what
the code says: `generate_stream` is B=1, and FastAPI launches independent B=1
executions.

It also isn't a speed problem, and that distinction matters because it points at
different work. The same chip's decode kernel does 2,888 tok/s at batch 32 —
about 20× what my server delivers. So the hardware is sitting idle. Batching
would have to claim that 20× before a faster kernel is worth writing.

**Memory: 43× more than I need. Speed: 20× more than I'm using. Batching: none.**
The fix is a request batcher, not a bigger chip.

To be precise: I mean **request** scheduling — which requests share a device
execution — not how HBM streams data. HBM efficiency is a real constraint, but
it's the next one. You can't tune a batch that doesn't exist.

### So how many agents should one chip run?

Weights cost the same per step no matter the batch — about 1.86 GB streamed, not
the full 6.56 GB, since the PLE table is gathered rather than streamed. KV grows
with agents × context. So agents are cheap until KV traffic overtakes the weight
read.

Where that crossover lands, I can't say yet: measuring it needs a batcher, and
this server doesn't have one. But I suspect it isn't a number of agents at all,
rather a combination of agent count and context size — eight agents at 8K should
behave much like thirty-two at 2K.

## Feasibility, part 2: what agent workloads need that raw JAX doesn't have

This is the honest core of the assessment. Three gaps, ranked by how much they'd hurt an agent workload specifically:

**1. No prefix caching — and agent loops are the pathological case.** Every agent turn resends the entire conversation plus every tool result so far. With prefix caching, turn 20 reprocesses only the new tokens. Without it, you re-prefill the whole growing context every single turn. Over a 20-turn loop with context growing 500 → 8,000 tokens, that's *quadratic* wasted prefill. vLLM ships this; raw JAX does not. **For agent workloads this is the single biggest gap**, and it isn't exotic to build — but you do have to build it. Measured from the other direction: prefix caching "is at its most effective in agent fleets, because agents share long system prompts and accumulated context, so the prompt-processing cost of inter-agent messages collapses" ([DOI 10.5281/zenodo.21212010](https://doi.org/10.5281/zenodo.21212010)).

**2. No guided decoding.** Tool calls have to be parseable JSON. vLLM has
schema/grammar-constrained generation; this raw JAX server does not. On the
measured Gemma 4 vLLM stack, schema enforcement was itself conditional: it worked
with thinking enabled and failed or partially worked with thinking disabled.
Client-side validation remains necessary on either path, but raw JAX still lacks
the server-side constraint layer entirely.

**3. Lockstep batching penalizes uneven turns.** A fixed batch runs together and a finished sequence holds its slot until the whole batch drains. Agent turns are wildly variable — one agent emits a 12-token tool call while another writes 400 tokens of reasoning. vLLM's continuous batching swaps in new work per step; here, short turns wait on long ones. At the low concurrency agent fleets run at, this is a real efficiency tax.

Plus the standing costs of the approach: static shapes mean any uncompiled `(batch, seq)` combination stalls on XLA compilation, so you bucket and spend FLOPs on padding; and every new architecture or quantization format is hand-written.

Against that, what raw JAX genuinely buys you: **it runs the checkpoint vLLM
can't**, and the whole stack is inspectable. A persistent JAX compilation cache
avoids about 17 seconds of recompilation on a warm restart. The vLLM deployment's
measured cold time-to-healthy was about 8.5 minutes, including image pull, model
load and 329 seconds of compilation. Those are not like-for-like restart timings,
so the supported claim is simply that the JAX cache shortens its own warm loop.

## The vLLM bar

For calibration, vLLM serving the BF16 checkpoint on the same chip, measured with `vllm bench serve` (1,024-token input, 128-token output):

| Concurrency | Output tok/s | Per-stream tok/s | Median TTFT | Median TPOT |
| :---: | ---: | ---: | ---: | ---: |
| 1 | 209 | 213 | 16 ms | 4.7 ms |
| 8 | 1,209 | 161 | 27 ms | 6.2 ms |
| 32 | 1,636 | 57 | 155 ms | 17.5 ms |
| 64 | 2,140 | 39 | 122 ms | 25.3 ms |
| 100 | 2,215 | 27 | 833 ms | 36.8 ms |

Note the shape at agent-scale concurrency: at 8 streams it holds **161 tok/s per stream at 27 ms TTFT**. That is a good bar, and it's what you give up by leaving vLLM. The honest framing is not "raw JAX beats vLLM" — it's "vLLM can't load this checkpoint, and here's what the alternative costs you."

## Measured: the raw JAX decode kernel on the same chip

This distinction matters: correctness was tested with the real checkpoint;
performance was measured with deterministic, architecture-shaped synthetic
parameters. Under static JAX shapes the values do not change the compiled work,
but this is still a **decode-kernel benchmark**, not an end-to-end serving result
and not a direct vLLM comparison.

The final revalidation used the checkpoint's dimensions, one configuration per
process, 15 timed samples after warmup, ctx 8,192 and B=32. The donation and
cache-dtype effects exceeded twice the measured IQR; the PLE differences did
not. The largest relative IQR was 1.32%.

| PLE | KV | cache update | step | aggregate |
| :--- | :--- | :--- | ---: | ---: |
| bf16 | bf16 | copying | 21.262 ms | 1,505 tok/s |
| bf16 | bf16 | **donated** | 13.143 ms | 2,435 tok/s |
| bf16 | int8 | **donated** | 11.088 ms | 2,886 tok/s |
| int4 | int8 | **donated** | **11.080 ms** | **2,888 tok/s** |

The largest fix was not quantization. It was buffer donation. Without
`donate_argnums`, `dynamic_update_slice` created a second full cache to write one
token, then attention read it again. Donation made bf16 1.62× faster and nearly
doubled its resident-token ceiling. On the corrected donated path, int8 KV was
1.17–1.19× faster than bf16 and retained 1.82–1.98× the KV-token capacity.
On the real checkpoint, int8 was also indistinguishable from bf16 in a 583-step
teacher-forced quality check (28.41 versus 28.73 perplexity, 97.08% greedy
match), and its error did not grow across a separate 968-step continuous decode.

Quantizing the 4.70 GB per-layer embedding table to int4 reduced the measured
weight tree from 6.618 to 3.113 GB, but did not move step time: 13.143 versus
13.099 ms with bf16 KV, a 0.3% difference. That table is gathered, not streamed
in full every token. Quantizing it buys residency, not throughput.

The roofline cross-check also changed the conclusion. A calibrated streaming
reduction reached 1,417 GB/s. The donated decode step's marginal KV read reached
794 GB/s, about 56% of that; without donation it reached 276 GB/s, or 19%.
This engine still has headroom, but eager attention was not the dominant mystery:
attention in isolation measured in the same 39–59% range. The extra cache copies
were.

### End to end: the server does not batch yet

I then ran the real checkpoint through the OpenAI-compatible HTTP endpoint. With
int8 KV and donation enabled, warm single-stream decode stayed nearly flat as
context grew. The prompt repeated a fixed sentence to reach each token count, so
this is a latency/scheduling test rather than a quality evaluation:

| Prompt tokens | HTTP wall time | Prefill | Decode |
| ---: | ---: | ---: | ---: |
| 506 | 240.5 ms | 9.3 ms | 141.1 tok/s |
| 2,045 | 267.7 ms | 32.0 ms | 140.5 tok/s |
| 7,679 | 570.9 ms | 318.6 ms | 138.5 tok/s |

The long-context penalty is almost entirely prefill. But simultaneous HTTP
requests expose the more important limitation. At a 2K prompt and 32 output
tokens, concurrency 2/4/8 produced 128.7/133.6/143.3 aggregate tok/s while
median request latency rose from 497 to 952 to 1,775 ms. Every request succeeded;
none formed a device batch.

That is exactly what the code says should happen: `generate_stream` is B=1, and
FastAPI launches independent B=1 executions. The device queues them. The
2,888 tok/s kernel point proves batched decode work can be fast; the current
server cannot reach it until it owns a real request batcher and batched KV state.

## The server

`jax_openai_server.py` runs generation on the JAX engine: `/v1/chat/completions`, `/v1/completions`, `/health`, Prometheus `/metrics`, and SSE streaming that emits one chunk per decoded token.

```bash
python3 jax_openai_server.py --port 8000 --max-model-len 4096
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it-qat-w4a16-ct",
    "messages": [{"role": "user", "content": "Explain TPU MXU vectorization."}],
    "stream": true
  }'
```

`tests/test_openai_server.py` drives the real endpoints, including an assertion that streaming and non-streaming produce **identical greedy text** — precisely the thing that silently rots when streaming is bolted on separately. `tests/test_jax_engine.py` additionally asserts the load path never imports `torch`.

## Verdict

**Is raw JAX viable for a small agent workload on one TPU chip? As an
experimental serving path, yes. As a vLLM replacement, not yet.**

The strongest evidence is correctness and capacity: the QAT checkpoint that
blocks vLLM loads and generates; cached decode matches a full re-forward; eight
8K bf16 contexts require 1.21 GB of KV; and donated decode has far more residency
headroom than that. The kernel result at B=32 is promising, but it is not an
OpenAI-serving benchmark. The real HTTP test establishes the current serving
limit instead: roughly **139–141 tok/s aggregate**, with latency growing almost
linearly when 2–8 independent requests contend for the device.

The deciding gaps are prefix caching, constrained decoding and continuous
batching. If agents resend a growing history, the current server re-prefills it
quadratically over the conversation. Chunked prefill makes large batches
admissible, but it does not reuse a previous turn's prefix and currently runs
about 5× slower than windowed one-shot prefill at B=8, ctx 2,048.

Use this path when you specifically need the QAT checkpoint, can tolerate an
experimental scheduler, and value a stack you can inspect end to end. For a
production agent service today, the evidence still favours vLLM with the BF16
checkpoint while waiting for #3225.

### Measure it on your own chip

```bash
python3 benchmarks/queued/revalidate.py --all --out results.json
```

This harness uses synthetic parameter values but checkpoint-matched dimensions,
runs each configuration in a fresh process and measures both donated and copying
paths. Keep performance and capacity separate: use it for step latency, and the
dedicated capacity bisection in the run directory for HBM ceilings. A doubling
ladder previously manufactured a false "constant" ceiling, and a non-donated
step later measured exactly half the usable capacity.

### Two optimizations that didn't work

Worth recording, because the reasoning was plausible and the result wasn't. Decode reads the full weight set per token, so 4-bit weights ought to cut traffic ~4× — a fused Pallas kernel that unpacks int4 inside the VMEM tile instead of dequantizing to BF16 in HBM first. Predicted ~3.4× on the memory floor. Measured: **0.59× at B=1, 0.21× at B=2**, and `CompileTimeScopedVmemOom` on five of eight cells because the kernel loads all of `x` as one VMEM block rather than tiling the sequence axis. Quantizing the LM head to int8 — the single largest read in a step — bought **1.00×/1.04×/1.05×** at B=1/2/8, for 0.8% logit error.

Neither failure was novel. The same series reports speculative decoding costing a **factor of six**, a precision change that "looked certain on paper" costing 14% and being reverted, and the general rule that this stack is best modified *above* the compiler (configuration) or *below* it (whole kernels) "rather than through point edits to the compiled middle" — which is exactly what a fused dequant-matmul is ([DOI 10.5281/zenodo.21212010](https://doi.org/10.5281/zenodo.21212010)). It also brackets the ceiling I was chasing: the entire logits pipeline is 18% of a decode step, so an int8 LM head could never have paid much.

Those failures do **not** prove weight traffic is irrelevant. The old ~3.6 ms
floor omitted KV traffic, and the old 20.6 ms step was measured before the
donation fix. The narrower conclusion survives: these two implementations did
not pay. Profile attribution and a calibrated bandwidth bound were more useful
than another speculative kernel.

The uncomfortable part is that this is a known failure pattern, not a novel one. The same series ran the fashionable strategies and measured them losing: a team "following the mainstream sequence without profiling would have deployed speculative decoding, measured here at six times slower in this regime; pursued quantization, measured at zero functioning routes on this stack, one failing silently; or hand-optimized a genuine numeric inefficiency, measured at 14 percent slower when removed at the wrong layer." I did the second one, and mine failed silently too.

There's a landmine in that kernel worth flagging if you write one: it originally unpacked nibbles plane-major while contracting activations in natural order — silently wrong — and nobody noticed because a bare `except Exception` fell back to the reference path on any host without a TPU. It only computes garbage where Pallas actually compiles. Verify kernels against a reference on the hardware you ship on, and never let a fallback swallow the exception that would have told you.

The next engineering step is no longer another kernel benchmark. It is a request
batcher with batched KV ownership, continuous admission and prefix reuse, followed
by the same 2–8 stream HTTP test. Peak aggregate kernel throughput cannot answer
that question by itself.

### Repository

- **Engine:** [`ports/gemma4/jax_e_model.py`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/ports/gemma4/jax_e_model.py) · **loader:** [`ports/gemma4/jax_e_loader.py`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/ports/gemma4/jax_e_loader.py)
- **Serving engine:** [`jax_engine.py`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/jax_engine.py) · **server:** [`jax_openai_server.py`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/jax_openai_server.py)
- **Tests:** [`tests/test_kv_cache_parity.py`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/tests/test_kv_cache_parity.py) (cached decode vs full re-forward) · [`tests/test_jax_engine.py`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/tests/test_jax_engine.py) (load path, torch-free assertion) · [`tests/test_openai_server.py`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/tests/test_openai_server.py) (endpoints, SSE)
- **Benchmark harness:** [`ports/gemma4/jax_e_benchmark_sweep_v2.py`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/ports/gemma4/jax_e_benchmark_sweep_v2.py)
- **Corrected validation report:** [`benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md)
- **Real-weight HTTP validation:** [`benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md)
- **vLLM baseline data:** [`benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json)
- **Upstream issue:** [tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225)

### Prior art

Rubens de Almeida Zimbres' TPU inference measurement series (CC-BY-4.0) is the closest published
work to this, and several of its results are cited above. Notes on how each bears on the engine
here are in [`docs/references/tpu-inference-measurement-series.md`](https://github.com/xbill9/tpu-jax-v5e1-2b/blob/main/docs/references/tpu-inference-measurement-series.md).

- [10.5281/zenodo.21212010](https://doi.org/10.5281/zenodo.21212010) — *From 1,540 to 19,511 Tokens per Second on a Single TPU v5e Chip* (12.7x; arithmetic-intensity framework; what failed and why)
- [10.5281/zenodo.21227936](https://doi.org/10.5281/zenodo.21227936) — *Token Velocity on a Single TPU v5e Chip* (decode rate vs. input length)
- [10.5281/zenodo.21404155](https://doi.org/10.5281/zenodo.21404155) — *The Decode Block Size Heuristic in TPU Ragged Paged Attention* (28–69% left on the table by one constant)
- [10.5281/zenodo.21462837](https://doi.org/10.5281/zenodo.21462837) — *Batch Scaling and Goodput of a Tuned Attention Kernel on TPU v6e* (the goodput cliff)
